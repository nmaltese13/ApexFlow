"""Self-computed Heatseeker levels.

These levels are derived directly from the raw options chain (strikes / OI /
volume / IV) plus Black-Scholes Greeks. Nothing is pulled from an external
"levels" feed — the heatseeker computes its own:

  hvl            High Volume Liquidity strike (max OI*volume product, total)
  call_wall      Strike with the largest call open interest (resistance)
  put_wall       Strike with the largest put open interest (support)
  zero_gamma     Cumulative-GEX sign-flip strike, linearly interpolated
  vol_trigger    Highest strike below spot whose cumulative GEX (from low)
                 is still negative — once price slips below it dealers go
                 short-gamma and volatility expands
  charm_high     Strike with peak |charm| above spot (call-side time-decay magnet)
  charm_low      Strike with peak |charm| below spot (put-side time-decay magnet)
  vanna_high     Strike with peak |vanna| above spot
  vanna_low      Strike with peak |vanna| below spot
  king           Largest |GEX| strike (Skylit "King Node")
  gatekeepers    Next gatekeeper_count strikes by |GEX|

All level values are floats (the strike), or None when unavailable.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .dealer_greeks import greek_grid
from .timeutil import years_to_expiry as _years


# ---------------------------------------------------------------------------
# Greeks specific to level computation (charm + vanna)
# ---------------------------------------------------------------------------
def _d1_d2(S, K, t, sigma, r=0.04, q=0.0):
    if S <= 0 or K <= 0 or t <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def charm(S, K, t, sigma, right="C", r=0.04, q=0.0):
    """∂Delta/∂t — the rate at which delta decays per year."""
    d1, d2 = _d1_d2(S, K, t, sigma, r, q)
    if d1 is None:
        return 0.0
    pdf = norm.pdf(d1)
    common = pdf * (2 * (r - q) * t - d2 * sigma * math.sqrt(t)) / (2 * t * sigma * math.sqrt(t))
    if right == "C":
        return float(-q * math.exp(-q * t) * norm.cdf(d1) + math.exp(-q * t) * common)
    return float(q * math.exp(-q * t) * norm.cdf(-d1) + math.exp(-q * t) * common)


def vanna(S, K, t, sigma, r=0.04, q=0.0):
    """∂Delta/∂σ = ∂Vega/∂S. Same for calls and puts."""
    d1, d2 = _d1_d2(S, K, t, sigma, r, q)
    if d1 is None:
        return 0.0
    return float(-math.exp(-q * t) * norm.pdf(d1) * d2 / sigma)


# ---------------------------------------------------------------------------
# Level computation
# ---------------------------------------------------------------------------
def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def compute_levels(
    raw_chains: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
    rolled_gex: pd.DataFrame,
    spot: float,
) -> dict:
    """
    raw_chains : iterable of (expiry, calls_df, puts_df) tuples already filtered
                  to the requested DTE window
    rolled_gex : DataFrame with columns ["strike", "call_gex", "put_gex",
                  "total_gex"] (already rolled-up across the same expiries)
    spot       : underlying spot price
    """
    out = {
        "hvl": None,
        "call_wall": None,
        "put_wall": None,
        "zero_gamma": None,
        "vol_trigger": None,
        "charm_high": None,
        "charm_low": None,
        "vanna_high": None,
        "vanna_low": None,
        "king": None,
        "gatekeepers": [],
    }
    if rolled_gex is None or rolled_gex.empty or spot <= 0:
        return out

    # ------------------------------------------------------------------
    # Aggregate raw chain stats per strike (OI / volume / IV-weighted)
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for expiry, calls, puts in raw_chains:
        t = _years(expiry)
        if calls is not None and not calls.empty:
            for _, r in calls.iterrows():
                rows.append({
                    "strike": _safe_float(r.get("strike")),
                    "right":  "C",
                    "oi":     _safe_float(r.get("openInterest")),
                    "vol":    _safe_float(r.get("volume")),
                    "iv":     max(_safe_float(r.get("impliedVolatility")), 0.05),
                    "t":      t,
                })
        if puts is not None and not puts.empty:
            for _, r in puts.iterrows():
                rows.append({
                    "strike": _safe_float(r.get("strike")),
                    "right":  "P",
                    "oi":     _safe_float(r.get("openInterest")),
                    "vol":    _safe_float(r.get("volume")),
                    "iv":     max(_safe_float(r.get("impliedVolatility")), 0.05),
                    "t":      t,
                })
    if not rows:
        return out

    df = pd.DataFrame(rows)
    df = df[df["strike"] > 0]
    if df.empty:
        return out

    # HVL = strike with the largest OI*volume product across all rights
    df["liquidity"] = df["oi"] * df["vol"]
    liq = df.groupby("strike", as_index=False)["liquidity"].sum()
    if not liq.empty and liq["liquidity"].max() > 0:
        out["hvl"] = float(liq.loc[liq["liquidity"].idxmax(), "strike"])

    # Call/Put walls = max OI by side
    calls_only = df[df["right"] == "C"].groupby("strike", as_index=False)["oi"].sum()
    puts_only = df[df["right"] == "P"].groupby("strike", as_index=False)["oi"].sum()
    if not calls_only.empty and calls_only["oi"].max() > 0:
        out["call_wall"] = float(calls_only.loc[calls_only["oi"].idxmax(), "strike"])
    if not puts_only.empty and puts_only["oi"].max() > 0:
        out["put_wall"] = float(puts_only.loc[puts_only["oi"].idxmax(), "strike"])

    # ------------------------------------------------------------------
    # Zero-gamma flip (linearly interpolated)
    # ------------------------------------------------------------------
    g = rolled_gex.sort_values("strike").reset_index(drop=True).copy()
    g["cum"] = g["total_gex"].cumsum()
    sign = np.sign(g["cum"].values)
    flip = None
    for i in range(len(sign) - 1):
        a, b = sign[i], sign[i + 1]
        if a == 0 or b == 0 or a == b:
            continue
        x0, x1 = float(g.iloc[i]["strike"]), float(g.iloc[i + 1]["strike"])
        y0, y1 = float(g.iloc[i]["cum"]), float(g.iloc[i + 1]["cum"])
        if y1 == y0:
            flip = (x0 + x1) / 2
        else:
            flip = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
        break
    out["zero_gamma"] = float(flip) if flip is not None else None

    # Vol trigger = highest strike below spot whose cum-GEX is still negative
    below = g[g["strike"] < spot]
    if not below.empty:
        neg = below[below["cum"] < 0]
        if not neg.empty:
            out["vol_trigger"] = float(neg["strike"].max())

    # ------------------------------------------------------------------
    # Charm + Vanna peaks (per strike, OI-weighted, summed across expiries)
    # ------------------------------------------------------------------
    # Vectorised per (right, expiry) block. The previous version looped
    # .iterrows() inside a groupby — O(contracts) Python-level scipy calls,
    # which dominated the whole heatseeker request on a wide chain.
    chv_parts: list[pd.DataFrame] = []
    for (right, t_val), grp in df.groupby(["right", "t"], sort=False):
        strikes = grp["strike"].to_numpy(dtype=float)
        ivs = grp["iv"].to_numpy(dtype=float)
        oi = grp["oi"].to_numpy(dtype=float)
        g = greek_grid(spot, strikes, float(t_val), ivs, right=str(right))
        chv_parts.append(pd.DataFrame({
            "strike": strikes,
            "charm_w": np.abs(g["charm"]) * oi,
            "vanna_w": np.abs(g["vanna"]) * oi,
        }))
    if chv_parts:
        chv_df = (pd.concat(chv_parts, ignore_index=True)
                    .groupby("strike", as_index=False)[["charm_w", "vanna_w"]].sum())
        above = chv_df[chv_df["strike"] >= spot]
        below = chv_df[chv_df["strike"] < spot]
        if not above.empty and above["charm_w"].max() > 0:
            out["charm_high"] = float(above.loc[above["charm_w"].idxmax(), "strike"])
        if not below.empty and below["charm_w"].max() > 0:
            out["charm_low"] = float(below.loc[below["charm_w"].idxmax(), "strike"])
        if not above.empty and above["vanna_w"].max() > 0:
            out["vanna_high"] = float(above.loc[above["vanna_w"].idxmax(), "strike"])
        if not below.empty and below["vanna_w"].max() > 0:
            out["vanna_low"] = float(below.loc[below["vanna_w"].idxmax(), "strike"])

    # ------------------------------------------------------------------
    # King + Gatekeepers (Skylit naming)
    # ------------------------------------------------------------------
    g2 = rolled_gex.copy()
    g2["abs"] = g2["total_gex"].abs()
    g2 = g2.sort_values("abs", ascending=False).head(5)
    if not g2.empty:
        out["king"] = float(g2.iloc[0]["strike"])
        out["gatekeepers"] = [float(s) for s in g2.iloc[1:]["strike"].tolist()]

    return out
