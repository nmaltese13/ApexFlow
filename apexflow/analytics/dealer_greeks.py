"""Aggregate dealer exposure: DEX, GEX, VEX, charm and vanna exposure.

``gex.py`` answers one question - how much dealer gamma sits at each strike.
This module answers the other four, from the same chain, under the same
stated positioning assumption, so the whole dealer-Greek surface is
computed once and consistently.

The positioning assumption
--------------------------
Every one of these numbers is a *model output conditioned on an assumption
about who holds what*. Open interest tells you a contract exists; it does
not tell you which side the market maker is on. The convention used here
(and by SqueezeMetrics, SpotGamma and most public GEX dashboards) is the
"naive dealer" model:

    dealers are **long call open interest** and **short put open interest**

The reasoning: the typical retail/institutional customer sells calls
(covered calls, overwriting) and buys puts (portfolio protection), so the
dealer takes the other side of both. This is a coarse assumption and it is
wrong for individual names - a ticker where customers are aggressively
buying calls has dealers *short* calls, which flips the sign of everything
below. Pass ``convention="inverted"`` to model that case, or
``convention="all_short"`` for the "dealers are short whatever customers
bought" reading. The default stays ``"naive"`` so numbers are comparable
with published dashboards.

Do not read these outputs as a forecast. They describe where hedging flow
would come from *if* the assumption holds, which is a statement about market
structure, not about what price will do.

Definitions
-----------
With ``sign = +1`` for calls and ``-1`` for puts under the naive convention,
``OI`` the open interest and ``100`` the contract multiplier:

============  ==========================================  ==================================
Exposure      Formula                                     Units
============  ==========================================  ==================================
DEX           ``sign * delta * OI * 100 * S``              $ of stock dealers hold per strike
GEX           ``sign * gamma * OI * 100 * S^2 * 0.01``     $ of delta bought/sold per +1% in S
VEX           ``sign * vega * OI * 100``                   $ of P&L per +1 vol point
Charm exp.    ``sign * charm * OI * 100 * S / 365``        $ of delta to re-hedge per day
Vanna exp.    ``sign * vanna * OI * 100 * S * 0.01``       $ of delta to re-hedge per +1 vol pt
============  ==========================================  ==================================

Reading them
------------
* **DEX** - net dealer stock position implied by the book. Large positive
  DEX means dealers are long a lot of stock as a hedge; that inventory has
  to be sold if the options that justify it roll off.
* **GEX** - the sign of the hedging feedback loop. Positive: dealers sell
  strength and buy weakness, damping realised vol. Negative: they chase,
  amplifying it.
* **VEX** - P&L sensitivity to a vol move, and therefore how much vega
  dealers must buy back if IV rises. Strongly negative VEX into a catalyst
  is the setup where a vol spike forces dealers to buy options near spot,
  which also buys them gamma and changes the hedging regime mid-move.
* **Charm exposure** - hedge drift from time alone. On expiry afternoons
  OTM deltas collapse, and dealers unwind the shares backing them; this is
  the mechanic behind the familiar OPEX-afternoon drift.
* **Vanna exposure** - hedge drift from vol alone. When IV falls, put
  deltas shrink, and dealers who were short stock against them buy it back.
  This is why a vol crush after an event so often comes with a grind up
  that has no news attached to it.

Everything is vectorised over the chain; nothing iterates rows.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from .timeutil import years_to_expiry

__all__ = [
    "CONTRACT_MULTIPLIER", "dealer_signs", "chain_exposures", "roll_up",
    "exposure_summary", "vex_summary", "greek_grid",
    "net_gex_curve", "gamma_flip_level",
]

CONTRACT_MULTIPLIER = 100.0

Convention = Literal["naive", "inverted", "all_short"]

_EXPOSURE_COLS = ("dex", "gex", "vex", "charm_exp", "vanna_exp")


def dealer_signs(convention: Convention = "naive") -> tuple[float, float]:
    """(call_sign, put_sign) for the chosen dealer-positioning assumption.

    naive       dealers long calls, short puts (the published-dashboard
                convention; customers overwrite calls and buy protection)
    inverted    dealers short calls, long puts (customers are the call
                buyers - the right reading for a name in a call-buying mania)
    all_short   dealers short both sides (customers bought everything)
    """
    if convention == "naive":
        return +1.0, -1.0
    if convention == "inverted":
        return -1.0, +1.0
    if convention == "all_short":
        return -1.0, -1.0
    raise ValueError(f"unknown convention {convention!r}")


# ---------------------------------------------------------------------------
# Vectorised Greeks (kept local so this module has no import cycle with greeks.py)
# ---------------------------------------------------------------------------
def _d1_d2(spot: float, K: np.ndarray, t: float, sigma: np.ndarray,
           r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    sqrt_t = math.sqrt(t)
    d1 = (np.log(spot / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def greek_grid(spot: float, strikes: np.ndarray, t: float, sigma: np.ndarray,
               right: str = "C", r: float = 0.04, q: float = 0.0) -> dict[str, np.ndarray]:
    """Per-contract delta / gamma / vega / charm / vanna for an array of strikes.

    Units match ``analytics.greeks``: vega is per +1 vol point, charm is
    per year (callers divide by 365 for a daily figure), vanna is per +1.0
    of sigma (callers scale by 0.01 for a vol point).
    """
    K = np.asarray(strikes, dtype=float)
    s = np.asarray(sigma, dtype=float)
    zeros = np.zeros_like(K, dtype=float)
    out = {k: zeros.copy() for k in ("delta", "gamma", "vega", "charm", "vanna")}
    if spot <= 0 or t <= 0:
        return out
    valid = (K > 0) & (s > 0) & np.isfinite(K) & np.isfinite(s)
    if not valid.any():
        return out

    Kv, sv = K[valid], s[valid]
    sqrt_t = math.sqrt(t)
    d1, d2 = _d1_d2(spot, Kv, t, sv, r, q)
    pdf = norm.pdf(d1)
    disc_q = math.exp(-q * t)

    if right.upper().startswith("C"):
        delta = disc_q * norm.cdf(d1)
        charm = (-q * disc_q * norm.cdf(d1)
                 + disc_q * pdf * (2 * (r - q) * t - d2 * sv * sqrt_t) / (2 * t * sv * sqrt_t))
    else:
        delta = -disc_q * norm.cdf(-d1)
        charm = (q * disc_q * norm.cdf(-d1)
                 + disc_q * pdf * (2 * (r - q) * t - d2 * sv * sqrt_t) / (2 * t * sv * sqrt_t))

    gamma = disc_q * pdf / (spot * sv * sqrt_t)
    vega = spot * disc_q * pdf * sqrt_t * 0.01     # per +1 vol point
    vanna = -disc_q * pdf * d2 / sv                # d(delta)/d(sigma), per 1.0 sigma

    for name, arr in (("delta", delta), ("gamma", gamma), ("vega", vega),
                      ("charm", charm), ("vanna", vanna)):
        buf = out[name]
        buf[valid] = np.where(np.isfinite(arr), arr, 0.0)
    return out


# ---------------------------------------------------------------------------
# Chain-level exposure
# ---------------------------------------------------------------------------
def _atm_iv(calls: pd.DataFrame | None, puts: pd.DataFrame | None, spot: float) -> float:
    """Median IV of the three strikes nearest spot, both sides pooled."""
    candidates = []
    for df in (calls, puts):
        if df is None or df.empty or "impliedVolatility" not in df.columns:
            continue
        d = df.copy()
        d["impliedVolatility"] = pd.to_numeric(d["impliedVolatility"], errors="coerce")
        d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
        d = d.dropna(subset=["impliedVolatility", "strike"])
        d = d[d["impliedVolatility"] > 0]
        if d.empty:
            continue
        d["dist"] = (d["strike"] - spot).abs()
        candidates.append(d.nsmallest(3, "dist"))
    if not candidates:
        return 0.0
    return float(pd.concat(candidates)["impliedVolatility"].median())


def _side_exposures(df: pd.DataFrame | None, spot: float, t: float, right: str,
                    sign: float, fallback_iv: float, r: float, q: float,
                    oi_col: str) -> pd.DataFrame:
    cols = ["strike", *_EXPOSURE_COLS, "oi"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["strike"])
    d = d[d["strike"] > 0]
    if d.empty:
        return pd.DataFrame(columns=cols)

    iv = pd.to_numeric(d.get("impliedVolatility", 0), errors="coerce")
    # Missing or zero IV falls back to the chain's ATM IV; floor at 1% so the
    # 1/sigma terms in gamma and vanna stay finite.
    iv = iv.where(iv > 0, fallback_iv).clip(lower=0.01).to_numpy()
    oi = pd.to_numeric(d.get(oi_col, 0), errors="coerce").fillna(0.0).to_numpy()

    g = greek_grid(spot, d["strike"].to_numpy(), t, iv, right, r, q)
    notional = sign * oi * CONTRACT_MULTIPLIER

    out = pd.DataFrame({
        "strike": d["strike"].to_numpy(),
        "dex": notional * g["delta"] * spot,
        "gex": notional * g["gamma"] * (spot ** 2) * 0.01,
        "vex": notional * g["vega"],
        "charm_exp": notional * g["charm"] * spot / 365.0,
        "vanna_exp": notional * g["vanna"] * spot * 0.01,
        "oi": oi,
    })
    return out.groupby("strike", as_index=False).sum()


def chain_exposures(calls: pd.DataFrame | None, puts: pd.DataFrame | None,
                    spot: float, expiry: str, now: datetime | None = None,
                    convention: Convention = "naive", r: float = 0.04,
                    q: float = 0.0, oi_col: str = "openInterest") -> pd.DataFrame:
    """Per-strike dealer exposures for one expiry.

    Returns a frame with columns::

        strike, call_dex, put_dex, dex, call_gex, put_gex, gex,
        call_vex, put_vex, vex, charm_exp, vanna_exp, call_oi, put_oi

    ``gex`` here matches ``gex.chain_gex``'s ``total_gex`` to within floating
    point under the default convention - ``tests/test_dealer_greeks.py``
    pins that agreement, so the two engines cannot silently drift apart.

    Pass ``oi_col="volume"`` to compute exposures from the day's volume
    instead of resting open interest, which approximates *newly added*
    positioning rather than the accumulated book.
    """
    base_cols = ["strike",
                 "call_dex", "put_dex", "dex",
                 "call_gex", "put_gex", "gex",
                 "call_vex", "put_vex", "vex",
                 "charm_exp", "vanna_exp", "call_oi", "put_oi"]
    empty = pd.DataFrame(columns=base_cols)
    if spot <= 0:
        return empty
    if (calls is None or calls.empty) and (puts is None or puts.empty):
        return empty

    t = years_to_expiry(expiry, now=now)
    fallback_iv = _atm_iv(calls, puts, spot) or 0.30
    call_sign, put_sign = dealer_signs(convention)

    c = _side_exposures(calls, spot, t, "C", call_sign, fallback_iv, r, q, oi_col)
    p = _side_exposures(puts, spot, t, "P", put_sign, fallback_iv, r, q, oi_col)
    if c.empty and p.empty:
        return empty

    c = c.rename(columns={k: f"call_{k}" for k in (*_EXPOSURE_COLS, "oi")})
    p = p.rename(columns={k: f"put_{k}" for k in (*_EXPOSURE_COLS, "oi")})
    out = pd.merge(c, p, on="strike", how="outer")
    for col in out.columns:
        if col != "strike":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for name in _EXPOSURE_COLS:
        out[name] = out[f"call_{name}"] + out[f"put_{name}"]
    out = out.drop(columns=["call_charm_exp", "put_charm_exp",
                            "call_vanna_exp", "put_vanna_exp"], errors="ignore")
    ordered = [c_ for c_ in base_cols if c_ in out.columns]
    return out[ordered].sort_values("strike").reset_index(drop=True)


def roll_up(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Sum several expiries' exposure frames onto a shared strike axis."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["strike", "dex", "gex", "vex", "charm_exp", "vanna_exp"])
    joined = pd.concat(frames, ignore_index=True)
    return joined.groupby("strike", as_index=False).sum().sort_values("strike").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _zero_cross(strikes: np.ndarray, cumulative: np.ndarray) -> float | None:
    """First strike where a cumulative series crosses zero, interpolated."""
    if len(strikes) < 2:
        return None
    sign = np.sign(cumulative)
    for i in range(len(sign) - 1):
        a, b = sign[i], sign[i + 1]
        if a == 0 or b == 0 or a == b:
            continue
        x0, x1 = float(strikes[i]), float(strikes[i + 1])
        y0, y1 = float(cumulative[i]), float(cumulative[i + 1])
        return (x0 + x1) / 2 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
    return None


def exposure_summary(exp_df: pd.DataFrame, spot: float) -> dict:
    """Headline totals and the structurally interesting strikes.

    ``gamma_flip`` / ``vega_flip`` are the levels where the cumulative
    exposure (summed from the lowest strike up) changes sign - the boundary
    between the two hedging regimes.
    """
    keys = ("total_dex", "total_gex", "total_vex", "total_charm", "total_vanna")
    out: dict = {k: 0.0 for k in keys}
    out.update({"gamma_flip": None, "vega_flip": None, "spot": float(spot),
                "max_gex_strike": None, "min_gex_strike": None,
                "max_vex_strike": None, "min_vex_strike": None,
                "regime": "unknown"})
    if exp_df is None or exp_df.empty:
        return out

    g = exp_df.sort_values("strike").reset_index(drop=True)
    strikes = g["strike"].to_numpy(dtype=float)
    for src, dst in (("dex", "total_dex"), ("gex", "total_gex"), ("vex", "total_vex"),
                     ("charm_exp", "total_charm"), ("vanna_exp", "total_vanna")):
        if src in g.columns:
            out[dst] = float(g[src].sum())

    if "gex" in g.columns:
        out["gamma_flip"] = _zero_cross(strikes, g["gex"].cumsum().to_numpy())
        out["max_gex_strike"] = float(g.loc[g["gex"].idxmax(), "strike"])
        out["min_gex_strike"] = float(g.loc[g["gex"].idxmin(), "strike"])
        out["regime"] = "positive_gamma" if out["total_gex"] > 0 else "negative_gamma"
    if "vex" in g.columns:
        out["vex_flip"] = _zero_cross(strikes, g["vex"].cumsum().to_numpy())
        out["vega_flip"] = out["vex_flip"]
        out["max_vex_strike"] = float(g.loc[g["vex"].idxmax(), "strike"])
        out["min_vex_strike"] = float(g.loc[g["vex"].idxmin(), "strike"])
    return out


def vex_summary(exp_df: pd.DataFrame, spot: float, window_pct: float = 0.05) -> dict:
    """Vega-exposure detail, including the concentration near spot.

    ``net_vex_near_spot`` is the vega dealers carry within ``window_pct`` of
    spot - the part of the book most likely to be re-hedged on a vol move,
    since those are the strikes whose vega is largest and most unstable.

    ``dollars_per_vol_point`` restates total VEX in plain terms: the P&L
    swing dealers take for a 1-point move in implied vol under the stated
    positioning assumption.
    """
    out = {"total_vex": 0.0, "net_vex_near_spot": 0.0, "pct_near_spot": 0.0,
           "dollars_per_vol_point": 0.0, "net_short_vega": False}
    if exp_df is None or exp_df.empty or "vex" not in exp_df.columns or spot <= 0:
        return out
    g = exp_df.copy()
    total = float(g["vex"].sum())
    g["dist_pct"] = (g["strike"] - spot).abs() / spot
    near = g[g["dist_pct"] <= window_pct]
    near_sum = float(near["vex"].sum()) if not near.empty else 0.0
    abs_total = float(g["vex"].abs().sum()) or 1.0
    out.update({
        "total_vex": total,
        "net_vex_near_spot": near_sum,
        "pct_near_spot": float(near["vex"].abs().sum() / abs_total) if not near.empty else 0.0,
        "dollars_per_vol_point": total,
        "net_short_vega": total < 0,
    })
    return out


# ---------------------------------------------------------------------------
# Zero-gamma level, computed properly
# ---------------------------------------------------------------------------
def net_gex_curve(chains: Iterable[tuple[str, pd.DataFrame | None, pd.DataFrame | None]],
                  spots: np.ndarray, now: datetime | None = None,
                  convention: Convention = "naive", r: float = 0.04,
                  q: float = 0.0, oi_col: str = "openInterest") -> np.ndarray:
    """Total dealer GEX evaluated at each hypothetical spot price.

    ``chains`` is an iterable of ``(expiry, calls, puts)``. For every price
    in ``spots`` the whole book is re-priced - gamma is recomputed with that
    price as the underlying - and the exposures summed. IV is held fixed per
    contract, so this is a pure "what does the gamma profile look like if
    price were there today" curve, not a forecast.
    """
    spots = np.asarray(spots, dtype=float)
    totals = np.zeros_like(spots)
    call_sign, put_sign = dealer_signs(convention)

    prepared: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for expiry, calls, puts in chains:
        t = years_to_expiry(expiry, now=now)
        ref_spot = float(spots[len(spots) // 2]) if spots.size else 0.0
        fallback_iv = _atm_iv(calls, puts, ref_spot) or 0.30
        for df, sign in ((calls, call_sign), (puts, put_sign)):
            if df is None or df.empty:
                continue
            d = df.copy()
            d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
            d = d.dropna(subset=["strike"])
            d = d[d["strike"] > 0]
            if d.empty:
                continue
            iv = pd.to_numeric(d.get("impliedVolatility", 0), errors="coerce")
            iv = iv.where(iv > 0, fallback_iv).clip(lower=0.01).to_numpy()
            oi = pd.to_numeric(d.get(oi_col, 0), errors="coerce").fillna(0.0).to_numpy()
            prepared.append((t, d["strike"].to_numpy(dtype=float), iv, oi, sign))

    if not prepared:
        return totals

    for i, s in enumerate(spots):
        if s <= 0:
            continue
        acc = 0.0
        for t, K, iv, oi, sign in prepared:
            # Gamma is right/side-independent, so one grid call covers both.
            g = greek_grid(s, K, t, iv, right="C", r=r, q=q)["gamma"]
            acc += float(np.sum(sign * g * oi * CONTRACT_MULTIPLIER * (s ** 2) * 0.01))
        totals[i] = acc
    return totals


def gamma_flip_level(chains: Iterable[tuple[str, pd.DataFrame | None, pd.DataFrame | None]],
                     spot: float, span_pct: float = 0.25, n_points: int = 81,
                     now: datetime | None = None, convention: Convention = "naive",
                     r: float = 0.04, q: float = 0.0) -> dict:
    """The price at which total dealer gamma flips sign.

    Why this is not the same as the cumulative-sum method
    -----------------------------------------------------
    ``gex_summary`` (and the ``gamma_flip`` field in
    :func:`exposure_summary`) finds where GEX *accumulated from the lowest
    strike upward* crosses zero. That is cheap and it is what a lot of code
    does, but the number it returns is a strike, not a price level, and on a
    put-heavy index chain it lands far below spot: cumulative GEX starts
    deeply negative because of the put wall at the bottom of the chain, and
    crosses zero as soon as enough call gamma accumulates - which can be
    hundreds of points below where dealer gamma actually turns positive.

    The correct question is: *at what underlying price would total dealer
    gamma be zero?* That requires re-pricing the whole book at candidate
    spot levels, because every contract's gamma depends on where spot is.
    This function does that over a grid spanning ``+/- span_pct`` around
    spot and interpolates the crossing.

    Returns ``{flip, curve, spot, total_gex_at_spot, regime, bracketed}``.
    ``bracketed`` is False when no sign change exists in the searched range -
    the honest answer there is "no flip within +/-25%", not a number.
    """
    out = {"flip": None, "curve": [], "spot": float(spot),
           "total_gex_at_spot": 0.0, "regime": "unknown", "bracketed": False}
    if spot <= 0:
        return out

    lo, hi = spot * (1 - span_pct), spot * (1 + span_pct)
    grid = np.linspace(lo, hi, int(n_points))
    chains = list(chains)
    curve = net_gex_curve(chains, grid, now=now, convention=convention, r=r, q=q)

    out["curve"] = [{"spot": float(s), "total_gex": float(g)} for s, g in zip(grid, curve)]
    at_spot = float(np.interp(spot, grid, curve))
    out["total_gex_at_spot"] = at_spot
    out["regime"] = "positive_gamma" if at_spot > 0 else "negative_gamma"

    sign = np.sign(curve)
    crossings = np.where(np.diff(sign) != 0)[0]
    if crossings.size:
        # Prefer the crossing nearest spot — a chain can have several.
        best = min(crossings, key=lambda i: abs((grid[i] + grid[i + 1]) / 2 - spot))
        x0, x1 = float(grid[best]), float(grid[best + 1])
        y0, y1 = float(curve[best]), float(curve[best + 1])
        out["flip"] = (x0 + x1) / 2 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0)
        out["bracketed"] = True
    return out
