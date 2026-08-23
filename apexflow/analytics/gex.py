"""Gamma exposure (GEX) calculation from an options chain.

Convention (SqueezeMetrics / SpotGamma "naive dealer" model)::

    GEX_call_per_strike = +Gamma * OI * 100 * spot^2 * 0.01
    GEX_put_per_strike  = -Gamma * OI * 100 * spot^2 * 0.01
    Total GEX           = sum(GEX_call + GEX_put)

The 0.01 scales the result to "$ of dealer delta bought or sold per 1% move
in the underlying". Dealers are assumed long calls and short puts - the
standard market-maker positioning assumption used by most public GEX
dashboards, and an *assumption*, not an observation. Positive total means
dealers buy dips and sell rips (pinning, suppressed realised vol); negative
means they sell dips and buy rips (a volatility-amplifying regime). See
``dealer_greeks`` for the full derivation, the other four exposures, and the
switch that inverts the positioning assumption.

Implementation notes:

  * The per-strike math lives in :func:`dealer_greeks.chain_exposures` and
    this module is a thin adapter over it, so GEX cannot silently disagree
    with DEX/VEX computed from the same chain.
  * Time to expiry comes from :mod:`analytics.timeutil`, which resolves the
    real 16:00 ET close through ``zoneinfo`` rather than assuming a fixed
    UTC hour. The previous fixed 20:30 UTC constant was up to half an hour
    wrong in each direction, which materially distorts 0DTE gamma.
  * IV fallback uses the chain's own ATM IV rather than a hardcoded 30%.
  * NaN-safe throughout (yfinance returns NaN for many fields).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from .dealer_greeks import chain_exposures
from .timeutil import years_to_expiry as _years_to_expiry

__all__ = ["years_to_expiry", "bs_gamma", "chain_gex", "gex_summary", "heatmap_rows"]


def years_to_expiry(expiry: str, now: datetime | None = None) -> float:
    """Calendar years until the expiry's 16:00 ET close.

    Delegates to :func:`analytics.timeutil.years_to_expiry`; kept as a name
    here because several modules import it from ``gex``.
    """
    return _years_to_expiry(expiry, now=now)


def bs_gamma(spot: float, strike: float, t: float, iv: float,
             r: float = 0.04, q: float = 0.0) -> float:
    """Scalar Black-Scholes gamma. Kept for compatibility with other modules."""
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return float(math.exp(-q * t) * norm.pdf(d1) / (spot * iv * math.sqrt(t)))


def chain_gex(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, expiry: str,
              now: datetime | None = None, r: float = 0.04, q: float = 0.0) -> pd.DataFrame:
    """Per-strike GEX rolled up across one expiry.

    Columns: ``strike, call_gex, put_gex, total_gex``.
    """
    cols = ["strike", "call_gex", "put_gex", "total_gex"]
    exp = chain_exposures(calls, puts, spot, expiry, now=now, r=r, q=q)
    if exp.empty:
        return pd.DataFrame(columns=cols)
    out = exp[["strike", "call_gex", "put_gex", "gex"]].rename(columns={"gex": "total_gex"})
    return out.sort_values("strike").reset_index(drop=True)


def gex_summary(gex_df: pd.DataFrame, spot: float) -> dict:
    """Total GEX, the zero-gamma flip strike, and the extreme strikes.

    ``gamma_flip`` is where cumulative GEX (summed upward from the lowest
    strike) crosses zero, linearly interpolated between the bracketing
    strikes. Below it dealers amplify moves; above it they dampen them.
    """
    if gex_df is None or gex_df.empty:
        return {"total_gex": 0.0, "gamma_flip": None,
                "max_pos_strike": None, "max_neg_strike": None, "spot": spot}
    total = float(gex_df["total_gex"].sum())
    cum = gex_df.sort_values("strike").reset_index(drop=True).copy()
    cum["cum_gex"] = cum["total_gex"].cumsum()
    flip = None
    sign_changes = np.where(np.diff(np.sign(cum["cum_gex"].to_numpy())))[0]
    if len(sign_changes):
        idx = int(sign_changes[0])
        a, b = cum.iloc[idx], cum.iloc[idx + 1]
        y0, y1 = float(a["cum_gex"]), float(b["cum_gex"])
        x0, x1 = float(a["strike"]), float(b["strike"])
        flip = float((x0 + x1) / 2) if y1 == y0 else float(x0 - y0 * (x1 - x0) / (y1 - y0))
    g = gex_df.reset_index(drop=True)
    return {
        "total_gex": total,
        "gamma_flip": flip,
        "max_pos_strike": float(g.loc[g["total_gex"].idxmax(), "strike"]),
        "max_neg_strike": float(g.loc[g["total_gex"].idxmin(), "strike"]),
        "spot": spot,
    }


def heatmap_rows(gex_df: pd.DataFrame, spot: float, n_strikes: int = 15) -> list[tuple]:
    """Return list of (strike, total_gex, bar_string) ordered by distance from spot."""
    if gex_df is None or gex_df.empty:
        return []
    g = gex_df.copy()
    g["dist"] = (g["strike"] - spot).abs()
    g = g.nsmallest(n_strikes, "dist").sort_values("strike", ascending=False)
    max_abs = g["total_gex"].abs().max() or 1.0
    rows = []
    for _, r in g.iterrows():
        norm_w = int(round(20 * abs(r["total_gex"]) / max_abs))
        bar = ("#" * norm_w).ljust(20) if r["total_gex"] >= 0 else ("-" * norm_w).ljust(20)
        rows.append((float(r["strike"]), float(r["total_gex"]), bar))
    return rows
