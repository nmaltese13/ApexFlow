"""GEX profile classifier — names the shape of the gamma landscape.

Each label describes how dealer gamma is *distributed* across strikes. The
distribution implies where hedging flow would come from; it does not imply
what price will do, and nothing here is a trade recommendation.

  * **Wall**   — gamma concentrated at a single strike, with light
                  neighbours. Hedging flow is focused at one level, so that
                  level tends to act as static support/resistance.
  * **Pillar** — the same strike stays heavy across several expiry buckets.
                  The most persistent structure, because dealers re-stack
                  there as old expiries roll off, so the level survives
                  individual expirations.
  * **Slide**  — gamma ramps smoothly across strikes rather than clustering.
                  The slope says which direction dealer hedging grows into.
  * **Pin**    — heavy *positive* gamma at a strike very close to spot.
                  Positive gamma means dealers sell strength and buy
                  weakness, which mechanically compresses realised vol
                  around that level.

Plus a **violence flag** for negative-gamma zones near spot: where the net
gamma around spot is negative, dealer hedging runs with the move rather than
against it, so realised vol expands instead of compressing.

Every label is conditional on the dealer-positioning assumption documented
in ``dealer_greeks`` — invert that assumption and the signs invert with it.

The classifier is pure-pandas and works on the same per-strike GEX DataFrame
that chain_gex() produces, so it slots in anywhere we already compute GEX.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-strike profile tags
# ---------------------------------------------------------------------------
@dataclass
class StrikeProfile:
    strike: float
    total_gex: float
    abs_pct_of_max: float    # |gex| relative to chain max, 0..1
    tags: list[str] = field(default_factory=list)
    score: float = 0.0       # 0..100 — how loud this profile is

    def to_dict(self) -> dict:
        return {
            "strike": float(self.strike),
            "total_gex": float(self.total_gex),
            "abs_pct_of_max": float(self.abs_pct_of_max),
            "tags": list(self.tags),
            "score": float(self.score),
        }


# ---------------------------------------------------------------------------
# Single-chain classification (one expiry's worth of GEX)
# ---------------------------------------------------------------------------
def classify_walls(gex_df: pd.DataFrame, min_pct_of_max: float = 0.55,
                   max_neighbor_ratio: float = 0.4) -> list[float]:
    """Return strikes whose gamma is *concentrated* — much heavier than their
    neighbors (a wall stands alone).

    A strike is a "wall" when:
      * |GEX| is at least ``min_pct_of_max`` of the chain's max
      * Both immediate neighbors have |GEX| ≤ max_neighbor_ratio × this strike
    """
    if gex_df is None or gex_df.empty:
        return []
    g = gex_df.copy().sort_values("strike").reset_index(drop=True)
    g["abs_gex"] = g["total_gex"].abs()
    max_abs = g["abs_gex"].max()
    if max_abs <= 0:
        return []

    walls: list[float] = []
    for i, row in g.iterrows():
        if row["abs_gex"] / max_abs < min_pct_of_max:
            continue
        left  = g.iloc[i - 1]["abs_gex"] if i > 0 else 0
        right = g.iloc[i + 1]["abs_gex"] if i < len(g) - 1 else 0
        center = row["abs_gex"]
        # Both neighbors small relative to this strike → wall
        if (left  / center <= max_neighbor_ratio and
            right / center <= max_neighbor_ratio):
            walls.append(float(row["strike"]))
    return walls


def classify_slide(gex_df: pd.DataFrame, min_r2: float = 0.5) -> dict:
    """Detect directional 'slide' — a smooth ramp of gamma across strikes.

    Fits a linear model ``GEX ~ strike`` and reports slope + R². When R² is
    high and slope is meaningfully non-zero, dealers are positioned along a
    slope: positive slope means more positive gamma at higher strikes
    (resistance accelerates as price rises).

    Returns dict with keys: slope, intercept, r2, direction ('up'/'down'/'flat').
    """
    if gex_df is None or gex_df.empty or len(gex_df) < 5:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "direction": "flat"}
    g = gex_df.copy().sort_values("strike")
    x = g["strike"].to_numpy(dtype=float)
    y = g["total_gex"].to_numpy(dtype=float)
    if x.std() == 0 or y.std() == 0:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "direction": "flat"}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    direction = "flat"
    if r2 >= min_r2:
        if slope > 0:
            direction = "up"
        elif slope < 0:
            direction = "down"
    return {"slope": float(slope), "intercept": float(intercept),
             "r2": r2, "direction": direction}


def classify_pin(gex_df: pd.DataFrame, spot: float,
                 min_pct_of_max: float = 0.7,
                 max_dist_pct: float = 0.005) -> float | None:
    """Detect a pin candidate: a heavy positive-gamma strike near spot.

    Returns the pin strike if found, else None. Criteria:
      * Strike within ``max_dist_pct`` of spot (default 0.5%)
      * total_gex > 0 (positive dealer gamma → pinning behavior)
      * |GEX| ≥ ``min_pct_of_max`` of the chain max
    """
    if gex_df is None or gex_df.empty or spot <= 0:
        return None
    g = gex_df.copy()
    g["abs_gex"] = g["total_gex"].abs()
    max_abs = g["abs_gex"].max()
    if max_abs <= 0:
        return None
    g["dist_pct"] = (g["strike"] - spot).abs() / spot
    cand = g[(g["dist_pct"] <= max_dist_pct) &
             (g["total_gex"] > 0) &
             (g["abs_gex"] / max_abs >= min_pct_of_max)]
    if cand.empty:
        return None
    pin = cand.nlargest(1, "abs_gex").iloc[0]
    return float(pin["strike"])


def violence_flag(gex_df: pd.DataFrame, spot: float,
                   window_pct: float = 0.03) -> dict:
    """Flag for the @t38p_flow framework: 'negative GEX zones cause violent
    action'. Sums the signed GEX in a band around spot (±window_pct) — when
    that net sum is meaningfully negative, dealer hedging amplifies moves
    (especially downside).

    Returns dict: net_gex_near_spot, ratio_neg, violent (bool).
    """
    if gex_df is None or gex_df.empty or spot <= 0:
        return {"net_gex_near_spot": 0.0, "ratio_neg": 0.0, "violent": False}
    g = gex_df.copy()
    g["dist_pct"] = (g["strike"] - spot).abs() / spot
    near = g[g["dist_pct"] <= window_pct]
    if near.empty:
        return {"net_gex_near_spot": 0.0, "ratio_neg": 0.0, "violent": False}
    net = float(near["total_gex"].sum())
    abs_total = float(near["total_gex"].abs().sum() or 1.0)
    ratio_neg = float(near[near["total_gex"] < 0]["total_gex"].abs().sum()) / abs_total
    # "Violent" = net negative AND >60% of nearby gamma is negative
    violent = (net < 0) and (ratio_neg > 0.60)
    return {"net_gex_near_spot": net, "ratio_neg": ratio_neg, "violent": violent}


# ---------------------------------------------------------------------------
# Multi-DTE pillar detection — needs 2+ expiry chains to compute
# ---------------------------------------------------------------------------
def classify_pillars(per_dte_gex: dict[str, pd.DataFrame],
                     min_dtes: int = 2,
                     min_pct_of_max: float = 0.40,
                     strike_tolerance: float = 0.005) -> list[dict]:
    """Find strikes that show up heavy across multiple expiry buckets.

    ``per_dte_gex`` is a dict of {bucket_name: gex_df} where each frame has
    columns 'strike' + 'total_gex'.

    A *pillar* is a strike that is in the top N% of GEX in ≥``min_dtes``
    different DTE buckets. These are the most reliable magnets because dealers
    keep re-stacking gamma there as old expiries roll off.

    Returns list of {strike, n_dtes, dtes, total_abs_gex} sorted by n_dtes desc.
    """
    if not per_dte_gex:
        return []

    # Pull each bucket's "heavy" strikes (top by |gex| above min_pct_of_max)
    heavy_per_dte: dict[str, list[float]] = {}
    for dte_name, gex_df in per_dte_gex.items():
        if gex_df is None or gex_df.empty:
            continue
        g = gex_df.copy()
        g["abs_gex"] = g["total_gex"].abs()
        max_abs = g["abs_gex"].max()
        if max_abs <= 0:
            continue
        heavy = g[g["abs_gex"] / max_abs >= min_pct_of_max]
        heavy_per_dte[dte_name] = heavy["strike"].astype(float).tolist()

    # Bucket strikes that appear in multiple DTEs (within tolerance)
    pillars: dict[float, dict] = {}
    for dte_name, strikes in heavy_per_dte.items():
        for k in strikes:
            # Find an existing pillar this k could merge with
            merged = False
            for existing in list(pillars.keys()):
                if abs(k - existing) / max(abs(existing), 1.0) <= strike_tolerance:
                    pillars[existing]["dtes"].add(dte_name)
                    merged = True
                    break
            if not merged:
                pillars[k] = {"strike": k, "dtes": {dte_name}}

    # Tally and filter
    out = []
    for k, info in pillars.items():
        if len(info["dtes"]) < min_dtes:
            continue
        # Sum |gex| of this strike across all contributing DTEs
        tot = 0.0
        for dte_name in info["dtes"]:
            df = per_dte_gex.get(dte_name)
            if df is None or df.empty:
                continue
            row = df[(df["strike"] - k).abs() / max(abs(k), 1.0) <= strike_tolerance]
            if not row.empty:
                tot += float(row["total_gex"].abs().sum())
        out.append({
            "strike": float(k),
            "n_dtes": len(info["dtes"]),
            "dtes": sorted(info["dtes"]),
            "total_abs_gex": tot,
        })
    out.sort(key=lambda r: (r["n_dtes"], r["total_abs_gex"]), reverse=True)
    return out


# ---------------------------------------------------------------------------
# One-shot orchestrator — bundle every profile signal for a chain
# ---------------------------------------------------------------------------
def profile_chain(gex_df: pd.DataFrame, spot: float) -> dict:
    """Compute every single-chain profile signal in one call.

    Returns:
        {
          walls:    [strike, ...],
          slide:    {slope, intercept, r2, direction},
          pin:      strike or None,
          violence: {net_gex_near_spot, ratio_neg, violent},
        }
    """
    return {
        "walls":    classify_walls(gex_df),
        "slide":    classify_slide(gex_df),
        "pin":      classify_pin(gex_df, spot),
        "violence": violence_flag(gex_df, spot),
    }
