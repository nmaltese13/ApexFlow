"""Real reactive key levels — the price levels traders actually defend / fade.

Replaces the previous "PivotPointsHighLow" study that produced essentially
random fractal pivots. Combines:

  prior_day_levels   PDH / PDL / PDC + ON H/L from 1h bars
  volume_profile     POC, VAH, VAL  (TPC volume profile, 70% value area)
  swing_levels       Recent swing highs/lows (Williams fractals, k=3)
  round_numbers      Psychological round-number magnets near spot
  hvn_lvn            High/Low volume nodes from the profile (acceptance / rejection)

Everything is derived from price/volume history so we do not depend on a
levels-feed and the levels are reactive to the actual tape.
"""
from __future__ import annotations

import math
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prior day / overnight
# ---------------------------------------------------------------------------
def prior_day_levels(daily: pd.DataFrame) -> Dict[str, float | None]:
    """Yesterday's H/L/C from a daily bar series."""
    out = {"pdh": None, "pdl": None, "pdc": None}
    if daily is None or daily.empty or len(daily) < 2:
        return out
    y = daily.iloc[-2]
    out["pdh"] = float(y["High"])
    out["pdl"] = float(y["Low"])
    out["pdc"] = float(y["Close"])
    return out


def overnight_levels(intraday_1h: pd.DataFrame) -> Dict[str, float | None]:
    """Overnight high / low from the most recent intraday bars before US open.

    Handled best-effort: if intraday data is missing (free providers often
    only return regular-session bars), returns ``None``s.
    """
    out = {"onh": None, "onl": None}
    if intraday_1h is None or intraday_1h.empty:
        return out
    try:
        last_session = intraday_1h.tail(48)  # ~2 days of 1h bars
        # Treat the after-hours window as the overnight piece
        if len(last_session) > 6:
            on = last_session.iloc[-6:]
            out["onh"] = float(on["High"].max())
            out["onl"] = float(on["Low"].min())
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Volume profile (TPO-style on price bins)
# ---------------------------------------------------------------------------
def volume_profile(intraday: pd.DataFrame, bins: int = 60,
                   value_area_pct: float = 0.70) -> Dict[str, Any]:
    """Build a price-binned volume profile and return POC / VAH / VAL.

    Parameters
    ----------
    intraday : DataFrame indexed by datetime with OHLCV columns.
    bins : number of price bins across the lookback range.
    value_area_pct : portion of total volume that defines the value area.

    Returns dict with keys: ``poc, vah, val, bins`` (list of {price, volume}).
    """
    out = {"poc": None, "vah": None, "val": None, "bins": []}
    if intraday is None or intraday.empty or len(intraday) < 5:
        return out

    df = intraday.dropna(subset=["High", "Low", "Volume"]).copy()
    if df.empty:
        return out

    lo = float(df["Low"].min())
    hi = float(df["High"].max())
    if hi <= lo:
        return out

    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vols = np.zeros(bins)

    # Spread each bar's volume across the bins it covers.
    for _, r in df.iterrows():
        bar_lo, bar_hi, vol = float(r["Low"]), float(r["High"]), float(r["Volume"] or 0)
        if vol <= 0 or bar_hi <= bar_lo:
            # Pin point bars to a single bin
            mid = (bar_lo + bar_hi) / 2 if bar_hi > 0 else bar_lo
            i = min(max(int((mid - lo) / (hi - lo) * bins), 0), bins - 1)
            vols[i] += vol
            continue
        i_lo = min(max(int((bar_lo - lo) / (hi - lo) * bins), 0), bins - 1)
        i_hi = min(max(int((bar_hi - lo) / (hi - lo) * bins), 0), bins - 1)
        n_covered = max(i_hi - i_lo + 1, 1)
        share = vol / n_covered
        vols[i_lo:i_hi + 1] += share

    if vols.sum() <= 0:
        return out

    poc_idx = int(np.argmax(vols))
    out["poc"] = float(centers[poc_idx])

    # 70% value area expanding outward from the POC
    total = vols.sum()
    target = total * value_area_pct
    inc = vols[poc_idx]
    lo_idx = hi_idx = poc_idx
    while inc < target and (lo_idx > 0 or hi_idx < bins - 1):
        next_lo = vols[lo_idx - 1] if lo_idx > 0 else -1
        next_hi = vols[hi_idx + 1] if hi_idx < bins - 1 else -1
        if next_lo >= next_hi and lo_idx > 0:
            lo_idx -= 1
            inc += vols[lo_idx]
        elif hi_idx < bins - 1:
            hi_idx += 1
            inc += vols[hi_idx]
        else:
            break

    out["val"] = float(centers[lo_idx])
    out["vah"] = float(centers[hi_idx])

    # Compact bin list (for client-side rendering of the profile bars)
    out["bins"] = [
        {"price": float(centers[i]), "volume": float(vols[i])}
        for i in range(bins) if vols[i] > 0
    ]
    return out


def hvn_lvn(profile_bins: List[Dict[str, float]], n: int = 3) -> Dict[str, List[float]]:
    """Pick top-N high-volume nodes (acceptance) and low-volume nodes (rejection)
    from a volume profile bin list. Skips trivial near-empty bins for LVN."""
    if not profile_bins:
        return {"hvn": [], "lvn": []}
    arr = sorted(profile_bins, key=lambda b: b["volume"], reverse=True)
    hvn = [b["price"] for b in arr[:n]]
    median = np.median([b["volume"] for b in profile_bins])
    nontrivial = [b for b in profile_bins if b["volume"] >= median * 0.05]
    lvn = sorted(nontrivial, key=lambda b: b["volume"])[:n]
    return {"hvn": hvn, "lvn": [b["price"] for b in lvn]}


# ---------------------------------------------------------------------------
# Swing pivots — Williams fractal (5-bar)
# ---------------------------------------------------------------------------
def swing_levels(daily: pd.DataFrame, k: int = 3, lookback: int = 60,
                 spot: float | None = None, max_above: int = 4,
                 max_below: int = 4) -> Dict[str, List[float]]:
    """Recent untested swing highs / lows.

    A swing high at bar i requires i's High > the High of the k bars on both
    sides. The level is *removed* from the result once a later bar trades
    through it (broken pivots are not real S/R anymore).
    """
    out = {"swing_highs": [], "swing_lows": []}
    if daily is None or daily.empty:
        return out

    df = daily.tail(max(lookback, 2 * k + 4)).reset_index(drop=True)
    if len(df) < 2 * k + 1:
        return out

    highs = df["High"].astype(float).values
    lows = df["Low"].astype(float).values
    closes = df["Close"].astype(float).values

    raw_highs = []
    raw_lows = []
    for i in range(k, len(df) - k):
        win_h = highs[i - k:i + k + 1]
        win_l = lows[i - k:i + k + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            raw_highs.append((i, float(highs[i])))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            raw_lows.append((i, float(lows[i])))

    # Remove pivots that have already been broken by a later close
    untested_highs: List[float] = []
    for idx, lvl in raw_highs:
        broken = (closes[idx + 1:] > lvl).any() if idx + 1 < len(df) else False
        if not broken:
            untested_highs.append(lvl)
    untested_lows: List[float] = []
    for idx, lvl in raw_lows:
        broken = (closes[idx + 1:] < lvl).any() if idx + 1 < len(df) else False
        if not broken:
            untested_lows.append(lvl)

    if spot is not None:
        above = sorted([h for h in untested_highs if h >= spot])[:max_above]
        below = sorted([l for l in untested_lows if l <= spot], reverse=True)[:max_below]
        out["swing_highs"] = above
        out["swing_lows"] = below
    else:
        out["swing_highs"] = sorted(set(untested_highs), reverse=True)[:max_above]
        out["swing_lows"] = sorted(set(untested_lows))[:max_below]
    return out


# ---------------------------------------------------------------------------
# Round numbers
# ---------------------------------------------------------------------------
def round_number_levels(spot: float, n: int = 4) -> List[float]:
    """Whole-number magnets near spot. Step size scales with price magnitude.

    Returns the ``n`` nearest round-number levels (mix of above / below).
    """
    if not spot or spot <= 0:
        return []
    if spot < 25:
        step = 1
    elif spot < 100:
        step = 5
    elif spot < 500:
        step = 10
    elif spot < 1500:
        step = 25
    else:
        step = 50
    base = round(spot / step) * step
    out = set()
    for i in range(-n, n + 1):
        v = base + i * step
        if v > 0:
            out.add(float(v))
    near = sorted(out, key=lambda v: abs(v - spot))[: 2 * n]
    return sorted(near, reverse=True)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def compute_key_levels(daily: pd.DataFrame, intraday: pd.DataFrame,
                        spot: float) -> Dict[str, Any]:
    """Bundle every reactive level the chart cares about."""
    out: Dict[str, Any] = {
        "pdh": None, "pdl": None, "pdc": None,
        "onh": None, "onl": None,
        "poc": None, "vah": None, "val": None,
        "swing_highs": [], "swing_lows": [],
        "hvn": [], "lvn": [],
        "round_numbers": [],
        "profile": [],
    }
    pd_lv = prior_day_levels(daily)
    out.update(pd_lv)

    on_lv = overnight_levels(intraday)
    out.update(on_lv)

    if intraday is not None and not intraday.empty:
        prof = volume_profile(intraday)
        out["poc"], out["vah"], out["val"] = prof["poc"], prof["vah"], prof["val"]
        out["profile"] = prof["bins"]
        nodes = hvn_lvn(prof["bins"])
        out["hvn"] = nodes["hvn"]
        out["lvn"] = nodes["lvn"]

    swings = swing_levels(daily, spot=spot)
    out["swing_highs"] = swings["swing_highs"]
    out["swing_lows"] = swings["swing_lows"]

    out["round_numbers"] = round_number_levels(spot)
    return out
