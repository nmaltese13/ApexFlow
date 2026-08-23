"""Squeeze Radar — one composite ranker that says 'these are about to pop'.

Layers six independent edges into a single 0-100 score:

  1. Short-squeeze pressure   (existing squeeze.py composite, normalised)
  2. GEX wall proximity       (close to a heavy negative wall → flush risk;
                               close to a heavy positive wall → magnet/breakout)
  3. Compression               (TTM squeeze + BB bandwidth shrinkage)
  4. RVOL spike                (5-day avg / 20-day avg volume)
  5. Catalyst proximity        (earnings within ≤7 days)
  6. Trend posture             (distance from 20EMA + slope)

Each layer outputs 0-100 independently; the radar score is a weighted average
plus a "synergy" bonus when ≥3 layers fire ≥60. The reason string spells out
exactly which edges contributed so you can audit it instead of trusting a
black-box number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from apexflow.analytics import indicators as ind
from apexflow.analytics.gex import chain_gex, gex_summary
from apexflow.analytics.gex_profile import (
    profile_chain, classify_walls, violence_flag,
)
from apexflow.analytics.squeeze import SqueezeInputs, squeeze_score
from apexflow.models import Signal
from .base import BaseScanner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer weights (sum should be 100 before synergy bonus)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "short":       25,   # short squeeze potential
    "gex":         20,   # gamma wall positioning
    "compression": 15,   # coiled spring
    "rvol":        15,   # volume spike
    "catalyst":    15,   # earnings proximity
    "trend":       10,   # trend posture
}
SYNERGY_THRESHOLD = 60.0
SYNERGY_BONUS_PER_LAYER = 5.0


@dataclass
class RadarBreakdown:
    short: float = 0.0
    gex: float = 0.0
    compression: float = 0.0
    rvol: float = 0.0
    catalyst: float = 0.0
    trend: float = 0.0
    synergy: float = 0.0

    def composite(self) -> float:
        wsum = sum(WEIGHTS.values())
        weighted = (
            self.short * WEIGHTS["short"] +
            self.gex * WEIGHTS["gex"] +
            self.compression * WEIGHTS["compression"] +
            self.rvol * WEIGHTS["rvol"] +
            self.catalyst * WEIGHTS["catalyst"] +
            self.trend * WEIGHTS["trend"]
        ) / wsum
        return float(min(100.0, weighted + self.synergy))


# ---------------------------------------------------------------------------
# Per-layer scorers (each returns 0-100)
# ---------------------------------------------------------------------------
def score_short(short_pct: float, dtc: float, borrow: float, accum: float) -> tuple[float, dict]:
    """Wrap the existing squeeze.py scorer."""
    inp = SqueezeInputs(
        short_pct_float=short_pct or 0,
        days_to_cover=dtc or 0,
        borrow_rate=borrow or 0,
        iv_hv_ratio=1.0,
        rvol=1.0,
        float_shares=0,
        accumulation_score=accum or 0,
    )
    score, components = squeeze_score(inp)
    return float(score), components


def score_gex_proximity(spot: float, gex_df: pd.DataFrame) -> tuple[float, dict]:
    """0-100 based on distance to nearest heavy GEX wall + sign.

    Inside ±0.5% of a major wall = max attention. Negative wall under spot
    or positive wall over spot is the breakout-setup geometry.
    """
    if gex_df is None or gex_df.empty or spot <= 0:
        return 0.0, {}
    g = gex_df.copy()
    g["abs"] = g["total_gex"].abs()
    if g["abs"].sum() <= 0:
        return 0.0, {}
    # Top 5 heaviest strikes
    heavy = g.nlargest(5, "abs")
    nearest = heavy.assign(dist=lambda d: (d["strike"] - spot).abs() / spot).sort_values("dist").iloc[0]
    dist_pct = float(nearest["dist"] * 100)
    # Score: 100 at 0% distance, 0 at 3% distance, linear
    proximity = max(0.0, min(100.0, 100.0 * (1 - dist_pct / 3.0)))

    # Geometry bonus: if a heavy negative wall is just below spot → flush risk
    # OR a heavy positive wall is just above spot → magnet to break through
    geometry_bonus = 0.0
    near_neg_below = heavy[(heavy["total_gex"] < 0) & (heavy["strike"] < spot)]
    near_pos_above = heavy[(heavy["total_gex"] > 0) & (heavy["strike"] > spot)]
    if not near_neg_below.empty:
        d = float(((spot - near_neg_below["strike"].max()) / spot) * 100)
        if d < 1.5:
            geometry_bonus = max(geometry_bonus, 15.0 * (1 - d / 1.5))
    if not near_pos_above.empty:
        d = float(((near_pos_above["strike"].min() - spot) / spot) * 100)
        if d < 1.5:
            geometry_bonus = max(geometry_bonus, 15.0 * (1 - d / 1.5))

    # Violence boost: when negative GEX dominates near spot, the @t38p_flow
    # framework says price action will be amplified. Adds up to +10 points.
    violence = violence_flag(gex_df, spot)
    violence_bonus = 10.0 if violence.get("violent") else 0.0

    walls = classify_walls(gex_df)

    return min(100.0, proximity + geometry_bonus + violence_bonus), {
        "nearest_strike": float(nearest["strike"]),
        "nearest_gex": float(nearest["total_gex"]),
        "nearest_dist_pct": dist_pct,
        "geometry_bonus": geometry_bonus,
        "violence_bonus": violence_bonus,
        "violent_zone": bool(violence.get("violent")),
        "walls": walls,
    }


def score_compression(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple[float, dict]:
    """TTM squeeze + Bollinger bandwidth shrinkage."""
    if len(close) < 30:
        return 0.0, {}
    bb = ind.bollinger(close, 20)
    bw_now = float(bb["bb_bw"].iloc[-1] or 0)
    bw_50d_min = float(bb["bb_bw"].tail(50).min() or 0)
    bw_50d_max = float(bb["bb_bw"].tail(50).max() or 0)
    if bw_50d_max <= 0:
        return 0.0, {}
    # Tightness: where in the last 50 days' bandwidth range we sit
    rng = (bw_now - bw_50d_min) / (bw_50d_max - bw_50d_min) if bw_50d_max > bw_50d_min else 1.0
    tight = max(0.0, min(1.0, 1.0 - rng))    # 1.0 = at tightest

    # TTM squeeze on?
    ttm_now = bool(ind.squeeze_on(close, high, low).iloc[-1])
    score = 100.0 * tight
    if ttm_now:
        score = min(100.0, score + 25.0)
    return float(score), {"bb_bw": bw_now, "bb_bw_pctile": float(rng), "ttm_squeeze": ttm_now}


def score_rvol(volume: pd.Series) -> tuple[float, dict]:
    """5d/20d volume swelling."""
    if len(volume) < 25:
        return 0.0, {}
    avg5 = float(volume.tail(5).mean() or 0)
    avg20 = float(volume.tail(20).mean() or 0)
    if avg20 <= 0:
        return 0.0, {}
    ratio = avg5 / avg20
    # 1x = 0, 2x = 50, 4x = 100, capped
    score = max(0.0, min(100.0, (ratio - 1.0) * 33.0))
    return float(score), {"vol_5d": avg5, "vol_20d": avg20, "rvol5_20": ratio}


def score_catalyst(earnings_dates: list[date]) -> tuple[float, dict]:
    """100 if earnings tomorrow, decays linearly to 0 at 14 days."""
    if not earnings_dates:
        return 0.0, {}
    today = date.today()
    upcoming = [d for d in earnings_dates if d >= today]
    if not upcoming:
        return 0.0, {}
    days = (min(upcoming) - today).days
    if days > 14:
        return 0.0, {"earnings_in_days": days}
    return float(max(0.0, 100.0 * (1 - days / 14.0))), {"earnings_in_days": days,
                                                          "earnings_date": min(upcoming).isoformat()}


def score_trend(close: pd.Series) -> tuple[float, dict]:
    """Combined posture score: above 20EMA, slope positive, not extended."""
    if len(close) < 30:
        return 0.0, {}
    ema20 = ind.ema(close, 20)
    last = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1] or last)
    e20_prev = float(ema20.iloc[-6] or e20)
    above = last >= e20
    slope_pct = (e20 - e20_prev) / e20_prev if e20_prev else 0.0
    dist_pct = (last - e20) / e20 if e20 else 0.0

    # Sweet spot: just above 20EMA with positive slope, not >5% extended
    score = 0.0
    if above:
        score += 40.0
    if slope_pct > 0:
        score += min(30.0, slope_pct * 1500.0)  # 2% slope/5d → 30 pts
    if 0 <= dist_pct <= 0.05:
        score += 30.0
    elif dist_pct > 0.10:
        score -= 20.0  # too extended
    return float(max(0.0, min(100.0, score))), {
        "ema20": e20, "above_ema20": above,
        "slope_5d_pct": slope_pct * 100, "dist_ema20_pct": dist_pct * 100,
    }


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------
class SqueezeRadarScanner(BaseScanner):
    name = "radar"
    label = "Squeeze Radar (composite)"

    def __init__(self, provider, min_score: float = 50.0):
        super().__init__(provider)
        self.min_score = float(min_score)

    def scan(self, universe: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        for sym in universe:
            try:
                signal = self._score_one(sym)
            except Exception as e:
                log.debug("radar: %s failed: %s", sym, e)
                continue
            if signal and signal.score >= self.min_score:
                signals.append(signal)
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def _score_one(self, sym: str) -> Signal | None:
        # ---- price history ----
        df = self.provider.history(sym, period="6mo", interval="1d")
        if df is None or len(df) < 30:
            return None
        close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]
        spot = float(close.iloc[-1])

        # ---- fundamentals + short interest ----
        fund = self.provider.fundamentals(sym) or {}
        sd: dict = {}
        if self.squeeze_client:
            try:
                sd = self.squeeze_client.short_interest(sym) or {}
            except Exception:
                sd = {}
        if not sd:
            sd = self.provider.short_interest_detail(sym) or {}
        short_pct = sd.get("short_pct_float") or fund.get("short_pct_float", 0) or 0
        dtc = sd.get("days_to_cover") or fund.get("short_ratio", 0) or 0
        borrow = float(sd.get("borrow_rate") or sd.get("fee") or 0)

        # Accumulation proxy
        up_vol = float((vol * (close.diff() > 0)).tail(20).sum())
        tot_vol = float(vol.tail(20).sum() or 1)
        accum = (up_vol / tot_vol) * 2 - 1

        # ---- options chain → GEX ----
        gex_df = pd.DataFrame()
        try:
            chain = self.provider.options_chain(sym)
            calls, puts = chain.get("calls"), chain.get("puts")
            chain_spot = chain.get("spot") or spot
            chain_exp = chain.get("expiry")
            if calls is not None and not calls.empty and chain_exp:
                gex_df = chain_gex(calls, puts if puts is not None else pd.DataFrame(),
                                    chain_spot, chain_exp)
        except Exception as e:
            log.debug("radar: chain fetch failed for %s: %s", sym, e)

        # ---- earnings ----
        earnings: list[date] = []
        if self.news_client:
            try:
                earnings = self.news_client.earnings_calendar(sym, days_ahead=21) or []
            except Exception:
                pass
        if not earnings:
            try:
                earnings = self.provider.earnings_calendar(sym) or []
            except Exception:
                earnings = []

        # ---- score every layer ----
        breakdown = RadarBreakdown()
        details: dict = {"spot": spot, "short_pct_float": short_pct, "days_to_cover": dtc,
                          "borrow_rate": borrow}

        s_short, _ = score_short(short_pct, dtc, borrow, accum)
        breakdown.short = s_short
        details["accumulation"] = accum

        s_gex, gex_d = score_gex_proximity(spot, gex_df)
        breakdown.gex = s_gex
        details.update({f"gex_{k}": v for k, v in gex_d.items()})

        s_comp, comp_d = score_compression(close, high, low)
        breakdown.compression = s_comp
        details.update(comp_d)

        s_rv, rv_d = score_rvol(vol)
        breakdown.rvol = s_rv
        details.update(rv_d)

        s_cat, cat_d = score_catalyst(earnings)
        breakdown.catalyst = s_cat
        details.update(cat_d)

        s_trd, trd_d = score_trend(close)
        breakdown.trend = s_trd
        details.update(trd_d)

        # Synergy bonus: how many layers cleared the threshold?
        firing = sum(1 for x in (s_short, s_gex, s_comp, s_rv, s_cat, s_trd) if x >= SYNERGY_THRESHOLD)
        if firing >= 3:
            breakdown.synergy = SYNERGY_BONUS_PER_LAYER * (firing - 2)
        composite = breakdown.composite()

        # ---- tags + reason ----
        tags = ["radar"]
        if s_short >= 60: tags.append("squeeze-pressure")
        if s_gex   >= 60: tags.append("gex-magnet")
        if s_comp  >= 60: tags.append("coiled")
        if s_rv    >= 60: tags.append("vol-swell")
        if s_cat   >= 60: tags.append("catalyst")
        if s_trd   >= 60: tags.append("trend-up")
        if firing >= 3:    tags.append("multi-edge")

        reason_bits = [f"short={s_short:.0f}", f"gex={s_gex:.0f}", f"comp={s_comp:.0f}",
                        f"rvol={s_rv:.0f}", f"cat={s_cat:.0f}", f"trend={s_trd:.0f}"]
        reason = " · ".join(reason_bits) + (f" · synergy +{breakdown.synergy:.0f}" if breakdown.synergy else "")

        return Signal(
            scanner=self.name, symbol=sym.upper(),
            score=composite, reason=reason, tags=tags,
            metrics={**details,
                     "layer_scores": {
                         "short": s_short, "gex": s_gex, "compression": s_comp,
                         "rvol": s_rv, "catalyst": s_cat, "trend": s_trd,
                         "synergy": breakdown.synergy,
                     }},
        )
