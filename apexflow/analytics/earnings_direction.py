"""Earnings direction predictor.

EarningsAlphaScanner already estimates the *magnitude* of the move
(implied vs historical realised). This module adds *which way* it's likely to
go using five orthogonal directional edges:

  1. 25-delta risk reversal       (call IV − put IV around 0.25Δ; positive = bullish)
  2. Call/Put OI imbalance trend  (last 10 sessions, slope of CP OI ratio)
  3. IV skew curvature            (smile vs smirk; smirk to puts = downside fear)
  4. Short interest direction     (heavy short into earnings = squeeze fuel up
                                    OR weak hands long that get flushed if miss)
  5. Post-ER drift history        (this name's typical sign after past prints)

Each returns a directional score in [-1, +1]. The composite is the
weighted average; the sign is the predicted direction, the magnitude is the
confidence (0-1). Conviction breaks below 0.20 should be treated as neutral.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


WEIGHTS = {
    "risk_reversal":    0.25,
    "oi_trend":         0.20,
    "iv_skew":          0.15,
    "short_setup":      0.20,
    "drift_history":    0.20,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nearest_delta(chain: pd.DataFrame, target_delta: float, right: str,
                    spot: float) -> dict:
    """Pick the row whose Black-Scholes delta is closest to `target_delta`.

    Free chains rarely include delta directly — we approximate it with the
    distance from spot in IV-units. Cheap proxy: for calls, +0.25Δ ≈ ~1σ OTM
    where σ scales with IV and time. We return the row plus its IV.
    """
    if chain is None or chain.empty or spot <= 0:
        return {}
    df = chain.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", 0), errors="coerce").fillna(0)
    df = df[df["strike"] > 0]
    if df.empty:
        return {}
    # Use the closest expiry → assume t ≈ 7 days unless caller filters
    # The exact target moneyness depends on IV; use the chain median IV as proxy.
    iv_med = float(df["impliedVolatility"].median() or 0.30)
    sigma = max(0.05, iv_med) * (7 / 365) ** 0.5
    if right.upper() == "C":
        target_strike = spot * (1 + sigma)        # ≈ +1σ OTM
    else:
        target_strike = spot * (1 - sigma)        # ≈ -1σ OTM
    df["dist"] = (df["strike"] - target_strike).abs()
    row = df.nsmallest(1, "dist").iloc[0]
    return {"strike": float(row["strike"]),
             "iv": float(row["impliedVolatility"]),
             "moneyness_pct": float((row["strike"] / spot - 1) * 100)}


def _safe(v) -> float:
    try:
        f = float(v)
        return 0.0 if f != f else f
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Layer scorers (each returns (signal in [-1, +1], detail dict))
# ---------------------------------------------------------------------------
def risk_reversal(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> tuple[float, dict]:
    """25-delta call IV minus 25-delta put IV. >0 = puts are cheaper → bullish."""
    cot = _nearest_delta(calls, 0.25, "C", spot)
    pot = _nearest_delta(puts, -0.25, "P", spot)
    if not cot or not pot or cot["iv"] <= 0 or pot["iv"] <= 0:
        return 0.0, {}
    rr = cot["iv"] - pot["iv"]
    # Normalise: ±10 vol pts → ±1.0
    sig = max(-1.0, min(1.0, rr / 0.10))
    return float(sig), {
        "call_25d_iv": cot["iv"], "put_25d_iv": pot["iv"],
        "risk_reversal_pts": rr * 100,
    }


def oi_trend(calls_history: List[float], puts_history: List[float]) -> tuple[float, dict]:
    """Slope of (call_OI / put_OI) over the last N daily snapshots.

    Free providers don't expose historical OI series; the caller passes
    whatever they have. Rising C/P OI ratio = bullish positioning.
    """
    if not calls_history or not puts_history or len(calls_history) != len(puts_history):
        return 0.0, {}
    arr = np.array([c / p if p else np.nan for c, p in zip(calls_history, puts_history)],
                    dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 4:
        return 0.0, {}
    # Linear slope normalised by mean to be ratio-invariant
    x = np.arange(arr.size, dtype=float)
    slope, _ = np.polyfit(x, arr, 1)
    norm = slope / (arr.mean() or 1.0)
    sig = max(-1.0, min(1.0, norm * 8.0))
    return float(sig), {"cp_oi_slope": slope, "cp_oi_mean": float(arr.mean())}


def iv_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> tuple[float, dict]:
    """Compare downside (-15% strike) put IV to upside (+15%) call IV.

    Heavy downside skew (puts much pricier) = market braced for crash → if it
    holds, slight bullish (squeeze). If it amplifies, bearish.
    Using the static reading: if downside skew is *extreme*, slight bearish
    bias because it usually means real fear with cause; mild downside skew is
    neutral; flat skew is bullish.
    """
    if calls is None or calls.empty or puts is None or puts.empty or spot <= 0:
        return 0.0, {}
    pdf = puts.copy(); cdf = calls.copy()
    for d in (pdf, cdf):
        d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
        d["impliedVolatility"] = pd.to_numeric(d.get("impliedVolatility", 0), errors="coerce")
    pdf["dist"] = (pdf["strike"] - spot * 0.85).abs()
    cdf["dist"] = (cdf["strike"] - spot * 1.15).abs()
    p_iv = float(pdf.nsmallest(1, "dist").iloc[0].get("impliedVolatility") or 0)
    c_iv = float(cdf.nsmallest(1, "dist").iloc[0].get("impliedVolatility") or 0)
    if p_iv <= 0 or c_iv <= 0:
        return 0.0, {}
    skew = p_iv - c_iv  # positive = downside more expensive (typical)
    # Normalise: 0..0.05 vol pts = neutral; >0.10 = extreme = mild bearish
    if skew < 0.02:
        sig = +0.4    # near-flat skew is unusual = bullish posture
    elif skew < 0.06:
        sig = +0.1
    elif skew < 0.10:
        sig = -0.1
    else:
        sig = -0.5    # extreme downside skew = real fear
    return float(sig), {"down_iv": p_iv, "up_iv": c_iv, "skew_pts": skew * 100}


def short_setup(short_pct_float: float, days_to_cover: float,
                price_pos_52w: float | None = None) -> tuple[float, dict]:
    """Direction of pressure from short positioning.

    Heavy short + low price (near 52w low) into earnings = either capitulation
    (bearish if miss) or the squeeze trigger (bullish if any beat). Slightly
    bullish on net because squeezes pay more than fades.
    """
    if not short_pct_float:
        return 0.0, {}
    base = 0.0
    if short_pct_float > 0.20:
        base += 0.4
    if days_to_cover and days_to_cover > 5:
        base += 0.2
    # Bonus if price is near 52w low → coiled spring
    if price_pos_52w is not None and price_pos_52w < 0.30:
        base += 0.2
    return float(min(1.0, base)), {
        "short_pct_float": short_pct_float, "days_to_cover": days_to_cover,
        "price_pct_of_52w_range": price_pos_52w,
    }


def drift_history(history_df: pd.DataFrame, earnings_dates: list[date],
                   lookback: int = 8) -> tuple[float, dict]:
    """Average signed close-to-close move on past earnings days for THIS name."""
    if history_df is None or history_df.empty or not earnings_dates:
        return 0.0, {}
    moves = []
    for ed in earnings_dates[-lookback:]:
        try:
            idx = history_df.index.get_indexer([ed], method="nearest")[0]
            if idx <= 0:
                continue
            prev = history_df["Close"].iloc[idx - 1]
            cur = history_df["Close"].iloc[idx]
            moves.append((cur / prev - 1))
        except Exception:
            continue
    if not moves:
        return 0.0, {}
    avg = float(np.mean(moves))
    # Average move of ±5% → signal of ±1.0
    sig = max(-1.0, min(1.0, avg / 0.05))
    return float(sig), {"avg_post_er_move_pct": avg * 100, "n_prior_quarters": len(moves)}


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
@dataclass
class DirectionalView:
    direction: str = "neutral"   # "bullish" | "bearish" | "neutral"
    confidence: float = 0.0      # 0..1
    composite: float = 0.0       # raw signed signal -1..+1
    layers: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "direction": self.direction, "confidence": self.confidence,
            "composite": self.composite, "layers": self.layers,
            "reason": self.reason,
        }


def predict_direction(*,
                      calls: pd.DataFrame, puts: pd.DataFrame, spot: float,
                      short_pct_float: float = 0.0, days_to_cover: float = 0.0,
                      history_df: pd.DataFrame | None = None,
                      earnings_dates: list[date] | None = None,
                      price_pos_52w: float | None = None,
                      cp_oi_history: tuple[list, list] | None = None,
                      ) -> DirectionalView:
    """Return a DirectionalView combining all five layers."""
    layers: dict = {}
    contributions: dict = {}

    rr_sig, rr_d = risk_reversal(calls, puts, spot)
    layers["risk_reversal"] = {"signal": rr_sig, **rr_d}
    contributions["risk_reversal"] = rr_sig * WEIGHTS["risk_reversal"]

    if cp_oi_history:
        oi_sig, oi_d = oi_trend(*cp_oi_history)
    else:
        oi_sig, oi_d = 0.0, {}
    layers["oi_trend"] = {"signal": oi_sig, **oi_d}
    contributions["oi_trend"] = oi_sig * WEIGHTS["oi_trend"]

    skew_sig, skew_d = iv_skew(calls, puts, spot)
    layers["iv_skew"] = {"signal": skew_sig, **skew_d}
    contributions["iv_skew"] = skew_sig * WEIGHTS["iv_skew"]

    short_sig, short_d = short_setup(short_pct_float, days_to_cover, price_pos_52w)
    layers["short_setup"] = {"signal": short_sig, **short_d}
    contributions["short_setup"] = short_sig * WEIGHTS["short_setup"]

    drift_sig, drift_d = drift_history(history_df, earnings_dates or [])
    layers["drift_history"] = {"signal": drift_sig, **drift_d}
    contributions["drift_history"] = drift_sig * WEIGHTS["drift_history"]

    composite = sum(contributions.values())
    confidence = abs(composite)
    if confidence < 0.20:
        direction = "neutral"
    elif composite > 0:
        direction = "bullish"
    else:
        direction = "bearish"

    # This function used to emit a BUY/SELL debit-spread leg pair here.
    # It no longer does. Naming strikes with a side attached turns a
    # positioning read into an instruction, and the five layers below do
    # not support that: they measure what the options market is currently
    # positioned for, which is a statement about the present, not a
    # forecast with an entry attached. The layer detail and `reason` string
    # carry everything the leg list encoded, without the framing.

    # Human reason
    bits = []
    for k, v in contributions.items():
        if abs(v) > 0.02:
            bits.append(f"{k}={v:+.2f}")
    reason = " · ".join(bits) if bits else "no clear directional edge"

    return DirectionalView(
        direction=direction, confidence=float(confidence),
        composite=float(composite), layers=layers, reason=reason,
    )
