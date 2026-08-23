"""Earnings Market Scanner — scan EVERY company reporting in a window.

For each company with earnings on/before ``end_date``, this scanner:

  1. Pulls the options chain
  2. Runs :func:`predict_direction` to score bullish / bearish / neutral
     (composite signal in [-1, +1] → translated to a -10..+10 score for
     consistency with Bull Bear X Insights / SkylitAi-style scorecards)
  3. Runs the squeeze radar to add a 0-100 "ready to pop" score
  4. Computes the implied earnings move from the ATM straddle
  5. Returns a sortable, spreadsheet-friendly list of records.

Designed to power a dedicated /earnings page where you scan tomorrow's +
this week's prints in one view, sorted by direction or radar score.

Uses Finnhub for the calendar when available (most reliable), falls back
to per-symbol yfinance lookups against an optionable universe otherwise.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from apexflow.analytics.earnings_direction import predict_direction
from apexflow.analytics.gex import chain_gex
from apexflow.analytics.projection import expected_move as proj_em, horizon_years
from apexflow.providers import get_provider, get_news_client, get_squeeze_client
from apexflow.scanners.squeeze_radar import SqueezeRadarScanner

log = logging.getLogger(__name__)


@dataclass
class EarningsRow:
    """One row in the earnings scan output."""
    symbol:            str
    date:              str            # YYYY-MM-DD
    hour:              str            # bmo | amc | (empty if intraday)
    spot:              float = 0.0

    # Direction
    direction:         str = "neutral"   # bullish | bearish | neutral
    direction_score:   int = 0           # -10..+10
    direction_conf:    float = 0.0       # 0..1
    direction_reason:  str = ""

    # Squeeze readiness
    radar_score:       float = 0.0       # 0..100
    radar_tags:        list[str] = field(default_factory=list)

    # Move setup
    implied_move_pct:  float = 0.0       # straddle-implied % move
    historical_move_pct: float = 0.0     # avg abs prior-ER move
    iv_hist_ratio:     float = 0.0
    iv_label:          str = ""          # "cheap" | "rich" | "fair"
    atm_iv_pct:        float = 0.0

    # Positioning
    cp_oi_ratio:       float = 0.0       # call OI / put OI
    short_pct_float:   float = 0.0
    days_to_cover:     float = 0.0

    # EPS context
    eps_estimate:      float | None = None
    revenue_estimate: float | None = None

    errors:            dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _signed_to_score(composite: float) -> int:
    """Map -1..+1 composite to integer -10..+10 (Bull Bear X Insights style)."""
    return int(round(max(-1.0, min(1.0, composite)) * 10))


def _label_iv(ratio: float) -> str:
    if ratio <= 0:
        return ""
    if ratio < 0.85:
        return "cheap"
    if ratio > 1.30:
        return "rich"
    return "fair"


def _historical_er_move(history_df: pd.DataFrame, earnings_dates: list[date],
                         lookbacks: int = 8) -> float:
    """Average absolute close-to-close move on past earnings dates."""
    import numpy as np
    if history_df is None or history_df.empty or not earnings_dates:
        return 0.0
    moves = []
    for ed in earnings_dates[-lookbacks:]:
        try:
            idx = history_df.index.get_indexer([ed], method="nearest")[0]
            if idx <= 0:
                continue
            prev = float(history_df["Close"].iloc[idx - 1])
            cur = float(history_df["Close"].iloc[idx])
            moves.append(abs(cur / prev - 1))
        except Exception:
            continue
    return float(np.mean(moves)) if moves else 0.0


# ---------------------------------------------------------------------------
# Calendar fetcher — prefers Finnhub, falls back to per-symbol yfinance
# ---------------------------------------------------------------------------
def fetch_calendar(from_date: date, to_date: date,
                    universe: list[str] | None = None) -> list[dict]:
    """Get the earnings calendar between two dates.

    With a Finnhub key set we get the FULL US market calendar in one call.
    Without it we fall back to walking ``universe`` symbol-by-symbol via
    yfinance (much slower; pass a focused universe to keep it manageable).
    """
    news = get_news_client()
    if news is not None:
        try:
            return news.earnings_calendar_market(from_date, to_date)
        except Exception as e:
            log.warning("Finnhub market calendar failed: %s", e)

    # Fallback — per-symbol via the active provider's earnings_calendar
    if not universe:
        log.warning("No Finnhub key + no universe provided → empty calendar")
        return []
    prov = get_provider()
    out = []
    for sym in universe:
        try:
            dates = prov.earnings_calendar(sym) or []
        except Exception:
            continue
        for d in dates:
            if from_date <= d <= to_date:
                out.append({"symbol": sym, "date": d.isoformat(), "hour": ""})
        time.sleep(0.05)
    return out


# ---------------------------------------------------------------------------
# Per-symbol scoring
# ---------------------------------------------------------------------------
def score_symbol(symbol: str, calendar_entry: dict | None = None) -> EarningsRow | None:
    """Run the full scoring pipeline for one symbol. Returns None if we can't
    even price the underlying."""
    sym = symbol.upper()
    prov = get_provider()
    row = EarningsRow(
        symbol=sym,
        date=(calendar_entry or {}).get("date", ""),
        hour=(calendar_entry or {}).get("hour", ""),
    )
    if calendar_entry:
        row.eps_estimate = calendar_entry.get("eps_estimate")
        row.revenue_estimate = calendar_entry.get("revenue_estimate")

    # ---- price + history ----
    try:
        df = prov.history(sym, period="2y", interval="1d")
    except Exception as e:
        row.errors["history"] = str(e); df = None
    if df is None or df.empty:
        return None
    spot = float(df["Close"].iloc[-1])
    row.spot = spot

    # ---- chain + ATM straddle implied move ----
    chain = {}; calls = puts = None
    try:
        chain = prov.options_chain(sym)
        calls, puts = chain.get("calls"), chain.get("puts")
    except Exception as e:
        row.errors["chain"] = str(e)

    cp_oi = 0.0
    atm_iv = 0.0
    implied = 0.0
    if calls is not None and puts is not None and not calls.empty and not puts.empty:
        try:
            atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]].iloc[0]
            implied = float(((atm_c.get("lastPrice") or 0) +
                              (atm_p.get("lastPrice") or 0)) / spot) if spot else 0.0
            atm_iv = float(atm_c.get("impliedVolatility") or 0)
            row.implied_move_pct = implied * 100
            row.atm_iv_pct = atm_iv * 100
            call_oi = float(calls["openInterest"].fillna(0).sum())
            put_oi = float(puts["openInterest"].fillna(0).sum())
            cp_oi = (call_oi / put_oi) if put_oi else 0.0
            row.cp_oi_ratio = cp_oi
        except Exception as e:
            row.errors["atm"] = str(e)

    # ---- historical ER move + IV/hist ratio ----
    try:
        ed_dates = prov.earnings_calendar(sym) or []
    except Exception:
        ed_dates = []
    hist_move = _historical_er_move(df, ed_dates)
    row.historical_move_pct = hist_move * 100
    if hist_move > 0 and implied > 0:
        ratio = implied / hist_move
        row.iv_hist_ratio = ratio
        row.iv_label = _label_iv(ratio)

    # ---- short interest ----
    sd = {}
    try:
        sd = prov.short_interest_detail(sym) or {}
    except Exception:
        pass
    sclient = get_squeeze_client()
    if sclient and not sd:
        try:
            sd = sclient.short_interest(sym) or {}
        except Exception:
            pass
    fund = prov.fundamentals(sym) or {}
    row.short_pct_float = float(sd.get("short_pct_float") or fund.get("short_pct_float", 0) or 0)
    row.days_to_cover = float(sd.get("days_to_cover") or fund.get("short_ratio", 0) or 0)

    # 52w price position (for the directional predictor)
    pos52 = None
    if len(df) > 200:
        win = df.tail(252)
        lo, hi = float(win["Low"].min()), float(win["High"].max())
        pos52 = float((spot - lo) / (hi - lo)) if hi > lo else None

    # ---- direction ----
    try:
        view = predict_direction(
            calls=calls if calls is not None else pd.DataFrame(),
            puts=puts if puts is not None else pd.DataFrame(),
            spot=spot,
            short_pct_float=row.short_pct_float,
            days_to_cover=row.days_to_cover,
            history_df=df, earnings_dates=ed_dates,
            price_pos_52w=pos52,
            cp_oi_history=None,
        )
        row.direction = view.direction
        row.direction_conf = view.confidence
        row.direction_score = _signed_to_score(view.composite)
        row.direction_reason = view.reason
    except Exception as e:
        row.errors["direction"] = str(e)

    # ---- squeeze radar ----
    try:
        scanner = SqueezeRadarScanner(prov, min_score=0)
        sigs = scanner.scan([sym])
        if sigs:
            s = sigs[0]
            row.radar_score = float(s.score)
            row.radar_tags = list(s.tags)
    except Exception as e:
        row.errors["radar"] = str(e)

    return row


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def scan_earnings(from_date: date, to_date: date,
                   universe: list[str] | None = None,
                   inter_pause: float = 0.4,
                   max_symbols: int = 200) -> dict:
    """Scan every company with earnings between from_date and to_date.

    Returns:
        {
          generated_at: unix int,
          window: {from, to},
          universe_size: int,
          rows: [EarningsRow.to_dict(), ...],
          errors: int,
        }
    """
    cal = fetch_calendar(from_date, to_date, universe=universe)
    # Dedupe by symbol+date (Finnhub sometimes returns multiple entries per quarter)
    seen = set()
    unique = []
    for entry in cal:
        key = (entry.get("symbol", "").upper(), entry.get("date", ""))
        if key in seen or not key[0]:
            continue
        seen.add(key)
        unique.append(entry)
    if max_symbols and len(unique) > max_symbols:
        unique = unique[:max_symbols]

    rows: list[dict] = []
    errors = 0
    for entry in unique:
        sym = entry.get("symbol", "").upper()
        if not sym:
            continue
        try:
            row = score_symbol(sym, entry)
            if row:
                rows.append(row.to_dict())
            else:
                errors += 1
        except Exception as e:
            log.debug("earnings scan: %s failed: %s", sym, e)
            errors += 1
        if inter_pause:
            time.sleep(inter_pause)

    rows.sort(key=lambda r: (-r.get("radar_score", 0), -abs(r.get("direction_score", 0))))
    return {
        "generated_at": int(datetime.now(timezone.utc).timestamp()),
        "window": {"from": from_date.isoformat(), "to": to_date.isoformat()},
        "universe_size": len(unique),
        "rows": rows,
        "errors": errors,
    }
