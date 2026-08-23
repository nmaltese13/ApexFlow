"""Earnings alpha scanner: cheap/expensive options into earnings + positioning."""
from __future__ import annotations
import logging
import math
from datetime import date, timedelta

import numpy as np

import config
from apexflow.analytics import indicators as ind
from apexflow.models import Signal
from .base import BaseScanner

log = logging.getLogger(__name__)


def implied_move_from_atm(spot: float, atm_call: float, atm_put: float) -> float:
    """Straddle-implied move as % of spot."""
    if not spot:
        return 0.0
    return (atm_call + atm_put) / spot


def historical_earnings_move(history_df, earnings_dates: list[date], lookbacks: int = 8) -> float:
    """Average abs(close-to-close) move on past earnings days."""
    if history_df is None or history_df.empty or not earnings_dates:
        return 0.0
    moves = []
    for ed in earnings_dates[-lookbacks:]:
        try:
            idx = history_df.index.get_indexer([ed], method="nearest")[0]
            if idx <= 0:
                continue
            prev = history_df["Close"].iloc[idx - 1]
            cur = history_df["Close"].iloc[idx]
            moves.append(abs(cur / prev - 1))
        except Exception:
            continue
    return float(np.mean(moves)) if moves else 0.0


class EarningsAlphaScanner(BaseScanner):
    name = "earnings"
    label = "Earnings Alpha Scanner"

    def __init__(self, provider, days_ahead: int = 14):
        super().__init__(provider)
        self.days_ahead = days_ahead

    def scan(self, universe: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        today = date.today()
        horizon = today + timedelta(days=self.days_ahead)
        for sym in universe:
            # Prefer Finnhub earnings calendar (more reliable); fall back to yfinance.
            earnings = []
            if self.news_client:
                try:
                    earnings = self.news_client.earnings_calendar(sym, days_ahead=self.days_ahead)
                except Exception as e:
                    log.debug("Finnhub earnings failed for %s: %s", sym, e)
            if not earnings:
                earnings = self.provider.earnings_calendar(sym)
            upcoming = [d for d in earnings if today <= d <= horizon]
            if not upcoming:
                continue
            ed = upcoming[0]

            try:
                chain = self.provider.options_chain(sym)
            except Exception:
                continue
            calls, puts, spot = chain.get("calls"), chain.get("puts"), chain.get("spot") or 0
            if calls is None or puts is None or calls.empty or puts.empty or spot <= 0:
                continue

            atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]].iloc[0]
            implied = implied_move_from_atm(spot, atm_c["lastPrice"] or 0, atm_p["lastPrice"] or 0)

            df_d = self.provider.history(sym, period="2y", interval="1d")
            historical = historical_earnings_move(df_d, list(earnings))

            ratio = implied / historical if historical else 0
            cheap = 0 < ratio < config.EARNINGS_IV_CHEAP_RATIO
            expensive = ratio > 1.3

            # Pre-earnings positioning: call/put OI ratio
            call_oi = float(calls["openInterest"].fillna(0).sum())
            put_oi = float(puts["openInterest"].fillna(0).sum())
            cp_oi = call_oi / put_oi if put_oi else float("inf")

            # Sector momentum proxy via SPY 20d return
            try:
                spy = self.provider.history("SPY", period="2mo", interval="1d")["Close"]
                sector_mom = float(spy.iloc[-1] / spy.iloc[-21] - 1) if len(spy) > 21 else 0
            except Exception:
                sector_mom = 0

            score = 0.0
            tags = ["earnings"]
            reasons = [f"reports {ed.isoformat()}", f"implied {implied*100:.1f}%", f"avg historical {historical*100:.1f}%"]
            if cheap:
                score += 40; tags.append("cheap-iv"); reasons.append(f"IV/hist={ratio:.2f} CHEAP")
            elif expensive:
                score += 25; tags.append("expensive-iv"); reasons.append(f"IV/hist={ratio:.2f} RICH")
            else:
                score += 15
            if cp_oi > 1.5:
                score += 15; tags.append("bullish-positioning"); reasons.append(f"C/P OI={cp_oi:.2f}")
            elif cp_oi < 0.67:
                score += 15; tags.append("bearish-positioning"); reasons.append(f"C/P OI={cp_oi:.2f}")
            if abs(sector_mom) > 0.05:
                score += 10
                tags.append("sector-momentum")
                reasons.append(f"SPY 20d {sector_mom*100:+.1f}%")
            days_until = (ed - today).days
            if days_until <= 3:
                score += 10; tags.append("imminent")

            signals.append(Signal(
                scanner=self.name, symbol=sym, score=min(score, 100.0),
                reason="; ".join(reasons), tags=tags,
                metrics={"implied_move": implied, "historical_move": historical,
                         "iv_hist_ratio": ratio, "cp_oi_ratio": cp_oi,
                         "earnings_date": ed.isoformat(), "days_until": days_until},
            ))
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
