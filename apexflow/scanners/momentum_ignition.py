"""Momentum ignition scanner — first 1-5 minutes of a big move.

Uses 1m bars when available (yfinance gives 1m for last 7 days).
"""
from __future__ import annotations
import logging

import config
from apexflow.analytics import indicators as ind
from apexflow.models import Signal
from .base import BaseScanner

log = logging.getLogger(__name__)


class MomentumIgnitionScanner(BaseScanner):
    name = "momentum"
    label = "Momentum Ignition Detector"

    def scan(self, universe: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        for sym in universe:
            try:
                df_1m = self.provider.history(sym, period="1d", interval="1m")
                df_d = self.provider.history(sym, period="3mo", interval="1d")
            except Exception:
                continue
            if df_1m is None or df_d is None or len(df_1m) < 5 or len(df_d) < 30:
                continue

            avg_dollar_vol_1m = (df_d["Volume"] * df_d["Close"]).tail(30).mean() / 390  # ~390 min/day
            recent_5 = df_1m.tail(5)
            cur_dollar_vol = (recent_5["Volume"] * recent_5["Close"]).sum() / 5
            rvol_1m = cur_dollar_vol / avg_dollar_vol_1m if avg_dollar_vol_1m else 0

            move_5m_pct = (recent_5["Close"].iloc[-1] / recent_5["Open"].iloc[0] - 1) * 100
            move_1m_pct = (recent_5["Close"].iloc[-1] / recent_5["Close"].iloc[-2] - 1) * 100 if len(recent_5) >= 2 else 0

            # Tape acceleration: each successive bar volume increasing
            vols = recent_5["Volume"].tolist()
            accel = sum(1 for i in range(1, len(vols)) if vols[i] > vols[i-1])

            fund = self.provider.fundamentals(sym)
            short_pct = fund.get("short_pct_float", 0) or 0
            squeeze_kicker = short_pct > 0.15

            # Prefer Finnhub for news (real catalysts); fall back to yfinance headlines.
            news = []
            if self.news_client:
                try:
                    news = self.news_client.news(sym, days=1, limit=3)
                except Exception as e:
                    log.debug("Finnhub news failed for %s: %s", sym, e)
            if not news:
                news = self.provider.news(sym, limit=3)
            has_news = bool(news)

            if rvol_1m < config.RVOL_IGNITION_THRESHOLD or abs(move_5m_pct) < config.MOMENTUM_ACCEL_PCT:
                continue

            score = min(100, 30 + min(rvol_1m, 10) * 4 + min(abs(move_5m_pct), 8) * 3
                        + accel * 2 + (15 if squeeze_kicker else 0) + (10 if has_news else 0))
            tags = ["ignition", "long" if move_5m_pct > 0 else "short"]
            if squeeze_kicker: tags.append("squeeze-candidate")
            if has_news: tags.append("catalyst")

            reason = (f"5m move {move_5m_pct:+.2f}% on {rvol_1m:.1f}x RVOL, "
                      f"accel={accel}/4 bars"
                      + (f", short {short_pct*100:.0f}%" if squeeze_kicker else "")
                      + (f", news: {news[0]['title'][:60]}" if has_news else ""))
            signals.append(Signal(
                scanner=self.name, symbol=sym, score=float(score),
                reason=reason, tags=tags,
                metrics={"rvol_1m": rvol_1m, "move_5m_pct": move_5m_pct,
                         "move_1m_pct": move_1m_pct, "tape_accel": accel,
                         "short_pct_float": short_pct, "has_news": has_news},
            ))
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
