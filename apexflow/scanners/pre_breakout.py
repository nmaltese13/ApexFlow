"""Pre-breakout scanner.

Detects:
  - BB squeeze (BB inside Keltner Channels — TTM-style)
  - Volume dry-up + recent accumulation candle
  - Multi-timeframe momentum alignment (daily/weekly EMA stacked)
  - High relative strength vs SPY building a base near 52w highs
  - Gap-and-go pre-market candidates (premarket gap + RVOL)
"""
from __future__ import annotations
import logging

import pandas as pd

import config
from apexflow.analytics import indicators as ind
from apexflow.models import Signal
from .base import BaseScanner

log = logging.getLogger(__name__)


class PreBreakoutScanner(BaseScanner):
    name = "pre-breakout"
    label = "Pre-Breakout Scanner"

    def scan(self, universe: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        try:
            spy = self.provider.history("SPY", period="6mo", interval="1d")["Close"]
        except Exception:
            spy = pd.Series(dtype=float)

        for sym in universe:
            try:
                df_d = self.provider.history(sym, period="6mo", interval="1d")
            except Exception:
                continue
            if df_d is None or len(df_d) < 60:
                continue

            close = df_d["Close"]; high = df_d["High"]; low = df_d["Low"]; vol = df_d["Volume"]
            bb = ind.bollinger(close, 20, 2.0)
            bw = bb["bb_bw"].iloc[-1]
            squeeze_now = bool(ind.squeeze_on(close, high, low).iloc[-1])
            dryup = ind.volume_dryup(vol, lookback=20, ratio=0.65)

            # Accumulation: last 3 closes upper-half of range
            recent = df_d.tail(3)
            acc = bool(((recent["Close"] - recent["Low"]) /
                        (recent["High"] - recent["Low"]).replace(0, 1) > 0.6).all())

            ema20 = ind.ema(close, 20).iloc[-1]
            ema50 = ind.ema(close, 50).iloc[-1]
            ema200 = ind.ema(close, 200).iloc[-1] if len(close) >= 200 else 0
            stacked = ema200 and (ema20 > ema50 > ema200)

            # 52w high proximity
            window = close.tail(252) if len(close) >= 252 else close
            hi52 = window.max()
            near_high = bool(close.iloc[-1] / hi52 > 0.92)

            rs = ind.relative_strength(close, spy, n=63) if len(spy) > 63 else 0.0

            # Gap-and-go premarket: use yfinance 5-min interval to look at last bar gap.
            gap = 0.0
            premkt_rvol = 0.0
            try:
                df_5m = self.provider.history(sym, period="5d", interval="5m")
                if len(df_5m) > 50:
                    last_close_d = df_d["Close"].iloc[-2]
                    last_5m_close = df_5m["Close"].iloc[-1]
                    gap = ind.gap_pct(last_5m_close, last_close_d)
                    avg5 = df_5m["Volume"].tail(50).mean() or 1
                    premkt_rvol = float(df_5m["Volume"].iloc[-1] / avg5)
            except Exception:
                pass

            score = 0.0
            tags: list[str] = []
            reasons: list[str] = []
            if squeeze_now or (bw is not None and bw < config.BB_SQUEEZE_BANDWIDTH):
                score += 30; tags.append("bb-squeeze")
                reasons.append(f"BB squeeze (bw={bw:.3f})")
            if dryup:
                score += 15; tags.append("dryup"); reasons.append("vol dry-up")
            if acc:
                score += 15; tags.append("accumulation"); reasons.append("3-bar accumulation")
            if stacked:
                score += 15; tags.append("stacked"); reasons.append("20>50>200 EMA")
            if near_high:
                score += 10; tags.append("base-at-high"); reasons.append("near 52w high")
            if rs > 1.2:
                score += 10; tags.append("high-rs"); reasons.append(f"RS={rs:.2f}")
            if abs(gap) > 2 and premkt_rvol > 2:
                score += 10; tags.append("gap-and-go"); reasons.append(f"gap {gap:+.1f}% rvol {premkt_rvol:.1f}x")

            if score < 40:
                continue
            signals.append(Signal(
                scanner=self.name, symbol=sym, score=min(score, 100.0),
                reason="; ".join(reasons),
                tags=tags,
                metrics={"bb_bw": float(bw or 0), "rs": rs, "gap_pct": gap,
                         "premkt_rvol": premkt_rvol, "near_52w_high": near_high},
            ))
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
