"""yfinance-backed DataProvider. Free, delayed (~15 min). No flow/darkpool data."""
from __future__ import annotations
import logging
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from .base import DataProvider

log = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """Class-level state so all instances share the same rate-limit backoff
    and cache. Yahoo throttles per IP, so a per-instance cache is useless for
    coordinating the alerts engine and HTTP request handlers."""
    name = "yfinance"

    # Shared across all instances
    _shared_cache: dict[tuple, tuple[float, Any]] = {}
    _backoff_until: float = 0.0
    _consecutive_429: int = 0

    def __init__(self):
        self._cache = YFinanceProvider._shared_cache
        self._ttl = 60  # seconds

    @classmethod
    def is_rate_limited(cls) -> bool:
        return time.time() < cls._backoff_until

    @classmethod
    def backoff_remaining(cls) -> float:
        return max(0.0, cls._backoff_until - time.time())

    @classmethod
    def _trip_backoff(cls) -> None:
        cls._consecutive_429 += 1
        # Exponential backoff: 60s, 120s, 240s, capped at 600s (10 min)
        wait = min(60 * (2 ** (cls._consecutive_429 - 1)), 600)
        cls._backoff_until = max(cls._backoff_until, time.time() + wait)
        log.warning("yfinance rate-limited; backing off for %ds (#%d)",
                    wait, cls._consecutive_429)

    @classmethod
    def _clear_backoff(cls) -> None:
        if cls._consecutive_429:
            log.info("yfinance recovered after %d 429s", cls._consecutive_429)
        cls._consecutive_429 = 0

    def _cached(self, key: tuple, fetch, ttl: int | None = None):
        ttl = ttl or self._ttl
        now = time.time()
        if key in self._cache:
            ts, val = self._cache[key]
            if now - ts < ttl:
                return val
        # If we are currently rate-limited, return the stale cached value
        # (better than burning the next request budget on a guaranteed 429).
        if YFinanceProvider.is_rate_limited():
            if key in self._cache:
                return self._cache[key][1]
            return None
        try:
            val = fetch()
            YFinanceProvider._clear_backoff()
        except Exception as e:
            msg = str(e)
            if "Too Many Requests" in msg or "429" in msg or "Rate limit" in msg:
                YFinanceProvider._trip_backoff()
            else:
                log.warning("yfinance fetch failed for %s: %s", key, e)
            # Serve stale on failure when we have something cached
            if key in self._cache:
                return self._cache[key][1]
            return None
        self._cache[key] = (now, val)
        return val

    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        df = self._cached(
            ("hist", symbol, period, interval),
            lambda: yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False),
            ttl=60 if interval.endswith("m") else 3600,
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        return df

    def quote(self, symbol: str) -> dict:
        info = self._cached(("info", symbol), lambda: yf.Ticker(symbol).fast_info, ttl=30)
        if info is None:
            return {}
        try:
            return {
                "symbol": symbol,
                "price": float(getattr(info, "last_price", 0) or 0),
                "open": float(getattr(info, "open", 0) or 0),
                "previous_close": float(getattr(info, "previous_close", 0) or 0),
                "day_high": float(getattr(info, "day_high", 0) or 0),
                "day_low": float(getattr(info, "day_low", 0) or 0),
                "volume": int(getattr(info, "last_volume", 0) or 0),
                "market_cap": float(getattr(info, "market_cap", 0) or 0),
            }
        except Exception:
            return {"symbol": symbol}

    def expiries(self, symbol: str) -> list[str]:
        exps = self._cached(("exp", symbol), lambda: list(yf.Ticker(symbol).options), ttl=600)
        return exps or []

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        exps = self.expiries(symbol)
        if not exps:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": None, "spot": 0.0}
        expiry = expiry or exps[0]
        chain = self._cached(
            ("chain", symbol, expiry),
            lambda: yf.Ticker(symbol).option_chain(expiry),
            ttl=120,
        )
        if chain is None:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": expiry, "spot": 0.0}
        spot = self.quote(symbol).get("price", 0.0)
        return {
            "calls": chain.calls.copy(),
            "puts": chain.puts.copy(),
            "expiry": expiry,
            "spot": spot,
        }

    def fundamentals(self, symbol: str) -> dict:
        def fetch():
            t = yf.Ticker(symbol)
            try:
                info = t.info or {}
            except Exception:
                info = {}
            return {
                "symbol": symbol,
                "shares_float": info.get("floatShares") or 0,
                "shares_outstanding": info.get("sharesOutstanding") or 0,
                "shares_short": info.get("sharesShort") or 0,
                "short_ratio": info.get("shortRatio") or 0.0,                  # days-to-cover
                "short_pct_float": info.get("shortPercentOfFloat") or 0.0,
                "market_cap": info.get("marketCap") or 0,
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or "",
                "beta": info.get("beta") or 0.0,
                "avg_volume": info.get("averageVolume") or 0,
            }
        return self._cached(("fund", symbol), fetch, ttl=3600) or {}

    def earnings_calendar(self, symbol: str) -> list[date]:
        def fetch():
            t = yf.Ticker(symbol)
            try:
                cal = t.calendar
                if isinstance(cal, dict):
                    raw = cal.get("Earnings Date") or []
                    if isinstance(raw, list):
                        out = []
                        for x in raw:
                            if isinstance(x, datetime):
                                out.append(x.date())
                            elif isinstance(x, date):
                                out.append(x)
                        return out
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    if isinstance(val, (datetime, pd.Timestamp)):
                        return [val.date()]
            except Exception:
                pass
            return []
        return self._cached(("earn", symbol), fetch, ttl=3600) or []

    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        def fetch():
            try:
                items = yf.Ticker(symbol).news or []
            except Exception:
                return []
            out = []
            for n in items[:limit]:
                out.append({
                    "title": n.get("title", ""),
                    "publisher": n.get("publisher", ""),
                    "link": n.get("link", ""),
                    "published": n.get("providerPublishTime", 0),
                })
            return out
        return self._cached(("news", symbol, limit), fetch, ttl=300) or []
