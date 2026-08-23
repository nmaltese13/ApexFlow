"""Tradier-backed DataProvider. Free with a Tradier brokerage account.

Sign up at https://developer.tradier.com/ — get a sandbox token immediately,
or a live token after opening a (free, no-minimum) brokerage account.

Tradier covers: real-time quotes, options chains+greeks, history, intraday bars.
Tradier does NOT cover: fundamentals, earnings calendar, news. Those fall
through to a yfinance fallback bundled inside this provider.
"""
from __future__ import annotations
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from .base import DataProvider
from .yfinance_provider import YFinanceProvider

log = logging.getLogger(__name__)


class TradierProvider(DataProvider):
    name = "tradier"
    BASE_LIVE = "https://api.tradier.com/v1"
    BASE_SANDBOX = "https://sandbox.tradier.com/v1"

    def __init__(self, token: str, sandbox: bool = True):
        self.token = token
        self.base = self.BASE_SANDBOX if sandbox else self.BASE_LIVE
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })
        self._cache: dict[tuple, tuple[float, Any]] = {}
        # Bundled fallback for things Tradier doesn't expose well.
        self._fallback = YFinanceProvider()

    def _cached(self, key, fetch, ttl=60):
        now = time.time()
        if key in self._cache and now - self._cache[key][0] < ttl:
            return self._cache[key][1]
        try:
            val = fetch()
        except Exception as e:
            log.warning("tradier fetch failed for %s: %s", key, e)
            return None
        self._cache[key] = (now, val)
        return val

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=10)
        r.raise_for_status()
        return r.json()

    # ---- Required interface ------------------------------------------------

    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """Tradier supports daily/weekly/monthly history and intraday timesales."""
        def fetch():
            end = datetime.now(timezone.utc).date()
            days = self._period_to_days(period)
            start = end - timedelta(days=days)
            if interval.endswith("m"):
                # intraday — use timesales (1m, 5m, 15m supported)
                minutes = int(interval[:-1])
                resp = self._get("/markets/timesales", {
                    "symbol": symbol,
                    "interval": f"{minutes}min",
                    "start": start.isoformat() + " 09:30",
                    "end": end.isoformat() + " 16:00",
                    "session_filter": "open",
                })
                series = (resp.get("series") or {}).get("data") or []
                if not series:
                    return pd.DataFrame()
                df = pd.DataFrame(series)
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
                df = df.rename(columns={"open": "Open", "high": "High",
                                         "low": "Low", "close": "Close", "volume": "Volume"})
                return df[["Open", "High", "Low", "Close", "Volume"]]
            else:
                tradier_int = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}.get(interval, "daily")
                resp = self._get("/markets/history", {
                    "symbol": symbol,
                    "interval": tradier_int,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                })
                day = (resp.get("history") or {}).get("day") or []
                if isinstance(day, dict):
                    day = [day]
                if not day:
                    return pd.DataFrame()
                df = pd.DataFrame(day)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                df = df.rename(columns={"open": "Open", "high": "High",
                                         "low": "Low", "close": "Close", "volume": "Volume"})
                return df[["Open", "High", "Low", "Close", "Volume"]]
        ttl = 60 if interval.endswith("m") else 3600
        df = self._cached(("hist", symbol, period, interval), fetch, ttl=ttl)
        return df if df is not None else pd.DataFrame()

    def quote(self, symbol: str) -> dict:
        def fetch():
            resp = self._get("/markets/quotes", {"symbols": symbol})
            q = (resp.get("quotes") or {}).get("quote") or {}
            if isinstance(q, list):
                q = q[0] if q else {}
            return {
                "symbol": symbol,
                "price": float(q.get("last") or 0),
                "open": float(q.get("open") or 0),
                "previous_close": float(q.get("prevclose") or 0),
                "day_high": float(q.get("high") or 0),
                "day_low": float(q.get("low") or 0),
                "volume": int(q.get("volume") or 0),
                "bid": float(q.get("bid") or 0),
                "ask": float(q.get("ask") or 0),
            }
        return self._cached(("quote", symbol), fetch, ttl=10) or {}

    def expiries(self, symbol: str) -> list[str]:
        def fetch():
            resp = self._get("/markets/options/expirations", {"symbol": symbol})
            dates = (resp.get("expirations") or {}).get("date") or []
            if isinstance(dates, str):
                return [dates]
            return list(dates)
        return self._cached(("exp", symbol), fetch, ttl=600) or []

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        exps = self.expiries(symbol)
        if not exps:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": None, "spot": 0.0}
        expiry = expiry or exps[0]

        def fetch():
            resp = self._get("/markets/options/chains", {
                "symbol": symbol, "expiration": expiry, "greeks": "true",
            })
            opts = (resp.get("options") or {}).get("option") or []
            if not opts:
                return None
            df = pd.DataFrame(opts)
            df["impliedVolatility"] = df["greeks"].apply(
                lambda g: float(g.get("smv_vol") or g.get("mid_iv") or 0) if isinstance(g, dict) else 0
            )
            df = df.rename(columns={
                "strike": "strike", "last": "lastPrice", "bid": "bid", "ask": "ask",
                "volume": "volume", "open_interest": "openInterest",
            })
            calls = df[df["option_type"] == "call"][
                ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
            ].copy()
            puts = df[df["option_type"] == "put"][
                ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
            ].copy()
            return {"calls": calls, "puts": puts}
        chain = self._cached(("chain", symbol, expiry), fetch, ttl=120)
        spot = self.quote(symbol).get("price", 0.0)
        if chain is None:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": expiry, "spot": spot}
        return {**chain, "expiry": expiry, "spot": spot}

    # ---- Tradier doesn't cover — fall through to yfinance ------------------
    def fundamentals(self, symbol: str) -> dict:
        return self._fallback.fundamentals(symbol)

    def earnings_calendar(self, symbol: str) -> list[date]:
        return self._fallback.earnings_calendar(symbol)

    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        return self._fallback.news(symbol, limit)

    # ---- Helpers -----------------------------------------------------------
    @staticmethod
    def _period_to_days(period: str) -> int:
        period = period.lower().strip()
        units = {"d": 1, "mo": 30, "y": 365}
        for suffix, mult in units.items():
            if period.endswith(suffix):
                try:
                    return int(period[: -len(suffix)]) * mult
                except ValueError:
                    pass
        return 180
