"""MarketData.app DataProvider.

Cheap, options-friendly REST API. Free tier 100 req/day, paid plans start
~$15-30/mo. Has live IV + greeks on options chains, which is the main reason
to keep it as a fallback when Schwab/Polygon aren't configured.

API docs: https://www.marketdata.app/docs/api
Auth: Bearer token in Authorization header

Endpoints:
  GET /v1/stocks/quotes/{symbol}/                   real-time quote
  GET /v1/stocks/candles/{resolution}/{symbol}/     OHLCV bars (D / 1 / 5 / 15 / 30 / 60)
  GET /v1/options/expirations/{symbol}/             list of expiration dates
  GET /v1/options/chain/{symbol}/                   full chain w/ IV+greeks
  GET /v1/options/quotes/{optionSymbol}/            individual contract quote
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from .base import DataProvider

log = logging.getLogger(__name__)


class MarketDataProvider(DataProvider):
    name = "marketdata"
    BASE = "https://api.marketdata.app/v1"

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept":         "application/json",
        })
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._ttl = 30

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        key = (path, tuple(sorted((params or {}).items())))
        now = time.time()
        if key in self._cache and now - self._cache[key][0] < self._ttl:
            return self._cache[key][1]
        try:
            r = self.session.get(f"{self.BASE}{path}", params=params or {}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("MarketData GET %s failed: %s", path, e)
            return None
        self._cache[key] = (now, data)
        return data

    # ------- DataProvider interface -----------------------------------------
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        sym = symbol.upper()
        # Map ApexFlow interval → MarketData resolution
        res_map = {
            "1d": "D", "1day": "D", "daily": "D",
            "1m": "1", "1min": "1",
            "5m": "5", "5min": "5",
            "15m": "15", "30m": "30",
            "1h": "60", "60m": "60",
        }
        resolution = res_map.get((interval or "").lower(), "D")

        days_back = {
            "1d": 2, "5d": 7, "10d": 14, "20d": 28,
            "1mo": 32, "2mo": 62, "3mo": 95,
            "6mo": 190, "1y": 366, "2y": 732, "5y": 1830,
        }.get((period or "").lower(), 190)
        end = date.today(); start = end - timedelta(days=days_back)

        data = self._get(f"/stocks/candles/{resolution}/{sym}/", params={
            "from": start.isoformat(),
            "to":   end.isoformat(),
        })
        if not data or data.get("s") not in ("ok", "no_data") or not data.get("t"):
            return pd.DataFrame()
        ts = data.get("t", [])
        df = pd.DataFrame({
            "Open":  data.get("o", []),
            "High":  data.get("h", []),
            "Low":   data.get("l", []),
            "Close": data.get("c", []),
            "Volume": data.get("v", []),
        }, index=[datetime.fromtimestamp(t, tz=timezone.utc) for t in ts])
        df.index.name = "Date"
        return df

    def quote(self, symbol: str) -> dict:
        sym = symbol.upper()
        data = self._get(f"/stocks/quotes/{sym}/")
        if not data or data.get("s") != "ok":
            return {}
        # MarketData returns arrays of length 1 in batch shape
        def _g(k, idx=0, default=None):
            v = data.get(k)
            if isinstance(v, list) and len(v) > idx:
                return v[idx]
            return default
        return {
            "symbol": sym,
            "price":          _g("last"),
            "previous_close": None,   # not in this endpoint; caller can pull from history
            "open":           None,
            "high":           _g("high"),
            "low":            _g("low"),
            "volume":         _g("volume"),
            "bid":            _g("bid"),
            "ask":            _g("ask"),
        }

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        sym = symbol.upper()
        params: dict = {}
        if expiry:
            params["expiration"] = expiry
        data = self._get(f"/options/chain/{sym}/", params=params)
        if not data or data.get("s") != "ok":
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "spot": 0, "expiry": expiry}

        underlying = (data.get("underlyingPrice") or [0])
        spot = float(underlying[0] if isinstance(underlying, list) else underlying or 0)

        def _arr(name):
            v = data.get(name) or []
            return v if isinstance(v, list) else []

        sides = _arr("side")
        rows = []
        n = len(sides)
        for i in range(n):
            rows.append({
                "side":   sides[i],
                "strike": _arr("strike")[i] if i < len(_arr("strike")) else 0,
                "expiry": _arr("expiration")[i] if i < len(_arr("expiration")) else "",
                "lastPrice":   _arr("last")[i] if i < len(_arr("last")) else 0,
                "bid":         _arr("bid")[i] if i < len(_arr("bid")) else 0,
                "ask":         _arr("ask")[i] if i < len(_arr("ask")) else 0,
                "volume":      _arr("volume")[i] if i < len(_arr("volume")) else 0,
                "openInterest": _arr("openInterest")[i] if i < len(_arr("openInterest")) else 0,
                "impliedVolatility": _arr("iv")[i] if i < len(_arr("iv")) else 0,
                "delta":       _arr("delta")[i] if i < len(_arr("delta")) else 0,
                "gamma":       _arr("gamma")[i] if i < len(_arr("gamma")) else 0,
                "theta":       _arr("theta")[i] if i < len(_arr("theta")) else 0,
                "vega":        _arr("vega")[i] if i < len(_arr("vega")) else 0,
            })
        if not rows:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "spot": spot, "expiry": expiry}

        df = pd.DataFrame(rows)
        # Normalise expiry strings — MarketData returns YYYY-MM-DD
        if expiry is None:
            soonest = sorted(df["expiry"].dropna().unique().tolist())[0]
            expiry = soonest
        df = df[df["expiry"] == expiry]
        calls = df[df["side"].str.lower() == "call"].drop(columns=["side", "expiry"])
        puts  = df[df["side"].str.lower() == "put"].drop(columns=["side", "expiry"])
        return {"calls": calls.reset_index(drop=True), "puts": puts.reset_index(drop=True),
                "spot": spot, "expiry": expiry}

    def expiries(self, symbol: str) -> list[str]:
        sym = symbol.upper()
        data = self._get(f"/options/expirations/{sym}/")
        if not data or data.get("s") != "ok":
            return []
        exps = data.get("expirations") or []
        return sorted(set(exps))

    def fundamentals(self, symbol: str) -> dict:
        # MarketData.app doesn't expose a fundamentals endpoint —
        # leave empty so callers fall back to other providers.
        return {}

    def earnings_calendar(self, symbol: str):
        return []
