"""Polygon.io DataProvider.

Implements the standard ApexFlow DataProvider interface against Polygon's REST
API. Auto-activates from `apexflow.providers.get_provider()` when POLYGON_KEY
is set in keys.py.

Subscription tiers tested:
  - **Stocks Starter** ($29/mo): real-time stock quotes + 5y of intraday history.
    quote() / history() work fully. options_chain() returns OI/volume only
    (no live IV or greeks).
  - **Options Starter** ($29/mo, recommended): real-time options chains with
    live IV / OI / greeks. options_chain() returns the full Heatseeker /
    Atlas-grade payload.
  - **Higher tiers** add WebSocket streaming, options-trades flow,
    sweep classification (we don't use these via REST yet).

Free Polygon tier is rate-limited to 5 calls/min — too tight for the briefing
engine. Use it only for one-off CLI commands.

Endpoints used:
  - /v2/aggs/ticker/{symbol}/range/{mult}/{span}/{from}/{to}    history
  - /v3/snapshot/options/{underlying}                           full options snapshot
  - /v3/reference/options/contracts                             list contracts (expiries)
  - /v2/last/trade/{symbol} + /v2/last/nbbo/{symbol}            quote
  - /v3/reference/tickers/{symbol}                              fundamentals
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


# Map ApexFlow period strings to (start_date, multiplier, timespan)
def _range_for(period: str, interval: str) -> tuple[date, date, int, str]:
    today = date.today()
    period = (period or "").lower()
    interval = (interval or "").lower()

    if interval in ("1d", "1day", "daily", "day"):
        mult, span = 1, "day"
    elif interval in ("1m", "1min"):
        mult, span = 1, "minute"
    elif interval in ("5m", "5min"):
        mult, span = 5, "minute"
    elif interval == "15m":
        mult, span = 15, "minute"
    elif interval == "30m":
        mult, span = 30, "minute"
    elif interval in ("1h", "60m", "1hour"):
        mult, span = 1, "hour"
    else:
        mult, span = 1, "day"

    days_back = {
        "1d": 2, "5d": 7, "10d": 14, "20d": 28,
        "1mo": 32, "2mo": 62, "3mo": 95, "6mo": 190, "1y": 366,
        "2y": 732, "5y": 1830, "ytd": (today - date(today.year, 1, 1)).days + 1,
    }.get(period, 190)
    start = today - timedelta(days=days_back)
    return start, today, mult, span


class PolygonProvider(DataProvider):
    name = "polygon"
    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.key = api_key
        self.session = requests.Session()
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._ttl = 30

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        key = (path, tuple(sorted((params or {}).items())))
        now = time.time()
        if key in self._cache and now - self._cache[key][0] < self._ttl:
            return self._cache[key][1]
        try:
            r = self.session.get(f"{self.BASE}{path}",
                                  params={**(params or {}), "apiKey": self.key},
                                  timeout=15)
            if r.status_code == 429:
                log.warning("Polygon rate-limit hit on %s — sleeping 12s", path)
                time.sleep(12)
                r = self.session.get(f"{self.BASE}{path}",
                                      params={**(params or {}), "apiKey": self.key},
                                      timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("Polygon GET %s failed: %s", path, e)
            return None
        self._cache[key] = (now, data)
        return data

    # ------- DataProvider interface -----------------------------------------
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        sym = symbol.upper()
        start, end, mult, span = _range_for(period, interval)
        path = f"/v2/aggs/ticker/{sym}/range/{mult}/{span}/{start.isoformat()}/{end.isoformat()}"
        data = self._get(path, params={"adjusted": "true", "sort": "asc", "limit": 50000})
        if not data or not data.get("results"):
            return pd.DataFrame()
        rows = data["results"]
        df = pd.DataFrame([{
            "Open": r.get("o"), "High": r.get("h"), "Low": r.get("l"),
            "Close": r.get("c"), "Volume": r.get("v"),
        } for r in rows], index=[datetime.fromtimestamp(r["t"]/1000, tz=timezone.utc) for r in rows])
        df.index.name = "Date"
        return df

    def quote(self, symbol: str) -> dict:
        sym = symbol.upper()
        # Snapshot endpoint gives us last + day open/high/low/close + change in one call
        snap = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{sym}")
        if not snap or not snap.get("ticker"):
            return {}
        t = snap["ticker"]
        last = t.get("lastTrade") or {}
        day = t.get("day") or {}
        prev = t.get("prevDay") or {}
        return {
            "symbol": sym,
            "price":          last.get("p") or day.get("c"),
            "previous_close": prev.get("c"),
            "open":           day.get("o"),
            "high":           day.get("h"),
            "low":            day.get("l"),
            "volume":         day.get("v"),
            "bid":            (t.get("lastQuote") or {}).get("p"),
            "ask":            (t.get("lastQuote") or {}).get("P"),
        }

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        """Use the v3 options snapshot endpoint to pull live IV + OI + greeks."""
        sym = symbol.upper()
        params = {"limit": 250}
        if expiry:
            params["expiration_date"] = expiry
        snap = self._get(f"/v3/snapshot/options/{sym}", params=params)
        if not snap or not snap.get("results"):
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "spot": 0, "expiry": None}

        spot = float((snap.get("results") or [{}])[0].get("underlying_asset", {}).get("price")
                      or (snap.get("results") or [{}])[0].get("underlying_asset", {}).get("last_quote", {}).get("midpoint")
                      or 0)
        calls_rows, puts_rows = [], []
        for c in snap.get("results", []):
            details = c.get("details") or {}
            day     = c.get("day") or {}
            greeks  = c.get("greeks") or {}
            quote   = c.get("last_quote") or {}
            row = {
                "strike":      details.get("strike_price"),
                "expiry":      details.get("expiration_date"),
                "lastPrice":   day.get("close") or (c.get("last_trade") or {}).get("price") or 0,
                "bid":         quote.get("bid") or 0,
                "ask":         quote.get("ask") or 0,
                "volume":      day.get("volume") or 0,
                "openInterest": c.get("open_interest") or 0,
                "impliedVolatility": c.get("implied_volatility") or 0,
                "delta":       greeks.get("delta") or 0,
                "gamma":       greeks.get("gamma") or 0,
                "theta":       greeks.get("theta") or 0,
                "vega":        greeks.get("vega")  or 0,
            }
            if details.get("contract_type") == "call":
                calls_rows.append(row)
            else:
                puts_rows.append(row)

        calls = pd.DataFrame(calls_rows)
        puts  = pd.DataFrame(puts_rows)

        # Filter to requested expiry, or pick the soonest if none requested
        if expiry is None:
            all_exp = sorted(set(calls["expiry"].dropna().tolist() + puts["expiry"].dropna().tolist()))
            expiry = all_exp[0] if all_exp else None
        if expiry:
            if not calls.empty: calls = calls[calls["expiry"] == expiry].drop(columns="expiry")
            if not puts.empty:  puts  = puts[puts["expiry"] == expiry].drop(columns="expiry")

        return {"calls": calls, "puts": puts, "spot": spot, "expiry": expiry}

    def expiries(self, symbol: str) -> list[str]:
        sym = symbol.upper()
        out: set[str] = set()
        next_url = "/v3/reference/options/contracts"
        params = {"underlying_ticker": sym, "limit": 1000}
        for _ in range(5):  # max 5 pages
            data = self._get(next_url, params=params)
            if not data:
                break
            for c in data.get("results", []) or []:
                d = c.get("expiration_date")
                if d:
                    out.add(d)
            nxt = data.get("next_url")
            if not nxt:
                break
            # next_url comes back fully qualified — strip BASE
            next_url = nxt.replace(self.BASE, "")
            params = {}
        return sorted(out)

    def fundamentals(self, symbol: str) -> dict:
        sym = symbol.upper()
        data = self._get(f"/v3/reference/tickers/{sym}")
        if not data or not data.get("results"):
            return {}
        r = data["results"]
        return {
            "symbol":        sym,
            "name":          r.get("name"),
            "market_cap":    r.get("market_cap"),
            "shares_outstanding": r.get("share_class_shares_outstanding"),
            "shares_float":  r.get("weighted_shares_outstanding"),
            "sector":        r.get("sic_description"),
            "exchange":      r.get("primary_exchange"),
            "currency":      r.get("currency_name"),
        }

    def earnings_calendar(self, symbol: str) -> list[date]:
        # Polygon does not expose an earnings calendar in the standard REST tier.
        # Scanners fall back to Finnhub or yfinance.
        return []
