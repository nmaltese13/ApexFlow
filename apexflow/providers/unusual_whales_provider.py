"""Unusual Whales API client.

Used as the *flow_client* by scanners — UnusualWhalesProvider does not own the
primary price/options chain feed; it provides flow alerts, dark pool prints,
and GEX/short-interest endpoints. Combine with Schwab/Polygon for chains and
Fintel/Schwab for fundamentals.

Sign up:    https://unusualwhales.com — Plus ~$48/mo, Pro ~$75/mo
API docs:   https://api.unusualwhales.com/docs
Auth:       Bearer token in Authorization header

Endpoints implemented here:
  - /api/option-trades/flow-alerts        SkylitAi-equivalent flow feed
  - /api/dark-pool/recent                 dark pool prints (timestamp, size, price)
  - /api/dark-pool/{ticker}               per-ticker dark pool history
  - /api/stock/{ticker}/greek-exposure    pre-computed GEX/DEX by strike
  - /api/stock/{ticker}/options-volume    daily volume vs OI per contract
  - /api/stock/{ticker}/short-interest    UW's short data (often 1-day fresher than FINRA)

For methods we don't implement, the BaseScanner falls back to its primary
provider — that's why DataProvider's optional methods like options_flow()
and dark_pool_prints() return [] by default.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from .base import DataProvider
from apexflow.models import OptionsPrint, DarkPoolPrint

log = logging.getLogger(__name__)


class UnusualWhalesProvider(DataProvider):
    name = "unusual_whales"
    BASE = "https://api.unusualwhales.com"

    def __init__(self, api_key: str):
        self.key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept":         "application/json",
        })
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._ttl = 30

    def _get(self, path: str, params: dict | None = None) -> Any:
        key = (path, tuple(sorted((params or {}).items())))
        now = time.time()
        if key in self._cache and now - self._cache[key][0] < self._ttl:
            return self._cache[key][1]
        try:
            r = self.session.get(f"{self.BASE}{path}", params=params or {}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("UW GET %s failed: %s", path, e)
            return None
        self._cache[key] = (now, data)
        return data

    # ---- Flow feed --------------------------------------------------------
    def options_flow(self, symbol: str | None = None,
                     since_minutes: int = 60) -> list[OptionsPrint]:
        """Fetch the flow-alerts feed, optionally filtered to one ticker.

        UW classifies each print as SWEEP / BLOCK / SWEEP_OF_BLOCKS in the
        ``rule_name`` field, plus a side classification (ASK/BID/MID).
        """
        params: dict = {"limit": 200}
        if symbol:
            params["ticker_symbol"] = symbol.upper()
        if since_minutes:
            cutoff = datetime.now(timezone.utc).timestamp() - since_minutes * 60
            params["min_executed_at"] = int(cutoff)
        data = self._get("/api/option-trades/flow-alerts", params=params)
        rows = (data or {}).get("data") or (data or {}).get("flow_alerts") or []
        out: list[OptionsPrint] = []
        for r in rows:
            try:
                ts = datetime.fromtimestamp(int(r.get("executed_at", 0)) / 1000,
                                              tz=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            try:
                out.append(OptionsPrint(
                    symbol=str(r.get("ticker") or r.get("underlying_symbol") or "").upper(),
                    strike=float(r.get("strike") or 0),
                    expiry=str(r.get("expiry") or r.get("expiration") or ""),
                    right=("C" if (r.get("type") or r.get("option_type") or "")[:1].upper() == "C" else "P"),
                    size=int(r.get("size") or r.get("trade_size") or 0),
                    price=float(r.get("price") or r.get("trade_price") or 0),
                    premium=float(r.get("premium") or 0),
                    side=(r.get("side") or r.get("trade_side") or "UNKNOWN").upper(),
                    volume=int(r.get("volume") or 0),
                    open_interest=int(r.get("open_interest") or 0),
                    iv=float(r.get("implied_volatility")) if r.get("implied_volatility") else None,
                    print_type=str(r.get("rule_name") or "TRADE").upper(),
                    timestamp=ts,
                ))
            except Exception as e:
                log.debug("UW row parse failed: %s", e)
        return out

    # ---- Dark pool --------------------------------------------------------
    def dark_pool_prints(self, symbol: str | None = None,
                          since_minutes: int = 60) -> list[DarkPoolPrint]:
        params: dict = {"limit": 200}
        if since_minutes:
            cutoff = datetime.now(timezone.utc).timestamp() - since_minutes * 60
            params["min_executed_at"] = int(cutoff)
        path = f"/api/dark-pool/{symbol.upper()}" if symbol else "/api/dark-pool/recent"
        data = self._get(path, params=params)
        rows = (data or {}).get("data") or []
        out: list[DarkPoolPrint] = []
        for r in rows:
            try:
                ts = datetime.fromtimestamp(int(r.get("executed_at", 0)) / 1000,
                                              tz=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            try:
                size = int(r.get("size") or r.get("share_count") or 0)
                price = float(r.get("price") or 0)
                out.append(DarkPoolPrint(
                    symbol=str(r.get("ticker") or symbol or "").upper(),
                    size=size, price=price, notional=size * price,
                    venue=str(r.get("venue") or r.get("ats") or "DARKPOOL"),
                    timestamp=ts,
                ))
            except Exception:
                continue
        return out

    # ---- Short interest --------------------------------------------------
    def short_interest_detail(self, symbol: str) -> dict:
        sym = symbol.upper()
        data = self._get(f"/api/stock/{sym}/short-interest")
        if not data:
            return {}
        d = data.get("data") if isinstance(data, dict) else data
        if isinstance(d, list) and d:
            d = d[0]
        if not isinstance(d, dict):
            return {}
        sf = d.get("short_percent_of_float") or d.get("short_percentage_float") or 0
        # Some endpoints return percentages (e.g. 22.5), some fractions (0.225) — normalise.
        sf = float(sf)
        if sf > 1.5:
            sf = sf / 100.0
        return {
            "short_pct_float": sf,
            "days_to_cover":   float(d.get("days_to_cover") or 0),
            "short_volume":    float(d.get("short_volume") or 0),
            "borrow_rate":     float(d.get("cost_to_borrow") or 0) / 100.0,
            "utilization":     float(d.get("utilization") or 0) / 100.0,
        }

    # ---- UW pre-computed GEX (handy cross-check vs our own chain_gex) ----
    def greek_exposure(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        data = self._get(f"/api/stock/{sym}/greek-exposure")
        rows = (data or {}).get("data") or []
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ---- DataProvider interface (UW is supplementary) --------------------
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        # Not provided — caller should use Schwab/Polygon/yfinance for prices
        return pd.DataFrame()

    def quote(self, symbol: str) -> dict:
        return {}

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "spot": 0, "expiry": expiry}

    def expiries(self, symbol: str) -> list[str]:
        return []

    def fundamentals(self, symbol: str) -> dict:
        return {}

    def earnings_calendar(self, symbol: str):
        return []
