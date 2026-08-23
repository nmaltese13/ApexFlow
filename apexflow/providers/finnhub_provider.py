"""Finnhub helper. Used by the momentum ignition scanner for catalyst tags
and by the earnings scanner as a more reliable earnings calendar than yfinance.

Sign up:  https://finnhub.io/  (free tier, 60 req/min — enough for personal use)
Docs:     https://finnhub.io/docs/api
"""
from __future__ import annotations
import logging
import time
from datetime import date, datetime
import requests

log = logging.getLogger(__name__)


class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, key: str):
        self.key = key
        self.session = requests.Session()
        self.session.params = {"token": key}  # type: ignore[assignment]

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        try:
            r = self.session.get(f"{self.BASE}{path}",
                                 params={**(params or {}), "token": self.key}, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("Finnhub %s failed: %s", path, e)
            return {}

    def news(self, symbol: str, days: int = 3, limit: int = 10) -> list[dict]:
        from_ = date.fromtimestamp(time.time() - days * 86400).isoformat()
        to_ = date.today().isoformat()
        data = self._get("/company-news", {"symbol": symbol, "from": from_, "to": to_})
        if not isinstance(data, list):
            return []
        return [{
            "title": n.get("headline", ""),
            "publisher": n.get("source", ""),
            "link": n.get("url", ""),
            "published": n.get("datetime", 0),
        } for n in data[:limit]]

    def earnings_calendar(self, symbol: str, days_ahead: int = 30) -> list[date]:
        from_ = date.today().isoformat()
        to_ = date.fromtimestamp(time.time() + days_ahead * 86400).isoformat()
        data = self._get("/calendar/earnings",
                         {"symbol": symbol, "from": from_, "to": to_})
        if not isinstance(data, dict):
            return []
        out = []
        for e in data.get("earningsCalendar", []) or []:
            try:
                out.append(date.fromisoformat(e["date"]))
            except (KeyError, ValueError):
                continue
        return out
    def earnings_calendar_market(self, from_date, to_date) -> list[dict]:
        """Full-market earnings calendar between two dates.

        Returns a list of dicts: [{symbol, date, eps_estimate, eps_actual,
        revenue_estimate, revenue_actual, hour}].
        ``hour`` is "bmo" (before market open), "amc" (after market close),
        or "" (during market).
        """
        from_ = from_date.isoformat() if hasattr(from_date, "isoformat") else str(from_date)
        to_ = to_date.isoformat() if hasattr(to_date, "isoformat") else str(to_date)
        data = self._get("/calendar/earnings", {"from": from_, "to": to_})
        if not isinstance(data, dict):
            return []
        out = []
        for e in data.get("earningsCalendar", []) or []:
            try:
                out.append({
                    "symbol":           e.get("symbol", "").upper(),
                    "date":             e.get("date"),
                    "eps_estimate":     e.get("epsEstimate"),
                    "eps_actual":       e.get("epsActual"),
                    "revenue_estimate": e.get("revenueEstimate"),
                    "revenue_actual":   e.get("revenueActual"),
                    "hour":             (e.get("hour") or "").lower(),
                    "year":             e.get("year"),
                    "quarter":          e.get("quarter"),
                })
            except Exception:
                continue
        return out

