"""Fintel short-interest helper. Used by the squeeze scanner; not a full DataProvider.

Sign up:  https://fintel.io/
Cost:     ~$20/mo
API docs: https://api.fintel.io/

Endpoints relevant to ApexFlow:
  GET /sf/us/{symbol}     → short interest, days-to-cover, utilization
  GET /so/us/{symbol}     → squeeze score
"""
from __future__ import annotations
import logging
import requests

log = logging.getLogger(__name__)


class FintelClient:
    BASE = "https://api.fintel.io"

    def __init__(self, key: str):
        self.key = key
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": key})

    def short_interest(self, symbol: str) -> dict:
        try:
            r = self.session.get(f"{self.BASE}/sf/us/{symbol}", timeout=10)
            r.raise_for_status()
            d = r.json()
            return {
                "short_pct_float": d.get("shortPercentFloat", 0) / 100,
                "days_to_cover": d.get("daysToCover", 0),
                "utilization": d.get("utilization", 0),
                "borrow_rate": d.get("costToBorrow", 0) / 100,
            }
        except Exception as e:
            log.warning("Fintel short interest fetch failed for %s: %s", symbol, e)
            return {}

    def squeeze_score(self, symbol: str) -> float:
        try:
            r = self.session.get(f"{self.BASE}/so/us/{symbol}", timeout=10)
            r.raise_for_status()
            return float(r.json().get("score", 0))
        except Exception:
            return 0.0
