"""Alpaca provider stub. Auto-activates when ALPACA_KEY_ID + ALPACA_SECRET_KEY are set.

Sign up:  https://alpaca.markets/  (free, no minimum)
Docs:     https://alpaca.markets/docs/api-references/market-data-api/

Auth headers:
  APCA-API-KEY-ID:     <ALPACA_KEY_ID>
  APCA-API-SECRET-KEY: <ALPACA_SECRET_KEY>

Useful endpoints:
  GET  https://data.alpaca.markets/v2/stocks/{symbol}/bars     → OHLCV
  GET  https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest
  GET  https://data.alpaca.markets/v1beta1/options/snapshots/{underlying}

This stub falls through to yfinance for everything until you fill in the calls.
"""
from __future__ import annotations
import requests

from .base import DataProvider
from .yfinance_provider import YFinanceProvider


class AlpacaProvider(DataProvider):
    name = "alpaca"
    DATA_BASE = "https://data.alpaca.markets"

    def __init__(self, key_id: str, secret: str):
        self.key_id = key_id
        self.secret = secret
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        })
        self._fallback = YFinanceProvider()

    # All methods delegate until implemented.
    def history(self, *a, **k):           return self._fallback.history(*a, **k)
    def quote(self, *a, **k):             return self._fallback.quote(*a, **k)
    def expiries(self, *a, **k):          return self._fallback.expiries(*a, **k)
    def options_chain(self, *a, **k):     return self._fallback.options_chain(*a, **k)
    def fundamentals(self, *a, **k):      return self._fallback.fundamentals(*a, **k)
    def earnings_calendar(self, *a, **k): return self._fallback.earnings_calendar(*a, **k)
