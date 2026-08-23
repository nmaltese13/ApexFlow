"""Abstract data provider interface. All providers implement these methods."""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from apexflow.models import OptionsPrint, DarkPoolPrint


class DataProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """OHLCV DataFrame indexed by datetime. Columns: Open, High, Low, Close, Volume."""

    @abstractmethod
    def quote(self, symbol: str) -> dict:
        """Latest snapshot: price, change, volume, etc."""

    @abstractmethod
    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        """Returns {'calls': DataFrame, 'puts': DataFrame, 'expiry': str, 'spot': float}.
        DataFrames have columns: strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility.
        """

    @abstractmethod
    def expiries(self, symbol: str) -> list[str]:
        """Available expirations as YYYY-MM-DD strings."""

    @abstractmethod
    def fundamentals(self, symbol: str) -> dict:
        """Snapshot of float, short interest, days-to-cover, market cap, sector, etc."""

    @abstractmethod
    def earnings_calendar(self, symbol: str) -> list[date]:
        """Upcoming earnings dates."""

    # Optional — paid providers override; free returns empty.
    def options_flow(self, symbol: str | None = None, since_minutes: int = 60) -> list[OptionsPrint]:
        return []

    def dark_pool_prints(self, symbol: str | None = None, since_minutes: int = 60) -> list[DarkPoolPrint]:
        return []

    def short_interest_detail(self, symbol: str) -> dict:
        """Borrow rate, utilization, etc. Free providers return {} or minimal info."""
        return {}

    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        return []
