"""Data providers — priority-selected at import time.

Architecture:
  - get_provider()       → main price/options data
                            priority: demo > Schwab > Polygon > MarketData.app > yfinance
  - get_flow_client()    → Unusual Whales (real flow + dark pool, optional)
  - get_squeeze_client() → Fintel (short interest, optional)
  - get_news_client()    → Finnhub (news + earnings, optional)

Set the relevant key(s) in keys.py and the right provider activates
automatically. If no paid keys are present, falls back to yfinance.

Demo mode (`APEXFLOW_DEMO=1`) overrides all of it and serves the frozen
snapshot in `data/demo/` instead — no key, no network, no rate limit. It is
checked first precisely so it is unambiguous: when demo mode is on, nothing
is reaching the network, which is what makes it safe to hand to someone else
to run.
"""
from __future__ import annotations
import logging

from .base import DataProvider
from .yfinance_provider import YFinanceProvider
from .demo_provider import demo_available

import config

log = logging.getLogger(__name__)

# Cache the chosen primary provider so we don't re-instantiate on every call.
_primary_cache: DataProvider | None = None


def _build_primary() -> DataProvider:
    """Pick the best primary provider based on configured keys."""
    # 0. Demo — frozen snapshot, wins over everything when explicitly enabled
    if config.DEMO_MODE:
        try:
            from .demo_provider import DemoProvider, demo_available
            if demo_available():
                p = DemoProvider()
                log.info("Provider: demo (frozen snapshot from %s)",
                         p.capture_date.isoformat())
                return p
            log.warning("APEXFLOW_DEMO=1 but no dataset in data/demo — "
                        "run scripts/capture_demo_dataset.py. Falling back.")
        except Exception as e:
            log.warning("Demo provider init failed: %s — falling back", e)

    # 1. Schwab — free with brokerage, best data when set up
    if (config.SCHWAB_APP_KEY and config.SCHWAB_APP_SECRET
            and config.schwab_tokens_present()):
        try:
            from .schwab_provider import SchwabProvider
            log.info("Provider: Schwab")
            return SchwabProvider(
                app_key=config.SCHWAB_APP_KEY,
                app_secret=config.SCHWAB_APP_SECRET,
                redirect_uri=config.SCHWAB_REDIRECT_URI,
            )
        except Exception as e:
            log.warning("Schwab init failed: %s — falling back", e)

    # 2. Polygon — paid, real-time
    if config.POLYGON_KEY:
        try:
            from .polygon_provider import PolygonProvider
            log.info("Provider: Polygon")
            return PolygonProvider(config.POLYGON_KEY)
        except Exception as e:
            log.warning("Polygon init failed: %s — falling back", e)

    # 3. MarketData.app — cheap fallback with greeks
    if config.MARKETDATA_TOKEN:
        try:
            from .marketdata_provider import MarketDataProvider
            log.info("Provider: MarketData.app")
            return MarketDataProvider(config.MARKETDATA_TOKEN)
        except Exception as e:
            log.warning("MarketData init failed: %s — falling back", e)

    # 4. yfinance — always available
    log.info("Provider: yfinance (free fallback)")
    return YFinanceProvider()


def get_provider() -> DataProvider:
    global _primary_cache
    if _primary_cache is None:
        _primary_cache = _build_primary()
    return _primary_cache


def reset_provider() -> None:
    """Force re-selection on next call (e.g. after running `schwab-auth`)."""
    global _primary_cache
    _primary_cache = None


def get_flow_client():
    """Unusual Whales client if configured, else None."""
    if config.UNUSUAL_WHALES_KEY:
        try:
            from .unusual_whales_provider import UnusualWhalesProvider
            return UnusualWhalesProvider(config.UNUSUAL_WHALES_KEY)
        except Exception as e:
            log.warning("Unusual Whales init failed: %s", e)
    return None


def get_squeeze_client():
    """Fintel client if configured, else None."""
    if config.FINTEL_KEY:
        try:
            from .fintel_provider import FintelClient
            return FintelClient(config.FINTEL_KEY)
        except Exception as e:
            log.warning("Fintel init failed: %s", e)
    return None


def get_news_client():
    """Finnhub client if configured, else None."""
    if config.FINNHUB_KEY:
        try:
            from .finnhub_provider import FinnhubClient
            return FinnhubClient(config.FINNHUB_KEY)
        except Exception as e:
            log.warning("Finnhub init failed: %s", e)
    return None


__all__ = [
    "DataProvider", "YFinanceProvider", "demo_available",
    "get_provider", "reset_provider",
    "get_flow_client", "get_squeeze_client", "get_news_client",
]
