"""Smoke tests for the provider registry + priority selection.

We don't make any real API calls — just verify imports, class shapes, and
that the priority selector returns the expected provider for each combination
of configured keys.
"""
import json
from pathlib import Path
import pytest


def test_all_provider_modules_import():
    from apexflow.providers.schwab_provider import (
        SchwabProvider, SchwabAuthRequired, build_auth_url,
        exchange_code_for_tokens, refresh_access_token,
    )
    from apexflow.providers.polygon_provider import PolygonProvider
    from apexflow.providers.unusual_whales_provider import UnusualWhalesProvider
    from apexflow.providers.marketdata_provider import MarketDataProvider
    from apexflow.providers.finnhub_provider import FinnhubClient
    from apexflow.providers.fintel_provider import FintelClient
    from apexflow.providers.yfinance_provider import YFinanceProvider
    # Each provider should expose a 'name' attribute
    for cls in [SchwabProvider, PolygonProvider, UnusualWhalesProvider,
                 MarketDataProvider, YFinanceProvider]:
        assert hasattr(cls, "name") and isinstance(cls.name, str)


def test_priority_selector_yfinance_default(monkeypatch):
    """No keys → falls back to yfinance."""
    import config
    from apexflow.providers import reset_provider, get_provider

    monkeypatch.setattr(config, "SCHWAB_APP_KEY", "")
    monkeypatch.setattr(config, "SCHWAB_APP_SECRET", "")
    monkeypatch.setattr(config, "POLYGON_KEY", "")
    monkeypatch.setattr(config, "MARKETDATA_TOKEN", "")
    reset_provider()
    assert get_provider().name == "yfinance"


def test_priority_polygon_when_only_polygon(monkeypatch):
    import config
    from apexflow.providers import reset_provider, get_provider
    monkeypatch.setattr(config, "SCHWAB_APP_KEY", "")
    monkeypatch.setattr(config, "POLYGON_KEY", "fake")
    monkeypatch.setattr(config, "MARKETDATA_TOKEN", "")
    reset_provider()
    assert get_provider().name == "polygon"


def test_priority_marketdata_when_only_marketdata(monkeypatch):
    import config
    from apexflow.providers import reset_provider, get_provider
    monkeypatch.setattr(config, "SCHWAB_APP_KEY", "")
    monkeypatch.setattr(config, "POLYGON_KEY", "")
    monkeypatch.setattr(config, "MARKETDATA_TOKEN", "fake")
    reset_provider()
    assert get_provider().name == "marketdata"


def test_priority_schwab_requires_tokens(monkeypatch, tmp_path):
    """Schwab keys WITHOUT a tokens file should NOT activate Schwab."""
    import config
    from apexflow.providers import reset_provider, get_provider

    monkeypatch.setattr(config, "SCHWAB_APP_KEY", "fake")
    monkeypatch.setattr(config, "SCHWAB_APP_SECRET", "fake")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "POLYGON_KEY", "")
    monkeypatch.setattr(config, "MARKETDATA_TOKEN", "")
    reset_provider()
    p = get_provider()
    assert p.name != "schwab"


def test_priority_schwab_wins_when_authed(monkeypatch, tmp_path):
    import config
    from apexflow.providers import reset_provider, get_provider

    # Write a tokens file with a refresh_token
    (tmp_path / "schwab_tokens.json").write_text(json.dumps({
        "access_token": "fake", "refresh_token": "fake",
        "access_expires_at": 9999999999, "refresh_expires_at": 9999999999,
    }))

    monkeypatch.setattr(config, "SCHWAB_APP_KEY", "fake")
    monkeypatch.setattr(config, "SCHWAB_APP_SECRET", "fake")
    monkeypatch.setattr(config, "SCHWAB_REDIRECT_URI", "https://127.0.0.1")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "POLYGON_KEY", "fake")  # Schwab should still win
    monkeypatch.setattr(config, "MARKETDATA_TOKEN", "fake")
    reset_provider()
    assert get_provider().name == "schwab"


def test_schwab_tokens_present_rejects_empty(tmp_path, monkeypatch):
    """Empty {} file should not count as authed."""
    import config
    (tmp_path / "schwab_tokens.json").write_text("{}")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert config.schwab_tokens_present() is False


def test_build_auth_url_includes_required_params():
    from apexflow.providers.schwab_provider import build_auth_url
    url = build_auth_url("MY_APP_KEY", "https://127.0.0.1:8443")
    assert "client_id=MY_APP_KEY" in url
    assert "redirect_uri=https" in url
    assert "response_type=code" in url


def test_supplementary_clients_return_none_without_keys(monkeypatch):
    import config
    from apexflow.providers import get_flow_client, get_squeeze_client, get_news_client
    monkeypatch.setattr(config, "UNUSUAL_WHALES_KEY", "")
    monkeypatch.setattr(config, "FINTEL_KEY", "")
    monkeypatch.setattr(config, "FINNHUB_KEY", "")
    assert get_flow_client() is None
    assert get_squeeze_client() is None
    assert get_news_client() is None
