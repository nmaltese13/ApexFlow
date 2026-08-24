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


def _no_paid_keys(monkeypatch):
    import config
    monkeypatch.setattr(config, "SCHWAB_APP_KEY", "")
    monkeypatch.setattr(config, "SCHWAB_APP_SECRET", "")
    monkeypatch.setattr(config, "POLYGON_KEY", "")
    monkeypatch.setattr(config, "MARKETDATA_TOKEN", "")
    return config


def test_priority_selector_cboe_default(monkeypatch):
    """No keys → Cboe, not yfinance.

    Cboe publishes exchange-computed IV and Greeks for the whole chain in a
    single keyless request, so it is the better free default. yfinance
    remains the last resort.
    """
    from apexflow.providers import reset_provider, get_provider

    config = _no_paid_keys(monkeypatch)
    monkeypatch.setattr(config, "CBOE_ENABLED", True)
    reset_provider()
    assert get_provider().name == "cboe"


def test_priority_selector_yfinance_when_cboe_disabled(monkeypatch):
    """APEXFLOW_DISABLE_CBOE must still leave a working free provider."""
    from apexflow.providers import reset_provider, get_provider

    config = _no_paid_keys(monkeypatch)
    monkeypatch.setattr(config, "CBOE_ENABLED", False)
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


# ---------------------------------------------------------------------------
# Cboe — OCC symbol parsing and index mapping (offline, no network)
# ---------------------------------------------------------------------------
class TestCboeSymbolHandling:
    def test_occ_parse_roundtrip(self):
        from apexflow.providers.cboe_provider import parse_occ
        assert parse_occ("AAPL260824C00205000") == ("2026-08-24", "C", 205.0)
        assert parse_occ("SPY261218P00700000") == ("2026-12-18", "P", 700.0)
        # Fractional strikes survive the /1000 scaling.
        assert parse_occ("GME260904C00018500") == ("2026-09-04", "C", 18.5)

    def test_occ_parse_handles_index_underscore(self):
        from apexflow.providers.cboe_provider import parse_occ
        assert parse_occ("_SPX260320C05000000") == ("2026-03-20", "C", 5000.0)

    def test_occ_parse_rejects_junk(self):
        from apexflow.providers.cboe_provider import parse_occ
        for bad in ("", "NOTASYMBOL", "AAPL2608C00205000", "AAPL269924C00205000"):
            assert parse_occ(bad) is None

    def test_index_symbols_get_underscore_prefix(self):
        from apexflow.providers.cboe_provider import cboe_symbol
        assert cboe_symbol("SPX") == "_SPX"
        assert cboe_symbol("VIX") == "_VIX"
        assert cboe_symbol("^VIX") == "_VIX"
        assert cboe_symbol("AAPL") == "AAPL"
        assert cboe_symbol("spy") == "SPY"


# ---------------------------------------------------------------------------
# Short interest — the publication lag is the whole point
# ---------------------------------------------------------------------------
class TestShortInterestPointInTime:
    def _history(self):
        from datetime import date
        from apexflow.providers.shortinterest_provider import (
            ShortInterestHistory, ShortInterestRecord, _publication_date)
        recs = []
        for settle, shares, dtc in [
            (date(2025, 1, 15), 1_000_000, 2.0),
            (date(2025, 1, 31), 2_000_000, 4.0),
            (date(2025, 2, 14), 3_000_000, 6.0),
        ]:
            recs.append(ShortInterestRecord(
                settlement_date=settle, publication_date=_publication_date(settle),
                shares_short=shares, avg_daily_volume=500_000, days_to_cover=dtc))
        return ShortInterestHistory(
            symbol="TEST", records=recs,
            share_counts=[(date(2024, 11, 1), 10_000_000),
                          (date(2025, 2, 10), 20_000_000)])

    def test_publication_lags_settlement(self):
        from datetime import date
        from apexflow.providers.shortinterest_provider import _publication_date
        for settle in (date(2025, 1, 15), date(2025, 1, 31), date(2025, 6, 30)):
            assert _publication_date(settle) > settle

    def test_record_invisible_before_publication(self):
        """The core anti-lookahead guarantee."""
        from datetime import timedelta
        h = self._history()
        first = h.records[0]
        assert h.as_of(first.settlement_date) is None, \
            "settlement-dated figure must not be visible on its settlement date"
        assert h.as_of(first.publication_date - timedelta(days=1)) is None
        assert h.as_of(first.publication_date) is first

    def test_as_of_returns_latest_published(self):
        from datetime import date
        h = self._history()
        got = h.as_of(date(2025, 3, 1))
        assert got is h.records[-1]

    def test_share_count_filtered_on_filed_date(self):
        from datetime import date
        h = self._history()
        assert h.shares_outstanding_as_of(date(2025, 1, 1)) == 10_000_000
        assert h.shares_outstanding_as_of(date(2025, 2, 9)) == 10_000_000
        assert h.shares_outstanding_as_of(date(2025, 2, 11)) == 20_000_000

    def test_short_pct_uses_point_in_time_share_count(self):
        from datetime import date
        h = self._history()
        # 2025-01-27: first record published (1M short), share count still 10M.
        pct = h.short_pct_as_of(date(2025, 1, 28))
        assert pct == pytest.approx(0.10)

    def test_short_pct_none_when_nothing_published(self):
        from datetime import date
        h = self._history()
        assert h.short_pct_as_of(date(2024, 1, 1)) is None

    def test_sec_user_agent_has_no_default(self, monkeypatch):
        """An email address must never be baked into the codebase."""
        from apexflow.providers.shortinterest_provider import (
            sec_user_agent, SEC_USER_AGENT_ENV)
        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        assert sec_user_agent() is None
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Me me@example.com")
        assert sec_user_agent() == "Me me@example.com"


class TestHistoricalShortInterestSource:
    def test_coverage_clears_the_verdict_gate(self):
        from apexflow.platform.squeeze_backtest import (
            HistoricalShortInterestSource, MIN_MEANINGFUL_COVERAGE)
        src = HistoricalShortInterestSource(client=object())
        assert {"short_float", "days_to_cover"} <= src.covers
        assert src.coverage > MIN_MEANINGFUL_COVERAGE

    def test_borrow_and_float_remain_uncovered(self):
        """Neither is available free; the source must not pretend otherwise."""
        from apexflow.platform.squeeze_backtest import HistoricalShortInterestSource
        src = HistoricalShortInterestSource(client=object())
        assert "borrow" in src.missing
        assert "float_size" in src.missing
