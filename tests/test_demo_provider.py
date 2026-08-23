"""The frozen offline dataset and the provider that reads it.

These tests run against the committed snapshot in ``data/demo/``, so they
also serve as a check that the shipped dataset is intact and loadable — the
thing that has to work for someone cloning the repo with no API key.
"""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from apexflow.providers.demo_provider import (
    DemoProvider, demo_available, _round_weeks, DEMO_DIR,
)

pytestmark = pytest.mark.skipif(
    not demo_available(),
    reason="no frozen dataset in data/demo (run scripts/capture_demo_dataset.py)",
)


@pytest.fixture(scope="module")
def provider():
    return DemoProvider(shift_dates=False)


@pytest.fixture(scope="module")
def symbol(provider):
    syms = provider.symbols()
    assert syms, "demo dataset has no symbols"
    return syms[0]


class TestDatasetIntegrity:
    def test_manifest_is_present_and_complete(self):
        m = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
        for key in ("captured_at", "capture_date", "source", "symbols",
                    "total_contracts"):
            assert key in m, f"manifest missing {key}"
        assert m["total_contracts"] > 0

    def test_every_manifest_symbol_has_a_file(self):
        m = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
        for entry in m["symbols"]:
            assert (DEMO_DIR / entry["file"]).exists()

    def test_snapshot_files_are_valid_gzip_json(self, provider):
        for sym in provider.symbols():
            with gzip.open(DEMO_DIR / f"{sym}.json.gz", "rt", encoding="utf-8") as fh:
                snap = json.load(fh)
            assert snap["symbol"] == sym
            assert snap["chains"]
            assert snap["spot"] > 0

    def test_dataset_is_small_enough_to_commit(self):
        total = sum(p.stat().st_size for p in DEMO_DIR.glob("*.json.gz"))
        assert total < 25 * 1024 * 1024, "demo dataset has grown past 25 MB"


class TestProviderSurface:
    def test_name_is_demo(self, provider):
        assert provider.name == "demo"

    def test_expiries_are_sorted_iso_dates(self, provider, symbol):
        exps = provider.expiries(symbol)
        assert exps == sorted(exps)
        for e in exps:
            datetime.strptime(e, "%Y-%m-%d")

    def test_options_chain_shape(self, provider, symbol):
        ch = provider.options_chain(symbol)
        assert ch["spot"] > 0
        assert ch["expiry"] in provider.expiries(symbol)
        for side in ("calls", "puts"):
            df = ch[side]
            assert isinstance(df, pd.DataFrame)
            for col in ("strike", "openInterest", "impliedVolatility", "bid", "ask"):
                assert col in df.columns
        assert not ch["calls"].empty

    def test_chain_columns_are_numeric(self, provider, symbol):
        ch = provider.options_chain(symbol)
        for col in ("strike", "openInterest", "impliedVolatility"):
            assert pd.api.types.is_numeric_dtype(ch["calls"][col])

    def test_specific_expiry_is_honoured(self, provider, symbol):
        exps = provider.expiries(symbol)
        if len(exps) < 2:
            pytest.skip("need two expiries")
        ch = provider.options_chain(symbol, exps[1])
        assert ch["expiry"] == exps[1]

    def test_unknown_expiry_falls_back_to_the_front(self, provider, symbol):
        ch = provider.options_chain(symbol, "1999-01-01")
        assert ch["expiry"] == provider.expiries(symbol)[0]

    def test_history_is_a_dated_ohlcv_frame(self, provider, symbol):
        df = provider.history(symbol, period="3mo")
        assert not df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing

    def test_history_period_limits_rows(self, provider, symbol):
        short = provider.history(symbol, period="1mo")
        long = provider.history(symbol, period="1y")
        assert len(short) <= len(long)

    def test_quote_has_a_price(self, provider, symbol):
        q = provider.quote(symbol)
        assert q["price"] > 0
        assert q["symbol"] == symbol

    def test_fundamentals_and_earnings_do_not_raise(self, provider, symbol):
        assert isinstance(provider.fundamentals(symbol), dict)
        assert isinstance(provider.earnings_calendar(symbol), list)

    def test_news_is_deliberately_empty(self, provider, symbol):
        """Stale headlines presented as current are worse than none."""
        assert provider.news(symbol) == []

    def test_unknown_symbol_degrades_quietly(self, provider):
        assert provider.expiries("NOTATICKER") == []
        assert provider.quote("NOTATICKER") == {}
        assert provider.history("NOTATICKER").empty
        ch = provider.options_chain("NOTATICKER")
        assert ch["calls"].empty and ch["spot"] == 0.0

    def test_info_reports_provenance(self, provider):
        info = provider.info()
        assert info["source"]
        assert info["total_contracts"] > 0
        date.fromisoformat(info["capture_date"])


class TestDateShifting:
    def test_round_weeks(self):
        assert _round_weeks(0) == 0
        assert _round_weeks(3) == 0        # rounds down to the same week
        assert _round_weeks(4) == 7
        assert _round_weeks(10) == 7
        assert _round_weeks(11) == 14
        assert _round_weeks(365) == 364

    def test_shift_is_always_a_whole_number_of_weeks(self, provider):
        """Weekday alignment is what keeps Friday expiries on Fridays."""
        p = DemoProvider(shift_dates=True)
        assert p.shift_days % 7 == 0

    def test_shifted_expiries_keep_their_weekday(self):
        raw = DemoProvider(shift_dates=False)
        shifted = DemoProvider(shift_dates=True)
        sym = raw.symbols()[0]
        for a, b in zip(raw.expiries(sym), shifted.expiries(sym)):
            da = datetime.strptime(a, "%Y-%m-%d").date()
            db = datetime.strptime(b, "%Y-%m-%d").date()
            assert da.weekday() == db.weekday()

    def test_shifting_preserves_the_gaps_between_expiries(self):
        """The term structure must keep its exact original shape."""
        raw = DemoProvider(shift_dates=False)
        shifted = DemoProvider(shift_dates=True)
        sym = raw.symbols()[0]
        ra = [datetime.strptime(e, "%Y-%m-%d").date() for e in raw.expiries(sym)]
        sh = [datetime.strptime(e, "%Y-%m-%d").date() for e in shifted.expiries(sym)]
        assert [(b - a).days for a, b in zip(ra, ra[1:])] == \
               [(b - a).days for a, b in zip(sh, sh[1:])]

    def test_shifting_does_not_touch_prices_or_open_interest(self):
        """Only labels move. Not one number changes."""
        raw = DemoProvider(shift_dates=False)
        shifted = DemoProvider(shift_dates=True)
        sym = raw.symbols()[0]
        a = raw.options_chain(sym, raw.expiries(sym)[0])
        b = shifted.options_chain(sym, shifted.expiries(sym)[0])
        assert a["spot"] == b["spot"]
        for side in ("calls", "puts"):
            pd.testing.assert_frame_equal(a[side], b[side])

    def test_shifted_chain_is_retrievable_by_its_shifted_expiry(self):
        """The round trip that makes shifting invisible to callers."""
        p = DemoProvider(shift_dates=True)
        sym = p.symbols()[0]
        for e in p.expiries(sym):
            assert p.options_chain(sym, e)["expiry"] == e

    def test_history_shifts_with_the_expiries(self):
        raw = DemoProvider(shift_dates=False)
        shifted = DemoProvider(shift_dates=True)
        sym = raw.symbols()[0]
        a = raw.history(sym, period="1mo")
        b = shifted.history(sym, period="1mo")
        assert len(a) == len(b)
        delta = (b.index[-1] - a.index[-1]).days
        assert delta == shifted.shift_days

    def test_unshifted_provider_returns_the_captured_dates(self, provider):
        m = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
        captured = {e["symbol"]: e["expiries"] for e in m["symbols"]}
        sym = provider.symbols()[0]
        assert provider.expiries(sym) == sorted(captured[sym])


class TestAnalyticsRunOnDemoData:
    """The point of the dataset: the real engines work against it offline."""

    def test_gex_computes_on_every_symbol(self, provider):
        from apexflow.analytics.gex import chain_gex, gex_summary
        for sym in provider.symbols():
            ch = provider.options_chain(sym)
            df = chain_gex(ch["calls"], ch["puts"], ch["spot"], ch["expiry"])
            assert not df.empty, f"{sym} produced no GEX"
            s = gex_summary(df, ch["spot"])
            assert s["total_gex"] == s["total_gex"]      # not NaN

    def test_dealer_exposures_compute(self, provider, symbol):
        from apexflow.analytics.dealer_greeks import (
            chain_exposures, exposure_summary, vex_summary)
        ch = provider.options_chain(symbol)
        e = chain_exposures(ch["calls"], ch["puts"], ch["spot"], ch["expiry"])
        assert not e.empty
        s = exposure_summary(e, ch["spot"])
        assert s["regime"] in ("positive_gamma", "negative_gamma")
        assert isinstance(vex_summary(e, ch["spot"])["net_short_vega"], bool)

    def test_atm_iv_is_extractable_for_every_symbol(self, provider):
        from apexflow.analytics.iv_surface import first_usable_atm_iv
        for sym in provider.symbols():
            chains = []
            spot = 0.0
            for e in provider.expiries(sym):
                ch = provider.options_chain(sym, e)
                spot = ch["spot"]
                chains.append((e, ch["calls"], ch["puts"]))
            iv, quality, _ = first_usable_atm_iv(chains, spot)
            assert iv > 0, f"{sym} yielded no ATM IV at all"
            assert quality in ("good", "fair", "poor")

    def test_monte_carlo_runs_on_a_demo_symbol(self, provider, symbol):
        from apexflow.analytics import montecarlo as mc
        from apexflow.analytics.iv_surface import first_usable_atm_iv
        chains, spot = [], 0.0
        for e in provider.expiries(symbol):
            ch = provider.options_chain(symbol, e)
            spot = ch["spot"]
            chains.append((e, ch["calls"], ch["puts"]))
        iv, _, _ = first_usable_atm_iv(chains, spot)
        res = mc.simulate_cone(spot, 5 / 365, iv, mc.MCConfig(n_paths=20_000, n_steps=16))
        assert res.n_paths == 20_000
        assert res.quantiles["p5"] < spot < res.quantiles["p95"]
