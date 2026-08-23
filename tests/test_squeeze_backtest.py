"""The squeeze-scorer backtest harness.

The most important tests here are the **power tests**: a harness that finds
nothing is worthless unless it can be shown to find something when
something is there. So the suite injects known relationships of known
strength and asserts they are detected, then injects pure noise and asserts
it is not. Without both halves, "no signal detected" is indistinguishable
from "the harness is broken".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexflow.analytics.squeeze import COMPONENT_MAX
from apexflow.platform.squeeze_backtest import (
    SqueezeBacktester, PriceDerivedSource, SqueezeBacktestReport,
    _spearman, _rank, _obv_trend, _bucket_table,
    MIN_BREADTH, MIN_MEANINGFUL_COVERAGE, TOTAL_POINTS,
)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def make_panel(n_dates: int = 200, n_names: int = 40, ic: float = 0.0,
               seed: int = 0) -> pd.DataFrame:
    """Synthetic panel where score predicts forward return with strength `ic`.

    `ic` is the per-date correlation planted between score and return. At
    ic=0 the two are independent.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        scores = rng.uniform(0, 100, n_names)
        z = (scores - scores.mean()) / (scores.std() or 1.0)
        noise = rng.standard_normal(n_names)
        signal = ic * z + np.sqrt(max(1 - ic * ic, 0.0)) * noise
        for s, r in zip(scores, signal):
            rows.append({"date": d, "symbol": f"S{int(s*100)%997:03d}",
                         "score": float(s), "fwd_return": float(r * 3.0)})
    return pd.DataFrame(rows)


class FullCoverageSource:
    """Stand-in source that claims full coverage, for testing the judge."""
    covers = frozenset(COMPONENT_MAX)
    coverage = 1.0
    missing: list[str] = []

    def inputs_at(self, symbol, as_of, history):  # pragma: no cover
        raise NotImplementedError


def evaluate(panel, hold_days=10, n_permutations=200, source=None):
    bt = SqueezeBacktester(provider=object(), source=source or FullCoverageSource())
    return bt.evaluate(panel, hold_days=hold_days, n_permutations=n_permutations)


# ---------------------------------------------------------------------------
# Power: can it find a signal that is really there?
# ---------------------------------------------------------------------------
class TestPower:
    def test_detects_a_strong_planted_signal(self):
        rep = evaluate(make_panel(ic=0.30, seed=1))
        assert rep.rank_ic_all_dates > 0.15
        assert rep.t_stat > 3.0
        assert rep.null_percentile > 97.5
        assert rep.conclusive
        assert "positive rank information" in rep.verdict

    def test_detects_a_moderate_planted_signal(self):
        rep = evaluate(make_panel(ic=0.15, n_dates=300, seed=2))
        assert rep.rank_ic_all_dates > 0.05
        assert rep.null_percentile > 97.5

    def test_detects_an_inverted_signal(self):
        """A score that predicts the wrong way must be flagged, not ignored."""
        rep = evaluate(make_panel(ic=-0.30, seed=3))
        assert rep.rank_ic_all_dates < -0.15
        assert rep.null_percentile < 2.5
        assert rep.conclusive
        assert "negative rank information" in rep.verdict

    def test_ic_estimate_tracks_the_planted_strength(self):
        weak = evaluate(make_panel(ic=0.05, seed=4)).rank_ic_all_dates
        strong = evaluate(make_panel(ic=0.35, seed=4)).rank_ic_all_dates
        assert strong > weak + 0.15


# ---------------------------------------------------------------------------
# Specificity: does it stay quiet when there is nothing there?
# ---------------------------------------------------------------------------
class TestSpecificity:
    @pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
    def test_pure_noise_is_not_called_a_signal(self, seed):
        rep = evaluate(make_panel(ic=0.0, seed=seed))
        assert abs(rep.rank_ic_all_dates) < 0.05
        assert "no detectable rank information" in rep.verdict

    def test_noise_null_percentile_is_roughly_uniform(self):
        """Across seeds, the null percentile should not cluster at an extreme."""
        pct = [evaluate(make_panel(ic=0.0, seed=s), n_permutations=150).null_percentile
               for s in range(20, 30)]
        assert 15 < float(np.mean(pct)) < 85
        assert sum(p > 97.5 or p < 2.5 for p in pct) <= 2

    def test_constant_scores_yield_no_ic(self):
        panel = make_panel(ic=0.0, seed=31)
        panel["score"] = 50.0
        rep = evaluate(panel)
        assert rep.rank_ic_all_dates == 0.0
        assert not rep.conclusive


# ---------------------------------------------------------------------------
# Overlap handling
# ---------------------------------------------------------------------------
class TestOverlapCorrection:
    def test_independent_subsample_is_smaller_than_the_date_count(self):
        rep = evaluate(make_panel(n_dates=200, ic=0.0, seed=41), hold_days=10)
        assert rep.n_dates == 200
        assert 15 <= rep.n_independent_dates <= 25

    def test_longer_holds_shrink_the_independent_sample(self):
        panel = make_panel(n_dates=200, ic=0.0, seed=42)
        short = evaluate(panel, hold_days=2)
        long = evaluate(panel, hold_days=20)
        assert short.n_independent_dates > long.n_independent_dates

    def test_offset_averaging_recovers_the_all_dates_mean(self):
        """Averaging every offset must reproduce the full-sample IC.

        This is what stops the significance test depending on an arbitrary
        starting point — an earlier version used offset 0 only and reported
        a 96th-percentile result that vanished once every offset was used.
        """
        rep = evaluate(make_panel(n_dates=200, ic=0.1, seed=43), hold_days=10)
        assert rep.rank_ic == pytest.approx(rep.rank_ic_all_dates, abs=1e-9)

    def test_offset_spread_is_reported(self):
        rep = evaluate(make_panel(n_dates=200, ic=0.0, seed=44), hold_days=10)
        assert rep.t_stat_spread >= 0.0

    def test_overlap_correction_lowers_significance(self):
        """The whole point: a naive N would give a much larger t."""
        panel = make_panel(n_dates=300, ic=0.06, seed=45)
        rep = evaluate(panel, hold_days=15)
        naive_t = (rep.rank_ic_all_dates / rep.rank_ic_std * np.sqrt(rep.n_dates)
                   if rep.rank_ic_std > 0 else 0.0)
        assert abs(rep.t_stat) < abs(naive_t)


# ---------------------------------------------------------------------------
# Coverage gating — the reason this module exists
# ---------------------------------------------------------------------------
class TestCoverageGate:
    def test_price_source_coverage_is_low_and_honest(self):
        src = PriceDerivedSource()
        assert src.covers == frozenset({"rvol", "accumulation"})
        expected = (COMPONENT_MAX["rvol"] + COMPONENT_MAX["accumulation"]) / TOTAL_POINTS
        assert src.coverage == pytest.approx(expected)
        assert src.coverage < MIN_MEANINGFUL_COVERAGE

    def test_missing_axes_named_explicitly(self):
        missing = PriceDerivedSource().missing
        for axis in ("short_float", "days_to_cover", "borrow", "float_size"):
            assert axis in missing

    def test_low_coverage_is_reported_inconclusive_even_with_a_strong_signal(self):
        """A planted signal must not produce a verdict on 18% of the model."""
        rep = evaluate(make_panel(ic=0.40, seed=51), source=PriceDerivedSource())
        assert rep.rank_ic_all_dates > 0.2      # the signal is plainly there
        assert not rep.conclusive               # but the claim is refused
        assert "coverage" in rep.verdict.lower()

    def test_realised_vol_proxy_raises_coverage_but_stays_below_the_gate(self):
        src = PriceDerivedSource(use_realised_vol_proxy=True)
        assert "iv_hv" in src.covers
        assert src.coverage > PriceDerivedSource().coverage
        assert src.coverage < MIN_MEANINGFUL_COVERAGE

    def test_proxy_is_off_by_default(self):
        assert PriceDerivedSource().use_realised_vol_proxy is False

    def test_full_coverage_allows_a_verdict(self):
        rep = evaluate(make_panel(ic=0.30, seed=52), source=FullCoverageSource())
        assert rep.conclusive

    def test_coverage_caveat_always_listed_when_partial(self):
        rep = evaluate(make_panel(seed=53), source=PriceDerivedSource())
        assert any("coverage" in c or "reconstructable" in c for c in rep.caveats)

    def test_survivorship_caveat_always_present(self):
        rep = evaluate(make_panel(seed=54))
        assert any("survivorship" in c for c in rep.caveats)


# ---------------------------------------------------------------------------
# Breadth gating
# ---------------------------------------------------------------------------
class TestBreadthGate:
    def test_narrow_cross_section_is_refused(self):
        rep = evaluate(make_panel(n_names=MIN_BREADTH - 5, seed=61))
        assert not rep.conclusive
        assert "breadth" in rep.verdict.lower() or "breadth" in " ".join(rep.caveats)

    def test_adequate_breadth_passes_the_gate(self):
        rep = evaluate(make_panel(n_names=MIN_BREADTH + 20, ic=0.3, seed=62))
        assert rep.conclusive

    def test_empty_panel_is_handled(self):
        rep = evaluate(pd.DataFrame())
        assert rep.n_observations == 0
        assert not rep.conclusive


# ---------------------------------------------------------------------------
# No lookahead in panel construction
# ---------------------------------------------------------------------------
class _RecordingProvider:
    """Provider that hands out a known series and records nothing else."""

    def __init__(self, n=400):
        idx = pd.bdate_range("2023-01-01", periods=n)
        rng = np.random.default_rng(7)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        self.df = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": rng.uniform(1e6, 5e6, n),
        }, index=idx)

    def history(self, symbol, period="2y", interval="1d"):
        return self.df.copy()


class _SpySource:
    """Records the last timestamp of every history window it is handed."""
    covers = frozenset({"rvol"})
    coverage = 0.12
    missing: list[str] = []

    def __init__(self):
        self.seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def inputs_at(self, symbol, as_of, history):
        from apexflow.analytics.squeeze import SqueezeInputs
        self.seen.append((as_of, history.index[-1]))
        return SqueezeInputs(rvol=1.0, price=float(history["Close"].iloc[-1]))


class TestNoLookahead:
    def test_history_window_never_extends_past_the_signal_date(self):
        spy = _SpySource()
        bt = SqueezeBacktester(provider=_RecordingProvider(), source=spy)
        bt.build_panel(["AAA"], lookback_days=100, hold_days=5)
        assert spy.seen
        for as_of, last in spy.seen:
            assert last == as_of, "history window extended beyond the signal date"

    def test_forward_return_uses_bars_after_the_signal(self):
        prov = _RecordingProvider()
        bt = SqueezeBacktester(provider=prov, source=PriceDerivedSource())
        panel = bt.build_panel(["AAA"], lookback_days=60, hold_days=5)
        assert not panel.empty
        close = prov.df["Close"]
        row = panel.iloc[0]
        i = prov.df.index.get_loc(row["date"])
        expected = (close.iloc[i + 5] / close.iloc[i] - 1) * 100
        assert row["fwd_return"] == pytest.approx(expected, abs=1e-9)

    def test_last_hold_days_bars_produce_no_signals(self):
        """There is no forward return to measure there, so they must be skipped."""
        prov = _RecordingProvider()
        bt = SqueezeBacktester(provider=prov, source=PriceDerivedSource())
        panel = bt.build_panel(["AAA"], lookback_days=400, hold_days=10)
        assert panel["date"].max() <= prov.df.index[-11]

    def test_short_history_is_skipped_not_crashed(self):
        class Tiny:
            def history(self, *a, **k):
                return pd.DataFrame({"Close": [1.0, 2.0], "Volume": [1.0, 2.0]})
        bt = SqueezeBacktester(provider=Tiny(), source=PriceDerivedSource())
        assert bt.build_panel(["AAA"]).empty

    def test_provider_errors_are_survived(self):
        class Broken:
            def history(self, *a, **k):
                raise RuntimeError("network down")
        bt = SqueezeBacktester(provider=Broken(), source=PriceDerivedSource())
        assert bt.build_panel(["AAA", "BBB"]).empty


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
class TestSpearman:
    def test_perfect_monotone_is_one(self):
        x = np.arange(30, dtype=float)
        assert _spearman(x, x * 2 + 1) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_inverse_is_minus_one(self):
        x = np.arange(30, dtype=float)
        assert _spearman(x, -x) == pytest.approx(-1.0, abs=1e-9)

    def test_rank_based_not_value_based(self):
        """A monotone but wildly non-linear map is still rank-correlation 1."""
        x = np.arange(1, 31, dtype=float)
        assert _spearman(x, np.exp(x / 3)) == pytest.approx(1.0, abs=1e-9)

    def test_returns_none_below_breadth_minimum(self):
        x = np.arange(MIN_BREADTH - 1, dtype=float)
        assert _spearman(x, x) is None

    def test_returns_none_for_constant_input(self):
        x = np.ones(30)
        assert _spearman(x, np.arange(30, dtype=float)) is None

    def test_ties_share_average_rank(self):
        r = _rank(np.array([10.0, 20.0, 20.0, 30.0]))
        assert r[1] == r[2] == pytest.approx(1.5)
        assert r[0] == 0.0 and r[3] == 3.0


class TestObvTrend:
    def test_steady_accumulation_is_positive(self):
        n = 60
        close = pd.Series(np.linspace(100, 130, n))
        volume = pd.Series(np.full(n, 1e6))
        assert _obv_trend(close, volume, 20) > 0.3

    def test_steady_distribution_is_negative(self):
        n = 60
        close = pd.Series(np.linspace(130, 100, n))
        volume = pd.Series(np.full(n, 1e6))
        assert _obv_trend(close, volume, 20) < -0.3

    def test_bounded_in_minus_one_to_one(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            n = 80
            close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))))
            volume = pd.Series(rng.uniform(1e5, 1e7, n))
            assert -1.0 <= _obv_trend(close, volume, 20) <= 1.0

    def test_short_series_returns_zero(self):
        assert _obv_trend(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), 20) == 0.0


class TestBucketTable:
    def test_buckets_are_ordered_and_complete(self):
        rows, tmb = _bucket_table(make_panel(ic=0.3, seed=71), 10)
        assert [r["bucket"] for r in rows] == list(range(1, 11))
        assert all(r["n"] > 0 for r in rows)

    def test_mean_score_increases_across_buckets(self):
        rows, _ = _bucket_table(make_panel(ic=0.0, seed=72), 10)
        scores = [r["mean_score"] for r in rows]
        assert scores == sorted(scores)

    def test_planted_signal_shows_in_top_minus_bottom(self):
        _, tmb = _bucket_table(make_panel(ic=0.4, seed=73), 10)
        assert tmb > 0

    def test_empty_panel_returns_empty(self):
        rows, tmb = _bucket_table(pd.DataFrame(), 10)
        assert rows == [] and tmb == 0.0


class TestReport:
    def test_is_json_serialisable(self):
        import json
        json.dumps(evaluate(make_panel(seed=81)).to_dict())

    def test_summary_flags_inconclusive_runs(self):
        rep = evaluate(make_panel(seed=82), source=PriceDerivedSource())
        assert rep.summary.startswith("INCONCLUSIVE")

    def test_summary_reports_the_statistic_when_conclusive(self):
        rep = evaluate(make_panel(ic=0.3, seed=83))
        assert "rank IC" in rep.summary


# ---------------------------------------------------------------------------
# The scanner backtester's scope guard (platform/backtest.py)
# ---------------------------------------------------------------------------
class TestScannerBacktestScope:
    """Unsupported scanners must fail loudly, not return an empty result.

    An empty BacktestResult reads as "no signals fired" — a statement about
    the market — when it actually means "this was never evaluated", a
    statement about the data. Conflating those is the kind of quiet wrong
    answer a backtest exists to avoid.
    """

    def test_options_driven_scanners_raise_with_a_reason(self):
        from apexflow.platform.backtest import Backtester, UNSUPPORTED_REASON
        bt = Backtester.__new__(Backtester)      # skip provider construction
        bt.provider = None
        for name in ("earnings", "options-flow"):
            with pytest.raises(ValueError) as exc:
                bt.run(name, ["SPY"])
            msg = str(exc.value)
            assert name in msg
            assert UNSUPPORTED_REASON[name].split(";")[0][:20] in msg
            assert "backtest-squeeze" in msg

    def test_unknown_scanner_still_raises(self):
        from apexflow.platform.backtest import Backtester
        bt = Backtester.__new__(Backtester)
        bt.provider = None
        with pytest.raises(ValueError, match="Unknown scanner"):
            bt.run("not-a-scanner", ["SPY"])

    def test_supported_set_is_price_only(self):
        from apexflow.platform.backtest import SUPPORTED_SCANNERS
        assert SUPPORTED_SCANNERS == frozenset({"pre-breakout", "momentum", "squeeze"})

    def test_every_unsupported_scanner_has_a_stated_reason(self):
        from apexflow.scanners import ALL_SCANNERS
        from apexflow.platform.backtest import SUPPORTED_SCANNERS, UNSUPPORTED_REASON
        for name in ALL_SCANNERS:
            if name not in SUPPORTED_SCANNERS:
                assert UNSUPPORTED_REASON.get(name), f"{name} has no stated reason"

    def test_would_fire_rejects_anything_unsupported(self):
        from apexflow.platform.backtest import Backtester
        bt = Backtester.__new__(Backtester)
        bt.provider = None
        window = pd.DataFrame({
            "Close": np.linspace(100, 110, 80), "High": np.linspace(101, 111, 80),
            "Low": np.linspace(99, 109, 80), "Volume": np.full(80, 1e6),
        })
        with pytest.raises(ValueError):
            bt._would_fire("earnings", window)
