"""Position sizing and data-freshness — the two safety layers.

Sizing is arithmetic, so it can be pinned exactly. Freshness is about
refusing to present old data as current, so most of these assert what the
module *declines* to call tradeable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from apexflow.platform.risk import (
    size_by_stop, size_options_by_premium, r_multiple, kelly_fraction,
    MAX_SANE_RISK_FRACTION, CONTRACT_MULTIPLIER,
)
from apexflow.platform.freshness import assess, REALTIME_SOURCES

OPEN = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)     # Mon 13:00 ET
SAT = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)      # Saturday


class TestFixedFractionalSizing:
    def test_worked_example(self):
        """$50k, 1% risk, $3 stop -> floor(500/3) = 166 shares."""
        s = size_by_stop(50_000, 0.01, 100.0, 97.0)
        assert s.ok and s.quantity == 166
        assert s.risk_amount == pytest.approx(498.0)

    def test_risk_never_exceeds_the_budget(self):
        """The defining property: loss is capped by construction."""
        for equity in (5_000, 50_000, 250_000):
            for frac in (0.0025, 0.005, 0.01, 0.02):
                for stop_dist in (0.25, 1.0, 3.0, 12.5):
                    s = size_by_stop(equity, frac, 100.0, 100.0 - stop_dist)
                    if s.ok:
                        assert s.risk_amount <= equity * frac + 1e-9

    def test_tighter_stop_buys_more_size(self):
        wide = size_by_stop(100_000, 0.01, 100.0, 90.0).quantity
        tight = size_by_stop(100_000, 0.01, 100.0, 98.0).quantity
        assert tight > wide

    def test_size_scales_linearly_with_equity(self):
        small = size_by_stop(25_000, 0.01, 100.0, 97.0).quantity
        big = size_by_stop(250_000, 0.01, 100.0, 97.0).quantity
        assert big == pytest.approx(small * 10, rel=0.01)

    def test_contracts_apply_the_multiplier(self):
        shares = size_by_stop(100_000, 0.01, 5.0, 4.0, unit="shares").quantity
        contracts = size_by_stop(100_000, 0.01, 5.0, 4.0, unit="contracts").quantity
        assert shares == contracts * CONTRACT_MULTIPLIER

    def test_notional_cap_is_separate_from_risk_cap(self):
        s = size_by_stop(50_000, 0.01, 100.0, 99.9, max_position_fraction=0.25)
        assert s.position_value <= 50_000 * 0.25 + 1e-6
        assert any("notional" in w.lower() for w in s.warnings)

    def test_quantity_is_a_whole_number(self):
        s = size_by_stop(33_333, 0.0137, 91.37, 88.11)
        assert isinstance(s.quantity, int)


class TestSizingRefusals:
    def test_refuses_absurd_risk_fraction(self):
        s = size_by_stop(50_000, 0.20, 100.0, 97.0)
        assert not s.ok and s.quantity == 0
        assert any("unrecoverable" in w for w in s.warnings)

    def test_max_sane_fraction_is_conservative(self):
        assert MAX_SANE_RISK_FRACTION <= 0.05

    def test_zero_stop_distance_is_refused(self):
        s = size_by_stop(50_000, 0.01, 100.0, 100.0)
        assert not s.ok
        assert "zero" in s.detail.lower()

    def test_stop_too_wide_for_account_is_explained(self):
        s = size_by_stop(2_000, 0.01, 500.0, 400.0)
        assert not s.ok and s.quantity == 0
        assert "too wide" in s.detail or "too big" in s.detail

    @pytest.mark.parametrize("equity,frac,entry,stop", [
        (0, 0.01, 100, 97), (-5, 0.01, 100, 97), (50_000, 0, 100, 97),
        (50_000, -0.01, 100, 97), (50_000, 0.01, 0, 97), (50_000, 0.01, 100, 0),
    ])
    def test_invalid_inputs_refuse_rather_than_raise(self, equity, frac, entry, stop):
        s = size_by_stop(equity, frac, entry, stop)
        assert not s.ok and s.quantity == 0

    def test_very_tight_stop_is_flagged(self):
        s = size_by_stop(500_000, 0.01, 100.0, 99.95)
        assert any("noise" in w for w in s.warnings)


class TestOptionSizing:
    def test_premium_is_the_max_loss(self):
        s = size_options_by_premium(50_000, 0.01, 3.50)
        assert s.ok
        assert s.risk_amount == pytest.approx(s.quantity * 3.50 * CONTRACT_MULTIPLIER)

    def test_says_the_whole_premium_is_at_risk(self):
        s = size_options_by_premium(50_000, 0.02, 2.00)
        assert any("entire premium" in w for w in s.warnings)

    def test_stop_on_underlying_caveat_is_stated(self):
        """The trap: an underlying stop does not cap an option loss."""
        s = size_by_stop(100_000, 0.01, 4.00, 2.00, unit="contracts")
        assert any("does not cap an option loss" in w for w in s.warnings)

    def test_expensive_premium_can_be_unaffordable(self):
        s = size_options_by_premium(5_000, 0.005, 40.0)
        assert not s.ok


class TestRMultiple:
    def test_three_to_one(self):
        r = r_multiple(100, 97, 109)
        assert r["r"] == pytest.approx(3.0)
        assert r["breakeven_win_rate"] == pytest.approx(0.25)

    def test_one_to_one_needs_half(self):
        assert r_multiple(100, 95, 105)["breakeven_win_rate"] == pytest.approx(0.5)

    def test_direction_agnostic(self):
        """Works for shorts as well as longs."""
        assert r_multiple(100, 103, 91)["r"] == pytest.approx(3.0)

    def test_zero_risk_is_undefined_not_infinite(self):
        assert r_multiple(100, 100, 110)["breakeven_win_rate"] is None


class TestKelly:
    def test_positive_edge_gives_positive_stake(self):
        assert kelly_fraction(0.55, 1.5)["kelly"] > 0

    def test_no_edge_gives_zero(self):
        assert kelly_fraction(0.40, 1.0)["kelly"] == 0.0

    def test_output_is_capped(self):
        k = kelly_fraction(0.90, 5.0)
        assert k["kelly"] > k["capped"]
        assert k["capped"] <= MAX_SANE_RISK_FRACTION

    def test_half_kelly_is_half(self):
        k = kelly_fraction(0.60, 2.0)
        assert k["half_kelly"] == pytest.approx(k["kelly"] / 2)

    def test_warns_about_estimation_error(self):
        assert "exactly" in kelly_fraction(0.60, 2.0)["detail"]

    @pytest.mark.parametrize("wr,rr", [(0, 1.5), (1, 1.5), (-0.1, 1.5), (0.5, 0)])
    def test_invalid_inputs_return_zero(self, wr, rr):
        assert kelly_fraction(wr, rr)["kelly"] == 0.0


class TestFreshness:
    def test_delayed_feed_is_never_called_live(self):
        """Cboe's file is seconds old; the prices in it are ~15 minutes old."""
        f = assess(OPEN - timedelta(seconds=5), source="cboe", now=OPEN)
        assert f.level == "delayed"
        assert "cboe" not in REALTIME_SOURCES

    def test_realtime_feed_can_be_live(self):
        f = assess(OPEN - timedelta(seconds=20), source="schwab", now=OPEN)
        assert f.level == "live" and f.tradeable

    def test_realtime_feed_going_quiet_is_stale(self):
        """12 minutes is fine for Cboe and alarming for Schwab."""
        assert assess(OPEN - timedelta(minutes=12), "schwab", OPEN).level == "stale"
        assert assess(OPEN - timedelta(minutes=12), "cboe", OPEN).level == "delayed"

    def test_stale_data_is_not_tradeable(self):
        f = assess(OPEN - timedelta(hours=3), source="cboe", now=OPEN)
        assert f.level == "stale" and not f.tradeable
        assert "may have stopped" in f.detail

    def test_missing_timestamp_is_unknown_not_assumed_fresh(self):
        f = assess(None, source="cboe", now=OPEN)
        assert f.level == "unknown" and not f.tradeable

    def test_demo_data_is_never_tradeable(self):
        f = assess(OPEN, source="demo", now=OPEN)
        assert not f.tradeable and "not tradeable" in f.detail

    def test_last_sessions_close_is_expected_when_shut(self):
        """Old data on a Saturday is correct, not a fault."""
        f = assess(SAT - timedelta(hours=19), source="cboe", now=SAT)
        assert f.level == "delayed"
        assert "last session" in f.detail

    def test_missing_a_whole_session_is_stale(self):
        f = assess(SAT - timedelta(days=3), source="cboe", now=SAT)
        assert f.level == "stale" and "missed a day" in f.detail

    def test_nothing_outside_an_open_session_is_tradeable(self):
        for delta in (timedelta(hours=19), timedelta(days=3)):
            assert not assess(SAT - delta, source="cboe", now=SAT).tradeable

    @pytest.mark.parametrize("stamp", [
        OPEN - timedelta(seconds=30),
        (OPEN - timedelta(seconds=30)).timestamp(),
        (OPEN - timedelta(seconds=30)).isoformat(),
    ])
    def test_accepts_datetime_epoch_or_iso(self, stamp):
        assert assess(stamp, source="cboe", now=OPEN).level == "delayed"

    def test_garbage_timestamp_is_unknown(self):
        assert assess("not-a-date", source="cboe", now=OPEN).level == "unknown"

    def test_serialises(self):
        import json
        json.dumps(assess(OPEN, source="cboe", now=OPEN).to_dict())


class TestMarketSessions:
    """The session logic freshness depends on — DST and holidays."""

    def test_dst_transition_is_handled(self):
        from apexflow.analytics.timeutil import market_state
        # 14:00 UTC is 10:00 ET in summer (open) and 09:00 ET in winter (pre).
        assert market_state(datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)) == "open"
        assert market_state(datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)) == "premarket"

    def test_holidays_are_closed(self):
        from apexflow.analytics.timeutil import market_state
        # MLK Day 2026 is a Monday.
        assert market_state(datetime(2026, 1, 19, 16, 0, tzinfo=timezone.utc)) == "closed"

    def test_good_friday_is_closed(self):
        from apexflow.analytics.timeutil import market_holidays
        from datetime import date
        assert date(2026, 4, 3) in market_holidays(2026)

    def test_weekend_is_closed(self):
        from apexflow.analytics.timeutil import market_state
        assert market_state(SAT) == "closed"

    def test_early_close_days_end_at_one(self):
        from apexflow.analytics.timeutil import session_close, EARLY_CLOSE
        from datetime import date
        assert session_close(date(2026, 11, 27)) == EARLY_CLOSE   # day after Thanksgiving
