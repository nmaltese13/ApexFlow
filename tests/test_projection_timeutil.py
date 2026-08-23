"""Closed-form projection maths, and the shared time-to-expiry helper."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from apexflow.analytics.projection import (
    expected_move, lognormal_std, above_prob, touch_prob,
    touch_prob_reflection, projection_cone, horizon_years,
)
from apexflow.analytics.timeutil import (
    years_to_expiry, days_to_expiry, is_expired, market_close_utc,
    parse_expiry, now_utc, MINUTE_YEARS,
)

SPOT, T, IV, R = 100.0, 0.25, 0.30, 0.04


class TestExpectedMove:
    def test_one_sigma_formula(self):
        assert expected_move(100.0, 0.25, 0.40) == pytest.approx(100 * 0.40 * 0.5)

    def test_scales_with_sqrt_time(self):
        a = expected_move(SPOT, 0.25, IV)
        b = expected_move(SPOT, 1.00, IV)
        assert b / a == pytest.approx(2.0, rel=1e-9)

    @pytest.mark.parametrize("args", [(0, T, IV), (SPOT, 0, IV), (SPOT, T, 0), (-1, T, IV)])
    def test_invalid_returns_zero(self, args):
        assert expected_move(*args) == 0.0

    def test_shorthand_is_close_to_exact_lognormal_sd(self):
        """The sqrt-t shorthand differs from the exact SD by O(sigma^2 t)."""
        approx = expected_move(SPOT, T, IV)
        exact = lognormal_std(SPOT, T, IV, R, 0.0)
        assert abs(approx - exact) / exact < 0.05


class TestAboveProb:
    def test_at_the_money_is_near_half(self):
        assert above_prob(SPOT, SPOT, T, IV, 0.0, 0.0) == pytest.approx(0.5, abs=0.05)

    def test_monotone_decreasing_in_strike(self):
        prev = 1.1
        for k in (80, 90, 100, 110, 120):
            p = above_prob(SPOT, k, T, IV, R)
            assert p < prev
            prev = p

    def test_far_otm_approaches_zero_and_deep_itm_approaches_one(self):
        assert above_prob(SPOT, 10_000.0, T, IV, R) == pytest.approx(0.0, abs=1e-6)
        assert above_prob(SPOT, 0.01, T, IV, R) == pytest.approx(1.0, abs=1e-6)

    def test_is_n_d2_exactly(self):
        from scipy.stats import norm
        k = 105.0
        d2 = (math.log(SPOT / k) + (R - 0.5 * IV * IV) * T) / (IV * math.sqrt(T))
        assert above_prob(SPOT, k, T, IV, R) == pytest.approx(float(norm.cdf(d2)), abs=1e-12)

    def test_invalid_inputs_return_half(self):
        assert above_prob(0.0, 100.0, T, IV) == 0.5


class TestTouchProb:
    def test_touch_always_exceeds_terminal_probability(self):
        """A level is easier to tag than to close beyond."""
        for k in (90.0, 95.0, 105.0, 110.0):
            beyond = above_prob(SPOT, k, T, IV, R)
            tail = beyond if k > SPOT else 1 - beyond
            assert touch_prob(SPOT, k, T, IV, R) > tail

    def test_barrier_at_spot_is_certain(self):
        assert touch_prob(SPOT, SPOT, T, IV, R) == pytest.approx(1.0)

    def test_bounded_in_zero_one(self):
        for k in (1.0, 50.0, 99.0, 101.0, 200.0, 10_000.0):
            assert 0.0 <= touch_prob(SPOT, k, T, IV, R) <= 1.0

    def test_monotone_in_distance(self):
        """The further the barrier, the less likely it is touched."""
        ups = [touch_prob(SPOT, k, T, IV, R) for k in (102, 105, 110, 120)]
        assert ups == sorted(ups, reverse=True)
        downs = [touch_prob(SPOT, k, T, IV, R) for k in (98, 95, 90, 80)]
        assert downs == sorted(downs, reverse=True)

    def test_monotone_in_time(self):
        probs = [touch_prob(SPOT, 110.0, t, IV, R) for t in (0.01, 0.1, 0.5, 2.0)]
        assert probs == sorted(probs)

    def test_zero_drift_matches_reflection_shortcut(self):
        """The old approximation is exactly right when drift is zero.

        With r = q = 0 the log drift is -sigma^2/2, not zero, so use the
        vol-adjusted rate that makes the log-drift vanish.
        """
        r_star = 0.5 * IV * IV
        exact = touch_prob(SPOT, 110.0, T, IV, r=r_star, q=0.0)
        approx = touch_prob_reflection(SPOT, 110.0, T, IV, r=r_star, q=0.0)
        assert exact == pytest.approx(approx, abs=1e-9)

    def test_reflection_shortcut_disagrees_under_drift(self):
        """With real drift the shortcut is wrong — quantify by how much."""
        r_big = 0.25
        exact = touch_prob(SPOT, 90.0, 1.0, IV, r=r_big)
        approx = touch_prob_reflection(SPOT, 90.0, 1.0, IV, r=r_big)
        assert abs(exact - approx) > 0.02

    def test_extreme_barrier_does_not_overflow(self):
        assert 0.0 <= touch_prob(SPOT, 1e-6, T, IV, R) <= 1.0
        assert 0.0 <= touch_prob(SPOT, 1e9, T, IV, R) <= 1.0

    def test_invalid_inputs_return_half(self):
        assert touch_prob(SPOT, 0.0, T, IV) == 0.5
        assert touch_prob(SPOT, 105.0, 0.0, IV) == 0.5


class TestProjectionCone:
    def test_quantiles_ordered_at_every_step(self):
        for row in projection_cone(SPOT, T, IV, 20, R):
            assert row["p5"] < row["p25"] < row["p50"] < row["p75"] < row["p95"]

    def test_widens_monotonically(self):
        widths = [r["p95"] - r["p5"] for r in projection_cone(SPOT, T, IV, 20, R)]
        assert all(b > a for a, b in zip(widths, widths[1:]))

    def test_length_and_frac_endpoints(self):
        cone = projection_cone(SPOT, T, IV, 12, R)
        assert len(cone) == 12
        assert cone[-1]["frac"] == pytest.approx(1.0)
        assert cone[-1]["t_years"] == pytest.approx(T)

    def test_median_follows_the_log_drift(self):
        cone = projection_cone(SPOT, T, IV, 5, R, 0.0)
        want = SPOT * math.exp((R - 0.5 * IV * IV) * T)
        assert cone[-1]["p50"] == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("args", [(0, T, IV), (SPOT, 0, IV), (SPOT, T, 0)])
    def test_invalid_returns_empty(self, args):
        assert projection_cone(*args) == []

    def test_zero_steps_returns_empty(self):
        assert projection_cone(SPOT, T, IV, 0) == []


class TestHorizonYears:
    def test_one_day(self):
        assert horizon_years(1.0) == pytest.approx(1 / 365)

    def test_floors_at_a_positive_value(self):
        assert horizon_years(0.0) > 0
        assert horizon_years(-5.0) > 0


class TestTimeUtil:
    def test_parse_accepts_several_shapes(self):
        assert parse_expiry("2027-03-19") == date(2027, 3, 19)
        assert parse_expiry(date(2027, 3, 19)) == date(2027, 3, 19)
        assert parse_expiry(datetime(2027, 3, 19, 12)) == date(2027, 3, 19)
        assert parse_expiry("2027-03-19T00:00:00") == date(2027, 3, 19)
        assert parse_expiry("garbage") is None
        assert parse_expiry(None) is None

    def test_dst_summer_close_is_2000_utc(self):
        """16:00 America/New_York in July is 20:00 UTC."""
        c = market_close_utc(date(2027, 7, 15))
        assert (c.hour, c.minute) == (20, 0)

    def test_standard_time_close_is_2100_utc(self):
        """16:00 America/New_York in January is 21:00 UTC."""
        c = market_close_utc(date(2027, 1, 15))
        assert (c.hour, c.minute) == (21, 0)

    def test_dst_and_std_closes_differ_by_an_hour(self):
        summer = market_close_utc(date(2027, 7, 15))
        winter = market_close_utc(date(2027, 1, 15))
        assert winter.hour - summer.hour == 1

    def test_zero_dte_shrinks_through_the_session(self):
        day = date(2027, 7, 15)
        morning = datetime(2027, 7, 15, 14, 0, tzinfo=timezone.utc)   # 10:00 ET
        afternoon = datetime(2027, 7, 15, 19, 0, tzinfo=timezone.utc)  # 15:00 ET
        t_am = years_to_expiry(day.isoformat(), now=morning)
        t_pm = years_to_expiry(day.isoformat(), now=afternoon)
        assert t_am > t_pm > 0
        # 10:00 ET to close is 6 hours.
        assert t_am == pytest.approx(6 / (365 * 24), rel=1e-6)

    def test_expired_contract_floors_instead_of_going_negative(self):
        past = datetime(2027, 7, 16, 12, 0, tzinfo=timezone.utc)
        t = years_to_expiry("2027-07-15", now=past)
        assert t == pytest.approx(MINUTE_YEARS)
        assert t > 0

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2027, 7, 15, 14, 0)
        aware = datetime(2027, 7, 15, 14, 0, tzinfo=timezone.utc)
        assert years_to_expiry("2027-08-20", now=naive) == \
            pytest.approx(years_to_expiry("2027-08-20", now=aware))

    def test_unparseable_expiry_falls_back_to_one_day(self):
        assert years_to_expiry("not-a-date") == pytest.approx(1 / 365)

    def test_thirty_days_out_is_about_a_twelfth_of_a_year(self):
        now = datetime(2027, 7, 15, 20, 0, tzinfo=timezone.utc)
        t = years_to_expiry("2027-08-14", now=now)
        assert t == pytest.approx(30 / 365, rel=0.01)

    def test_days_to_expiry(self):
        now = datetime(2027, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert days_to_expiry("2027-07-15", now=now) == 0
        assert days_to_expiry("2027-07-22", now=now) == 7
        assert days_to_expiry("2027-07-08", now=now) == -7

    def test_is_expired_flips_at_the_close(self):
        before = datetime(2027, 7, 15, 19, 0, tzinfo=timezone.utc)   # 15:00 ET
        after = datetime(2027, 7, 15, 20, 30, tzinfo=timezone.utc)   # 16:30 ET
        assert not is_expired("2027-07-15", now=before)
        assert is_expired("2027-07-15", now=after)

    def test_now_utc_is_aware(self):
        assert now_utc().tzinfo is not None

    def test_monotone_in_expiry_date(self):
        now = datetime(2027, 7, 15, 12, 0, tzinfo=timezone.utc)
        ts = [years_to_expiry((date(2027, 7, 15) + timedelta(days=d)).isoformat(), now=now)
              for d in (0, 1, 7, 30, 90)]
        assert ts == sorted(ts)

    def test_greeks_and_gex_agree_on_time_to_expiry(self):
        """The bug this module exists to prevent: two modules, one answer."""
        from apexflow.analytics.greeks import years_to_expiry as g_yte
        from apexflow.analytics.gex import years_to_expiry as x_yte
        now = datetime(2027, 7, 15, 15, 30, tzinfo=timezone.utc)
        for exp in ("2027-07-15", "2027-07-16", "2027-08-20", "2028-01-21"):
            assert g_yte(exp, now=now) == x_yte(exp, now=now)
