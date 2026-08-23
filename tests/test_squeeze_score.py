"""Composite squeeze scorer — anchors, monotonicity, and the cliff it removes.

The scoring curves are judgement, not a fitted model, so there is no ground
truth to check against. What *can* be pinned is the shape: the score must be
monotone in each input, bounded, equal to the old step-function values at
every anchor point (so previously logged signals stay comparable), and free
of the discontinuities that made a rounding difference in a vendor feed move
a name several places up the leaderboard.
"""
from __future__ import annotations

import pytest

from apexflow.analytics.squeeze import (
    SqueezeInputs, squeeze_score, interpolate, COMPONENT_MAX,
)


class TestInterpolate:
    def test_returns_anchor_values_exactly(self):
        bp = [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
        sc = [0, 5, 15, 25, 30, 35]
        for b, s in zip(bp, sc):
            assert interpolate(b, bp, sc) == pytest.approx(s)

    def test_midpoint_is_halfway_between_anchors(self):
        assert interpolate(0.15, [0.10, 0.20], [10, 20]) == pytest.approx(15.0)

    def test_below_first_anchor_clamps(self):
        assert interpolate(0.001, [0.05, 0.10], [0, 5]) == 0.0

    def test_above_last_anchor_clamps(self):
        assert interpolate(999.0, [1.0, 2.0], [10, 20]) == 20.0

    def test_descending_scores_interpolate_downhill(self):
        """Float-size scoring runs downhill; interpolation must follow."""
        assert interpolate(30.0, [10, 50], [10, 7]) == pytest.approx(8.5)

    def test_none_and_nan_are_treated_as_the_floor(self):
        assert interpolate(None, [1, 2], [5, 10]) == 5.0
        assert interpolate(float("nan"), [1, 2], [5, 10]) == 5.0


class TestNoCliff:
    def test_short_float_has_no_jump_at_the_ten_percent_boundary(self):
        """The exact bug the rewrite fixes: 9.9% vs 10.1% short float."""
        lo = squeeze_score(SqueezeInputs(short_pct_float=0.099))[0]
        hi = squeeze_score(SqueezeInputs(short_pct_float=0.101))[0]
        assert abs(hi - lo) < 0.5      # was a 10-point step

    @pytest.mark.parametrize("field,boundary", [
        ("short_pct_float", 0.20),
        ("days_to_cover", 3.0),
        ("borrow_rate", 0.50),
        ("iv_hv_ratio", 1.30),
        ("rvol", 2.50),
    ])
    def test_every_axis_is_continuous_at_its_anchors(self, field, boundary):
        eps = boundary * 1e-3
        lo = squeeze_score(SqueezeInputs(**{field: boundary - eps}))[0]
        hi = squeeze_score(SqueezeInputs(**{field: boundary + eps}))[0]
        assert abs(hi - lo) < 0.5

    def test_score_moves_smoothly_across_the_whole_short_float_range(self):
        """Sweep at 0.1% granularity; the response must be gradual throughout.

        The steepest stretch is 0.05->0.10 short float, worth 5 points over
        5 percentage points, so 0.1 point per 0.1% is the expected slope.
        Anything markedly larger would be a surviving discontinuity.
        """
        vals = [squeeze_score(SqueezeInputs(short_pct_float=x / 1000))[0]
                for x in range(0, 1001)]
        jumps = [abs(b - a) for a, b in zip(vals, vals[1:])]
        assert max(jumps) < 0.2


class TestMonotonicity:
    @pytest.mark.parametrize("field,values", [
        ("short_pct_float", [0.0, 0.05, 0.15, 0.25, 0.40, 0.60]),
        ("days_to_cover", [0.0, 1.0, 2.5, 4.0, 6.0, 10.0]),
        ("borrow_rate", [0.0, 0.10, 0.30, 0.80, 2.0, 6.0]),
        ("iv_hv_ratio", [0.5, 0.9, 1.1, 1.4, 1.8, 2.5]),
        ("rvol", [0.5, 1.2, 2.0, 3.0, 5.0, 8.0]),
        ("accumulation_score", [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0]),
    ])
    def test_more_pressure_never_lowers_the_score(self, field, values):
        scores = [squeeze_score(SqueezeInputs(**{field: v}))[0] for v in values]
        assert scores == sorted(scores)

    def test_smaller_float_scores_higher(self):
        small = squeeze_score(SqueezeInputs(float_shares=5_000_000))[0]
        large = squeeze_score(SqueezeInputs(float_shares=900_000_000))[0]
        assert small > large

    def test_missing_float_scores_no_points(self):
        _, comp = squeeze_score(SqueezeInputs(float_shares=0))
        assert comp["float_size"] == 0.0


class TestBounds:
    def test_zero_pressure_inputs_score_zero(self):
        score, comp = squeeze_score(SqueezeInputs(
            short_pct_float=0.0, days_to_cover=0.0, borrow_rate=0.0,
            iv_hv_ratio=0.0, rvol=0.0, float_shares=0.0,
            accumulation_score=0.0))
        assert score == 0.0
        assert all(v == 0.0 for v in comp.values())

    def test_defaults_are_not_zero_pressure(self):
        """The dataclass defaults are a *neutral* name, not a blank one.

        ``iv_hv_ratio`` and ``rvol`` both default to 1.0, which is the
        anchor meaning "IV equals realised vol" and "volume is normal" —
        the IV/HV curve scores 1 point there by design. Worth pinning so
        nobody later reads a baseline score of 1 as a bug.
        """
        score, comp = squeeze_score(SqueezeInputs())
        assert score == pytest.approx(1.0)
        assert comp["iv_hv"] == pytest.approx(1.0)
        assert comp["rvol"] == 0.0

    def test_maximum_inputs_cap_at_one_hundred(self):
        score, _ = squeeze_score(SqueezeInputs(
            short_pct_float=0.95, days_to_cover=50, borrow_rate=8.0,
            iv_hv_ratio=5.0, rvol=20.0, float_shares=1_000_000,
            accumulation_score=1.0))
        assert score == 100.0

    def test_score_always_in_range(self):
        import random
        rng = random.Random(0)
        for _ in range(500):
            s, _ = squeeze_score(SqueezeInputs(
                short_pct_float=rng.uniform(-0.1, 1.5),
                days_to_cover=rng.uniform(-1, 60),
                borrow_rate=rng.uniform(-0.1, 10),
                iv_hv_ratio=rng.uniform(0, 6),
                rvol=rng.uniform(0, 30),
                float_shares=rng.uniform(0, 3e9),
                accumulation_score=rng.uniform(-2, 2)))
            assert 0.0 <= s <= 100.0

    def test_negative_accumulation_does_not_subtract(self):
        """Distribution is neutral here, not a penalty."""
        _, comp = squeeze_score(SqueezeInputs(accumulation_score=-1.0))
        assert comp["accumulation"] == 0.0

    def test_accumulation_above_one_is_clamped(self):
        _, comp = squeeze_score(SqueezeInputs(accumulation_score=99.0))
        assert comp["accumulation"] == COMPONENT_MAX["accumulation"]


class TestComponents:
    def test_components_are_returned_and_sum_to_the_score(self):
        inp = SqueezeInputs(short_pct_float=0.25, days_to_cover=4.0,
                            borrow_rate=0.30, iv_hv_ratio=1.5, rvol=3.0,
                            float_shares=40_000_000, accumulation_score=0.5)
        score, comp = squeeze_score(inp)
        assert set(comp) == set(COMPONENT_MAX)
        assert score == pytest.approx(min(sum(comp.values()), 100.0))

    def test_no_component_exceeds_its_declared_maximum(self):
        score, comp = squeeze_score(SqueezeInputs(
            short_pct_float=5.0, days_to_cover=500, borrow_rate=50,
            iv_hv_ratio=50, rvol=200, float_shares=1, accumulation_score=1.0))
        for name, value in comp.items():
            assert value <= COMPONENT_MAX[name] + 1e-9

    def test_short_float_is_the_heaviest_axis(self):
        assert COMPONENT_MAX["short_float"] == max(COMPONENT_MAX.values())
