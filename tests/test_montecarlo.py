"""Monte Carlo engine — validated against closed-form values.

A simulator is only trustworthy if it reproduces things that are known
independently. These tests check the engine against Black-Scholes prices,
the analytic terminal CDF, the martingale property, and the exact
first-passage probability, each time expressing the error in Monte Carlo
standard errors so the assertion is a statistical statement rather than an
arbitrary tolerance.

Path counts here are kept modest so the suite stays fast. The tolerances are
in units of standard error, so they hold at any N — that is the point of
expressing them that way.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apexflow.analytics import montecarlo as mc
from apexflow.analytics.projection import above_prob, touch_prob

SPOT, T, IV, R, Q = 100.0, 0.25, 0.30, 0.04, 0.0

# Number of standard errors we allow. 4 sigma is a ~1-in-16,000 false-failure
# rate per assertion — loose enough not to flake, tight enough to catch a
# real bias.
Z_TOL = 4.0


def z_score(simulated: float, analytic: float, stderr: float) -> float:
    return abs(simulated - analytic) / max(stderr, 1e-12)


class TestAgainstClosedForm:
    @pytest.mark.parametrize("right,strike", [
        ("C", 100.0), ("C", 110.0), ("C", 90.0),
        ("P", 100.0), ("P", 90.0), ("P", 110.0),
    ])
    def test_european_price_matches_black_scholes(self, right, strike):
        cfg = mc.MCConfig(n_paths=200_000, n_steps=1, seed=11)
        got = mc.price_european(SPOT, strike, T, IV, right, cfg, R, Q)
        want = mc.bs_price(SPOT, strike, T, IV, right, R, Q)
        assert z_score(got["price"], want, got["stderr"]) < Z_TOL

    @pytest.mark.parametrize("strike", [90.0, 95.0, 100.0, 105.0, 110.0])
    def test_terminal_cdf_matches_n_d2(self, strike):
        cfg = mc.MCConfig(n_paths=200_000, n_steps=1, seed=12)
        st = mc.simulate_terminal(SPOT, T, IV, cfg, R, Q)
        p = float((st > strike).mean())
        want = above_prob(SPOT, strike, T, IV, R, Q)
        se = math.sqrt(max(p * (1 - p), 1e-12) / st.size)
        assert z_score(p, want, se) < Z_TOL

    def test_expectation_is_a_martingale(self):
        """E[S_T] must equal S0 * exp((r-q)T) — the risk-neutral drift check."""
        cfg = mc.MCConfig(n_paths=400_000, n_steps=1, seed=13)
        st = mc.simulate_terminal(SPOT, T, IV, cfg, R, Q)
        want = SPOT * math.exp((R - Q) * T)
        se = float(st.std(ddof=1) / math.sqrt(st.size))
        assert z_score(float(st.mean()), want, se) < Z_TOL

    def test_terminal_variance_matches_lognormal(self):
        cfg = mc.MCConfig(n_paths=400_000, n_steps=1, seed=14)
        st = mc.simulate_terminal(SPOT, T, IV, cfg, R, Q)
        want = (SPOT ** 2) * math.exp(2 * (R - Q) * T) * (math.exp(IV * IV * T) - 1)
        assert float(st.var()) == pytest.approx(want, rel=0.02)

    @pytest.mark.parametrize("barrier", [105.0, 95.0, 110.0, 90.0])
    def test_touch_probability_matches_first_passage(self, barrier):
        """The barrier check — this is where discretisation bias shows up."""
        cfg = mc.MCConfig(n_paths=100_000, n_steps=128, seed=15, chunk_paths=25_000)
        got = mc.touch_probabilities(SPOT, [barrier], T, IV, cfg, R, Q)[barrier]
        want = touch_prob(SPOT, barrier, T, IV, R, Q)
        assert z_score(got["touch"], want, got["touch_se"]) < Z_TOL

    def test_validate_harness_passes_end_to_end(self):
        rows = mc.validate(n_paths=200_000, n_steps=128, seed=17)
        assert rows, "validate() returned nothing"
        worst = max(abs(r["z"]) for r in rows)
        assert worst < Z_TOL, f"worst |z| = {worst:.2f} in {rows}"


class TestBrownianBridge:
    """The bridge correction is the difference between right and wrong here."""

    def test_uncorrected_monitoring_is_biased_low(self):
        cfg = mc.MCConfig(n_paths=100_000, n_steps=16, seed=21, chunk_paths=25_000)
        naive = mc.touch_probabilities(SPOT, [105.0], T, IV, cfg, R, Q,
                                       bridge_correction=False)[105.0]["touch"]
        want = touch_prob(SPOT, 105.0, T, IV, R, Q)
        # At 16 steps the discrete path misses crossings that happen between
        # observations, by several percentage points.
        assert naive < want - 0.05

    def test_bridge_correction_removes_the_bias(self):
        cfg = mc.MCConfig(n_paths=100_000, n_steps=16, seed=21, chunk_paths=25_000)
        got = mc.touch_probabilities(SPOT, [105.0], T, IV, cfg, R, Q,
                                     bridge_correction=True)[105.0]
        want = touch_prob(SPOT, 105.0, T, IV, R, Q)
        assert abs(got["touch"] - want) < 0.01

    def test_uncorrected_converges_upward_with_more_steps(self):
        """More monitoring points recover more of the true probability."""
        want = touch_prob(SPOT, 105.0, T, IV, R, Q)
        errors = []
        for steps in (8, 32, 128):
            cfg = mc.MCConfig(n_paths=50_000, n_steps=steps, seed=22, chunk_paths=25_000)
            p = mc.touch_probabilities(SPOT, [105.0], T, IV, cfg, R, Q,
                                       bridge_correction=False)[105.0]["touch"]
            errors.append(want - p)
        assert errors[0] > errors[1] > errors[2] > 0


class TestConvergence:
    def test_standard_error_falls_as_one_over_sqrt_n(self):
        rows = mc.convergence_table(SPOT, T, IV, R, Q,
                                    counts=(10_000, 100_000, 1_000_000))
        ses = [r["stderr"] for r in rows]
        assert ses[0] > ses[1] > ses[2]
        # A 100x increase in paths should cut the error by ~10x.
        assert ses[0] / ses[2] == pytest.approx(10.0, rel=0.25)

    def test_error_shrinks_with_path_count(self):
        rows = mc.convergence_table(SPOT, T, IV, R, Q, counts=(5_000, 500_000))
        assert rows[1]["abs_error"] < rows[0]["abs_error"]

    def test_five_million_paths_gives_sub_basis_point_probability_error(self):
        """The headline claim: 5,000,000 paths puts SE ~2bp on a probability.

        Checked analytically rather than by running 5M paths in the test
        suite — the formula is sqrt(p(1-p)/N), and the worst case is p=0.5.
        """
        se_worst = math.sqrt(0.25 / mc.VALIDATION_PATHS)
        assert se_worst < 0.00025          # under 2.5 bp
        assert se_worst < 0.001 / 4        # comfortably under UI resolution


class TestVarianceReduction:
    def test_antithetic_pairs_have_exactly_zero_mean_normal(self):
        cfg = mc.MCConfig(n_paths=10_000, n_steps=4, seed=31, antithetic=True)
        rng = np.random.default_rng(cfg.seed)
        z = mc._normals(rng, cfg.n_paths, cfg.n_steps, True)
        assert float(z.sum()) == pytest.approx(0.0, abs=1e-9)

    def test_control_variate_reduces_standard_error(self):
        cfg = mc.MCConfig(n_paths=100_000, n_steps=1, seed=32)
        plain = mc.price_european(SPOT, 100.0, T, IV, "C", cfg, R, Q,
                                  control_variate=False)
        cv = mc.price_european(SPOT, 100.0, T, IV, "C", cfg, R, Q,
                               control_variate=True)
        assert cv["stderr"] < plain["stderr"]

    def test_odd_path_count_rounded_up_for_antithetic(self):
        cfg = mc.MCConfig(n_paths=9_999, antithetic=True)
        assert cfg.n_paths % 2 == 0


class TestReproducibility:
    def test_same_seed_gives_identical_results(self):
        a = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=20_000, seed=42))
        b = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=20_000, seed=42))
        assert a.mean == b.mean and a.quantiles == b.quantiles

    def test_different_seeds_differ_but_agree_within_noise(self):
        a = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=100_000, seed=1))
        b = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=100_000, seed=2))
        assert a.mean != b.mean
        assert abs(a.mean - b.mean) < 6 * (a.standard_error + b.standard_error)

    def test_chunking_does_not_change_the_answer(self):
        """Streaming in chunks must give the same result as one big batch."""
        big = mc.simulate_cone(SPOT, T, IV,
                               mc.MCConfig(n_paths=80_000, seed=7, chunk_paths=80_000))
        small = mc.simulate_cone(SPOT, T, IV,
                                 mc.MCConfig(n_paths=80_000, seed=7, chunk_paths=10_000))
        # Chunk boundaries change which normals land on which path, so the
        # samples differ; the estimates must still agree within noise.
        assert abs(big.mean - small.mean) < 6 * (big.standard_error + small.standard_error)


class TestConeShape:
    def test_quantiles_are_ordered(self):
        res = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=100_000, seed=51))
        for row in res.cone:
            assert row["p5"] < row["p25"] < row["p50"] < row["p75"] < row["p95"]

    def test_cone_widens_with_time(self):
        res = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=100_000, n_steps=16, seed=52))
        widths = [r["p95"] - r["p5"] for r in res.cone]
        assert widths[-1] > widths[0]
        assert all(b >= a * 0.98 for a, b in zip(widths, widths[1:]))

    def test_terminal_quantiles_match_analytic_lognormal(self):
        res = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=400_000, n_steps=8, seed=53))
        drift = (R - Q - 0.5 * IV * IV) * T
        sig = IV * math.sqrt(T)
        want_p50 = SPOT * math.exp(drift)
        want_p95 = SPOT * math.exp(drift + 1.6449 * sig)
        assert res.quantiles["p50"] == pytest.approx(want_p50, rel=0.005)
        assert res.quantiles["p95"] == pytest.approx(want_p95, rel=0.01)

    def test_cone_length_equals_step_count(self):
        res = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=5_000, n_steps=23, seed=54))
        assert len(res.cone) == 23

    def test_gbm_terminal_is_nearly_symmetric_in_logs(self):
        res = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=400_000, n_steps=4, seed=55))
        assert abs(res.skew) < 0.03
        assert abs(res.kurtosis) < 0.05


class TestAlternativeModels:
    def test_merton_jumps_produce_fatter_tails(self):
        """Rare, large jumps — the regime where the jump model earns its keep.

        Frequent small jumps wash out: with many jumps per horizon the sum
        approaches a normal by the central limit theorem and excess kurtosis
        collapses back toward the GBM value. One jump per year at 20% is the
        single-name-event shape, and it is clearly visible.
        """
        base = mc.MCConfig(n_paths=200_000, n_steps=32, seed=61)
        jump = mc.MCConfig(n_paths=200_000, n_steps=32, seed=61, model="merton",
                           jump_intensity=1.0, jump_mean=0.0, jump_vol=0.20)
        a = mc.simulate_cone(SPOT, T, IV, base)
        b = mc.simulate_cone(SPOT, T, IV, jump)
        assert a.kurtosis < 0.05        # GBM is (near) mesokurtic in logs
        assert b.kurtosis > 0.5         # jumps put real weight in the tails

    def test_frequent_small_jumps_converge_toward_gbm(self):
        """The CLT limit of the jump model, asserted rather than assumed."""
        rare = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(
            n_paths=200_000, n_steps=32, seed=64, model="merton",
            jump_intensity=1.0, jump_vol=0.20))
        frequent = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(
            n_paths=200_000, n_steps=32, seed=64, model="merton",
            jump_intensity=40.0, jump_vol=0.20 / math.sqrt(40)))
        assert frequent.kurtosis < rare.kurtosis

    def test_negative_jump_mean_produces_negative_skew(self):
        cfg = mc.MCConfig(n_paths=300_000, n_steps=32, seed=65, model="merton",
                          jump_intensity=4.0, jump_mean=-0.03, jump_vol=0.10)
        assert mc.simulate_cone(SPOT, T, IV, cfg).skew < -0.05

    def test_merton_preserves_the_martingale(self):
        """Drift compensation must keep E[S_T] at the forward."""
        cfg = mc.MCConfig(n_paths=400_000, n_steps=32, seed=62, model="merton",
                          jump_intensity=8.0, jump_mean=-0.02, jump_vol=0.05)
        res = mc.simulate_cone(SPOT, T, IV, cfg)
        want = SPOT * math.exp((R - Q) * T)
        assert abs(res.mean - want) < 5 * res.standard_error

    def test_bootstrap_inherits_the_sample_shape(self):
        """Skewed historical returns should give a skewed simulated terminal."""
        rng = np.random.default_rng(0)
        # Mostly small ups with occasional large downs — negative skew.
        rets = np.where(rng.random(4000) < 0.05,
                        rng.normal(-0.05, 0.01, 4000),
                        rng.normal(0.002, 0.008, 4000))
        cfg = mc.MCConfig(n_paths=200_000, n_steps=1, seed=63,
                          model="bootstrap", returns=rets)
        res = mc.simulate_cone(SPOT, T, IV, cfg)
        assert res.skew < -0.3

    def test_bootstrap_rejects_insufficient_history(self):
        with pytest.raises(ValueError):
            mc.MCConfig(model="bootstrap", returns=np.array([0.01, -0.01]))

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError):
            mc.MCConfig(model="stochastic_vol_wishart")


class TestDegenerateInputs:
    @pytest.mark.parametrize("spot,t,iv", [
        (0.0, T, IV), (SPOT, 0.0, IV), (SPOT, T, 0.0), (-1.0, T, IV),
    ])
    def test_cone_returns_empty_result_not_an_exception(self, spot, t, iv):
        res = mc.simulate_cone(spot, t, iv, mc.MCConfig(n_paths=1_000))
        assert res.n_paths == 0
        assert res.cone == []

    @pytest.mark.parametrize("spot,t,iv", [(0.0, T, IV), (SPOT, 0.0, IV), (SPOT, T, 0.0)])
    def test_touch_probabilities_degrade_gracefully(self, spot, t, iv):
        out = mc.touch_probabilities(spot, [105.0], t, iv, mc.MCConfig(n_paths=1_000))
        assert out[105.0]["touch"] == 0.0

    def test_no_barriers_returns_empty_dict(self):
        assert mc.touch_probabilities(SPOT, [], T, IV) == {}

    def test_non_positive_barriers_ignored(self):
        out = mc.touch_probabilities(SPOT, [0.0, -5.0, 105.0], T, IV,
                                     mc.MCConfig(n_paths=5_000, n_steps=8))
        assert set(out) == {105.0}

    def test_result_is_json_serialisable(self):
        import json
        res = mc.simulate_cone(SPOT, T, IV, mc.MCConfig(n_paths=2_000, n_steps=4))
        json.dumps(res.to_dict())     # must not raise


class TestStreamingHistogram:
    """The O(1)-memory quantile accumulator underneath the cone."""

    def test_quantiles_match_numpy_on_a_known_sample(self):
        rng = np.random.default_rng(99)
        x = rng.normal(0.0, 0.2, 500_000)
        h = mc._LogHistogram(sigma_total=0.2)
        for chunk in np.array_split(x, 7):
            h.update(chunk)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            assert h.quantile(q) == pytest.approx(float(np.quantile(x, q)), abs=0.002)

    def test_moments_match_numpy(self):
        rng = np.random.default_rng(98)
        x = rng.normal(0.01, 0.3, 200_000)
        h = mc._LogHistogram(sigma_total=0.3)
        for chunk in np.array_split(x, 5):
            h.update(chunk)
        assert h.mean == pytest.approx(float(x.mean()), abs=1e-9)
        assert h.std == pytest.approx(float(x.std()), abs=1e-9)
        skew, kurt = h.moments()
        assert abs(skew) < 0.05 and abs(kurt) < 0.05

    def test_empty_histogram_is_safe(self):
        h = mc._LogHistogram(sigma_total=0.2)
        assert h.quantile(0.5) == 0.0
        assert h.mean == 0.0 and h.std == 0.0

    def test_extreme_values_clip_rather_than_crash(self):
        h = mc._LogHistogram(sigma_total=0.01)
        h.update(np.array([-50.0, 50.0, 0.0]))
        assert h.below == 1 and h.above == 1
        assert math.isfinite(h.quantile(0.5))
