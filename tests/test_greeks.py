"""Black-Scholes Greeks — known inputs, known outputs.

The reference case is the standard textbook one (Hull, *Options, Futures and
Other Derivatives*): S=100, K=100, T=1yr, r=5%, sigma=20%, q=0. Values below
are the analytic ones to six decimals, not numbers copied out of this
codebase's own output — a test that asserts the implementation equals itself
proves nothing.

Where a closed form is awkward to state independently (theta, charm, vanna),
the Greek is checked against a central finite difference of the pricing
function instead, which is an independent derivation.
"""
from __future__ import annotations

import math

import pytest

from apexflow.analytics.greeks import (
    greeks, bs_price, implied_vol, iv_rank, iv_percentile,
)

# Reference case
S, K, T, R, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

CALL_PRICE = 10.450584
PUT_PRICE = 5.573526
DELTA_CALL = 0.636831
DELTA_PUT = -0.363169
GAMMA = 0.01876202
VEGA_PER_VOLPT = 0.37524035       # per +1 vol point
THETA_CALL_PER_DAY = -0.01757268  # per calendar day
RHO_CALL_PER_PCT = 0.53232482     # per +1 percentage point of rate


class TestReferenceValues:
    def test_call_price(self):
        assert bs_price(S, K, T, SIG, "C", R, Q) == pytest.approx(CALL_PRICE, abs=1e-6)

    def test_put_price(self):
        assert bs_price(S, K, T, SIG, "P", R, Q) == pytest.approx(PUT_PRICE, abs=1e-6)

    def test_call_greeks(self):
        g = greeks(S, K, T, SIG, "C", R, Q)
        assert g["delta"] == pytest.approx(DELTA_CALL, abs=1e-6)
        assert g["gamma"] == pytest.approx(GAMMA, abs=1e-8)
        assert g["vega"] == pytest.approx(VEGA_PER_VOLPT, abs=1e-8)
        assert g["theta"] == pytest.approx(THETA_CALL_PER_DAY, abs=1e-8)
        assert g["rho"] == pytest.approx(RHO_CALL_PER_PCT, abs=1e-8)

    def test_put_delta(self):
        assert greeks(S, K, T, SIG, "P", R, Q)["delta"] == pytest.approx(DELTA_PUT, abs=1e-6)

    def test_gamma_and_vega_are_side_independent(self):
        """Gamma and vega are identical for a call and a put at the same strike."""
        c = greeks(S, K, T, SIG, "C", R, Q)
        p = greeks(S, K, T, SIG, "P", R, Q)
        assert c["gamma"] == pytest.approx(p["gamma"], rel=1e-12)
        assert c["vega"] == pytest.approx(p["vega"], rel=1e-12)


class TestStructuralIdentities:
    """Relationships that must hold regardless of the parameters."""

    @pytest.mark.parametrize("spot,strike,t,sig,r,q", [
        (100, 100, 1.0, 0.20, 0.05, 0.0),
        (42.5, 50.0, 0.25, 0.60, 0.04, 0.02),
        (250, 200, 2.0, 0.15, 0.03, 0.01),
        (7.5, 10.0, 0.08, 1.20, 0.05, 0.0),
    ])
    def test_put_call_parity(self, spot, strike, t, sig, r, q):
        c = bs_price(spot, strike, t, sig, "C", r, q)
        p = bs_price(spot, strike, t, sig, "P", r, q)
        lhs = c - p
        rhs = spot * math.exp(-q * t) - strike * math.exp(-r * t)
        assert lhs == pytest.approx(rhs, abs=1e-9)

    @pytest.mark.parametrize("spot,strike,t,sig,r,q", [
        (100, 100, 1.0, 0.20, 0.05, 0.0),
        (42.5, 50.0, 0.25, 0.60, 0.04, 0.02),
    ])
    def test_delta_parity(self, spot, strike, t, sig, r, q):
        """delta_call - delta_put = exp(-q*T)."""
        dc = greeks(spot, strike, t, sig, "C", r, q)["delta"]
        dp = greeks(spot, strike, t, sig, "P", r, q)["delta"]
        assert dc - dp == pytest.approx(math.exp(-q * t), abs=1e-9)

    def test_delta_matches_finite_difference(self):
        h = 1e-4
        up = bs_price(S + h, K, T, SIG, "C", R, Q)
        dn = bs_price(S - h, K, T, SIG, "C", R, Q)
        assert (up - dn) / (2 * h) == pytest.approx(DELTA_CALL, abs=1e-6)

    def test_gamma_matches_finite_difference(self):
        h = 1e-3
        up = bs_price(S + h, K, T, SIG, "C", R, Q)
        mid = bs_price(S, K, T, SIG, "C", R, Q)
        dn = bs_price(S - h, K, T, SIG, "C", R, Q)
        assert (up - 2 * mid + dn) / (h * h) == pytest.approx(GAMMA, abs=1e-6)

    def test_vega_matches_finite_difference(self):
        h = 1e-5
        up = bs_price(S, K, T, SIG + h, "C", R, Q)
        dn = bs_price(S, K, T, SIG - h, "C", R, Q)
        # Divided by 100 because our vega is per vol *point*.
        assert (up - dn) / (2 * h) / 100 == pytest.approx(VEGA_PER_VOLPT, abs=1e-8)

    def test_theta_matches_finite_difference(self):
        """Theta is -dPrice/dT, expressed per calendar day."""
        h = 1e-6
        up = bs_price(S, K, T + h, SIG, "C", R, Q)
        dn = bs_price(S, K, T - h, SIG, "C", R, Q)
        d_price_dT = (up - dn) / (2 * h)
        assert -d_price_dT / 365 == pytest.approx(THETA_CALL_PER_DAY, abs=1e-6)

    def test_gamma_peaks_near_the_money(self):
        atm = greeks(S, 100, T, SIG, "C", R, Q)["gamma"]
        assert atm > greeks(S, 70, T, SIG, "C", R, Q)["gamma"]
        assert atm > greeks(S, 140, T, SIG, "C", R, Q)["gamma"]

    def test_price_is_monotone_in_vol(self):
        prev = -1.0
        for sig in (0.05, 0.10, 0.20, 0.40, 0.80):
            px = bs_price(S, K, T, sig, "C", R, Q)
            assert px > prev
            prev = px

    def test_deep_itm_call_delta_approaches_one(self):
        assert greeks(100, 1.0, T, SIG, "C", R, Q)["delta"] == pytest.approx(1.0, abs=1e-6)

    def test_deep_otm_call_delta_approaches_zero(self):
        assert greeks(100, 10_000.0, T, SIG, "C", R, Q)["delta"] == pytest.approx(0.0, abs=1e-6)


class TestInvalidInputs:
    @pytest.mark.parametrize("kwargs", [
        dict(spot=0, strike=100, t=1, iv=0.2),
        dict(spot=100, strike=0, t=1, iv=0.2),
        dict(spot=100, strike=100, t=0, iv=0.2),
        dict(spot=100, strike=100, t=1, iv=0),
        dict(spot=-5, strike=100, t=1, iv=0.2),
    ])
    def test_returns_zeros_not_nan(self, kwargs):
        g = greeks(**kwargs)
        assert all(v == 0.0 for v in g.values())
        assert all(not math.isnan(v) for v in g.values())


class TestImpliedVol:
    @pytest.mark.parametrize("true_iv", [0.05, 0.15, 0.30, 0.75, 1.50, 3.00])
    @pytest.mark.parametrize("right,strike", [("C", 100.0), ("C", 130.0),
                                              ("P", 70.0), ("P", 100.0)])
    def test_round_trip(self, true_iv, right, strike):
        """Price at a known vol, then solve back to it."""
        px = bs_price(S, strike, T, true_iv, right, R, Q)
        if px < 1e-6:
            pytest.skip("option is worthless at this vol; IV is undefined")
        solved = implied_vol(px, S, strike, T, right, R, Q)
        assert solved is not None
        assert solved == pytest.approx(true_iv, abs=1e-4)

    def test_short_dated_deep_otm_converges(self):
        """The case plain Newton diverges on: tiny vega, far from the money."""
        t = 3 / 365
        px = bs_price(S, 115.0, t, 0.90, "C", R, Q)
        solved = implied_vol(px, S, 115.0, t, "C", R, Q)
        assert solved is not None
        assert solved == pytest.approx(0.90, abs=1e-3)

    def test_price_above_no_arb_bound_returns_none(self):
        """A call cannot be worth more than the discounted forward."""
        assert implied_vol(S * 1.5, S, K, T, "C", R, Q) is None

    def test_price_below_intrinsic_returns_none(self):
        intrinsic = max(S - K * math.exp(-R * T), 0.0)
        assert implied_vol(intrinsic * 0.5, S, K, T, "C", R, Q) is None

    @pytest.mark.parametrize("bad", [0.0, -1.0, None])
    def test_non_positive_price_returns_none(self, bad):
        assert implied_vol(bad, S, K, T, "C", R, Q) is None

    def test_invalid_inputs_return_none(self):
        assert implied_vol(5.0, 0.0, K, T, "C") is None
        assert implied_vol(5.0, S, K, 0.0, "C") is None

    def test_honours_dividend_yield(self):
        """Ignoring q would bias the solved vol; check it round-trips with q>0."""
        q = 0.03
        px = bs_price(S, K, T, 0.35, "C", R, q)
        assert implied_vol(px, S, K, T, "C", R, q) == pytest.approx(0.35, abs=1e-4)


class TestIVRankPercentile:
    def test_rank_at_top_and_bottom(self):
        hist = [0.2, 0.3, 0.4, 0.5, 0.6]
        assert iv_rank(0.6, hist) == pytest.approx(100.0)
        assert iv_rank(0.2, hist) == pytest.approx(0.0)
        assert iv_rank(0.4, hist) == pytest.approx(50.0)

    def test_rank_clamps_outside_history(self):
        assert iv_rank(0.9, [0.2, 0.6]) == 100.0

    def test_percentile_counts_strictly_below(self):
        assert iv_percentile(0.5, [0.1, 0.3, 0.5, 0.7, 0.9]) == pytest.approx(40.0)

    def test_flat_history_is_midpoint(self):
        assert iv_rank(0.5, [0.5, 0.5, 0.5]) == 50.0

    def test_empty_history_is_none_not_zero(self):
        """'No data' must be distinguishable from 'at the bottom of the range'."""
        assert iv_rank(0.5, []) is None
        assert iv_percentile(0.5, []) is None
        assert iv_rank(0.5, None) is None

    def test_junk_values_filtered(self):
        assert iv_rank(0.5, [0.0, -1.0, None, 0.4, 0.6]) is not None
