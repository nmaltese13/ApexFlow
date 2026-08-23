"""Robust ATM IV extraction, term structure, and skew.

The central case is the one that motivated the module: a 0DTE chain whose
vendor IVs are numerically meaningless. A correct implementation must not
return one of those numbers with a confident-looking quality flag.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from apexflow.analytics.iv_surface import (
    atm_iv, first_usable_atm_iv, iv_term_structure, risk_reversal_25d,
    IV_MIN, IV_MAX, DISPERSION_POOR,
)

SPOT = 100.0


def expiry_in(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


def chain(strikes, ivs, bids=None, asks=None, oi=500.0):
    n = len(strikes)
    bids = [1.00] * n if bids is None else bids
    asks = [1.10] * n if asks is None else asks
    return pd.DataFrame({
        "strike": strikes, "impliedVolatility": ivs,
        "bid": bids, "ask": asks,
        "openInterest": [oi] * n, "volume": [100.0] * n,
    })


def clean_chain(center=SPOT, iv=0.30, n=9, step=1.0):
    strikes = [center + (i - n // 2) * step for i in range(n)]
    # A gentle smile, as a real chain has.
    ivs = [iv + 0.002 * abs(i - n // 2) for i in range(n)]
    return chain(strikes, ivs), chain(strikes, [v + 0.005 for v in ivs])


class TestCleanChain:
    def test_recovers_the_atm_level(self):
        calls, puts = clean_chain(iv=0.30)
        res = atm_iv(calls, puts, SPOT, expiry=expiry_in(30))
        assert res.iv == pytest.approx(0.30, abs=0.01)
        assert res.quality == "good"
        assert res.fallback_used is False

    def test_reports_sample_size_and_dispersion(self):
        calls, puts = clean_chain()
        res = atm_iv(calls, puts, SPOT, expiry=expiry_in(30))
        assert res.n_samples > 0
        assert res.dispersion < 0.05

    def test_truthiness_follows_having_a_value(self):
        calls, puts = clean_chain()
        assert bool(atm_iv(calls, puts, SPOT, expiry=expiry_in(30))) is True
        assert bool(atm_iv(None, None, SPOT, expiry=expiry_in(30))) is False


class TestNoisyZeroDTEChain:
    """The real failure mode, reproduced from the frozen SPY snapshot."""

    def _noisy(self):
        # Adjacent strikes disagreeing by a factor of four — vendor solver
        # noise on near-worthless contracts, not a smile.
        strikes = [763.0, 764.0, 765.0, 766.0, 767.0, 768.0]
        ivs = [0.088, 0.076, 0.052, 0.029, 0.021, 0.033]
        bids = [2.87, 2.01, 1.11, 0.28, 0.01, 0.01]
        asks = [3.09, 2.19, 1.18, 0.30, 0.02, 0.02]
        return chain(strikes, ivs, bids, asks)

    def test_flagged_poor_not_good(self):
        calls = self._noisy()
        res = atm_iv(calls, None, 765.62, expiry=expiry_in(0))
        assert res.quality == "poor"

    def test_dispersion_is_reported_as_large(self):
        res = atm_iv(self._noisy(), None, 765.62, expiry=expiry_in(0))
        assert res.dispersion > DISPERSION_POOR

    def test_penny_wide_worthless_contracts_are_excluded(self):
        """The 0.01/0.02 quotes carry the silliest IVs and must be dropped."""
        calls = self._noisy()
        res = atm_iv(calls, None, 765.62, expiry=expiry_in(0))
        assert 767.0 not in res.strikes_used
        assert 768.0 not in res.strikes_used


class TestFiltering:
    def test_absurd_iv_values_rejected(self):
        calls = chain([98, 99, 100, 101, 102],
                      [0.30, 900.0, 0.31, 0.0, 0.29])
        res = atm_iv(calls, None, SPOT, expiry=expiry_in(30))
        assert IV_MIN <= res.iv <= IV_MAX
        assert res.iv == pytest.approx(0.30, abs=0.02)

    def test_single_wild_outlier_does_not_move_the_median(self):
        good = [0.30] * 8
        calls = chain(list(range(96, 104)), good)
        base = atm_iv(calls, None, SPOT, expiry=expiry_in(30)).iv

        polluted = chain(list(range(96, 105)), good + [4.5])
        after = atm_iv(polluted, None, SPOT, expiry=expiry_in(30)).iv
        assert after == pytest.approx(base, abs=0.005)

    def test_zero_bid_contracts_dropped_when_alternatives_exist(self):
        calls = chain([99, 100, 101],
                      [0.30, 0.31, 2.00],
                      bids=[1.0, 1.0, 0.0], asks=[1.1, 1.1, 0.05])
        res = atm_iv(calls, None, SPOT, expiry=expiry_in(30))
        assert 101.0 not in res.strikes_used

    def test_falls_back_to_a_wider_window_when_needed(self):
        """Nothing within 5% — must widen rather than return nothing."""
        calls = chain([115.0, 116.0, 117.0], [0.35, 0.36, 0.37])
        res = atm_iv(calls, None, SPOT, expiry=expiry_in(30), width_pct=0.05)
        assert res.iv > 0
        assert res.fallback_used is True
        assert res.quality == "poor"

    def test_nan_rows_ignored(self):
        calls = chain([99, 100, 101], [0.30, float("nan"), 0.31])
        res = atm_iv(calls, None, SPOT, expiry=expiry_in(30))
        assert res.iv == pytest.approx(0.305, abs=0.01)


class TestDegenerate:
    def test_empty_chain_returns_none_quality(self):
        res = atm_iv(None, None, SPOT, expiry=expiry_in(30))
        assert res.iv == 0.0 and res.quality == "none"

    def test_zero_spot_returns_none_quality(self):
        calls, puts = clean_chain()
        assert atm_iv(calls, puts, 0.0, expiry=expiry_in(30)).quality == "none"

    def test_all_junk_returns_none_quality(self):
        calls = chain([99, 100, 101], [0.0, -1.0, 900.0])
        assert atm_iv(calls, None, SPOT, expiry=expiry_in(30)).quality == "none"

    def test_result_is_json_serialisable(self):
        import json
        calls, puts = clean_chain()
        json.dumps(atm_iv(calls, puts, SPOT, expiry=expiry_in(30)).to_dict())


class TestFirstUsable:
    def test_skips_the_unusable_front_expiry(self):
        junk = chain([99, 100, 101], [0.02, 0.90, 0.03],
                     bids=[0.01, 0.01, 0.01], asks=[0.02, 0.02, 0.02])
        good_c, good_p = clean_chain(iv=0.28)
        chains = [(expiry_in(0), junk, None),
                  (expiry_in(7), good_c, good_p)]
        iv, quality, expiry = first_usable_atm_iv(chains, SPOT)
        assert quality in ("good", "fair")
        assert expiry == chains[1][0]
        assert iv == pytest.approx(0.28, abs=0.02)

    def test_falls_back_to_a_poor_estimate_rather_than_nothing(self):
        junk = chain([99, 100, 101], [0.05, 0.80, 0.06])
        iv, quality, _ = first_usable_atm_iv([(expiry_in(0), junk, None)], SPOT)
        assert iv > 0
        assert quality == "poor"

    def test_no_chains_returns_none_quality(self):
        assert first_usable_atm_iv([], SPOT) == (0.0, "none", None)


class TestTermStructure:
    def test_contango_detected(self):
        chains = []
        for i, (dte, iv) in enumerate([(7, 0.25), (30, 0.28), (60, 0.32)]):
            c, p = clean_chain(iv=iv)
            chains.append((expiry_in(dte), c, p))
        ts = iv_term_structure(chains, SPOT)
        assert ts["shape"] == "contango"
        assert ts["slope"] > 0

    def test_backwardation_detected(self):
        chains = []
        for dte, iv in [(7, 0.60), (30, 0.40), (60, 0.32)]:
            c, p = clean_chain(iv=iv)
            chains.append((expiry_in(dte), c, p))
        ts = iv_term_structure(chains, SPOT)
        assert ts["shape"] == "backwardation"
        assert ts["slope"] < 0

    def test_flat_detected(self):
        chains = [(expiry_in(d), *clean_chain(iv=0.30)) for d in (7, 30, 60)]
        assert iv_term_structure(chains, SPOT)["shape"] == "flat"

    def test_poor_quality_points_excluded_from_the_shape(self):
        """A noisy 0DTE row must not set the front of the curve."""
        junk = chain([99, 100, 101], [0.02, 0.95, 0.03],
                     bids=[0.01, 0.01, 0.01], asks=[0.02, 0.02, 0.02])
        chains = [(expiry_in(0), junk, None)]
        chains += [(expiry_in(d), *clean_chain(iv=iv)) for d, iv in [(7, 0.25), (30, 0.30)]]
        ts = iv_term_structure(chains, SPOT)
        assert ts["n_usable"] == 2
        assert ts["front_iv"] == pytest.approx(0.25, abs=0.02)
        assert ts["shape"] == "contango"

    def test_points_sorted_by_tenor(self):
        chains = [(expiry_in(d), *clean_chain(iv=0.30)) for d in (60, 7, 30)]
        dtes = [p["dte"] for p in iv_term_structure(chains, SPOT)["points"]]
        assert dtes == sorted(dtes)

    def test_empty_input_is_unknown(self):
        assert iv_term_structure([], SPOT)["shape"] == "unknown"


class TestRiskReversal:
    def _skewed(self, put_extra=0.10):
        """Puts bid over calls — the normal equity index shape."""
        strikes = np.arange(70.0, 131.0, 2.5)
        call_iv = 0.30 + 0.0005 * (strikes - SPOT)
        put_iv = call_iv + put_extra
        return (chain(strikes, call_iv, bids=[1.0] * len(strikes), asks=[1.1] * len(strikes)),
                chain(strikes, put_iv, bids=[1.0] * len(strikes), asks=[1.1] * len(strikes)))

    def test_put_skew_gives_positive_risk_reversal(self):
        calls, puts = self._skewed(put_extra=0.10)
        rr = risk_reversal_25d(calls, puts, SPOT, 0.25)
        assert rr["quality"] == "good"
        assert rr["rr_25d"] > 0

    def test_call_skew_gives_negative_risk_reversal(self):
        calls, puts = self._skewed(put_extra=-0.08)
        assert risk_reversal_25d(calls, puts, SPOT, 0.25)["rr_25d"] < 0

    def test_selected_strikes_straddle_spot(self):
        calls, puts = self._skewed()
        rr = risk_reversal_25d(calls, puts, SPOT, 0.25)
        assert rr["call_strike"] > SPOT > rr["put_strike"]

    def test_reported_in_vol_points(self):
        calls, puts = self._skewed(put_extra=0.10)
        rr = risk_reversal_25d(calls, puts, SPOT, 0.25)
        assert 5.0 < rr["rr_25d"] < 15.0     # ~10 vol points

    def test_missing_wing_returns_none_quality(self):
        calls, _ = self._skewed()
        assert risk_reversal_25d(calls, None, SPOT, 0.25)["quality"] == "none"

    def test_invalid_inputs_return_none_quality(self):
        calls, puts = self._skewed()
        assert risk_reversal_25d(calls, puts, 0.0, 0.25)["quality"] == "none"
        assert risk_reversal_25d(calls, puts, SPOT, 0.0)["quality"] == "none"
