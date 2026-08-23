"""GEX and the wider dealer-exposure engine — sign conventions and scaling.

The point of these tests is that GEX has no external reference value to
check against: it is a convention, and the only things that can go wrong are
the sign, the scale, and the aggregation. So they pin exactly those:

  * calls contribute positive gamma and puts negative, under the naive
    dealer assumption, and flipping the convention flips the sign;
  * the magnitude equals the hand-computed ``gamma * OI * 100 * S^2 * 0.01``
    for a one-contract chain, so a lost factor of 100 or a missing S^2 is
    caught;
  * ``gex.chain_gex`` and ``dealer_greeks.chain_exposures`` agree, since one
    now delegates to the other and they must never drift apart;
  * the zero-gamma *level* is found by re-pricing, not by cumulative sum.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from apexflow.analytics.gex import chain_gex, gex_summary, bs_gamma
from apexflow.analytics.dealer_greeks import (
    chain_exposures, roll_up, exposure_summary, vex_summary,
    dealer_signs, greek_grid, gamma_flip_level, net_gex_curve,
    CONTRACT_MULTIPLIER,
)
from apexflow.analytics.greeks import greeks as scalar_greeks
from apexflow.analytics.timeutil import years_to_expiry

SPOT = 100.0


def expiry_in(days: int) -> str:
    """An expiry `days` from today, so these tests never rot.

    Tenor matters here in a way it does not in the Greeks tests: a chain
    dated years out has gamma so diffuse that a big far-OTM open-interest
    cluster dominates the profile at every price, and the gamma flip
    genuinely does not exist within a sane search band. Tests about the flip
    therefore need a realistic near-dated tenor, not an arbitrarily distant
    one.
    """
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


EXPIRY = expiry_in(150)


def one_contract(strike: float, oi: float, iv: float = 0.30) -> pd.DataFrame:
    return pd.DataFrame([{"strike": strike, "openInterest": oi,
                          "impliedVolatility": iv, "volume": 0.0,
                          "bid": 1.0, "ask": 1.1}])


def build_chain(strikes, call_oi, put_oi, iv=0.30):
    calls = pd.DataFrame({"strike": strikes, "openInterest": call_oi,
                          "impliedVolatility": iv, "volume": 10.0,
                          "bid": 1.0, "ask": 1.1})
    puts = pd.DataFrame({"strike": strikes, "openInterest": put_oi,
                         "impliedVolatility": iv, "volume": 10.0,
                         "bid": 1.0, "ask": 1.1})
    return calls, puts


class TestSignConvention:
    def test_call_only_chain_is_positive_gex(self):
        df = chain_gex(one_contract(100, 1000), pd.DataFrame(), SPOT, EXPIRY)
        assert df["total_gex"].sum() > 0

    def test_put_only_chain_is_negative_gex(self):
        df = chain_gex(pd.DataFrame(), one_contract(100, 1000), SPOT, EXPIRY)
        assert df["total_gex"].sum() < 0

    def test_balanced_chain_nets_to_zero(self):
        """Equal call and put OI at the same strike cancels exactly."""
        df = chain_gex(one_contract(100, 500), one_contract(100, 500), SPOT, EXPIRY)
        assert df["total_gex"].sum() == pytest.approx(0.0, abs=1e-6)

    def test_inverted_convention_flips_every_sign(self):
        calls, puts = build_chain([95, 100, 105], [100, 200, 150], [180, 220, 90])
        naive = chain_exposures(calls, puts, SPOT, EXPIRY, convention="naive")
        inv = chain_exposures(calls, puts, SPOT, EXPIRY, convention="inverted")
        for col in ("gex", "dex", "vex"):
            assert np.allclose(naive[col].to_numpy(), -inv[col].to_numpy(), rtol=1e-9)

    def test_all_short_makes_both_sides_negative_gamma(self):
        calls, puts = build_chain([100], [500], [500])
        e = chain_exposures(calls, puts, SPOT, EXPIRY, convention="all_short")
        assert e["call_gex"].sum() < 0 and e["put_gex"].sum() < 0

    def test_dealer_signs_table(self):
        assert dealer_signs("naive") == (+1.0, -1.0)
        assert dealer_signs("inverted") == (-1.0, +1.0)
        assert dealer_signs("all_short") == (-1.0, -1.0)
        with pytest.raises(ValueError):
            dealer_signs("nonsense")


class TestScaling:
    def test_single_contract_magnitude_is_exact(self):
        """Hand-compute gamma * OI * 100 * S^2 * 0.01 and match it."""
        oi, iv = 1234.0, 0.30
        t = years_to_expiry(EXPIRY)
        gamma = bs_gamma(SPOT, 100.0, t, iv)
        expected = gamma * oi * CONTRACT_MULTIPLIER * (SPOT ** 2) * 0.01

        df = chain_gex(one_contract(100.0, oi, iv), pd.DataFrame(), SPOT, EXPIRY)
        assert df["total_gex"].sum() == pytest.approx(expected, rel=1e-6)

    def test_gex_scales_linearly_with_open_interest(self):
        a = chain_gex(one_contract(100, 100), pd.DataFrame(), SPOT, EXPIRY)["total_gex"].sum()
        b = chain_gex(one_contract(100, 1000), pd.DataFrame(), SPOT, EXPIRY)["total_gex"].sum()
        assert b == pytest.approx(10 * a, rel=1e-9)

    def test_vex_is_dollars_per_vol_point(self):
        """VEX must equal per-contract vega x OI x 100, with vega per +1 vol pt."""
        oi, iv = 500.0, 0.30
        t = years_to_expiry(EXPIRY)
        vega = scalar_greeks(SPOT, 100.0, t, iv, "C")["vega"]
        expected = vega * oi * CONTRACT_MULTIPLIER
        e = chain_exposures(one_contract(100.0, oi, iv), pd.DataFrame(), SPOT, EXPIRY)
        assert e["vex"].sum() == pytest.approx(expected, rel=1e-6)

    def test_dex_is_dollars_of_stock(self):
        oi, iv = 500.0, 0.30
        t = years_to_expiry(EXPIRY)
        delta = scalar_greeks(SPOT, 100.0, t, iv, "C")["delta"]
        expected = delta * oi * CONTRACT_MULTIPLIER * SPOT
        e = chain_exposures(one_contract(100.0, oi, iv), pd.DataFrame(), SPOT, EXPIRY)
        assert e["dex"].sum() == pytest.approx(expected, rel=1e-6)


class TestEngineAgreement:
    def test_chain_gex_matches_chain_exposures(self):
        """gex.chain_gex delegates to dealer_greeks — pin that they agree."""
        strikes = np.arange(80.0, 121.0, 1.0)
        calls, puts = build_chain(strikes,
                                  np.linspace(100, 2000, len(strikes)),
                                  np.linspace(2000, 100, len(strikes)))
        now = pd.Timestamp("2027-01-04T15:00:00Z").to_pydatetime()
        g = chain_gex(calls, puts, SPOT, EXPIRY, now=now)
        e = chain_exposures(calls, puts, SPOT, EXPIRY, now=now)
        assert np.allclose(g["total_gex"].to_numpy(), e["gex"].to_numpy(), rtol=1e-12)
        assert np.allclose(g["call_gex"].to_numpy(), e["call_gex"].to_numpy(), rtol=1e-12)

    def test_greek_grid_matches_scalar_greeks(self):
        """The vectorised grid must equal the scalar implementation."""
        t = 0.35
        strikes = np.array([80.0, 95.0, 100.0, 110.0, 130.0])
        ivs = np.array([0.42, 0.34, 0.30, 0.31, 0.38])
        for right in ("C", "P"):
            grid = greek_grid(SPOT, strikes, t, ivs, right)
            for i, (k, iv) in enumerate(zip(strikes, ivs)):
                s = scalar_greeks(SPOT, float(k), t, float(iv), right)
                assert grid["delta"][i] == pytest.approx(s["delta"], rel=1e-9)
                assert grid["gamma"][i] == pytest.approx(s["gamma"], rel=1e-9)
                assert grid["vega"][i] == pytest.approx(s["vega"], rel=1e-9)


class TestAggregation:
    def test_duplicate_strikes_are_summed(self):
        calls = pd.concat([one_contract(100, 500), one_contract(100, 500)],
                          ignore_index=True)
        single = chain_gex(one_contract(100, 1000), pd.DataFrame(), SPOT, EXPIRY)
        doubled = chain_gex(calls, pd.DataFrame(), SPOT, EXPIRY)
        assert len(doubled) == 1
        assert doubled["total_gex"].sum() == pytest.approx(single["total_gex"].sum(), rel=1e-9)

    def test_roll_up_sums_across_expiries(self):
        calls, puts = build_chain([95, 100, 105], [10, 20, 30], [30, 20, 10])
        a = chain_exposures(calls, puts, SPOT, expiry_in(30))
        b = chain_exposures(calls, puts, SPOT, expiry_in(120))
        rolled = roll_up([a, b])
        assert len(rolled) == 3
        assert rolled["gex"].sum() == pytest.approx(a["gex"].sum() + b["gex"].sum(), rel=1e-9)

    def test_roll_up_of_nothing_is_empty_not_an_error(self):
        assert roll_up([]).empty
        assert roll_up([None, pd.DataFrame()]).empty


class TestNaNHandling:
    def test_nan_iv_falls_back_to_chain_atm_iv(self):
        calls = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "openInterest": [100.0, 100.0, 100.0],
            "impliedVolatility": [0.30, float("nan"), 0.30],
            "volume": [1.0, 1.0, 1.0], "bid": [1.0]*3, "ask": [1.1]*3,
        })
        df = chain_gex(calls, pd.DataFrame(), SPOT, EXPIRY)
        assert len(df) == 3
        assert df["total_gex"].notna().all()
        assert (df["total_gex"] > 0).all()

    def test_nan_open_interest_becomes_zero(self):
        calls = one_contract(100, 500)
        calls.loc[0, "openInterest"] = float("nan")
        df = chain_gex(calls, pd.DataFrame(), SPOT, EXPIRY)
        assert df["total_gex"].sum() == pytest.approx(0.0)

    def test_empty_inputs_give_empty_frame_with_right_columns(self):
        df = chain_gex(pd.DataFrame(), pd.DataFrame(), SPOT, EXPIRY)
        assert df.empty
        assert list(df.columns) == ["strike", "call_gex", "put_gex", "total_gex"]

    def test_zero_spot_is_handled(self):
        assert chain_gex(one_contract(100, 100), pd.DataFrame(), 0.0, EXPIRY).empty

    def test_negative_and_zero_strikes_dropped(self):
        calls = pd.DataFrame({"strike": [-5.0, 0.0, 100.0],
                              "openInterest": [100.0]*3,
                              "impliedVolatility": [0.3]*3,
                              "volume": [1.0]*3, "bid": [1.0]*3, "ask": [1.1]*3})
        df = chain_gex(calls, pd.DataFrame(), SPOT, EXPIRY)
        assert list(df["strike"]) == [100.0]


class TestGexSummary:
    def test_flip_is_interpolated_between_bracketing_strikes(self):
        """Cumulative GEX crosses zero between 100 and 101 → flip lands inside."""
        df = pd.DataFrame({
            "strike": [99.0, 100.0, 101.0, 102.0],
            "call_gex": [0.0, 0.0, 150.0, 0.0],
            "put_gex": [-100.0, 0.0, 0.0, 0.0],
            "total_gex": [-100.0, 0.0, 150.0, 0.0],
        })
        s = gex_summary(df, SPOT)
        assert 100.0 < s["gamma_flip"] < 101.0

    def test_extremes_reported(self):
        df = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "call_gex": [0.0, 500.0, 0.0],
            "put_gex": [-800.0, 0.0, 0.0],
            "total_gex": [-800.0, 500.0, 0.0],
        })
        s = gex_summary(df, SPOT)
        assert s["max_pos_strike"] == 100.0
        assert s["max_neg_strike"] == 95.0
        assert s["total_gex"] == pytest.approx(-300.0)

    def test_no_crossing_gives_none(self):
        df = pd.DataFrame({"strike": [99.0, 100.0], "call_gex": [10.0, 10.0],
                           "put_gex": [0.0, 0.0], "total_gex": [10.0, 10.0]})
        assert gex_summary(df, SPOT)["gamma_flip"] is None

    def test_empty_summary_is_safe(self):
        s = gex_summary(pd.DataFrame(), SPOT)
        assert s["total_gex"] == 0.0 and s["gamma_flip"] is None


class TestZeroGammaLevel:
    """The re-priced flip level, versus the cumulative-sum shortcut."""

    def _put_heavy_index_chain(self):
        """Mimic an index book: a fat put wall well below spot, near-dated."""
        strikes = np.arange(70.0, 131.0, 1.0)
        call_oi = np.where(strikes >= SPOT, 800.0, 100.0)
        put_oi = np.where(strikes <= SPOT * 0.85, 6000.0, 200.0)
        calls, puts = build_chain(strikes, call_oi, put_oi)
        return calls, puts

    def test_flip_level_is_near_spot_not_at_the_put_wall(self):
        calls, puts = self._put_heavy_index_chain()
        repriced = gamma_flip_level([(EXPIRY, calls, puts)], SPOT)
        cumsum = exposure_summary(
            chain_exposures(calls, puts, SPOT, EXPIRY), SPOT)["gamma_flip"]

        # The re-priced level exists and sits in a sane band around spot.
        assert repriced["bracketed"] is True
        assert 0.80 * SPOT < repriced["flip"] < 1.20 * SPOT

        # The cumulative-sum shortcut fails outright on this shape: summing
        # upward from the lowest strike starts deep in the put wall and
        # never climbs back through zero, so it reports "no flip" for a book
        # that plainly has one just above spot. When it does return a
        # number on a chain like this it is dragged down toward the wall.
        assert cumsum is None or cumsum < repriced["flip"]

    def test_curve_is_returned_and_monotone_in_length(self):
        calls, puts = build_chain([95, 100, 105], [500, 500, 500], [500, 500, 500])
        out = gamma_flip_level([(EXPIRY, calls, puts)], SPOT, n_points=21)
        assert len(out["curve"]) == 21
        assert all("spot" in p and "total_gex" in p for p in out["curve"])

    def test_no_crossing_reports_unbracketed_rather_than_a_number(self):
        """An all-call book never flips; say so instead of inventing a level."""
        calls, _ = build_chain([95, 100, 105], [500, 500, 500], [0, 0, 0])
        out = gamma_flip_level([(EXPIRY, calls, pd.DataFrame())], SPOT)
        assert out["bracketed"] is False
        assert out["flip"] is None
        assert out["regime"] == "positive_gamma"

    def test_net_gex_curve_sign_follows_the_book(self):
        calls, _ = build_chain([100], [1000], [0])
        grid = np.array([90.0, 100.0, 110.0])
        curve = net_gex_curve([(EXPIRY, calls, pd.DataFrame())], grid)
        assert (curve > 0).all()

    def test_zero_spot_short_circuits(self):
        assert gamma_flip_level([], 0.0)["flip"] is None


class TestVexSummary:
    def test_naive_convention_on_put_heavy_book_is_short_vega(self):
        calls, puts = build_chain([95, 100, 105], [100, 100, 100], [900, 900, 900])
        e = chain_exposures(calls, puts, SPOT, EXPIRY)
        v = vex_summary(e, SPOT)
        assert v["net_short_vega"] is True
        assert v["total_vex"] < 0

    def test_near_spot_share_is_a_fraction(self):
        strikes = np.arange(70.0, 131.0, 1.0)
        calls, puts = build_chain(strikes, np.full(len(strikes), 500.0),
                                  np.full(len(strikes), 100.0))
        v = vex_summary(chain_exposures(calls, puts, SPOT, EXPIRY), SPOT)
        assert 0.0 <= v["pct_near_spot"] <= 1.0

    def test_empty_is_safe(self):
        v = vex_summary(pd.DataFrame(), SPOT)
        assert v["total_vex"] == 0.0 and v["net_short_vega"] is False


class TestVolumeBasis:
    def test_volume_basis_uses_volume_column(self):
        df = pd.DataFrame([{"strike": 100.0, "openInterest": 0.0, "volume": 750.0,
                            "impliedVolatility": 0.30, "bid": 1.0, "ask": 1.1}])
        by_oi = chain_exposures(df, pd.DataFrame(), SPOT, EXPIRY, oi_col="openInterest")
        by_vol = chain_exposures(df, pd.DataFrame(), SPOT, EXPIRY, oi_col="volume")
        assert by_oi["gex"].sum() == pytest.approx(0.0)
        assert by_vol["gex"].sum() > 0
