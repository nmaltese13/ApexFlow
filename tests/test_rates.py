"""Treasury yield curve — parsing, interpolation, and graceful failure.

Runs offline: the XML fixture below is a trimmed copy of the real Treasury
feed, so parsing is pinned against the actual format rather than a guess.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from apexflow.analytics import rates
from apexflow.analytics.rates import (
    TreasuryCurve, _parse, _to_continuous, risk_free_rate, FALLBACK_RATE,
)

# Trimmed from the live feed for 2026-08-21.
FIXTURE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><content type="application/xml"><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-08-20T00:00:00</d:NEW_DATE>
<d:BC_1MONTH m:type="Edm.Double">3.70</d:BC_1MONTH>
<d:BC_10YEAR m:type="Edm.Double">4.70</d:BC_10YEAR>
</m:properties></content></entry>
<entry><content type="application/xml"><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-08-21T00:00:00</d:NEW_DATE>
<d:BC_1MONTH m:type="Edm.Double">3.80</d:BC_1MONTH>
<d:BC_3MONTH m:type="Edm.Double">3.88</d:BC_3MONTH>
<d:BC_6MONTH m:type="Edm.Double">3.95</d:BC_6MONTH>
<d:BC_1YEAR m:type="Edm.Double">4.03</d:BC_1YEAR>
<d:BC_2YEAR m:type="Edm.Double">4.24</d:BC_2YEAR>
<d:BC_10YEAR m:type="Edm.Double">4.74</d:BC_10YEAR>
<d:BC_30YEAR m:type="Edm.Double">5.27</d:BC_30YEAR>
</m:properties></content></entry>
</feed>"""


class TestCompounding:
    def test_semiannual_to_continuous_is_lower(self):
        """Continuous compounding of the same cash flows implies a lower rate."""
        assert _to_continuous(4.0) < 0.04

    def test_conversion_matches_the_formula(self):
        assert _to_continuous(3.88) == pytest.approx(2 * math.log1p(0.0388 / 2), abs=1e-12)

    def test_zero_is_zero(self):
        assert _to_continuous(0.0) == 0.0

    def test_conversion_is_small_but_not_negligible(self):
        """~4bp at these levels — worth getting right since it is free."""
        diff = 0.0388 - _to_continuous(3.88)
        assert 0.0001 < diff < 0.0010


class TestParsing:
    def test_takes_the_most_recent_entry(self):
        curve = _parse(FIXTURE)
        assert curve is not None
        assert curve.as_of == date(2026, 8, 21)

    def test_reads_every_available_tenor(self):
        curve = _parse(FIXTURE)
        assert len(curve.points) == 7
        assert curve.points == sorted(curve.points)

    def test_missing_tenors_are_skipped_not_zeroed(self):
        """The fixture has no 5-year; it must be absent, not present as 0."""
        curve = _parse(FIXTURE)
        assert all(r > 0 for _, r in curve.points)

    def test_marks_itself_live(self):
        assert _parse(FIXTURE).live is True

    @pytest.mark.parametrize("junk", ["", "<feed></feed>", "not xml at all",
                                      "<entry><d:NEW_DATE>garbage</d:NEW_DATE></entry>"])
    def test_junk_returns_none_rather_than_raising(self, junk):
        assert _parse(junk) is None

    def test_entry_without_tenors_returns_none(self):
        xml = ('<feed><entry><d:NEW_DATE m:type="Edm.DateTime">2026-08-21T00:00:00'
               '</d:NEW_DATE></entry></feed>')
        assert _parse(xml) is None


class TestInterpolation:
    @pytest.fixture
    def curve(self):
        return _parse(FIXTURE)

    def test_hits_the_knots_exactly(self, curve):
        assert curve.rate(0.25) == pytest.approx(_to_continuous(3.88), abs=1e-12)
        assert curve.rate(2.0) == pytest.approx(_to_continuous(4.24), abs=1e-12)

    def test_interpolates_between_knots(self, curve):
        mid = curve.rate(0.375)          # halfway between 3M and 6M
        lo, hi = curve.rate(0.25), curve.rate(0.5)
        assert lo < mid < hi
        assert mid == pytest.approx((lo + hi) / 2, rel=1e-9)

    def test_flat_below_the_short_end(self, curve):
        """Never extrapolate down into negative rates."""
        assert curve.rate(1e-9) == curve.rate(1 / 12)
        assert curve.rate(0.0) == curve.rate(1 / 12)

    def test_flat_beyond_the_long_end(self, curve):
        assert curve.rate(50.0) == curve.rate(30.0)
        assert curve.rate(1000.0) == curve.rate(30.0)

    def test_upward_sloping_curve_is_monotone(self, curve):
        ts = [1 / 12, 0.25, 0.5, 1.0, 2.0, 10.0, 30.0]
        rs = [curve.rate(t) for t in ts]
        assert rs == sorted(rs)

    def test_tenor_matters(self, curve):
        """The point of the module: short and long differ materially."""
        assert abs(curve.rate(30.0) - curve.rate(1 / 12)) > 0.010   # >100bp

    def test_empty_curve_returns_fallback(self):
        assert TreasuryCurve(as_of=date(2026, 1, 1), points=[]).rate(1.0) == FALLBACK_RATE


class TestFailureBehaviour:
    def test_fallback_curve_is_flat_at_the_old_constant(self):
        c = TreasuryCurve.fallback()
        assert c.live is False
        for t in (1 / 365, 0.25, 1.0, 10.0):
            assert c.rate(t) == pytest.approx(FALLBACK_RATE)

    def test_network_failure_falls_back_without_raising(self, monkeypatch):
        rates.clear_cache()

        def boom(*a, **k):
            raise RuntimeError("network down")

        monkeypatch.setattr(rates.requests, "get", boom)
        r = risk_free_rate(0.25)
        assert r == pytest.approx(FALLBACK_RATE)
        assert rates.get_curve().live is False
        rates.clear_cache()

    def test_round_trip_through_the_cache_format(self):
        curve = _parse(FIXTURE)
        again = TreasuryCurve.from_dict(curve.to_dict())
        assert again.as_of == curve.as_of
        assert again.points == curve.points
        assert again.rate(1.0) == curve.rate(1.0)


class TestAgainstGreeks:
    def test_rate_error_grows_with_tenor(self):
        """The module's actual justification: one constant cannot fit the curve.

        A 50bp rate error barely touches a one-week option and moves a
        two-year one materially, which is exactly why the rate has to be
        matched to the contract's tenor rather than fixed.
        """
        from apexflow.analytics.greeks import bs_price

        def price_gap(t):
            lo = bs_price(100, 100, t, 0.30, "C", r=0.0376)
            hi = bs_price(100, 100, t, 0.30, "C", r=0.0426)
            return abs(hi - lo) / lo

        assert price_gap(2.0) > price_gap(0.25) > price_gap(7 / 365)

    def test_stale_rate_misprices_a_long_dated_option(self):
        """Concrete size of the error the old hardcoded constant produced."""
        from apexflow.analytics.greeks import bs_price
        stale = bs_price(100, 100, 3.0, 0.30, "C", r=0.04)
        live = bs_price(100, 100, 3.0, 0.30, "C", r=0.0426)   # 3y from the curve
        assert abs(live - stale) > 0.05      # more than five cents on a ~$25 option

    def test_put_call_parity_holds_at_the_live_rate(self):
        from apexflow.analytics.greeks import bs_price
        r = _parse(FIXTURE).rate(0.25)
        c = bs_price(100, 95, 0.25, 0.30, "C", r=r)
        p = bs_price(100, 95, 0.25, 0.30, "P", r=r)
        assert c - p == pytest.approx(100 - 95 * math.exp(-r * 0.25), abs=1e-9)
