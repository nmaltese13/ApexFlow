"""Smoke tests for earnings_direction predictor."""
from datetime import date, timedelta
import pandas as pd
import numpy as np
import pytest

from apexflow.analytics.earnings_direction import (
    risk_reversal, iv_skew, short_setup, drift_history, predict_direction,
    DirectionalView,
)


@pytest.fixture
def sample_chain():
    spot = 100.0
    calls = pd.DataFrame([
        {"strike": k, "impliedVolatility": 0.45 + 0.001*(k-100), "openInterest": 1000, "volume": 100}
        for k in range(80, 121, 5)
    ])
    puts = pd.DataFrame([
        {"strike": k, "impliedVolatility": 0.55 + 0.002*(100-k), "openInterest": 1500, "volume": 150}
        for k in range(80, 121, 5)
    ])
    return calls, puts, spot


def test_risk_reversal_negative_when_puts_pricier(sample_chain):
    calls, puts, spot = sample_chain
    sig, detail = risk_reversal(calls, puts, spot)
    # puts have higher IV than calls in our setup → negative RR
    assert sig < 0


def test_risk_reversal_zero_on_empty_chain():
    sig, _ = risk_reversal(pd.DataFrame(), pd.DataFrame(), 100.0)
    assert sig == 0.0


def test_iv_skew_normal_downside_skew(sample_chain):
    calls, puts, spot = sample_chain
    sig, detail = iv_skew(calls, puts, spot)
    # Has clear downside skew → mild bearish bias
    assert sig < 0
    assert detail.get("skew_pts", 0) > 0


def test_short_setup_bullish_when_heavy_short_low_price():
    sig, _ = short_setup(0.30, 7.0, 0.20)  # 30% short, DTC 7, near 52w low
    assert sig > 0.5


def test_short_setup_zero_when_no_short():
    sig, _ = short_setup(0.0, 0.0, 0.5)
    assert sig == 0.0


def test_drift_history_zero_on_empty():
    sig, _ = drift_history(None, [])
    assert sig == 0.0


def test_predict_direction_returns_view(sample_chain):
    calls, puts, spot = sample_chain
    hist = pd.DataFrame({
        "Close": np.linspace(95, 100, 500),
        "High":  np.linspace(96, 102, 500),
        "Low":   np.linspace(94, 99, 500),
    }, index=pd.date_range(end=date.today(), periods=500))
    earn = [date.today() - timedelta(days=90 * i) for i in range(1, 5)]

    view = predict_direction(
        calls=calls, puts=puts, spot=spot,
        short_pct_float=0.30, days_to_cover=7.0,
        history_df=hist, earnings_dates=earn, price_pos_52w=0.20,
    )
    assert isinstance(view, DirectionalView)
    assert view.direction in ("bullish", "bearish", "neutral")
    assert 0.0 <= view.confidence <= 1.0
    assert -1.0 <= view.composite <= 1.0


def test_predict_direction_neutral_below_threshold():
    """When all signals are weak, direction should land in 'neutral'."""
    # Calls and puts with identical IV — no risk reversal signal
    calls = pd.DataFrame([{"strike": 100, "impliedVolatility": 0.40, "openInterest": 100, "volume": 10}])
    puts = pd.DataFrame([{"strike": 100, "impliedVolatility": 0.40, "openInterest": 100, "volume": 10}])
    view = predict_direction(
        calls=calls, puts=puts, spot=100.0,
        short_pct_float=0.0, days_to_cover=0.0,
        history_df=None, earnings_dates=[], price_pos_52w=0.50,
    )
    assert view.direction == "neutral"
    assert view.confidence < 0.20
