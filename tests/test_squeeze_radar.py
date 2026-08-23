"""Smoke tests for the Squeeze Radar layer scorers."""
import pandas as pd
import pytest

from apexflow.scanners.squeeze_radar import (
    score_short, score_gex_proximity, score_compression,
    score_rvol, score_catalyst, score_trend,
    RadarBreakdown, WEIGHTS,
)


def test_short_layer_responds_to_short_pct():
    s_low, _ = score_short(0.05, 1.0, 0.05, 0.0)
    s_high, _ = score_short(0.40, 8.0, 0.50, 0.5)
    assert s_high > s_low


def test_gex_proximity_max_when_close_to_wall():
    spot = 100.0
    df = pd.DataFrame([{"strike": 100, "total_gex": 5e8}])
    score, _ = score_gex_proximity(spot, df)
    assert score >= 80


def test_gex_proximity_zero_when_far_from_wall():
    spot = 100.0
    df = pd.DataFrame([{"strike": 200, "total_gex": 5e8}])
    score, _ = score_gex_proximity(spot, df)
    assert score < 5


def test_gex_geometry_bonus_for_setup_geometry():
    """Negative wall just below + positive wall just above = breakout setup."""
    spot = 100.0
    df = pd.DataFrame([
        {"strike": 99, "total_gex": -2e8},
        {"strike": 101, "total_gex": +3e8},
    ])
    score, detail = score_gex_proximity(spot, df)
    assert detail.get("geometry_bonus", 0) > 5


def test_rvol_layer_zero_when_no_swell():
    import numpy as np
    vol = pd.Series([1_000_000] * 30)  # flat
    score, _ = score_rvol(vol)
    assert score == 0


def test_rvol_layer_high_when_swelling():
    import numpy as np
    # last 5 days at 6x the prior 20 days → 5d/20d ratio ≈ 3.0 → score ≈ 66
    vol = pd.Series([1_000_000] * 20 + [6_000_000] * 5)
    score, _ = score_rvol(vol)
    assert score > 50


def test_catalyst_decay():
    from datetime import date, timedelta
    today = date.today()
    s_imm, _ = score_catalyst([today + timedelta(days=1)])
    s_far, _ = score_catalyst([today + timedelta(days=12)])
    s_none, _ = score_catalyst([today + timedelta(days=30)])
    assert s_imm > s_far > s_none == 0


def test_breakdown_composite_respects_weights():
    """All-zero breakdown → 0; all-100 breakdown → 100."""
    b0 = RadarBreakdown()
    assert b0.composite() == 0.0
    b100 = RadarBreakdown(short=100, gex=100, compression=100, rvol=100,
                           catalyst=100, trend=100)
    # Sum of weights = 100, so composite max = 100
    assert b100.composite() == 100.0


def test_breakdown_synergy_clamped_at_100():
    b = RadarBreakdown(short=100, gex=100, compression=100, rvol=100,
                       catalyst=100, trend=100, synergy=20)
    assert b.composite() == 100.0  # clamped


def test_compression_returns_valid_shape():
    """Verify score_compression returns a 0-100 score and a detail dict
    with the expected keys, on real-shaped synthetic data."""
    import numpy as np
    rng = np.random.RandomState(42)
    close = pd.Series(100 + rng.normal(0, 1.0, 60).cumsum() / 5)
    high = close + 0.5
    low = close - 0.5
    score, detail = score_compression(close, high, low)
    assert 0.0 <= score <= 100.0
    assert "bb_bw" in detail and "bb_bw_pctile" in detail
    assert "ttm_squeeze" in detail
    assert 0.0 <= detail["bb_bw_pctile"] <= 1.0
