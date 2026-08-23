"""Smoke tests for gex_profile classifier (walls / pillars / slides / pins / violence)."""
import pandas as pd
import pytest

from apexflow.analytics.gex_profile import (
    classify_walls, classify_slide, classify_pin, classify_pillars,
    violence_flag, profile_chain,
)


def test_walls_finds_concentrated_strike():
    """A single dominant strike with weak neighbors = wall."""
    gex = pd.DataFrame([
        {"strike": 95,  "total_gex": 1e6},
        {"strike": 100, "total_gex": 5e8},   # dominant
        {"strike": 105, "total_gex": 2e6},
    ])
    walls = classify_walls(gex)
    assert 100 in walls


def test_walls_skips_smooth_distribution():
    """A smooth gamma distribution shouldn't produce walls."""
    gex = pd.DataFrame([
        {"strike": k, "total_gex": 4e8 - abs(k - 100) * 5e6}
        for k in range(90, 111)
    ])
    walls = classify_walls(gex)
    assert len(walls) <= 1   # at most the peak itself, but neighbors are too close


def test_slide_detects_directional_ramp():
    """Linearly increasing GEX should classify as 'up' slide."""
    gex = pd.DataFrame([
        {"strike": k, "total_gex": (k - 100) * 1e7}
        for k in range(80, 121, 5)
    ])
    slide = classify_slide(gex)
    assert slide["direction"] == "up"
    assert slide["slope"] > 0
    assert slide["r2"] > 0.95


def test_slide_flat_when_random():
    """Random gamma → no slide direction."""
    import numpy as np
    rng = np.random.RandomState(7)
    gex = pd.DataFrame([
        {"strike": k, "total_gex": rng.normal(0, 1e8)}
        for k in range(80, 121)
    ])
    slide = classify_slide(gex)
    assert slide["direction"] == "flat" or slide["r2"] < 0.5


def test_pin_finds_at_money_positive_gamma():
    spot = 100.0
    gex = pd.DataFrame([
        {"strike": 95,  "total_gex": -5e7},
        {"strike": 100, "total_gex": 5e8},  # heavy positive at spot
        {"strike": 105, "total_gex": 1e7},
    ])
    assert classify_pin(gex, spot) == 100


def test_pin_returns_none_when_negative_at_spot():
    spot = 100.0
    gex = pd.DataFrame([
        {"strike": 100, "total_gex": -5e8},  # negative — not a pin
    ])
    assert classify_pin(gex, spot) is None


def test_violence_flag_detects_negative_dominance():
    """Mostly-negative GEX near spot → violent."""
    spot = 100.0
    gex = pd.DataFrame([
        {"strike": 99,  "total_gex": -3e8},
        {"strike": 100, "total_gex": -2e8},
        {"strike": 101, "total_gex": -1e8},
        {"strike": 102, "total_gex": +5e7},  # tiny positive
    ])
    flag = violence_flag(gex, spot)
    assert flag["violent"] is True
    assert flag["net_gex_near_spot"] < 0


def test_violence_flag_calm_when_positive():
    spot = 100.0
    gex = pd.DataFrame([
        {"strike": 99,  "total_gex": +3e8},
        {"strike": 100, "total_gex": +5e8},
        {"strike": 101, "total_gex": +2e8},
    ])
    flag = violence_flag(gex, spot)
    assert flag["violent"] is False
    assert flag["net_gex_near_spot"] > 0


def test_pillars_finds_persistent_strike_across_dtes():
    """Strike 100 is heavy in all three DTE buckets → pillar."""
    base = lambda heavy: pd.DataFrame([
        {"strike": k, "total_gex": (5e8 if k == heavy else 1e7)}
        for k in [95, 100, 105]
    ])
    per_dte = {
        "0dte":   base(100),
        "wkly":   base(100),
        "monthly": base(100),
    }
    pillars = classify_pillars(per_dte, min_dtes=2)
    assert any(p["strike"] == 100 and p["n_dtes"] == 3 for p in pillars)


def test_pillars_skips_single_dte():
    """Strike that's only heavy in one DTE shouldn't be a pillar."""
    per_dte = {
        "0dte":   pd.DataFrame([{"strike": 100, "total_gex": 5e8}]),
        "wkly":   pd.DataFrame([{"strike": 200, "total_gex": 5e8}]),
    }
    pillars = classify_pillars(per_dte, min_dtes=2)
    assert pillars == []


def test_profile_chain_returns_all_keys():
    spot = 100.0
    gex = pd.DataFrame([
        {"strike": 95,  "total_gex": -2e8},
        {"strike": 100, "total_gex": +5e8},
        {"strike": 105, "total_gex": +1e8},
    ])
    profile = profile_chain(gex, spot)
    assert set(profile.keys()) == {"walls", "slide", "pin", "violence"}
    assert isinstance(profile["walls"], list)
    assert isinstance(profile["slide"], dict)
    assert isinstance(profile["violence"], dict)
