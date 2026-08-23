"""Smoke tests for apexflow.analytics.atlas — storage + bucket selection + metrics.

Runs without network or API keys.
"""
from datetime import date, timedelta
import pandas as pd
import pytest

from apexflow.analytics.atlas import (
    AtlasStore, _pick_bucket_expiries, compute_node_metrics, session_summary,
)


@pytest.fixture
def tmp_store(tmp_path):
    return AtlasStore(tmp_path / "atlas.db")


@pytest.fixture
def sample_frame():
    return pd.DataFrame([
        {"strike": 100, "total_gex": 1.5e8, "call_gex": 2e8, "put_gex": -5e7,
         "call_oi": 12000, "put_oi": 4000, "call_vol": 800, "put_vol": 300,
         "spot": 102.5, "expiry_bucket": "wkly"},
        {"strike": 105, "total_gex": -8e7, "call_gex": 1e8, "put_gex": -1.8e8,
         "call_oi": 8000, "put_oi": 11000, "call_vol": 600, "put_vol": 700,
         "spot": 102.5, "expiry_bucket": "wkly"},
        {"strike": 110, "total_gex": 4e7, "call_gex": 6e7, "put_gex": -2e7,
         "call_oi": 5000, "put_oi": 1000, "call_vol": 200, "put_vol": 100,
         "spot": 102.5, "expiry_bucket": "wkly"},
    ])


def test_bucket_picker_returns_correct_keys():
    today = date(2026, 5, 4)
    mk = lambda d: (today + timedelta(days=d)).strftime("%Y-%m-%d")
    expiries = [mk(0), mk(1), mk(7), mk(30), mk(60)]
    choices = _pick_bucket_expiries(expiries, today=today)
    buckets = {c.bucket: c.expiry for c in choices}
    assert "0dte" in buckets
    assert buckets["0dte"] == mk(0)
    assert buckets["wkly"] == mk(1)
    assert buckets["monthly"] == mk(30)


def test_bucket_picker_skips_missing_categories():
    today = date(2026, 5, 4)
    # Only weekly available
    expiries = [(today + timedelta(days=2)).strftime("%Y-%m-%d")]
    choices = _pick_bucket_expiries(expiries, today=today)
    assert {c.bucket for c in choices} == {"wkly"}


def test_store_roundtrip(tmp_store, sample_frame):
    n = tmp_store.write_frame("TSLA", 1717200000, sample_frame)
    assert n == 3
    df = tmp_store.read_range("TSLA", 1717199000, 1717201000)
    assert len(df) == 3
    assert set(df["strike"]) == {100, 105, 110}
    assert set(df["expiry_bucket"]) == {"wkly"}


def test_store_filters_by_bucket(tmp_store, sample_frame):
    tmp_store.write_frame("TSLA", 1717200000, sample_frame)
    df_w = tmp_store.read_range("TSLA", 1717199000, 1717201000, bucket="wkly")
    assert len(df_w) == 3
    df_0 = tmp_store.read_range("TSLA", 1717199000, 1717201000, bucket="0dte")
    assert len(df_0) == 0


def test_node_metrics_growth_pct(tmp_store, sample_frame):
    # Two snapshots 5 minutes apart, 40% gex growth
    tmp_store.write_frame("TSLA", 1717200000, sample_frame)
    growth = sample_frame.copy()
    growth["total_gex"] *= 1.4
    tmp_store.write_frame("TSLA", 1717200300, growth)

    df = tmp_store.read_range("TSLA", 1717199000, 1717201000)
    enriched = compute_node_metrics(df)
    later = enriched[enriched["ts"] == 1717200300]
    # All strikes should have ~+40% growth in the second snapshot
    assert all(abs(g - 40.0) < 1.0 for g in later["growth_pct"])


def test_node_metrics_king_is_max_abs(tmp_store, sample_frame):
    tmp_store.write_frame("TSLA", 1717200000, sample_frame)
    df = tmp_store.read_range("TSLA", 1717199000, 1717201000)
    enriched = compute_node_metrics(df)
    king_rows = enriched[enriched["is_king"]]
    assert len(king_rows) == 1
    # Strike 100 has largest |gex| (1.5e8)
    assert king_rows.iloc[0]["strike"] == 100


def test_session_summary_counts():
    df = pd.DataFrame({
        "ts": [1, 1, 1, 2, 2],
        "strike": [100, 105, 110, 100, 105],
        "expiry_bucket": ["wkly"] * 5,
    })
    summary = session_summary(df)
    assert summary["snapshots"] == 2
    assert summary["strikes"] == 3
    assert summary["buckets"] == ["wkly"]
