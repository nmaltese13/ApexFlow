"""Atlas — intraday GEX/OI node snapshot engine.

What this is
------------
SkylitAi's "Atlas" view shows orbs at each strike that grow throughout the day
as more open interest accumulates and gamma intensifies. Our existing gex.py
already computes a per-strike GEX *snapshot*; what was missing is a time-series
of those snapshots so the UI can animate growth.

This module provides:

  * AtlasStore — thin SQLite wrapper for snapshot persistence
  * snapshot_symbol(...) — capture one frame for a symbol
  * frames_for(symbol, ...) — query a time range of frames
  * compute_node_metrics(frames, spot) — per-strike growth / heat / dominance

Schema (single table, one row per (symbol, ts, expiry_bucket, strike)):

    snapshots(
      ts            INTEGER NOT NULL,    -- unix seconds, UTC
      symbol        TEXT    NOT NULL,
      expiry_bucket TEXT    NOT NULL,    -- '0dte' | 'wkly' | 'monthly'
      strike        REAL    NOT NULL,
      total_gex     REAL    NOT NULL,    -- $ per 1% (signed)
      call_gex      REAL    NOT NULL,
      put_gex       REAL    NOT NULL,
      call_oi       INTEGER NOT NULL,
      put_oi        INTEGER NOT NULL,
      call_vol      INTEGER NOT NULL,
      put_vol       INTEGER NOT NULL,
      spot          REAL    NOT NULL,
      PRIMARY KEY (symbol, ts, expiry_bucket, strike)
    )

The "node growth" metric the UI consumes is computed at query time, not stored
— that way we can change the formula without re-snapshotting.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config
from apexflow.analytics.gex import chain_gex
from apexflow.providers.base import DataProvider

log = logging.getLogger(__name__)

ATLAS_DB_PATH = config.DATA_DIR / "atlas.db"

# Number of (call,put) snapshots we keep around the spot per expiry bucket.
# 30 strikes × 3 buckets × ~80 snapshots/day = ~7k rows/day/symbol.
_STRIKES_PER_SIDE = 20

# Bucket the front expiries we snapshot. yfinance gives us full chains so we
# select the closest expiry in each bucket and tag rows accordingly.
BUCKETS = ("0dte", "wkly", "monthly")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class AtlasStore:
    """SQLite-backed snapshot store. Thread-safe via per-connection locks
    (FastAPI runs handlers in a threadpool)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS snapshots (
        ts            INTEGER NOT NULL,
        symbol        TEXT    NOT NULL,
        expiry_bucket TEXT    NOT NULL,
        strike        REAL    NOT NULL,
        total_gex     REAL    NOT NULL,
        call_gex      REAL    NOT NULL,
        put_gex       REAL    NOT NULL,
        call_oi       INTEGER NOT NULL,
        put_oi        INTEGER NOT NULL,
        call_vol      INTEGER NOT NULL,
        put_vol       INTEGER NOT NULL,
        spot          REAL    NOT NULL,
        PRIMARY KEY (symbol, ts, expiry_bucket, strike)
    );
    CREATE INDEX IF NOT EXISTS idx_snap_sym_ts ON snapshots(symbol, ts);
    CREATE INDEX IF NOT EXISTS idx_snap_sym_bucket_ts ON snapshots(symbol, expiry_bucket, ts);

    CREATE TABLE IF NOT EXISTS sessions (
        ts        INTEGER PRIMARY KEY,
        symbol    TEXT    NOT NULL,
        spot      REAL,
        notes     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_sess_sym_ts ON sessions(symbol, ts);
    """

    def __init__(self, path: Path | str = ATLAS_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as c:
            c.executescript(self._SCHEMA)

    @contextmanager
    def _connect(self):
        # `check_same_thread=False` because the snapshot loop and request
        # handlers may run on different threads. The lock guards writes.
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- writes -------------------------------------------------------
    def write_frame(self, symbol: str, ts: int, frame: pd.DataFrame) -> int:
        """Insert (or replace) one frame. Returns row count written."""
        if frame is None or frame.empty:
            return 0
        rows = []
        for _, r in frame.iterrows():
            rows.append((
                int(ts), symbol.upper(), str(r["expiry_bucket"]),
                float(r["strike"]), float(r["total_gex"]),
                float(r.get("call_gex") or 0), float(r.get("put_gex") or 0),
                int(r.get("call_oi") or 0), int(r.get("put_oi") or 0),
                int(r.get("call_vol") or 0), int(r.get("put_vol") or 0),
                float(r.get("spot") or 0),
            ))
        with self._lock, self._connect() as c:
            c.executemany(
                "INSERT OR REPLACE INTO snapshots "
                "(ts, symbol, expiry_bucket, strike, total_gex, call_gex, put_gex, "
                " call_oi, put_oi, call_vol, put_vol, spot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # ----- reads --------------------------------------------------------
    def read_range(self, symbol: str, from_ts: int, to_ts: int,
                   bucket: str | None = None) -> pd.DataFrame:
        sql = ("SELECT ts, symbol, expiry_bucket, strike, total_gex, call_gex, put_gex, "
               "call_oi, put_oi, call_vol, put_vol, spot "
               "FROM snapshots WHERE symbol = ? AND ts BETWEEN ? AND ?")
        params: list = [symbol.upper(), int(from_ts), int(to_ts)]
        if bucket:
            sql += " AND expiry_bucket = ?"
            params.append(bucket)
        sql += " ORDER BY ts ASC, strike ASC"
        with self._connect() as c:
            df = pd.read_sql_query(sql, c, params=params)
        return df

    def latest_ts(self, symbol: str) -> int | None:
        with self._connect() as c:
            row = c.execute("SELECT MAX(ts) AS ts FROM snapshots WHERE symbol = ?",
                            (symbol.upper(),)).fetchone()
        return int(row["ts"]) if row and row["ts"] is not None else None

    def symbols(self) -> list[str]:
        with self._connect() as c:
            rows = c.execute("SELECT DISTINCT symbol FROM snapshots ORDER BY symbol").fetchall()
        return [r["symbol"] for r in rows]

    def prune_before(self, before_ts: int) -> int:
        """Drop snapshots older than `before_ts`. Returns rows deleted."""
        with self._lock, self._connect() as c:
            cur = c.execute("DELETE FROM snapshots WHERE ts < ?", (int(before_ts),))
            return cur.rowcount


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------
@dataclass
class BucketChoice:
    bucket: str
    expiry: str
    days: int


def _pick_bucket_expiries(expiries: list[str], today: date | None = None) -> list[BucketChoice]:
    """From the full expiry list, pick one expiry per bucket (0dte/wkly/monthly).

    * 0dte    — only if today's expiry exists
    * wkly    — closest expiry > today and ≤ 9d away
    * monthly — closest expiry between 18 and 60 days
    """
    today = today or date.today()
    chosen: list[BucketChoice] = []

    parsed: list[tuple[date, str]] = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            parsed.append((d, e))
        except (ValueError, TypeError):
            continue
    parsed.sort()

    # 0DTE
    same_day = [(d, e) for d, e in parsed if d == today]
    if same_day:
        chosen.append(BucketChoice("0dte", same_day[0][1], 0))

    # Weekly: 1..9 days
    weekly = [(d, e) for d, e in parsed if 0 < (d - today).days <= 9]
    if weekly:
        d, e = weekly[0]
        chosen.append(BucketChoice("wkly", e, (d - today).days))

    # Monthly: 18..60 days, prefer the one nearest 30
    monthly = [(d, e) for d, e in parsed if 18 <= (d - today).days <= 60]
    if monthly:
        d, e = min(monthly, key=lambda t: abs((t[0] - today).days - 30))
        chosen.append(BucketChoice("monthly", e, (d - today).days))

    return chosen


def _build_frame(provider: DataProvider, symbol: str,
                 buckets: Iterable[BucketChoice]) -> pd.DataFrame:
    """Pull each bucket's chain, compute GEX, return one stacked DataFrame."""
    pieces: list[pd.DataFrame] = []
    spot_used = 0.0

    for choice in buckets:
        try:
            chain = provider.options_chain(symbol, choice.expiry)
        except Exception as e:
            log.debug("atlas: chain fetch failed for %s %s: %s", symbol, choice.expiry, e)
            continue
        calls = chain.get("calls"); puts = chain.get("puts")
        spot = float(chain.get("spot") or 0)
        if spot <= 0 or calls is None or puts is None or calls.empty:
            continue
        spot_used = spot

        gex_df = chain_gex(calls, puts, spot, choice.expiry)
        if gex_df.empty:
            continue

        # Trim to ±N strikes around spot — the UI doesn't render the full chain.
        gex_df = gex_df.copy()
        gex_df["dist"] = (gex_df["strike"] - spot).abs()
        gex_df = gex_df.nsmallest(_STRIKES_PER_SIDE * 2, "dist").drop(columns="dist")

        # Augment with per-strike OI / volume by re-merging from raw chain
        def _agg(df: pd.DataFrame, oi_col: str, vol_col: str) -> pd.DataFrame:
            d = df.copy()
            d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
            d["openInterest"] = pd.to_numeric(d.get("openInterest", 0), errors="coerce").fillna(0)
            d["volume"] = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)
            return d.groupby("strike", as_index=False).agg(
                **{oi_col: ("openInterest", "sum"), vol_col: ("volume", "sum")}
            )

        c_agg = _agg(calls, "call_oi", "call_vol")
        p_agg = _agg(puts, "put_oi", "put_vol")
        merged = gex_df.merge(c_agg, on="strike", how="left").merge(p_agg, on="strike", how="left")
        for col in ("call_oi", "put_oi", "call_vol", "put_vol"):
            merged[col] = merged[col].fillna(0).astype(int)
        merged["expiry_bucket"] = choice.bucket
        merged["spot"] = spot
        pieces.append(merged)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def snapshot_symbol(provider: DataProvider, symbol: str,
                    store: AtlasStore | None = None,
                    now: datetime | None = None) -> int:
    """Capture one Atlas frame for ``symbol``. Returns rows written.

    Designed to be cheap and idempotent — safe to call from a tight loop.
    """
    store = store or AtlasStore()
    now = now or datetime.now(timezone.utc)
    ts = int(now.timestamp())

    try:
        expiries = provider.expiries(symbol) or []
    except Exception as e:
        log.debug("atlas: expiries fetch failed for %s: %s", symbol, e)
        return 0

    choices = _pick_bucket_expiries(expiries, today=now.date())
    if not choices:
        log.debug("atlas: no bucketed expiries for %s", symbol)
        return 0

    frame = _build_frame(provider, symbol, choices)
    if frame.empty:
        return 0

    return store.write_frame(symbol, ts, frame)


# ---------------------------------------------------------------------------
# Query / metric helpers consumed by the API
# ---------------------------------------------------------------------------
def _ema(series: np.ndarray, span: float) -> np.ndarray:
    """Vectorised EMA. ``span`` is the EMA span (≈ window length)."""
    if series.size == 0:
        return series
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, series.size):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def compute_node_metrics(frames: pd.DataFrame, spot: float | None = None) -> pd.DataFrame:
    """Add growth / heat / dominance columns to a multi-frame DataFrame.

    Input: rows from AtlasStore.read_range (sorted by ts asc, strike asc).

    Output adds:
        node_strength   — sqrt(|total_gex|) × OI weighting, normalised 0..1
        growth_pct      — change since the strike's first appearance today
        heat            — EMA(|total_gex|, span=4 frames) — smooths jitter
        is_king         — top-1 |gex| of latest frame
        is_wall_pos     — top-3 positive gex of latest frame
        is_wall_neg     — top-3 negative gex of latest frame
    """
    if frames is None or frames.empty:
        return pd.DataFrame()

    df = frames.copy()
    df = df.sort_values(["expiry_bucket", "strike", "ts"]).reset_index(drop=True)

    # Per (bucket, strike) time series → rolling growth + EMA heat
    out_pieces = []
    for (_bucket, _strike), grp in df.groupby(["expiry_bucket", "strike"], sort=False):
        g = grp.copy()
        gex_abs = g["total_gex"].abs().to_numpy()
        # Heat: EMA of |gex| (span=4 ≈ ~20 min if 5-min snapshots)
        g["heat"] = _ema(gex_abs, span=4.0)
        # Growth: (current - first) / first, NaN-safe
        first = float(gex_abs[0]) if gex_abs.size else 0.0
        g["growth_pct"] = ((gex_abs - first) / first * 100.0) if first else 0.0
        # Total OI as supplemental size signal
        g["total_oi"] = (g["call_oi"] + g["put_oi"]).astype(float)
        out_pieces.append(g)
    df = pd.concat(out_pieces, ignore_index=True)

    # Node strength: sqrt(|gex|) × log1p(OI) normalised against the session max
    raw = np.sqrt(df["total_gex"].abs().to_numpy()) * np.log1p(df["total_oi"].to_numpy())
    max_raw = raw.max() if raw.size else 0.0
    df["node_strength"] = (raw / max_raw) if max_raw > 0 else 0.0

    # Per-frame dominance flags (king / walls) from the latest frame only —
    # cheaper than per-frame and the UI only needs them on the live snapshot.
    df["is_king"] = False
    df["is_wall_pos"] = False
    df["is_wall_neg"] = False
    if not df.empty:
        latest_ts = df["ts"].max()
        latest = df[df["ts"] == latest_ts].copy()
        for bucket, grp in latest.groupby("expiry_bucket"):
            idx_king = grp["total_gex"].abs().idxmax()
            df.loc[idx_king, "is_king"] = True
            pos = grp[grp["total_gex"] > 0].nlargest(3, "total_gex").index
            neg = grp[grp["total_gex"] < 0].nsmallest(3, "total_gex").index
            df.loc[pos, "is_wall_pos"] = True
            df.loc[neg, "is_wall_neg"] = True

    return df


def session_summary(frames_df: pd.DataFrame) -> dict:
    """Quick session-level numbers for the page header."""
    if frames_df is None or frames_df.empty:
        return {"snapshots": 0, "first_ts": None, "last_ts": None,
                "buckets": [], "strikes": 0}
    return {
        "snapshots": int(frames_df["ts"].nunique()),
        "first_ts": int(frames_df["ts"].min()),
        "last_ts": int(frames_df["ts"].max()),
        "buckets": sorted(frames_df["expiry_bucket"].unique().tolist()),
        "strikes": int(frames_df["strike"].nunique()),
    }


# ---------------------------------------------------------------------------
# Snapshot loop (for the background runner)
# ---------------------------------------------------------------------------
def is_market_hours(now: datetime | None = None) -> bool:
    """Crude US RTH check (Mon-Fri, 13:30-20:00 UTC ≈ 9:30-16:00 ET).

    Doesn't account for holidays — yfinance just returns stale data on
    closures, which is fine; we just won't write new rows.
    """
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    minutes = now.hour * 60 + now.minute
    return 13 * 60 + 30 <= minutes <= 20 * 60


def snapshot_universe(provider: DataProvider, symbols: Iterable[str],
                      store: AtlasStore | None = None) -> dict[str, int]:
    """Snapshot a universe; returns {symbol: rows_written}."""
    store = store or AtlasStore()
    out: dict[str, int] = {}
    for sym in symbols:
        try:
            n = snapshot_symbol(provider, sym, store)
            out[sym] = n
            if n:
                log.info("atlas: %s wrote %d rows", sym, n)
        except Exception as e:
            log.warning("atlas: snapshot failed for %s: %s", sym, e)
            out[sym] = 0
        # Tiny inter-symbol pacing — yfinance hates burst requests
        time.sleep(0.4)
    return out
