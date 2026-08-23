"""Offline provider backed by the frozen snapshot in ``data/demo/``.

Point the whole terminal at a real, committed options chain with no API key,
no network, and no rate limit::

    APEXFLOW_DEMO=1 python main.py web
    python main.py web --demo

The data is verbatim provider output captured by
``scripts/capture_demo_dataset.py`` - see ``docs/demo_dataset.md`` for what
is in it and how it was taken.

Keeping the snapshot from going stale
-------------------------------------
A frozen chain has a problem a frozen price series does not: its expiries
are dated. Left alone, every contract in the file is expired within a month
of capture, time-to-expiry floors at one minute, and every gamma number goes
to nonsense - the demo would quietly break instead of failing loudly.

So dates are shifted forward by default. The shift is
``round_to_whole_weeks(today - capture_date)``, applied to every expiry and
every history timestamp. Whole weeks specifically, because equity option
expiries are weekday-anchored: a Friday monthly stays a Friday, the 0DTE
contract stays 0DTE, and the term structure keeps its exact original shape.
Only the labels move; not one price, IV, or open-interest figure is touched.

Set ``APEXFLOW_DEMO_SHIFT=0`` to see the raw captured dates instead - the
right choice when checking the snapshot against the tape from that day, and
the wrong one for a live demo.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .base import DataProvider

log = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "demo"


def demo_available(demo_dir: Path | None = None) -> bool:
    """True when a usable frozen dataset is present on disk."""
    d = Path(demo_dir or DEMO_DIR)
    return (d / "manifest.json").exists() and any(d.glob("*.json.gz"))


def _round_weeks(days: int) -> int:
    """Round a day count to the nearest whole number of weeks."""
    return int(round(days / 7.0)) * 7


class DemoProvider(DataProvider):
    """Reads the frozen snapshot. Implements the full DataProvider surface."""

    name = "demo"

    def __init__(self, demo_dir: Path | None = None, shift_dates: bool | None = None):
        self.dir = Path(demo_dir or DEMO_DIR)
        if shift_dates is None:
            shift_dates = os.environ.get("APEXFLOW_DEMO_SHIFT", "1") not in ("0", "false", "False")
        self.shift_dates = bool(shift_dates)
        self.manifest = self._load_manifest()
        self.capture_date = self._parse_capture_date()
        self.shift_days = self._compute_shift()
        if self.shift_days:
            log.info("DemoProvider: shifting frozen dates forward by %d days "
                     "(captured %s)", self.shift_days, self.capture_date)

    # -- setup ------------------------------------------------------------
    def _load_manifest(self) -> dict:
        p = self.dir / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(
                f"No demo dataset at {self.dir}. Build one with:\n"
                f"    python scripts/capture_demo_dataset.py")
        return json.loads(p.read_text(encoding="utf-8"))

    def _parse_capture_date(self) -> date:
        raw = self.manifest.get("capture_date") or self.manifest.get("captured_at", "")
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except ValueError:
            return datetime.now(timezone.utc).date()

    def _compute_shift(self) -> int:
        if not self.shift_dates:
            return 0
        gap = (datetime.now(timezone.utc).date() - self.capture_date).days
        return _round_weeks(gap) if gap > 0 else 0

    # -- date shifting ----------------------------------------------------
    def _shift_expiry(self, iso: str) -> str:
        if not self.shift_days:
            return iso
        try:
            d = datetime.strptime(iso[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return iso
        return (d + timedelta(days=self.shift_days)).isoformat()

    def _unshift_expiry(self, iso: str | None) -> str | None:
        """Map a caller-facing expiry back to the key stored in the file."""
        if iso is None or not self.shift_days:
            return iso
        try:
            d = datetime.strptime(str(iso)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return iso
        return (d - timedelta(days=self.shift_days)).isoformat()

    # -- loading ----------------------------------------------------------
    @lru_cache(maxsize=64)
    def _load(self, symbol: str) -> dict | None:
        p = self.dir / f"{symbol.upper()}.json.gz"
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("demo snapshot %s unreadable: %s", p.name, e)
            return None

    def symbols(self) -> list[str]:
        """Tickers present in the frozen dataset."""
        return sorted(p.stem.replace(".json", "") for p in self.dir.glob("*.json.gz"))

    def info(self) -> dict:
        """Provenance, for the UI footer and ``main.py status``."""
        return {
            "capture_date": self.capture_date.isoformat(),
            "source": self.manifest.get("source", "unknown"),
            "symbols": self.symbols(),
            "total_contracts": self.manifest.get("total_contracts", 0),
            "shift_days": self.shift_days,
            "shifted": bool(self.shift_days),
        }

    # -- DataProvider surface ---------------------------------------------
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        snap = self._load(symbol)
        if not snap:
            return empty
        h = snap.get("history") or {}
        idx = h.get("index") or []
        if not idx:
            return empty
        shift = timedelta(days=self.shift_days)
        df = pd.DataFrame({
            c: h.get(c, []) for c in ("Open", "High", "Low", "Close", "Volume")
        }, index=pd.DatetimeIndex([datetime.fromtimestamp(t, tz=timezone.utc) + shift
                                   for t in idx]))
        # Only the daily bars were captured; an intraday request gets the
        # daily series rather than a silently empty frame.
        n = _period_to_rows(period)
        return df.tail(n) if n else df

    def quote(self, symbol: str) -> dict:
        snap = self._load(symbol)
        if not snap:
            return {}
        q = dict(snap.get("quote") or {})
        q.setdefault("symbol", symbol.upper())
        q["price"] = float(q.get("price") or snap.get("spot") or 0.0)
        q["as_of"] = snap.get("captured_at")
        return q

    def expiries(self, symbol: str) -> list[str]:
        snap = self._load(symbol)
        if not snap:
            return []
        return [self._shift_expiry(e) for e in sorted(snap.get("expiries") or [])]

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        snap = self._load(symbol)
        if not snap:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(),
                    "expiry": None, "spot": 0.0}
        chains = snap.get("chains") or {}
        stored_keys = sorted(chains.keys())
        if not stored_keys:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(),
                    "expiry": None, "spot": float(snap.get("spot") or 0.0)}

        key = self._unshift_expiry(expiry) if expiry else stored_keys[0]
        if key not in chains:
            key = stored_keys[0]
        side = chains[key]
        return {
            "calls": _to_frame(side.get("calls")),
            "puts": _to_frame(side.get("puts")),
            "expiry": self._shift_expiry(key),
            "spot": float(snap.get("spot") or 0.0),
        }

    def fundamentals(self, symbol: str) -> dict:
        snap = self._load(symbol)
        return dict(snap.get("fundamentals") or {}) if snap else {}

    def earnings_calendar(self, symbol: str) -> list[date]:
        snap = self._load(symbol)
        if not snap:
            return []
        out = []
        for iso in snap.get("earnings") or []:
            try:
                d = datetime.strptime(str(iso)[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            out.append(d + timedelta(days=self.shift_days) if self.shift_days else d)
        return out

    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        # Headlines were deliberately not frozen: they go stale in a way
        # prices do not, and a week-old headline presented as current is
        # worse than no headline.
        return []


_CHAIN_NUMERIC = ("strike", "lastPrice", "bid", "ask", "volume",
                  "openInterest", "impliedVolatility")


def _to_frame(records: list[dict] | None) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(_CHAIN_NUMERIC))
    df = pd.DataFrame(records)
    for c in _CHAIN_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0
    return df


def _period_to_rows(period: str) -> int | None:
    """Approximate trading-day count for a yfinance-style period string."""
    table = {"1d": 1, "5d": 5, "1mo": 22, "3mo": 66, "6mo": 126,
             "1y": 252, "2y": 504, "5y": 1260, "max": None, "ytd": None}
    return table.get(str(period).lower(), 252)
