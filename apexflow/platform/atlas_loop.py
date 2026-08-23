"""Background snapshot worker for Atlas — RTH + extended hours + weekends.

Cadence is tiered by session window (all times in UTC, US ET basis):

    RTH       Mon-Fri 13:30 → 20:00     full cadence (e.g. 5 min)
    Pre-mkt   Mon-Fri 08:00 → 13:30     reduced cadence (every 30 min)
    Post-mkt  Mon-Fri 20:00 → 00:00     reduced cadence (every 30 min)
    Weekend   Sat/Sun all day            occasional (every 4 hours) — captures
                                          OI re-pricing on Sunday evening when
                                          dealers update positions ahead of Mon

Each tier runs the same rotating-slice scheduler so over a few cycles every
symbol gets coverage. The cadence selector is pure (no I/O) so it's easy to
unit-test.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import config
from apexflow.analytics.atlas import AtlasStore, snapshot_symbol
from apexflow.providers import get_provider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session windows (UTC) — tied to US equity options sessions
# ---------------------------------------------------------------------------
# Pre-market starts ~04:00 ET (08:00 UTC outside DST, 09:00 during DST).
# We use 08:00 UTC as a single approximation — yfinance returns stale data
# before that anyway.
RTH_START_UTC      = 13 * 60 + 30   # 09:30 ET ≈ 13:30 UTC (no DST handling)
RTH_END_UTC        = 20 * 60         # 16:00 ET ≈ 20:00 UTC
PREMKT_START_UTC   = 8 * 60          # 04:00 ET ≈ 08:00 UTC
POSTMKT_END_UTC    = 24 * 60         # 20:00 ET ≈ 00:00 UTC next day


@dataclass
class CadenceProfile:
    rth_seconds: float = 300.0           # 5 min during RTH
    extended_seconds: float = 1800.0     # 30 min pre/post market
    weekend_seconds: float = 14400.0     # 4 hours on Sat/Sun
    rth_slice: int = 6
    extended_slice: int = 4
    weekend_slice: int = 3


def session_kind(now: datetime | None = None) -> str:
    """Return one of 'rth' | 'premkt' | 'postmkt' | 'weekend' | 'overnight'."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:                    # Saturday / Sunday
        return "weekend"
    minutes = now.hour * 60 + now.minute
    if RTH_START_UTC <= minutes < RTH_END_UTC:
        return "rth"
    if PREMKT_START_UTC <= minutes < RTH_START_UTC:
        return "premkt"
    if RTH_END_UTC <= minutes < POSTMKT_END_UTC:
        return "postmkt"
    return "overnight"


def cadence_for(kind: str, profile: CadenceProfile) -> tuple[float, int] | None:
    """Return (sleep_seconds, slice_size) for a session, or None to skip."""
    if kind == "rth":
        return profile.rth_seconds, profile.rth_slice
    if kind in ("premkt", "postmkt"):
        return profile.extended_seconds, profile.extended_slice
    if kind == "weekend":
        return profile.weekend_seconds, profile.weekend_slice
    return None  # overnight 00:00-08:00 UTC: skip entirely


class AtlasLoop:
    """Tiered-cadence snapshot scheduler. Idempotent start/stop."""

    def __init__(self,
                 universe: Iterable[str],
                 profile: CadenceProfile | None = None,
                 inter_symbol_pause: float = 0.4,
                 startup_delay: float = 15.0,
                 store: AtlasStore | None = None):
        self.universe = list(dict.fromkeys(s.upper() for s in universe if s))
        self.profile = profile or CadenceProfile()
        self.inter_pause = float(inter_symbol_pause)
        self.startup_delay = float(startup_delay)
        self.store = store or AtlasStore()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor = 0
        self._last_run: float = 0.0
        self._last_written: int = 0
        self._last_kind: str = "init"
        self._sweeps: int = 0

    # ---------------------------------------------------------------- API
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="atlas-loop")
        self._thread.start()
        log.info("AtlasLoop started — universe=%d (tiered: rth=%ds, ext=%ds, wknd=%ds)",
                 len(self.universe), self.profile.rth_seconds,
                 self.profile.extended_seconds, self.profile.weekend_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "universe_size": len(self.universe),
            "current_session": session_kind(),
            "last_session": self._last_kind,
            "sweeps_completed": self._sweeps,
            "last_run_unix": self._last_run,
            "last_rows_written": self._last_written,
            "cursor": self._cursor,
            "profile": {
                "rth_seconds":     self.profile.rth_seconds,
                "ext_seconds":     self.profile.extended_seconds,
                "weekend_seconds": self.profile.weekend_seconds,
            },
        }

    # ------------------------------------------------------------ thread
    def _next_slice(self, slice_size: int) -> list[str]:
        if not self.universe:
            return []
        n = len(self.universe)
        start = self._cursor % n
        end = start + slice_size
        if end <= n:
            slc = self.universe[start:end]
        else:
            slc = self.universe[start:] + self.universe[: end - n]
        self._cursor = end % n
        if self._cursor == 0:
            self._sweeps += 1
        return slc

    def _run(self) -> None:
        time.sleep(self.startup_delay)
        provider = get_provider()
        while not self._stop.is_set():
            try:
                kind = session_kind()
                cad = cadence_for(kind, self.profile)
                if cad is None:
                    # Overnight dead zone — sleep 30 min and re-check
                    self._last_kind = kind
                    self._stop.wait(timeout=1800)
                    continue

                sleep_s, slice_size = cad
                slc = self._next_slice(slice_size)
                rows_total = 0
                for sym in slc:
                    if self._stop.is_set():
                        break
                    try:
                        rows = snapshot_symbol(provider, sym, self.store)
                        rows_total += rows
                    except Exception as e:
                        log.debug("atlas-loop: %s failed: %s", sym, e)
                    if self.inter_pause:
                        self._stop.wait(timeout=self.inter_pause)
                self._last_run = time.time()
                self._last_written = rows_total
                self._last_kind = kind
                if rows_total:
                    log.info("atlas-loop[%s]: wrote %d rows across %d symbols",
                             kind, rows_total, len(slc))
                self._stop.wait(timeout=sleep_s)
            except Exception as e:
                log.exception("atlas-loop: unexpected error: %s", e)
                self._stop.wait(timeout=60)


def default_universe() -> list[str]:
    """Default symbols to snapshot — focused on actively-traded options names."""
    return [
        "SPY", "QQQ", "IWM",
        "TSLA", "NVDA", "AAPL", "AMZN", "META", "MSFT", "GOOGL",
        "AMD", "NFLX", "BABA", "PLTR", "INTC", "NOK", "CAR",
        "GME", "AMC", "MARA", "RIOT", "HOOD", "COIN", "SOFI",
    ]
