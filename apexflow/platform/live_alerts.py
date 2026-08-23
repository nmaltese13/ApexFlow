"""LiveAlertsEngine — background scanner that produces real, evolving alerts.

Why: the previous AlertManager only fired when something else called
ScannerHub.run_once(). Nothing in the web app actually does that on a schedule,
so the alerts feed was effectively static. This engine owns its own thread,
runs the full scanner hub on a rotating slice of the universe, dedupes hits in
a cooldown window, and exposes them through a thread-safe ring buffer the API
can read with no shared lock contention beyond a deque snapshot.

It is intentionally tolerant of failures — a single bad ticker, scanner crash,
or provider timeout never stops the loop. Each cycle records a heartbeat so
the UI can show "engine is alive, last sweep N seconds ago".
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from apexflow.models import Signal
from apexflow.platform.scanner_hub import ScannerHub

log = logging.getLogger(__name__)


class LiveAlertsEngine:
    def __init__(
        self,
        universe: list[str],
        enabled_scanners: list[str] | None = None,
        cooldown_seconds: int = 900,
        slice_size: int = 12,
        sweep_pause: float = 90.0,
        threshold: float = 70.0,
        buffer_size: int = 500,
        startup_delay: float = 20.0,
    ):
        self.full_universe = list(universe)
        self.enabled = enabled_scanners
        self.cooldown = cooldown_seconds
        self.slice_size = max(5, slice_size)
        self.sweep_pause = sweep_pause
        self.threshold = threshold
        self.startup_delay = startup_delay

        self._alerts: deque[dict] = deque(maxlen=buffer_size)
        self._last_fired: dict[str, float] = {}
        self._slice_idx = 0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.last_sweep_at: float = 0.0
        self.last_sweep_size: int = 0
        self.last_sweep_hits: int = 0
        self.last_error: str | None = None
        self.cycles: int = 0
        self.total_hits: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="apex-alerts", daemon=True)
        self._thread.start()
        log.info("LiveAlertsEngine started (universe=%d, slice=%d)",
                 len(self.full_universe), self.slice_size)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "universe_size": len(self.full_universe),
            "slice_size": self.slice_size,
            "cycles": self.cycles,
            "total_hits": self.total_hits,
            "last_sweep_at": self.last_sweep_at,
            "last_sweep_size": self.last_sweep_size,
            "last_sweep_hits": self.last_sweep_hits,
            "last_error": self.last_error,
            "buffer_size": len(self._alerts),
            "cooldown_seconds": self.cooldown,
            "threshold": self.threshold,
            "now": time.time(),
        }

    def recent(self, limit: int = 100, min_severity: str | None = None,
               scanner: str | None = None, symbol: str | None = None) -> list[dict]:
        rank = {"info": 0, "warn": 1, "critical": 2}
        cutoff = rank.get(min_severity or "info", 0)
        snap = list(self._alerts)
        out = []
        for a in reversed(snap):  # newest first
            if rank.get(a.get("severity", "info"), 0) < cutoff:
                continue
            if scanner and a.get("scanner") != scanner:
                continue
            if symbol and a.get("symbol", "").upper() != symbol.upper():
                continue
            out.append(a)
            if len(out) >= limit:
                break
        return out

    def clear(self) -> None:
        self._alerts.clear()
        self._last_fired.clear()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        # Give the web app room to serve a few user requests before
        # the engine starts hammering the provider.
        if self.startup_delay > 0:
            self._stop.wait(self.startup_delay)
        while not self._stop.is_set():
            # If the underlying provider is currently in a rate-limit backoff,
            # idle until it recovers (don't waste cycles or stale-cache reads).
            try:
                from apexflow.providers.yfinance_provider import YFinanceProvider
                wait = YFinanceProvider.backoff_remaining()
            except Exception:
                wait = 0.0
            if wait > 0:
                self.last_error = f"provider rate-limited; sleeping {int(wait)}s"
                self._stop.wait(min(wait + 1, 60))
                continue
            try:
                self._sweep_once()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("LiveAlertsEngine sweep crashed: %s", e)
            self.cycles += 1
            # Pause but stay responsive to stop()
            self._stop.wait(self.sweep_pause)

    def _next_slice(self) -> list[str]:
        n = len(self.full_universe)
        if n == 0:
            return []
        if n <= self.slice_size:
            return list(self.full_universe)
        start = self._slice_idx % n
        end = start + self.slice_size
        if end <= n:
            sl = self.full_universe[start:end]
        else:
            sl = self.full_universe[start:] + self.full_universe[: end - n]
        self._slice_idx = (start + self.slice_size) % n
        return sl

    def _sweep_once(self) -> None:
        slice_syms = self._next_slice()
        if not slice_syms:
            return
        hub = ScannerHub(universe=slice_syms, enabled=self.enabled)
        results = hub.run_once()

        all_signals: list[Signal] = [s for sigs in results.values() for s in sigs]
        hits = 0
        now = time.time()
        for sig in all_signals:
            if sig.score < self.threshold:
                continue
            key = f"{sig.scanner}:{sig.symbol}"
            last = self._last_fired.get(key, 0.0)
            if now - last < self.cooldown:
                continue
            self._last_fired[key] = now
            sev = "critical" if sig.score >= 85 else "warn" if sig.score >= 70 else "info"
            self._alerts.append({
                "symbol":    sig.symbol,
                "scanner":   sig.scanner,
                "score":     float(sig.score),
                "severity":  sev,
                "reason":    sig.reason,
                "tags":      list(sig.tags),
                "metrics":   dict(sig.metrics),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ts":        now,
            })
            hits += 1
        self.last_sweep_at = now
        self.last_sweep_size = len(slice_syms)
        self.last_sweep_hits = hits
        self.total_hits += hits
        self.last_error = None
