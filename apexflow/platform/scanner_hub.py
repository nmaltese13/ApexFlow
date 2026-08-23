"""Runs all scanners in parallel, dispatches signals to logger / watchlist / alerts."""
from __future__ import annotations
import concurrent.futures as cf
import logging
import time
from typing import Callable

from apexflow.models import Signal
from apexflow.providers import get_provider
from apexflow.scanners import ALL_SCANNERS
from .alerts import AlertManager
from .signal_logger import SignalLogger
from .watchlist import Watchlist
import config

log = logging.getLogger(__name__)


class ScannerHub:
    def __init__(self,
                 universe: list[str],
                 enabled: list[str] | None = None,
                 alert_threshold: float = 70.0,
                 watchlist_threshold: float = 60.0,
                 alert_sink: Callable | None = None):
        self.provider = get_provider()
        self.universe = universe
        self.enabled = enabled or list(ALL_SCANNERS.keys())
        self.alert_threshold = alert_threshold
        self.watchlist_threshold = watchlist_threshold
        self.scanners = {n: ALL_SCANNERS[n](self.provider) for n in self.enabled if n in ALL_SCANNERS}
        self.logger = SignalLogger()
        self.watchlist = Watchlist()
        self.alerts = AlertManager(sink=alert_sink)
        self.last_results: dict[str, list[Signal]] = {}
        self.last_run_time: float = 0.0

    def run_once(self) -> dict[str, list[Signal]]:
        results: dict[str, list[Signal]] = {}
        with cf.ThreadPoolExecutor(max_workers=len(self.scanners) or 1) as ex:
            futures = {ex.submit(s.scan_safe, self.universe): name for name, s in self.scanners.items()}
            for fut in cf.as_completed(futures):
                name = futures[fut]
                sigs = fut.result()
                results[name] = sigs
                self.logger.log_many(sigs)
                self.watchlist.add_many(sigs, min_score=self.watchlist_threshold)
                self.alerts.process(sigs, threshold=self.alert_threshold)
        self.last_results = results
        self.last_run_time = time.time()
        return results

    def loop(self, iterations: int | None = None, sleep_s: int = config.HUB_LOOP_SECONDS):
        i = 0
        while iterations is None or i < iterations:
            log.info("Hub run #%d starting", i + 1)
            self.run_once()
            i += 1
            if iterations is None or i < iterations:
                time.sleep(sleep_s)
