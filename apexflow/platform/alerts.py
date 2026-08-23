"""Terminal alerts with cooldown to prevent spam."""
from __future__ import annotations
import time
from collections import deque
from typing import Callable

from apexflow.models import Alert, Signal
import config


class AlertManager:
    def __init__(self, cooldown_seconds: int = config.ALERT_COOLDOWN_SECONDS,
                 sink: Callable[[Alert], None] | None = None):
        self.cooldown = cooldown_seconds
        self.sink = sink or self._default_sink
        self._last_fired: dict[str, float] = {}
        self.recent: deque[Alert] = deque(maxlen=200)

    def _default_sink(self, alert: Alert) -> None:
        # Plain stdout — UI layer overrides this to display in dashboard.
        ts = alert.timestamp.strftime("%H:%M:%S")
        marker = {"info": "•", "warn": "!", "critical": "!!"}.get(alert.severity, "•")
        print(f"[{ts}] {marker} {alert.symbol} ({alert.scanner}) — {alert.message}")

    def maybe_fire(self, signal: Signal, threshold: float = 70.0,
                   severity: str | None = None) -> bool:
        if signal.score < threshold:
            return False
        key = f"{signal.scanner}:{signal.symbol}"
        now = time.time()
        last = self._last_fired.get(key, 0)
        if now - last < self.cooldown:
            return False
        self._last_fired[key] = now
        sev = severity or ("critical" if signal.score >= 85 else "warn" if signal.score >= 70 else "info")
        alert = Alert(symbol=signal.symbol, scanner=signal.scanner,
                      message=f"score={signal.score:.0f} — {signal.reason}", severity=sev)
        self.recent.append(alert)
        try:
            self.sink(alert)
        except Exception:
            pass
        return True

    def process(self, signals: list[Signal], threshold: float = 70.0) -> int:
        fired = 0
        for s in signals:
            if self.maybe_fire(s, threshold):
                fired += 1
        return fired
