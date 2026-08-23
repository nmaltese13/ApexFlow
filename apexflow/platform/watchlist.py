"""Auto-populated watchlist from scan results."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from apexflow.models import Signal
import config


class Watchlist:
    def __init__(self, path: Path | str = config.WATCHLIST_PATH, max_size: int = 50):
        self.path = Path(path)
        self.max_size = max_size
        self.entries: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text())
            except Exception:
                self.entries = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))

    def add(self, signal: Signal) -> None:
        existing = self.entries.get(signal.symbol)
        new = {
            "symbol": signal.symbol,
            "best_score": signal.score,
            "best_scanner": signal.scanner,
            "tags": signal.tags,
            "reason": signal.reason,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "scanners_hit": [signal.scanner],
        }
        if existing:
            new["best_score"] = max(existing.get("best_score", 0), signal.score)
            if existing.get("best_score", 0) >= signal.score:
                new["best_scanner"] = existing["best_scanner"]
                new["reason"] = existing["reason"]
            scanners = set(existing.get("scanners_hit", []))
            scanners.add(signal.scanner)
            new["scanners_hit"] = sorted(scanners)
            new["added_at"] = existing["added_at"]
        self.entries[signal.symbol] = new

    def add_many(self, signals: list[Signal], min_score: float = 50.0) -> None:
        for s in signals:
            if s.score >= min_score:
                self.add(s)
        self._trim()
        self.save()

    def _trim(self) -> None:
        if len(self.entries) <= self.max_size:
            return
        sorted_entries = sorted(self.entries.items(),
                                key=lambda kv: kv[1].get("best_score", 0), reverse=True)
        self.entries = dict(sorted_entries[: self.max_size])

    def remove(self, symbol: str) -> None:
        self.entries.pop(symbol, None)
        self.save()

    def list(self) -> list[dict]:
        return sorted(self.entries.values(), key=lambda e: e.get("best_score", 0), reverse=True)
