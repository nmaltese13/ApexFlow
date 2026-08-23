"""Append-only JSONL log of every signal emitted, for audit + backtesting."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from apexflow.models import Signal
import config


class SignalLogger:
    def __init__(self, path: Path | str = config.SIGNAL_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(self, signal: Signal) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(signal.to_dict()) + "\n")

    def log_many(self, signals: list[Signal]) -> None:
        if not signals:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for s in signals:
                f.write(json.dumps(s.to_dict()) + "\n")

    def iter_recent(self, n: int = 50) -> Iterator[dict]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return iter([])
        for line in lines[-n:]:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
