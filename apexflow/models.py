"""Shared dataclasses used across scanners and the platform layer."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Signal:
    """A single scanner emission. Every scanner returns a list of these."""
    scanner: str
    symbol: str
    score: float                 # 0-100; higher = stronger setup
    reason: str                  # human-readable summary
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class OptionsPrint:
    """A single options trade print (sweep/block/sweep-of-blocks)."""
    symbol: str
    strike: float
    expiry: str                  # YYYY-MM-DD
    right: str                   # "C" or "P"
    size: int                    # contracts
    price: float                 # per-contract price
    premium: float               # size * price * 100
    side: str                    # "ASK", "BID", "MID", "UNKNOWN"
    volume: int
    open_interest: int
    iv: float | None = None
    print_type: str = "TRADE"    # SWEEP, BLOCK, SWEEP_OF_BLOCKS, TRADE
    timestamp: datetime = field(default_factory=_utcnow)

    @property
    def volume_oi_ratio(self) -> float:
        return self.volume / self.open_interest if self.open_interest else float("inf")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class DarkPoolPrint:
    symbol: str
    size: int                    # shares
    price: float
    notional: float              # size * price
    venue: str                   # ATS venue code, e.g. "UBSA"
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass
class Alert:
    symbol: str
    scanner: str
    message: str
    severity: str = "info"       # info | warn | critical
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass
class BacktestResult:
    scanner: str
    signals_tested: int
    wins: int
    losses: int
    avg_return_pct: float
    median_return_pct: float
    win_rate: float
    sharpe: float
    lookback_days: int
    hold_days: int
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"{self.scanner}: {self.signals_tested} signals, "
            f"win_rate={self.win_rate:.1%}, "
            f"avg={self.avg_return_pct:+.2f}%, "
            f"sharpe={self.sharpe:.2f}, "
            f"sortino={self.sortino:.2f}, "
            f"max_dd={self.max_drawdown_pct:+.2f}%"
        )
