"""Backtest a scanner: historically replay signals and measure forward returns.

Scope, and what this is not
---------------------------
This replays a scanner's *boolean* daily signal against price bars. It is a
much weaker exercise than validating a scoring model, and it is separate
from ``squeeze_backtest.py``, which evaluates whether the squeeze *score*
ranks forward returns cross-sectionally with a permutation null and an
overlap correction.

Only scanners whose logic can be reconstructed from daily OHLCV are
supported. The options-driven ones cannot be: historical option chains with
point-in-time open interest are not available from any free provider, so
replaying them would require using today's chain to score a past date, which
is lookahead bias. :data:`SUPPORTED_SCANNERS` is the honest list, and
:meth:`Backtester.run` raises on anything outside it rather than quietly
returning zero signals — an empty result that looks like "no signals fired"
when it really means "this was never evaluated" is worse than an error.

Risk metrics computed:
  Sharpe       — annualized risk-adjusted return (mean / stdev * sqrt(252/hold))
  Sortino      — same but uses downside deviation only
  Max Drawdown — worst peak-to-trough on the equity curve
  Calmar       — annualized return / |max drawdown|
  Profit factor — sum(wins) / |sum(losses)|

Every one of those is computed on overlapping, equally-weighted signals with
no transaction costs and a survivorship-biased universe. Treat them as a
sanity check on the scanner's shape, not as performance.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from apexflow.models import BacktestResult
from apexflow.providers import get_provider
from apexflow.scanners import ALL_SCANNERS

log = logging.getLogger(__name__)


#: Scanners whose signal can be reconstructed from daily OHLCV alone.
#: The rest depend on an options chain or a flow feed as it stood on the
#: signal date, which no free provider archives.
SUPPORTED_SCANNERS = frozenset({"pre-breakout", "momentum", "squeeze"})

#: Why each unsupported scanner cannot be replayed, surfaced in the error.
UNSUPPORTED_REASON = {
    "earnings": "needs the historical option chain and IV as of each signal "
                "date; free providers archive neither",
    "options-flow": "needs historical volume/open-interest per strike and "
                    "trade-level prints, which are not archived",
    "radar": "composites the options-driven layers above, so it inherits "
             "their data requirements",
}


class Backtester:
    """Replays a scanner's daily signal against historical bars.

    An approximation: today's scanner logic is applied to past data, with no
    costs, no slippage, and a universe that contains only names still listed
    today. A sanity check on shape, not a performance estimate.
    """

    def __init__(self):
        self.provider = get_provider()

    def run(self, scanner_name: str, universe: list[str],
            lookback_days: int = 90, hold_days: int = 5) -> BacktestResult:
        if scanner_name not in ALL_SCANNERS:
            raise ValueError(f"Unknown scanner: {scanner_name}")
        if scanner_name not in SUPPORTED_SCANNERS:
            reason = UNSUPPORTED_REASON.get(
                scanner_name, "its inputs are not reconstructable point-in-time")
            raise ValueError(
                f"Cannot backtest '{scanner_name}': {reason}. "
                f"Backtestable scanners: {', '.join(sorted(SUPPORTED_SCANNERS))}. "
                f"For the squeeze *score* (rather than the boolean signal), use "
                f"`python main.py backtest-squeeze`."
            )

        returns: list[float] = []
        wins = losses = 0
        signals_tested = 0

        for sym in universe:
            try:
                df = self.provider.history(sym, period=f"{lookback_days + hold_days + 60}d", interval="1d")
            except Exception:
                continue
            if df is None or len(df) < 60:
                continue
            close = df["Close"]

            for i in range(60, len(df) - hold_days):
                window = df.iloc[: i + 1]
                if not self._would_fire(scanner_name, window):
                    continue
                entry = close.iloc[i]
                exit_ = close.iloc[i + hold_days]
                if entry <= 0:
                    continue
                ret = (exit_ / entry - 1) * 100
                returns.append(float(ret))
                signals_tested += 1
                if ret > 0:
                    wins += 1
                else:
                    losses += 1

        if not returns:
            return BacktestResult(
                scanner=scanner_name, signals_tested=0,
                wins=0, losses=0, avg_return_pct=0,
                median_return_pct=0, win_rate=0, sharpe=0,
                lookback_days=lookback_days, hold_days=hold_days,
            )

        rets = np.array(returns)
        mean_r = float(rets.mean())
        std_r = float(rets.std(ddof=0))
        ann = float(np.sqrt(252 / max(hold_days, 1)))

        sharpe = (mean_r / std_r * ann) if std_r > 0 else 0.0

        downside = rets[rets < 0]
        ddev = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
        sortino = (mean_r / ddev * ann) if ddev > 0 else 0.0

        # Equity curve: compounded
        equity = [100.0]
        for r in rets:
            equity.append(equity[-1] * (1 + r / 100.0))
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = float(dd.min())

        annualized_return = ((eq[-1] / eq[0]) ** (252 / max(len(rets) * hold_days, 1)) - 1) * 100
        calmar = (annualized_return / abs(max_dd)) if max_dd < 0 else 0.0

        wins_sum = float(rets[rets > 0].sum())
        loss_sum = float(rets[rets < 0].sum())
        profit_factor = (wins_sum / abs(loss_sum)) if loss_sum < 0 else 0.0

        best = float(rets.max()); worst = float(rets.min())

        # Streaks
        longest_w = longest_l = cur_w = cur_l = 0
        for r in rets:
            if r > 0:
                cur_w += 1; cur_l = 0
                longest_w = max(longest_w, cur_w)
            else:
                cur_l += 1; cur_w = 0
                longest_l = max(longest_l, cur_l)

        return BacktestResult(
            scanner=scanner_name,
            signals_tested=signals_tested,
            wins=wins, losses=losses,
            avg_return_pct=mean_r,
            median_return_pct=float(np.median(rets)),
            win_rate=wins / signals_tested,
            sharpe=float(sharpe),
            sortino=float(sortino),
            max_drawdown_pct=max_dd,
            calmar=float(calmar),
            profit_factor=float(profit_factor),
            best_trade_pct=best,
            worst_trade_pct=worst,
            longest_win_streak=longest_w,
            longest_loss_streak=longest_l,
            equity_curve=[float(x) for x in eq],
            returns=[float(x) for x in rets],
            lookback_days=lookback_days, hold_days=hold_days,
        )

    def _would_fire(self, scanner_name: str, window: pd.DataFrame) -> bool:
        """Historical proxy for a scanner's daily-data signal.

        These are *proxies*, not the live scanner code: the live scanners
        read an options chain and a quote feed that no historical replay can
        supply. Each condition below is the price/volume shadow of the real
        rule, which is why only the three price-driven scanners are
        supported at all.
        """
        from apexflow.analytics import indicators as ind
        close = window["Close"]; high = window["High"]; low = window["Low"]; vol = window["Volume"]

        if scanner_name == "pre-breakout":
            if len(close) < 60:
                return False
            bb = ind.bollinger(close, 20).iloc[-1]
            sq = bool(ind.squeeze_on(close, high, low).iloc[-1])
            dryup = ind.volume_dryup(vol)
            return bool(sq and dryup and bb["bb_bw"] < 0.10)

        if scanner_name == "momentum":
            if len(close) < 30:
                return False
            move = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            r = ind.rvol(vol, 30).iloc[-1] or 0
            return bool(abs(move) > 3 and r > 3)

        if scanner_name == "squeeze":
            # NB: this is a price/volume proxy — a sharp up-move on heavy
            # volume — and NOT `analytics.squeeze.squeeze_score`, whose
            # short-interest inputs are not available point-in-time. Do not
            # read a result here as evidence about the composite score;
            # `squeeze_backtest.py` is the module that evaluates that.
            if len(close) < 30:
                return False
            move = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            r = ind.rvol(vol, 30).iloc[-1] or 0
            return bool(move > 5 and r > 4)

        # Unreachable: run() rejects unsupported scanners before we get here.
        raise ValueError(f"no historical proxy for scanner {scanner_name!r}")
