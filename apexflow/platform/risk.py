"""Position sizing and risk arithmetic.

What this is
------------
The unglamorous half of a trading tool, and the half that decides whether
you survive being wrong. Everything here is arithmetic on numbers *you*
supply — account equity, how much you are willing to lose, where you would
admit the idea failed. Nothing in this module reads market data, forms a
view, or suggests a trade.

That separation is deliberate. The analytics in this project describe market
structure and have no demonstrated predictive edge — the one model that was
formally tested showed no rank information (see ``docs/methodology.md`` §10).
Sizing is the part that works regardless: it is true by construction rather
than by backtest.

Why fixed-fractional
--------------------
Risk a constant fraction of equity per trade and position size falls out of
the stop distance:

    shares = (equity * risk_fraction) / |entry - stop|

Two properties make this the default worth having:

* **You cannot be ruined by one trade.** Loss is capped at the fraction by
  construction, before any judgement about the trade enters.
* **Size scales with conviction only through the stop.** A tight stop buys
  more size for the same dollar risk. That is the correct relationship, and
  it is the one people invert when sizing by gut.

The cost is that it says nothing about whether the trade is *good*. It
cannot: that is what the stop is for.

On options specifically
-----------------------
A long option's maximum loss is the premium, so "stop distance" is often
the whole position. Sizing by contracts against premium at risk is
supported, and the important caveat is stated in the output rather than
buried: a stop on the *underlying* does not cap an option loss, because
gaps, vol crush and time decay can all take the premium through it.

Kelly
-----
:func:`kelly_fraction` is included because people ask for it, and clamped
hard because full Kelly is close to unusable in practice: it assumes you
know your edge exactly, and being wrong about the edge is far more likely
than the edge existing. The output carries that warning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["PositionSize", "size_by_stop", "size_options_by_premium",
           "kelly_fraction", "r_multiple", "MAX_SANE_RISK_FRACTION"]

#: Risking more than this per trade is almost always a mistake, so the
#: sizer refuses to compute silently past it and says why.
MAX_SANE_RISK_FRACTION = 0.05          # 5% of equity on one trade

CONTRACT_MULTIPLIER = 100


@dataclass
class PositionSize:
    """A sizing answer, with everything needed to sanity-check it."""
    quantity: int = 0
    unit: str = "shares"               # shares | contracts
    entry: float = 0.0
    stop: float | None = None
    risk_amount: float = 0.0           # dollars at risk if the stop holds
    risk_fraction: float = 0.0
    position_value: float = 0.0
    position_pct_of_equity: float = 0.0
    stop_distance: float = 0.0
    stop_distance_pct: float = 0.0
    warnings: list[str] = field(default_factory=list)
    ok: bool = True
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity, "unit": self.unit, "entry": self.entry,
            "stop": self.stop, "risk_amount": round(self.risk_amount, 2),
            "risk_fraction": self.risk_fraction,
            "position_value": round(self.position_value, 2),
            "position_pct_of_equity": round(self.position_pct_of_equity, 4),
            "stop_distance": round(self.stop_distance, 4),
            "stop_distance_pct": round(self.stop_distance_pct, 4),
            "warnings": self.warnings, "ok": self.ok, "detail": self.detail,
        }


def _validate(equity: float, risk_fraction: float) -> list[str]:
    problems = []
    if equity <= 0:
        problems.append("Account equity must be positive.")
    if risk_fraction <= 0:
        problems.append("Risk per trade must be positive.")
    elif risk_fraction > MAX_SANE_RISK_FRACTION:
        problems.append(
            f"Risking {risk_fraction:.1%} of equity on one trade is beyond the "
            f"{MAX_SANE_RISK_FRACTION:.0%} this tool will size for. A handful of "
            f"consecutive losses at that rate is unrecoverable.")
    return problems


def size_by_stop(equity: float, risk_fraction: float, entry: float,
                 stop: float, unit: str = "shares",
                 max_position_fraction: float = 1.0) -> PositionSize:
    """Fixed-fractional size from a stop distance.

    ``risk_fraction`` is a decimal (0.01 = 1% of equity). ``unit`` is
    ``shares`` or ``contracts``; contracts multiply by 100.

    ``max_position_fraction`` caps notional exposure separately from risk —
    a very tight stop can otherwise size into a position far larger than the
    account, which is fine on paper and impossible in practice.
    """
    out = PositionSize(entry=float(entry), stop=float(stop), unit=unit,
                       risk_fraction=float(risk_fraction))
    problems = _validate(equity, risk_fraction)
    if entry <= 0:
        problems.append("Entry price must be positive.")
    if stop is None or stop <= 0:
        problems.append("Stop price must be positive.")
    if problems:
        out.ok = False
        out.warnings = problems
        out.detail = "Cannot size: " + " ".join(problems)
        return out

    distance = abs(float(entry) - float(stop))
    if distance <= 0:
        out.ok = False
        out.detail = ("Entry and stop are the same price, so risk per unit is "
                      "zero and no finite size caps the loss.")
        out.warnings = [out.detail]
        return out

    out.stop_distance = distance
    out.stop_distance_pct = distance / entry

    multiplier = CONTRACT_MULTIPLIER if unit == "contracts" else 1
    budget = equity * risk_fraction
    raw_qty = budget / (distance * multiplier)
    qty = int(math.floor(raw_qty))

    if qty < 1:
        out.ok = False
        out.quantity = 0
        out.detail = (
            f"A {risk_fraction:.2%} risk budget of ${budget:,.2f} does not "
            f"cover one {unit[:-1]} at a ${distance:,.2f} stop distance "
            f"(${distance * multiplier:,.2f} of risk per {unit[:-1]}). Either "
            f"the stop is too wide for this account or the trade is too big.")
        out.warnings.append(out.detail)
        return out

    # Notional cap — risk and exposure are different constraints.
    value = qty * entry * multiplier
    cap = equity * max_position_fraction
    if value > cap and cap > 0:
        qty = int(math.floor(cap / (entry * multiplier)))
        value = qty * entry * multiplier
        out.warnings.append(
            f"Size reduced to {qty} to keep notional within "
            f"{max_position_fraction:.0%} of equity. The stop was tight enough "
            f"to justify a larger position than the account can hold.")
        if qty < 1:
            out.ok = False
            out.quantity = 0
            out.detail = "Notional cap leaves room for less than one unit."
            return out

    out.quantity = qty
    out.position_value = value
    out.position_pct_of_equity = value / equity if equity else 0.0
    out.risk_amount = qty * distance * multiplier

    if out.stop_distance_pct < 0.005:
        out.warnings.append(
            f"The stop is {out.stop_distance_pct:.2%} away. Normal noise will "
            f"take you out of this before the idea has a chance to work.")
    if unit == "contracts":
        out.warnings.append(
            "This sizes by a stop on the option's own price. A stop on the "
            "underlying does not cap an option loss — a gap, a vol crush or "
            "simple decay can move the premium through it.")

    out.detail = (
        f"{qty} {unit} at ${entry:,.2f}, stop ${stop:,.2f}. "
        f"Risking ${out.risk_amount:,.2f} ({out.risk_amount / equity:.2%} of "
        f"${equity:,.0f}) over a ${distance:,.2f} stop.")
    return out


def size_options_by_premium(equity: float, risk_fraction: float,
                            premium: float,
                            max_loss_fraction_of_premium: float = 1.0) -> PositionSize:
    """Size long options where the max loss is the premium itself.

    ``max_loss_fraction_of_premium`` is how much of the premium you actually
    expect to lose in the bad case — 1.0 treats the whole thing as at risk,
    which is the honest default for a long option held through an event.
    """
    out = PositionSize(entry=float(premium), stop=None, unit="contracts",
                       risk_fraction=float(risk_fraction))
    problems = _validate(equity, risk_fraction)
    if premium <= 0:
        problems.append("Premium must be positive.")
    if problems:
        out.ok = False
        out.warnings = problems
        out.detail = "Cannot size: " + " ".join(problems)
        return out

    per_contract_risk = premium * CONTRACT_MULTIPLIER * max_loss_fraction_of_premium
    budget = equity * risk_fraction
    qty = int(math.floor(budget / per_contract_risk))

    if qty < 1:
        out.ok = False
        out.detail = (
            f"A {risk_fraction:.2%} budget of ${budget:,.2f} does not cover one "
            f"contract at ${premium:,.2f} (${per_contract_risk:,.2f} at risk).")
        out.warnings.append(out.detail)
        return out

    out.quantity = qty
    out.position_value = qty * premium * CONTRACT_MULTIPLIER
    out.position_pct_of_equity = out.position_value / equity
    out.risk_amount = qty * per_contract_risk
    out.stop_distance = premium
    out.stop_distance_pct = 1.0
    out.warnings.append(
        "Max loss on a long option is the entire premium. This assumes you "
        "are willing to lose "
        f"{max_loss_fraction_of_premium:.0%} of it.")
    out.detail = (
        f"{qty} contracts at ${premium:,.2f} = ${out.position_value:,.2f} of "
        f"premium, ${out.risk_amount:,.2f} at risk "
        f"({out.risk_amount / equity:.2%} of ${equity:,.0f}).")
    return out


def r_multiple(entry: float, stop: float, target: float) -> dict:
    """Reward-to-risk for a trade, in units of the risk taken (R).

    Says nothing about whether the target is *likely* — only what the payoff
    would be if both levels were reached. The breakeven win rate is the
    useful number: the frequency at which this R just breaks even.
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return {"r": 0.0, "risk": 0.0, "reward": reward,
                "breakeven_win_rate": None,
                "detail": "Zero stop distance — R is undefined."}
    r = reward / risk
    be = 1.0 / (1.0 + r) if r > 0 else None
    return {
        "r": round(r, 3), "risk": round(risk, 4), "reward": round(reward, 4),
        "breakeven_win_rate": round(be, 4) if be is not None else None,
        "detail": (f"{r:.2f}R. You would need to be right more than "
                   f"{be:.1%} of the time for this to break even before costs."
                   if be is not None else "R undefined."),
    }


def kelly_fraction(win_rate: float, win_loss_ratio: float,
                   cap: float = MAX_SANE_RISK_FRACTION) -> dict:
    """Kelly stake, and a blunt warning about using it.

    ``f* = W - (1-W)/R``. Returned capped, because full Kelly assumes the
    edge is known exactly. Overestimating it — far more common than having
    one — produces catastrophic sizing, and the drawdowns even at a correct
    full Kelly are larger than almost anyone tolerates.
    """
    if not (0 < win_rate < 1) or win_loss_ratio <= 0:
        return {"kelly": 0.0, "half_kelly": 0.0, "capped": 0.0,
                "detail": "Needs a win rate strictly between 0 and 1 and a "
                          "positive win/loss ratio."}
    f = win_rate - (1 - win_rate) / win_loss_ratio
    f = max(f, 0.0)
    capped = min(f, cap)
    return {
        "kelly": round(f, 4),
        "half_kelly": round(f / 2, 4),
        "capped": round(capped, 4),
        "detail": (
            f"Full Kelly is {f:.1%} of equity. That number assumes you know "
            f"your edge exactly; if the estimate is optimistic, Kelly sizing "
            f"is ruinous rather than merely bad. Most practitioners use half "
            f"Kelly or less — shown capped at {cap:.0%} here."
            if f > 0 else
            "Kelly is zero or negative: at this win rate and payoff there is "
            "no edge to stake."),
    }
