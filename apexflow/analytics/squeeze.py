"""Composite short-squeeze pressure score (0-100).

What this measures
------------------
How *fragile* the short side of a name is, given the inputs below. It is a
description of positioning, not a prediction: a 90 means the ingredients for
a forced-covering move are unusually present, not that one will happen.
Squeezes need a catalyst, and nothing here can see one.

Scoring shape
-------------
Each axis maps its input through a monotone piecewise-linear curve onto a
capped contribution. The curves are anchored at the same breakpoints the
step-function version used, but they interpolate *between* them rather than
jumping. That matters: under the old step buckets a name at 9.9% short
float scored 5 and one at 10.1% scored 15, so a rounding difference in a
vendor feed moved the composite by ten points and reordered the leaderboard.
Interpolating removes the cliff without changing the score at any anchor.

Weights (max contribution):

===================  ===  =========================================
Axis                 Max  Rationale
===================  ===  =========================================
short % of float     35   The core constraint - how much stock has
                          to be bought back relative to what trades.
days to cover        20   How long that buying takes at normal
                          volume. The squeeze's duration term.
borrow rate          15   Cost of staying short; high fees mean
                          holders are already being squeezed on P&L.
IV / HV ratio        10   Options market pricing a move the tape
                          has not delivered yet.
relative volume      12   Whether anything is actually happening.
float size           10   Small floats move further on equal flow.
accumulation          8   Directional tape confirmation.
===================  ===  =========================================

The weights are judgement, not fit - no backtest has calibrated them yet.
Treat the ranking as more meaningful than the absolute level, and see
``docs/methodology.md`` for what would be needed to validate them.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

__all__ = ["SqueezeInputs", "squeeze_score", "COMPONENT_MAX", "interpolate"]


@dataclass
class SqueezeInputs:
    short_pct_float: float = 0.0      # 0.20 = 20% of float short
    days_to_cover: float = 0.0        # short interest / average daily volume
    borrow_rate: float = 0.0          # annualized, 0.50 = 50%
    iv_hv_ratio: float = 1.0          # IV / 30d HV
    rvol: float = 1.0                 # current relative volume
    float_shares: float = 0.0         # shares
    price: float = 0.0
    accumulation_score: float = 0.0   # -1..+1, distribution..accumulation


#: Maximum contribution of each component, so callers can render a share-of-max.
COMPONENT_MAX = {
    "short_float": 35.0, "days_to_cover": 20.0, "borrow": 15.0,
    "iv_hv": 10.0, "rvol": 12.0, "float_size": 10.0, "accumulation": 8.0,
}


def interpolate(value: float, breakpoints: Sequence[float],
                scores: Sequence[float]) -> float:
    """Monotone piecewise-linear map from `value` onto `scores`.

    ``breakpoints`` must be ascending. Below the first breakpoint the result
    is ``scores[0]``; above the last it is ``scores[-1]``; in between it
    interpolates linearly. At any breakpoint the result equals the old step
    function's value there, so scores stay comparable with previously
    logged signals.
    """
    if value is None or value != value:      # None or NaN
        return float(scores[0])
    if value <= breakpoints[0]:
        return float(scores[0])
    for i in range(1, len(breakpoints)):
        if value <= breakpoints[i]:
            lo_b, hi_b = breakpoints[i - 1], breakpoints[i]
            lo_s, hi_s = scores[i - 1], scores[i]
            if hi_b <= lo_b:
                return float(hi_s)
            frac = (value - lo_b) / (hi_b - lo_b)
            return float(lo_s + frac * (hi_s - lo_s))
    return float(scores[-1])


def squeeze_score(inp: SqueezeInputs) -> tuple[float, dict]:
    """Returns (score 0-100, per-component contributions).

    Components sum to at most 110 before the cap; the cap is deliberate, so
    a name that maxes six of seven axes still reads as 100 rather than being
    dragged down by the seventh.
    """
    sf = interpolate(inp.short_pct_float, [0.05, 0.10, 0.20, 0.30, 0.50, 1.0],
                     [0,    5,    15,   25,   30,   35])
    dtc = interpolate(inp.days_to_cover, [1, 2, 3, 5, 8, 100],
                      [0, 3, 8, 12, 16, 20])
    br = interpolate(inp.borrow_rate, [0.05, 0.20, 0.50, 1.0, 5.0, 100],
                     [0,    3,    6,   10,  13,   15])
    ivhv = interpolate(inp.iv_hv_ratio, [0.8, 1.0, 1.3, 1.6, 2.0, 100],
                       [0,    1,   3,   6,   8,   10])
    rv = interpolate(inp.rvol, [1.0, 1.5, 2.5, 4.0, 7.0, 100],
                     [0,   2,   5,   8,   10,  12])
    # Smaller float = more explosive; tiers in millions of shares. Note the
    # scores descend, so the interpolation runs downhill — a 30M-share float
    # now lands between the 10M and 50M anchors instead of snapping to one.
    # A missing float is treated as "very large" so it cannot score points.
    floatm = inp.float_shares / 1_000_000 if inp.float_shares else 1e9
    fl = interpolate(floatm, [10, 50, 150, 500, 2000, 1e9],
                     [10,  7,   4,   2,   1,    0])
    acc = max(0.0, min(inp.accumulation_score, 1.0)) * COMPONENT_MAX["accumulation"]

    components = {
        "short_float": sf, "days_to_cover": dtc, "borrow": br,
        "iv_hv": ivhv, "rvol": rv, "float_size": fl, "accumulation": acc,
    }
    return min(sum(components.values()), 100.0), components
