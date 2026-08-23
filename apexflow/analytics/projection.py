"""Forward price projection - expected move, touch / above probabilities.

Models price as geometric Brownian motion under the risk-neutral measure::

    dS/S = (r - q) dt + sigma dW
    ln(S_t / S_0) ~ Normal( (r - q - sigma^2/2) t,  sigma^2 t )

Everything here is closed-form and exact under that model. For anything the
model cannot express - fat tails, an earnings jump, path-dependent payoffs -
use ``analytics.montecarlo``, which simulates instead and is validated
against the functions below.

Provides:
  expected_move(spot, t, iv)            - 1 sigma dollar move at the horizon
  above_prob(spot, K, t, iv)            - P(S_T > K) at the horizon
  touch_prob(spot, K, t, iv)            - P(price touches K at any time <= T)
  projection_cone(spot, t, iv, n_steps) - quantile cone (P5/P25/P50/P75/P95)

A note on the two probabilities: *above* is a terminal question and *touch*
is a path question. Touch is always the larger of the two, and roughly twice
as large for a driftless barrier - a strike price is far more likely to be
tagged intraday than to be closed beyond.
"""
from __future__ import annotations
import math
from typing import Dict, List

from scipy.stats import norm


def expected_move(spot: float, t: float, iv: float) -> float:
    """1 sigma expected dollar move over horizon `t` years given annualized vol `iv`.

    This is the conventional ``S * sigma * sqrt(t)`` shorthand: the standard
    deviation of the *arithmetic* return to first order. It differs from the
    exact lognormal standard deviation by O(sigma^2 t), which is under 1% of
    the move for a 30-day horizon at 30 vol.
    """
    if spot <= 0 or t <= 0 or iv <= 0:
        return 0.0
    return float(spot * iv * math.sqrt(t))


def lognormal_std(spot: float, t: float, iv: float, r: float = 0.04, q: float = 0.0) -> float:
    """Exact standard deviation of S_T under GBM.

    Used to cross-check ``expected_move`` and to compare against the
    simulated terminal standard deviation from ``montecarlo.simulate_cone``.
    """
    if spot <= 0 or t <= 0 or iv <= 0:
        return 0.0
    var = (spot ** 2) * math.exp(2 * (r - q) * t) * (math.exp(iv * iv * t) - 1.0)
    return float(math.sqrt(max(var, 0.0)))


def above_prob(spot: float, strike: float, t: float, iv: float,
               r: float = 0.04, q: float = 0.0) -> float:
    """Risk-neutral probability that S_T > K at horizon t (years).

    This is ``N(d2)`` - the same quantity as a digital call's undiscounted
    price. It is *not* the option's delta; delta is N(d1), which is larger.
    """
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        return 0.5
    d2 = (math.log(spot / strike) + (r - q - 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return float(norm.cdf(d2))


def touch_prob(spot: float, strike: float, t: float, iv: float,
               r: float = 0.04, q: float = 0.0) -> float:
    """P(the price touches barrier K at any time in [0, t]) under GBM.

    Exact first-passage probability for Brownian motion with drift. Writing
    ``X_t = ln(S_t/S_0) = mu*t + sigma*W_t`` with ``mu = r - q - sigma^2/2``
    and ``b = ln(K/S_0)``:

    up barrier (b > 0)::

        P(max X >= b) = N((mu*t - b)/(sigma*sqrt(t)))
                        + exp(2*mu*b/sigma^2) * N((-b - mu*t)/(sigma*sqrt(t)))

    down barrier (b < 0)::

        P(min X <= b) = N((b - mu*t)/(sigma*sqrt(t)))
                        + exp(2*mu*b/sigma^2) * N((b + mu*t)/(sigma*sqrt(t)))

    The older implementation used the reflection shortcut ``2 * P(S_T beyond
    K)``, which is only correct at exactly zero drift and overstates the
    probability of touching a downside barrier in a positive-drift regime
    (and understates the upside one). The difference is a couple of
    percentage points at a 1-week horizon and grows with the horizon; for a
    barrier 5% away over 3 months at 30 vol the shortcut is off by ~1.5 pts.
    ``montecarlo.validate()`` checks this function against simulated path
    maxima.
    """
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        return 0.5
    if abs(strike - spot) / spot < 1e-9:
        return 1.0

    b = math.log(strike / spot)
    mu = r - q - 0.5 * iv * iv
    sig_t = iv * math.sqrt(t)
    # exp(2*mu*b/sigma^2) can overflow for a distant barrier; the paired CDF
    # term underflows faster, so clamp the exponent rather than the product.
    expo = 2.0 * mu * b / (iv * iv)
    expo = max(min(expo, 700.0), -700.0)
    scale = math.exp(expo)

    if b > 0:   # up barrier
        p = norm.cdf((mu * t - b) / sig_t) + scale * norm.cdf((-b - mu * t) / sig_t)
    else:       # down barrier
        p = norm.cdf((b - mu * t) / sig_t) + scale * norm.cdf((b + mu * t) / sig_t)
    return float(min(max(p, 0.0), 1.0))


def touch_prob_reflection(spot: float, strike: float, t: float, iv: float,
                          r: float = 0.04, q: float = 0.0) -> float:
    """The old ``2 x tail`` approximation, kept so the error can be quantified.

    Correct only at zero drift. ``tests/test_projection.py`` pins the size of
    its disagreement with the exact formula.
    """
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        return 0.5
    p_above = above_prob(spot, strike, t, iv, r, q)
    tail = p_above if strike > spot else 1.0 - p_above
    return float(min(1.0, 2.0 * tail))


def projection_cone(spot: float, t: float, iv: float, n_steps: int = 30,
                    r: float = 0.04, q: float = 0.0) -> List[Dict]:
    """Cone of terminal-price quantiles (P5/P25/P50/P75/P95) over [0, t].

    Each step is the *marginal* distribution of S at that time, not a
    simultaneous confidence band for the whole path - 90% of paths finish
    inside the P5/P95 envelope at any given time, but far fewer stay inside
    it the entire way. ``montecarlo.simulate_cone`` produces the same
    marginals by simulation and can also answer the path question.
    """
    if spot <= 0 or t <= 0 or iv <= 0 or n_steps <= 0:
        return []
    out = []
    for i in range(1, n_steps + 1):
        ti = t * i / n_steps
        sig_t = iv * math.sqrt(ti)
        drift = (r - q - 0.5 * iv * iv) * ti
        median = spot * math.exp(drift)
        out.append({
            "frac": ti / t,
            "t_years": ti,
            "p5":  float(spot * math.exp(drift - 1.6449 * sig_t)),
            "p25": float(spot * math.exp(drift - 0.6745 * sig_t)),
            "p50": float(median),
            "p75": float(spot * math.exp(drift + 0.6745 * sig_t)),
            "p95": float(spot * math.exp(drift + 1.6449 * sig_t)),
        })
    return out


def horizon_years(horizon_days: float) -> float:
    """Convert calendar-day horizon to years (calendar 365)."""
    return max(horizon_days, 0.001) / 365.0
