"""Black-Scholes-Merton Greeks for European options.

Conventions:
  - Time `t` is in years (calendar 365, matching quoted IV).
  - Risk-free rate `r` in decimal (e.g. 0.04 = 4%).
  - Dividend yield `q` in decimal (defaults to 0).
  - IV `sigma` in decimal (e.g. 0.35 = 35%).

Output Greeks:
  delta : dC/dS                 per +$1 in spot
  gamma : d2C/dS2               per +$1 in spot
  vega  : dC/dsigma * 0.01      per +1 vol point (so vega is "per 1%")
  theta : dC/dt / 365           per calendar day
  rho   : dC/dr * 0.01          per +1 percentage point of rate

American exercise is not modelled. For US equity options that is a good
approximation for calls on non-dividend payers and a poor one for deep ITM
puts, where early exercise carries real value - treat put deltas near the
money-forward boundary as indicative rather than exact.
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Literal

from scipy.stats import norm

from .timeutil import years_to_expiry as _years_to_expiry

__all__ = ["years_to_expiry", "greeks", "bs_price", "implied_vol",
           "implied_vol_newton", "iv_rank", "iv_percentile"]


def years_to_expiry(expiry: str, now: datetime | None = None) -> float:
    """Calendar years until the expiry's 16:00 ET close.

    Delegates to :mod:`analytics.timeutil`. This used to be
    ``(exp - now).days + 0.5`` over 365, which disagreed with the version in
    ``gex.py`` - the same contract had two different times to expiry
    depending on which module reached it first.
    """
    return _years_to_expiry(expiry, now=now)


def _d1_d2(S: float, K: float, t: float, sigma: float, r: float = 0.04, q: float = 0.0):
    if S <= 0 or K <= 0 or t <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def bs_price(spot: float, strike: float, t: float, iv: float,
             right: Literal["C", "P"] = "C", r: float = 0.04, q: float = 0.0) -> float:
    """European option price under Black-Scholes-Merton."""
    d1, d2 = _d1_d2(spot, strike, t, iv, r, q)
    if d1 is None:
        intrinsic = spot - strike if right == "C" else strike - spot
        return max(intrinsic, 0.0)
    if right == "C":
        return float(spot * math.exp(-q * t) * norm.cdf(d1)
                     - strike * math.exp(-r * t) * norm.cdf(d2))
    return float(strike * math.exp(-r * t) * norm.cdf(-d2)
                 - spot * math.exp(-q * t) * norm.cdf(-d1))


def greeks(
    spot: float, strike: float, t: float, iv: float,
    right: Literal["C", "P"] = "C", r: float = 0.04, q: float = 0.0,
) -> dict:
    """Return delta/gamma/vega/theta/rho for a single option.

    Returns zeros if any input is invalid (still useful as defaults in tables).
    """
    out = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(spot, strike, t, iv, r, q)
    if d1 is None:
        return out
    sqrt_t = math.sqrt(t)
    discount_q = math.exp(-q * t)
    discount_r = math.exp(-r * t)

    if right == "C":
        delta = discount_q * norm.cdf(d1)
        theta_y = (
            -spot * norm.pdf(d1) * iv * discount_q / (2 * sqrt_t)
            - r * strike * discount_r * norm.cdf(d2)
            + q * spot * discount_q * norm.cdf(d1)
        )
        rho_pct = strike * t * discount_r * norm.cdf(d2) * 0.01
    else:
        delta = -discount_q * norm.cdf(-d1)
        theta_y = (
            -spot * norm.pdf(d1) * iv * discount_q / (2 * sqrt_t)
            + r * strike * discount_r * norm.cdf(-d2)
            - q * spot * discount_q * norm.cdf(-d1)
        )
        rho_pct = -strike * t * discount_r * norm.cdf(-d2) * 0.01

    gamma_v = discount_q * norm.pdf(d1) / (spot * iv * sqrt_t)
    vega_pct = spot * discount_q * norm.pdf(d1) * sqrt_t * 0.01  # per 1 vol point
    theta_day = theta_y / 365.0

    out.update({
        "delta": float(delta),
        "gamma": float(gamma_v),
        "vega":  float(vega_pct),
        "theta": float(theta_day),
        "rho":   float(rho_pct),
    })
    return out


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------
def _no_arb_bounds(spot: float, strike: float, t: float, right: str,
                   r: float, q: float) -> tuple[float, float]:
    """(lower, upper) no-arbitrage price bounds for a European option."""
    fwd = spot * math.exp(-q * t)
    pv_k = strike * math.exp(-r * t)
    if right == "C":
        return max(fwd - pv_k, 0.0), fwd
    return max(pv_k - fwd, 0.0), pv_k


def implied_vol(price: float, spot: float, strike: float, t: float,
                right: Literal["C", "P"] = "C", r: float = 0.04, q: float = 0.0,
                tol: float = 1e-6, max_iter: int = 100) -> float | None:
    """Solve for implied volatility. Returns None when no solution exists.

    Newton-Raphson converges in a handful of iterations near the money, but
    it is unreliable exactly where option chains are messiest: deep OTM
    contracts, where vega is nearly zero and a single step can overshoot
    into negative vol, and stale/wide quotes that sit outside the no-arb
    band. So:

    1. Reject prices outside the no-arbitrage bounds up front - those have
       no implied vol, and a solver that returns one is inventing data.
    2. Seed with the Brenner-Subrahmanyam approximation
       ``sigma ~ sqrt(2*pi/t) * price / spot`` rather than a flat 0.5, which
       lands close enough that Newton usually converges in 2-3 steps.
    3. Fall back to bisection on [1e-4, 5.0] whenever Newton leaves the
       bracket or stalls. Bisection is slower but cannot diverge, and the
       price is monotone in vol so the bracket is guaranteed to contain the
       root when one exists.

    The old implementation always started at 0.5, had no bounds check, and
    returned its last iterate on non-convergence - so a garbage quote came
    back as a confident-looking number rather than None.
    """
    if price is None or price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return None
    lo_px, hi_px = _no_arb_bounds(spot, strike, t, right, r, q)
    # Allow a hair of slack for rounding in the quote feed.
    if price < lo_px - 1e-8 or price > hi_px + 1e-8:
        return None
    if price <= lo_px + 1e-12:
        return None       # at intrinsic - implied vol is zero/undefined

    lo, hi = 1e-4, 5.0
    if bs_price(spot, strike, t, hi, right, r, q) < price:
        return None       # not reachable even at 500% vol

    # Brenner-Subrahmanyam seed, clamped into the bracket.
    sigma = min(max(math.sqrt(2 * math.pi / t) * price / spot, lo), hi)

    for _ in range(max_iter):
        theo = bs_price(spot, strike, t, sigma, right, r, q)
        diff = theo - price
        if abs(diff) < tol:
            return float(sigma)
        # Maintain the bracket from every evaluation, so the bisection
        # fallback below is always working with a valid interval.
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        d1, _ = _d1_d2(spot, strike, t, sigma, r, q)
        vega = spot * math.exp(-q * t) * norm.pdf(d1) * math.sqrt(t) if d1 is not None else 0.0
        if vega > 1e-8:
            step = sigma - diff / vega
            if lo < step < hi:
                sigma = step
                continue
        sigma = 0.5 * (lo + hi)      # Newton unusable here - bisect instead
        if hi - lo < 1e-10:
            break
    return float(sigma) if abs(bs_price(spot, strike, t, sigma, right, r, q) - price) < 1e-3 else None


def implied_vol_newton(price: float, spot: float, strike: float, t: float,
                       right: Literal["C", "P"] = "C", r: float = 0.04, q: float = 0.0,
                       tol: float = 1e-4, max_iter: int = 60) -> float | None:
    """Backwards-compatible alias for :func:`implied_vol`."""
    return implied_vol(price, spot, strike, t, right, r, q, tol=tol, max_iter=max_iter)


# ---------------------------------------------------------------------------
# IV rank / percentile
# ---------------------------------------------------------------------------
def iv_rank(current_iv: float, iv_history: list[float]) -> float | None:
    """Where current IV sits in its own 52-week range, 0-100.

    ``(current - min) / (max - min) * 100``. Returns None on empty history
    rather than 0.0 - "no data" and "at the bottom of the range" are very
    different statements and the caller needs to be able to tell them apart.
    """
    hist = [v for v in (iv_history or []) if v is not None and v > 0]
    if not hist or current_iv <= 0:
        return None
    lo, hi = min(hist), max(hist)
    if hi - lo < 1e-6:
        return 50.0
    return float(max(0.0, min(100.0, (current_iv - lo) / (hi - lo) * 100)))


def iv_percentile(current_iv: float, iv_history: list[float]) -> float | None:
    """Fraction of history strictly below current IV, 0-100. None on empty."""
    hist = [v for v in (iv_history or []) if v is not None and v > 0]
    if not hist or current_iv <= 0:
        return None
    below = sum(1 for v in hist if v < current_iv)
    return float(below / len(hist) * 100)
