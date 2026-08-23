"""Robust ATM implied vol, term structure, and skew from a raw chain.

The problem this solves
-----------------------
The obvious way to get "the" implied vol for a ticker is to take the strike
nearest spot and read its ``impliedVolatility``. That is what most of this
repo used to do, and on real data it is a trap. From the frozen SPY snapshot
in ``data/demo/``, the six strikes closest to spot on the front (0DTE)
expiry reported:

    766 -> 2.9%    765 -> 5.2%    767 -> 2.1%
    764 -> 7.6%    768 -> 3.3%    763 -> 8.8%

Those numbers are not a volatility smile. They are what a vendor's solver
returns when it is handed a contract worth one or two cents with a
penny-wide spread and hours of life left: vega is nearly zero, so the
inversion from price to vol is numerically hopeless and any rounding in the
last trade throws the answer around by a factor of four. Picking "the
nearest strike" from that set is a coin flip between 2.1% and 8.8%, and
whichever one you get is then propagated into the expected move, the cone,
the Monte Carlo, and every probability on the page.

What this module does instead
-----------------------------
1. **Filter before averaging.** Drop contracts with a zero/absent bid (a
   contract nobody will pay for has no meaningful IV), with a spread wider
   than the option's own value, and with IV outside a plausible band.
2. **Pool both sides.** Calls and puts at the same strike should imply the
   same vol under put-call parity. Using both doubles the sample and
   surfaces disagreement.
3. **Take a weighted median, not a mean.** The median ignores the outliers
   that survive filtering; weighting by vega concentrates the estimate on
   the contracts whose IV is actually well determined.
4. **Report quality, never guess silently.** Every estimate comes back with
   the sample size, the dispersion, and a ``quality`` of good/fair/poor.
   When the answer is unreliable the caller is told so, rather than being
   handed a confident-looking number.

None of this makes a bad chain good. It makes a bad chain *legible*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dealer_greeks import greek_grid
from .timeutil import years_to_expiry

__all__ = ["ATMResult", "atm_iv", "first_usable_atm_iv",
           "iv_term_structure", "risk_reversal_25d",
           "IV_MIN", "IV_MAX", "DISPERSION_GOOD", "DISPERSION_POOR",
           "MAX_REL_SPREAD"]

# Plausibility band for an annualised equity IV. Outside this, the vendor's
# solver has failed rather than found something interesting.
IV_MIN = 0.02
IV_MAX = 5.00

# Interquartile spread of the accepted sample, as a fraction of the median.
#
# Relative, not absolute, and that choice matters. A 0DTE chain quoting
# 2.1%-8.8% across adjacent strikes has an absolute IQR of only ~4 vol
# points, which looks tight next to a 30%-vol name — yet those strikes
# disagree by a factor of four and the estimate is worthless. Measuring
# spread against the level itself catches that; measuring it in absolute vol
# points does not.
DISPERSION_GOOD = 0.15   # IQR within 15% of the median
DISPERSION_POOR = 0.40   # beyond 40%, the strikes do not agree on a level

# A quote whose bid-ask spread exceeds this fraction of its mid is not a
# price anyone trades at, and inverting it for vol is meaningless. A penny
# -wide market on a 1.5-cent option is a 67% spread.
MAX_REL_SPREAD = 0.50


@dataclass
class ATMResult:
    iv: float = 0.0
    quality: str = "none"          # good | fair | poor | none
    n_samples: int = 0
    dispersion: float = 0.0        # IQR of the sample / its median (relative)
    strikes_used: list[float] = field(default_factory=list)
    source: str = ""               # what produced the number
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {"iv": self.iv, "quality": self.quality, "n_samples": self.n_samples,
                "dispersion": self.dispersion, "source": self.source,
                "fallback_used": self.fallback_used}

    def __bool__(self) -> bool:
        return self.iv > 0


def _clean_side(df: pd.DataFrame | None, spot: float, width_pct: float,
                require_bid: bool) -> pd.DataFrame:
    """Numeric, de-junked view of one side of the chain near the money."""
    cols = ["strike", "impliedVolatility", "bid", "ask", "openInterest", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    for c in cols:
        d[c] = pd.to_numeric(d.get(c, 0), errors="coerce")
    d = d.dropna(subset=["strike", "impliedVolatility"])
    d = d[(d["strike"] > 0) & (d["impliedVolatility"].between(IV_MIN, IV_MAX))]
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["dist_pct"] = (d["strike"] - spot).abs() / spot
    d = d[d["dist_pct"] <= width_pct]
    if d.empty:
        return pd.DataFrame(columns=cols)

    if require_bid:
        bid = d["bid"].fillna(0.0)
        ask = d["ask"].fillna(0.0)
        mid = (bid + ask) / 2.0
        # A contract with no bid is uninvertible. A *relatively* wide spread
        # means the mid is not a real price: on a 1-cent-bid contract a
        # one-tick market is a 67% spread, and its implied vol is a rounding
        # artefact even though the absolute spread looks tiny.
        rel_spread = np.divide((ask - bid).to_numpy(), mid.to_numpy(),
                               out=np.full(len(d), np.inf),
                               where=mid.to_numpy() > 0)
        keep = (bid > 0) & (mid > 0) & (rel_spread <= MAX_REL_SPREAD)
        if keep.any():
            d = d[keep]
    return d[cols + ["dist_pct"]]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if weights is None or not np.isfinite(weights).any() or weights.sum() <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w) / w.sum()
    return float(v[int(np.searchsorted(c, 0.5))])


def atm_iv(calls: pd.DataFrame | None, puts: pd.DataFrame | None, spot: float,
           expiry: str | None = None, t: float | None = None,
           width_pct: float = 0.05, min_samples: int = 3,
           require_bid: bool = True, r: float = 0.04, q: float = 0.0) -> ATMResult:
    """Vega-weighted median IV of the tradeable strikes near the money.

    ``width_pct`` is the moneyness window (default +/-5% of spot). Falls back
    to a wider window, then to dropping the bid requirement, before giving
    up - each relaxation is recorded in ``source`` so the caller can see
    which one produced the answer.

    ``quality`` is:
      good  >= 6 samples, relative IQR under 15%, no relaxation needed
      fair  >= min_samples samples and relative IQR under 40%
      poor  relative IQR above 40%, a thin sample, or a relaxed pass -
            the number is returned but should not be trusted
      none  nothing usable
    """
    out = ATMResult()
    if spot is None or spot <= 0:
        return out

    if t is None:
        t = years_to_expiry(expiry) if expiry else 0.0

    attempts = [
        (width_pct, require_bid, "atm_window"),
        (width_pct * 2, require_bid, "atm_window_wide"),
        (width_pct * 2, False, "atm_window_wide_nobid"),
        # Last resort: a chain whose nearest strike is far from spot has no
        # at-the-money contract at all. Return something so the caller is
        # not left with nothing, flagged poor so nobody mistakes it for one.
        (0.35, False, "nearest_available"),
    ]

    for w_pct, need_bid, label in attempts:
        frames = [_clean_side(calls, spot, w_pct, need_bid),
                  _clean_side(puts, spot, w_pct, need_bid)]
        pool = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame()
        if pool.empty:
            continue

        ivs = pool["impliedVolatility"].to_numpy(dtype=float)
        strikes = pool["strike"].to_numpy(dtype=float)

        # Weight by vega: the IV of a contract with real vega is a real
        # measurement; the IV of a near-zero-vega contract is a rounding
        # artefact, and should barely count.
        if t and t > 0:
            weights = greek_grid(spot, strikes, t, ivs, "C", r, q)["vega"]
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        else:
            weights = np.ones_like(ivs)
        if weights.sum() <= 0:
            weights = np.ones_like(ivs)

        est = _weighted_median(ivs, weights)
        if not (IV_MIN <= est <= IV_MAX):
            continue

        q75, q25 = np.percentile(ivs, [75, 25])
        abs_iqr = float(q75 - q25)
        dispersion = abs_iqr / est if est > 0 else float("inf")
        n = int(ivs.size)

        # Dispersion vetoes sample size. A 0DTE chain can offer fifty
        # near-the-money quotes whose IVs span thirty vol points; that is a
        # large sample of noise, not a good measurement, and counting the
        # rows would rate it higher than a clean eight-strike chain.
        if dispersion > DISPERSION_POOR:
            quality = "poor"
        elif label != "atm_window":
            quality = "poor"
        elif n >= 6 and dispersion < DISPERSION_GOOD:
            quality = "good"
        elif n >= min_samples and dispersion < DISPERSION_POOR:
            quality = "fair"
        else:
            quality = "poor"

        out = ATMResult(
            iv=float(est), quality=quality, n_samples=n, dispersion=dispersion,
            strikes_used=[float(s) for s in np.unique(strikes)],
            source=label, fallback_used=(label != "atm_window"),
        )
        return out
    return out


def iv_term_structure(chains: list[tuple[str, pd.DataFrame | None, pd.DataFrame | None]],
                      spot: float, now=None) -> dict:
    """ATM IV by expiry, plus the shape of the curve.

    ``shape`` is ``contango`` when far-dated IV exceeds front (the normal
    resting state), ``backwardation`` when the front is bid over the back
    (an event is priced into the near term), or ``flat``.

    Backwardation is the structurally interesting case: it means the market
    is paying up for near-dated optionality specifically, which is what an
    earnings date, an FDA decision, or a macro print looks like from the
    options side.
    """
    points = []
    for expiry, calls, puts in chains:
        t = years_to_expiry(expiry, now=now)
        res = atm_iv(calls, puts, spot, expiry=expiry, t=t)
        if not res:
            continue
        points.append({
            "expiry": expiry,
            "dte": round(t * 365.0, 2),
            "t_years": t,
            "atm_iv": res.iv,
            "quality": res.quality,
            "n_samples": res.n_samples,
        })
    points.sort(key=lambda p: p["t_years"])

    shape, slope = "unknown", 0.0
    usable = [p for p in points if p["quality"] in ("good", "fair")]
    if len(usable) >= 2:
        front, back = usable[0], usable[-1]
        slope = back["atm_iv"] - front["atm_iv"]
        if slope > 0.01:
            shape = "contango"
        elif slope < -0.01:
            shape = "backwardation"
        else:
            shape = "flat"
    return {"points": points, "shape": shape, "slope": float(slope),
            "front_iv": usable[0]["atm_iv"] if usable else 0.0,
            "back_iv": usable[-1]["atm_iv"] if usable else 0.0,
            "n_usable": len(usable)}


def risk_reversal_25d(calls: pd.DataFrame | None, puts: pd.DataFrame | None,
                      spot: float, t: float, r: float = 0.04, q: float = 0.0) -> dict:
    """25-delta risk reversal: put IV minus call IV, in vol points.

    The standard measure of skew. Positive means downside protection is bid
    relative to upside - the normal state for equity indices. A move toward
    zero or negative means calls are being bid up relative to puts, which is
    what an upside-chasing / potential-squeeze book looks like.

    Strikes are selected by *computed* delta rather than by a fixed
    percentage from spot, so the number stays comparable across tickers and
    tenors with very different volatilities.
    """
    out = {"rr_25d": 0.0, "call_iv": 0.0, "put_iv": 0.0,
           "call_strike": None, "put_strike": None, "quality": "none"}
    if spot <= 0 or t <= 0:
        return out

    def pick(df: pd.DataFrame | None, right: str, target: float):
        d = _clean_side(df, spot, width_pct=0.50, require_bid=True)
        if d.empty:
            return None, None
        strikes = d["strike"].to_numpy(dtype=float)
        ivs = d["impliedVolatility"].to_numpy(dtype=float)
        delta = greek_grid(spot, strikes, t, ivs, right, r, q)["delta"]
        idx = int(np.argmin(np.abs(np.abs(delta) - target)))
        # Reject if the closest available delta is nowhere near the target.
        if abs(abs(delta[idx]) - target) > 0.12:
            return None, None
        return float(strikes[idx]), float(ivs[idx])

    ck, civ = pick(calls, "C", 0.25)
    pk, piv = pick(puts, "P", 0.25)
    if civ is None or piv is None:
        return out
    out.update({
        "rr_25d": float((piv - civ) * 100.0),   # vol points
        "call_iv": civ, "put_iv": piv,
        "call_strike": ck, "put_strike": pk,
        "quality": "good",
    })
    return out


def first_usable_atm_iv(chains: list[tuple[str, "pd.DataFrame | None", "pd.DataFrame | None"]],
                        spot: float, now=None,
                        accept: tuple[str, ...] = ("good", "fair")) -> tuple[float, str, str | None]:
    """Walk expiries front to back and return the first trustworthy ATM IV.

    Returns ``(iv, quality, expiry)``. The front expiry is very often 0DTE,
    where vendor IV is numerically meaningless (see the module docstring),
    so "the nearest expiry" is the wrong default — this takes the nearest
    expiry that actually produces a usable estimate, and falls back to the
    best poor one if nothing qualifies, so the caller always learns both the
    number and how much to trust it.
    """
    best: tuple[float, str, str | None] | None = None
    for expiry, calls, puts in chains:
        t = years_to_expiry(expiry, now=now)
        res = atm_iv(calls, puts, spot, expiry=expiry, t=t)
        if not res:
            continue
        if res.quality in accept:
            return res.iv, res.quality, expiry
        if best is None:
            best = (res.iv, res.quality, expiry)
    return best if best is not None else (0.0, "none", None)
