"""Risk-free rate from the US Treasury par yield curve. Free, no key.

Why this exists
---------------
Every Greek in this project took ``r = 0.04`` as a default. That was listed
as limitation 9 in ``docs/methodology.md``, and it is the kind of constant
that is quietly wrong twice over:

* **It goes stale.** Hardcoding 4% was roughly right when it was written.
  If policy rates move to 2% or 6%, every delta, theta and rho in the
  terminal is wrong and nothing in the code notices.
* **It ignores tenor.** A 0DTE contract should be discounted at the 1-month
  bill, a LEAPS at the 2-year note. On 2026-08-21 those were 3.80% and
  4.24% — a 44bp spread across the curve that a single constant cannot
  represent.

Treasury publishes the whole par yield curve daily as XML, with no key and
no rate limit. This module fetches it, caches it to disk, and interpolates
to whatever tenor a contract actually has.

How much does it matter?
------------------------
Less than the dealer-positioning assumption, and much less than bad IV —
this is a second-order correction, and it is worth being clear about that
rather than overselling it. For a 30-day option a 50bp rate error moves
delta by well under a percentage point. It matters most for:

* **long-dated contracts**, where discounting compounds;
* **rho**, which is *entirely* a rate sensitivity and is meaningless
  against a stale constant;
* **put-call parity checks**, which fail visibly when the discount rate is
  wrong.

Conventions
-----------
Treasury quotes par yields as bond-equivalent percentages (``3.88`` means
3.88%). Black-Scholes wants a continuously-compounded decimal, so the
conversion is ``r_cc = ln(1 + y/2) * 2`` for the semi-annual convention,
applied in :func:`_to_continuous`. Skipping that step is a common shortcut
worth about 4bp at these levels — small, but free to get right.

Failure behaviour
-----------------
Network failures fall back to :data:`FALLBACK_RATE` and log once. The
terminal keeps working with a stale constant rather than breaking, which is
the right trade for a second-order input — but ``is_live()`` reports which
one you got, so nothing silently claims more precision than it has.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import requests

import config

log = logging.getLogger(__name__)

__all__ = ["TreasuryCurve", "risk_free_rate", "get_curve", "FALLBACK_RATE",
           "clear_cache"]

TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/pages/xml?data=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}")

#: Used when the curve cannot be fetched. Matches the constant that was
#: previously hardcoded throughout, so behaviour is unchanged on failure.
FALLBACK_RATE = 0.04

CACHE_PATH = config.CACHE_DIR / "treasury_curve.json"
CACHE_TTL_SECONDS = 12 * 3600

#: Treasury XML field -> tenor in years.
_TENORS: dict[str, float] = {
    "BC_1MONTH": 1 / 12, "BC_2MONTH": 2 / 12, "BC_3MONTH": 0.25,
    "BC_4MONTH": 4 / 12, "BC_6MONTH": 0.5, "BC_1YEAR": 1.0,
    "BC_2YEAR": 2.0, "BC_3YEAR": 3.0, "BC_5YEAR": 5.0, "BC_7YEAR": 7.0,
    "BC_10YEAR": 10.0, "BC_20YEAR": 20.0, "BC_30YEAR": 30.0,
}


def _to_continuous(par_yield_pct: float) -> float:
    """Semi-annual bond-equivalent percent -> continuously-compounded decimal."""
    y = par_yield_pct / 100.0
    if y <= -1.0:
        return 0.0
    return 2.0 * math.log1p(y / 2.0)


@dataclass
class TreasuryCurve:
    """A single day's par yield curve, in continuously-compounded decimals."""
    as_of: date
    points: list[tuple[float, float]] = field(default_factory=list)  # (tenor_yrs, rate)
    live: bool = True

    def rate(self, t_years: float) -> float:
        """Interpolate the curve at `t_years`, flat beyond either end.

        Linear in tenor, which is standard for a par curve at this
        precision. Extrapolation is deliberately flat rather than linear:
        continuing the slope past 30 years produces nonsense, and past the
        short end produces negative rates.
        """
        if not self.points:
            return FALLBACK_RATE
        t = max(float(t_years), 1e-6)
        xs = [p[0] for p in self.points]
        if t <= xs[0]:
            return self.points[0][1]
        if t >= xs[-1]:
            return self.points[-1][1]
        i = bisect_left(xs, t)
        x0, y0 = self.points[i - 1]
        x1, y1 = self.points[i]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)

    def to_dict(self) -> dict:
        return {"as_of": self.as_of.isoformat(),
                "points": [[t, r] for t, r in self.points],
                "live": self.live}

    @classmethod
    def from_dict(cls, d: dict) -> "TreasuryCurve":
        return cls(as_of=date.fromisoformat(d["as_of"]),
                   points=[(float(t), float(r)) for t, r in d["points"]],
                   live=bool(d.get("live", True)))

    @classmethod
    def fallback(cls) -> "TreasuryCurve":
        return cls(as_of=datetime.now(timezone.utc).date(),
                   points=[(1 / 12, FALLBACK_RATE), (30.0, FALLBACK_RATE)],
                   live=False)


def _parse(xml_text: str) -> TreasuryCurve | None:
    """Pull the most recent entry out of Treasury's XML feed.

    Parsed with regex rather than an XML library on purpose: the feed is a
    fixed-shape Atom document, the fields are namespaced awkwardly, and a
    tolerant scan degrades to "no curve" on a format change instead of
    raising somewhere deep in a parser.
    """
    entries = re.findall(r"<entry>.*?</entry>", xml_text, re.S)
    if not entries:
        return None
    last = entries[-1]

    m = re.search(r"<d:NEW_DATE[^>]*>([^<]+)<", last)
    if not m:
        return None
    try:
        as_of = datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).date()
    except ValueError:
        return None

    points: list[tuple[float, float]] = []
    for field_name, tenor in _TENORS.items():
        fm = re.search(rf"<d:{field_name}[^>]*>([^<]*)<", last)
        if not fm or not fm.group(1).strip():
            continue
        try:
            points.append((tenor, _to_continuous(float(fm.group(1)))))
        except ValueError:
            continue
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    return TreasuryCurve(as_of=as_of, points=points, live=True)


def _read_cache() -> TreasuryCurve | None:
    if not CACHE_PATH.exists():
        return None
    try:
        if time.time() - CACHE_PATH.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        return TreasuryCurve.from_dict(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(curve: TreasuryCurve) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(curve.to_dict()), encoding="utf-8")
    except OSError as e:
        log.debug("treasury cache write failed: %s", e)


_memo: TreasuryCurve | None = None
_warned = False


def get_curve(refresh: bool = False) -> TreasuryCurve:
    """The current Treasury curve, from memory, disk cache, then network."""
    global _memo, _warned
    if _memo is not None and not refresh:
        return _memo
    if not refresh:
        cached = _read_cache()
        if cached is not None:
            _memo = cached
            return cached

    year = datetime.now(timezone.utc).year
    try:
        r = requests.get(TREASURY_URL.format(year=year), timeout=20,
                         headers={"User-Agent": "ApexFlow/1.0 (options research)"})
        r.raise_for_status()
        curve = _parse(r.text)
    except Exception as e:
        curve = None
        if not _warned:
            log.warning("Treasury curve unavailable (%s); falling back to %.2f%%",
                        e, FALLBACK_RATE * 100)
            _warned = True

    if curve is None:
        stale = _read_cache()
        curve = stale if stale is not None else TreasuryCurve.fallback()
    else:
        _write_cache(curve)

    _memo = curve
    return curve


def risk_free_rate(t_years: float = 0.25, refresh: bool = False) -> float:
    """Continuously-compounded risk-free rate for a `t_years` horizon.

    Falls back to :data:`FALLBACK_RATE` when the curve cannot be fetched,
    so callers never have to handle an error for a second-order input.
    """
    return get_curve(refresh=refresh).rate(t_years)


def clear_cache() -> None:
    """Drop the memo and the disk cache. Used by tests."""
    global _memo, _warned
    _memo = None
    _warned = False
    try:
        CACHE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
