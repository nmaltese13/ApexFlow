"""Single source of truth for time-to-expiry.

Three modules used to compute this independently and disagree with each
other, which meant the same option had a different gamma depending on which
code path reached it:

  * ``greeks.years_to_expiry``  ``(exp - now).days + 0.5``, over 365
  * ``levels._years``           the same expression, duplicated
  * ``gex.years_to_expiry``     calendar-day difference plus an intraday
                                remainder for 0DTE, using a fixed 20:30 UTC
                                close

All three are now thin wrappers over :func:`years_to_expiry` here.

Two things the old implementations got wrong:

**DST.** US equity options stop trading at 16:00 America/New_York. That is
20:00 UTC while daylight saving is in effect and 21:00 UTC otherwise - the
opposite of what the old comment in ``gex.py`` claimed. A fixed 20:30 UTC
constant is up to half an hour off in both directions, which on a 0DTE
contract with two hours left is a 25% error in ``t`` and therefore a ~13%
error in gamma. This module resolves the real close using ``zoneinfo``.

**Naive datetimes.** ``datetime.utcnow()`` is deprecated in Python 3.12 and
returns a naive datetime that silently compares wrong against an aware one.
Everything here is timezone-aware internally; naive inputs are assumed UTC,
which is what the previous callers meant.

Convention note: expiry is measured in *calendar* years (365 days), not
trading years (252). Implied vol quoted by every provider in this repo is
annualised on a calendar basis, so mixing in a 252 convention here would
misprice every contract. The intraday remainder on expiry day is expressed
as a fraction of a 24-hour day for the same reason.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - tzdata missing on a bare Windows box
    _NY = None

# Fallback used only when tzdata is unavailable: 16:00 ET expressed in UTC,
# split by the (approximate) US DST window.
_FALLBACK_CLOSE_UTC_DST = 20.0
_FALLBACK_CLOSE_UTC_STD = 21.0

MINUTE_YEARS = 1.0 / (365.0 * 24.0 * 60.0)
# Fraction of a calendar day that a 6.5-hour regular session occupies. Used
# as the remaining stub on the final day for contracts expiring in the future.
SESSION_DAY_FRACTION = 6.5 / 24.0


def now_utc() -> datetime:
    """Timezone-aware current UTC time (``utcnow()`` is deprecated in 3.12)."""
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return now_utc()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def market_close_utc(day: date) -> datetime:
    """16:00 America/New_York on `day`, as an aware UTC datetime.

    Handles DST correctly, so this is 20:00 UTC in summer and 21:00 UTC in
    winter rather than a fixed constant.
    """
    if _NY is not None:
        return datetime.combine(day, time(16, 0), tzinfo=_NY).astimezone(timezone.utc)
    # tzdata unavailable - approximate the US DST window (Mar 2nd Sun to Nov 1st Sun).
    dst = 3 <= day.month <= 10
    hour = _FALLBACK_CLOSE_UTC_DST if dst else _FALLBACK_CLOSE_UTC_STD
    return datetime.combine(day, time(int(hour), int((hour % 1) * 60)), tzinfo=timezone.utc)


def parse_expiry(expiry: str | date | datetime | None) -> date | None:
    """Accept 'YYYY-MM-DD', a date, or a datetime. Returns None if unparseable."""
    if expiry is None:
        return None
    if isinstance(expiry, datetime):
        return expiry.date()
    if isinstance(expiry, date):
        return expiry
    try:
        return datetime.strptime(str(expiry)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def years_to_expiry(expiry: str | date | datetime | None,
                    now: datetime | None = None,
                    min_years: float = MINUTE_YEARS) -> float:
    """Calendar years from `now` until `expiry`'s 16:00 ET close.

    Never returns zero: an expired or same-minute contract floors at
    ``min_years`` (one minute) so that gamma stays finite instead of
    dividing by zero. An unparseable expiry falls back to one day, matching
    the previous behaviour of every caller.
    """
    now = _as_utc(now)
    exp = parse_expiry(expiry)
    if exp is None:
        return 1.0 / 365.0
    close = market_close_utc(exp)
    delta = (close - now).total_seconds()
    if delta <= 0:
        return min_years
    return max(delta / (365.0 * 24.0 * 3600.0), min_years)


def days_to_expiry(expiry: str | date | datetime | None,
                   now: datetime | None = None) -> int:
    """Whole calendar days until expiry (0 = expires today, negative = past)."""
    exp = parse_expiry(expiry)
    if exp is None:
        return 0
    return (exp - _as_utc(now).date()).days


def is_expired(expiry: str | date | datetime | None,
               now: datetime | None = None) -> bool:
    """True once the expiry's 16:00 ET close has passed."""
    exp = parse_expiry(expiry)
    if exp is None:
        return False
    return _as_utc(now) >= market_close_utc(exp)
