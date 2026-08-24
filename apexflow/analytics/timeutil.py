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


# ---------------------------------------------------------------------------
# Market sessions
# ---------------------------------------------------------------------------
# The previous session check lived in atlas.py and compared against fixed UTC
# minutes (13:30-20:00). That is 9:30-16:00 ET only while daylight saving is
# in effect; for the roughly four months either side of it the same constants
# mean 8:30-15:00 ET, so the market read as open an hour before it was and
# closed an hour before it did. It also ignored holidays entirely.
#
# Being wrong about whether the market is open matters more in a trading
# context than an analytical one: it is what separates "this quote is
# correctly hours old because it is Sunday" from "this feed has died".

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous algorithm) — for Good Friday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth `weekday` of a month (n=-1 for the last one)."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + 7 * (n - 1))
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """US convention: Saturday holidays observe Friday, Sunday observes Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def market_holidays(year: int) -> set[date]:
    """NYSE/Nasdaq full-day closures for `year`.

    Excludes ad-hoc closures (presidential funerals, Hurricane Sandy), which
    cannot be computed and are rare enough to accept.
    """
    return {
        _observed(date(year, 1, 1)),                     # New Year's Day
        _nth_weekday(year, 1, 0, 3),                     # MLK Day
        _nth_weekday(year, 2, 0, 3),                     # Presidents Day
        _easter(year) - timedelta(days=2),               # Good Friday
        _nth_weekday(year, 5, 0, -1),                    # Memorial Day
        _observed(date(year, 6, 19)),                    # Juneteenth
        _observed(date(year, 7, 4)),                     # Independence Day
        _nth_weekday(year, 9, 0, 1),                     # Labor Day
        _nth_weekday(year, 11, 3, 4),                    # Thanksgiving
        _observed(date(year, 12, 25)),                   # Christmas
    }


def early_close_days(year: int) -> set[date]:
    """Sessions that end at 13:00 ET instead of 16:00."""
    days = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}   # day after Thanksgiving
    for d in (date(year, 7, 3), date(year, 12, 24)):
        if d.weekday() < 5:
            days.add(d)
    return days


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def session_close(day: date) -> time:
    return EARLY_CLOSE if day in early_close_days(day.year) else REGULAR_CLOSE


def market_state(now: datetime | None = None) -> str:
    """One of: ``open``, ``premarket``, ``afterhours``, ``closed``.

    ``closed`` means a weekend or holiday; ``premarket``/``afterhours`` mean
    a trading day outside the regular session.
    """
    now = _as_utc(now)
    if _NY is None:                      # no tz database — assume open, warn elsewhere
        return "open"
    local = now.astimezone(_NY)
    day = local.date()
    if not is_trading_day(day):
        return "closed"
    t = local.time()
    if t < REGULAR_OPEN:
        return "premarket"
    if t > session_close(day):
        return "afterhours"
    return "open"


def is_market_open(now: datetime | None = None) -> bool:
    """True only during the regular session on a trading day."""
    return market_state(now) == "open"


def last_session_close(now: datetime | None = None) -> datetime:
    """UTC datetime of the most recent regular-session close.

    What a quote's age should be measured against outside market hours: a
    price that is sixteen hours old at 8am Saturday is correct, not broken.
    """
    now = _as_utc(now)
    if _NY is None:
        return now
    local = now.astimezone(_NY)
    day = local.date()
    if is_trading_day(day) and local.time() > session_close(day):
        close_local = datetime.combine(day, session_close(day), tzinfo=_NY)
        return close_local.astimezone(timezone.utc)
    for back in range(1, 12):
        d = day - timedelta(days=back)
        if is_trading_day(d):
            return datetime.combine(d, session_close(d), tzinfo=_NY).astimezone(timezone.utc)
    return now
