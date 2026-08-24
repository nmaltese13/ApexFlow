"""How old is this number, and is it old enough to be dangerous?

Why this exists
---------------
Until now a fifteen-minute-delayed quote rendered identically to a live one,
and a feed that had silently died rendered identically to both. The only
acknowledgement of the problem was a static sentence in the page footer
saying data "may be delayed".

That is survivable in an analysis tool. It is the single most expensive
failure mode in a tool used to make decisions with money, because the
failure is *invisible*: the numbers still look precise, the charts still
render, and nothing distinguishes a live book from a snapshot of one taken
before lunch.

It is not hypothetical here. ``CboeProvider`` explicitly serves its last
cached chain when a fetch fails ("serve stale rather than nothing"), and
``YFinanceProvider`` does the same under rate-limit backoff. Both are the
right behaviour for a dashboard and both need to be *visible*.

Measuring age against the session, not the clock
------------------------------------------------
Wall-clock age is the wrong measure. At 08:00 on a Saturday the last trade
is sixteen hours old and that is completely correct; at 14:30 on a Wednesday
a sixteen-minute-old quote may mean the feed has stopped. So age is measured
against :func:`analytics.timeutil.last_session_close` when the market is
shut, and against now when it is open.

Two kinds of age, and why the distinction matters
------------------------------------------------
A provider timestamp says when the *payload* was produced, not how old the
quotes inside it are. Cboe regenerates its file constantly, so a fetch is
always seconds old — but the prices in it are on a fifteen-minute delay.
Reporting that as "live" would be precisely the misleading thing this
module exists to prevent.

So a feed can never be classified fresher than its own documented delay
allows. :data:`REALTIME_SOURCES` lists the providers that can genuinely be
live; everything else tops out at ``delayed`` no matter how recently the
file was fetched.

Levels
------
``live``     real-time feed, under a minute old, session open
``delayed``  current for what this feed can deliver
``stale``    older than this feed should ever be. Something is wrong
``unknown``  the provider gave no timestamp at all

``tradeable`` is deliberately conservative: only ``live`` and ``delayed``
during an open session. Everything else is fine to *look* at and should not
be the basis of an execution decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from apexflow.analytics.timeutil import (
    market_state, last_session_close, now_utc, _as_utc,
)

__all__ = ["Freshness", "assess", "provider_delay_seconds", "DELAYS",
           "REALTIME_SOURCES"]

#: Expected delay per provider, in seconds. Anything materially beyond this
#: is a fault rather than the documented behaviour of the feed.
DELAYS: dict[str, int] = {
    "schwab": 60,          # real-time with a brokerage account
    "polygon": 60,         # real-time on the options plans
    "marketdata": 900,
    "cboe": 900,           # ~15 minutes, documented
    "yfinance": 900,       # ~15 minutes, and rate-limits into staleness
    "tradier": 60,
    "demo": None,          # a frozen snapshot; age is meaningless by design
}

#: Providers whose quotes are genuinely real-time. Everything else is
#: capped at "delayed" however recently its file was fetched, because the
#: fetch timestamp says nothing about the age of the prices inside.
REALTIME_SOURCES = frozenset({"schwab", "polygon", "tradier"})

#: Under this many seconds during an open session, a real-time feed is live.
LIVE_SECONDS = 60

#: How far past a provider's documented delay before calling it stale.
STALE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class Freshness:
    """The age of a payload, and whether it should drive a decision."""
    level: str                      # live | delayed | stale | unknown
    age_seconds: float | None
    as_of: datetime | None
    source: str
    market: str                     # open | premarket | afterhours | closed
    tradeable: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "age_seconds": None if self.age_seconds is None else round(self.age_seconds, 1),
            "age_label": self.age_label,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "source": self.source,
            "market": self.market,
            "tradeable": self.tradeable,
            "detail": self.detail,
        }

    @property
    def age_label(self) -> str:
        a = self.age_seconds
        if a is None:
            return "unknown"
        if a < 90:
            return f"{int(a)}s"
        if a < 5400:
            return f"{int(a // 60)}m"
        if a < 172800:
            return f"{a / 3600:.1f}h"
        return f"{a / 86400:.1f}d"


def provider_delay_seconds(source: str) -> int | None:
    return DELAYS.get((source or "").lower(), 900)


def assess(as_of: datetime | float | str | None, source: str = "unknown",
           now: datetime | None = None) -> Freshness:
    """Classify how old a payload is, relative to the market session.

    ``as_of`` accepts a datetime, a unix timestamp, or an ISO string —
    providers stamp it in whatever form is natural to them.
    """
    now = _as_utc(now) if now is not None else now_utc()
    state = market_state(now)
    stamped = _coerce(as_of)

    if source.lower() == "demo":
        return Freshness(
            level="delayed", age_seconds=None, as_of=stamped, source=source,
            market=state, tradeable=False,
            detail="Frozen demo snapshot — not live data and not tradeable.")

    if stamped is None:
        return Freshness(
            level="unknown", age_seconds=None, as_of=None, source=source,
            market=state, tradeable=False,
            detail="Provider returned no timestamp, so the age of this data "
                   "cannot be established.")

    # Outside the session, measure against the last close: a price that is
    # correctly hours old on a Sunday is not a fault.
    reference = now if state == "open" else last_session_close(now)
    age = max((reference - stamped).total_seconds(), 0.0)

    expected = provider_delay_seconds(source)
    if expected is None:
        expected = 900

    if state != "open":
        # With the market shut the useful question is not "how many minutes
        # old" but "is this from the most recent session?". Data from the
        # last session is exactly what should be on screen; data predating
        # it means the feed missed a day.
        session_start = reference - timedelta(hours=7)     # 09:30 -> 16:00 ET
        from_last_session = stamped >= session_start
        return Freshness(
            level="delayed" if from_last_session else "stale",
            age_seconds=age, as_of=stamped, source=source, market=state,
            tradeable=False,
            detail=(f"Market is {state}. This is the last session's close, "
                    f"which is what should be showing."
                    if from_last_session else
                    f"Market is {state}, but this data predates the most "
                    f"recent session — the feed appears to have missed a day."))

    realtime = source.lower() in REALTIME_SOURCES

    if realtime and age <= LIVE_SECONDS:
        return Freshness(
            level="live", age_seconds=age, as_of=stamped, source=source,
            market=state, tradeable=True,
            detail=f"Live: {_fmt(age)} old from {source}.")

    if age <= expected * STALE_MULTIPLIER:
        # A freshly fetched file from a delayed feed is still delayed data.
        note = (f"Fetched {_fmt(age)} ago, but {source} publishes on a "
                f"~{expected // 60} minute delay, so the prices are about "
                f"that old.") if not realtime else (
                f"Delayed {_fmt(age)} — within what {source} is expected to have.")
        return Freshness(
            level="delayed", age_seconds=age, as_of=stamped, source=source,
            market=state, tradeable=True, detail=note)

    return Freshness(
        level="stale", age_seconds=age, as_of=stamped, source=source,
        market=state, tradeable=False,
        detail=f"STALE: {_fmt(age)} old during an open session, well beyond "
               f"the ~{expected // 60} minutes expected from {source}. The "
               f"feed may have stopped and a cached copy may be showing.")


def _coerce(v) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return _as_utc(v)
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return _as_utc(datetime.fromisoformat(str(v).replace("Z", "+00:00")))
    except ValueError:
        return None


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds // 60)}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
