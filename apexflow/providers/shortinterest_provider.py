"""Point-in-time short interest, free, from FINRA and SEC EDGAR.

This module exists to answer one question the project could not previously
answer: **does the squeeze score actually predict anything?**

``platform/squeeze_backtest.py`` could only reconstruct 18% of the score
from price history, because short interest, days-to-cover and float were
available only as *current* values. Scoring a past date with today's short
interest is lookahead bias, and it biases results upward, so the harness
refused to issue a verdict. Both missing pieces are in fact free:

* **FINRA** publishes consolidated short interest twice a month, per
  symbol, back to 2020 — shares short, average daily volume and
  days-to-cover, each stamped with its settlement date.
* **SEC EDGAR** publishes shares outstanding in XBRL, and critically stamps
  each observation with the date it was *filed*.

Together they take backtest coverage from 18% to about 75%, past the
threshold at which the harness will report a result at all.

The publication lag, which is the whole game
--------------------------------------------
FINRA short interest is *settled* on the 15th and the last business day of
each month, but it is not *disseminated* until roughly eight business days
later. A backtest that keys off the settlement date is quietly using
information that nobody had yet — the exact error this module exists to
avoid, dressed up as a fix for it.

So every lookup here takes a knowledge date and returns only records whose
**publication** date is on or before it. The lag is applied explicitly
(:data:`FINRA_PUBLICATION_LAG_BUSINESS_DAYS`) rather than assumed away, and
EDGAR observations are filtered on ``filed``, never on ``end``.

Float versus shares outstanding
-------------------------------
EDGAR reports shares *outstanding*; short interest is conventionally quoted
against the *float*, which excludes insider and restricted holdings. Float
is smaller, so ``shares_short / shares_outstanding`` **understates** short
percentage — often materially for founder-controlled names.

That is a real approximation and it is deliberately the conservative
direction: it under-states squeeze pressure rather than inventing it. Every
result carries ``float_is_approximate=True`` so nothing downstream can
mistake it for a true short-%-of-float.

Caching
-------
Both APIs are slow and rate-limited by courtesy rather than by key. Results
are cached to ``data/cache/short_interest/`` as JSON, keyed by symbol, so a
backtest sweeping hundreds of dates hits the network once per ticker.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

import config

log = logging.getLogger(__name__)

__all__ = [
    "ShortInterestRecord", "ShortInterestHistory", "ShortInterestClient",
    "FINRA_PUBLICATION_LAG_BUSINESS_DAYS", "SEC_USER_AGENT_ENV",
    "sec_user_agent",
]

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CONCEPT_URL = ("https://data.sec.gov/api/xbrl/companyconcept/"
                   "CIK{cik}/dei/EntityCommonStockSharesOutstanding.json")

#: FINRA disseminates short interest about eight business days after the
#: settlement date. Applied to every lookup so a backtest never sees a
#: figure before the market did.
FINRA_PUBLICATION_LAG_BUSINESS_DAYS = 8

#: SEC requires a User-Agent identifying you and carrying a contact email,
#: and returns 403 without one. There is deliberately no default: an email
#: address is personal data, and baking one into a shared codebase would
#: send it to a third party on behalf of whoever runs the code. Set:
#:
#:     APEXFLOW_SEC_USER_AGENT="Your Name your.email@example.com"
#:
#: FINRA needs nothing. Without this only the share-count half is skipped,
#: which costs the short-%-of-float axis and drops backtest coverage from
#: ~75% to ~40%.
SEC_USER_AGENT_ENV = "APEXFLOW_SEC_USER_AGENT"

CACHE_DIR = config.CACHE_DIR / "short_interest"


def sec_user_agent() -> str | None:
    """The SEC User-Agent, or None when the operator has not supplied one."""
    import os
    ua = (os.environ.get(SEC_USER_AGENT_ENV) or "").strip()
    return ua or None


def _finra_user_agent() -> str:
    """FINRA has no contact requirement, so a plain product string is fine."""
    return "ApexFlow/1.0 (options research)"


def _publication_date(settlement: date,
                      lag_business_days: int = FINRA_PUBLICATION_LAG_BUSINESS_DAYS) -> date:
    """When a settlement-dated figure actually became public."""
    return np.busday_offset(np.datetime64(settlement, "D"),
                            lag_business_days, roll="forward").astype("O")


@dataclass(frozen=True)
class ShortInterestRecord:
    """One FINRA observation, with the date it became knowable."""
    settlement_date: date
    publication_date: date
    shares_short: float
    avg_daily_volume: float
    days_to_cover: float

    def to_dict(self) -> dict:
        return {
            "settlement_date": self.settlement_date.isoformat(),
            "publication_date": self.publication_date.isoformat(),
            "shares_short": self.shares_short,
            "avg_daily_volume": self.avg_daily_volume,
            "days_to_cover": self.days_to_cover,
        }


@dataclass
class ShortInterestHistory:
    """A symbol's short-interest and share-count history, queryable by date."""
    symbol: str
    records: list[ShortInterestRecord] = field(default_factory=list)
    #: (filed_date, shares_outstanding), ascending by filed date.
    share_counts: list[tuple[date, float]] = field(default_factory=list)

    def as_of(self, when: date) -> ShortInterestRecord | None:
        """The most recent record *published* on or before `when`."""
        best = None
        for r in self.records:
            if r.publication_date <= when and (best is None or
                                               r.publication_date > best.publication_date):
                best = r
        return best

    def shares_outstanding_as_of(self, when: date) -> float | None:
        """Most recent share count *filed* on or before `when`."""
        best = None
        for filed, val in self.share_counts:
            if filed <= when and (best is None or filed > best[0]):
                best = (filed, val)
        return best[1] if best else None

    def short_pct_as_of(self, when: date) -> float | None:
        """shares_short / shares_outstanding, or None if either is missing.

        An approximation of short-%-of-float; see the module docstring.
        """
        rec = self.as_of(when)
        shares = self.shares_outstanding_as_of(when)
        if rec is None or not shares or shares <= 0:
            return None
        return float(rec.shares_short / shares)

    @property
    def covered_span(self) -> tuple[date, date] | None:
        if not self.records:
            return None
        pubs = [r.publication_date for r in self.records]
        return min(pubs), max(pubs)


class ShortInterestClient:
    """Fetches and caches FINRA short interest plus SEC share counts."""

    def __init__(self, cache_dir: Path | None = None, ttl_days: int = 7,
                 timeout: int = 30, sec_delay: float = 0.15):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_days * 86400
        self.timeout = timeout
        # SEC asks for <10 req/s; this keeps us well under without being slow.
        self.sec_delay = sec_delay
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _finra_user_agent()})
        self._cik_map: dict[str, int] | None = None
        self._mem: dict[str, ShortInterestHistory] = {}
        self._warned_no_sec_ua = False

    # -- caching ----------------------------------------------------------
    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}.json"

    def _read_cache(self, symbol: str) -> dict | None:
        p = self._cache_path(symbol)
        if not p.exists() or time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, symbol: str, payload: dict) -> None:
        try:
            self._cache_path(symbol).write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        except OSError as e:
            log.warning("short-interest cache write failed for %s: %s", symbol, e)

    # -- FINRA ------------------------------------------------------------
    def _fetch_finra(self, symbol: str, start: date, end: date) -> list[ShortInterestRecord]:
        body = {
            "limit": 5000,
            "compareFilters": [{"fieldName": "symbolCode",
                                "fieldValue": symbol.upper(),
                                "compareType": "EQUAL"}],
            "dateRangeFilters": [{"fieldName": "settlementDate",
                                  "startDate": start.isoformat(),
                                  "endDate": end.isoformat()}],
        }
        try:
            r = self._session.post(FINRA_URL, json=body, timeout=self.timeout,
                                   headers={"Content-Type": "application/json"})
            r.raise_for_status()
        except Exception as e:
            log.warning("FINRA fetch failed for %s: %s", symbol, e)
            return []

        out: list[ShortInterestRecord] = []
        for row in csv.DictReader(io.StringIO(r.text)):
            try:
                settle = datetime.strptime(row["settlementDate"][:10], "%Y-%m-%d").date()
                shares = float(row["currentShortPositionQuantity"] or 0)
                adv = float(row["averageDailyVolumeQuantity"] or 0)
                dtc = float(row["daysToCoverQuantity"] or 0)
            except (KeyError, ValueError, TypeError):
                continue
            if shares <= 0:
                continue
            out.append(ShortInterestRecord(
                settlement_date=settle,
                publication_date=_publication_date(settle),
                shares_short=shares, avg_daily_volume=adv, days_to_cover=dtc,
            ))
        out.sort(key=lambda r: r.settlement_date)
        return out

    # -- SEC EDGAR --------------------------------------------------------
    def _sec_headers(self) -> dict | None:
        ua = sec_user_agent()
        if not ua:
            if not self._warned_no_sec_ua:
                log.warning(
                    "%s is not set, so SEC share counts are unavailable and the "
                    "short-%%-of-float axis will be missing. Set it to "
                    "'Your Name your.email@example.com' — the SEC requires a "
                    "contact address and returns 403 without one.",
                    SEC_USER_AGENT_ENV)
                self._warned_no_sec_ua = True
            return None
        return {"User-Agent": ua}

    def _cik(self, symbol: str) -> int | None:
        headers = self._sec_headers()
        if headers is None:
            return None
        if self._cik_map is None:
            try:
                r = self._session.get(SEC_TICKERS_URL, timeout=self.timeout,
                                      headers=headers)
                r.raise_for_status()
                self._cik_map = {v["ticker"].upper(): int(v["cik_str"])
                                 for v in r.json().values()}
            except Exception as e:
                log.warning("SEC ticker map failed: %s", e)
                self._cik_map = {}
        return self._cik_map.get(symbol.upper())

    def _fetch_shares(self, symbol: str) -> list[tuple[date, float]]:
        headers = self._sec_headers()
        if headers is None:
            return []
        cik = self._cik(symbol)
        if cik is None:
            return []
        time.sleep(self.sec_delay)
        try:
            r = self._session.get(SEC_CONCEPT_URL.format(cik=str(cik).zfill(10)),
                                  timeout=self.timeout, headers=headers)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            units = r.json().get("units", {})
        except Exception as e:
            log.warning("SEC shares fetch failed for %s: %s", symbol, e)
            return []

        rows = units.get("shares") or next(iter(units.values()), [])
        out: list[tuple[date, float]] = []
        for row in rows:
            # `filed` is the knowledge date; `end` is the as-of date the
            # filing describes and is NOT when the number became public.
            filed = row.get("filed")
            val = row.get("val")
            if not filed or not val:
                continue
            try:
                out.append((datetime.strptime(filed[:10], "%Y-%m-%d").date(), float(val)))
            except (ValueError, TypeError):
                continue
        out.sort(key=lambda t: t[0])
        return out

    # -- public -----------------------------------------------------------
    def history(self, symbol: str, start: date | None = None,
                end: date | None = None) -> ShortInterestHistory:
        """Full short-interest and share-count history for one symbol."""
        symbol = symbol.upper()
        if symbol in self._mem:
            return self._mem[symbol]

        start = start or (datetime.now(timezone.utc).date() - timedelta(days=365 * 6))
        end = end or datetime.now(timezone.utc).date()

        cached = self._read_cache(symbol)
        if cached:
            hist = ShortInterestHistory(
                symbol=symbol,
                records=[ShortInterestRecord(
                    settlement_date=date.fromisoformat(r["settlement_date"]),
                    publication_date=date.fromisoformat(r["publication_date"]),
                    shares_short=r["shares_short"],
                    avg_daily_volume=r["avg_daily_volume"],
                    days_to_cover=r["days_to_cover"],
                ) for r in cached.get("records", [])],
                share_counts=[(date.fromisoformat(d), v)
                              for d, v in cached.get("share_counts", [])],
            )
            self._mem[symbol] = hist
            return hist

        records = self._fetch_finra(symbol, start, end)
        shares = self._fetch_shares(symbol)
        hist = ShortInterestHistory(symbol=symbol, records=records, share_counts=shares)

        self._write_cache(symbol, {
            "symbol": symbol,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "records": [r.to_dict() for r in records],
            "share_counts": [[d.isoformat(), v] for d, v in shares],
        })
        self._mem[symbol] = hist
        return hist

    def warm(self, symbols: list[str], progress: bool = False) -> dict[str, int]:
        """Pre-fetch a universe. Returns {symbol: record count}."""
        out: dict[str, int] = {}
        for i, sym in enumerate(symbols, 1):
            h = self.history(sym)
            out[sym] = len(h.records)
            if progress:
                span = h.covered_span
                span_s = f"{span[0]} to {span[1]}" if span else "no data"
                print(f"  [{i}/{len(symbols)}] {sym:<6} {len(h.records):>3} records  {span_s}")
        return out
