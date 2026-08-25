"""Cboe delayed option quotes — the best free options data available.

No API key and no registration. Cboe publishes its own delayed quote feed
as JSON, and for this project's purposes it beats yfinance on every axis
that matters:

============================  ==========================  =====================
                              Cboe                        yfinance
============================  ==========================  =====================
Implied volatility            exchange-computed           vendor solver, often
                                                          garbage near expiry
Greeks                        delta/gamma/vega/theta/rho  none
Requests per full chain       **1** (all expiries)        one *per expiry*
Rate limiting                 429s under burst            aggressive 429s
Index options (SPX, VIX)      yes                         poor/absent
============================  ==========================  =====================

The IV point is the important one. ``analytics/iv_surface.py`` exists
because yfinance's implied vols on near-expiry contracts are numerically
meaningless — the six SPY strikes nearest spot on a 0DTE expiry came back
between 2.1% and 8.8%. Cboe computes IV as the exchange, from its own
book, and publishes the Greeks alongside. That does not make the data
perfect (see *Caveats*), but it removes the single largest source of
garbage in the free-data path.

The single-request property matters nearly as much. A full SPY chain from
yfinance is one HTTP call per expiry — twenty-odd calls, each a chance to
trip a 429 — whereas Cboe returns all 13,000 contracts in one ~0.7s
response. Everything downstream that wanted a multi-expiry roll-up
(heatseeker, dealer exposures, the gamma flip curve) gets much cheaper.

Hybrid by design
----------------
Cboe publishes options, not price history. So this provider serves chains,
expiries and the underlying quote itself, and delegates ``history``,
``fundamentals`` and ``earnings_calendar`` to yfinance. That split plays to
each source's strength rather than pretending one covers everything.

Caveats
-------
* **Delayed**, like every free feed — roughly 15 minutes. Fine for
  positioning work, where open interest is T+1 anyway; not fine for
  execution.
* **Greeks come with Cboe's own rate and dividend assumptions**, which are
  not published. They are used as *supplementary* columns; every exposure
  in ``dealer_greeks`` still recomputes from IV under this project's own
  stated conventions, so nothing silently mixes two models. Set
  ``prefer_exchange_greeks=True`` on the caller if you want the exchange's
  numbers instead, and know that you are then mixing assumptions.
* **Not every contract carries IV or Greeks** — typically 85-97% do. The
  gaps are the illiquid wings, which is exactly where a solver would have
  struggled anyway. Missing values come through as NaN rather than zero so
  ``iv_surface`` can filter them honestly.
* **It does rate-limit.** An earlier version of this docstring claimed
  otherwise; sweeping a universe produced HTTP 429 quickly enough to prove
  it wrong. There is no published quota, so the provider treats 429 as a
  signal to back off: an exponential pause (capped at five minutes) during
  which cached chains are served and no new requests are made. Because one
  request returns *every* expiry, normal single-symbol use rarely comes
  near the limit — it is universe sweeps that do.
* This is an undocumented public endpoint. It could change or disappear;
  the provider degrades to yfinance rather than breaking the app.

Index symbols
-------------
Cboe prefixes cash indices with an underscore: ``SPX`` is ``_SPX``, ``VIX``
is ``_VIX``. :func:`cboe_symbol` handles the mapping so callers keep using
ordinary tickers.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import requests

from .base import DataProvider
from .yfinance_provider import YFinanceProvider

log = logging.getLogger(__name__)

BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"

#: Cash indices Cboe exposes under an underscore prefix.
INDEX_SYMBOLS = {"SPX", "VIX", "NDX", "RUT", "DJX", "OEX", "XSP", "VVIX"}

#: OCC contract symbol: root + YYMMDD + C/P + strike in thousandths.
_OCC_RE = re.compile(r"^(?P<root>[A-Z_]+)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

_CHAIN_COLUMNS = [
    "strike", "lastPrice", "bid", "ask", "volume", "openInterest",
    "impliedVolatility", "delta", "gamma", "vega", "theta", "rho",
    "theo", "contractSymbol", "expiry",
]


def cboe_symbol(symbol: str) -> str:
    """Map an ordinary ticker to Cboe's naming (``SPX`` -> ``_SPX``)."""
    s = symbol.upper().lstrip("^")
    return f"_{s}" if s in INDEX_SYMBOLS else s


def parse_occ(contract: str) -> tuple[str, str, float] | None:
    """Split an OCC symbol into (expiry ISO, right, strike).

    ``AAPL260824C00205000`` -> ``("2026-08-24", "C", 205.0)``.
    Returns None if the symbol does not parse, so a feed change degrades to
    dropped rows rather than wrong strikes.
    """
    m = _OCC_RE.match(contract.strip().upper())
    if not m:
        return None
    ymd = m.group("ymd")
    try:
        expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])).isoformat()
    except ValueError:
        return None
    return expiry, m.group("cp"), int(m.group("strike")) / 1000.0


class CboeProvider(DataProvider):
    """Options from Cboe; price history and fundamentals from yfinance.

    Backoff state is class-level on purpose: every instance is talking to
    the same host under the same (unpublished) quota, so a 429 seen by one
    must pause all of them. A per-instance counter would let the scanner
    hub and the request handlers each keep hammering independently.
    """

    name = "cboe"

    # Shared across instances — see the class docstring.
    _backoff_until: float = 0.0
    _consecutive_429: int = 0

    def __init__(self, ttl: int = 300, timeout: int = 25,
                 fallback: DataProvider | None = None):
        self._ttl = ttl
        self._timeout = timeout
        # yfinance covers what Cboe does not publish, and is the safety net
        # if the endpoint changes shape.
        self._fallback = fallback or YFinanceProvider()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "ApexFlow/1.0 (options research; +https://github.com/nmaltese13/ApexFlow)",
            "Accept": "application/json",
        })

    # -- rate limiting ----------------------------------------------------
    @classmethod
    def is_rate_limited(cls) -> bool:
        return time.time() < cls._backoff_until

    @classmethod
    def backoff_remaining(cls) -> float:
        return max(0.0, cls._backoff_until - time.time())

    @classmethod
    def _trip_backoff(cls) -> None:
        """Exponential pause after a 429: 30s, 60s, 120s … capped at 5 min."""
        cls._consecutive_429 += 1
        wait = min(30 * (2 ** (cls._consecutive_429 - 1)), 300)
        cls._backoff_until = max(cls._backoff_until, time.time() + wait)
        log.warning("Cboe rate-limited; backing off %ds (#%d). Serving cached "
                    "chains until then.", wait, cls._consecutive_429)

    @classmethod
    def _clear_backoff(cls) -> None:
        if cls._consecutive_429:
            log.info("Cboe recovered after %d rate-limit responses",
                     cls._consecutive_429)
        cls._consecutive_429 = 0
        cls._backoff_until = 0.0

    # -- fetching ---------------------------------------------------------
    def _fetch_chain(self, symbol: str) -> pd.DataFrame | None:
        """Whole chain, every expiry, as one tidy frame. Cached."""
        key = symbol.upper()
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self._ttl:
            return hit[1]

        # While backing off, spend nothing on requests that will 429 again.
        # A stale chain is still useful and freshness.py makes its age
        # visible; a burst of failures is useful to nobody.
        if self.is_rate_limited():
            if hit:
                return hit[1]
            return self._fallback_chain(key)

        url = BASE_URL.format(symbol=cboe_symbol(key))
        try:
            r = self._session.get(url, timeout=self._timeout)
            if r.status_code == 404:
                log.info("Cboe has no chain for %s", key)
                self._cache[key] = (now, None)
                return None
            if r.status_code == 429:
                self._trip_backoff()
                return hit[1] if hit else self._fallback_chain(key)
            r.raise_for_status()
            payload = r.json()
            self._clear_backoff()
        except Exception as e:
            log.warning("Cboe fetch failed for %s: %s", key, e)
            # Serve stale rather than nothing.
            return hit[1] if hit else None

        data = payload.get("data") or {}
        rows = data.get("options") or []
        if not rows:
            self._cache[key] = (now, None)
            return None

        df = pd.DataFrame(rows)
        parsed = df["option"].map(parse_occ)
        keep = parsed.notna()
        if not keep.any():
            self._cache[key] = (now, None)
            return None
        df = df[keep].copy()
        parsed = parsed[keep]
        df["expiry"] = [p[0] for p in parsed]
        df["right"] = [p[1] for p in parsed]
        df["strike"] = [p[2] for p in parsed]

        df = df.rename(columns={
            "iv": "impliedVolatility",
            "open_interest": "openInterest",
            "last_trade_price": "lastPrice",
            "option": "contractSymbol",
        })
        for col in ("bid", "ask", "lastPrice", "volume", "openInterest",
                    "impliedVolatility", "delta", "gamma", "vega", "theta",
                    "rho", "theo"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        # Zero IV means "not computed", not "zero volatility" — NaN so that
        # iv_surface's filters treat it as missing instead of implausible.
        df.loc[df["impliedVolatility"] <= 0, "impliedVolatility"] = pd.NA

        spot = data.get("current_price") or data.get("close") or 0.0
        df.attrs["spot"] = float(spot or 0.0)
        df.attrs["fetched_at"] = payload.get("timestamp")

        self._cache[key] = (now, df)
        return df

    # -- DataProvider surface ---------------------------------------------
    def expiries(self, symbol: str) -> list[str]:
        df = self._fetch_chain(symbol)
        if df is None or df.empty:
            return self._fallback.expiries(symbol)
        today = datetime.now(timezone.utc).date().isoformat()
        return sorted({e for e in df["expiry"].unique() if e >= today})

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        df = self._fetch_chain(symbol)
        if df is None or df.empty:
            return self._fallback.options_chain(symbol, expiry)

        exps = self.expiries(symbol)
        if not exps:
            return {"calls": pd.DataFrame(columns=_CHAIN_COLUMNS),
                    "puts": pd.DataFrame(columns=_CHAIN_COLUMNS),
                    "expiry": None, "spot": df.attrs.get("spot", 0.0)}
        target = expiry if expiry in exps else exps[0]

        block = df[df["expiry"] == target]
        out = {}
        for right, name in (("C", "calls"), ("P", "puts")):
            side = block[block["right"] == right]
            cols = [c for c in _CHAIN_COLUMNS if c in side.columns]
            out[name] = (side[cols].sort_values("strike").reset_index(drop=True)
                         if not side.empty else pd.DataFrame(columns=_CHAIN_COLUMNS))
        out["expiry"] = target
        out["spot"] = df.attrs.get("spot", 0.0)
        # Stamp when this payload was actually produced. Without it, a chain
        # served from cache after a failed fetch is indistinguishable from a
        # fresh one — see platform/freshness.py.
        out["as_of"] = df.attrs.get("fetched_at")
        out["source"] = self.name
        return out

    def _fallback_chain(self, symbol: str) -> pd.DataFrame | None:
        """Nothing cached and Cboe is unavailable — return None.

        Deliberately does not silently substitute yfinance chain data here:
        the columns differ (no exchange Greeks), and quietly swapping the
        source mid-session would make the freshness reporting lie about
        where the numbers came from. `options_chain` handles the swap
        explicitly at a level where the source can be reported honestly.
        """
        return None

    def full_chain(self, symbol: str) -> pd.DataFrame | None:
        """Every expiry in one frame — the reason to prefer this provider.

        Callers that roll exposures across expiries can take this once
        instead of looping ``options_chain`` per expiry.
        """
        df = self._fetch_chain(symbol)
        return None if df is None else df.copy()

    def quote(self, symbol: str) -> dict:
        df = self._fetch_chain(symbol)
        spot = float(df.attrs.get("spot", 0.0)) if df is not None else 0.0
        if spot <= 0:
            return self._fallback.quote(symbol)
        # Cboe's option payload carries the underlying price but not volume
        # or the day's range, so fill those from the fallback when cheap.
        base = {}
        try:
            base = self._fallback.quote(symbol) or {}
        except Exception:
            pass
        base.update({"symbol": symbol.upper(), "price": spot})
        return base

    # Cboe publishes options, not price history — delegate the rest.
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        return self._fallback.history(symbol, period=period, interval=interval)

    def fundamentals(self, symbol: str) -> dict:
        return self._fallback.fundamentals(symbol)

    def earnings_calendar(self, symbol: str) -> list[date]:
        return self._fallback.earnings_calendar(symbol)

    def news(self, symbol: str, limit: int = 5) -> list[dict]:
        return self._fallback.news(symbol, limit=limit)


def cboe_available(symbol: str = "SPY", timeout: int = 10) -> bool:
    """Cheap reachability probe, for `status` and the provider selector."""
    try:
        r = requests.head(BASE_URL.format(symbol=cboe_symbol(symbol)),
                          timeout=timeout,
                          headers={"User-Agent": "ApexFlow/1.0"})
        return r.status_code == 200
    except Exception:
        return False
