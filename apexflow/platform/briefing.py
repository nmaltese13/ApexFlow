"""Briefing engine — pre-computes everything you need to walk into a chart cold.

Two cadences run in one background daemon:

  * Watchlist deep brief   — every 10 min while running. For each watchlist
                             symbol, runs the radar scanner + earnings
                             direction predictor + GEX walls + key levels
                             + projection cone. Persists per-symbol JSON.
  * Universe radar pass    — every 30 min. Lighter. Runs the radar across the
                             full universe and stores the leaderboard.

Both run 24/7 (extended hours and weekends). The deep brief uses the most
recent options chain available — on weekends that's Friday's close, which is
exactly what dealers re-price against on Sunday evening.

Persistence: data/briefings/<SYMBOL>.json + data/briefings/_universe.json.
The /api/brief and /api/brief/{sym} endpoints just serve those files plus a
"freshness" timestamp so the UI knows when to nag for a refresh.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import config
from apexflow.analytics.gex import chain_gex, gex_summary
from apexflow.analytics.key_levels import compute_key_levels
from apexflow.analytics.projection import (
    expected_move as proj_em, above_prob as proj_above,
    touch_prob as proj_touch, projection_cone as proj_cone, horizon_years,
)
from apexflow.analytics.earnings_direction import predict_direction
from apexflow.scanners.squeeze_radar import SqueezeRadarScanner
from apexflow.providers import get_provider
from apexflow.platform.watchlist import Watchlist

log = logging.getLogger(__name__)


BRIEF_DIR = config.DATA_DIR / "briefings"
BRIEF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# One symbol → full brief
# ---------------------------------------------------------------------------
def build_brief(symbol: str) -> dict | None:
    """Pull everything we need for `symbol` and return a brief dict.

    Returns None if we can't even get a price for the symbol (provider failure
    or de-listed). All other failures are caught per-section and noted in the
    brief so the UI can show partial data instead of blanking.
    """
    sym = symbol.upper()
    prov = get_provider()
    brief: dict = {
        "symbol": sym,
        "generated_at": int(datetime.now(timezone.utc).timestamp()),
        "errors": {},
    }

    # ---- price + history ----
    try:
        df = prov.history(sym, period="6mo", interval="1d")
    except Exception as e:
        brief["errors"]["history"] = str(e); df = None
    if df is None or df.empty:
        return None
    spot = float(df["Close"].iloc[-1])
    brief["spot"] = spot
    brief["change_pct_5d"] = float((spot / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0.0
    brief["change_pct_20d"] = float((spot / df["Close"].iloc[-21] - 1) * 100) if len(df) > 21 else 0.0

    # 52w price position
    if len(df) > 200:
        win = df.tail(252)
        lo, hi = float(win["Low"].min()), float(win["High"].max())
        brief["pct_of_52w_range"] = float((spot - lo) / (hi - lo)) if hi > lo else None
    else:
        brief["pct_of_52w_range"] = None

    # ---- options chain + GEX ----
    chain = {}; calls = puts = None; expiry = None
    try:
        chain = prov.options_chain(sym)
        calls, puts, expiry = chain.get("calls"), chain.get("puts"), chain.get("expiry")
    except Exception as e:
        brief["errors"]["chain"] = str(e)

    walls = []
    flip = None
    atm_iv = 0.0
    if calls is not None and not calls.empty and expiry:
        try:
            gex_df = chain_gex(calls, puts if puts is not None else None, spot, expiry)
            summary = gex_summary(gex_df, spot)
            flip = summary.get("gamma_flip")
            # Top 8 |gex| strikes
            gex_df["abs_gex"] = gex_df["total_gex"].abs()
            top = gex_df.nlargest(8, "abs_gex").sort_values("strike")
            walls = [{
                "strike":     float(r["strike"]),
                "total_gex":  float(r["total_gex"]),
                "dist_pct":   float((r["strike"] / spot - 1) * 100),
            } for _, r in top.iterrows()]
            # ATM IV
            atm_row = calls.copy()
            atm_row["dist"] = (atm_row["strike"] - spot).abs()
            atm_iv = float(atm_row.nsmallest(1, "dist").iloc[0].get("impliedVolatility") or 0)
        except Exception as e:
            brief["errors"]["gex"] = str(e)
    brief["gamma_walls"] = walls
    brief["gamma_flip"]  = flip
    brief["atm_iv"]      = atm_iv

    # ---- expected move + projection cone ----
    if atm_iv > 0:
        em = proj_em(spot, horizon_years(1.0), atm_iv)
        em_pct = (em / spot * 100) if spot else 0
        brief["em_1d"] = em; brief["em_1d_pct"] = em_pct
        brief["em_5d"] = proj_em(spot, horizon_years(5.0), atm_iv)
        brief["em_5d_pct"] = (brief["em_5d"] / spot * 100) if spot else 0
        brief["projection_cone"] = proj_cone(spot, horizon_years(5.0), atm_iv, n_steps=10)
    else:
        brief["em_1d"] = brief["em_5d"] = 0.0
        brief["em_1d_pct"] = brief["em_5d_pct"] = 0.0
        brief["projection_cone"] = []

    # ---- key levels (PDH/PDL, POC/VAH/VAL, swing, round numbers) ----
    intraday = None
    try:
        for period, interval in (("20d", "30m"), ("60d", "1h")):
            try:
                idf = prov.history(sym, period=period, interval=interval)
                if idf is not None and not idf.empty:
                    intraday = idf; break
            except Exception:
                continue
    except Exception:
        pass
    try:
        brief["key_levels"] = compute_key_levels(df, intraday, spot)
    except Exception as e:
        brief["errors"]["key_levels"] = str(e)
        brief["key_levels"] = {}

    # ---- earnings direction ----
    earnings = []
    try:
        earnings = prov.earnings_calendar(sym) or []
    except Exception:
        pass
    sd = {}
    try:
        sd = prov.short_interest_detail(sym) or {}
    except Exception:
        pass
    fund = prov.fundamentals(sym) or {}
    short_pct = sd.get("short_pct_float") or fund.get("short_pct_float", 0) or 0
    dtc = sd.get("days_to_cover") or fund.get("short_ratio", 0) or 0

    try:
        view = predict_direction(
            calls=calls, puts=puts, spot=spot,
            short_pct_float=short_pct, days_to_cover=dtc,
            history_df=df, earnings_dates=earnings,
            price_pos_52w=brief.get("pct_of_52w_range"),
            cp_oi_history=None,
        )
        brief["direction"] = view.to_dict()
    except Exception as e:
        brief["errors"]["direction"] = str(e)
        brief["direction"] = None

    today = date.today()
    upcoming = [d for d in earnings if d >= today]
    brief["next_earnings"] = upcoming[0].isoformat() if upcoming else None
    brief["days_to_earnings"] = (upcoming[0] - today).days if upcoming else None

    # ---- radar score (single-symbol scan) ----
    try:
        scanner = SqueezeRadarScanner(prov, min_score=0.0)
        sigs = scanner.scan([sym])
        if sigs:
            s = sigs[0]
            brief["radar"] = {
                "score":   s.score, "tags": s.tags, "reason": s.reason,
                "layers":  s.metrics.get("layer_scores", {}),
            }
        else:
            brief["radar"] = {"score": 0, "tags": [], "reason": "no signal", "layers": {}}
    except Exception as e:
        brief["errors"]["radar"] = str(e)
        brief["radar"] = None

    # ---- short interest snapshot ----
    brief["short_interest"] = {
        "short_pct_float": short_pct, "days_to_cover": dtc,
        "borrow_rate": float(sd.get("borrow_rate") or sd.get("fee") or 0),
    }

    return brief


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_brief(brief: dict) -> Path:
    sym = brief["symbol"]
    path = BRIEF_DIR / f"{sym}.json"
    path.write_text(json.dumps(brief, default=str, separators=(",", ":")))
    return path


def read_brief(symbol: str) -> dict | None:
    path = BRIEF_DIR / f"{symbol.upper()}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def all_briefs() -> list[dict]:
    out = []
    for p in BRIEF_DIR.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def write_universe_leaderboard(symbols: Iterable[str]) -> Path:
    """Run the radar across `symbols` and persist the ranked board."""
    prov = get_provider()
    scanner = SqueezeRadarScanner(prov, min_score=0.0)
    sigs = scanner.scan_safe(list(symbols))
    payload = {
        "generated_at": int(datetime.now(timezone.utc).timestamp()),
        "universe_size": len(list(symbols)),
        "signals": [s.to_dict() for s in sigs],
    }
    path = BRIEF_DIR / "_universe.json"
    path.write_text(json.dumps(payload, default=str, separators=(",", ":")))
    return path


def read_universe_leaderboard() -> dict | None:
    path = BRIEF_DIR / "_universe.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Background engine
# ---------------------------------------------------------------------------
@dataclass
class BriefingProfile:
    watchlist_seconds: float = 600.0     # 10 min
    universe_seconds: float = 1800.0     # 30 min
    inter_symbol_pause: float = 0.5
    startup_delay: float = 30.0


class BriefingEngine:
    """Daemon that maintains pre-computed briefs."""

    def __init__(self,
                 universe: Iterable[str],
                 profile: BriefingProfile | None = None):
        self.universe = list(dict.fromkeys(s.upper() for s in universe if s))
        self.profile = profile or BriefingProfile()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_watchlist_run: float = 0.0
        self._last_universe_run: float = 0.0
        self._briefs_written: int = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="briefing")
        self._thread.start()
        log.info("BriefingEngine started — universe=%d watchlist_every=%ds universe_every=%ds",
                 len(self.universe),
                 int(self.profile.watchlist_seconds),
                 int(self.profile.universe_seconds))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "universe_size": len(self.universe),
            "briefs_written": self._briefs_written,
            "last_watchlist_run_unix": self._last_watchlist_run,
            "last_universe_run_unix": self._last_universe_run,
            "watchlist_seconds": self.profile.watchlist_seconds,
            "universe_seconds": self.profile.universe_seconds,
        }

    # ----- one-shot helpers (callable from API) -------------------------
    def refresh_one(self, symbol: str) -> dict | None:
        brief = build_brief(symbol)
        if brief:
            write_brief(brief)
            self._briefs_written += 1
        return brief

    # ----- main loop ----------------------------------------------------
    def _run(self) -> None:
        time.sleep(self.profile.startup_delay)
        while not self._stop.is_set():
            try:
                now = time.time()
                # Watchlist deep brief
                if now - self._last_watchlist_run >= self.profile.watchlist_seconds:
                    self._do_watchlist_pass()
                    self._last_watchlist_run = time.time()
                # Universe leaderboard
                if now - self._last_universe_run >= self.profile.universe_seconds:
                    self._do_universe_pass()
                    self._last_universe_run = time.time()
                # Sleep until the soonest next deadline (clamped 30..600s)
                sleep_for = max(30, min(
                    self.profile.watchlist_seconds - (time.time() - self._last_watchlist_run),
                    self.profile.universe_seconds - (time.time() - self._last_universe_run),
                ))
                self._stop.wait(timeout=min(600, sleep_for))
            except Exception as e:
                log.exception("briefing: unexpected error: %s", e)
                self._stop.wait(timeout=60)

    def _do_watchlist_pass(self) -> None:
        try:
            wl_syms = [e["symbol"] for e in Watchlist().list()]
        except Exception:
            wl_syms = []
        # Always include a tiny core set so the brief page isn't empty when
        # the watchlist is fresh / has been cleared.
        core = ["SPY", "QQQ", "TSLA", "NVDA"]
        targets = list(dict.fromkeys(wl_syms + core))[:30]
        log.info("briefing: watchlist pass on %d symbols", len(targets))
        for sym in targets:
            if self._stop.is_set():
                return
            try:
                brief = build_brief(sym)
                if brief:
                    write_brief(brief)
                    self._briefs_written += 1
            except Exception as e:
                log.debug("briefing: %s failed: %s", sym, e)
            self._stop.wait(timeout=self.profile.inter_symbol_pause)

    def _do_universe_pass(self) -> None:
        if not self.universe:
            return
        log.info("briefing: universe radar pass on %d symbols", len(self.universe))
        try:
            write_universe_leaderboard(self.universe)
        except Exception as e:
            log.debug("briefing: universe pass failed: %s", e)


def default_universe() -> list[str]:
    """Default universe for the radar pass (broader than Atlas to surface
    less-watched names that occasionally squeeze)."""
    return [
        "SPY", "QQQ", "IWM", "DIA",
        "TSLA", "NVDA", "AAPL", "AMZN", "META", "MSFT", "GOOGL", "AVGO",
        "AMD", "NFLX", "BABA", "PLTR", "INTC", "NOK", "CAR", "SMCI",
        "GME", "AMC", "MARA", "RIOT", "HOOD", "COIN", "SOFI", "AFRM",
        "RIVN", "LCID", "F", "GM", "BAC", "JPM", "WFC", "GS",
        "BA", "GE", "DIS", "NKE", "WMT", "TGT", "COST",
        "CVX", "XOM", "OXY", "FCX", "X", "CLF",
        "MRNA", "BNTX", "PFE", "JNJ", "UNH", "LLY", "ABBV",
        "SNAP", "PINS", "RBLX", "U", "DKNG", "PENN",
    ]
