"""ApexFlow web app — FastAPI backend.

Launch:  python main.py web
Then:    http://localhost:8501
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from apexflow.providers import get_provider
from apexflow.scanners import ALL_SCANNERS
from apexflow.platform import ScannerHub, Backtester, SignalLogger, Watchlist, LiveAlertsEngine
from apexflow.universe import load_universe, universe_choices
from apexflow.analytics import indicators as ind
from apexflow.analytics.gex import chain_gex, gex_summary
from apexflow.analytics.greeks import greeks as bs_greeks, years_to_expiry, iv_rank, iv_percentile
from apexflow.analytics.levels import compute_levels as compute_hs_levels
from apexflow.analytics.key_levels import compute_key_levels
from apexflow.analytics.projection import (
    expected_move as proj_em, above_prob as proj_above, touch_prob as proj_touch,
    projection_cone as proj_cone, horizon_years,
)
from apexflow.analytics import montecarlo as mc
from apexflow.analytics.rates import risk_free_rate
from apexflow.platform.freshness import assess as assess_freshness
from apexflow.analytics.timeutil import market_state
from apexflow.analytics.iv_surface import (
    atm_iv as robust_atm_iv, first_usable_atm_iv, iv_term_structure,
    risk_reversal_25d,
)
from apexflow.analytics.dealer_greeks import (
    chain_exposures, roll_up as roll_exposures, exposure_summary,
    vex_summary, gamma_flip_level,
)
from apexflow.platform.briefing import (
    BriefingEngine, BriefingProfile, build_brief, write_brief,
    read_brief, all_briefs, read_universe_leaderboard,
    default_universe as briefing_default_universe,
)

# -----------------------------------------------------------------------------
# App + templates
# -----------------------------------------------------------------------------
log = logging.getLogger("apexflow.webapp")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop the background engines.

    Replaces four @app.on_event handlers, which FastAPI deprecated and will
    eventually remove. Each engine is started defensively: a failure in the
    alerts or briefing loop must not stop the app from serving pages, since
    every page works fine without them.
    """
    for start in (_start_alerts_engine, _start_briefing_engine):
        try:
            start()
        except Exception as e:
            log.warning("%s failed at startup: %s", start.__name__, e)
    try:
        yield
    finally:
        for stop in (_stop_alerts_engine, _stop_briefing_engine):
            try:
                stop()
            except Exception as e:
                log.warning("%s failed at shutdown: %s", stop.__name__, e)


app = FastAPI(title="ApexFlow", docs_url=None, redoc_url=None, lifespan=lifespan)
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


# -----------------------------------------------------------------------------
# Tiny in-memory cache (provider already caches; this avoids repeating scans)
# -----------------------------------------------------------------------------
_cache: dict[str, tuple[float, Any]] = {}

def freshness_for(sym: str | None = None) -> dict:
    """How fresh is the data this request is built on?

    Derived from the provider's own timestamp where it gives one. Returned
    on the payloads that drive decisions so the UI can surface it rather
    than leaving the user to assume the numbers are current.
    """
    prov = get_provider()
    as_of = None
    try:
        if sym:
            q = prov.quote(sym) or {}
            as_of = q.get("as_of") or q.get("timestamp")
            if as_of is None:
                ch = prov.options_chain(sym)
                as_of = ch.get("as_of")
    except Exception as e:
        log.debug("freshness probe failed for %s: %s", sym, e)
    return assess_freshness(as_of, source=getattr(prov, "name", "unknown")).to_dict()


def rate_for(t_years: float) -> float:
    """Tenor-matched risk-free rate, or the static default when disabled.

    Second-order next to the IV and positioning assumptions, but free to get
    right — and rho is meaningless computed against a stale constant.
    """
    if not getattr(config, "LIVE_RATES", True):
        from apexflow.analytics.rates import FALLBACK_RATE
        return FALLBACK_RATE
    try:
        return risk_free_rate(t_years)
    except Exception as e:          # never let a rate lookup break a request
        log.warning("risk-free rate lookup failed: %s", e)
        from apexflow.analytics.rates import FALLBACK_RATE
        return FALLBACK_RATE


def _opt_float(v) -> float | None:
    """float(), but preserves None instead of raising."""
    return None if v is None else float(v)


def cached(key: str, ttl: float, fetch):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    val = fetch()
    _cache[key] = (now, val)
    return val


# -----------------------------------------------------------------------------
# HTML routes
# -----------------------------------------------------------------------------
def asset_version() -> str:
    """Cache-busting token derived from the newest static file's mtime.

    Browsers cache /static/js/*.js aggressively, so an edit can leave a user
    running yesterday's JavaScript against today's markup — which looks
    exactly like "the page is broken" and is invisible from the server side.
    Appending this to every asset URL makes a changed file a changed URL.
    """
    try:
        newest = max(f.stat().st_mtime
                     for pattern in ("js/*.js", "css/*.css")
                     for f in (ROOT / "static").glob(pattern))
        return str(int(newest))
    except (ValueError, OSError):
        return "0"


def _base_ctx(request: Request) -> dict:
    return {
        "request": request,
        "providers": config.configured_providers(),
        "active_provider": get_provider().name,
        "now": datetime.now().strftime("%H:%M:%S"),
        "asset_v": asset_version(),
    }


@app.get("/")
def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html",
        {**_base_ctx(request), "page": "dashboard"})


@app.get("/heatmap")
def page_heatmap_redirect(sym: str = "SPY"):
    """Retired: the per-strike GEX view now lives on the symbol page.

    Kept as a redirect rather than deleted so bookmarks and the browser
    history of anyone who used it keep working.
    """
    return RedirectResponse(f"/symbol/{sym.upper()}#gex", status_code=308)


@app.get("/dealer")
def page_dealer(request: Request):
    return templates.TemplateResponse(request, "dealer.html",
        {**_base_ctx(request), "page": "dealer"})


@app.get("/heatseeker")
def page_heatseeker(request: Request):
    return templates.TemplateResponse(request, "heatseeker.html",
        {**_base_ctx(request), "page": "heatseeker"})


@app.get("/atlas")
def page_atlas(request: Request):
    return templates.TemplateResponse(request, "atlas.html",
        {**_base_ctx(request), "page": "atlas"})


@app.get("/vol")
def page_vol(request: Request):
    return templates.TemplateResponse(request, "vol.html",
        {**_base_ctx(request), "page": "vol"})


@app.get("/radar")
def page_radar(request: Request):
    return templates.TemplateResponse(request, "radar.html",
        {**_base_ctx(request), "page": "radar"})


@app.get("/brief")
def page_brief(request: Request):
    return templates.TemplateResponse(request, "brief.html",
        {**_base_ctx(request), "page": "brief"})


@app.get("/earnings")
def page_earnings(request: Request):
    return templates.TemplateResponse(request, "earnings.html",
        {**_base_ctx(request), "page": "earnings"})


@app.get("/guide")
def page_guide(request: Request):
    return templates.TemplateResponse(request, "guide.html",
        {**_base_ctx(request), "page": "guide"})


@app.get("/journal")
def page_journal(request: Request, limit: int = 50):
    """Watchlist and signal log on one page.

    Two thin list views merged: both answered "what has this thing flagged
    recently", neither justified its own nav slot, and keeping them apart
    meant checking two places for one question.
    """
    limit = max(10, min(limit, 500))
    return templates.TemplateResponse(request, "journal.html",
        {**_base_ctx(request), "page": "journal",
         "entries": Watchlist().list(),
         "records": list(SignalLogger().iter_recent(limit))[::-1],
         "limit": limit})


@app.get("/watchlist")
def page_watchlist_redirect():
    return RedirectResponse("/journal", status_code=308)


@app.get("/log")
def page_log(request: Request, limit: int = 50):
    """Retired in favour of /journal, kept as a redirect for bookmarks."""
    return RedirectResponse(f"/journal?limit={limit}", status_code=308)


def _page_log_unused(request: Request, limit: int = 50):
    """Recent signals.

    Renders `limit` rows server-side. The default was 200, which shipped a
    ~113KB HTML document on every page load for a view where nobody reads
    past the first screen — the tail is available through /api/log, which is
    what a "load more" control should call.
    """
    limit = max(10, min(limit, 500))
    sl = SignalLogger()
    records = list(sl.iter_recent(limit))
    return templates.TemplateResponse(request, "log.html",
        {**_base_ctx(request), "page": "log",
         "records": records[::-1], "limit": limit})


@app.get("/symbol/{sym}")
def page_symbol(request: Request, sym: str):
    return templates.TemplateResponse(request, "symbol.html",
        {**_base_ctx(request), "page": "symbol", "symbol": sym.upper()})


@app.get("/backtest")
def page_backtest_redirect():
    """Retired.

    The page only replayed three price-based scanners, which is the weakest
    question this project can ask. The real validation work — the
    cross-sectional rank test with a permutation null and a coverage gate —
    lives in `python main.py backtest-squeeze`, and a browser form was never
    going to represent it honestly. The Guide explains both.
    """
    return RedirectResponse("/guide#backtesting", status_code=308)


# -----------------------------------------------------------------------------
# JSON API
# -----------------------------------------------------------------------------
@app.get("/api/scan")
def api_scan(universe: str = "data/test_universe.txt",
             scanners: str = "options-flow,pre-breakout,momentum,squeeze,earnings"):
    enabled = [s for s in scanners.split(",") if s in ALL_SCANNERS]
    if not enabled:
        raise HTTPException(400, "no valid scanners")

    def fetch():
        syms = load_universe(universe)
        hub = ScannerHub(universe=syms, enabled=enabled)
        results = hub.run_once()
        return {
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "universe": universe,
            "universe_size": len(syms),
            "scanners": {name: [s.to_dict() for s in sigs] for name, sigs in results.items()},
            "totals": {name: len(sigs) for name, sigs in results.items()},
            "top_signals": sorted(
                (s.to_dict() for sigs in results.values() for s in sigs),
                key=lambda x: x["score"], reverse=True
            )[:20],
        }
    key = f"scan:{universe}:{','.join(sorted(enabled))}"
    return cached(key, ttl=60, fetch=fetch)


@app.get("/api/quote/{sym}")
def api_quote(sym: str):
    sym = sym.upper()
    return cached(f"q:{sym}", ttl=15,
                   fetch=lambda: get_provider().quote(sym))


@app.get("/api/ribbon")
def api_ribbon():
    """Market overview ticker tape."""
    syms = ["SPY", "QQQ", "IWM", "DIA", "^VIX"]
    p = get_provider()
    out = []
    for s in syms:
        try:
            q = cached(f"q:{s}", 15, lambda s=s: p.quote(s))
            price = q.get("price") or 0
            prev = q.get("previous_close") or 0
            chg = price - prev
            chg_pct = (chg / prev * 100) if prev else 0
            out.append({
                "symbol": "VIX" if s == "^VIX" else s,
                "price": price, "change": chg, "change_pct": chg_pct,
            })
        except Exception:
            out.append({"symbol": s, "price": 0, "change": 0, "change_pct": 0})
    return out


@app.get("/api/history/{sym}")
def api_history(sym: str, period: str = "6mo", interval: str = "1d"):
    sym = sym.upper()
    def fetch():
        df = get_provider().history(sym, period=period, interval=interval)
        if df is None or df.empty:
            return []
        return [{
            "t": int(idx.timestamp()),
            "o": float(row["Open"]), "h": float(row["High"]),
            "l": float(row["Low"]),  "c": float(row["Close"]),
            "v": int(row["Volume"]),
        } for idx, row in df.iterrows()]
    return cached(f"h:{sym}:{period}:{interval}", ttl=120, fetch=fetch)


@app.get("/api/indicators/{sym}")
def api_indicators(sym: str):
    sym = sym.upper()
    def fetch():
        prov = get_provider()
        df = prov.history(sym, period="1y", interval="1d")
        if df is None or df.empty or len(df) < 30:
            return {}
        close = df["Close"]; vol = df["Volume"]
        bb = ind.bollinger(close, 20)

        # IV rank from rolling 30-day HV history (proxy when we have no IV history feed)
        import numpy as np
        log_ret = (close / close.shift(1)).apply(lambda x: 0 if x <= 0 else np.log(x))
        hv_window = log_ret.rolling(30, min_periods=20).std() * np.sqrt(252)
        hv_history = [float(v) for v in hv_window.dropna().tolist() if v > 0]
        current_hv = float(ind.historical_volatility(close, 30))

        # ATM IV from the chain, falling back to realised vol. Walk the
        # expiries until one yields a usable estimate: the front expiry is
        # often 0DTE, where vendor IV is numerically meaningless.
        atm_iv = current_hv
        iv_quality = "hv_fallback"
        try:
            chains, spot = [], 0.0
            for exp in (prov.expiries(sym) or [])[:4]:
                chain = prov.options_chain(sym, exp)
                spot = float(chain.get("spot") or spot)
                if spot:
                    chains.append((exp, chain.get("calls"), chain.get("puts")))
            if chains and spot:
                iv_val, q, _ = first_usable_atm_iv(chains, spot)
                if iv_val > 0:
                    atm_iv, iv_quality = iv_val, q
        except Exception as e:
            log.warning("ATM IV lookup failed for %s: %s", sym, e)

        return {
            "rsi": float(ind.rsi(close, 14).iloc[-1] or 0),
            "bb_bw": float(bb["bb_bw"].iloc[-1] or 0),
            "ema20": float(ind.ema(close, 20).iloc[-1] or 0),
            "ema50": float(ind.ema(close, 50).iloc[-1] or 0),
            "rvol": float(ind.rvol(vol, 30).iloc[-1] or 0),
            "hv30": float(current_hv * 100),
            "atr14": float(ind.atr(df["High"], df["Low"], close, 14).iloc[-1] or 0),
            "atm_iv": float(atm_iv * 100),
            "atm_iv_quality": iv_quality,
            # None means "no history to rank against" — pass it through as
            # null rather than coercing to 0.0, which the UI would render as
            # "IV at the bottom of its range".
            "iv_rank": _opt_float(iv_rank(atm_iv, hv_history)),
            "iv_percentile": _opt_float(iv_percentile(atm_iv, hv_history)),
        }
    return cached(f"ind:{sym}", ttl=120, fetch=fetch)


@app.get("/api/gex/{sym}")
def api_gex(sym: str):
    sym = sym.upper()
    def fetch():
        chain = get_provider().options_chain(sym)
        calls, puts, spot, expiry = chain.get("calls"), chain.get("puts"), chain.get("spot"), chain.get("expiry")
        if calls is None or puts is None or calls.empty or not spot:
            return {"strikes": [], "summary": {}, "spot": 0, "expiry": None}
        df = chain_gex(calls, puts, spot, expiry)
        summary = gex_summary(df, spot)
        df = df.sort_values("strike")
        df["dist"] = (df["strike"] - spot).abs()
        df = df.nsmallest(40, "dist").sort_values("strike")
        return {
            "spot": float(spot),
            "expiry": expiry,
            "summary": summary,
            "strikes": [{
                "strike": float(r["strike"]),
                "call_gex": float(r["call_gex"]),
                "put_gex": float(r["put_gex"]),
                "total_gex": float(r["total_gex"]),
            } for _, r in df.iterrows()],
        }
    return cached(f"gex:{sym}", ttl=120, fetch=fetch)


@app.get("/api/iv_surface/{sym}")
def api_iv_surface(sym: str, max_expiries: int = 8):
    """ATM IV per expiry, the term-structure shape, and 25-delta skew.

    Each expiry carries a `quality` flag. Front-month rows are frequently
    `poor` — a 0DTE contract's vendor IV is a numerical artefact, not a
    measurement — and the term-structure shape is computed from the usable
    rows only. See analytics.iv_surface for why this matters.

    `shape` = backwardation means near-dated vol is bid over far-dated:
    the options market is pricing an event inside the front window.
    """
    sym = sym.upper()
    def fetch():
        import pandas as pd
        prov = get_provider()
        try:
            expiries = (prov.expiries(sym) or [])[:max(1, min(max_expiries, 20))]
        except Exception:
            expiries = []
        if not expiries:
            return {"symbol": sym, "spot": 0.0, "points": [], "shape": "unknown",
                    "skew": {}, "atm": {}}

        chains, spot = [], 0.0
        for e in expiries:
            try:
                ch = prov.options_chain(sym, e)
            except Exception:
                continue
            spot = float(ch.get("spot") or spot)
            puts = ch.get("puts") if ch.get("puts") is not None else pd.DataFrame()
            chains.append((e, ch.get("calls"), puts))
        if not chains or spot <= 0:
            return {"symbol": sym, "spot": spot, "points": [], "shape": "unknown",
                    "skew": {}, "atm": {}}

        term = iv_term_structure(chains, spot)
        iv_val, quality, expiry = first_usable_atm_iv(chains, spot)

        # Skew off the first expiry with a trustworthy surface, so the
        # 25-delta strikes are selected using a believable vol.
        skew = {}
        for e, calls, puts in chains:
            res = robust_atm_iv(calls, puts, spot, expiry=e)
            if res and res.quality in ("good", "fair"):
                skew = risk_reversal_25d(calls, puts, spot, years_to_expiry(e))
                skew["expiry"] = e
                break

        return {
            "symbol": sym,
            "spot": spot,
            "atm": {"iv": iv_val, "quality": quality, "expiry": expiry},
            "points": term["points"],
            "shape": term["shape"],
            "slope": term["slope"],
            "front_iv": term["front_iv"],
            "back_iv": term["back_iv"],
            "n_usable": term["n_usable"],
            "skew": skew,
        }
    return cached(f"ivs:{sym}:{max_expiries}", ttl=180, fetch=fetch)


@app.get("/api/atlas/{sym}")
def api_atlas(sym: str, hours: float = 6.5, bucket: str | None = None,
              max_frames: int = 120):
    """Intraday GEX node history for the Atlas replay view.

    Reads the snapshots the AtlasLoop has captured into `data/atlas.db` and
    returns them as a time-ordered list of frames, each holding the
    per-strike node metrics at that instant.

    Returns an empty `frames` list rather than an error when nothing has been
    captured yet — Atlas needs the background loop to have been running, and
    a brand-new install legitimately has no history. `capture_hint` says so,
    so the page can explain itself instead of looking broken.
    """
    sym = sym.upper()

    def fetch():
        from apexflow.analytics.atlas import (
            AtlasStore, compute_node_metrics, session_summary)

        store = AtlasStore()
        latest = store.latest_ts(sym)
        empty = {
            "symbol": sym, "spot": 0.0, "frames": [], "intraday_candles": [],
            "session": session_summary(None), "bucket": bucket,
            "generated_at": int(datetime.now(timezone.utc).timestamp()),
            "capture_hint": (
                "No Atlas snapshots for this symbol yet. The AtlasLoop "
                "captures them during and around market hours; give it "
                "30+ minutes of session time, and check that "
                "APEXFLOW_ATLAS_DISABLE is not set."),
        }
        if latest is None:
            return empty

        from_ts = int(latest - hours * 3600)
        df = store.read_range(sym, from_ts, latest, bucket=bucket)
        if df is None or df.empty:
            return empty

        spot = float(pd.to_numeric(df["spot"], errors="coerce").dropna().iloc[-1]) \
            if "spot" in df.columns and df["spot"].notna().any() else 0.0

        nodes = compute_node_metrics(df, spot=spot or None)

        # Group into frames. Thinning keeps the scrubber responsive on a long
        # session without dropping the most recent state, which is the one
        # the page opens on.
        stamps = sorted(nodes["ts"].unique().tolist())
        if len(stamps) > max_frames:
            step = len(stamps) / max_frames
            keep = {stamps[min(int(i * step), len(stamps) - 1)] for i in range(max_frames)}
            keep.add(stamps[-1])
            stamps = sorted(keep)

        frames = []
        for ts in stamps:
            block = nodes[nodes["ts"] == ts]
            frames.append({
                "ts": int(ts),
                "strikes": [
                    {k: (None if pd.isna(v) else
                         (bool(v) if isinstance(v, (bool,)) else
                          float(v) if isinstance(v, (int, float)) else v))
                     for k, v in row.items() if k != "ts"}
                    for row in block.to_dict(orient="records")
                ],
            })

        candles = []
        try:
            hist = get_provider().history(sym, period="5d", interval="5m")
            if hist is not None and not hist.empty:
                recent = hist[hist.index >= pd.Timestamp(from_ts, unit="s", tz="UTC")]
                for idx, row in recent.iterrows():
                    candles.append({
                        "t": int(pd.Timestamp(idx).timestamp()),
                        "o": float(row["Open"]), "h": float(row["High"]),
                        "l": float(row["Low"]), "c": float(row["Close"]),
                    })
        except Exception as e:
            log.warning("atlas candles failed for %s: %s", sym, e)

        return {
            "symbol": sym,
            "spot": spot,
            "bucket": bucket,
            "frames": frames,
            "intraday_candles": candles,
            "session": session_summary(df),
            "generated_at": int(datetime.now(timezone.utc).timestamp()),
        }

    return cached(f"atlas:{sym}:{hours}:{bucket}:{max_frames}", ttl=30, fetch=fetch)


@app.get("/api/atlas_symbols")
def api_atlas_symbols():
    """Symbols that currently have captured Atlas history."""
    from apexflow.analytics.atlas import AtlasStore
    try:
        return {"symbols": AtlasStore().symbols()}
    except Exception as e:
        log.warning("atlas symbols failed: %s", e)
        return {"symbols": []}


@app.get("/api/dealer_greeks/{sym}")
def api_dealer_greeks(sym: str, dte_max: int = 30, convention: str = "naive",
                      n_strikes: int = 40, basis: str = "openInterest"):
    """Full dealer-exposure surface for one ticker: DEX, GEX, VEX, charm, vanna.

    Query params:
      dte_max     roll every expiry within this many days onto one strike axis
      convention  naive | inverted | all_short — the dealer-positioning
                  assumption. Everything returned is conditional on it;
                  see analytics.dealer_greeks for what each one means.
      basis       openInterest (the resting book) or volume (today's adds)

    `gamma_flip` here is the re-priced zero-gamma *level*, not the
    cumulative-sum strike — the two differ substantially on put-heavy index
    chains.
    """
    sym = sym.upper()
    def fetch():
        from datetime import datetime, timedelta
        import pandas as pd
        prov = get_provider()
        try:
            expiries = prov.expiries(sym) or []
        except Exception:
            expiries = []
        if not expiries:
            return {"symbol": sym, "spot": 0.0, "strikes": [], "summary": {},
                    "vex": {}, "expiries_used": []}

        cutoff = datetime.now(timezone.utc).date() + timedelta(days=dte_max)
        used = []
        for e in expiries:
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d <= cutoff:
                used.append(e)
        if not used:
            used = [expiries[0]]

        frames, raw, spot = [], [], 0.0
        for exp in used:
            try:
                ch = prov.options_chain(sym, exp)
            except Exception:
                continue
            calls, puts = ch.get("calls"), ch.get("puts")
            spot = float(ch.get("spot") or spot)
            if (calls is None or calls.empty) and (puts is None or puts.empty):
                continue
            puts_df = puts if puts is not None else pd.DataFrame()
            raw.append((exp, calls, puts_df))
            frames.append(chain_exposures(calls, puts_df, spot, exp,
                                          convention=convention, oi_col=basis,
                                          r=rate_for(years_to_expiry(exp))))
        if not frames or spot <= 0:
            return {"symbol": sym, "spot": spot, "strikes": [], "summary": {},
                    "vex": {}, "expiries_used": used}

        rolled = roll_exposures(frames)
        summary = exposure_summary(rolled, spot)
        vex = vex_summary(rolled, spot)

        try:
            flip = gamma_flip_level(raw, spot, convention=convention,
                                    r=rate_for(years_to_expiry(used[0])))
            summary["gamma_flip"] = flip.get("flip")
            summary["gamma_flip_bracketed"] = flip.get("bracketed", False)
            summary["regime"] = flip.get("regime", summary.get("regime"))
        except Exception as e:
            log.warning("gamma_flip_level failed for %s: %s", sym, e)

        near = rolled.copy()
        near["dist"] = (near["strike"] - spot).abs()
        near = near.nsmallest(n_strikes, "dist").sort_values("strike")

        cols = ["dex", "gex", "vex", "charm_exp", "vanna_exp"]
        return {
            "symbol": sym,
            "spot": spot,
            "freshness": freshness_for(sym),
            "convention": convention,
            "basis": basis,
            "expiries_used": used,
            "summary": summary,
            "vex": vex,
            "strikes": [
                {"strike": float(r["strike"]),
                 **{c: float(r.get(c, 0.0)) for c in cols}}
                for _, r in near.iterrows()
            ],
        }
    return cached(f"dg:{sym}:{dte_max}:{convention}:{n_strikes}:{basis}",
                  ttl=120, fetch=fetch)


@app.get("/api/mc/validate")
def api_mc_validate(paths: int = 200_000, steps: int = 128):
    """Run the Monte Carlo validation harness and return the comparison table.

    Exposed over HTTP so the claim "the simulator agrees with closed form"
    is checkable from the running app, not just from a test file.
    """
    paths = max(10_000, min(int(paths), 2_000_000))
    steps = max(8, min(int(steps), 512))
    rows = mc.validate(n_paths=paths, n_steps=steps)
    worst = max((abs(r["z"]) for r in rows), default=0.0)
    return {
        "rows": rows,
        "max_abs_z": worst,
        "passed": worst < 4.0,
        "note": ("z is the error in Monte Carlo standard errors. |z| < ~3-4 "
                 "across all rows is consistent with a correct simulator."),
    }


@app.get("/api/chain/{sym}")
def api_chain(sym: str, expiry: str | None = None):
    sym = sym.upper()
    def fetch():
        chain = get_provider().options_chain(sym, expiry)
        calls, puts, spot = chain.get("calls"), chain.get("puts"), chain.get("spot") or 0
        exp = chain.get("expiry")
        t = years_to_expiry(exp) if exp else 0.0
        out = {"spot": float(spot), "expiry": exp, "calls": [], "puts": []}

        import math
        def _f(v):
            try:
                f = float(v)
                return 0.0 if math.isnan(f) else f
            except (TypeError, ValueError):
                return 0.0
        def _i(v):
            return int(_f(v))

        def row_with_greeks(r, right: str):
            iv = _f(r.get("impliedVolatility"))
            strike = _f(r["strike"])
            g = bs_greeks(spot, strike, t, iv, right=right) if (spot and t and iv) else {
                "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
            }
            return {
                "strike": strike,
                "bid": _f(r.get("bid")),
                "ask": _f(r.get("ask")),
                "last": _f(r.get("lastPrice")),
                "vol": _i(r.get("volume")),
                "oi":  _i(r.get("openInterest")),
                "iv":  iv,
                "delta": g["delta"], "gamma": g["gamma"],
                "vega":  g["vega"],  "theta": g["theta"], "rho": g["rho"],
            }
        if calls is not None and not calls.empty:
            out["calls"] = [row_with_greeks(r, "C") for _, r in calls.iterrows()]
        if puts is not None and not puts.empty:
            out["puts"] = [row_with_greeks(r, "P") for _, r in puts.iterrows()]
        return out
    return cached(f"chain:{sym}:{expiry or 'next'}", ttl=120, fetch=fetch)


@app.get("/api/expiries/{sym}")
def api_expiries(sym: str):
    sym = sym.upper()
    return cached(f"exp:{sym}", ttl=600, fetch=lambda: get_provider().expiries(sym))


@app.get("/api/watchlist")
def api_watchlist():
    return Watchlist().list()


@app.delete("/api/watchlist")
def api_watchlist_clear():
    wl = Watchlist()
    wl.entries = {}
    wl.save()
    return {"cleared": True}


@app.get("/api/log")
def api_log(limit: int = 100):
    sl = SignalLogger()
    return list(sl.iter_recent(limit))[::-1]


@app.post("/api/backtest")
def api_backtest(scanner: str, universe: str = "data/test_universe.txt",
                 lookback: int = 90, hold: int = 5):
    if scanner not in ALL_SCANNERS:
        raise HTTPException(400, "unknown scanner")
    bt = Backtester()
    syms = load_universe(universe)
    try:
        res = bt.run(scanner, syms, lookback_days=lookback, hold_days=hold)
    except ValueError as e:
        # Options-driven scanners cannot be replayed on free data. Say so
        # with the reason rather than 500-ing or returning an empty result
        # that reads as "no signals fired".
        raise HTTPException(400, str(e))
    return {
        "scanner": res.scanner,
        "signals_tested": res.signals_tested,
        "wins": res.wins, "losses": res.losses,
        "win_rate": res.win_rate,
        "avg_return_pct": res.avg_return_pct,
        "median_return_pct": res.median_return_pct,
        "sharpe": res.sharpe,
        "sortino": res.sortino,
        "max_drawdown_pct": res.max_drawdown_pct,
        "calmar": res.calmar,
        "profit_factor": res.profit_factor,
        "best_trade_pct": res.best_trade_pct,
        "worst_trade_pct": res.worst_trade_pct,
        "longest_win_streak": res.longest_win_streak,
        "longest_loss_streak": res.longest_loss_streak,
        "equity_curve": res.equity_curve,
        "returns": res.returns,
        "summary": res.summary,
    }


@app.get("/api/projection/{sym}")
def api_projection(sym: str, dte_max: int = 7, horizon_days: float = 1.0,
                   n_walls: int = 8, model: str = "gbm",
                   paths: int = mc.DEFAULT_PATHS, steps: int = 64,
                   simulate: bool = True):
    """Forward distribution for a single ticker, analytic and simulated.

    Query params:
      dte_max       roll GEX across expiries within this many days
      horizon_days  projection horizon
      model         gbm | merton | bootstrap  (see analytics.montecarlo)
      paths/steps   Monte Carlo size; steps only affects path-dependent output
      simulate      set false to skip the simulation and return closed-form only

    Returns closed-form values under `analytic` and simulated ones under
    `mc`, deliberately side by side: where they disagree, the disagreement
    is the interesting part (fat tails, jump risk, discrete monitoring), and
    hiding one behind the other would make that invisible.

    Every gamma wall carries both `above_prob` (P(S_T beyond it) - a
    terminal question) and `touch_prob` (P(tagged at any point) - a path
    question). Touch is always the larger of the two.
    """
    sym = sym.upper()
    def fetch():
        from datetime import datetime, timedelta
        prov = get_provider()

        # Spot + recent candles
        df = prov.history(sym, period="3mo", interval="1d")
        if df is None or df.empty:
            return {"symbol": sym, "spot": 0.0, "atm_iv": 0.0, "horizon_days": horizon_days,
                    "expected_move": 0.0, "expected_move_pct": 0.0,
                    "candles": [], "gamma_walls": [], "cone": [], "summary": {}}
        spot = float(df["Close"].iloc[-1])
        candles = [{
            "t": int(idx.timestamp()),
            "o": float(row["Open"]), "h": float(row["High"]),
            "l": float(row["Low"]),  "c": float(row["Close"]),
            "v": int(row["Volume"]) if not (row["Volume"] != row["Volume"]) else 0,
        } for idx, row in df.tail(60).iterrows()]

        # Aggregate GEX across expiries within dte_max
        try:
            expiries = prov.expiries(sym) or []
        except Exception:
            expiries = []

        today = datetime.now(timezone.utc).date()
        cutoff = today + timedelta(days=dte_max)
        used = []
        for e in expiries:
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d <= cutoff:
                used.append(e)
        if not used and expiries:
            used = [expiries[0]]

        import pandas as pd
        agg = []
        raw_chains = []
        atm_iv = 0.0
        for exp in used:
            try:
                chain = prov.options_chain(sym, exp)
            except Exception:
                continue
            calls, puts = chain.get("calls"), chain.get("puts")
            if calls is not None and not calls.empty:
                puts_df = puts if puts is not None else pd.DataFrame()
                raw_chains.append((exp, calls, puts_df))
                df_g = chain_gex(calls, puts_df, spot, exp)
                if not df_g.empty:
                    agg.append(df_g)
        # ATM IV: vega-weighted median of the tradeable near-the-money
        # strikes, taking the first expiry that produces a usable estimate.
        # Picking the single nearest strike (the old behaviour) reads pure
        # noise off a 0DTE chain — see analytics.iv_surface.
        atm_iv, iv_quality, iv_expiry = first_usable_atm_iv(raw_chains, spot)

        if atm_iv <= 0:
            atm_iv = float(ind.historical_volatility(df["Close"], 30))
            iv_quality, iv_expiry = "hv_fallback", None

        # Horizon
        t = horizon_years(horizon_days)
        # Named `rf`, not `r`: the gamma-wall loop below binds `r` to each
        # DataFrame row, and shadowing the rate there silently passes a
        # pandas Series into the pricing maths.
        rf = rate_for(t)
        em = proj_em(spot, t, atm_iv) if atm_iv > 0 else 0.0
        em_pct = (em / spot * 100) if spot else 0.0

        # Magnet strikes from aggregated GEX
        walls = []
        strike_list: list[float] = []
        if agg:
            full = pd.concat(agg, ignore_index=True)
            rolled = full.groupby("strike", as_index=False)[["call_gex","put_gex","total_gex"]].sum()
            rolled["abs_gex"] = rolled["total_gex"].abs()
            rolled = rolled.sort_values("abs_gex", ascending=False).head(n_walls)
            rolled = rolled.sort_values("strike", ascending=False)
            for _, r in rolled.iterrows():
                K = float(r["strike"])
                strike_list.append(K)
                walls.append({
                    "strike":     K,
                    "call_gex":   float(r["call_gex"]),
                    "put_gex":    float(r["put_gex"]),
                    "total_gex":  float(r["total_gex"]),
                    "dist_pct":   float((K / spot - 1) * 100) if spot else 0.0,
                    "above_prob": proj_above(spot, K, t, atm_iv, rf) if atm_iv > 0 else 0.5,
                    "touch_prob": proj_touch(spot, K, t, atm_iv, rf) if atm_iv > 0 else 0.5,
                    "sigma_dist": float((K - spot) / em) if em > 0 else 0.0,
                })

        # Zero-gamma level, computed by re-pricing the book across candidate
        # spot levels rather than by cumulative sum. See the long note in
        # dealer_greeks.gamma_flip_level for why the two differ so much.
        flip = None
        flip_detail = {}
        if raw_chains:
            try:
                flip_detail = gamma_flip_level(raw_chains, spot, r=rf)
                flip = flip_detail.get("flip")
            except Exception as e:
                log.warning("gamma_flip_level failed for %s: %s", sym, e)

        cone = proj_cone(spot, t, atm_iv, n_steps=30, r=rf) if atm_iv > 0 else []

        out = {
            "symbol": sym,
            "spot": float(spot),
            "atm_iv": float(atm_iv * 100),
            "atm_iv_quality": iv_quality,
            "atm_iv_expiry": iv_expiry,
            "horizon_days": float(horizon_days),
            "horizon_years": float(t),
            "risk_free_rate": float(rf),
            "expected_move": float(em),
            "expected_move_pct": float(em_pct),
            "expiries_used": used,
            "candles": candles,
            "gamma_walls": walls,
            "cone": cone,
            "gamma_flip": flip,
            "freshness": freshness_for(sym),
            "gamma_flip_detail": {
                "regime": flip_detail.get("regime", "unknown"),
                "bracketed": flip_detail.get("bracketed", False),
                "total_gex_at_spot": flip_detail.get("total_gex_at_spot", 0.0),
            },
            "analytic": {
                "expected_move": float(em),
                "cone": cone,
                "method": "closed-form GBM",
            },
            "mc": None,
        }

        if simulate and atm_iv > 0 and spot > 0:
            try:
                out["mc"] = _run_projection_mc(
                    spot, t, atm_iv, strike_list, df["Close"],
                    model=model, paths=paths, steps=steps, r=rf)
            except Exception as e:
                log.warning("Monte Carlo failed for %s: %s", sym, e)
                out["mc"] = {"error": str(e)}
        return out
    key = f"proj:{sym}:{dte_max}:{horizon_days}:{n_walls}:{model}:{paths}:{steps}:{simulate}"
    return cached(key, ttl=60, fetch=fetch)


def _run_projection_mc(spot: float, t: float, iv: float, strikes: list[float],
                       closes, model: str, paths: int, steps: int,
                       r: float = 0.04) -> dict:
    """Simulate the forward distribution and the gamma-wall touch probabilities.

    Path counts are capped here so a browser refresh cannot pin a core; the
    validation CLI is where the large runs belong.
    """
    import numpy as np

    paths = max(1_000, min(int(paths), 2_000_000))
    steps = max(1, min(int(steps), 512))
    model = (model or "gbm").lower()

    kwargs = {}
    if model == "bootstrap":
        # Use the ticker's own daily log returns, so the simulated shape
        # carries its realised skew/kurtosis rather than a normal's.
        c = closes.dropna()
        rets = np.diff(np.log(c.to_numpy(dtype=float)))
        rets = rets[np.isfinite(rets)]
        if rets.size < 20:
            model = "gbm"
        else:
            kwargs["returns"] = rets
    elif model == "merton":
        # A generic single-name jump prior: ~4 jumps/yr, zero mean, 4% sd.
        # Not calibrated per ticker - stated here so it is not mistaken for
        # a fitted parameter.
        kwargs.update(jump_intensity=4.0, jump_mean=0.0, jump_vol=0.04)

    cfg = mc.MCConfig(n_paths=paths, n_steps=steps, model=model,
                      chunk_paths=min(paths, 100_000), **kwargs)
    res = mc.simulate_cone(spot, t, iv, cfg, r=r)
    payload = res.to_dict()

    if strikes:
        touch = mc.touch_probabilities(spot, strikes, t, iv, cfg, r=r)
        payload["walls"] = [{
            "strike": float(k),
            "touch_prob": touch[k]["touch"],
            "touch_se": touch[k]["touch_se"],
            "beyond_prob": touch[k]["beyond"],
        } for k in strikes if k in touch]
    else:
        payload["walls"] = []

    payload["notes"] = {
        "bridge_correction": True,
        "antithetic": True,
        "se_note": "touch_se is the Monte Carlo standard error of that probability",
    }
    return payload


@app.get("/api/keylevels/{sym}")
def api_keylevels(sym: str, lookback_days: int = 20):
    """Real reactive S/R levels for the chart overlay.

    Combines:
      - Prior-day H/L/C
      - Volume-profile POC, VAH, VAL on the recent intraday window
      - High/Low volume nodes (acceptance / rejection levels)
      - Williams-fractal swing highs/lows that haven't been broken
      - Round-number magnets near spot
    """
    sym = sym.upper()
    def fetch():
        prov = get_provider()
        spot = 0.0
        try:
            q = prov.quote(sym)
            spot = float(q.get("price") or 0)
        except Exception:
            pass

        try:
            daily = prov.history(sym, period="6mo", interval="1d")
        except Exception:
            daily = None
        if (not spot) and daily is not None and not daily.empty:
            spot = float(daily["Close"].iloc[-1])

        # Intraday for volume profile — try a few window/interval combos so
        # different providers cooperate. yfinance for example only allows 60d
        # of 5m data and 730d of 1h.
        intraday = None
        for period, interval in (("20d", "30m"), ("30d", "1h"), ("60d", "1h")):
            try:
                df = prov.history(sym, period=period, interval=interval)
                if df is not None and not df.empty:
                    intraday = df
                    break
            except Exception:
                continue

        # Trim to roughly the requested lookback (intraday)
        if intraday is not None and not intraday.empty:
            intraday = intraday.tail(lookback_days * 14)  # ~14 30m bars per session

        levels = compute_key_levels(daily if daily is not None else None,
                                    intraday if intraday is not None else None,
                                    spot)
        levels["symbol"] = sym
        levels["spot"] = float(spot)
        return levels
    return cached(f"klv:{sym}:{lookback_days}", ttl=120, fetch=fetch)


# --- Heatseeker layer parsing helpers ---
# Legacy bucket presets — still supported in `layers=` for backwards compat,
# but the UI now drives via `dtes=0,1,2,7,30` (single-DTE columns, max 5).
PRESET_LAYERS = {
    "0dte":   {"label": "0DTE",   "min": 0,  "max": 0},
    "front":  {"label": "Front",  "min": 0,  "max": 2},
    "middle": {"label": "Middle", "min": 7,  "max": 30},
    "back":   {"label": "Back",   "min": 45, "max": 365},
    "all":    {"label": "All",    "min": 0,  "max": 365},
}

MAX_LAYERS = 5


def _label_for_dte(d: int) -> str:
    if d == 0:
        return "0DTE"
    return f"{d}DTE"


def _parse_layers(layers: str | None, dtes: str | None,
                  dte_min: int, dte_max: int) -> list[dict]:
    """Build the list of DTE columns to render.

    Priority:
      1. ``dtes=0,1,2,7,30`` — comma-separated list of *individual* DTE
         values. Each one becomes its own column. Capped at ``MAX_LAYERS``.
      2. ``layers=front,middle,7-30`` — preset names or numeric ranges.
      3. fallback to a single `dte_min`-`dte_max` window (legacy).
    """
    out: list[dict] = []

    if dtes:
        for raw in dtes.split(","):
            tok = raw.strip()
            if not tok:
                continue
            try:
                d = int(tok)
            except ValueError:
                continue
            if d < 0:
                continue
            out.append({"label": _label_for_dte(d), "min": d, "max": d})

    if not out and layers:
        for raw in layers.split(","):
            tok = raw.strip().lower()
            if not tok:
                continue
            if tok in PRESET_LAYERS:
                out.append(dict(PRESET_LAYERS[tok]))
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    lo, hi = int(a), int(b)
                except ValueError:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                out.append({"label": f"{lo}-{hi} DTE", "min": lo, "max": hi})
                continue
            try:
                d = int(tok)
                out.append({"label": _label_for_dte(d), "min": d, "max": d})
            except ValueError:
                pass

    if not out:
        out.append({"label": f"{dte_min}-{dte_max} DTE",
                    "min": int(dte_min), "max": int(dte_max)})

    seen = set(); deduped = []
    for L in out:
        key = (L["min"], L["max"])
        if key not in seen:
            seen.add(key); deduped.append(L)
    return deduped[:MAX_LAYERS]


def _build_layer(sym: str, spot_hint: float, expiries: list[str], lo: int, hi: int,
                 n_strikes: int, king_count: int, gatekeeper_count: int) -> dict | None:
    """Aggregate GEX + compute levels for one DTE band. Returns None if empty."""
    from datetime import datetime, timedelta
    import pandas as pd

    today = datetime.now(timezone.utc).date()
    win_lo = today + timedelta(days=max(0, lo))
    win_hi = today + timedelta(days=max(0, hi))
    used: list[str] = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        if win_lo <= d <= win_hi:
            used.append(e)

    spot = float(spot_hint or 0)
    raw_chains: list[tuple[str, "pd.DataFrame", "pd.DataFrame"]] = []
    agg_frames = []
    prov = get_provider()
    for exp in used:
        try:
            chain = prov.options_chain(sym, exp)
        except Exception:
            continue
        calls, puts = chain.get("calls"), chain.get("puts")
        spot = spot or float(chain.get("spot") or 0)
        if calls is None or calls.empty:
            continue
        raw_chains.append((exp, calls, puts if puts is not None else pd.DataFrame()))
        df = chain_gex(calls, puts if puts is not None else pd.DataFrame(), spot, exp)
        if not df.empty:
            df["expiry"] = exp
            agg_frames.append(df)

    if not agg_frames or not spot:
        return {
            "expiries_used": used,
            "strikes": [],
            "max_abs_gex": 0.0,
            "summary": {},
            "levels": {},
            "stacks": [],
        }

    full = pd.concat(agg_frames, ignore_index=True)
    rolled_full = full.groupby("strike", as_index=False)[["call_gex","put_gex","total_gex"]].sum()
    # Levels are computed on the FULL rolled chain (not just the windowed view)
    levels = compute_hs_levels(raw_chains, rolled_full, spot)

    rolled = rolled_full.copy()
    rolled["abs_gex"] = rolled["total_gex"].abs()
    rolled["dist"] = (rolled["strike"] - spot).abs()
    rolled = rolled.nsmallest(n_strikes, "dist").sort_values("strike", ascending=False).reset_index(drop=True)

    max_abs = float(rolled["abs_gex"].max()) if not rolled.empty else 0.0
    ranked = rolled.sort_values("abs_gex", ascending=False).head(king_count + gatekeeper_count).index.tolist()
    king_idx = ranked[:king_count]
    gate_idx = ranked[king_count:king_count + gatekeeper_count]

    node_types: list[str | None] = []
    is_air: list[bool] = []
    for i, r in rolled.iterrows():
        mag_ratio = (r["abs_gex"] / max_abs) if max_abs else 0.0
        if i in king_idx: node_types.append("king")
        elif i in gate_idx: node_types.append("gatekeeper")
        elif mag_ratio < 0.10: node_types.append("air")
        else: node_types.append(None)
        is_air.append(mag_ratio < 0.10)

    # Stacked-node detection (rug-pull / slingshot) on adjacent strikes
    stacks = []
    sorted_rows = rolled.sort_values("strike").reset_index(drop=True)
    for j in range(len(sorted_rows) - 1):
        a = sorted_rows.iloc[j]; b = sorted_rows.iloc[j + 1]
        if a["total_gex"] * b["total_gex"] >= 0:
            continue
        mag = max(abs(a["total_gex"]), abs(b["total_gex"]))
        if max_abs and mag / max_abs < 0.25:
            continue
        kind = "rug_pull" if a["total_gex"] < 0 and b["total_gex"] > 0 else "slingshot"
        stacks.append({
            "lower_strike": float(a["strike"]),
            "upper_strike": float(b["strike"]),
            "kind": kind,
            "magnitude": float(mag),
        })

    summary = gex_summary(rolled_full, spot)

    return {
        "expiries_used": used,
        "max_abs_gex": float(max_abs),
        "strikes": [{
            "strike":    float(r["strike"]),
            "call_gex":  float(r["call_gex"]),
            "put_gex":   float(r["put_gex"]),
            "total_gex": float(r["total_gex"]),
            "node_type": node_types[i],
            "is_air":    bool(is_air[i]),
        } for i, r in rolled.iterrows()],
        "stacks": stacks,
        "levels": levels,
        "profile": _gex_profile_for_layer(rolled_full, spot),
        "summary": {
            "total_gex": float(summary.get("total_gex") or 0),
            "gamma_flip": float(summary.get("gamma_flip")) if summary.get("gamma_flip") else None,
            "max_pos_strike": float(summary.get("max_pos_strike")) if summary.get("max_pos_strike") is not None else None,
            "max_neg_strike": float(summary.get("max_neg_strike")) if summary.get("max_neg_strike") is not None else None,
        },
    }


def _gex_profile_for_layer(rolled_full, spot):
    """Wrapper around gex_profile.profile_chain that's safe on empty data."""
    try:
        from apexflow.analytics.gex_profile import profile_chain
        return profile_chain(rolled_full, spot)
    except Exception:
        return {}


@app.get("/api/heatseeker")
def api_heatseeker(symbols: str = "SPY,QQQ,IWM",
                   dte_min: int = 0, dte_max: int = 7,
                   layers: str | None = None,
                   dtes: str | None = None,
                   n_strikes: int = 40,
                   king_count: int = 1, gatekeeper_count: int = 4):
    """Multi-DTE Skylit-style GEX matrix.

    Pass `layers` as a comma-separated list — preset names (front, middle,
    back, all, 0dte) or numeric ranges like `0-2,7-30,45-365` — to compute
    several DTE bands in a single request. Each ticker comes back with one
    entry per layer plus a unified `strike_grid` listing every strike that
    appears across the layers (so the client can render row-by-row).

    Each layer also includes self-computed levels (HVL, walls, zero-gamma,
    vol trigger, charm/vanna peaks) derived directly from the chain.
    """
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(400, "no symbols")
    if len(syms) > 8:
        syms = syms[:8]

    layer_specs = _parse_layers(layers, dtes, dte_min, dte_max)

    def fetch_one(sym: str):
        prov = get_provider()
        try:
            expiries = prov.expiries(sym) or []
        except Exception:
            expiries = []

        # Spot (one quote, shared across layers)
        spot = 0.0
        change_pct = 0.0
        try:
            q = prov.quote(sym)
            spot = float(q.get("price") or 0)
            prev = q.get("previous_close") or spot
            change_pct = ((q.get("price") or spot) - prev) / prev * 100 if prev else 0.0
        except Exception:
            pass

        if not expiries:
            return {
                "symbol": sym, "spot": float(spot), "change_pct": float(change_pct),
                "layers": [], "strike_grid": [],
            }

        layers_out = []
        for spec in layer_specs:
            payload = _build_layer(sym, spot, expiries, spec["min"], spec["max"],
                                   n_strikes, king_count, gatekeeper_count) or {}
            layers_out.append({
                "label":   spec["label"],
                "dte_min": spec["min"],
                "dte_max": spec["max"],
                **payload,
            })

        # Build unified strike grid (union of all layer strikes near spot)
        strike_set: set[float] = set()
        for L in layers_out:
            for s in L.get("strikes", []):
                strike_set.add(round(float(s["strike"]), 4))
        if spot:
            grid_sorted = sorted(strike_set, key=lambda k: abs(k - spot))[: n_strikes]
            strike_grid = sorted(grid_sorted, reverse=True)
        else:
            strike_grid = sorted(strike_set, reverse=True)[: n_strikes]

        return {
            "symbol": sym,
            "spot": float(spot),
            "change_pct": float(change_pct),
            "layers": layers_out,
            "strike_grid": strike_grid,
        }

    layers_key = ",".join(f"{L['min']}-{L['max']}" for L in layer_specs)
    key = f"hs:{','.join(syms)}:L[{layers_key}]:{n_strikes}:{king_count}:{gatekeeper_count}"
    def fetch_all():
        return {
            "symbols": [fetch_one(s) for s in syms],
            "layers":  layer_specs,
        }
    return cached(key, ttl=60, fetch=fetch_all)



# -----------------------------------------------------------------------------
# Earnings direction predictor
# -----------------------------------------------------------------------------
@app.get("/api/earnings_direction/{sym}")
def api_earnings_direction(sym: str):
    """Predict which way earnings will move for `sym`.

    Returns:
      direction      "bullish" | "bearish" | "neutral"
      confidence     0..1
      composite      -1..+1 raw signed signal
      layers         per-layer signal + supporting numbers
      reason         which layers contributed
    """
    from apexflow.analytics.earnings_direction import predict_direction
    sym = sym.upper()

    def fetch():
        prov = get_provider()
        try:
            chain = prov.options_chain(sym)
        except Exception:
            chain = {}
        calls, puts = chain.get("calls"), chain.get("puts")
        spot = float(chain.get("spot") or 0)

        df = prov.history(sym, period="2y", interval="1d")
        if (not spot) and df is not None and not df.empty:
            spot = float(df["Close"].iloc[-1])

        # 52w price position
        pos52 = None
        if df is not None and len(df) > 200:
            window = df.tail(252)
            lo, hi = float(window["Low"].min()), float(window["High"].max())
            if hi > lo:
                pos52 = float((spot - lo) / (hi - lo))

        # Short interest
        sd = {}
        try:
            sd = prov.short_interest_detail(sym) or {}
        except Exception:
            pass
        fund = prov.fundamentals(sym) or {}
        short_pct = sd.get("short_pct_float") or fund.get("short_pct_float", 0) or 0
        dtc = sd.get("days_to_cover") or fund.get("short_ratio", 0) or 0

        # Earnings
        earnings = []
        try:
            earnings = prov.earnings_calendar(sym) or []
        except Exception:
            pass

        view = predict_direction(
            calls=calls, puts=puts, spot=spot,
            short_pct_float=short_pct, days_to_cover=dtc,
            history_df=df, earnings_dates=earnings,
            price_pos_52w=pos52,
            cp_oi_history=None,   # not available from yfinance; placeholder
        )
        d = view.to_dict()
        d.update({"symbol": sym, "spot": spot, "earnings_dates_seen": [e.isoformat() for e in earnings[-8:]]})
        return d

    return cached(f"earndir:{sym}", ttl=300, fetch=fetch)


# -----------------------------------------------------------------------------
# Squeeze Radar — composite scanner across the universe
# -----------------------------------------------------------------------------
@app.get("/api/radar")
def api_radar(universe: str = "data/test_universe.txt", min_score: float = 50.0,
              limit: int = 50):
    """Run the SqueezeRadarScanner across `universe` and return ranked signals.

    Each signal carries a `metrics.layer_scores` block (short, gex,
    compression, rvol, catalyst, trend, synergy) so the UI can show the
    edge breakdown.
    """
    from apexflow.scanners.squeeze_radar import SqueezeRadarScanner

    def fetch():
        syms = load_universe(universe)
        scanner = SqueezeRadarScanner(get_provider(), min_score=min_score)
        signals = scanner.scan_safe(syms)
        return {
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "universe": universe,
            "universe_size": len(syms),
            "min_score": float(min_score),
            "signals": [s.to_dict() for s in signals[:limit]],
            "count": len(signals),
        }
    return cached(f"radar:{universe}:{min_score}:{limit}", ttl=180, fetch=fetch)


@app.get("/api/regime")
def api_regime():
    """Market regime snapshot: VIX-based regime + sector breadth."""
    def fetch():
        prov = get_provider()
        # VIX
        vix = 0.0
        try:
            q = prov.quote("^VIX")
            vix = float(q.get("price") or 0)
        except Exception:
            pass
        if vix == 0:
            try:
                df = prov.history("^VIX", period="5d", interval="1d")
                if df is not None and not df.empty:
                    vix = float(df["Close"].iloc[-1])
            except Exception:
                pass
        regime = "neutral"
        if vix and vix < 15:
            regime = "calm"
        elif vix and vix > 25:
            regime = "risk"
        elif vix and vix > 19:
            regime = "neutral"
        else:
            regime = "calm"

        # SPY trend (above/below 50EMA)
        spy_trend = "—"; spy_pct = 0.0
        try:
            df = prov.history("SPY", period="6mo", interval="1d")
            if df is not None and len(df) > 60:
                close = df["Close"]
                ema50 = ind.ema(close, 50).iloc[-1]
                last = float(close.iloc[-1])
                spy_pct = (last / ema50 - 1) * 100 if ema50 else 0.0
                spy_trend = "above 50EMA" if last >= ema50 else "below 50EMA"
        except Exception:
            pass

        # Sector ETFs
        sectors = [
            ("XLK", "Tech"), ("XLF", "Fin"), ("XLE", "Energy"),
            ("XLV", "Health"), ("XLY", "Discr"), ("XLP", "Staples"),
            ("XLI", "Indust"), ("XLU", "Util"), ("XLB", "Mater"), ("XLC", "Comm"),
            ("XLRE", "RE"),
        ]
        heat = []
        for sym, label in sectors:
            try:
                q = cached(f"q:{sym}", 60, lambda s=sym: prov.quote(s))
                price = q.get("price") or 0
                prev = q.get("previous_close") or 0
                pct = ((price - prev) / prev * 100) if prev else 0.0
                heat.append({"symbol": sym, "label": label, "change_pct": float(pct)})
            except Exception:
                heat.append({"symbol": sym, "label": label, "change_pct": 0.0})

        breadth_up = sum(1 for h in heat if h["change_pct"] > 0)
        breadth_pct = (breadth_up / len(heat) * 100) if heat else 50.0

        return {
            "vix": float(vix),
            "regime": regime,
            "spy_trend": spy_trend,
            "spy_vs_ema50_pct": float(spy_pct),
            "sectors": heat,
            "breadth_up": breadth_up,
            "breadth_total": len(heat),
            "breadth_pct": float(breadth_pct),
        }
    return cached("regime", ttl=60, fetch=fetch)


@app.get("/api/size")
def api_size(equity: float, risk_pct: float, entry: float,
             stop: float | None = None, target: float | None = None,
             unit: str = "shares", premium: float | None = None,
             max_position_pct: float = 100.0):
    """Position size from your own numbers. Reads no market data.

    Fixed-fractional: risk a constant fraction of equity and let the stop
    distance determine size. Pass `premium` with unit=contracts to size a
    long option against the premium instead of a stop.

    Deliberately not connected to any signal — it answers "how much", never
    "whether". Refuses rather than guesses on inputs that would produce an
    unsurvivable position.
    """
    from apexflow.platform.risk import (
        size_by_stop, size_options_by_premium, r_multiple)

    frac = max(risk_pct, 0.0) / 100.0
    if premium is not None and premium > 0:
        result = size_options_by_premium(equity, frac, premium)
    elif stop is not None:
        result = size_by_stop(equity, frac, entry, stop, unit=unit,
                              max_position_fraction=max(max_position_pct, 0.0) / 100.0)
    else:
        raise HTTPException(400, "provide either a stop price or a premium")

    out = result.to_dict()
    if target is not None and stop is not None:
        out["reward"] = r_multiple(entry, stop, target)
    return out


@app.get("/api/freshness")
def api_freshness(sym: str = "SPY"):
    """Data age and whether it is current enough to act on.

    Polled by the UI banner. Deliberately cheap and never cached, since a
    cached freshness reading is a contradiction in terms.
    """
    f = freshness_for(sym)
    f["market_state"] = market_state()
    f["provider"] = getattr(get_provider(), "name", "unknown")
    return f


@app.get("/api/health")
def health():
    """One-stop check: provider, configured keys, background engine statuses."""
    out = {
        "ok": True,
        "provider": get_provider().name,
        "primary_provider_priority": config.primary_provider_name(),
        "providers_configured": config.configured_providers(),
        "engines": {},
    }
    if _alerts_engine is not None:
        try:
            out["engines"]["alerts"] = _alerts_engine.status()
        except Exception:
            out["engines"]["alerts"] = {"running": False, "error": "status failed"}
    if _briefing_engine is not None:
        try:
            out["engines"]["briefing"] = _briefing_engine.status()
        except Exception:
            out["engines"]["briefing"] = {"running": False, "error": "status failed"}
    return out


# -----------------------------------------------------------------------------
# Live alerts engine — background scanner that produces real, evolving alerts
#
# yfinance throttles aggressively per IP, so the free-tier defaults are very
# conservative: small slice, long pause, startup delay so the page can load
# before the engine starts. With paid providers configured, the engine
# scales up automatically.
# -----------------------------------------------------------------------------
_alerts_engine: LiveAlertsEngine | None = None


def get_alerts_engine() -> LiveAlertsEngine | None:
    return _alerts_engine


def _alerts_profile() -> dict:
    """Tune universe + cadence to whether paid providers are available."""
    has_paid = any(config.configured_providers().values())
    if has_paid:
        return {"universe": "optionable", "slice_size": 40,
                "sweep_pause": 30.0, "threshold": 65.0,
                "startup_delay": 10.0}
    # Free yfinance — must avoid 429s
    return {"universe": "sp100", "slice_size": 10,
            "sweep_pause": 120.0, "threshold": 70.0,
            "startup_delay": 30.0}


def _start_alerts_engine() -> None:
    global _alerts_engine
    prof = _alerts_profile()
    try:
        universe = load_universe(prof["universe"])
    except Exception:
        universe = load_universe("sp100")
    _alerts_engine = LiveAlertsEngine(
        universe=universe,
        enabled_scanners=list(ALL_SCANNERS.keys()),
        cooldown_seconds=900,
        slice_size=prof["slice_size"],
        sweep_pause=prof["sweep_pause"],
        threshold=prof["threshold"],
        startup_delay=prof["startup_delay"],
        buffer_size=500,
    )
    _alerts_engine.start()


def _stop_alerts_engine() -> None:
    if _alerts_engine:
        _alerts_engine.stop()



# -----------------------------------------------------------------------------
# Briefing engine — pre-computes per-symbol analysis + universe leaderboard
# -----------------------------------------------------------------------------
_briefing_engine: BriefingEngine | None = None


def get_briefing_engine() -> BriefingEngine | None:
    return _briefing_engine


def _start_briefing_engine() -> None:
    import os
    if os.environ.get("APEXFLOW_BRIEFING_DISABLE"):
        return
    global _briefing_engine
    has_paid = any(config.configured_providers().values())
    profile = BriefingProfile(
        watchlist_seconds=600.0 if has_paid else 900.0,
        universe_seconds=1800.0 if has_paid else 3600.0,
        inter_symbol_pause=0.3 if has_paid else 0.7,
        startup_delay=20.0 if has_paid else 60.0,
    )
    _briefing_engine = BriefingEngine(
        universe=briefing_default_universe(),
        profile=profile,
    )
    _briefing_engine.start()


def _stop_briefing_engine() -> None:
    if _briefing_engine:
        _briefing_engine.stop()


@app.get("/api/brief/status")
def api_brief_status():
    if _briefing_engine is None:
        return {"running": False}
    return _briefing_engine.status()


@app.get("/api/brief")
def api_brief_list(scope: str = "watchlist", limit: int = 100):
    """List pre-computed briefs.

    scope='watchlist' returns the per-symbol briefs (sorted by radar score).
    scope='universe'  returns the universe-wide radar leaderboard.
    """
    if scope == "universe":
        board = read_universe_leaderboard() or {}
        signals = board.get("signals", [])[:limit]
        return {
            "scope": "universe",
            "generated_at": board.get("generated_at"),
            "universe_size": board.get("universe_size", 0),
            "signals": signals,
        }
    briefs = all_briefs()
    # Sort by radar score desc; missing radar = 0
    briefs.sort(key=lambda b: (b.get("radar") or {}).get("score", 0), reverse=True)
    return {
        "scope": "watchlist",
        "count": len(briefs),
        "briefs": briefs[:limit],
    }



@app.get("/api/brief/{sym}")
def api_brief_one(sym: str, refresh: bool = False):
    """Return the cached brief for `sym`. ?refresh=1 rebuilds it on demand."""
    sym = sym.upper()
    if refresh and _briefing_engine is not None:
        brief = _briefing_engine.refresh_one(sym)
        if brief:
            return brief
    cached_b = read_brief(sym)
    if cached_b:
        return cached_b
    brief = build_brief(sym)
    if brief:
        write_brief(brief)
        return brief
    raise HTTPException(404, f"could not build brief for {sym}")


# -----------------------------------------------------------------------------
# Earnings Market Scanner — full-market earnings + direction + radar table
# -----------------------------------------------------------------------------
@app.get("/api/earnings_scan")
def api_earnings_scan(window: str = "week", max_symbols: int = 80):
    """Scan all companies with earnings in the requested window.

    window:
        'tomorrow' → next trading day only
        'week'     → next 5 trading days (default)
        '2week'    → next 10 trading days
    """
    from apexflow.scanners.earnings_market import scan_earnings
    from datetime import date as _date, timedelta as _td

    today = _date.today()
    if window == "tomorrow":
        start = today + _td(days=1); end = start
    elif window == "2week":
        start = today; end = today + _td(days=14)
    else:
        start = today; end = today + _td(days=7)

    def fetch():
        return scan_earnings(start, end, max_symbols=max_symbols)

    return cached(f"earnings_scan:{window}:{max_symbols}", ttl=900, fetch=fetch)
