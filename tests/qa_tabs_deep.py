"""ApexFlow per-tab deep validation.

Goes through every UI tab and validates that:
  - The page renders (status 200)
  - Every API endpoint the tab calls returns sane LIVE data (not just 200)
  - Cross-endpoint values are consistent (e.g. spot matches across endpoints)
  - Numbers are in plausible ranges for the symbol

Run after qa_full_audit.py passes.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["APEXFLOW_ATLAS_DISABLE"] = "1"
os.environ["APEXFLOW_BRIEFING_DISABLE"] = "1"

from fastapi.testclient import TestClient
import webapp

client = TestClient(webapp.app)
results: list[tuple[str, str, str]] = []

def _add(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    tag = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[ -- ]"}[status]
    print(f"{tag}  {name}" + (f"  ::  {detail}" if detail else ""), flush=True)

def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}", flush=True)

def get(path: str, **kw):
    r = client.get(path, timeout=180, **kw)
    return r

def post(path: str):
    return client.post(path, timeout=300)


# ============================================================================
# Tab 1: Dashboard  (/)  - calls /api/ribbon, /api/regime, /api/scan
# ============================================================================
def tab_dashboard():
    section("TAB 1 - Dashboard  (/)")
    r = get("/")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")

    # Ribbon: 5 tickers (SPY, QQQ, IWM, DIA, VIX)
    j = get("/api/ribbon").json()
    syms = {x["symbol"] for x in j}
    expected = {"SPY", "QQQ", "IWM", "DIA", "VIX"}
    _add("PASS" if expected.issubset(syms) else "FAIL", "ribbon: all 5 indices",
         f"got={sorted(syms)}")
    nonzero = [x for x in j if (x.get("price") or 0) > 0]
    _add("PASS" if len(nonzero) >= 4 else "WARN",
         "ribbon: at least 4 of 5 have live prices",
         f"with_price={len(nonzero)}/5")
    for x in j:
        if (x.get("price") or 0) > 0:
            # Sanity: change_pct typically -10..10 in normal markets
            cp = x.get("change_pct", 0)
            if abs(cp) > 25:
                _add("WARN", f"ribbon {x['symbol']} change_pct={cp:.2f}%",
                     "outside +/-25% range -- review")

    # Regime
    j = get("/api/regime").json()
    _add("PASS" if j.get("regime") in ("calm", "neutral", "risk") else "FAIL",
         "regime: enum value", f"regime={j.get('regime')}")
    _add("PASS" if 5 < (j.get("vix") or 0) < 100 else "WARN",
         "regime: VIX in plausible range (5..100)",
         f"vix={j.get('vix')}")
    sectors = j.get("sectors", [])
    _add("PASS" if len(sectors) >= 10 else "FAIL",
         "regime: sector breadth has 10+ entries",
         f"count={len(sectors)}")
    _add("PASS" if 0 <= (j.get("breadth_pct") or 0) <= 100 else "FAIL",
         "regime: breadth % in [0,100]",
         f"breadth={j.get('breadth_pct')}")

    # Scan (small universe so it's fast)
    j = get("/api/scan?universe=data/test_universe.txt&scanners=options-flow,momentum").json()
    _add("PASS" if "scanners" in j and "top_signals" in j else "FAIL",
         "scan: response shape", f"keys={list(j.keys())[:6]}")
    _add("PASS" if j.get("universe_size", 0) > 0 else "FAIL",
         "scan: universe loaded", f"size={j.get('universe_size')}")


# ============================================================================
# Tab 2: Heatmap  (/heatmap)  - single-symbol heatseeker
# ============================================================================
def tab_heatmap():
    section("TAB 2 - Heatmap  (/heatmap)")
    r = get("/heatmap")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/heatseeker?symbols=AAPL&dtes=0,7,30&n_strikes=20").json()
    syms = j.get("symbols", [])
    _add("PASS" if syms else "FAIL", "heatseeker returns symbols", f"n={len(syms)}")
    if syms:
        s = syms[0]
        _add("PASS" if s.get("spot", 0) > 0 else "FAIL",
             "heatmap AAPL: spot live", f"spot={s.get('spot')}")
        layers = s.get("layers", [])
        _add("PASS" if len(layers) == 3 else "FAIL",
             "heatmap: 3 DTE layers returned",
             f"got={len(layers)} labels={[L.get('label') for L in layers]}")
        # At least one layer should have strikes populated
        any_strikes = any(len(L.get("strikes", [])) > 0 for L in layers)
        _add("PASS" if any_strikes else "WARN",
             "heatmap: at least one layer has strikes",
             f"per_layer_strikes={[len(L.get('strikes', [])) for L in layers]}")


# ============================================================================
# Tab 3: Heatseeker  (/heatseeker)  - multi-symbol
# ============================================================================
def tab_heatseeker():
    section("TAB 3 - Heatseeker  (/heatseeker)")
    r = get("/heatseeker")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/heatseeker?symbols=SPY,QQQ&dtes=0,1,7&n_strikes=15").json()
    syms = j.get("symbols", [])
    _add("PASS" if len(syms) == 2 else "FAIL",
         "multi: 2 symbols returned", f"n={len(syms)}")
    if syms:
        for s in syms:
            grid = s.get("strike_grid", [])
            _add("PASS" if len(grid) > 0 else "WARN",
                 f"multi {s.get('symbol')}: strike_grid populated",
                 f"size={len(grid)}")
            # All layer 'kings' should have node_type='king'
            for L in s.get("layers", []):
                kings = [st for st in L.get("strikes", []) if st.get("node_type") == "king"]
                if kings:
                    _add("PASS", f"{s['symbol']} {L.get('label')}: king node tagged",
                         f"king_strike={kings[0]['strike']}")


# ============================================================================
# Tab 4: Atlas  (/atlas)  - intraday GEX history
# ============================================================================
def tab_atlas():
    section("TAB 4 - Atlas  (/atlas)")
    r = get("/atlas")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/atlas/AAPL?hours=6.5&live_snapshot=false").json()
    _add("PASS" if "frames" in j and "intraday_candles" in j else "FAIL",
         "atlas response shape", f"keys={list(j.keys())[:8]}")
    candles = j.get("intraday_candles", [])
    _add("PASS" if len(candles) > 0 else "WARN",
         "atlas: intraday candles available", f"n={len(candles)}")
    j2 = get("/api/atlas/status").json()
    _add("PASS" if "running" in j2 else "FAIL",
         "atlas/status field", f"got={list(j2.keys())[:6]}")


# ============================================================================
# Tab 5: Radar  (/radar)
# ============================================================================
def tab_radar():
    section("TAB 5 - Radar  (/radar)")
    r = get("/radar")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/radar?universe=data/test_universe.txt&min_score=0&limit=10").json()
    _add("PASS" if "signals" in j and "count" in j else "FAIL",
         "radar response shape", f"keys={list(j.keys())[:6]}")
    _add("PASS" if j.get("universe_size", 0) > 0 else "FAIL",
         "radar: universe loaded", f"size={j.get('universe_size')}")
    # Each signal must have layer_scores
    for s in j.get("signals", [])[:3]:
        m = s.get("metrics", {})
        ls = m.get("layer_scores")
        _add("PASS" if isinstance(ls, dict) and len(ls) >= 5 else "WARN",
             f"radar signal {s.get('symbol')}: layer_scores present",
             f"keys={list(ls.keys()) if ls else 'missing'}")


# ============================================================================
# Tab 6: Brief  (/brief)
# ============================================================================
def tab_brief():
    section("TAB 6 - Brief  (/brief)")
    r = get("/brief")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/brief?scope=watchlist&limit=20").json()
    _add("PASS" if "scope" in j else "FAIL",
         "brief watchlist scope", f"keys={list(j.keys())[:5]}")
    j2 = get("/api/brief?scope=universe&limit=20").json()
    _add("PASS" if j2.get("scope") == "universe" else "FAIL",
         "brief universe scope", f"keys={list(j2.keys())[:5]}")
    j3 = get("/api/brief/status").json()
    _add("PASS" if "running" in j3 else "FAIL",
         "brief engine status", f"got={j3}")


# ============================================================================
# Tab 7: Watchlist  (/watchlist)
# ============================================================================
def tab_watchlist():
    section("TAB 7 - Watchlist  (/watchlist)")
    r = get("/watchlist")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/watchlist").json()
    _add("PASS" if isinstance(j, list) else "FAIL",
         "watchlist returns list", f"type={type(j).__name__}")


# ============================================================================
# Tab 8: Log  (/log)
# ============================================================================
def tab_log():
    section("TAB 8 - Log  (/log)")
    r = get("/log")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    j = get("/api/log?limit=10").json()
    _add("PASS" if isinstance(j, list) else "FAIL",
         "log returns list", f"type={type(j).__name__}")


# ============================================================================
# Tab 9: Symbol  (/symbol/AAPL)  - the single-ticker view
# ============================================================================
def tab_symbol():
    section("TAB 9 - Symbol  (/symbol/AAPL)")
    r = get("/symbol/AAPL")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")

    # Quote
    q = get("/api/quote/AAPL").json()
    spot = q.get("price") or 0
    _add("PASS" if spot > 0 else "FAIL",
         "symbol AAPL: live price", f"price={spot}")

    # Indicators
    i = get("/api/indicators/AAPL").json()
    needed = {"rsi", "ema20", "ema50", "atr14", "rvol", "atm_iv", "iv_rank"}
    _add("PASS" if needed.issubset(i.keys()) else "FAIL",
         "indicators: all fields present",
         f"missing={needed - set(i.keys())}")
    _add("PASS" if 0 <= (i.get("rsi") or 0) <= 100 else "FAIL",
         "indicators: RSI in [0,100]", f"rsi={i.get('rsi')}")
    _add("PASS" if 0 <= (i.get("iv_rank") or 0) <= 100 else "FAIL",
         "indicators: IV rank in [0,100]", f"iv_rank={i.get('iv_rank')}")

    # History (price chart)
    h = get("/api/history/AAPL?period=6mo&interval=1d").json()
    _add("PASS" if isinstance(h, list) and len(h) > 100 else "FAIL",
         "history: 6mo daily has 100+ bars", f"n={len(h) if isinstance(h, list) else 'n/a'}")
    if isinstance(h, list) and h:
        last = h[-1]
        _add("PASS" if all(k in last for k in ("t","o","h","l","c","v")) else "FAIL",
             "history: OHLCV row schema", f"keys={list(last.keys())}")

    # Expiries
    e = get("/api/expiries/AAPL").json()
    _add("PASS" if isinstance(e, list) and len(e) >= 5 else "FAIL",
         "expiries: at least 5 expiration dates",
         f"n={len(e) if isinstance(e, list) else 'n/a'}")

    # Chain (Greeks attached)
    if isinstance(e, list) and e:
        c = get(f"/api/chain/AAPL?expiry={e[0]}").json()
        calls = c.get("calls", [])
        _add("PASS" if len(calls) > 0 else "FAIL",
             "chain: calls populated", f"n={len(calls)}")
        if calls:
            row = calls[0]
            _add("PASS" if all(k in row for k in ("strike","bid","ask","delta","gamma","vega","theta")) else "FAIL",
                 "chain: row has Greeks attached",
                 f"keys={list(row.keys())}")
            # Chain spot matches quote spot within 1%
            cs = c.get("spot", 0)
            if cs and spot:
                drift = abs(cs - spot) / spot
                _add("PASS" if drift < 0.02 else "WARN",
                     "chain.spot ~ quote.price (cross-endpoint sanity)",
                     f"chain={cs} quote={spot} drift={drift*100:.2f}%")

    # Projection
    p = get("/api/projection/AAPL").json()
    _add("PASS" if "expected_move" in p and "cone" in p else "FAIL",
         "projection: expected_move + cone",
         f"keys={list(p.keys())[:8]}")
    if (p.get("expected_move") or 0) > 0 and spot:
        em_pct = p.get("expected_move_pct") or 0
        _add("PASS" if 0.1 < em_pct < 30 else "WARN",
             "projection: 1-sigma move plausible (0.1..30%)",
             f"em_pct={em_pct:.2f}%")

    # Key levels
    kl = get("/api/keylevels/AAPL").json()
    _add("PASS" if "spot" in kl else "FAIL",
         "keylevels: response", f"keys={list(kl.keys())[:8]}")


# ============================================================================
# Tab 10: Backtest  (/backtest)
# ============================================================================
def tab_backtest():
    section("TAB 10 - Backtest  (/backtest)")
    r = get("/backtest")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code}")
    r = post("/api/backtest?scanner=momentum&universe=data/test_universe.txt&lookback=60&hold=5")
    _add("PASS" if r.status_code == 200 else "FAIL",
         "backtest POST momentum", f"status={r.status_code}")
    if r.status_code == 200:
        j = r.json()
        for k in ("scanner", "signals_tested", "win_rate", "sharpe", "equity_curve"):
            _add("PASS" if k in j else "FAIL",
                 f"backtest result has '{k}'", f"got={k in j}")


# ============================================================================
# Tab 11: Guide  (/guide)
# ============================================================================
def tab_guide():
    section("TAB 11 - Guide  (/guide)")
    r = get("/guide")
    _add("PASS" if r.status_code == 200 else "FAIL", "page renders",
         f"status={r.status_code} ct={r.headers.get('content-type','')}")
    # Just confirm it has substantial content
    _add("PASS" if len(r.text) > 1000 else "WARN",
         "guide: page has content (>1KB)", f"size={len(r.text)}B")


# ============================================================================
# Cross-tab consistency
# ============================================================================
def cross_consistency():
    section("X - Cross-endpoint consistency")
    # Quote spot vs. GEX spot vs. projection spot for AAPL
    q = (get("/api/quote/AAPL").json().get("price") or 0)
    g = (get("/api/gex/AAPL").json().get("spot") or 0)
    p = (get("/api/projection/AAPL").json().get("spot") or 0)
    if q and g:
        _add("PASS" if abs(g - q) / q < 0.02 else "WARN",
             "consistency: quote.price ~ gex.spot",
             f"q={q:.2f} g={g:.2f}")
    if q and p:
        _add("PASS" if abs(p - q) / q < 0.02 else "WARN",
             "consistency: quote.price ~ projection.spot",
             f"q={q:.2f} p={p:.2f}")


def main():
    print("ApexFlow per-tab deep validation\n")
    tab_dashboard()
    tab_heatmap()
    tab_heatseeker()
    tab_atlas()
    tab_radar()
    tab_brief()
    tab_watchlist()
    tab_log()
    tab_symbol()
    tab_backtest()
    tab_guide()
    cross_consistency()

    section("REPORT SUMMARY")
    npass = sum(1 for s, _, _ in results if s == "PASS")
    nwarn = sum(1 for s, _, _ in results if s == "WARN")
    nfail = sum(1 for s, _, _ in results if s == "FAIL")
    print(f"\n  PASS: {npass}\n  WARN: {nwarn}\n  FAIL: {nfail}\n")
    if nfail or nwarn:
        print("Issues:")
        for s, n, d in results:
            if s in ("FAIL", "WARN"):
                print(f"  [{s}] {n}  ::  {d}")
    return nfail


if __name__ == "__main__":
    sys.exit(main())
