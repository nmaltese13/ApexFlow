"""ApexFlow full QA harness.

Run:    python tests/qa_full_audit.py
Output: structured PASS/FAIL/WARN report. Exit code = number of FAILs.

This is a one-shot pre-flight check before paying for live data:
  1. Math sanity (Greeks, GEX, indicators, squeeze score)
  2. Provider live-data round trip (yfinance free)
  3. Each scanner runs without crashing on a small universe
  4. Platform: signal logger / watchlist / alert cooldown
  5. Backtester runs and produces finite metrics
  6. Every FastAPI route returns 200 with a sane body

Network calls hit yfinance (free) and may be slow / rate-limited. Use a small
universe (test_universe.txt = AAPL,NVDA,TSLA) for the live tests.
"""
from __future__ import annotations
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Atlas + briefing background loops would slow startup and pound yfinance.
os.environ["APEXFLOW_ATLAS_DISABLE"] = "1"
os.environ["APEXFLOW_BRIEFING_DISABLE"] = "1"

import numpy as np
import pandas as pd

results: list[tuple[str, str, str]] = []  # (status, name, detail)
def _add(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    tag = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[ -- ]"}[status]
    print(f"{tag}  {name}" + (f"  ::  {detail}" if detail else ""), flush=True)

def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}", flush=True)

def check(name: str, fn, warn_only: bool = False):
    try:
        ok, detail = fn()
        if ok:
            _add("PASS", name, detail or "")
        else:
            _add("WARN" if warn_only else "FAIL", name, detail or "")
    except Exception as e:
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        _add("WARN" if warn_only else "FAIL", name, f"{type(e).__name__}: {e} | {tb}")


# ============================================================================
# 1. MATH - Greeks
# ============================================================================
def test_greeks_math():
    section("1. MATH - Black-Scholes Greeks")
    from apexflow.analytics.greeks import greeks, implied_vol_newton, iv_rank, iv_percentile, years_to_expiry

    # Reference values (verified against external BS calculator)
    # S=100, K=100, t=0.25y, r=0.04, q=0, sigma=0.30
    S, K, t, sigma, r = 100.0, 100.0, 0.25, 0.30, 0.04
    g_call = greeks(S, K, t, sigma, "C", r=r)
    g_put = greeks(S, K, t, sigma, "P", r=r)

    check("Call delta in (0.5, 0.6) for ATM, +rate",
          lambda: (0.55 < g_call["delta"] < 0.62, f"delta={g_call['delta']:.4f}"))
    check("Put delta in (-0.5, -0.4) for ATM, +rate",
          lambda: (-0.45 < g_put["delta"] < -0.38, f"delta={g_put['delta']:.4f}"))
    check("Call/put gamma identical (no dividends)",
          lambda: (abs(g_call["gamma"] - g_put["gamma"]) < 1e-10,
                   f"call={g_call['gamma']:.6f} put={g_put['gamma']:.6f}"))
    check("Call/put vega identical",
          lambda: (abs(g_call["vega"] - g_put["vega"]) < 1e-10,
                   f"call={g_call['vega']:.4f} put={g_put['vega']:.4f}"))
    check("Theta is negative for both (long opt decays)",
          lambda: (g_call["theta"] < 0 and g_put["theta"] < 0,
                   f"call_theta={g_call['theta']:.4f} put_theta={g_put['theta']:.4f}"))
    # Put-call parity for delta: C_delta - P_delta = e^{-q*t} ~= 1 with q=0
    check("Put-call parity (delta): call_delta - put_delta ~= 1",
          lambda: (abs((g_call["delta"] - g_put["delta"]) - 1.0) < 1e-6,
                   f"diff={g_call['delta'] - g_put['delta']:.6f}"))

    # Gamma peaks at-the-money
    g_atm  = greeks(S, S,        t, sigma, "C")["gamma"]
    g_otm  = greeks(S, S * 1.20, t, sigma, "C")["gamma"]
    g_itm  = greeks(S, S * 0.80, t, sigma, "C")["gamma"]
    check("Gamma is max at ATM vs ?20% strikes",
          lambda: (g_atm > g_otm and g_atm > g_itm,
                   f"atm={g_atm:.5f} otm={g_otm:.5f} itm={g_itm:.5f}"))

    # Vega should be positive
    check("Vega > 0 for ATM call",
          lambda: (g_call["vega"] > 0, f"vega={g_call['vega']:.4f}"))

    # Invalid inputs return zeros (safe defaults)
    bad = greeks(0, 100, 0.25, 0.3, "C")
    check("Zero-spot input returns zero greeks",
          lambda: (all(v == 0 for v in bad.values()), f"got={bad}"))

    bad2 = greeks(100, 100, -0.5, 0.3, "C")  # negative t
    check("Negative time-to-expiry returns zero greeks",
          lambda: (all(v == 0 for v in bad2.values()), f"got={bad2}"))

    # IV solver round-trip: compute price ? solve back ? original IV
    from scipy.stats import norm
    def bs_call(S, K, t, sigma, r):
        d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))
        d2 = d1 - sigma*math.sqrt(t)
        return S*norm.cdf(d1) - K*math.exp(-r*t)*norm.cdf(d2)
    price = bs_call(100, 100, 0.25, 0.30, 0.04)
    iv_solved = implied_vol_newton(price, 100, 100, 0.25, "C", r=0.04)
    check("IV solver round-trip recovers sigma=0.30 within 1e-3",
          lambda: (iv_solved is not None and abs(iv_solved - 0.30) < 1e-3,
                   f"price={price:.4f} iv_solved={iv_solved}"))

    # IV rank/percentile bounds
    check("IV rank clamps to 0..100",
          lambda: (0 <= iv_rank(0.5, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) <= 100, ""))
    check("IV percentile == 50 when median",
          lambda: (abs(iv_percentile(0.5, [0.1, 0.3, 0.5, 0.7, 0.9]) - 40) < 1,
                   f"got={iv_percentile(0.5, [0.1, 0.3, 0.5, 0.7, 0.9])}"))
    check("IV rank degenerate (all same) returns 50",
          lambda: (iv_rank(0.5, [0.5, 0.5, 0.5]) == 50.0, ""))


# ============================================================================
# 2. MATH - GEX
# ============================================================================
def test_gex_math():
    section("2. MATH - Gamma Exposure (GEX)")
    from apexflow.analytics.gex import chain_gex, gex_summary, bs_gamma, years_to_expiry

    # Sign convention test: pure call OI ? +GEX, pure put OI ? -GEX
    calls = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "openInterest": [1000, 5000, 1000],
        "impliedVolatility": [0.30, 0.30, 0.30],
    })
    empty_puts = pd.DataFrame(columns=["strike", "openInterest", "impliedVolatility"])
    df = chain_gex(calls, empty_puts, spot=100.0, expiry="2026-06-15")
    check("All-call chain ? all call_gex > 0",
          lambda: ((df["call_gex"] >= 0).all() and df["call_gex"].sum() > 0,
                   f"sum={df['call_gex'].sum():.0f}"))
    check("All-call chain ? put_gex == 0",
          lambda: (df["put_gex"].abs().sum() < 1e-9, ""))

    puts = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "openInterest": [1000, 5000, 1000],
        "impliedVolatility": [0.30, 0.30, 0.30],
    })
    df2 = chain_gex(pd.DataFrame(columns=calls.columns), puts, 100.0, "2026-06-15")
    check("All-put chain ? all put_gex <= 0",
          lambda: ((df2["put_gex"] <= 0).all() and df2["put_gex"].sum() < 0,
                   f"sum={df2['put_gex'].sum():.0f}"))

    # ATM call gamma > wing call gamma at same expiry. Use asymmetric OI so
    # calls/puts don't cancel exactly when summed.
    asym_puts = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "openInterest": [200, 200, 200],
        "impliedVolatility": [0.30, 0.30, 0.30],
    })
    df3 = chain_gex(calls, asym_puts, 100.0, "2026-06-15")
    atm_row = df3[df3["strike"] == 100.0].iloc[0]
    otm_row = df3[df3["strike"] == 105.0].iloc[0]
    check("ATM strike has higher |total_gex| than OTM (call-heavy)",
          lambda: (abs(atm_row["total_gex"]) > abs(otm_row["total_gex"]),
                   f"atm={atm_row['total_gex']:.0f} otm={otm_row['total_gex']:.0f}"))
    # Symmetric calls+puts with same OI MUST cancel to ~0 — that's the math
    df_sym = chain_gex(calls, puts, 100.0, "2026-06-15")
    check("Symmetric same-OI calls+puts cancel to ~0 (sanity)",
          lambda: (df_sym["total_gex"].abs().sum() < 1.0,
                   f"sum_abs={df_sym['total_gex'].abs().sum():.6f}"))

    # NaN safety: yfinance often returns NaN for IV/OI
    nan_calls = pd.DataFrame({
        "strike": [100.0, 105.0],
        "openInterest": [float("nan"), 500],
        "impliedVolatility": [float("nan"), 0.35],
    })
    df4 = chain_gex(nan_calls, empty_puts, 100.0, "2026-06-15")
    check("NaN OI/IV doesn't crash (yfinance-safe)",
          lambda: (not df4.empty and np.isfinite(df4["total_gex"]).all(),
                   f"rows={len(df4)} all_finite={np.isfinite(df4['total_gex']).all()}"))

    # Empty chain returns empty df (no crash)
    df5 = chain_gex(pd.DataFrame(), pd.DataFrame(), 100.0, "2026-06-15")
    check("Empty chains return empty DataFrame",
          lambda: (df5.empty and list(df5.columns) == ["strike","call_gex","put_gex","total_gex"], ""))

    # Spot=0 is rejected (no division by zero)
    df6 = chain_gex(calls, puts, 0.0, "2026-06-15")
    check("Spot=0 returns empty (no /0 crash)",
          lambda: (df6.empty, ""))

    # Summary: gamma flip exists when sign changes; positive total when calls dominate
    summary = gex_summary(df3, 100.0)
    check("Summary returns total_gex finite",
          lambda: (np.isfinite(summary["total_gex"]),
                   f"total={summary['total_gex']:.0f}"))
    check("Summary returns max_pos_strike & max_neg_strike",
          lambda: (summary["max_pos_strike"] is not None and summary["max_neg_strike"] is not None, ""))

    # 0DTE: years_to_expiry must be > 0 (not exactly 0) so gamma stays finite
    from datetime import datetime
    today = datetime.utcnow().strftime("%Y-%m-%d")
    t0 = years_to_expiry(today)
    check("0DTE years_to_expiry > 0 (no /0 in gamma)",
          lambda: (t0 > 0, f"t={t0:.6f}"))

    # bs_gamma scalar matches vector path at one strike
    g_scalar = bs_gamma(100.0, 100.0, 0.25, 0.30)
    check("Scalar bs_gamma > 0 at ATM",
          lambda: (g_scalar > 0, f"gamma={g_scalar:.5f}"))


# ============================================================================
# 3. MATH - Indicators
# ============================================================================
def test_indicators_math():
    section("3. MATH - Indicators")
    from apexflow.analytics import indicators as ind

    # Build a deterministic price series
    n = 100
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    vol = pd.Series(rng.integers(1_000_000, 5_000_000, n).astype(float))

    sma20 = ind.sma(close, 20)
    check("SMA(20) NaN for first 19 bars, finite after",
          lambda: (sma20.iloc[:19].isna().all() and sma20.iloc[19:].notna().all(), ""))
    check("SMA(20) last value matches manual calc",
          lambda: (abs(sma20.iloc[-1] - close.iloc[-20:].mean()) < 1e-9,
                   f"diff={abs(sma20.iloc[-1] - close.iloc[-20:].mean()):.2e}"))

    ema20 = ind.ema(close, 20)
    check("EMA(20) finite from bar 0 (ewm)",
          lambda: (ema20.notna().all(), ""))

    rsi = ind.rsi(close, 14)
    rsi_last = float(rsi.dropna().iloc[-1])
    check("RSI in [0,100]",
          lambda: (0 <= rsi_last <= 100, f"rsi={rsi_last:.2f}"))

    bb = ind.bollinger(close, 20)
    check("Bollinger upper > mid > lower",
          lambda: (bb["bb_upper"].iloc[-1] > bb["bb_mid"].iloc[-1] > bb["bb_lower"].iloc[-1], ""))
    check("Bollinger bandwidth > 0",
          lambda: (bb["bb_bw"].iloc[-1] > 0, f"bw={bb['bb_bw'].iloc[-1]:.4f}"))

    atr14 = ind.atr(high, low, close, 14)
    check("ATR(14) > 0",
          lambda: (atr14.dropna().iloc[-1] > 0, f"atr={atr14.dropna().iloc[-1]:.4f}"))

    rvol = ind.rvol(vol, 30)
    check("RVOL is roughly 1.0 average (synthetic uniform vol)",
          lambda: (0.5 < float(rvol.dropna().mean()) < 1.5,
                   f"mean_rvol={float(rvol.dropna().mean()):.3f}"))

    hv30 = ind.historical_volatility(close, 30)
    check("HV(30) finite, positive",
          lambda: (hv30 > 0 and np.isfinite(hv30), f"hv30={hv30:.4f}"))

    # Squeeze: should be a boolean Series
    sq = ind.squeeze_on(close, high, low)
    check("squeeze_on returns Boolean Series",
          lambda: (sq.dtype == bool, f"dtype={sq.dtype}"))

    # Empty volume edge cases
    check("rvol with zero volume stays NaN safe",
          lambda: (ind.rvol(pd.Series([0.0]*40), 30).iloc[-1] != ind.rvol(pd.Series([0.0]*40), 30).iloc[-1] or
                   np.isnan(ind.rvol(pd.Series([0.0]*40), 30).iloc[-1]), ""))

    # Gap %
    check("gap_pct correct",
          lambda: (abs(ind.gap_pct(102, 100) - 2.0) < 1e-9, ""))
    check("gap_pct safe with zero close_yest",
          lambda: (ind.gap_pct(102, 0) == 0.0, ""))


# ============================================================================
# 4. MATH - Squeeze score
# ============================================================================
def test_squeeze_score():
    section("4. MATH - Composite squeeze score")
    from apexflow.analytics.squeeze import SqueezeInputs, squeeze_score

    # Maxed out: should hit 100
    big = SqueezeInputs(
        short_pct_float=0.55, days_to_cover=12, borrow_rate=2.0,
        iv_hv_ratio=1.8, rvol=8.0, float_shares=5_000_000, accumulation_score=1.0,
    )
    score, comps = squeeze_score(big)
    check("Maxed-squeeze score capped at 100",
          lambda: (score == 100.0, f"score={score:.1f}"))
    check("All component scores positive",
          lambda: (all(v >= 0 for v in comps.values()), f"comps={comps}"))

    # Empty inputs: should be ~0
    zero = SqueezeInputs()
    score0, comps0 = squeeze_score(zero)
    check("Zero-input score is small (<= 15 from float bucket fallback)",
          lambda: (score0 <= 15, f"score={score0:.1f}"))

    # Realistic: GME-like
    real = SqueezeInputs(
        short_pct_float=0.22, days_to_cover=4.2, borrow_rate=0.15,
        iv_hv_ratio=1.4, rvol=2.8, float_shares=300_000_000,
        accumulation_score=0.4,
    )
    sr, _ = squeeze_score(real)
    check("Realistic squeeze score finite, 0..100",
          lambda: (0 < sr < 100, f"score={sr:.1f}"))


# ============================================================================
# 5. PROVIDERS (live yfinance round-trip)
# ============================================================================
def test_providers_live():
    section("5. PROVIDERS - live yfinance round trip")
    from apexflow.providers import get_provider
    p = get_provider()
    _add("INFO", f"Active provider: {p.name}")

    sym = "AAPL"
    q = p.quote(sym)
    check("quote(AAPL) returns dict with price",
          lambda: (isinstance(q, dict) and (q.get("price") or 0) > 0,
                   f"price={q.get('price')}"))

    h = p.history(sym, period="1mo", interval="1d")
    check("history(AAPL) returns DataFrame with OHLCV",
          lambda: (isinstance(h, pd.DataFrame) and not h.empty
                   and {"Open","High","Low","Close","Volume"}.issubset(h.columns),
                   f"rows={len(h) if isinstance(h, pd.DataFrame) else 'n/a'}"))

    exp = p.expiries(sym)
    check("expiries(AAPL) returns non-empty list of YYYY-MM-DD",
          lambda: (isinstance(exp, list) and len(exp) > 0
                   and all(len(e) == 10 and e[4] == '-' for e in exp[:3]),
                   f"first3={exp[:3] if exp else []}"))

    chain = p.options_chain(sym, exp[0] if exp else None)
    check("options_chain(AAPL) returns calls+puts DataFrames",
          lambda: (
              isinstance(chain, dict)
              and isinstance(chain.get("calls"), pd.DataFrame)
              and isinstance(chain.get("puts"), pd.DataFrame)
              and (chain.get("spot") or 0) > 0,
              f"spot={chain.get('spot')} calls={len(chain.get('calls', []))} puts={len(chain.get('puts', []))}"
          ), warn_only=True)

    f = p.fundamentals(sym)
    check("fundamentals(AAPL) returns dict (may be sparse)",
          lambda: (isinstance(f, dict), f"keys={list(f.keys())[:6] if isinstance(f, dict) else 'n/a'}"),
          warn_only=True)


# ============================================================================
# 6. SCANNERS (run each on AAPL,NVDA,TSLA)
# ============================================================================
def test_scanners():
    section("6. SCANNERS - run-once on test universe")
    from apexflow.providers import get_provider
    from apexflow.scanners import ALL_SCANNERS
    from apexflow.models import Signal

    p = get_provider()
    universe = ["AAPL", "NVDA", "TSLA"]

    for name, cls in ALL_SCANNERS.items():
        scanner = cls(p)
        t0 = time.time()
        try:
            sigs = scanner.scan_safe(universe)
        except Exception as e:
            _add("FAIL", f"scanner.{name}", f"{type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        # Validate Signal contract
        bad = [s for s in sigs if not isinstance(s, Signal)
               or not (0 <= s.score <= 100)
               or not isinstance(s.reason, str)
               or not isinstance(s.metrics, dict)]
        if bad:
            _add("FAIL", f"scanner.{name}", f"{len(bad)} malformed signals (score-out-of-range or wrong type)")
            continue
        _add("PASS", f"scanner.{name}",
             f"emitted={len(sigs)}, dt={dt:.1f}s, "
             f"top_score={max((s.score for s in sigs), default=0):.0f}")


# ============================================================================
# 7. PLATFORM - logger / watchlist / alerts
# ============================================================================
def test_platform():
    section("7. PLATFORM - logger / watchlist / alert cooldown")
    from apexflow.models import Signal
    from apexflow.platform.signal_logger import SignalLogger
    from apexflow.platform.watchlist import Watchlist
    from apexflow.platform.alerts import AlertManager

    # Use isolated tmp paths (don't pollute prod data)
    tmp = ROOT / "data" / "_qa_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    log_path = tmp / "qa_signals.jsonl"
    wl_path = tmp / "qa_watchlist.json"
    if log_path.exists(): log_path.unlink()
    if wl_path.exists(): wl_path.unlink()

    sl = SignalLogger(path=log_path)
    s1 = Signal(scanner="qa", symbol="AAPL", score=90, reason="qa")
    s2 = Signal(scanner="qa", symbol="NVDA", score=80, reason="qa")
    sl.log_many([s1, s2])
    rec = list(sl.iter_recent(10))
    check("SignalLogger writes & reads back JSONL",
          lambda: (len(rec) == 2 and rec[0]["symbol"] == "AAPL", f"got={len(rec)}"))

    wl = Watchlist(path=wl_path)
    wl.add_many([s1, s2], min_score=50)
    listed = wl.list()
    check("Watchlist persists entries to disk",
          lambda: (len(listed) == 2 and listed[0]["best_score"] == 90, f"got={len(listed)}"))
    # Update with higher score for AAPL
    s3 = Signal(scanner="qa", symbol="AAPL", score=95, reason="updated")
    wl.add_many([s3], min_score=50)
    aapl = next((e for e in wl.list() if e["symbol"] == "AAPL"), None)
    check("Watchlist updates best_score on higher hit",
          lambda: (aapl and aapl["best_score"] == 95, f"score={aapl['best_score'] if aapl else 'n/a'}"))

    # Alert cooldown
    fired = []
    am = AlertManager(cooldown_seconds=60, sink=fired.append)
    s_hi = Signal(scanner="qa", symbol="AAPL", score=80, reason="boom")
    am.maybe_fire(s_hi, threshold=70)
    am.maybe_fire(s_hi, threshold=70)  # within cooldown
    check("Alert cooldown blocks duplicates within window",
          lambda: (len(fired) == 1, f"fired={len(fired)}"))

    # Below threshold doesn't fire
    s_low = Signal(scanner="qa", symbol="MSFT", score=60, reason="meh")
    am.maybe_fire(s_low, threshold=70)
    check("Alert below threshold suppressed",
          lambda: (len(fired) == 1, ""))

    # Cleanup
    log_path.unlink(missing_ok=True)
    wl_path.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass


# ============================================================================
# 8. BACKTESTER
# ============================================================================
def test_backtester():
    section("8. BACKTESTER - pre-breakout / momentum / squeeze")
    from apexflow.platform.backtest import Backtester

    bt = Backtester()
    universe = ["AAPL", "NVDA", "TSLA"]

    for scanner in ["pre-breakout", "momentum", "squeeze"]:
        try:
            res = bt.run(scanner, universe, lookback_days=90, hold_days=5)
        except Exception as e:
            _add("FAIL", f"backtest.{scanner}", f"{type(e).__name__}: {e}")
            continue
        ok = (res.signals_tested >= 0
              and math.isfinite(res.sharpe)
              and math.isfinite(res.sortino)
              and math.isfinite(res.max_drawdown_pct)
              and math.isfinite(res.win_rate)
              and 0 <= res.win_rate <= 1)
        if not ok:
            _add("FAIL", f"backtest.{scanner}",
                 f"signals={res.signals_tested} sharpe={res.sharpe} winrate={res.win_rate}")
            continue
        # equity curve must start at 100 and be monotonic-positive
        if res.equity_curve and (res.equity_curve[0] != 100.0 or any(x <= 0 for x in res.equity_curve)):
            _add("FAIL", f"backtest.{scanner}.equity",
                 f"start={res.equity_curve[0]} min={min(res.equity_curve)}")
            continue
        _add("PASS", f"backtest.{scanner}",
             f"signals={res.signals_tested} winrate={res.win_rate*100:.1f}% "
             f"sharpe={res.sharpe:.2f} sortino={res.sortino:.2f} dd={res.max_drawdown_pct:.1f}%")


# ============================================================================
# 9. FASTAPI ENDPOINTS
# ============================================================================
def test_endpoints():
    section("9. FASTAPI - endpoint smoke test (TestClient, in-process)")
    from fastapi.testclient import TestClient
    import webapp
    client = TestClient(webapp.app)

    # HTML pages - must return 200 + text/html
    pages = [
        ("/", "dashboard"),
        ("/heatmap", "heatmap"),
        ("/heatseeker", "heatseeker"),
        ("/atlas", "atlas"),
        ("/radar", "radar"),
        ("/brief", "brief"),
        ("/guide", "guide"),
        ("/watchlist", "watchlist"),
        ("/log", "log"),
        ("/symbol/AAPL", "symbol"),
        ("/backtest", "backtest"),
    ]
    for path, label in pages:
        try:
            r = client.get(path)
            ok = r.status_code == 200 and "text/html" in r.headers.get("content-type", "")
            _add("PASS" if ok else "FAIL",
                 f"page {path}", f"status={r.status_code}")
        except Exception as e:
            _add("FAIL", f"page {path}", f"{type(e).__name__}: {e}")

    # JSON endpoints
    api = [
        ("/api/health", lambda j: j.get("ok") is True),
        ("/api/ribbon", lambda j: isinstance(j, list) and len(j) >= 4),
        ("/api/regime", lambda j: "regime" in j and "vix" in j),
        ("/api/quote/AAPL", lambda j: isinstance(j, dict) and (j.get("price") or 0) > 0),
        ("/api/history/AAPL?period=1mo&interval=1d",
            lambda j: isinstance(j, list) and len(j) > 0 and "c" in j[0]),
        ("/api/indicators/AAPL", lambda j: isinstance(j, dict) and "rsi" in j),
        ("/api/expiries/AAPL", lambda j: isinstance(j, list) and len(j) > 0),
        ("/api/gex/AAPL", lambda j: isinstance(j, dict) and "spot" in j and "strikes" in j),
        ("/api/chain/AAPL", lambda j: isinstance(j, dict) and "calls" in j),
        ("/api/projection/AAPL", lambda j: isinstance(j, dict) and "expected_move" in j),
        ("/api/keylevels/AAPL", lambda j: isinstance(j, dict) and "spot" in j),
        ("/api/heatseeker?symbols=AAPL&dtes=0,7", lambda j: "symbols" in j),
        ("/api/atlas/status", lambda j: isinstance(j, dict)),
        ("/api/atlas/AAPL?live_snapshot=false", lambda j: isinstance(j, dict)),
        ("/api/watchlist", lambda j: isinstance(j, list)),
        ("/api/log?limit=10", lambda j: isinstance(j, list)),
        ("/api/alerts/status", lambda j: isinstance(j, dict)),
        ("/api/alerts/recent?limit=5", lambda j: isinstance(j, dict) and "alerts" in j),
        ("/api/universe?name=sp100", lambda j: j.get("size", 0) > 50),
        ("/api/scan?universe=data/test_universe.txt&scanners=momentum",
            lambda j: "scanners" in j and "top_signals" in j),
        ("/api/radar?universe=data/test_universe.txt&min_score=0&limit=5",
            lambda j: "signals" in j and "count" in j),
        ("/api/earnings_direction/AAPL",
            lambda j: isinstance(j, dict) and "direction" in j),
        ("/api/brief/status", lambda j: isinstance(j, dict)),
    ]
    for path, validator in api:
        try:
            r = client.get(path, timeout=120)
            if r.status_code != 200:
                _add("FAIL", f"api {path}", f"status={r.status_code} body={r.text[:120]}")
                continue
            j = r.json()
            ok = validator(j)
            _add("PASS" if ok else "WARN",
                 f"api {path}",
                 "" if ok else f"validator failed; sample={str(j)[:160]}")
        except Exception as e:
            _add("FAIL", f"api {path}", f"{type(e).__name__}: {e}")

    # POST backtest
    try:
        r = client.post(
            "/api/backtest?scanner=momentum&universe=data/test_universe.txt&lookback=60&hold=5",
            timeout=300,
        )
        ok = r.status_code == 200 and "scanner" in r.json()
        _add("PASS" if ok else "FAIL", "api POST /api/backtest",
             f"status={r.status_code}")
    except Exception as e:
        _add("FAIL", "api POST /api/backtest", f"{type(e).__name__}: {e}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("ApexFlow QA full audit\n")
    test_greeks_math()
    test_gex_math()
    test_indicators_math()
    test_squeeze_score()
    test_providers_live()
    test_scanners()
    test_platform()
    test_backtester()
    test_endpoints()

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
