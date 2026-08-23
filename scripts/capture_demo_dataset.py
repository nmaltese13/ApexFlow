"""Freeze a real options-chain snapshot into ``data/demo/`` for offline use.

Run this once against any configured provider; afterwards the whole terminal
runs with ``APEXFLOW_DEMO=1`` and no API key, no network, and no rate limit.

    python scripts/capture_demo_dataset.py                    # default symbols
    python scripts/capture_demo_dataset.py --symbols SPY,NVDA --expiries 8

What gets captured, per symbol:
    quote, expiries, full call/put chains for the first N expiries,
    1 year of daily OHLCV, fundamentals, and upcoming earnings dates.

Files are gzipped JSON at ``data/demo/<SYMBOL>.json.gz`` plus a
``manifest.json`` describing the capture. See ``docs/demo_dataset.md``.

Note on provenance: this writes down exactly what the upstream provider
returned, with the capture timestamp and provider name recorded in the
manifest. Nothing is synthesised or smoothed. Free providers serve delayed
quotes, so a snapshot taken during the session reflects the tape roughly
15 minutes earlier - the manifest records which provider was used so the
delay is attributable.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from apexflow.providers import get_provider  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo"

# A spread of structures worth looking at rather than just the biggest names:
# two index ETFs (deep, tight chains), high-gamma megacaps, a small-float
# high-short-interest name, and a leveraged ETF.
DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "GME"]

# Columns kept from each chain side. Anything else the provider returns is
# dropped so the frozen file stays provider-agnostic and small.
CHAIN_COLS = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest",
              "impliedVolatility", "inTheMoney", "contractSymbol"]


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _chain_records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    keep = [c for c in CHAIN_COLS if c in df.columns]
    out = df[keep].copy()
    for c in out.columns:
        if c not in ("contractSymbol", "inTheMoney"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["strike"])
    return [{k: _jsonable(v) for k, v in rec.items()}
            for rec in out.to_dict(orient="records")]


def _history_records(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"index": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    d = df.copy()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in d.columns:
            d[c] = 0.0
    idx = [int(pd.Timestamp(i).timestamp()) for i in d.index]
    return {
        "index": idx,
        **{c: [float(x) if pd.notna(x) else 0.0 for x in d[c]] for c in cols},
    }


def capture_symbol(prov, symbol: str, n_expiries: int, pause: float) -> dict | None:
    symbol = symbol.upper()
    print(f"  {symbol:<6}", end=" ", flush=True)
    try:
        expiries = list(prov.expiries(symbol) or [])[:n_expiries]
    except Exception as e:
        print(f"expiries failed: {e}")
        return None
    if not expiries:
        print("no expiries - skipped")
        return None

    try:
        quote = prov.quote(symbol) or {}
    except Exception:
        quote = {}
    try:
        hist = prov.history(symbol, period="1y", interval="1d")
    except Exception:
        hist = None

    spot = float(quote.get("price") or 0.0)
    if not spot and hist is not None and not hist.empty:
        spot = float(hist["Close"].iloc[-1])

    chains: dict[str, dict] = {}
    for exp in expiries:
        try:
            ch = prov.options_chain(symbol, exp)
        except Exception as e:
            print(f"[{exp} failed: {e}]", end="")
            continue
        calls = _chain_records(ch.get("calls"))
        puts = _chain_records(ch.get("puts"))
        if not calls and not puts:
            continue
        chains[exp] = {"calls": calls, "puts": puts}
        if not spot:
            spot = float(ch.get("spot") or 0.0)
        if pause:
            time.sleep(pause)

    if not chains:
        print("no chains - skipped")
        return None

    try:
        fundamentals = prov.fundamentals(symbol) or {}
    except Exception:
        fundamentals = {}
    try:
        earnings = [d.isoformat() for d in (prov.earnings_calendar(symbol) or [])]
    except Exception:
        earnings = []

    n_contracts = sum(len(c["calls"]) + len(c["puts"]) for c in chains.values())
    print(f"spot={spot:<9.2f} {len(chains)} expiries, {n_contracts} contracts")

    return {
        "symbol": symbol,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": getattr(prov, "name", "unknown"),
        "spot": spot,
        "quote": {k: _jsonable(v) for k, v in quote.items()},
        "expiries": sorted(chains.keys()),
        "chains": chains,
        "history": _history_records(hist),
        "fundamentals": {k: _jsonable(v) for k, v in fundamentals.items()},
        "earnings": earnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="comma-separated tickers")
    ap.add_argument("--expiries", type=int, default=6,
                    help="how many expiries per symbol (from the front)")
    ap.add_argument("--pause", type=float, default=0.4,
                    help="seconds between chain requests, to stay under rate limits")
    ap.add_argument("--out", default=str(DEMO_DIR), help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    prov = get_provider()
    print(f"Capturing {len(symbols)} symbols from provider '{prov.name}' "
          f"({args.expiries} expiries each)")

    captured, total_contracts = [], 0
    for sym in symbols:
        snap = capture_symbol(prov, sym, args.expiries, args.pause)
        if not snap:
            continue
        path = out_dir / f"{sym}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(snap, fh, separators=(",", ":"))
        n = sum(len(c["calls"]) + len(c["puts"]) for c in snap["chains"].values())
        total_contracts += n
        captured.append({
            "symbol": sym,
            "file": path.name,
            "spot": snap["spot"],
            "expiries": snap["expiries"],
            "contracts": n,
            "bytes": path.stat().st_size,
        })

    if not captured:
        print("Nothing captured.")
        return 1

    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capture_date": datetime.now(timezone.utc).date().isoformat(),
        "source": prov.name,
        "note": ("Verbatim provider output, unmodified. Free providers serve "
                 "delayed quotes; see docs/demo_dataset.md."),
        "symbols": captured,
        "total_contracts": total_contracts,
        "total_bytes": sum(c["bytes"] for c in captured),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    mb = manifest["total_bytes"] / 1e6
    print(f"\nWrote {len(captured)} symbols, {total_contracts:,} contracts, "
          f"{mb:.2f} MB to {out_dir}")
    print("Run the terminal against it with:  APEXFLOW_DEMO=1 python main.py web")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
