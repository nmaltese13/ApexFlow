"""Request every page and API route against the frozen demo dataset.

A fast end-to-end check that the whole app still serves: the unit tests
cover the maths, this covers the wiring between it and the HTTP layer.
Needs no network and no API keys — it forces demo mode, so a failure here
is a real regression rather than a rate limit.

    python scripts/smoke_test.py

Exits non-zero on the first failing route, printing the response body.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before webapp imports a provider.
os.environ["APEXFLOW_DEMO"] = "1"
os.environ.setdefault("APEXFLOW_ATLAS_DISABLE", "1")
os.environ.setdefault("APEXFLOW_BRIEFING_DISABLE", "1")

from fastapi.testclient import TestClient  # noqa: E402

import webapp  # noqa: E402

PAGES = [
    "/", "/heatmap", "/heatseeker", "/atlas", "/radar", "/brief",
    "/earnings", "/guide", "/watchlist", "/log", "/backtest", "/symbol/SPY",
]

API = [
    "/api/health",
    "/api/quote/SPY",
    "/api/gex/SPY",
    "/api/chain/SPY",
    "/api/expiries/SPY",
    "/api/indicators/SPY",
    "/api/history/SPY",
    "/api/keylevels/SPY",
    # Projection across all three simulation models, plus the analytic-only path.
    "/api/projection/SPY?dte_max=7&horizon_days=5",
    "/api/projection/NVDA?model=merton&paths=20000",
    "/api/projection/GME?model=bootstrap&paths=20000",
    "/api/projection/SPY?simulate=false",
    # Dealer exposures under each positioning convention and basis.
    "/api/dealer_greeks/SPY",
    "/api/dealer_greeks/NVDA?convention=inverted",
    "/api/dealer_greeks/AMD?basis=volume",
    "/api/iv_surface/NVDA",
    "/api/mc/validate?paths=20000&steps=32",
    "/api/heatseeker?symbols=SPY,QQQ",
    # Atlas returns an empty frame list on a fresh clone (no captured
    # history yet) — a 200 with a capture_hint, not an error.
    "/api/atlas/SPY",
    "/api/atlas_symbols",
    "/api/earnings_direction/NVDA",
    "/api/watchlist",
    "/api/log",
    "/api/regime",
    "/api/brief/status",
]


def check(client: TestClient, path: str, kind: str) -> tuple[bool, str]:
    try:
        r = client.get(path)
    except Exception:
        return False, traceback.format_exc()[-600:]
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:400]}"
    if kind == "api":
        try:
            body = r.json()
        except ValueError:
            return False, "response was not JSON"
        if isinstance(body, dict):
            # A 200 carrying an error field is still a failure.
            if body.get("error"):
                return False, f"body error: {body['error']}"
            sim = body.get("mc")
            if isinstance(sim, dict) and sim.get("error"):
                return False, f"monte carlo error: {sim['error']}"
    return True, ""


def main() -> int:
    client = TestClient(webapp.app)
    failures: list[tuple[str, str]] = []

    info = getattr(webapp.get_provider(), "info", lambda: {})()
    if info:
        print(f"Demo snapshot: {info.get('capture_date')} from "
              f"{info.get('source')} — {info.get('total_contracts', 0):,} contracts")
        if not info.get("symbols"):
            print("ERROR: demo dataset has no symbols", file=sys.stderr)
            return 1
    print()

    for kind, paths in (("page", PAGES), ("api", API)):
        print(f"{kind.upper()}S")
        for path in paths:
            ok, detail = check(client, path, kind)
            print(f"  {'ok  ' if ok else 'FAIL'} {path}")
            if not ok:
                failures.append((path, detail))
        print()

    total = len(PAGES) + len(API)
    if failures:
        print(f"{len(failures)} of {total} routes FAILED\n", file=sys.stderr)
        for path, detail in failures:
            print(f"--- {path}\n{detail}\n", file=sys.stderr)
        return 1
    print(f"all {total} routes OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
