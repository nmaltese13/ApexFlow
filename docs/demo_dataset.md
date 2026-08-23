# The frozen demo dataset

A real options-chain snapshot committed to the repo, so the whole terminal
runs with **no API key, no network, and no rate limit**.

```sh
python main.py --demo web          # or: APEXFLOW_DEMO=1 python main.py web
python main.py --demo demo-info    # what's in the dataset
python main.py --demo dealer-greeks NVDA
```

---

## What's in it

`data/demo/` holds one gzipped JSON file per symbol plus a `manifest.json`.

| | |
|---|---|
| Captured | 2026-08-21 |
| Source | yfinance (free tier) |
| Symbols | SPY, QQQ, IWM, NVDA, TSLA, AAPL, AMD, GME |
| Expiries | 6 per symbol, from the front |
| Contracts | 7,665 |
| Size | 244 KB gzipped |

Per symbol: the quote, the expiry list, full call and put chains for each
expiry, one year of daily OHLCV, fundamentals, and upcoming earnings dates.

The symbol mix is deliberate rather than just "the biggest names" — two
index ETFs with deep tight chains, high-gamma megacaps, and a small-float
high-short-interest name, so the different structures the terminal is meant
to show up are all present in one dataset.

### Provenance

This is **verbatim provider output**, unmodified. Nothing is synthesised,
smoothed, or hand-adjusted. The capture timestamp and provider name are
recorded in `manifest.json`.

It was captured from a free tier, which means quotes are delayed roughly
15 minutes — the snapshot reflects the tape about 15 minutes before the
capture timestamp. It is fine for exercising the analytics and completely
unsuitable as a record of what any instrument was worth at a given instant.

The dataset also contains a genuinely bad chain, on purpose: the front SPY
expiry is 0DTE and its near-the-money implied vols are numerical garbage
(2.1% to 8.8% across adjacent strikes). That is what real free-tier data
looks like, and it is the case
[`analytics/iv_surface.py`](../apexflow/analytics/iv_surface.py) exists to
handle. Freezing a clean chain would have hidden the problem the code is
built around.

---

## Dates are shifted forward

A frozen chain has a problem a frozen price series does not: its expiries
are dated. Left alone, every contract in the file expires within a month of
capture, time-to-expiry floors at one minute, and every gamma number becomes
nonsense — the demo would quietly break rather than fail loudly.

So `DemoProvider` shifts dates forward by default:

```
shift = round_to_whole_weeks(today − capture_date)
```

**Whole weeks specifically**, because equity option expiries are
weekday-anchored. A Friday monthly stays a Friday, the 0DTE contract stays
0DTE, and the gaps between expiries are preserved exactly — so the term
structure keeps its original shape. The same shift is applied to history
timestamps and earnings dates.

Only labels move. **Not one price, implied vol, or open-interest figure is
touched** — `tests/test_demo_provider.py` asserts the chains are identical
frame-for-frame between the shifted and unshifted providers.

To see the raw captured dates instead:

```sh
APEXFLOW_DEMO_SHIFT=0 python main.py --demo demo-info
```

That is the right setting for checking the snapshot against the tape from
that day, and the wrong one for a live demo.

---

## Rebuilding or extending it

```sh
# defaults: 8 symbols, 6 expiries each
python scripts/capture_demo_dataset.py

# your own selection
python scripts/capture_demo_dataset.py --symbols SPY,NVDA,GME --expiries 8
```

The script captures from whatever provider is currently configured
(`python main.py status` shows which), so pointing it at a paid provider
produces a higher-quality snapshot with the same layout. Options:

| Flag | Default | Meaning |
|---|---|---|
| `--symbols` | 8 defaults | comma-separated tickers |
| `--expiries` | 6 | how many expiries per symbol, from the front |
| `--pause` | 0.4 | seconds between chain requests, to stay under rate limits |
| `--out` | `data/demo` | output directory |

Capturing more symbols or expiries is cheap in disk terms — the whole
current dataset is 244 KB — but the free tier rate-limits aggressively, so
keep `--pause` non-zero.

---

## How provider selection works

Demo mode is checked **first**, ahead of every real provider:

```
demo → Schwab → Polygon → MarketData.app → yfinance
```

That ordering is deliberate. When demo mode is on, nothing is reaching the
network at all — which is what makes the dataset safe to hand to someone
else to run, and what makes results reproducible. If `APEXFLOW_DEMO=1` is
set but no dataset is present, the app logs a warning and falls back to the
normal chain rather than failing.

`DemoProvider` implements the full `DataProvider` interface, so every
scanner, endpoint and CLI command works against it unchanged. The one
deliberate gap is `news()`, which returns empty: headlines go stale in a way
prices do not, and a week-old headline presented as current is worse than no
headline.

---

## What it's good for

- **Running the terminal with no credentials** — the whole point.
- **Reproducible tests.** `tests/test_demo_provider.py` runs the real GEX,
  dealer-exposure, IV-extraction and Monte Carlo engines against it, so the
  analytics are exercised on real market data with no network in CI.
- **Comparing changes.** A fixed dataset means a change in output is a
  change in the code, not in the market.

## What it isn't

- **Not a backtest.** One day's snapshot, no point-in-time history. See the
  limitations section of [`methodology.md`](methodology.md).
- **Not a record of prices.** Delayed free-tier data (see Provenance).
- **Not live.** Everything is frozen at the capture instant; only the date
  labels advance.
