# Data Sources for ApexFlow

> What to subscribe to, in what order, to get from "free yfinance" to
> "Atlas-grade signal quality." Pricing as of 2026-Q1; verify before you sign.

---

## TL;DR — recommended path

| Tier | Cost / mo | Add | What it unlocks |
|------|----------:|-----|-----------------|
| **Demo** | $0 | Nothing — frozen snapshot in the repo | Runs the entire terminal offline with no key. See [`demo_dataset.md`](demo_dataset.md). |
| **0 — Free start** | $0 | Default yfinance | OHLCV + delayed options chains. Good enough to wire/tune the math. |
| **1 — Real-time options** | $19 | **Tradier Pro** | Real-time chains + Greeks + IV. Kills the 15-min delay. **Biggest single upgrade.** |
| **2 — Squeeze data** | $20 | **Fintel API** | SI%, days-to-cover, CTB, FTDs, ownership changes. Required for short-squeeze scanner to be predictive. |
| **3 — Earnings + news** | $0–$25 | **Finnhub free → $25** | Earnings calendar, news, transcripts. Free tier already wired in repo. |
| **4 — Real flow** | $50–$100 | **Unusual Whales API** | Sweeps, blocks, dark pool, real-time flow stream. Replaces inferred flow with observed flow — the largest single improvement in what the positioning numbers are actually measuring. |
| **5 — Pro term-structure** | $100+ | **Theta Data** | Full historical chain + IV surface. Required for serious backtesting and skew modeling. |

Total to run a *real* setup: about **$110–$150/mo** at Tiers 1–4. Skip Tier
5 unless you're backtesting earnings strategies seriously.

---

## Keyless sources actually in use

Every one of these was probed and verified working; none needs an account.

| Source | Provides | Used for |
|---|---|---|
| **Cboe** delayed quotes | Full option chain, exchange-computed IV + Greeks, all expiries in **one** request. Covers SPX/VIX. | The default options provider |
| **FINRA** consolidated short interest | Shares short, ADV, days-to-cover, twice monthly back to 2020 | Point-in-time squeeze backtest |
| **SEC EDGAR** XBRL | Shares outstanding stamped with the **filed** date | Short-%-of-float, point-in-time |
| **US Treasury** par yield curve | Full curve daily, 1M to 30Y | Tenor-matched risk-free rate |
| **Nasdaq Trader** symbol directory | ~5,600 listed symbols with ETF / test-issue flags | Universe construction |

SEC is the only one wanting anything from you: a contact email in the
User-Agent (`APEXFLOW_SEC_USER_AGENT`), which is their published policy.
There is deliberately no default — see `providers/shortinterest_provider.py`.

### Probed and rejected

Worth recording so nobody re-checks them:

| Source | Why not |
|---|---|
| IBKR shortable-instruments file | The public `usa.txt` path 404s and the FTP mirror times out. **Borrow rate remains genuinely paid-only.** |
| Stooq CSV | JavaScript-gated; returns a browser-check page, not data |
| FRED `fredgraph.csv` | Intermittently unreachable. Treasury's own feed covers the same need more directly |
| Cboe market-statistics JSON | 403 on the daily put/call and volume endpoints |

### Free tiers that need a signup

Not wired in, listed for completeness. Each needs an account and a key, and
none was verified here — treat the limits as documented rather than tested:

| Service | Free tier | Would add |
|---|---|---|
| **Alpaca** | IEX real-time, generous | Real-time equity quotes; IEX-only depth |
| **Tiingo** | ~50 symbols/hr | Clean EOD history, fundamentals |
| **Twelve Data** | 800 req/day | Quotes + some fundamentals |
| **Alpha Vantage** | 25 req/day | Very restrictive at this point |
| **Polygon** | 5 req/min | Same data as the paid tier, throttled hard |
| **Finnhub** | 60 req/min | Earnings calendar + news (already wired) |

The honest summary: **for options specifically, the free keyless path is now
about as good as it gets short of paying.** Cboe gives exchange-computed IV
and Greeks, which is the thing that actually limits analysis quality. The
remaining gaps are borrow rate and true free float, and neither is available
free from any source found.

---

## The $0 path, in detail

There are two free modes, and they answer different questions.

### Demo mode — frozen snapshot, no network at all

```sh
python main.py --demo web
```

Serves a real options-chain snapshot committed to `data/demo/` (7,665
contracts across 8 symbols, 244 KB). No API key, no network, no rate limit,
and identical output on every run. This is the mode to use for running the
terminal on a machine with no credentials, for reproducible tests, and for
handing the project to someone else.

Its limits are exactly what you'd expect: one day, eight symbols, and
nothing live. Full detail in [`demo_dataset.md`](demo_dataset.md).

### Tier 0 — live yfinance

* **What you get:** delayed (≈15 min) OHLCV, options chains for most
  tickers, vendor IV, 60 days of intraday at 5m, 730 days of 1h,
  fundamentals.
* **Why it's enough to build on:** the GEX/VEX/charm/vanna math doesn't
  care about latency. Every calculation in
  [`methodology.md`](methodology.md) can be written, tested and inspected
  against free data.
* **Where it actually hurts**, in order of severity:
  1. **Vendor IV is unreliable on illiquid and near-expiry contracts.**
     This is the real problem, and it is worse than the delay. Near-dated
     ATM strikes can report implied vols that disagree by a factor of four
     (see §6 of [`methodology.md`](methodology.md)). Since IV feeds every
     Greek, every exposure and every probability, bad IV corrupts
     everything downstream. `analytics/iv_surface.py` filters and
     quality-flags it; it cannot manufacture data that was never there.
  2. **Open interest is T+1 anyway.** Worth noting that the *delay* mostly
     doesn't bite for positioning work: OI is published after the fact by
     every provider, free or paid, so dealer exposures are inherently a
     snapshot of yesterday's book plus today's volume.
  3. **15-minute quote delay**, which matters for spot and therefore for
     where spot sits relative to the computed levels.
  4. **Aggressive rate limiting** — HTTP 429 after a few hundred requests
     per IP. The provider implements shared exponential backoff (60s → 600s)
     and serves stale cache while throttled.
* **The practical universe size on free data** is a few dozen symbols on a
  slow refresh, not the S&P 500 in real time. `data/test_universe.txt` and
  the tiered `AtlasLoop` cadences exist for this reason.
* **Wired in:** `apexflow/providers/yfinance_provider.py`.

### What money actually buys you

Ranked by how much it improves the *numbers*, which is not the same as
ranked by price:

| Upgrade | Fixes |
|---|---|
| Real-time chain with **vendor Greeks** (Tradier, Schwab, Polygon) | The IV quality problem above — the single biggest accuracy win |
| **Trade-level flow with aggressor side** (Unusual Whales) | Turns the dealer-positioning *assumption* into a measurement (§2 of `methodology.md`) |
| **Historical chain archive** (Theta Data) | Makes backtesting the squeeze scorer possible at all |
| Real-time **short-interest / borrow** (Fintel) | Squeeze inputs stop being stale weekly figures |

Note that **Schwab is free with a brokerage account** and gives real-time
chains with Greeks — for anyone who already has one, that is the cheapest
route past the worst limitation on this list.

---

## Tier 1 — Real-time options chains (the big upgrade)

### Tradier — recommended

* **Cost:** $0/mo brokerage account → free real-time data, **but** for
  programmatic use the **Pro tier** is $20/mo and removes the 100 req/min
  cap.
* **What it gives you:** real-time chains (no delay), real-time bid/ask,
  Greeks computed server-side, all expiries, all strikes. The 15-min delay
  on your GEX numbers vanishes — your kings and gatekeepers update with
  the chain.
* **Coverage:** US equity options + ETFs. Indexes via SPX symbol.
* **API quality:** REST + websocket. Reasonable docs.
* **Already stubbed:** `apexflow/providers/tradier_provider.py` — needs
  `TRADIER_KEY` filled in.
* **Sign up:** tradier.com/products/brokerage — open an account → Developer
  Portal for API keys.

### Alternatives at this tier

* **MarketData.app** — $19/mo, also real-time chains, slightly different
  REST shape. Stub already in repo. Good if Tradier gives you trouble.
* **Polygon.io Options Starter** — $29/mo. Cleaner historical data than
  Tradier, websocket flow built-in. Good if you also need historical
  back-data for research.
* **Schwab/TD API** — free if you have a brokerage account, but the OAuth
  flow is a hassle and they rate-limit.

### What it changes

The composite scanner's `gamma_squeeze_score` flips from "indicative" to
"actionable." You can run the heatseeker page during market hours and the
nodes update every 30s instead of being 15 min old.

---

## Tier 2 — Short-squeeze data (Fintel)

* **Cost:** $20/mo (Insider tier) or $40/mo (Premium with API).
* **What it gives you:** short interest %, days-to-cover, **cost to borrow
  (CTB)**, **FTDs (Failures To Deliver)**, institutional ownership changes,
  short squeeze scores. CTB and FTDs are the lead indicators
  `methodology.md §4` calls out — neither is on yfinance.
* **Why it matters:** without CTB, the short-squeeze scanner runs on
  semi-monthly Finra data that's 2-3 weeks old. With CTB, you can detect
  borrow-rate inflections (the +15pp / 5-day rule from §4) in real time.
* **Already stubbed:** `apexflow/providers/fintel_provider.py` — needs
  `FINTEL_KEY`.

### Cheaper alternative

* **Ortex** — full-featured but $100/mo+.
* **IBKR's "Securities Lending" data** — free if you have an Interactive
  Brokers account, with a clunky API but accurate borrow rates.
* **finra_short_volume CSVs** — free, daily, but only short volume, not
  borrow.

---

## Tier 3 — Earnings + news (Finnhub)

* **Cost:** Free tier 60 calls/min; Pro $25/mo for 300/min and longer
  history.
* **What you need it for:** confirmed earnings dates (yfinance is wrong
  often enough to be dangerous), pre/post times, EPS estimates, news flow,
  press releases.
* **Already wired:** `apexflow/providers/finnhub_provider.py` — needs
  `FINNHUB_KEY`. Free tier is fine to start.
* **Alternative:** Benzinga ($199/mo) is the gold standard for newsflow but
  expensive. Polygon.io includes some news on its options tier.

---

## Tier 4 — Real options flow (Unusual Whales)

* **Cost:** $48/mo (Trader) → $100/mo (Pro with API access).
* **What it gives you:** real-time sweep/block detection, dark pool prints,
  premium-weighted aggressors, alerts. Without this, the
  `OptionsFlowScanner` runs on yfinance volume/OI ratios — directional
  hints, not actual flow.
* **Why it matters:** every meaningful gamma squeeze is preceded by sweep
  buying at OTM call strikes. UW shows you that sweep buying *as it
  happens*, with size, premium, and aggressor side. CAR last week was
  visible in UW two days before the move.
* **Already stubbed:** `apexflow/providers/unusual_whales_provider.py` —
  needs `UNUSUAL_WHALES_KEY`.
* **Alternative:** Cheddar Flow ($50/mo) — similar product, slightly
  different UI. Flowalgo, BlackBoxStocks — also similar. UW is the most
  developer-friendly.

---

## Tier 5 — Pro IV surface (Theta Data)

* **Cost:** $80/mo (Standard) → $200/mo (Pro).
* **What it gives you:** historical options chain bars, full IV surface
  history, no rate limits worth speaking of. The data you need to backtest
  earnings strategies properly (per `methodology.md §5`).
* **When to add it:** only when you've validated the live signal stack and
  want to backtest 5+ years of data with real chain history. Not needed
  to *find* squeezes; needed to *prove* a strategy historically.
* **Alternative:** ORATS ($79/mo+) — broader pro coverage but pricier API
  tiers. CBOE LiveVol is the institutional standard but $1000+/mo.

---

## Free supplementary feeds (always-on)

* **stocktwits.com/api** — sentiment + message volume. Useful as a
  "retail rotation" detector. No key needed, low rate-limited.
* **NYSE Reg SHO threshold list** — free CSV of stocks with persistent
  FTDs. Strong squeeze precursor.
* **Finra Short Sale Volume daily files** — free, daily, public.
* **CBOE put/call ratio + VIX term-structure** — free CSVs.
* **FRED** for risk-free rate (we use 4% as a default in greeks calc; pull
  the actual 3-month T-bill if you want exactness).

---

## Configuration in this repo

All keys live in `keys.py` (template at `keys.example.py`):

```python
# keys.py
UNUSUAL_WHALES_KEY = "uw_..."
FINTEL_KEY         = "fintel_..."
FINNHUB_KEY        = "fh_..."
TRADIER_KEY        = "..."
POLYGON_KEY        = "..."
MARKETDATA_KEY     = "..."
ALPACA_KEY         = "..."
ALPACA_SECRET      = "..."
```

Or override via environment:

```
APEXFLOW_UNUSUAL_WHALES_KEY=...
APEXFLOW_FINTEL_KEY=...
APEXFLOW_FINNHUB_KEY=...
```

`config.configured_providers()` reports which are wired. The dashboard top
bar shows it live, and `LiveAlertsEngine._alerts_profile()` automatically
scales the universe + cadence based on which paid providers are present
(see `webapp.py` lines 934–944).

---

## Suggested upgrade order

1. **Tradier first ($20)** — biggest quality jump.
2. **Fintel second ($20)** — unlocks the squeeze prediction lead time.
3. **UW third ($50)** — turns the flow scanner from heuristic to real.
4. **Finnhub Pro fourth ($25)** — only if free-tier rate limits start
   biting once your watchlist grows.
5. Theta Data — only when you're serious about historical backtests.

Anything below Tier 4 and you're guessing at flow. Anything below Tier 2
and you're guessing at borrow stress. Anything below Tier 1 and your
heatseeker is showing 15-minute-old kings — useful for context, useless
for entries.
