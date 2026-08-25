# ApexFlow

A terminal for visualising options-market structure and dealer positioning.
It computes dealer exposures (DEX / GEX / VEX / charm / vanna) from an
options chain, extracts a trustworthy implied-vol surface from messy vendor
data, and produces forward price distributions both in closed form and by
Monte Carlo simulation.

It describes market structure. It does not produce trade recommendations,
entries, or signals to act on — see
[`docs/methodology.md`](docs/methodology.md) for what each number means, the
assumptions behind it, and where it breaks.

---

## Run it in 30 seconds, with no API key

A real options-chain snapshot is committed to the repo, so the whole
terminal runs offline:

```sh
pip install -r requirements.txt
python main.py --demo web          # → http://localhost:8501
```

Drop `--demo` for live data. With no API key at all it uses **Cboe** —
exchange-computed IV and Greeks, the whole chain in one keyless request.

7,665 real contracts across SPY, QQQ, IWM, NVDA, TSLA, AAPL, AMD and GME.
No key, no network, no rate limit. Details in
[`docs/demo_dataset.md`](docs/demo_dataset.md).

```sh
python main.py --demo demo-info               # what's in the dataset
python main.py --demo dealer-greeks NVDA      # exposure surface for one name
python main.py validate-mc                    # check the simulator against closed form
```

---

## Analytics

| Module | What it computes |
|---|---|
| [`dealer_greeks.py`](apexflow/analytics/dealer_greeks.py) | DEX, GEX, VEX, charm and vanna exposure per strike; the re-priced zero-gamma level; switchable dealer-positioning convention |
| [`montecarlo.py`](apexflow/analytics/montecarlo.py) | GBM / Merton-jump / historical-bootstrap path simulation, antithetic + control variates, Brownian-bridge barrier correction, streaming O(1)-memory quantiles, closed-form validation harness |
| [`iv_surface.py`](apexflow/analytics/iv_surface.py) | Vega-weighted median ATM IV with quality flags, term structure, 25-delta risk reversal |
| [`projection.py`](apexflow/analytics/projection.py) | Expected move, P(S_T > K), exact first-passage touch probability, quantile cone |
| [`greeks.py`](apexflow/analytics/greeks.py) | Black–Scholes–Merton Greeks; bracketed Newton/bisection IV solver with no-arbitrage rejection |
| [`gex.py`](apexflow/analytics/gex.py) | Per-strike gamma exposure (a thin adapter over `dealer_greeks`, so the two cannot disagree) |
| [`gex_profile.py`](apexflow/analytics/gex_profile.py) | Classifies the gamma landscape: walls, pillars, slides, pins, negative-gamma zones |
| [`rates.py`](apexflow/analytics/rates.py) | Tenor-matched risk-free rate from the Treasury par yield curve — free, keyless, replaces a hardcoded 4% |
| [`timeutil.py`](apexflow/analytics/timeutil.py) | Single source of truth for time-to-expiry — DST-correct 16:00 ET close via `zoneinfo` |
| [`squeeze.py`](apexflow/analytics/squeeze.py) | Composite short-squeeze pressure score, piecewise-linear (no bucket cliffs) |
| [`squeeze_backtest.py`](apexflow/platform/squeeze_backtest.py) | Cross-sectional rank-IC evaluation of that score, with a permutation null, overlap correction, and a coverage gate that refuses under-powered verdicts |
| [`atlas.py`](apexflow/analytics/atlas.py) | Intraday OI/GEX snapshot store + node growth metrics |
| [`earnings_direction.py`](apexflow/analytics/earnings_direction.py) | Five-layer directional positioning read |
| [`key_levels.py`](apexflow/analytics/key_levels.py), [`levels.py`](apexflow/analytics/levels.py) | PDH/PDL, POC/VAH/VAL, HVN/LVN; HVL, walls, charm/vanna peaks |

### Using this with real money

Two safety layers exist because the analytics alone are not a trading tool.

**Data freshness.** A 15-minute-delayed quote used to render identically to
a live one, and a dead feed identically to both — the most expensive failure
mode a tool like this has, because it is invisible. Every payload now
carries an age, measured against the *market session* rather than the wall
clock (data that is 16 hours old on a Saturday is correct; 16 minutes old
on a Wednesday may mean the feed stopped). A feed is never reported fresher
than it can be: Cboe's file is seconds old but its prices are 15 minutes
old, so it tops out at `delayed` and only a real-time provider can read
`live`. Anything stale raises a banner that says not to act on it.

**Position sizing** (`/dealer`, `/api/size`). Fixed-fractional: risk a
constant fraction of equity and let the stop distance set the size, so one
trade cannot ruin you by construction. It reads no market data and takes no
view — it answers "how much", never "whether" — and refuses rather than
guesses when the inputs would produce an unsurvivable position.

**What is deliberately absent:** no broker connection, no order routing, no
automated execution, and no signal generation. The one model formally tested
here showed *no* rank information ([§10](docs/methodology.md#10-does-the-squeeze-score-work)),
so there is no validated edge in this repository to trade. It is a lens on
market structure and a discipline for sizing; the decisions are yours.

### Four things worth knowing about the numbers

**The dealer positioning assumption dominates everything.** GEX, VEX and
DEX all depend on guessing which side of the open interest the market maker
holds. The default ("dealers long calls, short puts") matches published
dashboards and is *frequently wrong for individual names* — invert it and
every sign flips. It's an explicit, switchable parameter, not a hidden
constant. [§2 of the methodology](docs/methodology.md#2-the-dealer-positioning-assumption).

**The zero-gamma level is computed by re-pricing, not cumulative sum.** The
common shortcut — running GEX upward from the lowest strike until it crosses
zero — returns a *strike*, not a price level, and on a put-heavy index chain
it lands hundreds of points below the real flip. On the frozen SPY snapshot:
770.55 re-priced versus 532.27 by cumulative sum, with spot at 765.62.
[§5](docs/methodology.md#5-the-zero-gamma-level).

**Vendor IV on near-expiry contracts is often garbage.** The six SPY strikes
nearest spot on a 0DTE expiry reported 2.1% – 8.8% implied vol. Taking "the
nearest strike" is a coin flip that then propagates into every probability
on the page. `iv_surface.py` filters by relative spread, pools both sides,
takes a vega-weighted median, and returns a quality flag rather than a
confident-looking number. [§6](docs/methodology.md#6-implied-volatility-extraction).

**The squeeze scorer was backtested, and the result is not what a rank
statistic sees.** Point-in-time short interest is free after all — FINRA
twice-monthly (filtered on *publication* date, not settlement, so the
8-business-day lag is respected) plus SEC EDGAR share counts filtered on
their `filed` date. That lifts coverage from 18% to 68%. Over 15,639
observations on high-short-interest names: rank IC −0.049, indistinguishable
from noise. But the top decile's **mean is +14.5% while its median is
−1.6%**, and the share of names moving >20% rises monotonically from 7.2%
to 13.0% across deciles. Most high-scored names drift down; a few explode.
No evidence as a ranking model; suggestive as a tail-exposure filter.
[§10](docs/methodology.md#10-does-the-squeeze-score-work).

---

## The Monte Carlo engine

Simulates forward price distributions under GBM, Merton jump-diffusion, or a
bootstrap of the ticker's own historical returns.

**Validated against closed form.** `python main.py validate-mc` prices
European options, terminal digitals, the martingale expectation and a
one-touch barrier by simulation, and reports each error in Monte Carlo
standard errors:

```
Check                    Analytic     Simulated    Std err       z
European C K=100.00      6.459483     6.450951     0.004458    -1.91
P(S_T > 105.00)          0.369340     0.369208     0.000483    -0.27
E[S_T] (martingale)    101.005017   101.002123     0.015216    -0.19
P(touch 105.00)          0.742956     0.742976     0.000952    +0.02
```

**Why 5,000,000 paths.** The standard error of a simulated probability is
`√(p(1−p)/N)` — 2.2 bp at 5M, against a UI that renders to 0.1%. That puts
sampling noise an order of magnitude below display resolution, so numbers
don't move when the seed changes. It's a claim about display stability, not
model accuracy: more paths do nothing about the error from assuming GBM.
Interactive endpoints default to 200,000 paths because 5M buys nothing
visible.

**The Brownian-bridge correction matters more than the path count.**
Discretely monitored paths under-count barrier hits, because a path can
cross and return between observations. At 64 steps that bias is −5.6
percentage points — more than an order of magnitude larger than the sampling
error all those paths were spent removing:

| Steps | Naive | Bridged | Analytic |
|---:|---:|---:|---:|
| 16 | 0.6377 | 0.7428 | 0.7430 |
| 64 | 0.6871 | 0.7415 | 0.7430 |
| 256 | 0.7151 | 0.7430 | 0.7430 |

Memory is O(chunk × steps) regardless of path count — 5M paths never
materialise, they stream into per-step histograms and counters.

---

## Web pages

`python main.py web` → http://localhost:8501

Nav is grouped into **Structure** (what the market looks like), **Screening**
(what to look at) and **Record** (what happened).

- **Flow** (`/`) — dashboard with live alerts, scanner output, ticker tape
- **Dealer** (`/dealer`) — the full exposure surface: DEX / GEX / VEX / charm / vanna per strike, the re-priced zero-gamma level, vega concentration, and a **switchable dealer-positioning convention** — flip it and every sign flips, which is the honest way to present a number most dashboards quote as observed
- **Seek** (`/heatseeker`) — multi-DTE matrix of strikes as magnitude bars, king/wall annotations
- **Vol** (`/vol`) — implied-vol term structure and 25Δ skew, with a **confidence flag on every expiry**: near-expiry contracts routinely quote IVs that disagree by a factor of four across adjacent strikes, and points that fail that test are hidden rather than plotted next to good ones
- **Atlas** (`/atlas`) — intraday GEX node history; strikes render as orb bands that grow as they persist, with a time scrubber to replay the session
- **Radar** (`/radar`) — composite squeeze scanner across the universe, six layer scores plus a synergy bonus
- **Brief** (`/brief`) — pre-computed analysis on a background cadence
- **Symbol** (`/symbol/{sym}`) — price, indicators, options chain, and the per-strike GEX panel (the old standalone `/heatmap` page, folded in — that URL now redirects here)
- **Journal** (`/journal`) — watchlist and signal log as two tabs (they were separate pages answering the same question)
- **Guide** (`/guide`) — how every number is computed, plus the backtesting commands

Retired, all redirecting rather than 404ing: `/heatmap` → the symbol page,
`/watchlist` and `/log` → `/journal`, `/backtest` → the Guide (the page only
replayed three price-based scanners; the real validation is
`backtest-squeeze` in the CLI).

### Selected API endpoints

| Endpoint | Returns |
|---|---|
| `/api/dealer_greeks/{sym}` | Full exposure surface; `?convention=naive\|inverted\|all_short`, `?basis=openInterest\|volume`. Rendered by the **Dealer** page. |
| `/api/projection/{sym}` | Closed-form **and** simulated forward distribution side by side; `?model=gbm\|merton\|bootstrap` |
| `/api/iv_surface/{sym}` | ATM IV per expiry with quality flags, term-structure shape, 25Δ skew |
| `/api/mc/validate` | The validation table above, from the running app |
| `/api/gex/{sym}`, `/api/keylevels/{sym}`, `/api/heatseeker` | Per-strike GEX, S/R levels, multi-symbol matrix |

---

## Scanners

| Name | What it does |
|---|---|
| `options-flow` | Volume >> OI, sweep/block detection |
| `pre-breakout` | BB squeeze + volume dry-up + alignment |
| `momentum` | First-5-min RVOL ignition + catalyst tags |
| `squeeze` | Short pressure composite |
| `earnings` | IV vs historical move, C/P OI bias |
| `radar` | Composite of all six layers + synergy bonus |

---

## Data providers

Selection order: **demo → Schwab → Polygon → MarketData.app → Cboe →
yfinance**. Demo is checked first so that when it's on, nothing reaches the
network; Cboe outranks yfinance because it is better on every axis that
matters and needs no key.

Keys go in `keys.py` (copy `keys.example.py`). `python main.py status` shows
what's configured and which provider is active.

| Key | Cost | Notes |
|---|---|---|
| *(none)* | **Free** | **Cboe** delayed chains — exchange-computed IV + Greeks, all expiries in one request, no key. The default. |
| *(none)* | **Free** | Demo snapshot (offline), or yfinance as last-resort fallback |
| `APEXFLOW_SEC_USER_AGENT` | **Free** | Your name + email. Unlocks SEC share counts for the point-in-time backtest. FINRA needs nothing. |
| `SCHWAB_APP_KEY` + `SECRET` | **Free** with a Schwab brokerage account | Real-time chains with Greeks. Run `python main.py schwab-auth` once. |
| `POLYGON_KEY` | ~$29/mo | Real-time options chains + IV + Greeks |
| `MARKETDATA_TOKEN` | Free 100/day, ~$15–30/mo | Cheap fallback with Greeks |
| `FINNHUB_KEY` | Free tier | Earnings calendar + news |
| `FINTEL_KEY` | ~$30/mo | Short interest, days-to-cover, borrow fee |
| `UNUSUAL_WHALES_KEY` | ~$48–75/mo | Trade-level flow with aggressor side, dark pool |

The upgrade that most improves the *numbers* is a real-time chain with
vendor Greeks, because it fixes the IV-quality problem — and Schwab provides
that free with a brokerage account. Full cost breakdown and what each tier
actually fixes: [`docs/data_sources.md`](docs/data_sources.md).

---

## CLI

| Command | What it does |
|---|---|
| `python main.py --demo web` | Launch the web app against the frozen snapshot |
| `python main.py web --port 9000 --reload` | Different port / auto-reload |
| `python main.py validate-mc` | Validate the Monte Carlo engine against closed form |
| `python main.py dealer-greeks TSLA --dte 30` | Exposure surface for one symbol |
| `python main.py demo-info` | Describe the frozen dataset |
| `python main.py status` | Configured keys + active provider |
| `python main.py gex TSLA` | CLI GEX heatmap |
| `python main.py scan radar` | Run a scanner |
| `python main.py hub --once` | Run all scanners once |
| `python main.py backtest pre-breakout --hold 5` | Backtest a price-based scanner |
| `python main.py backtest-squeeze --universe squeeze` | Point-in-time backtest of the squeeze score (FINRA + SEC) |

## Environment toggles

| Var | Effect |
|---|---|
| `APEXFLOW_DEMO=1` | Use the frozen snapshot (same as `--demo`) |
| `APEXFLOW_SEC_USER_AGENT` | `"Your Name you@example.com"` — required by the SEC for share counts |
| `APEXFLOW_DISABLE_CBOE=1` | Fall back to yfinance for options |
| `APEXFLOW_STATIC_RATE=1` | Use a flat 4% instead of the live Treasury curve |
| `APEXFLOW_DEMO_SHIFT=0` | Show raw captured dates instead of shifting them forward |
| `APEXFLOW_ATLAS_DISABLE=1` | Skip the Atlas snapshot loop |
| `APEXFLOW_BRIEFING_DISABLE=1` | Skip the briefing engine |
| `APEXFLOW_<KEY_NAME>` | Override any `keys.py` value |

---

## Tests

```sh
python -m pytest tests/ -q
```

329 tests, no network access and no API keys required. The numerical
engines are checked against **independently derived** values rather than
against their own output:

- **Greeks** — the standard textbook reference case (S=K=100, t=1yr, r=5%,
  σ=20%), plus put-call parity, delta parity, and central finite differences
  of the pricing function.
- **Monte Carlo** — every simulated quantity compared to its closed form
  with the error expressed in standard errors; convergence asserted to fall
  as `1/√N`; the bridge correction asserted to remove a bias that is
  asserted to exist without it.
- **GEX / exposures** — hand-computed magnitude for a one-contract chain (so
  a lost factor of 100 or a missing `S²` fails), sign conventions under all
  three dealer assumptions, and agreement between `gex.py` and
  `dealer_greeks.py`.
- **Time to expiry** — DST correctness on both sides of the transition, and
  that `greeks.py` and `gex.py` return the *same* answer.
- **IV extraction** — the real noisy 0DTE chain from the snapshot must come
  back flagged `poor`.
- **Demo dataset** — integrity, date-shift invariants (prices identical
  frame-for-frame), and the real analytics engines running against it.
- **Backtest harness** — *power* tests, not just correctness: planted
  signals of known strength must be detected (and inverted ones flagged as
  negative), pure noise must not be, and low coverage must block a verdict
  even when a strong signal is plainly present.

---

## Layout

```
apexflow/
  analytics/     dealer_greeks, montecarlo, iv_surface, projection, greeks,
                 gex, gex_profile, timeutil, squeeze, atlas, levels, ...
  platform/      atlas_loop, briefing, live_alerts, scanner_hub,
                 backtest, squeeze_backtest
  providers/     demo, cboe, shortinterest (FINRA+SEC), schwab, polygon,
                 marketdata, tradier, yfinance, unusual_whales, fintel, finnhub
  scanners/      squeeze_radar, earnings_*, momentum_ignition, options_flow
  ui/            terminal UI
scripts/         capture_demo_dataset.py
docs/            methodology.md, data_sources.md, demo_dataset.md
data/demo/       frozen chain snapshot (committed)
templates/       Jinja2 HTML
static/          css + js
tests/
```

## Disclaimer

Educational research tool. Not investment advice. Data may be delayed or
wrong; several outputs are conditional on modelling assumptions documented
in [`docs/methodology.md`](docs/methodology.md).
