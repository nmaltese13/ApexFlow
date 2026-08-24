# ApexFlow — Calculation Methodology

How every number in this terminal is computed, what it assumes, and where it
breaks. Each section names the module that implements it, and every module
named here exists — if a formula appears below, you can open the file and
read the code that produces it.

**What this system is.** It visualises options-market structure and dealer
positioning. It computes exposures, probabilities and distributions from an
options chain, and displays them. It does not produce trade
recommendations, entries, targets, or signals to act on, and the outputs
should not be read as forecasts. Several of the quantities below are
*conditional on a modelling assumption about who holds which side of the
open interest* — an assumption that is stated explicitly, is often wrong for
an individual name, and is switchable.

---

## Contents

1. [Conventions](#1-conventions)
2. [The dealer positioning assumption](#2-the-dealer-positioning-assumption)
3. [Greeks](#3-greeks)
4. [Dealer exposures: DEX, GEX, VEX, charm, vanna](#4-dealer-exposures)
5. [The zero-gamma level](#5-the-zero-gamma-level)
6. [Implied volatility extraction](#6-implied-volatility-extraction)
7. [Forward distributions: closed form](#7-forward-distributions-closed-form)
8. [Forward distributions: Monte Carlo](#8-forward-distributions-monte-carlo)
9. [Composite scores](#9-composite-scores)
10. [Does the squeeze score work?](#10-does-the-squeeze-score-work)
11. [Known limitations](#11-known-limitations)
12. [Not yet implemented](#12-not-yet-implemented)
13. [References](#13-references)

---

## 1. Conventions

| Symbol | Meaning |
|---|---|
| `S` | spot price of the underlying |
| `K` | strike |
| `t` | time to expiry, in **calendar** years |
| `σ` | implied volatility, annualised, decimal (0.30 = 30%) |
| `r` | risk-free rate, decimal (default 0.04) |
| `q` | continuous dividend yield, decimal (default 0.0) |
| `OI` | open interest, in contracts |
| `100` | contract multiplier (shares per US equity option contract) |

**Calendar years, not trading years.** `t` uses 365 days, not 252. Every
provider in this repo quotes IV annualised on a calendar basis, so a 252-day
convention here would misprice every contract by roughly √(365/252) ≈ 1.20 —
a 20% error in every Greek. This is a common and expensive mix-up.

**Time to expiry** is computed in one place, `analytics/timeutil.py`, and
measured to the option's actual last moment of trading: 16:00
America/New_York on the expiry date, resolved through `zoneinfo` so daylight
saving is handled. That is 20:00 UTC in summer and 21:00 UTC in winter.

> This used to be wrong in two different ways at once. `greeks.py` and
> `levels.py` each computed `(expiry - now).days + 0.5` over 365, while
> `gex.py` used a calendar-day difference plus an intraday remainder pinned
> to a fixed 20:30 UTC close. The same contract therefore had a different
> `t` — and so a different gamma — depending on which module reached it
> first, and the 20:30 constant was up to half an hour off in each
> direction. On a 0DTE contract with two hours left, half an hour is a 25%
> error in `t` and roughly a 13% error in gamma.

Expired contracts floor at one minute of remaining life rather than zero, so
gamma stays finite instead of dividing by zero.

---

## 2. The dealer positioning assumption

Every exposure in section 4 depends on knowing **which side of the open
interest the market maker holds**. Open interest tells you a contract
exists. It does not tell you who is long it.

The default here is the "naive dealer" convention used by SqueezeMetrics,
SpotGamma, and most public GEX dashboards:

> **Dealers are long call open interest and short put open interest.**

The reasoning is that the typical customer sells calls (covered calls,
overwriting) and buys puts (portfolio protection), so the dealer takes the
other side of both. Under this assumption dealers are net long delta —
long calls plus short puts is a synthetic long — which is why DEX comes out
positive for most names.

**This assumption is frequently wrong for individual tickers.** A name where
customers are aggressively buying calls has dealers *short* calls, which
inverts the sign of every number below. `analytics/dealer_greeks.py`
therefore takes a `convention` argument:

| Convention | Call sign | Put sign | When it applies |
|---|---:|---:|---|
| `naive` (default) | +1 | −1 | Comparable with published dashboards |
| `inverted` | −1 | +1 | Customers are the call buyers — a call-buying mania |
| `all_short` | −1 | −1 | Customers bought both sides |

The default stays `naive` so numbers line up with other dashboards, not
because it is correct. Anywhere a GEX figure is quoted without a stated
convention — here or anywhere else — it is carrying this assumption
silently.

Distinguishing these properly needs trade-level data with an aggressor flag
(which side hit the bid or lifted the offer), which is what a flow feed like
Unusual Whales provides and what open interest alone cannot. See
[`data_sources.md`](data_sources.md).

---

## 3. Greeks

`analytics/greeks.py` (scalar), `analytics/dealer_greeks.py::greek_grid`
(vectorised). Black–Scholes–Merton, European exercise.

With

```
d1 = [ln(S/K) + (r − q + σ²/2)·t] / (σ√t)
d2 = d1 − σ√t
```

| Greek | Formula | Units as reported |
|---|---|---|
| delta | `e^(−qt)·N(d1)` (call), `−e^(−qt)·N(−d1)` (put) | per +$1 in S |
| gamma | `e^(−qt)·φ(d1) / (S·σ·√t)` | per +$1 in S |
| vega | `S·e^(−qt)·φ(d1)·√t · 0.01` | **per +1 vol point** |
| theta | annual θ / 365 | per calendar day |
| rho | `K·t·e^(−rt)·N(d2) · 0.01` | per +1 percentage point of rate |
| charm | `∂Δ/∂t` (see below) | per year |
| vanna | `−e^(−qt)·φ(d1)·d2 / σ` | per +1.0 of σ |

```
charm_call = −q·e^(−qt)·N(d1) + e^(−qt)·φ(d1)·[2(r−q)t − d2·σ√t] / (2t·σ√t)
charm_put  = +q·e^(−qt)·N(−d1) + e^(−qt)·φ(d1)·[2(r−q)t − d2·σ√t] / (2t·σ√t)
```

The vega and rho scalings are the fiddly part: both are divided by 100 so
they read "per one point" rather than "per 1.0 of the raw variable". Vanna
is **not** scaled, so callers using it in a per-vol-point context multiply by
0.01 themselves. `tests/test_greeks.py` pins all of these against the
standard textbook reference case (S=K=100, t=1, r=5%, σ=20%) and against
central finite differences of the pricing function, which is an independent
derivation rather than the implementation checking itself.

**American exercise is not modelled.** For US equity options this is a good
approximation for calls on non-dividend payers and a poor one for deep ITM
puts, where early exercise carries real value. Treat put deltas near the
money-forward boundary as indicative.

### Implied volatility

`greeks.implied_vol` inverts the pricing function. Newton–Raphson converges
in a few iterations near the money, and is unreliable exactly where chains
are messiest — deep OTM contracts where vega is near zero and one step
overshoots into negative vol. So:

1. **Reject prices outside the no-arbitrage band** up front. A price above
   the discounted forward or below intrinsic has no implied vol, and a
   solver that returns one is inventing data.
2. **Seed with Brenner–Subrahmanyam**, `σ ≈ √(2π/t)·price/S`, rather than a
   flat 0.5.
3. **Maintain a bracket and bisect** whenever Newton leaves it or stalls.
   Price is monotone in vol, so the bracket always contains the root when
   one exists.
4. **Return `None` on failure**, rather than the last iterate.

---

## 4. Dealer exposures

`analytics/dealer_greeks.py::chain_exposures`. With `sign` = +1 for calls
and −1 for puts under the naive convention:

| Exposure | Formula | Units |
|---|---|---|
| **DEX** | `sign · Δ · OI · 100 · S` | $ of stock dealers hold |
| **GEX** | `sign · Γ · OI · 100 · S² · 0.01` | $ of delta traded per **+1%** in S |
| **VEX** | `sign · vega · OI · 100` | $ of P&L per **+1 vol point** |
| **Charm exposure** | `sign · charm · OI · 100 · S / 365` | $ of delta to re-hedge per day |
| **Vanna exposure** | `sign · vanna · OI · 100 · S · 0.01` | $ of delta to re-hedge per +1 vol point |

Total exposure is the sum across every strike and every expiry in the
window.

**Where the GEX scaling comes from.** Gamma is `∂²V/∂S²` — delta per dollar
per dollar. Multiplying by `OI · 100` converts per-contract to the whole
book. Multiplying by `S` once converts "delta per $1 move" into "delta
per 1% move"… but delta is itself a share count, so a second factor of `S`
converts share count to dollars. The `0.01` then rescales from "per 100%"
to "per 1%". Net: `Γ · OI · 100 · S² · 0.01` is **dollars of stock the
dealer must buy or sell for a 1% move in the underlying**. This is why GEX
is quoted in dollars-per-percent and not in any Greek's native units.

### Reading them

- **DEX** — the net stock position implied by the book. Large positive DEX
  means dealers hold a lot of stock as a hedge; that inventory has to be
  sold when the options justifying it roll off.
- **GEX** — the sign of the hedging feedback loop. Positive: dealers sell
  strength and buy weakness, damping realised vol. Negative: they buy
  strength and sell weakness, amplifying it.
- **VEX** — P&L sensitivity to a vol move, and therefore how much vega
  dealers must buy back if IV rises. Strongly negative VEX into a catalyst
  is the configuration where a vol spike forces dealers to buy options near
  spot — which also buys them gamma, changing the hedging regime mid-move.
- **Charm exposure** — hedge drift from time alone. On expiry afternoons
  OTM deltas collapse and dealers unwind the shares backing them; this is
  the mechanism behind the familiar OPEX-afternoon drift.
- **Vanna exposure** — hedge drift from vol alone. When IV falls, put
  deltas shrink, and dealers short stock against those puts buy it back.
  This is why a post-event vol crush so often comes with a grind higher
  that has no news attached.

`gex.py` is a thin adapter over `chain_exposures`, so GEX cannot drift out
of agreement with DEX and VEX computed from the same chain;
`tests/test_gex_engine.py` pins that agreement.

**IV fallback.** Contracts with missing or zero IV fall back to the chain's
own ATM IV (section 6) rather than a hardcoded 30%, floored at 1% to keep
the `1/σ` terms in gamma and vanna finite.

**Basis.** Exposures are computed from open interest by default — the
resting book. Passing `oi_col="volume"` computes them from the day's volume
instead, which approximates newly-added positioning rather than accumulated
positioning.

---

## 5. The zero-gamma level

`analytics/dealer_greeks.py::gamma_flip_level`.

The price at which total dealer gamma changes sign. Above it dealers damp
moves; below it they amplify. This is the single most-quoted number in
dealer-positioning analysis, and there are two ways to compute it that give
very different answers.

**The shortcut (what most code does).** Sum GEX across strikes from the
lowest upward and find where the running total crosses zero. Cheap, and
what `gex_summary`'s `gamma_flip` field returns. But the number it produces
is *a strike*, not a price level, and on a put-heavy index chain it is badly
wrong: the cumulative sum starts deeply negative because of the put wall at
the bottom of the chain, and crosses zero as soon as enough call gamma
accumulates — which can be hundreds of points below where dealer gamma
actually turns positive. On the frozen SPY snapshot in `data/demo/` it
returns 532 against a spot of 766.

**The correct question** is: *at what underlying price would total dealer
gamma be zero?* Every contract's gamma depends on where spot is, so
answering it requires re-pricing the whole book at candidate spot levels:

```
for S' in grid spanning spot ± 25%:
    GEX_total(S') = Σ_contracts  sign · Γ(S', K, t, σ) · OI · 100 · S'² · 0.01
find the sign change in GEX_total(S'), interpolate
```

IV is held fixed per contract, so this is a "what does the gamma profile
look like if price were there today" curve, not a forecast.

On the frozen snapshot the two methods give:

| Symbol | Spot | Re-priced level | Cumulative-sum strike |
|---|---:|---:|---:|
| SPY | 765.62 | **770.55** | 532.27 |
| NVDA | 214.72 | **208.83** | 66.50 |
| GME | 18.19 | **17.62** | 13.97 |

When no sign change exists in the searched band, `bracketed` comes back
`False` and `flip` is `None`. "No flip within ±25%" is the honest answer
there; inventing a level is not.

---

## 6. Implied volatility extraction

`analytics/iv_surface.py`.

Getting "the" IV for a ticker looks trivial — take the strike nearest spot
and read its `impliedVolatility`. On real data this is a trap. From the
frozen SPY snapshot, the six strikes nearest spot on the front (0DTE)
expiry reported:

```
766 → 2.9%    765 → 5.2%    767 → 2.1%
764 → 7.6%    768 → 3.3%    763 → 8.8%
```

That is not a volatility smile. It is what a vendor's solver returns when
handed a contract worth one or two cents with a penny-wide spread and hours
of life left: vega is near zero, the inversion from price to vol is
numerically hopeless, and rounding in the last trade throws the answer
around by a factor of four. Picking "the nearest strike" is a coin flip
between 2.1% and 8.8%, and whichever you get then propagates into the
expected move, the cone, the Monte Carlo, and every probability on the page.

The extraction therefore:

1. **Filters before averaging** — drops contracts with no bid, with a
   bid-ask spread exceeding 50% of mid, and with IV outside [2%, 500%]. The
   spread test is *relative*: a one-tick market on a 1.5-cent option is a
   67% spread even though the absolute spread looks tiny.
2. **Pools calls and puts** — put-call parity says they should imply the
   same vol, so using both doubles the sample and surfaces disagreement.
3. **Takes a vega-weighted median** — the median ignores surviving
   outliers; the weighting concentrates the estimate on contracts whose IV
   is actually well determined.
4. **Reports quality, never guesses silently:**

   | Quality | Criterion |
   |---|---|
   | `good` | ≥6 samples, relative IQR under 15%, no relaxation needed |
   | `fair` | ≥3 samples, relative IQR under 40% |
   | `poor` | relative IQR above 40%, thin sample, or a widened window |
   | `none` | nothing usable |

   Dispersion is measured **relative to the median**, not in absolute vol
   points. The 0DTE chain above has an absolute IQR of only ~4 vol points,
   which looks tight next to a 30%-vol name — yet those strikes disagree by
   a factor of four. Relative dispersion catches that; absolute does not.

Callers use `first_usable_atm_iv`, which walks expiries front to back and
takes the nearest one that is actually trustworthy — the front expiry is
frequently 0DTE and unusable.

None of this makes a bad chain good. It makes a bad chain legible.

### Term structure and skew

- **Term structure** — ATM IV per expiry, classified `contango` (far > front,
  the normal resting state), `backwardation` (front > far), or `flat`. Only
  `good`/`fair` points contribute. Backwardation is the structurally
  interesting case: the market is paying up for near-dated optionality
  specifically, which is what an earnings date or macro print looks like
  from the options side.
- **25-delta risk reversal** — 25Δ put IV minus 25Δ call IV, in vol points.
  Strikes are chosen by *computed* delta, not by a fixed percentage from
  spot, so the measure stays comparable across tickers and tenors with very
  different volatilities. Positive means downside protection is bid
  relative to upside — the normal equity index state.

---

## 7. Forward distributions: closed form

`analytics/projection.py`. Geometric Brownian motion under the risk-neutral
measure:

```
dS/S = (r − q)·dt + σ·dW
ln(S_t/S_0) ~ Normal( (r − q − σ²/2)·t , σ²·t )
```

**Expected move** — `S·σ·√t`. The conventional shorthand: the standard
deviation of the arithmetic return to first order. It differs from the exact
lognormal standard deviation by O(σ²t), under 1% of the move for a 30-day
horizon at 30 vol. `lognormal_std` gives the exact figure.

**P(S_T > K)** — `N(d2)`. Note this is *not* the option's delta; delta is
`N(d1)`, which is larger. Confusing the two is a common error and overstates
the probability of finishing in the money.

**P(touch K before T)** — the exact first-passage probability for Brownian
motion with drift. With `X_t = ln(S_t/S_0) = μt + σW_t`, `μ = r − q − σ²/2`,
and `b = ln(K/S_0)`:

```
up barrier (b > 0):
  P(max X ≥ b) = N((μt − b)/(σ√t)) + e^(2μb/σ²)·N((−b − μt)/(σ√t))

down barrier (b < 0):
  P(min X ≤ b) = N((b − μt)/(σ√t)) + e^(2μb/σ²)·N((b + μt)/(σ√t))
```

> The previous implementation used the reflection shortcut `2 × P(S_T beyond
> K)`, while its docstring claimed to implement Hull's closed form. The
> shortcut is exact only at zero drift; it overstates the chance of touching
> a downside barrier in a positive-drift regime and understates the upside
> one. For a barrier 5% away over 3 months at 30 vol it is off by ~1.5
> points. `touch_prob_reflection` is retained so the size of the
> disagreement can be measured, and `tests/test_projection_timeutil.py`
> pins both the agreement at zero drift and the divergence under drift.

**Touch is always larger than terminal.** A level is far easier to tag
intraday than to close beyond, and roughly twice as likely for a driftless
barrier. Both are reported for every gamma wall, because they answer
different questions.

**The cone** is the marginal distribution of S at each time, not a
simultaneous confidence band for the path. 90% of paths finish inside the
P5/P95 envelope at any *given* time; far fewer stay inside it the whole way.

---

## 8. Forward distributions: Monte Carlo

`analytics/montecarlo.py`. Run `python main.py validate-mc` to reproduce
everything in this section.

### What is simulated

Three models, all under the risk-neutral drift `r − q`:

| Model | Process | When to use it |
|---|---|---|
| `gbm` | `dS/S = (r−q)dt + σdW` | Default. Terminal draws use the exact lognormal solution — no discretisation error at all. |
| `merton` | GBM + compound-Poisson jumps, `ln J ~ N(μ_j, σ_j)`, drift-compensated | Across a catalyst, where the move is a jump rather than accumulated diffusion. |
| `bootstrap` | IID resampling of the ticker's own historical log returns, standardised and rescaled | Keeps the ticker's realised skew and kurtosis; makes no distributional claim. |

The Merton drift is compensated by `−λk·dt` with `k = e^(μ_j + σ_j²/2) − 1`
so the discounted price stays a martingale — a jump model that skips this
step silently injects drift, and the check is in the test suite.

### Why simulate at all

Closed form already answers the GBM questions exactly and instantly. The
simulator earns its place on the questions closed form cannot reach:

- **Path-dependent quantities.** Touch probabilities come from actual
  simulated path maxima rather than a reflection formula, so they remain
  correct under models where no closed form exists.
- **Non-Gaussian shapes.** Fat tails, jump risk, and a specific ticker's
  empirical return distribution.
- **Distributional output**, not just moments — full quantiles, skew, and
  excess kurtosis of the terminal distribution.

### Variance reduction

- **Antithetic variates** — every draw `Z` is paired with `−Z`. Halves the
  independent draws needed for a given standard error on symmetric
  functionals and removes sampling error in the mean of `W` exactly.
- **Control variate** (pricing only) — `S_T` has known expectation
  `S_0·e^((r−q)T)`; regressing the payoff on that residual removes the
  error component explained by the terminal price. The regression
  coefficient is estimated from the same sample, so the bias is O(1/N) and
  negligible against the variance removed.
- **Brownian-bridge barrier correction** — see below.

### The Brownian bridge correction

Discretely monitored paths systematically **under**-count barrier hits,
because a path can cross the barrier and come back between two observations.
Conditional on the endpoints of a step, the probability of having crossed
has a closed form:

```
p_cross = exp( −2 · (b − x_i) · (b − x_{i+1}) / (σ²·Δt) )      both endpoints on the same side
P(no touch) = Π (1 − p_cross)
```

The size of the effect, on P(touch 105) with spot 100, σ=30%, t=0.25,
200,000 paths (analytic answer 0.7430):

| Steps | Naive discrete | With bridge | Naive error | Bridge error |
|---:|---:|---:|---:|---:|
| 16 | 0.6377 | 0.7428 | **−0.1053** | −0.0002 |
| 64 | 0.6871 | 0.7415 | **−0.0559** | −0.0015 |
| 256 | 0.7151 | 0.7430 | **−0.0278** | +0.0000 |

Uncorrected monitoring at 64 steps is off by 5.6 percentage points. That
bias is more than an order of magnitude larger than the sampling error the
path count was chosen to eliminate — so it is on by default, and correcting
it matters far more than adding paths.

### Why 5,000,000 paths

The standard error of a simulated probability is `√(p(1−p)/N)`, worst case
at `p = 0.5`:

| N | Worst-case SE |
|---:|---:|
| 100,000 | 0.00158 (16 bp) |
| 1,000,000 | 0.00050 (5 bp) |
| **5,000,000** | **0.00022 (2.2 bp)** |

The UI renders probabilities to 0.1% (10 bp). Five million paths puts the
sampling error roughly an order of magnitude below display resolution — the
numbers do not move when the seed changes.

**That is the entire justification.** It is a claim about *display
stability*, not about model accuracy: more paths reduce Monte Carlo error
and do nothing whatsoever about the far larger error from assuming GBM,
from a mis-estimated IV, or from the dealer positioning assumption in
section 2. Measured convergence for an ATM call, no control variate:

| Paths | Simulated | Abs error | Std err | Seconds |
|---:|---:|---:|---:|---:|
| 10,000 | 6.41653 | 0.04295 | 0.098224 | 0.00 |
| 100,000 | 6.43345 | 0.02603 | 0.031184 | 0.00 |
| 1,000,000 | 6.44927 | 0.01021 | 0.009886 | 0.02 |
| 5,000,000 | 6.45642 | 0.00306 | 0.004426 | 0.12 |

Error falls as `1/√N` as expected. **Interactive endpoints default to
200,000 paths** (SE ≈ 11 bp, well under a rendered pixel) because five
million buys nothing a user can see; the large counts are for validation.

### Memory

5,000,000 paths × 64 steps × 8 bytes would be 2.5 GB. The full path array is
never materialised. Paths are generated in chunks and folded into

- per-step **streaming log-price histograms** for quantiles — 4096 bins over
  ±8σ, so bin error is 0.004σ, far below Monte Carlo noise at any realistic
  path count; and
- plain **counters** for barrier hits.

Memory is O(chunk × steps), independent of the path count.

### Validation

`montecarlo.validate()` compares simulated values against closed-form ones
and reports the error in standard errors. A correct simulator gives |z|
below about 3 across the board. At 1,000,000 paths / 128 steps:

| Check | Analytic | Simulated | Std err | z |
|---|---:|---:|---:|---:|
| European C K=100 | 6.459483 | 6.450951 | 0.004458 | −1.91 |
| European C K=110 | 2.773003 | 2.765633 | 0.004300 | −1.71 |
| European P K=90 | 1.793866 | 1.788016 | 0.003268 | −1.79 |
| P(S_T > 95) | 0.630668 | 0.630799 | 0.000483 | +0.27 |
| P(S_T > 100) | 0.496676 | 0.496609 | 0.000500 | −0.13 |
| P(S_T > 105) | 0.369340 | 0.369208 | 0.000483 | −0.27 |
| E[S_T] (martingale) | 101.005017 | 101.002123 | 0.015216 | −0.19 |
| P(touch 105) | 0.742956 | 0.742976 | 0.000952 | +0.02 |
| P(touch 95) | 0.734469 | 0.734544 | 0.000963 | +0.08 |

The three option rows share one random sample and one control variate, so
their errors are correlated — that is a single draw landing ~1.8 SE low, not
three independent misses. Also checkable from the running app at
`/api/mc/validate`, and asserted in `tests/test_montecarlo.py` so a
regression fails the build.

---

## 9. Composite scores

### Squeeze pressure — `analytics/squeeze.py`

Measures how *fragile* the short side of a name is. It is a description of
positioning, not a prediction.

| Axis | Max points | Rationale |
|---|---:|---|
| Short % of float | 35 | How much stock must be bought back relative to what trades |
| Days to cover | 20 | How long that buying takes at normal volume |
| Borrow rate | 15 | Holders are already being squeezed on P&L |
| IV / HV ratio | 10 | Options pricing a move the tape has not delivered |
| Relative volume | 12 | Whether anything is actually happening |
| Float size | 10 | Small floats move further on equal flow |
| Accumulation | 8 | Directional tape confirmation |

Each axis maps through a **monotone piecewise-linear** curve anchored at the
same breakpoints the original step function used.

> The step version had cliffs: a name at 9.9% short float scored 5 and one
> at 10.1% scored 15, so a rounding difference in a vendor feed moved the
> composite ten points and reordered the leaderboard. Interpolating removes
> the discontinuity while giving identical values at every anchor, so
> previously logged signals stay comparable.

**The weights are judgement, not a fitted model.** No backtest has
calibrated them, and [§10](#10-does-the-squeeze-score-work) explains why
one cannot be run on free data. Treat the ranking as more meaningful than
the level.

### GEX profile classification — `analytics/gex_profile.py`

Labels the *shape* of the gamma landscape: **Wall** (concentrated at one
strike with light neighbours), **Pillar** (the same strike stays heavy
across several expiries), **Slide** (a smooth ramp across strikes, fitted by
least squares with an R² gate), **Pin** (heavy positive gamma very close to
spot). Plus a **violence flag** where net gamma near spot is negative.

Each describes a distribution and what hedging flow it implies. None is a
recommendation, and all invert with the section-2 convention.

---

## 10. Does the squeeze score work?

`platform/squeeze_backtest.py`, `python main.py backtest-squeeze`.

**Short answer: not as a ranking model, but it does concentrate tail
outcomes — and those are different claims.**

### Getting the data to ask the question at all

An earlier version of this document said the question was unanswerable on
free data, because four of the seven axes were only available as *current*
values and scoring a past date with them is lookahead bias. Two of the four
turn out to be free after all:

| Source | Provides | Cost |
|---|---|---|
| **FINRA** consolidated short interest | shares short, ADV, days-to-cover, twice monthly back to 2020 | free, no key |
| **SEC EDGAR** XBRL | shares outstanding, stamped with the **filed** date | free, contact email in User-Agent |

That lifts point-in-time coverage from 18% to **68%**, past the threshold at
which the harness will report anything.

**The publication lag is the whole game.** FINRA short interest settles on
the 15th and the last business day, but is not disseminated until roughly
eight business days later. Keying a backtest off the settlement date uses
information nobody had — the exact error the exercise exists to avoid,
disguised as a fix for it. Every lookup filters on *publication* date, and
EDGAR observations filter on `filed`, never on `end`.

Two axes remain genuinely unavailable: **borrow rate** (paid only) and
**true free float** (EDGAR gives shares outstanding, which is larger, so
short percentage is understated — conservative, but not the conventional
number).

### The result

45 high-short-interest names, 500 days, 15,639 point-in-time observations,
10-bar forward returns:

| Metric | Value |
|---|---:|
| Rank IC (all dates) | −0.0493 |
| t-statistic (offset-averaged) | −1.64 |
| Permutation-null percentile | 3.0 |
| Top-minus-bottom decile (mean) | **+13.5%** |

Those last two rows disagree, and the disagreement is the finding.

| Score decile | Mean return | Median return | P90 | Share >+20% |
|---:|---:|---:|---:|---:|
| 1 (lowest) | +0.95% | −0.49% | +17.2% | 7.2% |
| 5 | +0.79% | +0.11% | +16.3% | 6.6% |
| 8 | +1.85% | −0.88% | +20.3% | 10.2% |
| 9 | +2.95% | −1.94% | +20.0% | 10.0% |
| 10 (highest) | **+14.45%** | **−1.56%** | +23.9% | **13.0%** |

The top decile has a mean of +14.5% and a median of −1.6%. Both are
correct. Most high-scored names drift down; a few move violently up and
carry the average alone.

### Why rank IC was the wrong lens

Section 9 argued that win rate is a poor measure because it mostly reflects
market direction, and that rank IC is the right one. That reasoning holds
for a factor expected to shift the whole distribution. It does **not** hold
for a squeeze model, and this data is why.

A rank correlation is driven by typical ordering. When most of a bucket
drifts down while a minority explodes, it reports *negative* information —
even though an equal-weighted holder of that bucket made money. The
statistic is not wrong; it is answering a question nobody asked.

So the harness now reports tail statistics alongside the rank statistics:
the share of each bucket exceeding a large-move threshold, the P90 outcome,
and a `tail_signature` flag when a bucket's mean and median disagree in
sign. On this run the large-move rate rises monotonically from 7.2% in the
bottom decile to 13.0% in the top — a **1.8x** concentration, with P90
rising from +17.2% to +23.9%.

### What can honestly be claimed

- **As a ranking model: no evidence it works.** The rank IC is
  indistinguishable from noise, and mildly negative.
- **As a tail-exposure filter: suggestive, not established.** The
  large-move gradient is monotone across all ten deciles, which is more
  structure than noise usually produces. But the t-statistic does not clear
  a conventional bar, 68% coverage is not 100%, the universe is
  survivorship-biased, and the mean is dominated by a handful of
  observations — exactly the regime where a result is least stable.
- **Nothing here justifies trading it.** A concentration of tail outcomes
  is not an edge until you have accounted for the cost of holding the
  losers that produce it.

### Is the harness capable of finding anything?

A test that finds nothing is worthless unless it can be shown to find
something. `tests/test_squeeze_backtest.py` plants synthetic relationships
of known strength and asserts they are recovered: an IC of 0.30 is detected
at t > 3 above the 97.5th null percentile, 0.15 is detected, an *inverted*
relationship is detected and flagged negative, and ten independent noise
samples produce null percentiles that do not cluster at the extremes. It
also asserts that a planted IC of 0.40 still returns **inconclusive** at 18%
coverage — the gate holds even when a signal is plainly present.

### Overlapping windows

A 10-day forward return sampled daily overlaps its neighbour by 9 days, so
400 dates are nowhere near 400 independent observations. Both the
t-statistic and the null run on non-overlapping subsamples (every h-th
date), averaged across all h starting offsets so nothing depends on where
the sample begins.

> An earlier version got this wrong instructively. It corrected the
> t-statistic but computed the null on all 400 dates, and the two
> disagreed. Fixing the null to use one subsample moved it to the 96.5th
> percentile — *more* significant-looking. Averaging over all ten offsets
> collapsed it to the 63rd. The apparent signal was entirely an artifact of
> which date the subsample happened to start on.

---

## 11. Known limitations

Ordered by how much they affect the output — the modelling assumptions at
the top dominate everything below them.

1. **The dealer positioning assumption (§2) is the largest source of error
   by far.** It is a guess about who holds what, applied uniformly to every
   ticker. On a name in a call-buying mania it has the sign backwards, and
   no amount of numerical care downstream recovers from that.
2. **IV is taken as given, per contract.** No arbitrage-free surface is
   fitted, so a chain with crossed or stale quotes produces a locally
   inconsistent surface. Section 6 filters the worst of it and flags what
   survives; it does not repair it.
3. **European exercise.** Deep ITM put deltas are biased (§3).
4. **Static, single-snapshot IV in the flip curve.** `gamma_flip_level`
   holds each contract's IV fixed while moving spot. Real skew means IV
   would change as spot moves, so the curve is a same-day approximation,
   not a path.
5. **Open interest is T+1.** Every provider publishes OI after the fact, so
   today's exposures are computed from yesterday's resting book plus
   today's volume where the volume basis is used. Intraday positioning
   changes are invisible until the next publication.
6. **No aggressor side.** Without trade-level bid/ask flags, "customer
   bought" versus "customer sold" cannot be distinguished — which is what
   makes §2 an assumption rather than a measurement.
7. **The squeeze weights are uncalibrated** (§9), and cannot be validated on free data ([§10](#10-does-the-squeeze-score-work)).
8. **Free-tier data is delayed ~15 minutes.** Since switching the default
   free source to Cboe (exchange-computed IV and Greeks, whole chain in one
   keyless request) the *quality* problem is much reduced, but the delay
   remains. See [`data_sources.md`](data_sources.md).
9. **`r` defaults to 4% and `q` to 0** unless a caller passes otherwise.
   Both are second-order for short horizons but not for LEAPS or for
   high-yield names.

---

## 12. Not yet implemented

Listed explicitly because earlier revisions of this document described
these as though they existed.

- **The last 32% of backtest coverage.** Borrow rate is paid-only and true
  free float is not published free, so those two axes stay untested
  ([§10](#10-does-the-squeeze-score-work)).
- **A proper evaluation of the tail result.** The large-move concentration
  in §10 deserves a purpose-built test — a payoff-weighted statistic rather
  than a rank one, out-of-sample replication, and a delisting-inclusive
  universe to remove survivorship bias.
- **Intraday node-strength trajectories.** `analytics/atlas.py` stores
  intraday OI/GEX snapshots and computes growth metrics; the fuller
  node typology (air pockets, rug pulls, slingshots) sketched in earlier
  drafts is not built.
- **Rehedge-intensity metric** (`|∂Γ/∂S| × realised vol`).
- **FTD persistence and borrow-rate acceleration** as squeeze inputs —
  both need a paid data feed.
- **A fitted, arbitrage-free IV surface** (SVI or similar), which would
  replace per-contract IV with a smooth interpolation and fix limitation 2.
- **Real-time streaming and alerting.**

---

## 13. References

- SqueezeMetrics, *The Implied Order Book* / *Dealer's Hedge* (2017) — the
  origin of the naive-dealer GEX convention used in §2 and §4.
- SpotGamma, *Gamma Trading* (2020) — zero-gamma level and its
  interpretation.
- Hull, *Options, Futures and Other Derivatives*, 9th ed. — Black–Scholes
  Greeks (ch. 19), barrier/first-passage probabilities (ch. 26). The
  reference case in `tests/test_greeks.py` is Hull's.
- Merton, R. (1976), "Option pricing when underlying stock returns are
  discontinuous", *Journal of Financial Economics* 3 — the jump model in §8.
- Glasserman, P., *Monte Carlo Methods in Financial Engineering* (2003) —
  antithetic and control variates (ch. 4), Brownian-bridge barrier
  correction (ch. 6.4).
- Brenner, M. and Subrahmanyam, M. (1988), "A simple formula to compute the
  implied standard deviation" — the IV solver seed in §3.
