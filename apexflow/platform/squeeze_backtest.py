"""Backtest the squeeze scorer: does a higher score predict a better outcome?

This is the question that decides whether `analytics/squeeze.py` is a model
or a decoration, and it is deliberately separate from `backtest.py` — that
one replays *scanner* boolean signals against price bars, which is a
different and much easier question.

What makes this hard, stated up front
-------------------------------------
The scorer takes seven inputs. Backtesting it honestly means reconstructing
all seven **as they were known on the signal date**. Most of them cannot be:

============  ===================================  ==========================
Axis          Point-in-time on free data?          Max points
============  ===================================  ==========================
rvol          yes, from volume history                     12
accumulation  yes, from OBV/price history                   8
iv_hv_ratio   no real IV history; proxy only               10
short % float **no** - only the current value                35
days to cover **no** - only the current value                20
borrow rate   **no** - paid, current only                   15
float shares  **no** - only the current value               10
============  ===================================  ==========================

The four unavailable axes are **80 of the 100 points**, and they include
every axis that actually describes short-side fragility — the entire thesis
of the model. yfinance reports one current `shortPercentOfFloat`; using it
to score a signal from eight months ago is lookahead bias of the purest
kind, and it would make the backtest look *better* than reality, because
today's short interest partly reflects what already happened.

So this module does not silently fill those in. `PriceDerivedSource`
populates only what a date's own history supports, reports its coverage as
a fraction of the total score, and the report refuses to present a headline
verdict when coverage is too low to mean anything. A backtest of 20% of a
model is not a backtest of the model.

The harness itself is complete and takes a pluggable
:class:`PointInTimeSource`, so pointing it at a historical short-interest
archive (FINRA bi-monthly files, Fintel, Ortex) makes it a real validation
with no changes here.

How it evaluates
----------------
Not by win rate. Win rate on a long-only sample mostly measures whether the
market went up during the window, which says nothing about the *score*. The
question is whether the score **ranks** outcomes, so the harness computes:

* **Rank IC** — per-date Spearman correlation between score and forward
  return, averaged across dates. This is the standard cross-sectional
  measure: it asks "on a given day, did the higher-scored names do better
  than the lower-scored ones?" and is immune to overall market direction.
* **A permutation null** — the same statistic with scores shuffled within
  each date, many times over. Reported as a percentile, so an IC of 0.04
  can be seen for what it is against the spread of ICs pure noise produces
  at this sample size.
* **Decile means** — average forward return by score bucket, to see whether
  any relationship is monotone or driven by one extreme bucket.

Overlapping windows, and why both tests use the same subsample
---------------------------------------------------------------
An h-day forward return sampled every day overlaps its neighbour by h-1
days. Consecutive ICs are therefore strongly autocorrelated, and treating
400 daily observations as 400 independent ones overstates significance by
roughly sqrt(h).

The first version of this module corrected the t-statistic for that but
computed the permutation null on all 400 dates — so the two disagreed, with
the t-stat saying "nothing here" and the null percentile saying "almost
significant". The null was wrong: shuffling scores destroys the
cross-sectional relationship but leaves the date count untouched, so it was
still implicitly claiming 400 independent draws.

Both statistics are now computed on **non-overlapping subsamples**: every
h-th date, so within any one subsample no two forward windows share a day.

Picking a single subsample would mean picking an arbitrary starting offset
and discarding the other h-1 of the data, so the t-statistic and the null
percentile are computed for *every* offset 0..h-1 and averaged. Each
individual statistic is therefore built only from independent observations,
while the average still sees every date. The spread across offsets is
reported as ``t_stat_spread`` — a wide spread is itself a warning that the
result depends on where you happened to start.

The all-dates IC remains the better point estimate and is reported as
``rank_ic_all_dates``; it is simply not the basis for the significance
claim.

Every one of these choices exists to make it *harder* to conclude the model
works.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from apexflow.analytics import indicators as ind
from apexflow.analytics.squeeze import SqueezeInputs, squeeze_score, COMPONENT_MAX

log = logging.getLogger(__name__)

__all__ = [
    "PointInTimeSource", "PriceDerivedSource", "SqueezeBacktestReport",
    "SqueezeBacktester", "TOTAL_POINTS", "MIN_MEANINGFUL_COVERAGE",
]

TOTAL_POINTS = sum(COMPONENT_MAX.values())

#: Below this fraction of the score, a result is reported as inconclusive
#: rather than as a verdict on the model. 0.5 is a judgement call, but any
#: threshold that admits the price-only source (0.20) would be dishonest.
MIN_MEANINGFUL_COVERAGE = 0.50

#: Minimum names per date for a cross-sectional rank correlation to carry
#: any information. Spearman on 5 points is noise with a decimal place.
MIN_BREADTH = 15


# ---------------------------------------------------------------------------
# Point-in-time input sources
# ---------------------------------------------------------------------------
class PointInTimeSource(Protocol):
    """Supplies squeeze inputs as they were knowable on a given date.

    Implementations must use only information available at ``as_of``.
    ``history`` is pre-sliced to end on that date, so the price-derived
    axes cannot leak; anything else an implementation reaches for is its own
    responsibility.
    """

    #: Axis names (keys of COMPONENT_MAX) this source can populate.
    covers: frozenset[str]

    def inputs_at(self, symbol: str, as_of: pd.Timestamp,
                  history: pd.DataFrame) -> SqueezeInputs:
        ...


@dataclass
class PriceDerivedSource:
    """Reconstructs only the axes that daily OHLCV honestly supports.

    Leaves every unavailable axis at its zero-pressure value rather than
    guessing, so the composite is a partial score and
    :attr:`coverage` says how partial.

    ``use_realised_vol_proxy`` substitutes a realised-vol term-structure
    ratio (short-window HV over long-window HV) for the IV/HV axis. It is
    **off by default and should stay off for anything load-bearing**: the
    real axis measures options pricing a move the tape has not delivered,
    and a ratio of two realised windows cannot see the options market at
    all. It measures vol expansion that has already happened, which is a
    different thing wearing the same slot.
    """

    rvol_window: int = 30
    accumulation_window: int = 20
    use_realised_vol_proxy: bool = False

    @property
    def covers(self) -> frozenset[str]:
        c = {"rvol", "accumulation"}
        if self.use_realised_vol_proxy:
            c.add("iv_hv")
        return frozenset(c)

    @property
    def coverage(self) -> float:
        """Fraction of the total score this source can reach."""
        return sum(COMPONENT_MAX[a] for a in self.covers) / TOTAL_POINTS

    @property
    def missing(self) -> list[str]:
        return sorted(set(COMPONENT_MAX) - self.covers)

    def inputs_at(self, symbol: str, as_of: pd.Timestamp,
                  history: pd.DataFrame) -> SqueezeInputs:
        close = history["Close"]
        volume = history["Volume"]

        rvol = 0.0
        if len(volume) > self.rvol_window:
            series = ind.rvol(volume, self.rvol_window)
            val = series.iloc[-1]
            rvol = float(val) if pd.notna(val) else 0.0

        accumulation = _obv_trend(close, volume, self.accumulation_window)

        iv_hv = 0.0
        if self.use_realised_vol_proxy and len(close) > 65:
            short_hv = ind.historical_volatility(close, 10)
            long_hv = ind.historical_volatility(close, 60)
            if long_hv > 0:
                iv_hv = float(short_hv / long_hv)

        return SqueezeInputs(
            short_pct_float=0.0,      # not knowable point-in-time on free data
            days_to_cover=0.0,        # ditto
            borrow_rate=0.0,          # ditto
            float_shares=0.0,         # ditto
            iv_hv_ratio=iv_hv,
            rvol=rvol,
            price=float(close.iloc[-1]),
            accumulation_score=accumulation,
        )


def _obv_trend(close: pd.Series, volume: pd.Series, window: int) -> float:
    """On-balance-volume slope over `window`, squashed to -1..+1.

    Normalised by the OBV series' own standard deviation so the result is
    comparable across tickers of very different volume, then passed through
    tanh so extremes saturate instead of dominating.
    """
    if len(close) < window + 2:
        return 0.0
    direction = np.sign(close.diff().fillna(0.0).to_numpy())
    obv = np.cumsum(direction * volume.to_numpy())
    tail = obv[-window:]
    if tail.size < 3:
        return 0.0
    scale = float(np.std(obv[-(window * 3):])) or 1.0
    x = np.arange(tail.size, dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    return float(np.tanh(slope * tail.size / scale))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class SqueezeBacktestReport:
    hold_days: int
    n_observations: int = 0
    n_dates: int = 0
    mean_breadth: float = 0.0
    coverage: float = 0.0
    covered_axes: list[str] = field(default_factory=list)
    missing_axes: list[str] = field(default_factory=list)

    rank_ic: float = 0.0                 # mean across non-overlapping subsamples
    rank_ic_all_dates: float = 0.0       # every date — better point estimate
    rank_ic_std: float = 0.0
    t_stat: float = 0.0                  # averaged over all stride offsets
    t_stat_spread: float = 0.0           # min..max across offsets, as a range
    effective_n: float = 0.0
    n_independent_dates: int = 0
    null_percentile: float = 50.0
    null_ic_std: float = 0.0

    decile_returns: list[dict] = field(default_factory=list)
    top_minus_bottom: float = 0.0

    conclusive: bool = False
    verdict: str = ""
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hold_days": self.hold_days,
            "n_observations": self.n_observations,
            "n_dates": self.n_dates,
            "mean_breadth": self.mean_breadth,
            "coverage": self.coverage,
            "covered_axes": self.covered_axes,
            "missing_axes": self.missing_axes,
            "rank_ic": self.rank_ic,
            "rank_ic_all_dates": self.rank_ic_all_dates,
            "rank_ic_std": self.rank_ic_std,
            "t_stat": self.t_stat,
            "t_stat_spread": self.t_stat_spread,
            "effective_n": self.effective_n,
            "n_independent_dates": self.n_independent_dates,
            "null_percentile": self.null_percentile,
            "null_ic_std": self.null_ic_std,
            "decile_returns": self.decile_returns,
            "top_minus_bottom": self.top_minus_bottom,
            "conclusive": self.conclusive,
            "verdict": self.verdict,
            "caveats": self.caveats,
        }

    @property
    def summary(self) -> str:
        if not self.conclusive:
            return f"INCONCLUSIVE — {self.verdict}"
        return (f"rank IC {self.rank_ic:+.4f} (t={self.t_stat:+.2f}, "
                f"null pct {self.null_percentile:.1f}) over "
                f"{self.n_independent_dates} independent dates — {self.verdict}")


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class SqueezeBacktester:
    """Cross-sectional evaluation of the squeeze score against forward returns."""

    def __init__(self, provider=None, source: PointInTimeSource | None = None):
        if provider is None:
            from apexflow.providers import get_provider
            provider = get_provider()
        self.provider = provider
        self.source = source or PriceDerivedSource()

    # -- data assembly ----------------------------------------------------
    def build_panel(self, universe: Sequence[str], lookback_days: int = 365,
                    hold_days: int = 10, min_history: int = 90,
                    step_days: int = 1) -> pd.DataFrame:
        """Score every (symbol, date) and attach its forward return.

        Returns a long frame with columns ``date, symbol, score, fwd_return``.
        The forward return is measured from the close on the signal date to
        the close ``hold_days`` bars later — both strictly after the data
        used to compute the score, which is what keeps this out of lookahead
        territory.
        """
        rows: list[dict] = []
        for sym in universe:
            try:
                df = self.provider.history(sym, period="2y", interval="1d")
            except Exception as e:
                log.warning("history failed for %s: %s", sym, e)
                continue
            if df is None or len(df) < min_history + hold_days + 5:
                continue

            df = df.dropna(subset=["Close", "Volume"])
            close = df["Close"].to_numpy(dtype=float)
            n = len(df)
            start = max(min_history, n - lookback_days - hold_days)

            for i in range(start, n - hold_days, max(1, step_days)):
                # History is sliced to end at i — nothing after the signal
                # date is visible to the scorer.
                window = df.iloc[: i + 1]
                try:
                    inputs = self.source.inputs_at(sym, df.index[i], window)
                    score, _ = squeeze_score(inputs)
                except Exception as e:
                    log.debug("scoring failed %s@%s: %s", sym, df.index[i], e)
                    continue

                entry = close[i]
                exit_ = close[i + hold_days]
                if entry <= 0 or not np.isfinite(entry) or not np.isfinite(exit_):
                    continue
                rows.append({
                    "date": df.index[i],
                    "symbol": sym,
                    "score": float(score),
                    "fwd_return": float((exit_ / entry - 1) * 100.0),
                })

        return pd.DataFrame(rows)

    # -- evaluation -------------------------------------------------------
    def evaluate(self, panel: pd.DataFrame, hold_days: int = 10,
                 n_permutations: int = 500, seed: int = 7,
                 n_buckets: int = 10) -> SqueezeBacktestReport:
        """Turn a scored panel into a report, with the null comparison."""
        cov = getattr(self.source, "coverage", 1.0)
        covered = sorted(getattr(self.source, "covers", frozenset(COMPONENT_MAX)))
        missing = getattr(self.source, "missing", [])

        rep = SqueezeBacktestReport(
            hold_days=hold_days, coverage=cov,
            covered_axes=covered, missing_axes=list(missing),
        )

        if panel is None or panel.empty:
            rep.verdict = "no observations — check the universe and data source"
            return rep

        rep.n_observations = len(panel)
        by_date = panel.groupby("date")
        rep.n_dates = by_date.ngroups
        rep.mean_breadth = float(by_date.size().mean())

        # --- per-date rank IC -------------------------------------------
        # Keep dates ordered so the non-overlapping stride below is a real
        # stride through time, not through an arbitrary grouping order.
        dated_groups = [(d, g) for d, g in by_date]
        dated_groups.sort(key=lambda kv: kv[0])

        all_ics: list[float] = []
        usable: list[tuple[np.ndarray, np.ndarray]] = []
        for _, grp in dated_groups:
            scores = grp["score"].to_numpy()
            rets = grp["fwd_return"].to_numpy()
            ic = _spearman(scores, rets)
            if ic is not None:
                all_ics.append(ic)
                usable.append((scores, rets))

        if not all_ics:
            rep.verdict = (f"no date had enough cross-sectional breadth "
                           f"(need >= {MIN_BREADTH} names with varying scores)")
            rep.caveats.append(
                f"mean breadth was {rep.mean_breadth:.1f} names per date")
            return rep

        rep.rank_ic_all_dates = float(np.mean(all_ics))

        # Independent subsamples: every h-th date, so within a subsample no
        # two forward windows share a day. Every starting offset is used and
        # the statistics averaged, so nothing hinges on where we start and
        # no date is thrown away.
        stride = max(hold_days, 1)
        rng = np.random.default_rng(seed)
        perms_per_offset = max(n_permutations // stride, 50)

        ic_means, ic_stds, t_stats, pctiles, null_stds = [], [], [], [], []
        sizes = []

        for offset in range(stride):
            idx = list(range(offset, len(all_ics), stride))
            if len(idx) < 3:
                continue
            sub = np.array([all_ics[i] for i in idx], dtype=float)
            groups = [usable[i] for i in idx]

            m = float(sub.mean())
            sd = float(sub.std(ddof=1))
            ic_means.append(m)
            ic_stds.append(sd)
            sizes.append(sub.size)
            t_stats.append(m / sd * np.sqrt(sub.size) if sd > 0 else 0.0)

            null = np.empty(perms_per_offset, dtype=float)
            for k in range(perms_per_offset):
                vals = [ic for ic in
                        (_spearman(rng.permutation(sc), rt) for sc, rt in groups)
                        if ic is not None]
                null[k] = float(np.mean(vals)) if vals else 0.0
            null_stds.append(float(null.std(ddof=1)))
            pctiles.append(float((null < m).mean() * 100.0))

        if not ic_means:
            rep.verdict = "not enough dates to form an independent subsample"
            return rep

        rep.n_independent_dates = int(np.mean(sizes))
        rep.effective_n = float(np.mean(sizes))
        rep.rank_ic = float(np.mean(ic_means))
        rep.rank_ic_std = float(np.mean(ic_stds))
        rep.t_stat = float(np.mean(t_stats))
        rep.t_stat_spread = float(max(t_stats) - min(t_stats)) if len(t_stats) > 1 else 0.0
        rep.null_ic_std = float(np.mean(null_stds))
        rep.null_percentile = float(np.mean(pctiles))

        # --- decile table -----------------------------------------------
        rep.decile_returns, rep.top_minus_bottom = _bucket_table(panel, n_buckets)

        # --- verdict -----------------------------------------------------
        rep.conclusive, rep.verdict, rep.caveats = _judge(rep)
        return rep

    def run(self, universe: Sequence[str], lookback_days: int = 365,
            hold_days: int = 10, n_permutations: int = 500,
            step_days: int = 1) -> SqueezeBacktestReport:
        panel = self.build_panel(universe, lookback_days=lookback_days,
                                 hold_days=hold_days, step_days=step_days)
        return self.evaluate(panel, hold_days=hold_days,
                             n_permutations=n_permutations)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation, or None when it would be meaningless.

    Returns None for thin cross-sections or when either side is constant —
    a degenerate correlation is not a zero correlation, and averaging the
    two together would quietly bias the result toward zero.
    """
    if a.size < MIN_BREADTH or b.size != a.size:
        return None
    ra = _rank(a)
    rb = _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the standard Spearman tie handling)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    # Average the ranks within each group of equal values.
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def _bucket_table(panel: pd.DataFrame, n_buckets: int) -> tuple[list[dict], float]:
    """Mean forward return by score bucket, ranked within each date."""
    if panel.empty:
        return [], 0.0
    df = panel.copy()
    # Bucket within each date, so the table is a cross-sectional statement
    # rather than a comparison of different market days.
    def _assign(scores: pd.Series) -> pd.Series:
        if len(scores) < n_buckets:
            return pd.Series([np.nan] * len(scores), index=scores.index)
        try:
            return pd.qcut(scores.rank(method="first"), n_buckets,
                           labels=False, duplicates="drop")
        except ValueError:
            return pd.Series([np.nan] * len(scores), index=scores.index)

    # Transform on the score column only — passing whole frames through
    # groupby.apply is deprecated and pulls the grouping column in.
    df["bucket"] = df.groupby("date", group_keys=False)["score"].apply(_assign)
    df = df.dropna(subset=["bucket"])
    if df.empty:
        return [], 0.0

    rows = []
    for b, grp in df.groupby("bucket"):
        rows.append({
            "bucket": int(b) + 1,
            "n": int(len(grp)),
            "mean_score": float(grp["score"].mean()),
            "mean_return": float(grp["fwd_return"].mean()),
            "median_return": float(grp["fwd_return"].median()),
        })
    rows.sort(key=lambda r: r["bucket"])
    tmb = (rows[-1]["mean_return"] - rows[0]["mean_return"]) if len(rows) >= 2 else 0.0
    return rows, float(tmb)


def _judge(rep: SqueezeBacktestReport) -> tuple[bool, str, list[str]]:
    """Decide whether the run supports any claim, and enumerate the caveats.

    Deliberately conservative. The default answer is "this does not tell you
    the model works", and each gate has to be cleared to move off it.
    """
    caveats: list[str] = []

    if rep.coverage < 1.0:
        pct = rep.coverage * 100
        caveats.append(
            f"only {pct:.0f}% of the score was reconstructable point-in-time; "
            f"missing axes: {', '.join(rep.missing_axes)}")
    if rep.mean_breadth < MIN_BREADTH:
        caveats.append(
            f"mean cross-sectional breadth {rep.mean_breadth:.1f} is below the "
            f"{MIN_BREADTH}-name minimum for a rank correlation to mean anything")
    caveats.append(
        f"significance measured on non-overlapping subsamples of ~"
        f"{rep.n_independent_dates} dates (every {rep.hold_days}th of "
        f"{rep.n_dates}), averaged over all {rep.hold_days} starting offsets")
    if rep.t_stat_spread > 1.5:
        caveats.append(
            f"t-statistic ranges {rep.t_stat_spread:.1f} across starting "
            f"offsets — the result is sensitive to where the sample begins")
    caveats.append(
        "universe is currently-listed names only — delisted and acquired "
        "tickers are absent, which biases results upward (survivorship)")

    if rep.coverage < MIN_MEANINGFUL_COVERAGE:
        return False, (
            f"coverage {rep.coverage:.0%} is below the {MIN_MEANINGFUL_COVERAGE:.0%} "
            f"threshold — this measures a fragment of the score, not the model. "
            f"The axes that carry the squeeze thesis "
            f"({', '.join(rep.missing_axes)}) need a point-in-time archive."
        ), caveats

    if rep.mean_breadth < MIN_BREADTH:
        return False, "cross-section too narrow to rank", caveats

    # Two-sided read against the permutation null.
    extreme = rep.null_percentile > 97.5 or rep.null_percentile < 2.5
    if not extreme:
        return True, ("no detectable rank information — the observed IC sits "
                      f"at the {rep.null_percentile:.1f}th percentile of the "
                      "shuffled null, i.e. within noise"), caveats
    direction = "positive" if rep.rank_ic > 0 else "negative"
    return True, (f"{direction} rank information detected "
                  f"({rep.null_percentile:.1f}th percentile of null); "
                  "treat as provisional until replicated out of sample"), caveats
