"""Monte Carlo path simulation for forward price distributions.

Why this module exists
----------------------
``projection.py`` gives closed-form answers under geometric Brownian motion.
That is exact, fast, and *wrong in the ways GBM is wrong*: it cannot express
fat tails, jump risk around a catalyst, or the empirical return distribution
of a specific ticker. This module simulates paths so those questions can be
asked, and so path-dependent quantities (barrier / "does price touch this
gamma wall" probabilities) come from actual path maxima rather than the
reflection-principle approximation.

What is simulated
-----------------
Three models, all under the risk-neutral drift ``r - q``:

``gbm``
    dS/S = (r-q) dt + sigma dW.  Terminal draws use the *exact* lognormal
    solution (no discretisation error); stepped paths use the exact
    log-Euler recursion, which is also exact at the observation points.

``merton``
    GBM plus a compound-Poisson jump component (Merton 1976):
    dS/S = (r-q-lambda*k) dt + sigma dW + (J-1) dN,  ln J ~ N(mu_j, sigma_j).
    The drift is compensated so the discounted price stays a martingale.
    This is the model to use across an earnings print, where the move is a
    jump rather than accumulated diffusion.

``bootstrap``
    IID resampling of a supplied array of historical log returns, rescaled
    to the requested horizon and re-centred to the risk-neutral drift.
    Keeps the ticker's own skew/kurtosis; makes no distributional claim.

Variance reduction
------------------
* **Antithetic variates** - every normal draw Z is paired with -Z. Halves
  the number of independent draws needed for a given standard error on
  symmetric functionals, and removes sampling error in the mean of W.
* **Control variate** (option pricing only) - S_T has a known expectation
  ``S0 * exp((r-q)T)``; regressing the payoff on that residual removes the
  component of Monte Carlo error explained by the terminal price.
* **Brownian-bridge barrier correction** - discretely monitored paths
  systematically *under*-count barrier hits, because the path can cross and
  come back between two observations. Conditional on the endpoints of a
  step, the probability of having crossed has a closed form (the Brownian
  bridge crossing probability) which is applied per step. Without it, touch
  probabilities are biased low by several points at coarse step counts.

Memory
------
5,000,000 paths x 64 steps x 8 bytes would be 2.5 GB. Nothing here ever
materialises the full path array: paths are generated in chunks and folded
into (a) per-step streaming log-price histograms for quantiles and (b) plain
counters for barrier hits. Memory is O(chunk_paths * n_steps), independent of
``n_paths``.

Iteration count
---------------
The standard error of a simulated probability p is ``sqrt(p(1-p)/N)``::

    N =   100,000 -> SE <= 0.00158   (16 bp)
    N = 1,000,000 -> SE <= 0.00050   (5 bp)
    N = 5,000,000 -> SE <= 0.00022   (2.2 bp)

The UI renders probabilities to 0.1% (10 bp), so 5,000,000 paths puts the
sampling error roughly an order of magnitude below display resolution - the
numbers do not move when the seed changes. That is the entire justification;
it is not a claim that more paths make the *model* more correct. Interactive
endpoints default to 200,000 paths (SE ~ 11 bp) because that is already
below the width of a rendered pixel; ``validate()`` and the CLI use the
large counts.

Validation
----------
``validate()`` prices European options, terminal-distribution digitals, the
martingale expectation, and a one-touch barrier by simulation, comparing
each against its closed-form value and reporting the error in units of
standard error (a |z| under ~3 is consistent with correct code).
``convergence_table()`` shows the standard error shrinking as 1/sqrt(N).
Run both with ``python main.py validate-mc``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "MCConfig", "MCResult", "simulate_terminal", "simulate_cone",
    "touch_probabilities", "price_european", "bs_price", "analytic_touch",
    "validate", "convergence_table", "DEFAULT_PATHS", "VALIDATION_PATHS",
]

# Interactive default - SE ~ 11 bp on a probability, well under display resolution.
DEFAULT_PATHS = 200_000
# The count used for validation runs and the headline figure.
VALIDATION_PATHS = 5_000_000


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MCConfig:
    """Simulation settings.

    n_paths      total simulated paths (rounded up to an even number when
                 antithetic sampling is on, since paths come in +/- pairs)
    n_steps      observation points between now and the horizon. Only affects
                 path-dependent output (cone shape, barrier monitoring);
                 terminal-only quantities are exact at n_steps = 1.
    chunk_paths  paths generated per batch. Caps peak memory at roughly
                 chunk_paths * n_steps * 8 bytes.
    """
    n_paths: int = DEFAULT_PATHS
    n_steps: int = 64
    seed: int | None = 7
    antithetic: bool = True
    chunk_paths: int = 250_000
    model: str = "gbm"           # gbm | merton | bootstrap
    # Merton jump parameters (ignored unless model == "merton")
    jump_intensity: float = 0.0  # lambda, expected jumps per year
    jump_mean: float = 0.0       # mu_j, mean of ln(J)
    jump_vol: float = 0.0        # sigma_j, stdev of ln(J)
    # Bootstrap sample (ignored unless model == "bootstrap")
    returns: np.ndarray | None = None   # historical *log* returns, any frequency

    def __post_init__(self) -> None:
        self.n_paths = max(int(self.n_paths), 2)
        if self.antithetic and self.n_paths % 2:
            self.n_paths += 1
        self.n_steps = max(int(self.n_steps), 1)
        self.chunk_paths = max(int(self.chunk_paths), 2)
        if self.antithetic and self.chunk_paths % 2:
            self.chunk_paths += 1
        if self.model not in ("gbm", "merton", "bootstrap"):
            raise ValueError(f"unknown model {self.model!r}")
        if self.model == "bootstrap":
            r = np.asarray(self.returns, dtype=float).ravel() if self.returns is not None else np.array([])
            r = r[np.isfinite(r)]
            if r.size < 20:
                raise ValueError("bootstrap model needs >=20 finite historical log returns")
            self.returns = r


@dataclass
class MCResult:
    """Simulation output. All prices are in the underlying's units."""
    spot: float
    horizon_years: float
    iv: float
    model: str
    n_paths: int
    n_steps: int
    elapsed_s: float
    mean: float = 0.0
    std: float = 0.0
    quantiles: dict[str, float] = field(default_factory=dict)
    cone: list[dict] = field(default_factory=list)
    prob_up: float = 0.0
    expected_move: float = 0.0          # 1-sigma of the simulated terminal distribution
    expected_move_pct: float = 0.0
    standard_error: float = 0.0         # SE of the mean terminal price
    skew: float = 0.0
    kurtosis: float = 0.0               # excess kurtosis

    def to_dict(self) -> dict:
        return {
            "spot": self.spot, "horizon_years": self.horizon_years, "iv": self.iv,
            "model": self.model, "n_paths": self.n_paths, "n_steps": self.n_steps,
            "elapsed_s": round(self.elapsed_s, 4),
            "mean": self.mean, "std": self.std, "quantiles": self.quantiles,
            "cone": self.cone, "prob_up": self.prob_up,
            "expected_move": self.expected_move,
            "expected_move_pct": self.expected_move_pct,
            "standard_error": self.standard_error,
            "skew": self.skew, "kurtosis": self.kurtosis,
        }


# ---------------------------------------------------------------------------
# Streaming quantile accumulator
# ---------------------------------------------------------------------------
class _LogHistogram:
    """Fixed-grid histogram over log-returns, for O(1)-memory quantiles.

    Bins span +/- ``span_sigma`` standard deviations of the terminal log
    return. Quantiles are read off the cumulative counts with linear
    interpolation inside the containing bin, so the discretisation error is
    at most one bin width - with 4096 bins over +/-8 sigma that is 0.004
    sigma, far below Monte Carlo noise at any realistic path count.
    """

    __slots__ = ("lo", "hi", "n_bins", "width", "counts", "total", "below", "above",
                 "_sum", "_sum2", "_sum3", "_sum4")

    def __init__(self, sigma_total: float, n_bins: int = 4096, span_sigma: float = 8.0):
        s = max(float(sigma_total), 1e-9)
        self.lo = -span_sigma * s
        self.hi = +span_sigma * s
        self.n_bins = int(n_bins)
        self.width = (self.hi - self.lo) / self.n_bins
        self.counts = np.zeros(self.n_bins, dtype=np.int64)
        self.total = 0
        self.below = 0          # samples clipped off the low edge
        self.above = 0          # samples clipped off the high edge
        self._sum = 0.0
        self._sum2 = 0.0
        self._sum3 = 0.0
        self._sum4 = 0.0

    def update(self, log_returns: np.ndarray) -> None:
        x = np.ascontiguousarray(log_returns).ravel()
        self.total += x.size
        self._sum += float(x.sum())
        self._sum2 += float(np.dot(x, x))
        self._sum3 += float((x ** 3).sum())
        self._sum4 += float((x ** 4).sum())
        idx = np.floor((x - self.lo) / self.width).astype(np.int64)
        under = idx < 0
        over = idx >= self.n_bins
        n_under = int(under.sum())
        n_over = int(over.sum())
        self.below += n_under
        self.above += n_over
        if n_under or n_over:
            idx = idx[~(under | over)]
        if idx.size:
            self.counts += np.bincount(idx, minlength=self.n_bins)

    def quantile(self, q: float) -> float:
        """Return the q-quantile of the accumulated log returns."""
        if self.total == 0:
            return 0.0
        target = q * self.total
        if target <= self.below:
            return self.lo
        if target >= self.total - self.above:
            return self.hi
        target -= self.below
        cum = np.cumsum(self.counts)
        i = int(np.searchsorted(cum, target, side="left"))
        i = min(i, self.n_bins - 1)
        prev = float(cum[i - 1]) if i > 0 else 0.0
        in_bin = float(self.counts[i])
        frac = (target - prev) / in_bin if in_bin > 0 else 0.5
        return self.lo + (i + frac) * self.width

    @property
    def mean(self) -> float:
        return self._sum / self.total if self.total else 0.0

    @property
    def std(self) -> float:
        if self.total < 2:
            return 0.0
        var = self._sum2 / self.total - self.mean ** 2
        return math.sqrt(max(var, 0.0))

    def moments(self) -> tuple[float, float]:
        """(skew, excess kurtosis) of the accumulated log returns.

        Computed from raw power sums, so it is exact for a stream rather
        than an approximation that depends on chunk boundaries.
        """
        n = self.total
        if n < 4:
            return 0.0, 0.0
        m = self._sum / n
        m2 = self._sum2 / n - m * m
        if m2 <= 0:
            return 0.0, 0.0
        m3 = self._sum3 / n - 3 * m * (self._sum2 / n) + 2 * m ** 3
        m4 = (self._sum4 / n - 4 * m * (self._sum3 / n)
              + 6 * m * m * (self._sum2 / n) - 3 * m ** 4)
        s = math.sqrt(m2)
        return m3 / s ** 3, m4 / (m2 * m2) - 3.0


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------
def _normals(rng: np.random.Generator, n_paths: int, n_steps: int,
             antithetic: bool) -> np.ndarray:
    """(n_paths, n_steps) standard normals, antithetically paired if asked."""
    if not antithetic:
        return rng.standard_normal((n_paths, n_steps))
    half = n_paths // 2
    z = rng.standard_normal((half, n_steps))
    return np.concatenate([z, -z], axis=0)


def _log_increments(cfg: MCConfig, rng: np.random.Generator, n_paths: int,
                    t: float, iv: float, r: float, q: float) -> np.ndarray:
    """(n_paths, n_steps) log-price increments for one chunk."""
    n = cfg.n_steps
    dt = t / n

    if cfg.model == "bootstrap":
        # Resample historical log returns, then standardise and rescale so the
        # simulated horizon variance matches sigma^2 * t and the drift is
        # risk-neutral. Shape (skew/kurtosis) is preserved; scale is not.
        src = cfg.returns
        idx = rng.integers(0, src.size, size=(n_paths, n))
        raw = src[idx]
        src_mu = float(src.mean())
        src_sd = float(src.std(ddof=1)) or 1e-9
        z = (raw - src_mu) / src_sd
        return (r - q - 0.5 * iv * iv) * dt + iv * math.sqrt(dt) * z

    z = _normals(rng, n_paths, n, cfg.antithetic)
    diff = (r - q - 0.5 * iv * iv) * dt + iv * math.sqrt(dt) * z

    if cfg.model == "merton":
        lam, mj, sj = cfg.jump_intensity, cfg.jump_mean, cfg.jump_vol
        if lam > 0:
            # E[J-1] = exp(mj + sj^2/2) - 1; compensate the drift so the
            # discounted price remains a martingale.
            k = math.exp(mj + 0.5 * sj * sj) - 1.0
            counts = rng.poisson(lam * dt, size=(n_paths, n))
            # Sum of `counts` iid N(mj, sj^2) is N(counts*mj, counts*sj^2).
            jump = counts * mj
            if sj > 0:
                jump = jump + np.sqrt(counts) * sj * rng.standard_normal((n_paths, n))
            diff = diff - lam * k * dt + jump
    return diff


def _chunks(cfg: MCConfig) -> Iterable[int]:
    remaining = cfg.n_paths
    while remaining > 0:
        take = min(cfg.chunk_paths, remaining)
        if cfg.antithetic and take % 2:
            take -= 1
            if take == 0:
                break
        yield take
        remaining -= take


# ---------------------------------------------------------------------------
# Terminal-distribution simulation (exact - one step, no discretisation)
# ---------------------------------------------------------------------------
def simulate_terminal(spot: float, t: float, iv: float, cfg: MCConfig | None = None,
                      r: float = 0.04, q: float = 0.0) -> np.ndarray:
    """Draw ``cfg.n_paths`` terminal prices S_T.

    For the GBM model this uses the exact lognormal solution in a single
    step - there is no time-discretisation bias to worry about, only Monte
    Carlo sampling error. Returns the full array, so keep n_paths sane
    (<= ~20M) when calling this directly; the streaming entry points below
    do not hold arrays this large.
    """
    cfg = cfg or MCConfig()
    if spot <= 0 or t <= 0 or iv <= 0:
        return np.full(cfg.n_paths, max(spot, 0.0), dtype=float)
    one_step = MCConfig(
        n_paths=cfg.n_paths, n_steps=1, seed=cfg.seed, antithetic=cfg.antithetic,
        chunk_paths=cfg.chunk_paths, model=cfg.model,
        jump_intensity=cfg.jump_intensity, jump_mean=cfg.jump_mean,
        jump_vol=cfg.jump_vol, returns=cfg.returns,
    )
    rng = np.random.default_rng(cfg.seed)
    pieces: list[np.ndarray] = []
    for take in _chunks(one_step):
        inc = _log_increments(one_step, rng, take, t, iv, r, q)
        pieces.append(spot * np.exp(inc[:, 0]))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)


# ---------------------------------------------------------------------------
# Cone + terminal statistics, streamed
# ---------------------------------------------------------------------------
_CONE_Q = (("p5", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95))


def simulate_cone(spot: float, t: float, iv: float, cfg: MCConfig | None = None,
                  r: float = 0.04, q: float = 0.0) -> MCResult:
    """Simulate paths and return the quantile cone plus terminal statistics.

    Memory is bounded by ``cfg.chunk_paths * cfg.n_steps`` regardless of
    ``cfg.n_paths`` - per-step quantiles come from streaming histograms.
    """
    cfg = cfg or MCConfig()
    started = time.perf_counter()
    result = MCResult(spot=float(spot), horizon_years=float(t), iv=float(iv),
                      model=cfg.model, n_paths=0, n_steps=cfg.n_steps, elapsed_s=0.0)
    if spot <= 0 or t <= 0 or iv <= 0:
        result.elapsed_s = time.perf_counter() - started
        return result

    n = cfg.n_steps
    sigma_total = iv * math.sqrt(t)
    hists = [_LogHistogram(sigma_total * math.sqrt((i + 1) / n)) for i in range(n)]
    rng = np.random.default_rng(cfg.seed)

    total_paths = 0
    sum_terminal = 0.0
    sum_terminal_sq = 0.0
    n_up = 0

    for take in _chunks(cfg):
        inc = _log_increments(cfg, rng, take, t, iv, r, q)
        cum = np.cumsum(inc, axis=1)
        for i in range(n):
            hists[i].update(cum[:, i])
        terminal = spot * np.exp(cum[:, -1])
        sum_terminal += float(terminal.sum())
        sum_terminal_sq += float(np.dot(terminal, terminal))
        n_up += int((terminal > spot).sum())
        total_paths += take

    if total_paths == 0:
        result.elapsed_s = time.perf_counter() - started
        return result

    mean = sum_terminal / total_paths
    var = max(sum_terminal_sq / total_paths - mean * mean, 0.0)
    std = math.sqrt(var)

    cone: list[dict] = []
    for i, h in enumerate(hists):
        row = {"frac": (i + 1) / n, "t_years": t * (i + 1) / n}
        for name, qq in _CONE_Q:
            row[name] = float(spot * math.exp(h.quantile(qq)))
        cone.append(row)

    skew, kurt = hists[-1].moments()
    result.n_paths = total_paths
    result.mean = float(mean)
    result.std = float(std)
    result.quantiles = {name: cone[-1][name] for name, _ in _CONE_Q}
    result.cone = cone
    result.prob_up = n_up / total_paths
    result.expected_move = float(std)
    result.expected_move_pct = float(std / spot * 100.0) if spot else 0.0
    result.standard_error = float(std / math.sqrt(total_paths))
    result.skew = float(skew)
    result.kurtosis = float(kurt)
    result.elapsed_s = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# Barrier / "does price reach this strike" probabilities
# ---------------------------------------------------------------------------
def touch_probabilities(spot: float, barriers: Sequence[float], t: float, iv: float,
                        cfg: MCConfig | None = None, r: float = 0.04, q: float = 0.0,
                        bridge_correction: bool = True) -> dict[float, dict]:
    """P(path touches each barrier before T) and P(S_T beyond it).

    ``bridge_correction`` applies the Brownian-bridge crossing probability
    between consecutive observations. Discrete monitoring alone undercounts
    hits, because a path can cross the barrier and return within one step;
    that bias is larger than the sampling error we spent millions of paths
    removing, so it is on by default.

    Returns ``{barrier: {"touch": p, "beyond": p, "touch_se": se}}`` where
    *beyond* means at-or-above the barrier for up-barriers and at-or-below
    it for down-barriers.
    """
    cfg = cfg or MCConfig()
    ks = [float(k) for k in barriers if k and float(k) > 0]
    if spot <= 0 or t <= 0 or iv <= 0 or not ks:
        return {k: {"touch": 0.0, "beyond": 0.0, "touch_se": 0.0} for k in ks}

    n = cfg.n_steps
    dt = t / n
    var_step = iv * iv * dt
    log_b = np.array([math.log(k / spot) for k in ks])
    is_up = log_b > 0

    # With the bridge correction each path contributes a *probability* of
    # touching rather than a 0/1, so accumulate sums of squares as well in
    # order to report an honest standard error.
    sum_touch = np.zeros(len(ks))
    sum_touch_sq = np.zeros(len(ks))
    sum_beyond = np.zeros(len(ks))
    total_paths = 0

    rng = np.random.default_rng(cfg.seed)
    for take in _chunks(cfg):
        inc = _log_increments(cfg, rng, take, t, iv, r, q)
        cum = np.cumsum(inc, axis=1)                     # (take, n) log S_t / S_0
        prev = np.concatenate([np.zeros((take, 1)), cum[:, :-1]], axis=1)
        end = cum[:, -1]
        for j, b in enumerate(log_b):
            if is_up[j]:
                hit = cum >= b
                gap_prev = b - prev
                gap_cur = b - cum
            else:
                hit = cum <= b
                gap_prev = prev - b
                gap_cur = cum - b
            if bridge_correction and var_step > 0:
                # P(crossed inside the step | both endpoints on the same side)
                with np.errstate(over="ignore", under="ignore"):
                    p_cross = np.exp(-2.0 * gap_prev * gap_cur / var_step)
                # A step that ends past the barrier crossed with probability 1.
                p_cross = np.where(hit, 1.0, p_cross)
                p_survive = np.prod(1.0 - np.clip(p_cross, 0.0, 1.0), axis=1)
                p_touch = 1.0 - p_survive
            else:
                p_touch = hit.any(axis=1).astype(float)
            sum_touch[j] += float(p_touch.sum())
            sum_touch_sq[j] += float(np.dot(p_touch, p_touch))
            sum_beyond[j] += float((end >= b).sum() if is_up[j] else (end <= b).sum())
        total_paths += take

    out: dict[float, dict] = {}
    for j, k in enumerate(ks):
        p = sum_touch[j] / total_paths
        var = max(sum_touch_sq[j] / total_paths - p * p, 0.0)
        out[k] = {
            "touch": float(min(max(p, 0.0), 1.0)),
            "beyond": float(sum_beyond[j] / total_paths),
            "touch_se": float(math.sqrt(var / total_paths)),
        }
    return out


# ---------------------------------------------------------------------------
# Option pricing (used for validation, and useful on its own)
# ---------------------------------------------------------------------------
def price_european(spot: float, strike: float, t: float, iv: float,
                   right: str = "C", cfg: MCConfig | None = None,
                   r: float = 0.04, q: float = 0.0,
                   control_variate: bool = True) -> dict:
    """Monte Carlo price of a European option.

    With ``control_variate`` on, the payoff is regressed against S_T (whose
    expectation ``S0*exp((r-q)T)`` is known exactly) and the residual used
    instead. The optimal regression coefficient is estimated from the same
    sample; the resulting bias is O(1/N) and negligible next to the variance
    it removes.

    Returns ``{price, stderr, n_paths}``.
    """
    cfg = cfg or MCConfig()
    st = simulate_terminal(spot, t, iv, cfg, r, q)
    if st.size == 0:
        return {"price": 0.0, "stderr": 0.0, "n_paths": 0}
    payoff = (np.maximum(st - strike, 0.0) if right.upper().startswith("C")
              else np.maximum(strike - st, 0.0))
    y = math.exp(-r * t) * payoff

    if control_variate and t > 0 and iv > 0:
        ex = spot * math.exp((r - q) * t)
        vx = float(st.var())
        if vx > 0:
            beta = float(np.cov(y, st, ddof=1)[0, 1] / vx)
            y = y - beta * (st - ex)

    n = y.size
    return {
        "price": float(y.mean()),
        "stderr": float(y.std(ddof=1) / math.sqrt(n)),
        "n_paths": int(n),
    }


# ---------------------------------------------------------------------------
# Closed-form references (used only to check the simulator)
# ---------------------------------------------------------------------------
def bs_price(spot: float, strike: float, t: float, iv: float, right: str = "C",
             r: float = 0.04, q: float = 0.0) -> float:
    """Black-Scholes-Merton European price."""
    if spot <= 0 or strike <= 0 or t <= 0 or iv <= 0:
        intrinsic = spot - strike if right.upper().startswith("C") else strike - spot
        return max(intrinsic, 0.0)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    if right.upper().startswith("C"):
        return float(spot * math.exp(-q * t) * norm.cdf(d1)
                     - strike * math.exp(-r * t) * norm.cdf(d2))
    return float(strike * math.exp(-r * t) * norm.cdf(-d2)
                 - spot * math.exp(-q * t) * norm.cdf(-d1))


def analytic_touch(spot: float, barrier: float, t: float, iv: float,
                   r: float = 0.04, q: float = 0.0) -> float:
    """Exact first-passage probability for GBM (see ``projection.touch_prob``)."""
    from .projection import touch_prob
    return touch_prob(spot, barrier, t, iv, r, q)


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------
def validate(spot: float = 100.0, t: float = 0.25, iv: float = 0.30,
             r: float = 0.04, q: float = 0.0, n_paths: int = 1_000_000,
             n_steps: int = 128, seed: int = 7) -> list[dict]:
    """Compare simulated values against closed-form ones.

    Each row reports the analytic value, the simulated value, the Monte
    Carlo standard error, and the error expressed in standard errors (``z``).
    Correct code produces |z| under about 3 for essentially every row; a
    systematically large |z| on the barrier rows specifically points at the
    discretisation/bridge handling rather than at the sampling.
    """
    rows: list[dict] = []
    cfg_terminal = MCConfig(n_paths=n_paths, n_steps=1, seed=seed,
                            chunk_paths=min(n_paths, 1_000_000))

    # --- European options: MC price vs Black-Scholes -----------------------
    for right, k in (("C", spot), ("C", spot * 1.10), ("P", spot * 0.90)):
        analytic = bs_price(spot, k, t, iv, right, r, q)
        mc = price_european(spot, k, t, iv, right, cfg_terminal, r, q)
        se = mc["stderr"] or 1e-12
        rows.append({
            "check": f"European {right} K={k:.2f}",
            "analytic": analytic, "simulated": mc["price"],
            "stderr": mc["stderr"], "z": (mc["price"] - analytic) / se,
            "n_paths": mc["n_paths"],
        })

    # --- Digital / terminal CDF: MC P(S_T > K) vs N(d2) --------------------
    from .projection import above_prob
    st = simulate_terminal(spot, t, iv, cfg_terminal, r, q)
    for k in (spot * 0.95, spot, spot * 1.05):
        analytic = above_prob(spot, k, t, iv, r, q)
        p = float((st > k).mean())
        se = math.sqrt(max(p * (1 - p), 1e-12) / st.size)
        rows.append({
            "check": f"P(S_T > {k:.2f})",
            "analytic": analytic, "simulated": p, "stderr": se,
            "z": (p - analytic) / se, "n_paths": int(st.size),
        })

    # --- Martingale check: E[S_T] must equal S0*exp((r-q)T) ----------------
    analytic = spot * math.exp((r - q) * t)
    mean = float(st.mean())
    se = float(st.std(ddof=1) / math.sqrt(st.size))
    rows.append({
        "check": "E[S_T] (martingale)", "analytic": analytic, "simulated": mean,
        "stderr": se, "z": (mean - analytic) / (se or 1e-12), "n_paths": int(st.size),
    })

    # --- Barrier: MC one-touch vs closed-form first passage ----------------
    cfg_path = MCConfig(n_paths=max(n_paths // 5, 20_000), n_steps=n_steps, seed=seed,
                        chunk_paths=50_000)
    barriers = [spot * 1.05, spot * 0.95]
    sim = touch_probabilities(spot, barriers, t, iv, cfg_path, r, q, bridge_correction=True)
    for b in barriers:
        analytic = analytic_touch(spot, b, t, iv, r, q)
        got = sim[b]
        se = got["touch_se"] or 1e-12
        rows.append({
            "check": f"P(touch {b:.2f})", "analytic": analytic,
            "simulated": got["touch"], "stderr": got["touch_se"],
            "z": (got["touch"] - analytic) / se, "n_paths": cfg_path.n_paths,
        })
    return rows


def convergence_table(spot: float = 100.0, t: float = 0.25, iv: float = 0.30,
                      r: float = 0.04, q: float = 0.0,
                      counts: Sequence[int] = (10_000, 100_000, 1_000_000, 5_000_000),
                      seed: int = 7) -> list[dict]:
    """Show ATM-call pricing error and standard error shrinking as 1/sqrt(N).

    This is the empirical justification for the path count: read off the N
    at which the standard error drops below the resolution you display.
    Run without the control variate so the numbers show raw convergence.
    """
    analytic = bs_price(spot, spot, t, iv, "C", r, q)
    rows = []
    for n in counts:
        cfg = MCConfig(n_paths=int(n), n_steps=1, seed=seed,
                       chunk_paths=min(int(n), 1_000_000))
        started = time.perf_counter()
        mc = price_european(spot, spot, t, iv, "C", cfg, r, q, control_variate=False)
        rows.append({
            "n_paths": int(n),
            "analytic": analytic,
            "simulated": mc["price"],
            "abs_error": abs(mc["price"] - analytic),
            "stderr": mc["stderr"],
            "elapsed_s": time.perf_counter() - started,
        })
    return rows
