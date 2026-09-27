"""Combinatorial Purged Cross-Validation (CPCV) + Deflated Sharpe Ratio.

Roadmap Phase 1, Task 1.3 — replace basic walk-forward validation with
López de Prado's CPCV (Advances in Financial Machine Learning, ch. 7 & 12) to
eliminate data leakage / overlapping labels and to produce a *distribution* of
out-of-sample Sharpe ratios instead of a single point estimate.

Mechanics
---------
- Partition N observations into ``n_groups`` contiguous groups.
- Enumerate every combination of ``n_test_groups`` groups as the test set
  (C(N_g, k) splits). The rest is training, with two leakage guards:
    * **Purge**  — drop training observations whose forward label window
      ``[i, i + label_horizon]`` overlaps a test block (the label peeks into
      test).
    * **Embargo** — drop training observations in a window immediately AFTER a
      test block (serial-correlation leakage from test → train).
- Reassemble ``φ = C(N_g − 1, k − 1)`` complete back-test PATHS: each path
  covers every group exactly once, each group's slice coming from a model
  trained on a different (purged) training set. One Sharpe per path → a
  distribution.

Deflated Sharpe Ratio
---------------------
The Probabilistic Sharpe Ratio evaluated against the Sharpe you'd expect from
the *best* of ``n_trials`` independent strategy variations under the null
(so more variations tested ⇒ a higher bar). All Sharpes here are
per-observation (not annualised); annualisation cancels in the PSR z-score but
we keep it explicit to avoid ambiguity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

_EULER_MASCHERONI = 0.5772156649015329


# ─── Group partitioning ────────────────────────────────────────────────────────
def group_bounds(n_samples: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) bounds partitioning ``n_samples`` into groups
    as equal as possible (earlier groups absorb the remainder)."""
    if n_groups < 2 or n_groups > n_samples:
        raise ValueError(f"n_groups must be in [2, n_samples]; got {n_groups}")
    edges = np.linspace(0, n_samples, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def num_backtest_paths(n_groups: int, n_test_groups: int) -> int:
    """φ = C(N_g − 1, k − 1): the number of complete OOS paths CPCV yields."""
    return math.comb(n_groups - 1, n_test_groups - 1)


def _contiguous_blocks(sorted_idx: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield inclusive (start, end) runs of consecutive integers."""
    if len(sorted_idx) == 0:
        return
    start = prev = sorted_idx[0]
    for x in sorted_idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            yield (int(start), int(prev))
            start = prev = x
    yield (int(start), int(prev))


# ─── Purge + embargo ───────────────────────────────────────────────────────────
def purge_embargo_train_mask(
    n_samples: int,
    test_idx: np.ndarray,
    label_horizon: int,
    embargo_pct: float,
) -> np.ndarray:
    """Boolean train mask after removing the test set, purged and embargoed.

    Purge: the ``label_horizon`` observations *before* each test block (their
    forward label overlaps test). Embargo: ``ceil(embargo_pct * n_samples)``
    observations *after* each test block.
    """
    train = np.ones(n_samples, dtype=bool)
    test_idx = np.sort(np.asarray(test_idx, dtype=int))
    train[test_idx] = False
    embargo_span = int(math.ceil(embargo_pct * n_samples)) if embargo_pct > 0 else 0

    for a, b in _contiguous_blocks(test_idx):
        lo = max(0, a - label_horizon)      # purge the horizon before the block
        train[lo:b + 1] = False
        hi = min(n_samples, b + 1 + embargo_span)   # embargo after the block
        train[b + 1:hi] = False
    return train


# ─── The splitter ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    n_groups: int = 6
    n_test_groups: int = 2
    embargo_pct: float = 0.01
    label_horizon: int = 1

    def __post_init__(self) -> None:
        if not (1 <= self.n_test_groups < self.n_groups):
            raise ValueError("n_test_groups must be in [1, n_groups)")

    @property
    def num_paths(self) -> int:
        return num_backtest_paths(self.n_groups, self.n_test_groups)

    def test_group_combinations(self) -> list[tuple[int, ...]]:
        return list(_combinations(range(self.n_groups), self.n_test_groups))

    def split(self, n_samples: int) -> Iterator[CPCVSplit]:
        bounds = group_bounds(n_samples, self.n_groups)
        for combo in self.test_group_combinations():
            test_idx = np.sort(np.concatenate(
                [np.arange(bounds[g][0], bounds[g][1]) for g in combo]
            ))
            mask = purge_embargo_train_mask(
                n_samples, test_idx, self.label_horizon, self.embargo_pct
            )
            yield CPCVSplit(
                train_idx=np.where(mask)[0],
                test_idx=test_idx,
                test_groups=tuple(combo),
            )


def _combinations(iterable, r):
    from itertools import combinations
    return combinations(iterable, r)


# ─── Path assembly → Sharpe distribution ───────────────────────────────────────
Evaluator = Callable[[np.ndarray, np.ndarray], Sequence[float]]
"""``evaluate(train_idx, test_idx) -> per-observation returns aligned to test_idx``.

For an ML strategy this refits on ``train_idx`` and predicts on ``test_idx`` —
which makes each path genuinely different. For a fixed rule-based strategy the
returns are independent of training, so every path collapses to the same series
(zero dispersion) — honest, and the framework is ready for Task 1.5's model.
"""


def generate_paths(
    n_samples: int,
    cv: CombinatorialPurgedCV,
    evaluate: Evaluator,
) -> list[np.ndarray]:
    """Reassemble the φ complete OOS return paths."""
    bounds = group_bounds(n_samples, cv.n_groups)
    n_paths = cv.num_paths
    # path_group_returns[path][group] = that group's test-return slice
    path_group_returns: list[list[np.ndarray | None]] = [
        [None] * cv.n_groups for _ in range(n_paths)
    ]
    group_fill = [0] * cv.n_groups

    for split in cv.split(n_samples):
        test_returns = np.asarray(evaluate(split.train_idx, split.test_idx), dtype=float)
        if test_returns.shape[0] != split.test_idx.shape[0]:
            raise ValueError(
                "evaluate() returned {} values for {} test observations".format(
                    test_returns.shape[0], split.test_idx.shape[0]
                )
            )
        # test_idx is sorted → groups appear in ascending order; split back out.
        offset = 0
        for g in sorted(split.test_groups):
            glen = bounds[g][1] - bounds[g][0]
            seg = test_returns[offset:offset + glen]
            offset += glen
            path_group_returns[group_fill[g]][g] = seg
            group_fill[g] += 1

    paths = []
    for p in range(n_paths):
        paths.append(np.concatenate([path_group_returns[p][g] for g in range(cv.n_groups)]))
    return paths


def signals_to_returns(
    close: Sequence[float],
    signals: Sequence[float],
    cost_per_turn: float = 0.0,
) -> np.ndarray:
    """Per-bar strategy returns from discrete +1/-1/0 signals.

    A +1 opens/holds long, −1 opens/holds short, 0 holds the current position.
    Bar return_t = position_{t−1} · (close_t / close_{t−1} − 1), minus
    ``cost_per_turn`` on each unit of position change (a simple friction proxy
    so CPCV can run on cost-adjusted returns). This is the returns series CPCV
    (and the Deflated Sharpe) operate on.
    """
    close = np.asarray(close, dtype=float)
    sig = np.asarray(signals, dtype=float)
    pos = np.zeros(len(sig))
    cur = 0.0
    for i, s in enumerate(sig):
        cur = 1.0 if s == 1 else (-1.0 if s == -1 else cur)
        pos[i] = cur
    ret = np.zeros(len(close))
    ret[1:] = pos[:-1] * (close[1:] / close[:-1] - 1.0)
    if cost_per_turn:
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        ret = ret - turnover * cost_per_turn
    return ret


def _sharpe(returns: np.ndarray) -> float:
    """Per-observation Sharpe: mean / std(ddof=1). 0 for <2 points or no variance."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def path_sharpes(paths: Sequence[np.ndarray]) -> list[float]:
    """The Sharpe distribution — one Sharpe per reconstructed OOS path."""
    return [_sharpe(p) for p in paths]


# ─── Deflated / Probabilistic Sharpe Ratio ─────────────────────────────────────
def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """E[max Sharpe] of ``n_trials`` iid trials under the null (SR≈0).

    López de Prado's benchmark for the Deflated Sharpe Ratio. Returns 0 for a
    single trial (no selection bias to correct for)."""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    sigma = math.sqrt(sharpe_variance)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR: P(true SR > benchmark) given non-normal returns. ``kurtosis`` is
    non-excess (3 = normal)."""
    if n_obs < 2:
        return 0.0
    denom = math.sqrt(
        max(1e-12, 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2)
    )
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_variance: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR against the expected-max-Sharpe benchmark of ``n_trials``.

    More strategy variations tested ⇒ higher benchmark ⇒ lower DSR. A result is
    typically deemed significant at DSR > 0.95.
    """
    benchmark = expected_max_sharpe(sharpe_variance, n_trials)
    return probabilistic_sharpe_ratio(observed_sharpe, benchmark, n_obs, skew, kurtosis)


# ─── High-level result ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CPCVResult:
    n_paths: int
    path_sharpes: list[float]
    observed_sharpe: float          # mean across paths (the CPCV point estimate)
    sharpe_std: float               # dispersion across paths
    sharpe_min: float
    sharpe_max: float
    n_trials: int
    n_obs: int
    deflated_sharpe: float
    is_significant: bool = field(default=False)

    def summary(self) -> str:
        return (
            f"CPCV: {self.n_paths} paths | Sharpe mean={self.observed_sharpe:.3f} "
            f"std={self.sharpe_std:.3f} [{self.sharpe_min:.3f}, {self.sharpe_max:.3f}] "
            f"| trials={self.n_trials} | DSR={self.deflated_sharpe:.3f} "
            f"({'PASS' if self.is_significant else 'FAIL'} @0.95)"
        )


def run_cpcv(
    n_samples: int,
    evaluate: Evaluator,
    *,
    cv: CombinatorialPurgedCV | None = None,
    n_trials: int = 1,
    significance: float = 0.95,
) -> CPCVResult:
    """Run CPCV and deflate: assemble paths, build the Sharpe distribution, and
    compute the Deflated Sharpe Ratio against ``n_trials`` strategy variations."""
    cv = cv or CombinatorialPurgedCV()
    paths = generate_paths(n_samples, cv, evaluate)
    srs = path_sharpes(paths)
    all_returns = np.concatenate(paths) if paths else np.array([])

    observed = float(np.mean(srs)) if srs else 0.0
    sr_var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    sk = float(_skew(all_returns)) if all_returns.size > 2 else 0.0
    kt = float(_kurtosis(all_returns, fisher=False)) if all_returns.size > 3 else 3.0

    dsr = deflated_sharpe_ratio(
        observed_sharpe=observed,
        sharpe_variance=sr_var if sr_var > 0 else max(1e-12, np.var(srs) if srs else 0.0),
        n_trials=max(1, n_trials),
        n_obs=all_returns.size,
        skew=sk,
        kurtosis=kt,
    )
    return CPCVResult(
        n_paths=len(paths),
        path_sharpes=srs,
        observed_sharpe=observed,
        sharpe_std=float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0,
        sharpe_min=min(srs) if srs else 0.0,
        sharpe_max=max(srs) if srs else 0.0,
        n_trials=max(1, n_trials),
        n_obs=all_returns.size,
        deflated_sharpe=dsr,
        is_significant=dsr > significance,
    )
