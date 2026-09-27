"""Task 1.3 — CPCV purging, path/Sharpe-distribution generation, Deflated Sharpe."""
import math

import numpy as np
import pytest

from backtesting.cpcv import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    generate_paths,
    group_bounds,
    num_backtest_paths,
    path_sharpes,
    probabilistic_sharpe_ratio,
    purge_embargo_train_mask,
    run_cpcv,
)


class TestPartition:
    def test_group_bounds_cover_all_samples_contiguously(self):
        b = group_bounds(100, 6)
        assert b[0][0] == 0 and b[-1][1] == 100
        for (s, e), (s2, _) in zip(b, b[1:]):
            assert e == s2 and e > s          # contiguous, non-empty, no gaps/overlaps

    def test_num_paths_matches_lopez_de_prado(self):
        assert num_backtest_paths(6, 2) == math.comb(5, 1) == 5
        assert num_backtest_paths(6, 3) == math.comb(5, 2) == 10
        assert num_backtest_paths(4, 2) == 3


class TestPurgeEmbargo:
    def test_purges_horizon_before_and_embargoes_after(self):
        # test block [8, 11]; horizon=2 purges 6,7 before; embargo≈1 drops 12 after.
        mask = purge_embargo_train_mask(
            n_samples=20, test_idx=np.arange(8, 12), label_horizon=2, embargo_pct=0.05
        )
        train = set(np.where(mask)[0].tolist())
        assert {6, 7, 8, 9, 10, 11, 12}.isdisjoint(train)   # purged/test/embargoed
        assert {0, 1, 2, 3, 4, 5}.issubset(train)           # untouched before purge
        assert {13, 14, 15, 19}.issubset(train)             # untouched after embargo

    def test_test_observations_are_never_in_train(self):
        test_idx = np.array([0, 1, 2, 15, 16])
        mask = purge_embargo_train_mask(20, test_idx, label_horizon=1, embargo_pct=0.0)
        assert mask[test_idx].sum() == 0

    def test_zero_horizon_zero_embargo_only_removes_test(self):
        test_idx = np.arange(5, 10)
        mask = purge_embargo_train_mask(20, test_idx, label_horizon=0, embargo_pct=0.0)
        assert set(np.where(~mask)[0]) == set(test_idx.tolist())


class TestSplitter:
    def test_number_of_splits_is_C_N_k(self):
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0, label_horizon=0)
        splits = list(cv.split(120))
        assert len(splits) == math.comb(6, 2) == 15

    def test_train_and_test_are_disjoint_every_split(self):
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
        for s in cv.split(100):
            assert set(s.train_idx).isdisjoint(set(s.test_idx))


class TestPathGeneration:
    def test_paths_count_and_full_coverage(self):
        # evaluate returns the test indices themselves → every reconstructed path
        # must equal arange(n) exactly (each observation appears once, in order).
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.0, label_horizon=0)
        n = 120
        paths = generate_paths(n, cv, evaluate=lambda tr, te: te.astype(float))
        assert len(paths) == cv.num_paths == 5
        for p in paths:
            assert np.array_equal(p, np.arange(n, dtype=float))

    def test_distribution_is_non_degenerate_when_evaluation_depends_on_training(self):
        # A model that refits on train produces different test returns per split,
        # so the reconstructed paths — and their Sharpes — genuinely differ.
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_pct=0.01, label_horizon=1)

        def evaluate(train_idx, test_idx):
            rng = np.random.default_rng(int(train_idx.sum()) % (2**31))
            drift = 0.001 * (train_idx.size / 120)
            return rng.normal(drift, 0.01, size=test_idx.size)

        srs = path_sharpes(generate_paths(120, cv, evaluate))
        assert len(srs) == cv.num_paths == 5
        assert np.std(srs) > 0            # a real distribution, not a point estimate


class TestExpectedMaxSharpe:
    def test_increases_with_more_trials(self):
        e2 = expected_max_sharpe(1.0, n_trials=2)
        e10 = expected_max_sharpe(1.0, n_trials=10)
        e100 = expected_max_sharpe(1.0, n_trials=100)
        assert 0 < e2 < e10 < e100        # more variations tried → higher bar

    def test_single_trial_has_no_selection_bias(self):
        assert expected_max_sharpe(1.0, n_trials=1) == 0.0
        assert expected_max_sharpe(0.0, n_trials=50) == 0.0


class TestDeflatedSharpe:
    def test_psr_half_when_observed_equals_benchmark(self):
        assert probabilistic_sharpe_ratio(0.2, 0.2, n_obs=500) == pytest.approx(0.5, abs=1e-9)

    def test_dsr_in_unit_interval(self):
        for nt in (1, 5, 50, 500):
            dsr = deflated_sharpe_ratio(0.1, sharpe_variance=0.01, n_trials=nt, n_obs=1000)
            assert 0.0 <= dsr <= 1.0

    def test_more_trials_deflate_the_ratio(self):
        few = deflated_sharpe_ratio(0.12, sharpe_variance=0.02, n_trials=1, n_obs=1000)
        many = deflated_sharpe_ratio(0.12, sharpe_variance=0.02, n_trials=500, n_obs=1000)
        assert few > many                 # the whole point: penalise multiple testing

    def test_strong_single_trial_is_significant_but_overtested_is_not(self):
        strong = deflated_sharpe_ratio(0.30, sharpe_variance=0.01, n_trials=1, n_obs=1000)
        overtested = deflated_sharpe_ratio(0.03, sharpe_variance=0.01, n_trials=1000, n_obs=1000)
        assert strong > 0.95
        assert overtested < 0.5


class TestRunCPCV:
    def test_end_to_end_result_shape(self):
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)

        def evaluate(train_idx, test_idx):
            rng = np.random.default_rng(int(train_idx.sum()) % (2**31))
            return rng.normal(0.0008, 0.01, size=test_idx.size)

        res = run_cpcv(120, evaluate, cv=cv, n_trials=20)
        assert res.n_paths == 5
        assert len(res.path_sharpes) == 5
        assert 0.0 <= res.deflated_sharpe <= 1.0
        assert res.sharpe_min <= res.observed_sharpe <= res.sharpe_max
        assert isinstance(res.summary(), str)


class TestSignalsToReturns:
    def test_long_captures_up_move_and_cost_bites_on_turnover(self):
        from backtesting.cpcv import signals_to_returns
        close = np.array([100.0, 110.0, 121.0])   # +10% then +10%
        sig = np.array([1.0, 0.0, 0.0])           # long from bar 0, held
        r = signals_to_returns(close, sig)
        assert r[1] == pytest.approx(0.10) and r[2] == pytest.approx(0.10)
        # A cost per turn reduces the entry bar's realized return.
        rc = signals_to_returns(close, np.array([0.0, 1.0, 0.0]), cost_per_turn=0.001)
        assert rc[1] == pytest.approx(-0.001)     # opened at bar1, no price move yet, pays cost

    def test_flat_signal_holds_no_position(self):
        from backtesting.cpcv import signals_to_returns
        r = signals_to_returns([100.0, 200.0], [0.0, 0.0])
        assert r[1] == 0.0
