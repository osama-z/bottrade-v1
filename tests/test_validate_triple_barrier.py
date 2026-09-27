"""Unit tests for the triple-barrier validation script's pure helpers."""
import numpy as np
import pytest

from scripts.validate_triple_barrier import annualized, per_trade_sharpe


def test_per_trade_sharpe_ignores_flat_bars():
    # Zeros (no-bet bars) must not count as observations.
    path = np.array([0.0, 0.02, 0.0, -0.01, 0.03, 0.0])
    bets = path[path != 0.0]
    assert per_trade_sharpe(path) == pytest.approx(bets.mean() / bets.std(ddof=1))


def test_per_trade_sharpe_zero_when_fewer_than_two_bets():
    assert per_trade_sharpe(np.array([0.0, 0.0, 0.05])) == 0.0
    assert per_trade_sharpe(np.zeros(5)) == 0.0


def test_annualized_scales_by_sqrt_trades_per_year():
    assert annualized(0.1, 8760) == pytest.approx(0.1 * np.sqrt(8760))
