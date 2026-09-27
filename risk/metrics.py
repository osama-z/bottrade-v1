"""metrics.py — the single home for performance metrics (audit V-59).

Three incompatible Sharpe implementations previously coexisted
(walk-forward: per-trade; backtest engine: annualised per-candle;
dashboard: annualised hourly equity curve), plus three win/loss
classifications that flipped zero-PnL trades between categories per
module. Every consumer now imports from here, and every threshold in
config/docs must name the variant it gates.

Conventions (fixed project-wide):
- A trade is a WIN when pnl > 0; zero-PnL trades count as losses
  (conservative direction).
- ``sharpe_per_trade`` is NOT annualised — mean(pnl)/std(pnl, ddof=1)
  over closed trades, holding-period agnostic. This is the variant the
  Stage 5 walk-forward gate uses.
- ``sharpe_annualized`` operates on a *return series* (per-period
  fractional returns) and scales by sqrt(periods_per_year). This is the
  backtest/dashboard variant.
"""

from __future__ import annotations

import math
from typing import Sequence

PERIODS_PER_YEAR: dict[str, int] = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}


def sharpe_per_trade(pnl_series: Sequence[float]) -> float:
    """Per-trade Sharpe: mean(pnl) / std(pnl, ddof=1). Not annualised.

    Returns 0.0 with fewer than 2 trades or zero variance.
    """
    n = len(pnl_series)
    if n < 2:
        return 0.0
    mean = sum(pnl_series) / n
    variance = sum((x - mean) ** 2 for x in pnl_series) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    return mean / std if std > 0 else 0.0


def sharpe_annualized(
    returns: Sequence[float], periods_per_year: int
) -> float:
    """Annualised Sharpe over a per-period fractional return series."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def sortino_annualized(
    returns: Sequence[float], periods_per_year: int
) -> float:
    """Annualised Sortino (downside deviation, ddof=1) over returns."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    downside = [r for r in returns if r < 0]
    if len(downside) < 2:
        return 0.0
    d_mean = sum(downside) / len(downside)
    d_var = sum((r - d_mean) ** 2 for r in downside) / (len(downside) - 1)
    d_std = math.sqrt(d_var) if d_var > 0 else 0.0
    if d_std == 0:
        return 0.0
    return mean / d_std * math.sqrt(periods_per_year)


def is_win(pnl: float) -> bool:
    """Project-wide win definition: strictly positive PnL.

    Zero-PnL trades are counted as losses everywhere (conservative).
    """
    return pnl > 0


def win_rate(pnl_series: Sequence[float]) -> float:
    """Fraction of winning trades per ``is_win``. 0.0 when empty."""
    if not pnl_series:
        return 0.0
    return sum(1 for p in pnl_series if is_win(p)) / len(pnl_series)
