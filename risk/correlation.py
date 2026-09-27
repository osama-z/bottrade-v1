"""Rolling correlation tracker for correlation-aware heat caps (Roadmap Task 2.2).

Maintains a rolling window (default 30 days) of per-asset returns and exposes the
correlation matrix + the average pairwise correlation for a set of symbols. The
RiskManager multiplies nominal portfolio heat by (1 + average correlation), so
two highly-correlated longs count as far more risk than their nominal sum.
"""

from __future__ import annotations

import pandas as pd

from risk.manager import average_pairwise_correlation


class CorrelationTracker:
    """Rolling per-asset return store → correlation matrix / average correlation."""

    def __init__(self, window_days: int = 30, min_periods: int = 10) -> None:
        self.window_days = window_days
        self.min_periods = min_periods
        self._returns: dict[str, pd.Series] = {}

    # ─── Ingest ───────────────────────────────────────────────────────────────
    def update_returns(self, symbol: str, returns: pd.Series) -> None:
        """Store/replace a symbol's return series (indexed by timestamp)."""
        self._returns[symbol] = pd.Series(returns).astype(float).dropna()

    def update_from_prices(self, symbol: str, close: pd.Series) -> None:
        """Convenience: derive returns from a close-price series."""
        self.update_returns(symbol, pd.Series(close).astype(float).pct_change())

    @property
    def symbols(self) -> list[str]:
        return list(self._returns)

    # ─── Correlation ──────────────────────────────────────────────────────────
    def matrix(self) -> pd.DataFrame:
        """Correlation matrix over the trailing ``window_days`` of returns."""
        if not self._returns:
            return pd.DataFrame()
        df = pd.DataFrame(self._returns).sort_index()
        if isinstance(df.index, pd.DatetimeIndex) and len(df):
            cutoff = df.index.max() - pd.Timedelta(days=self.window_days)
            df = df[df.index >= cutoff]
        return df.corr(min_periods=self.min_periods)

    def average_correlation(self, symbols) -> float:
        """Average pairwise correlation among ``symbols`` (0.0 if not estimable)."""
        return average_pairwise_correlation(self.matrix(), symbols)
