"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


@pytest.fixture(scope="session")
def ohlcv_300() -> pd.DataFrame:
    """300-row synthetic OHLCV DataFrame, shared across all tests."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n, freq="1h"
    )
    price = 50000.0
    prices = [price]
    for _ in range(n - 1):
        price *= (1 + np.random.normal(0, 0.005))
        prices.append(price)

    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * np.random.uniform(0.998, 1.002, n),
        "high": prices * np.random.uniform(1.001, 1.015, n),
        "low": prices * np.random.uniform(0.985, 0.999, n),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
        "returns": pd.Series(prices).pct_change().values,
        "log_returns": np.log(pd.Series(prices) / pd.Series(prices).shift(1)).values,
        "body_pct": np.random.uniform(-0.01, 0.01, n),
        "upper_wick": np.random.uniform(0, 50, n),
        "lower_wick": np.random.uniform(0, 50, n),
        "range": np.random.uniform(100, 500, n),
        "is_bullish": np.random.randint(0, 2, n),
        "volatility_24h": np.random.uniform(0.001, 0.02, n),
    }, index=dates)
