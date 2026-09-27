"""
Tests for DataPreprocessor — verify data cleaning and derived column math.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from data.preprocessor import DataPreprocessor
from core.exceptions import InsufficientDataError


@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    """100 rows of raw OHLCV with some imperfections."""
    np.random.seed(42)
    n = 100
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
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.uniform(50, 500, n),
    }, index=dates)


class TestDataPreprocessor:

    def test_process_returns_dataframe(self, raw_ohlcv):
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_derived_columns_added(self, raw_ohlcv):
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        expected_cols = ["returns", "log_returns", "body", "body_pct",
                         "upper_wick", "lower_wick", "range", "is_bullish"]
        for col in expected_cols:
            assert col in result.columns, f"Missing derived column: {col}"

    def test_returns_calculation_correct(self, raw_ohlcv):
        """returns = (close_t - close_{t-1}) / close_{t-1}"""
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        # Check a specific return calculation
        idx = 10
        expected = (result["close"].iloc[idx] - result["close"].iloc[idx - 1]) / result["close"].iloc[idx - 1]
        actual = result["returns"].iloc[idx]
        assert abs(actual - expected) < 1e-10

    def test_log_returns_calculation_correct(self, raw_ohlcv):
        """log_returns = ln(close_t / close_{t-1})"""
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        idx = 10
        expected = np.log(result["close"].iloc[idx] / result["close"].iloc[idx - 1])
        actual = result["log_returns"].iloc[idx]
        assert abs(actual - expected) < 1e-10

    def test_candle_body_is_close_minus_open(self, raw_ohlcv):
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        idx = 15
        expected = result["close"].iloc[idx] - result["open"].iloc[idx]
        assert abs(result["body"].iloc[idx] - expected) < 1e-10

    def test_upper_wick_always_non_negative(self, raw_ohlcv):
        """Upper wick = high - max(open, close), should always be >= 0."""
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        assert (result["upper_wick"] >= -1e-10).all()  # Allow floating point tolerance

    def test_lower_wick_always_non_negative(self, raw_ohlcv):
        """Lower wick = min(open, close) - low, should always be >= 0."""
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        assert (result["lower_wick"] >= -1e-10).all()

    def test_range_is_high_minus_low(self, raw_ohlcv):
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        expected = result["high"] - result["low"]
        assert (abs(result["range"] - expected) < 1e-10).all()

    def test_is_bullish_flag(self, raw_ohlcv):
        """is_bullish = 1 if close > open, else 0."""
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        for i in range(10, 20):
            if result["close"].iloc[i] > result["open"].iloc[i]:
                assert result["is_bullish"].iloc[i] == 1
            else:
                assert result["is_bullish"].iloc[i] == 0

    def test_insufficient_data_raises(self):
        """Should raise if too few rows after processing."""
        pp = DataPreprocessor()
        small_df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [50, 60],
        }, index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))

        with pytest.raises(InsufficientDataError):
            pp.process(small_df, min_rows=50)

    def test_missing_columns_raises(self):
        """Should raise if required OHLCV columns are missing."""
        pp = DataPreprocessor()
        bad_df = pd.DataFrame({"close": [100, 101], "volume": [50, 60]},
                              index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))
        with pytest.raises(ValueError):
            pp.process(bad_df)

    def test_duplicates_removed(self):
        """Duplicate timestamps should be removed."""
        pp = DataPreprocessor()
        dates = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"])
        dates = dates.tz_localize("UTC")
        df = pd.DataFrame({
            "open": [100, 101, 102], "high": [105, 106, 107],
            "low": [95, 96, 97], "close": [103, 104, 105],
            "volume": [50, 60, 70],
        }, index=dates)
        result = pp.process(df, min_rows=2)
        assert len(result) == 2  # One duplicate removed

    def test_normalize_min_max(self, raw_ohlcv):
        """Min-max normalization should put values in [0, 1]."""
        pp = DataPreprocessor()
        result = pp.process(raw_ohlcv)
        normalized = pp.normalize(result, ["close", "volume"])
        assert normalized["close"].min() >= 0.0
        assert normalized["close"].max() <= 1.0
