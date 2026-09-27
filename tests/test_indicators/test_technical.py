"""
Tests for TechnicalIndicators module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from indicators.technical import TechnicalIndicators


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 300 rows of realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="1h",
    )

    price = 50000.0
    prices = [price]
    for _ in range(n - 1):
        change = np.random.normal(0, 0.005)
        price *= (1 + change)
        prices.append(price)

    prices = np.array(prices)

    df = pd.DataFrame({
        "open": prices * np.random.uniform(0.998, 1.002, n),
        "high": prices * np.random.uniform(1.001, 1.015, n),
        "low": prices * np.random.uniform(0.985, 0.999, n),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    }, index=dates)

    return df


class TestTechnicalIndicators:

    def test_compute_all_returns_dataframe(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_rsi_in_range(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        assert "RSI" in result.columns
        rsi = result["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bollinger_bands_logic(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        assert "BB_upper" in result.columns
        assert "BB_lower" in result.columns
        assert "BB_mid" in result.columns
        valid = result[["BB_upper", "BB_mid", "BB_lower"]].dropna()
        assert (valid["BB_upper"] >= valid["BB_mid"]).all()
        assert (valid["BB_mid"] >= valid["BB_lower"]).all()

    def test_macd_columns_present(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        assert "MACD" in result.columns
        assert "MACD_signal" in result.columns
        assert "MACD_hist" in result.columns

    def test_supertrend_adx_columns_survive_short_frame(self, sample_ohlcv):
        """R1: on a frame too short for pandas-ta to compute Supertrend/ADX,
        the columns must still exist (as NaN) rather than vanish — a missing
        Supertrend_dir/ADX makes trend_following silently stop trading."""
        ti = TechnicalIndicators()
        short = sample_ohlcv.head(6)  # shorter than Supertrend/ADX warm-up
        result = ti.add_trend_indicators(short.copy())
        for col in ("Supertrend", "Supertrend_dir", "ADX"):
            assert col in result.columns, f"{col} column vanished on short frame"

    def test_volatility_columns_survive_short_frame(self, sample_ohlcv):
        """BB/KC columns must exist even when pandas-ta returns None (short
        frame): compute_all's BB_squeeze line and dropna read them, so a
        vanished column would KeyError instead of warming up."""
        ti = TechnicalIndicators()
        result = ti.add_volatility_indicators(sample_ohlcv.head(6).copy())
        for col in ("BB_lower", "BB_mid", "BB_upper", "BB_width", "BB_pct",
                    "KC_lower", "KC_mid", "KC_upper"):
            assert col in result.columns, f"{col} column vanished on short frame"

    def test_macd_columns_survive_short_frame(self, sample_ohlcv):
        """MACD columns must exist even when the frame is too short for MACD
        but long enough for RSI (the MACD_cross line reads them)."""
        ti = TechnicalIndicators()
        # 20 rows: RSI(14) computes, MACD(26) returns None → else branch fires.
        result = ti.add_momentum_indicators(sample_ohlcv.head(20).copy())
        for col in ("MACD", "MACD_signal", "MACD_hist", "MACD_cross"):
            assert col in result.columns, f"{col} column vanished on short frame"

    def test_get_latest_signals_returns_dict(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        signals = ti.get_latest_signals(result)
        assert isinstance(signals, dict)
        assert "price" in signals
        assert "rsi" in signals
        assert "macd" in signals
        assert "composite_score" in signals

    def test_composite_score_in_range(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.compute_all(sample_ohlcv)
        scores = result["composite_score"].dropna()
        assert (scores >= -1).all() and (scores <= 1).all()
