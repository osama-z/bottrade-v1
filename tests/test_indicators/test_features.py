"""FeatureEngineer tests — the two properties that, if they regress, silently
corrupt the ML model: no future leakage, and clean (NaN/inf-free) output.

FeatureEngineer feeds ai/ml_predictor. A leakage bug would inflate backtest
accuracy and mislead every downstream gate; a NaN/inf leak would poison XGBoost.
Both are cheap to pin and expensive to discover in production.
"""
import numpy as np
import pandas as pd
import pytest

from indicators.features import FeatureEngineer


def _frame(periods: int = 60, with_indicators: bool = True) -> pd.DataFrame:
    """A realistic OHLCV + preprocessed frame (returns/log_returns/candle stats)."""
    idx = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, periods)), index=idx)
    df = pd.DataFrame(index=idx)
    df["close"] = close
    df["open"] = close.shift(1).fillna(close.iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = rng.uniform(50, 150, periods)
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["body_pct"] = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    if with_indicators:
        df["RSI"] = rng.uniform(20, 80, periods)
        df["MACD"] = rng.normal(0, 1, periods)
        df["MACD_hist"] = rng.normal(0, 1, periods)
        df["volume_ratio"] = df["volume"] / df["volume"].mean()
    return df


class TestNoFutureLeakage:
    def test_target_is_the_future_direction(self):
        df = _frame()
        _, y = FeatureEngineer().build_features(df, target_periods=1)
        expected = (df["close"].shift(-1) > df["close"]).astype(int).reindex(y.index)
        assert (y == expected).all()

    def test_lag_feature_uses_only_the_past(self):
        # return_lag_1 at row i must equal returns at row i-1, never a future row.
        df = _frame()
        X, _ = FeatureEngineer().build_features(df, target_periods=1)
        assert "return_lag_1" in X.columns
        for ts in X.index:
            pos = df.index.get_loc(ts)
            assert X.loc[ts, "return_lag_1"] == pytest.approx(df["returns"].iloc[pos - 1])

    def test_target_periods_shifts_further(self):
        df = _frame()
        _, y = FeatureEngineer().build_features(df, target_periods=3)
        expected = (df["close"].shift(-3) > df["close"]).astype(int).reindex(y.index)
        assert (y == expected).all()


class TestCleanOutput:
    def test_no_nan_or_inf_reaches_the_matrix(self):
        X, y = FeatureEngineer().build_features(_frame())
        arr = X.to_numpy(dtype=float)
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()
        assert not y.isna().any()

    def test_x_and_y_are_index_aligned(self):
        X, y = FeatureEngineer().build_features(_frame())
        assert X.index.equals(y.index)
        assert len(X) == len(y) > 0

    def test_zero_volume_window_does_not_leak_inf(self):
        # 0 / rolling-mean-0 = NaN (dropped), never +inf — the model never sees inf.
        df = _frame()
        df["volume"] = 0.0
        X, _ = FeatureEngineer().build_features(df)
        assert not np.isinf(X.to_numpy(dtype=float)).any()


class TestGracefulDegradation:
    def test_missing_indicator_does_not_empty_the_matrix(self):
        # Regression: an absent indicator used to create an all-NaN lag column that
        # survived into feature_cols, so dropna() wiped every row (0 samples).
        df = _frame(with_indicators=False)
        X, y = FeatureEngineer().build_features(df)
        assert len(X) == len(y) > 0            # not silently emptied
        assert "rsi_lag_1" not in X.columns    # absent indicator is skipped, not poisoned

    def test_time_features_present_for_datetime_index(self):
        X, _ = FeatureEngineer().build_features(_frame())
        for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            assert col in X.columns
