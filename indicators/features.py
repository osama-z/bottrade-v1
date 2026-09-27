"""
Feature engineering for ML models.
Takes raw OHLCV + indicators DataFrame and produces ML-ready features.

Key principles:
- No future leakage (only use data available at prediction time)
- Normalize/encode all features
- Add lag features for time-series context
"""

import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    """
    Transforms OHLCV + indicator DataFrame into ML features.

    Output features include:
    - Lagged price returns (1, 3, 7, 14 periods)
    - Lagged indicator values
    - Time features (hour, day of week)
    - Rolling statistics
    - Categorical features (bullish/bearish candle)
    """

    def build_features(
        self,
        df: pd.DataFrame,
        target_periods: int = 1,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix X and target vector y.

        Args:
            df: DataFrame with OHLCV + all indicators
            target_periods: How many periods ahead to predict direction

        Returns:
            X: Feature DataFrame
            y: Target Series (1=price up, 0=price down)
        """
        df = df.copy()

        # ── Target variable ─────────────────────────────────────────────────────
        # 1 if price goes up after `target_periods` candles, 0 if down
        df["target"] = (df["close"].shift(-target_periods) > df["close"]).astype(int)

        # ── Lag features ────────────────────────────────────────────────────────
        # Only lag indicators that are actually present. Creating an all-NaN
        # column for an absent indicator would survive into feature_cols and make
        # the final dropna() wipe out every row (silent empty matrix).
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)
            if "RSI" in df.columns:
                df[f"rsi_lag_{lag}"] = df["RSI"].shift(lag)
            if "MACD" in df.columns:
                df[f"macd_lag_{lag}"] = df["MACD"].shift(lag)

        # ── Time features (cyclical encoding) ───────────────────────────────────
        if hasattr(df.index, "hour"):
            df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # ── Rolling statistics ───────────────────────────────────────────────────
        df["rolling_return_7"] = df["returns"].rolling(7).mean()
        df["rolling_volatility_7"] = df["returns"].rolling(7).std()
        df["rolling_return_14"] = df["returns"].rolling(14).mean()
        df["rolling_volume_ratio"] = (
            df["volume"] / df["volume"].rolling(14).mean()
        )

        # ── Select feature columns ────────────────────────────────────────────────
        feature_cols = [
            # Price-based
            "returns", "log_returns", "body_pct", "upper_wick", "lower_wick",
            "range", "is_bullish",

            # Lag returns
            *[f"return_lag_{lag}" for lag in [1, 2, 3, 5, 7, 14]],
            *[f"rsi_lag_{lag}" for lag in [1, 2, 3, 5, 7, 14]],
            *[f"macd_lag_{lag}" for lag in [1, 2, 3, 5, 7, 14]],

            # Indicators
            "RSI", "MACD", "MACD_hist", "MACD_cross",
            "BB_pct", "BB_width", "BB_squeeze",
            "ATR_pct", "CCI", "Williams_R", "ROC",
            "Stoch_K", "Stoch_D",
            "volume_ratio", "MFI", "OBV_trend",
            "trend_score", "composite_score",
            "trend_200", "SMA_cross",

            # Rolling
            "rolling_return_7", "rolling_volatility_7",
            "rolling_return_14", "rolling_volume_ratio",
        ]

        # Time features (only if available)
        time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        for col in time_cols:
            if col in df.columns:
                feature_cols.append(col)

        # Keep only columns that actually exist in df
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Drop rows with NaN in features or target
        df_clean = df[feature_cols + ["target"]].dropna()

        X = df_clean[feature_cols]
        y = df_clean["target"]

        logger.debug(
            "Built feature matrix: {} samples × {} features",
            len(X), len(feature_cols)
        )

        return X, y
