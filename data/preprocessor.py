"""
Data preprocessor — cleans and normalizes raw market data.
All data passes through here before reaching indicators or ML models.
"""

import pandas as pd
import numpy as np
from loguru import logger
from core.exceptions import InsufficientDataError


class DataPreprocessor:
    """
    Cleans, validates, and normalizes OHLCV data.

    Steps:
    1. Validate required columns
    2. Remove duplicates
    3. Sort by timestamp
    4. Handle missing values
    5. Remove outliers (optional)
    6. Add derived columns (returns, log returns)
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def process(self, df: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            df: Raw OHLCV DataFrame
            min_rows: Minimum rows required after cleaning

        Returns:
            Cleaned DataFrame ready for analysis
        """
        df = df.copy()

        df = self._validate(df)
        df = self._remove_duplicates(df)
        df = self._sort(df)
        df = self._handle_missing(df)
        df = self._add_derived_columns(df)

        if len(df) < min_rows:
            raise InsufficientDataError(
                f"Only {len(df)} rows after preprocessing, minimum is {min_rows}"
            )

        logger.debug("Preprocessed {} rows of OHLCV data", len(df))
        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check required columns exist."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps."""
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        removed = before - len(df)
        if removed > 0:
            logger.debug("Removed {} duplicate rows", removed)
        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by timestamp ascending."""
        return df.sort_index()

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing OHLCV values.
        - Forward-fill close price
        - Use close for open/high/low if missing
        - Fill volume with 0
        """
        # Forward fill close price — BOUNDED: unlimited ffill silently
        # manufactures hours of flat "prices" after a feed outage (the
        # silent degradation claude.md forbids). Longer gaps stay NaN and
        # are dropped below.
        n_missing = int(df["close"].isna().sum())
        if n_missing:
            logger.warning(
                "Filling {} missing close value(s) (ffill, limit=3)", n_missing
            )
        df["close"] = df["close"].ffill(limit=3)

        # If open/high/low are missing, use close
        for col in ["open", "high", "low"]:
            df[col] = df[col].fillna(df["close"])

        # Volume missing = 0
        df["volume"] = df["volume"].fillna(0)

        # Drop rows where close is still NaN (start of data)
        df.dropna(subset=["close"], inplace=True)

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns for analysis."""
        # Reject zero/negative closes BEFORE log-returns: log(0) → -inf
        # poisons every downstream indicator (claude.md: zero/negative
        # prices must be rejected, never processed).
        bad = df["close"] <= 0
        if bad.any():
            logger.warning(
                "Dropping {} row(s) with non-positive close", int(bad.sum())
            )
            df = df[~bad].copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Candle properties
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["range"] = df["high"] - df["low"]
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # Rolling stats
        df["volatility_24h"] = df["returns"].rolling(24).std()

        return df

    def normalize(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Min-max normalize specific columns (for ML input).
        Returns normalized copy — does NOT modify original.
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max != col_min:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
        return df
