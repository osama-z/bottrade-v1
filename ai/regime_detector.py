"""Hidden Markov Model (HMM) Regime Detector — Stage 3.

Uses GaussianHMM from hmmlearn to identify 3 distinct market regimes:
- State 0: Bearish (lowest mean return)
- State 1: Neutral / Choppy (middle mean return)
- State 2: Bullish (highest mean return)

To ensure state labels are consistent across model retrains, they are sorted
by their mean log returns in ascending order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger


class RegimePrediction(TypedDict):
    state: int
    name: str
    confidence: float


class RegimeDetector:
    """Detects and predicts market regimes using a Gaussian Hidden Markov Model."""

    def __init__(
        self,
        pair: str = "BTC/USDT",
        n_components: int = 3,
        random_state: int = 42,
        model_dir: str | Path | None = None,
    ) -> None:
        self.pair = pair
        self.n_components = n_components
        self.random_state = random_state

        if model_dir is None:
            # Locate default model directory: project_root/ai/models
            self.model_dir = Path(__file__).resolve().parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"regime_hmm_{self.pair.replace('/', '_')}.joblib"

        self.model: hmm.GaussianHMM | None = None
        self._state_map: dict[int, int] = {}  # model state -> sorted state (0: Bear, 1: Choppy, 2: Bull)
        self._state_names = {
            0: "bearish",
            1: "neutral/choppy",
            2: "bullish",
        }

    # ── Feature Engineering ───────────────────────────────────────────────────

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features required for HMM regime detection.

        Features:
        1. log_return = ln(close_t / close_{t-1})
        2. log_range = ln(high_t / low_t) (proxy for volatility)

        Returns:
            DataFrame with columns 'log_return' and 'log_range'.
        """
        # Guard against missing columns
        for col in ["high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain column '{col}'")

        log_return = np.log(df["close"] / df["close"].shift(1))
        # Guard against zero-division (e.g. flat illiquid sessions)
        log_range = np.log(np.maximum(df["high"] / np.maximum(df["low"], 1e-8), 1.0001))

        features = pd.DataFrame(
            {"log_return": log_return, "log_range": log_range},
            index=df.index,
        )
        return features

    # ── Fit / Train ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """Fit Gaussian HMM model to the historical OHLCV data.

        Handles NaN drops automatically.
        """
        features_df = self.extract_features(df).dropna()

        if len(features_df) < 500:
            raise ValueError(
                f"Insufficient data to train HMM: need at least 500 rows, got {len(features_df)}"
            )

        X = features_df.values

        model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",    # 'diag' is more stable than 'full' for 2 features
            random_state=self.random_state,
            n_iter=200,
            tol=1e-4,
        )
        model.fit(X)

        self.model = model
        self._compute_state_map()
        logger.info(
            "Trained HMM regime model for {} | log-likelihood: {:.4f}",
            self.pair,
            model.score(X),
        )

    def _compute_state_map(self) -> None:
        """Compute state map to sort states by mean return.

        Map: model state -> sorted index
        - Sorted index 0: Bearish (lowest mean return)
        - Sorted index 1: Neutral/Choppy (middle mean return)
        - Sorted index 2: Bullish (highest mean return)
        """
        if self.model is None:
            return

        # model.means_ has shape (n_components, n_features)
        # Feature 0 is log_return
        means = self.model.means_[:, 0]
        sorted_indices = np.argsort(means)

        # Map original state labels to sorted index
        self._state_map = {orig: int(new) for new, orig in enumerate(sorted_indices)}
        logger.debug("HMM state mapping computed: {}", self._state_map)
        for orig, new in self._state_map.items():
            logger.debug(
                "State {} ({}) -> Mean Log Return: {:.6f}, Mean Volatility: {:.6f}",
                new,
                self._state_names[new],
                self.model.means_[orig, 0],
                self.model.means_[orig, 1],
            )

    # ── Predict / Query ───────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.60,
    ) -> RegimePrediction:
        """Predict the current regime based on the latest candle in df.

        If the model is not trained/loaded, returns a default neutral state.
        If confidence is below confidence_threshold, returns "neutral/choppy".

        Returns:
            RegimePrediction containing:
            - state: 0 (Bear), 1 (Neutral), 2 (Bull)
            - name: state name
            - confidence: probability of the active state
        """
        default_prediction: RegimePrediction = {
            "state": 1,
            "name": "neutral/choppy",
            "confidence": 1.0,
        }

        if self.model is None:
            return default_prediction

        try:
            features_df = self.extract_features(df).dropna()
            if features_df.empty:
                return default_prediction

            # HMM predicts using the full sequence up to current
            X = features_df.values
            posteriors = self.model.predict_proba(X)
            latest_posterior = posteriors[-1]

            raw_state = int(np.argmax(latest_posterior))
            confidence = float(latest_posterior[raw_state])

            sorted_state = self._state_map.get(raw_state, 1)

            if confidence < confidence_threshold:
                logger.debug(
                    "Regime confidence {:.2f} < threshold {:.2f} — falling back to neutral/choppy",
                    confidence,
                    confidence_threshold,
                )
                return default_prediction

            return {
                "state": sorted_state,
                "name": self._state_names[sorted_state],
                "confidence": confidence,
            }

        except Exception as e:
            logger.error("HMM regime prediction failed: {}", e)
            return default_prediction

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Predict the regime sequence for the entire DataFrame.

        Useful for backtesting.

        Returns:
            Series of integer states (0, 1, 2) aligned with df.index.
        """
        if self.model is None:
            return pd.Series(1, index=df.index)

        try:
            features_df = self.extract_features(df)
            clean_features = features_df.dropna()
            if clean_features.empty:
                return pd.Series(1, index=df.index)

            # Predict states
            raw_states = self.model.predict(clean_features.values)
            sorted_states = [self._state_map.get(s, 1) for s in raw_states]

            result = pd.Series(1, index=df.index, dtype=int)
            result.loc[clean_features.index] = sorted_states
            return result

        except Exception as e:
            logger.error("HMM regime series prediction failed: {}", e)
            return pd.Series(1, index=df.index)

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Serialize HMM model and state map to disk."""
        if self.model is None:
            raise ValueError("No model trained to save")

        data = {
            "model": self.model,
            "state_map": self._state_map,
            "pair": self.pair,
            "n_components": self.n_components,
        }
        joblib.dump(data, self.model_path)
        logger.info("Saved HMM model to {}", self.model_path)

    def load(self) -> bool:
        """Load HMM model and state map from disk.

        Returns:
            True if loaded successfully, False if file doesn't exist.
        """
        if not self.model_path.exists():
            logger.debug("HMM model file not found at {}", self.model_path)
            return False

        try:
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self._state_map = data["state_map"]
            logger.info("Loaded HMM model from {}", self.model_path)
            return True
        except Exception as e:
            logger.error("Failed to load HMM model from {}: {}", self.model_path, e)
            return False
