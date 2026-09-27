"""
Tests for MLPredictor — verifies training, prediction, save/load cycle, and edge cases.
Uses synthetic indicator data — no real market data or API needed.
"""

import pytest
import numpy as np
import pandas as pd

from ai.ml_predictor import MLPredictor, PredictionResult, TrainingResult
from config.constants import Signal


@pytest.fixture
def synthetic_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic X, y for training.
    400 samples, 10 features, random binary labels.
    """
    np.random.seed(42)
    n = 400
    X = pd.DataFrame(
        np.random.randn(n, 10),
        columns=[f"feature_{i}" for i in range(10)],
    )
    # Slightly learnable target: up if first feature > 0
    y = pd.Series((X["feature_0"] > 0).astype(int), name="target")
    return X, y


@pytest.fixture
def trained_predictor(synthetic_features, tmp_path, monkeypatch) -> MLPredictor:
    """Train a predictor using temp directory for model storage."""
    X, y = synthetic_features
    monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
    predictor = MLPredictor(pair="TEST/USDT")
    predictor.model_path = tmp_path / "xgb_TEST_USDT.joblib"
    predictor.scaler_path = tmp_path / "scaler_TEST_USDT.joblib"
    predictor.train(X, y)
    return predictor


class TestMLPredictorTraining:
    """Test model training."""

    def test_train_returns_training_result(self, synthetic_features, tmp_path, monkeypatch) -> None:
        X, y = synthetic_features
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="TEST/USDT")
        predictor.model_path = tmp_path / "model.joblib"
        predictor.scaler_path = tmp_path / "scaler.joblib"
        result = predictor.train(X, y)
        assert isinstance(result, TrainingResult)

    def test_train_accuracy_above_chance(self, synthetic_features, tmp_path, monkeypatch) -> None:
        """With a learnable target, accuracy should exceed 50%."""
        X, y = synthetic_features
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="TEST/USDT")
        predictor.model_path = tmp_path / "model.joblib"
        predictor.scaler_path = tmp_path / "scaler.joblib"
        result = predictor.train(X, y)
        assert result.accuracy > 0.50, f"Expected > 50% accuracy, got {result.accuracy:.1%}"

    def test_train_saves_model_file(self, synthetic_features, tmp_path, monkeypatch) -> None:
        X, y = synthetic_features
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="TEST/USDT")
        model_path = tmp_path / "xgb_TEST_USDT.joblib"
        predictor.model_path = model_path
        predictor.scaler_path = tmp_path / "scaler.joblib"
        predictor.train(X, y)
        assert model_path.exists(), "Model file should be saved to disk"

    def test_training_result_has_correct_sample_count(self, synthetic_features, tmp_path, monkeypatch) -> None:
        X, y = synthetic_features
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="TEST/USDT")
        predictor.model_path = tmp_path / "model.joblib"
        predictor.scaler_path = tmp_path / "scaler.joblib"
        result = predictor.train(X, y)
        assert result.n_samples == len(X)
        assert result.n_features == len(X.columns)

    def test_auc_roc_between_0_and_1(self, synthetic_features, tmp_path, monkeypatch) -> None:
        X, y = synthetic_features
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="TEST/USDT")
        predictor.model_path = tmp_path / "model.joblib"
        predictor.scaler_path = tmp_path / "scaler.joblib"
        result = predictor.train(X, y)
        assert 0.0 <= result.auc_roc <= 1.0


class TestMLPredictorPrediction:
    """Test model prediction output."""

    def test_predict_returns_prediction_result(self, trained_predictor, synthetic_features) -> None:
        X, _ = synthetic_features
        result = trained_predictor.predict(X)
        assert isinstance(result, PredictionResult)

    def test_predict_probabilities_sum_to_one(self, trained_predictor, synthetic_features) -> None:
        X, _ = synthetic_features
        result = trained_predictor.predict(X)
        assert abs(result.p_up + result.p_down - 1.0) < 1e-6, (
            f"p_up + p_down should = 1.0, got {result.p_up + result.p_down}"
        )

    def test_predict_probabilities_in_range(self, trained_predictor, synthetic_features) -> None:
        X, _ = synthetic_features
        result = trained_predictor.predict(X)
        assert 0.0 <= result.p_up <= 1.0
        assert 0.0 <= result.p_down <= 1.0

    def test_predict_signal_is_valid(self, trained_predictor, synthetic_features) -> None:
        X, _ = synthetic_features
        result = trained_predictor.predict(X)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_predict_score_equals_p_up_minus_p_down(self, trained_predictor, synthetic_features) -> None:
        X, _ = synthetic_features
        result = trained_predictor.predict(X)
        expected_score = round(result.p_up - result.p_down, 4)
        assert abs(result.score - expected_score) < 1e-4

    def test_predict_confidence_between_0_and_1(self, trained_predictor, synthetic_features) -> None:
        X, _ = synthetic_features
        result = trained_predictor.predict(X)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_no_model_returns_neutral(self, tmp_path, monkeypatch) -> None:
        """Predictor without a saved model returns neutral prediction."""
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="NOTRAINED/USDT")
        predictor.model_path = tmp_path / "nonexistent.joblib"
        predictor.scaler_path = tmp_path / "nonexistent_scaler.joblib"
        result = predictor.predict(pd.DataFrame({"a": [1, 2, 3]}))
        assert result.signal == Signal.HOLD
        assert result.error is not None


class TestMLPredictorSaveLoad:
    """Test model persistence."""

    def test_save_and_load_cycle(self, synthetic_features, tmp_path, monkeypatch) -> None:
        """Train, save, load in a new predictor, and predict successfully."""
        X, y = synthetic_features
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)

        # Train and save
        predictor1 = MLPredictor(pair="TEST/USDT")
        predictor1.model_path = tmp_path / "model.joblib"
        predictor1.scaler_path = tmp_path / "scaler.joblib"
        predictor1.train(X, y)

        # Load in new instance
        predictor2 = MLPredictor(pair="TEST/USDT")
        predictor2.model_path = tmp_path / "model.joblib"
        predictor2.scaler_path = tmp_path / "scaler.joblib"
        loaded = predictor2.load()

        assert loaded is True
        assert predictor2.is_trained()
        result = predictor2.predict(X)
        assert isinstance(result, PredictionResult)
        assert result.error is None

    def test_load_returns_false_if_no_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("ai.ml_predictor.MODEL_DIR", tmp_path)
        predictor = MLPredictor(pair="GHOST/USDT")
        predictor.model_path = tmp_path / "ghost.joblib"
        loaded = predictor.load()
        assert loaded is False
