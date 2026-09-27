"""Stage 3 tests — HMM RegimeDetector.

All tests use synthetic data — no exchange connections, no network.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from ai.regime_detector import RegimeDetector


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with distinct regimes baked in.

    First 1/3: uptrend (positive drift)
    Middle 1/3: choppy (zero drift, high vol)
    Last 1/3: downtrend (negative drift)
    """
    rng = np.random.default_rng(seed)
    n1 = rows // 3
    n2 = rows // 3
    n3 = rows - n1 - n2

    # Uptrend: positive drift, moderate vol
    r1 = rng.normal(0.002, 0.005, n1)
    # Choppy: zero drift, high vol
    r2 = rng.normal(0.0, 0.012, n2)
    # Downtrend: negative drift, moderate vol
    r3 = rng.normal(-0.002, 0.005, n3)

    returns = np.concatenate([r1, r2, r3])
    close = 50_000.0 * np.cumprod(1.0 + returns)

    spread = rng.uniform(0.001, 0.005, rows)
    high = close * (1.0 + spread)
    low = close * (1.0 - spread * rng.uniform(0.6, 1.2, rows))

    idx = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": np.roll(close, 1), "high": high, "low": low, "close": close,
         "volume": rng.uniform(100, 5000, rows)},
        index=idx,
    )


# ─── Feature Extraction ──────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_extracts_two_columns(self) -> None:
        df = _make_ohlcv(100)
        features = RegimeDetector.extract_features(df)
        assert "log_return" in features.columns
        assert "log_range" in features.columns

    def test_log_return_is_correct(self) -> None:
        df = _make_ohlcv(50)
        features = RegimeDetector.extract_features(df)
        # log_return[1] = ln(close[1] / close[0])
        expected = np.log(df["close"].iloc[1] / df["close"].iloc[0])
        assert features["log_return"].iloc[1] == pytest.approx(expected, abs=1e-10)

    def test_log_range_always_positive(self) -> None:
        df = _make_ohlcv(200)
        features = RegimeDetector.extract_features(df)
        assert (features["log_range"] > 0).all()

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="high"):
            RegimeDetector.extract_features(df)


# ─── Fit / Train ──────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_requires_minimum_data(self) -> None:
        df = _make_ohlcv(100)
        detector = RegimeDetector(pair="TEST/USDT")
        with pytest.raises(ValueError, match="at least 500"):
            detector.fit(df)

    def test_fit_succeeds_with_enough_data(self) -> None:
        df = _make_ohlcv(1000)
        detector = RegimeDetector(pair="TEST/USDT")
        detector.fit(df)
        assert detector.model is not None

    def test_state_map_has_three_entries(self) -> None:
        df = _make_ohlcv(1000)
        detector = RegimeDetector(pair="TEST/USDT")
        detector.fit(df)
        assert len(detector._state_map) == 3
        # All sorted states must be present: {0, 1, 2}
        assert set(detector._state_map.values()) == {0, 1, 2}


# ─── State Label Stability ────────────────────────────────────────────────────

class TestLabelStability:
    def test_states_sorted_by_mean_return(self) -> None:
        """After fitting, state 0 must have the lowest mean return and
        state 2 the highest — regardless of HMM internal label assignment."""
        df = _make_ohlcv(1500, seed=7)
        detector = RegimeDetector(pair="TEST/USDT")
        detector.fit(df)

        inv_map = {v: k for k, v in detector._state_map.items()}
        mean_bear = detector.model.means_[inv_map[0], 0]
        mean_neut = detector.model.means_[inv_map[1], 0]
        mean_bull = detector.model.means_[inv_map[2], 0]

        assert mean_bear < mean_neut < mean_bull, (
            f"State means not sorted: bear={mean_bear:.6f}, "
            f"neut={mean_neut:.6f}, bull={mean_bull:.6f}"
        )

    def test_labels_stable_across_retrains(self) -> None:
        """Training twice on the same data with different random_state
        must produce the same label ordering (Bear < Neutral < Bull)."""
        df = _make_ohlcv(1500, seed=7)

        det_a = RegimeDetector(pair="TEST/USDT", random_state=42)
        det_a.fit(df)
        inv_a = {v: k for k, v in det_a._state_map.items()}
        means_a = [det_a.model.means_[inv_a[i], 0] for i in range(3)]

        det_b = RegimeDetector(pair="TEST/USDT", random_state=99)
        det_b.fit(df)
        inv_b = {v: k for k, v in det_b._state_map.items()}
        means_b = [det_b.model.means_[inv_b[i], 0] for i in range(3)]

        # Both must have bear < neutral < bull ordering
        assert means_a[0] < means_a[1] < means_a[2]
        assert means_b[0] < means_b[1] < means_b[2]


# ─── Prediction ───────────────────────────────────────────────────────────────

class TestPredict:
    def _fitted_detector(self) -> RegimeDetector:
        df = _make_ohlcv(1000)
        det = RegimeDetector(pair="TEST/USDT")
        det.fit(df)
        return det

    def test_predict_returns_valid_state(self) -> None:
        det = self._fitted_detector()
        df = _make_ohlcv(100, seed=99)
        result = det.predict(df)
        assert result["state"] in (0, 1, 2)
        assert result["name"] in ("bearish", "neutral/choppy", "bullish")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_without_model_returns_neutral(self) -> None:
        det = RegimeDetector(pair="TEST/USDT")
        df = _make_ohlcv(100)
        result = det.predict(df)
        assert result["state"] == 1
        assert result["name"] == "neutral/choppy"

    def test_low_confidence_falls_back_to_neutral(self) -> None:
        """With an absurdly high threshold, every prediction should fall back."""
        det = self._fitted_detector()
        df = _make_ohlcv(100, seed=99)
        result = det.predict(df, confidence_threshold=0.9999)
        assert result["state"] == 1
        assert result["name"] == "neutral/choppy"

    def test_predict_series_length_matches_input(self) -> None:
        det = self._fitted_detector()
        df = _make_ohlcv(200, seed=99)
        series = det.predict_series(df)
        assert len(series) == len(df)
        assert set(series.unique()).issubset({0, 1, 2})


# ─── Save / Load (Serialization) ─────────────────────────────────────────────

class TestSerialization:
    def test_save_and_load_preserves_predictions(self) -> None:
        """Saving and loading the model must produce identical predictions."""
        df_train = _make_ohlcv(1000, seed=42)
        df_test = _make_ohlcv(100, seed=99)

        with tempfile.TemporaryDirectory() as tmpdir:
            det1 = RegimeDetector(pair="TEST/USDT", model_dir=tmpdir)
            det1.fit(df_train)
            pred1 = det1.predict(df_test)
            det1.save()

            det2 = RegimeDetector(pair="TEST/USDT", model_dir=tmpdir)
            loaded = det2.load()
            assert loaded is True
            pred2 = det2.predict(df_test)

            assert pred1["state"] == pred2["state"]
            assert pred1["confidence"] == pytest.approx(pred2["confidence"])

    def test_load_returns_false_when_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            det = RegimeDetector(pair="TEST/USDT", model_dir=tmpdir)
            assert det.load() is False

    def test_save_without_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            det = RegimeDetector(pair="TEST/USDT", model_dir=tmpdir)
            with pytest.raises(ValueError, match="No model"):
                det.save()


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe_predict_returns_neutral(self) -> None:
        det = RegimeDetector(pair="TEST/USDT")
        df_train = _make_ohlcv(1000)
        det.fit(df_train)
        # Empty DataFrame
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = det.predict(empty)
        assert result["state"] == 1

    def test_single_row_predict_returns_neutral(self) -> None:
        det = RegimeDetector(pair="TEST/USDT")
        df_train = _make_ohlcv(1000)
        det.fit(df_train)
        # Single row — log_return will be NaN, features will be empty after dropna
        single = _make_ohlcv(1000).iloc[:1]
        result = det.predict(single)
        assert result["state"] == 1
