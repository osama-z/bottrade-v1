"""
XGBoost ML Price Predictor — predicts short-term price direction.

Pipeline:
    Training:
        1. Load OHLCV + indicators DataFrame (from Phase 1 pipeline)
        2. Build feature matrix X and target y using FeatureEngineer
        3. Train XGBoost classifier with cross-validation
        4. Evaluate accuracy, precision, recall, AUC
        5. Save trained model to ai/models/

    Prediction:
        1. Load saved model
        2. Take latest row of indicator features
        3. Return P(up), P(down), and signal

No API key needed — runs entirely locally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ai.triple_barrier import triple_barrier_labels

try:
    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        roc_auc_score, classification_report,
    )
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("xgboost/sklearn not installed — ML predictor unavailable")

from config.constants import Signal

# Model save directory
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResult:
    """Result from a model training run."""
    accuracy: float
    precision: float
    recall: float
    auc_roc: float
    n_samples: int
    n_features: int
    feature_names: list[str]
    model_path: str

    def print_report(self) -> None:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║           XGBoost Training Report                        ║
╠══════════════════════════════════════════════════════════╣
║  Samples:    {self.n_samples:<10}                              ║
║  Features:   {self.n_features:<10}                              ║
║  Accuracy:   {self.accuracy:<10.1%}                              ║
║  Precision:  {self.precision:<10.1%}                              ║
║  Recall:     {self.recall:<10.1%}                              ║
║  AUC-ROC:    {self.auc_roc:<10.3f}                              ║
║  Model:      {Path(self.model_path).name:<38}  ║
╚══════════════════════════════════════════════════════════╝
        """)


@dataclass
class PredictionResult:
    """Result from a single model prediction."""
    p_up: float           # Probability price goes UP (0.0–1.0)
    p_down: float         # Probability price goes DOWN (0.0–1.0)
    score: float          # Normalized score: p_up - p_down → (-1, +1)
    signal: Signal
    confidence: float     # Max(p_up, p_down) — how sure the model is
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "p_up": self.p_up,
            "p_down": self.p_down,
            "score": self.score,
            "signal": self.signal.value,
            "confidence": self.confidence,
        }


class MLPredictor:
    """
    XGBoost-based price direction predictor.

    Predicts whether price will be higher or lower after `target_periods` candles.
    Outputs a probability and a BUY/SELL/HOLD signal.

    Usage:
        predictor = MLPredictor(pair="BTC/USDT")

        # Training (one-time)
        result = predictor.train(df_with_indicators)

        # Prediction (every candle)
        pred = predictor.predict(df_with_indicators)
        print(pred.signal, pred.p_up)
    """

    # Signal thresholds based on predicted probability
    BUY_THRESHOLD = 0.55     # P(up) > 55% → BUY
    SELL_THRESHOLD = 0.55    # P(down) > 55% → SELL

    def __init__(
        self,
        pair: str = "BTC/USDT",
        model_name: Optional[str] = None,
    ) -> None:
        self.pair = pair
        self._safe_pair = pair.replace("/", "_")
        self.model_name = model_name or f"xgb_{self._safe_pair}.joblib"
        self.model_path = MODEL_DIR / self.model_name
        self.scaler_path = MODEL_DIR / f"scaler_{self._safe_pair}.joblib"
        self.features_path = MODEL_DIR / f"features_{self._safe_pair}.joblib"

        self._model: Optional[object] = None
        self._scaler: Optional[object] = None
        self._feature_names: list[str] = []

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> TrainingResult:
        """
        Train XGBoost classifier on feature matrix X and target y.

        Args:
            X: Feature DataFrame (from FeatureEngineer.build_features)
            y: Target Series (1=up, 0=down)
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility

        Returns:
            TrainingResult with accuracy and model path
        """
        if not ML_AVAILABLE:
            raise RuntimeError("xgboost and scikit-learn are required for training")

        logger.info("Training XGBoost on {} samples × {} features", len(X), len(X.columns))

        self._feature_names = list(X.columns)

        # ── Train/test split — time-based (no shuffling to avoid leakage) ────
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # ── Scale features ──────────────────────────────────────────────────
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        # ── XGBoost model ────────────────────────────────────────────────────
        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

        # Train with early stopping on eval set
        self._model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        # ── Evaluation ───────────────────────────────────────────────────────
        y_pred = self._model.predict(X_test_scaled)
        y_prob = self._model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        logger.info(
            "XGBoost trained — accuracy={:.1%} precision={:.1%} recall={:.1%} AUC={:.3f}",
            accuracy, precision, recall, auc
        )
        logger.debug("\n{}", classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

        # ── Save model and scaler ─────────────────────────────────────────────
        joblib.dump(self._model, self.model_path)
        joblib.dump(self._scaler, self.scaler_path)
        joblib.dump(self._feature_names, self.features_path)
        logger.info("Model saved to {}", self.model_path)

        return TrainingResult(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            auc_roc=round(auc, 4),
            n_samples=len(X),
            n_features=len(X.columns),
            feature_names=self._feature_names,
            model_path=str(self.model_path),
        )

    # ─── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load saved model and scaler from disk.

        Returns:
            True if loaded successfully, False if model file not found
        """
        if not ML_AVAILABLE:
            return False

        if not self.model_path.exists():
            logger.warning("No saved model found at {} — train first", self.model_path)
            return False

        self._model = joblib.load(self.model_path)
        if self.scaler_path.exists():
            self._scaler = joblib.load(self.scaler_path)
        if self.features_path.exists():
            self._feature_names = joblib.load(self.features_path)
        logger.info("Model loaded from {} ({} features)", self.model_path, len(self._feature_names))
        return True

    def is_trained(self) -> bool:
        """Check if model is loaded and ready to predict."""
        return self._model is not None

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """
        Predict price direction from feature row(s).

        Args:
            X: Feature DataFrame — uses the LAST row for prediction.
               Must contain the same columns as used during training.

        Returns:
            PredictionResult with probabilities and signal
        """
        if not self.is_trained():
            loaded = self.load()
            if not loaded:
                return self._neutral_prediction("Model not trained — run train() first")

        if not ML_AVAILABLE:
            return self._neutral_prediction("xgboost not installed")

        try:
            # Use only the feature columns the model was trained on
            if self._feature_names:
                available = [c for c in self._feature_names if c in X.columns]
                X = X[available]

            # Take the last valid row
            latest = X.dropna().iloc[[-1]]  # Keep as 2D for scaler

            if self._scaler is not None:
                latest_scaled = self._scaler.transform(latest)
            else:
                latest_scaled = latest.values

            proba = self._model.predict_proba(latest_scaled)[0]
            p_down, p_up = float(proba[0]), float(proba[1])
            score = round(p_up - p_down, 4)

            if p_up >= self.BUY_THRESHOLD:
                signal = Signal.BUY
                confidence = p_up
            elif p_down >= self.SELL_THRESHOLD:
                signal = Signal.SELL
                confidence = p_down
            else:
                signal = Signal.HOLD
                confidence = max(p_up, p_down)

            logger.debug(
                "XGBoost prediction: p_up={:.1%} p_down={:.1%} → {}",
                p_up, p_down, signal.value
            )

            return PredictionResult(
                p_up=round(p_up, 4),
                p_down=round(p_down, 4),
                score=score,
                signal=signal,
                confidence=round(confidence, 4),
            )

        except Exception as e:
            logger.error("Prediction failed: {}", e)
            return self._neutral_prediction(str(e))

    def get_feature_importance(self, top_n: int = 10) -> dict[str, float]:
        """Return top N most important features for the trained model."""
        if not self.is_trained() or not hasattr(self._model, "feature_importances_"):
            return {}

        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]

        importance_dict = dict(zip(names, importances))
        return dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

    def _neutral_prediction(self, error: str = "") -> PredictionResult:
        return PredictionResult(
            p_up=0.5,
            p_down=0.5,
            score=0.0,
            signal=Signal.HOLD,
            confidence=0.0,
            error=error,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Triple-Barrier + Meta-Labelling predictor (Roadmap Task 1.5)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class MetaPrediction:
    """Primary side + meta-label decision for one bar.

    ``take`` gates trading and ``size`` (0 when not taken) is the position
    fraction — you trade ONLY when the meta-label says the bet is worth taking.
    """
    side: int              # +1 long / -1 short / 0 flat (primary model)
    p_down: float
    p_flat: float
    p_up: float
    p_take: float          # meta probability the proposed bet wins
    take: bool             # p_take >= threshold and side != 0
    size: float            # 0.0 unless taken; else p_take (fractional Kelly-lite)
    confidence: float      # max primary class probability
    signal: Signal = Signal.HOLD
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "side": self.side, "signal": self.signal.value,
            "p_down": self.p_down, "p_flat": self.p_flat, "p_up": self.p_up,
            "p_take": self.p_take, "take": self.take, "size": self.size,
        }


@dataclass
class TripleBarrierTrainingResult:
    n_samples: int
    n_features: int
    class_balance: dict          # {-1: n, 0: n, +1: n}
    primary_accuracy: float
    meta_precision: float        # precision of "take" on the OOF meta set
    meta_trained: bool
    primary_path: str


class TripleBarrierPredictor:
    """Two-stage XGBoost: primary predicts DIRECTION from triple-barrier labels;
    meta predicts whether to TAKE the primary's bet. Position size is non-zero
    only when the meta-label is 1 (Roadmap Task 1.5).
    """

    META_THRESHOLD = 0.5
    _CLASS_ORDER = (-1, 0, 1)      # xgboost classes 0,1,2 map to these

    def __init__(self, pair: str = "BTC/USDT") -> None:
        self.pair = pair
        self._safe_pair = pair.replace("/", "_")
        self.primary_path = MODEL_DIR / f"tb_primary_{self._safe_pair}.joblib"
        self.meta_path = MODEL_DIR / f"tb_meta_{self._safe_pair}.joblib"
        self.scaler_path = MODEL_DIR / f"tb_scaler_{self._safe_pair}.joblib"
        self.features_path = MODEL_DIR / f"tb_features_{self._safe_pair}.joblib"
        self._primary = None
        self._meta = None
        self._scaler = None
        self._feature_names: list[str] = []

    @staticmethod
    def _make_primary(random_state: int):
        return xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1,
        )

    @staticmethod
    def _make_meta(random_state: int):
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1,
        )

    # ─── Training ─────────────────────────────────────────────────────────────
    def train(
        self,
        X: pd.DataFrame,
        close: pd.Series,
        *,
        pt_mult: float = 2.0,
        sl_mult: float = 2.0,
        max_holding: int = 10,
        oof_split: float = 0.6,
        random_state: int = 42,
        save: bool = True,
    ) -> TripleBarrierTrainingResult:
        """Label with the triple barrier, then fit (see fit_labeled)."""
        if not ML_AVAILABLE:
            raise RuntimeError("xgboost and scikit-learn are required for training")

        tb = triple_barrier_labels(close, pt_mult, sl_mult, max_holding)
        common = X.index.intersection(tb.index)
        Xa, y = X.loc[common], tb.loc[common, "label"]
        keep = y.notna() & Xa.notna().all(axis=1)
        Xa, y = Xa[keep], y[keep].astype(int)
        return self.fit_labeled(
            Xa, y, oof_split=oof_split, random_state=random_state, save=save
        )

    def fit_labeled(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        oof_split: float = 0.6,
        random_state: int = 42,
        save: bool = True,
    ) -> TripleBarrierTrainingResult:
        """Fit primary + meta from PRE-COMPUTED triple-barrier labels y∈{-1,0,+1}.

        Meta is trained on OUT-OF-FOLD primary predictions (time split) so it sees
        no leakage. ``save=False`` skips disk writes — used by the CPCV validation
        loop which refits per split. Labels are computed once on the full
        (contiguous) series so per-split fitting needs no contiguity.
        """
        if not ML_AVAILABLE:
            raise RuntimeError("xgboost and scikit-learn are required for training")
        y = y.astype(int)
        if len(X) < 50:
            raise ValueError(f"Too few labelled samples to train: {len(X)}")

        Xa = X
        self._feature_names = list(Xa.columns)
        self._scaler = StandardScaler().fit(Xa)
        Xs = self._scaler.transform(Xa)
        y3 = (y + 1).to_numpy()                 # {-1,0,1} → {0,1,2}
        y_arr = y.to_numpy()

        # ── Out-of-fold primary preds for meta (time split, no shuffle) ───────
        split = int(len(Xs) * oof_split)
        meta_trained = False
        meta_precision = 0.0
        try:
            oof_primary = self._make_primary(random_state)
            oof_primary.fit(Xs[:split], y3[:split])
            oof_proba = oof_primary.predict_proba(Xs[split:])
            oof_side = np.array(self._CLASS_ORDER)[oof_proba.argmax(axis=1)]
            actual = y_arr[split:]
            trades = oof_side != 0
            if trades.sum() >= 20 and len(np.unique((oof_side[trades] == actual[trades]))) == 2:
                meta_target = (oof_side[trades] == actual[trades]).astype(int)
                meta_X = np.hstack([Xs[split:], oof_proba])[trades]
                self._meta = self._make_meta(random_state)
                self._meta.fit(meta_X, meta_target)
                meta_pred = self._meta.predict(meta_X)
                meta_precision = float(precision_score(meta_target, meta_pred, zero_division=0))
                meta_trained = True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Meta-model training skipped: {}", e)

        # ── Deploy primary refit on ALL labelled data ─────────────────────────
        self._primary = self._make_primary(random_state)
        self._primary.fit(Xs, y3)
        primary_acc = float(accuracy_score(y3, self._primary.predict(Xs)))

        if save:
            joblib.dump(self._primary, self.primary_path)
            joblib.dump(self._scaler, self.scaler_path)
            joblib.dump(self._feature_names, self.features_path)
            if meta_trained:
                joblib.dump(self._meta, self.meta_path)

        balance = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
        logger.info(
            "Triple-barrier trained — {} samples, classes {}, primary acc {:.1%}, "
            "meta {} (precision {:.1%})",
            len(Xa), balance, primary_acc,
            "trained" if meta_trained else "skipped", meta_precision,
        )
        return TripleBarrierTrainingResult(
            n_samples=len(Xa), n_features=Xa.shape[1], class_balance=balance,
            primary_accuracy=round(primary_acc, 4), meta_precision=round(meta_precision, 4),
            meta_trained=meta_trained, primary_path=str(self.primary_path),
        )

    # ─── Loading ──────────────────────────────────────────────────────────────
    def load(self) -> bool:
        if not ML_AVAILABLE or not self.primary_path.exists():
            return False
        self._primary = joblib.load(self.primary_path)
        self._scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None
        self._feature_names = joblib.load(self.features_path) if self.features_path.exists() else []
        self._meta = joblib.load(self.meta_path) if self.meta_path.exists() else None
        return True

    def is_trained(self) -> bool:
        return self._primary is not None

    # ─── Prediction ───────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> MetaPrediction:
        """Primary proposes a side; meta gates + sizes it. Size is 0 unless the
        meta-label is 1 (or, if no meta model, unless the primary is non-flat)."""
        if not self.is_trained() and not self.load():
            return self._neutral("Primary model not trained")
        if not ML_AVAILABLE:
            return self._neutral("xgboost not installed")
        try:
            if self._feature_names:
                X = X[[c for c in self._feature_names if c in X.columns]]
            row = X.dropna().iloc[[-1]]
            Xs = self._scaler.transform(row) if self._scaler is not None else row.to_numpy()

            proba = self._primary.predict_proba(Xs)[0]
            p_down, p_flat, p_up = (float(proba[0]), float(proba[1]), float(proba[2]))
            side = int(self._CLASS_ORDER[int(np.argmax(proba))])
            confidence = float(np.max(proba))

            if side == 0:
                return MetaPrediction(
                    side=0, p_down=p_down, p_flat=p_flat, p_up=p_up,
                    p_take=0.0, take=False, size=0.0, confidence=confidence,
                    signal=Signal.HOLD,
                )

            if self._meta is not None:
                meta_X = np.hstack([Xs, proba.reshape(1, -1)])
                p_take = float(self._meta.predict_proba(meta_X)[0][1])
            else:
                p_take = confidence      # no meta model → primary confidence gates
            take = p_take >= self.META_THRESHOLD
            size = p_take if take else 0.0
            signal = (Signal.BUY if side == 1 else Signal.SELL) if take else Signal.HOLD
            return MetaPrediction(
                side=side, p_down=p_down, p_flat=p_flat, p_up=p_up,
                p_take=round(p_take, 4), take=take, size=round(size, 4),
                confidence=round(confidence, 4), signal=signal,
            )
        except Exception as e:
            logger.error("Triple-barrier prediction failed: {}", e)
            return self._neutral(str(e))

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row predictions for a whole frame (for backtesting / CPCV).

        Returns a DataFrame indexed by the (NaN-free) rows of X with columns:
            side, p_take, take, size, signed_size (= side * size).
        ``signed_size`` is the fractional directional position — 0 when the
        meta-label declines the trade.
        """
        if not self.is_trained() and not self.load():
            raise RuntimeError("Primary model not trained")
        if self._feature_names:
            X = X[[c for c in self._feature_names if c in X.columns]]
        Xv = X.dropna()
        Xs = self._scaler.transform(Xv) if self._scaler is not None else Xv.to_numpy()

        proba = self._primary.predict_proba(Xs)
        sides = np.array(self._CLASS_ORDER)[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        if self._meta is not None:
            p_take = self._meta.predict_proba(np.hstack([Xs, proba]))[:, 1]
        else:
            p_take = conf
        take = (sides != 0) & (p_take >= self.META_THRESHOLD)
        size = np.where(take, p_take, 0.0)
        return pd.DataFrame(
            {"side": sides, "p_take": p_take, "take": take,
             "size": size, "signed_size": sides * size},
            index=Xv.index,
        )

    def _neutral(self, error: str = "") -> MetaPrediction:
        return MetaPrediction(
            side=0, p_down=0.0, p_flat=1.0, p_up=0.0, p_take=0.0,
            take=False, size=0.0, confidence=0.0, signal=Signal.HOLD, error=error,
        )
