"""Task 1.5 — triple-barrier labelling + meta-labelling."""
import numpy as np
import pandas as pd
import pytest

from ai.triple_barrier import daily_volatility, meta_labels, triple_barrier_labels


def _close(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="1h", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


class TestTripleBarrierLabels:
    # Constant vol 0.01, pt=sl=2 → ±2% barriers, horizon 3 bars.
    def test_first_touch_wins_across_a_full_series(self):
        close = _close([100, 100.5, 103, 100, 98, 100, 100, 100])
        out = triple_barrier_labels(close, pt_mult=2, sl_mult=2, max_holding=3, vol=0.01)
        assert out["label"].tolist()[:5] == [1.0, 1.0, -1.0, -1.0, 1.0]
        # Bars without a full 3-bar forward horizon are unlabelled.
        assert out["label"].iloc[5:].isna().all()

    def test_upper_barrier_labels_plus_one_and_records_touch(self):
        close = _close([100, 100.5, 102.5, 100, 100, 100])
        out = triple_barrier_labels(close, pt_mult=2, sl_mult=2, max_holding=3, vol=0.01)
        assert out["label"].iloc[0] == 1.0
        assert out["t1"].iloc[0] == 2                 # touched at bar 2
        assert out["ret"].iloc[0] == pytest.approx(0.025)

    def test_lower_barrier_labels_minus_one(self):
        close = _close([100, 99.5, 97.5, 100, 100, 100])
        out = triple_barrier_labels(close, pt_mult=2, sl_mult=2, max_holding=3, vol=0.01)
        assert out["label"].iloc[0] == -1.0
        assert out["t1"].iloc[0] == 2

    def test_time_barrier_labels_zero(self):
        # Stays within ±2% for the whole horizon → time stop → 0.
        close = _close([100, 100.5, 100.8, 101.0, 100.5, 100])
        out = triple_barrier_labels(close, pt_mult=2, sl_mult=2, max_holding=3, vol=0.01)
        assert out["label"].iloc[0] == 0.0
        assert out["t1"].iloc[0] == 3                 # vertical barrier at i+max_holding

    def test_precedence_stop_before_profit(self):
        # Drops to the stop at bar1, then rockets up — the STOP (first touch) wins.
        close = _close([100, 97, 110, 100, 100])
        out = triple_barrier_labels(close, pt_mult=2, sl_mult=2, max_holding=3, vol=0.01)
        assert out["label"].iloc[0] == -1.0 and out["t1"].iloc[0] == 1

    def test_unlabelled_when_no_volatility_estimate(self):
        # vol=None → daily_volatility: the first bar has no return → NaN vol → unlabelled.
        close = _close(list(np.linspace(100, 130, 60)))
        out = triple_barrier_labels(close, max_holding=5)
        assert np.isnan(out["label"].iloc[0])


class TestMetaLabels:
    def test_win_when_side_matches_barrier(self):
        side = pd.Series([1, -1, 1, -1])
        tb = pd.Series([1.0, -1.0, -1.0, 1.0])
        assert meta_labels(side, tb).tolist() == [1.0, 1.0, 0.0, 0.0]

    def test_time_stop_is_a_loss_for_either_side(self):
        side = pd.Series([1, -1])
        tb = pd.Series([0.0, 0.0])
        assert meta_labels(side, tb).tolist() == [0.0, 0.0]

    def test_no_bet_and_unlabelled_are_nan(self):
        side = pd.Series([0, 1])
        tb = pd.Series([1.0, np.nan])
        m = meta_labels(side, tb)
        assert m.isna().all()


class TestDailyVolatility:
    def test_is_positive_and_nan_at_start(self):
        vol = daily_volatility(_close(list(np.linspace(100, 120, 50))))
        assert np.isnan(vol.iloc[0])
        assert (vol.dropna() >= 0).all()


# ─── Predictor smoke (needs xgboost) ──────────────────────────────────────────
from ai.ml_predictor import ML_AVAILABLE, MetaPrediction, TripleBarrierPredictor  # noqa: E402


@pytest.mark.skipif(not ML_AVAILABLE, reason="xgboost/sklearn not installed")
class TestTripleBarrierPredictor:
    @pytest.fixture(autouse=True)
    def _cleanup_model_files(self):
        yield
        from ai.ml_predictor import MODEL_DIR
        for f in MODEL_DIR.glob("tb_*_TEST_USDT.joblib"):
            f.unlink(missing_ok=True)

    def _learnable(self, n=800, seed=0):
        rng = np.random.default_rng(seed)
        feat = rng.normal(0, 1, n)
        # forward returns driven by the feature → the label is learnable from it
        rets = 0.003 * np.sign(feat) + rng.normal(0, 0.003, n)
        close = pd.Series(100 * np.exp(np.cumsum(np.concatenate([[0.0], rets[:-1]]))),
                          index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))
        X = pd.DataFrame(
            {"feat": feat, "n1": rng.normal(0, 1, n), "n2": rng.normal(0, 1, n)},
            index=close.index,
        )
        return X, close

    def test_train_produces_labels_and_a_usable_predictor(self):
        X, close = self._learnable()
        pred = TripleBarrierPredictor(pair="TEST/USDT")
        res = pred.train(X, close, pt_mult=1.5, sl_mult=1.5, max_holding=10)
        assert res.n_samples > 100
        assert set(res.class_balance).issubset({-1, 0, 1})
        assert pred.is_trained()

    def test_prediction_obeys_the_sizing_gate(self):
        X, close = self._learnable()
        pred = TripleBarrierPredictor(pair="TEST/USDT")
        pred.train(X, close, pt_mult=1.5, sl_mult=1.5, max_holding=10)
        out = pred.predict(X)
        assert isinstance(out, MetaPrediction)
        assert out.side in (-1, 0, 1)
        # Size is non-zero ONLY when the meta-label takes the trade.
        if out.take:
            assert out.side != 0 and out.p_take >= 0.5 and out.size == out.p_take
        else:
            assert out.size == 0.0

    def test_flat_primary_never_takes(self):
        # A neutral single row: whatever the side, the size gate holds.
        X, close = self._learnable()
        pred = TripleBarrierPredictor(pair="TEST/USDT")
        pred.train(X, close, pt_mult=1.5, sl_mult=1.5, max_holding=10)
        out = pred.predict(X)
        if out.side == 0:
            assert out.take is False and out.size == 0.0

    def test_fit_labeled_no_save_and_predict_batch_gate(self):
        from ai.triple_barrier import triple_barrier_labels
        X, close = self._learnable()
        tb = triple_barrier_labels(close, 1.5, 1.5, 10)
        y = tb["label"]
        keep = y.notna() & X.notna().all(axis=1)
        pred = TripleBarrierPredictor(pair="TEST/USDT")
        pred.fit_labeled(X[keep], y[keep], save=False)
        assert not pred.primary_path.exists()          # save=False writes nothing
        pb = pred.predict_batch(X[keep])
        assert list(pb.columns) == ["side", "p_take", "take", "size", "signed_size"]
        # signed_size is 0 exactly where the meta-label declines the trade.
        assert (pb.loc[~pb["take"], "signed_size"] == 0).all()
