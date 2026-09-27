"""Final integration — shadow-parity divergence computation (Task 5.1 closure)."""
import pandas as pd

from scripts.shadow_parity import backtest_decisions, signal_divergence, signal_to_decision


class TestSignalToDecision:
    def test_maps_signals(self):
        assert signal_to_decision(1) == "BUY"
        assert signal_to_decision(-1) == "SELL"
        assert signal_to_decision(0) == "HOLD"
        assert signal_to_decision(99) == "HOLD"      # unknown → HOLD


class TestSignalDivergence:
    def test_identical_is_zero(self):
        s = {("BTC", "t1"): "BUY", ("BTC", "t2"): "HOLD"}
        assert signal_divergence(s, dict(s))["divergence_pct"] == 0.0

    def test_mismatch_percentage(self):
        s = {("BTC", "t1"): "BUY", ("BTC", "t2"): "HOLD"}
        b = {("BTC", "t1"): "SELL", ("BTC", "t2"): "HOLD"}
        r = signal_divergence(s, b)
        assert r["compared"] == 2 and r["mismatches"] == 1
        assert r["divergence_pct"] == 50.0
        assert r["details"][0] == (("BTC", "t1"), "BUY", "SELL")

    def test_only_shared_keys_are_compared(self):
        s = {("BTC", "t1"): "BUY", ("BTC", "t2"): "BUY"}
        b = {("BTC", "t1"): "BUY", ("ETH", "t9"): "SELL"}
        r = signal_divergence(s, b)
        assert r["compared"] == 1 and r["mismatches"] == 0

    def test_no_overlap_reports_zero_compared(self):
        r = signal_divergence({("a", 1): "BUY"}, {("b", 2): "SELL"})
        assert r["compared"] == 0 and r["divergence_pct"] == 0.0


class TestBacktestDecisions:
    def test_maps_generate_signals_to_decisions_by_timestamp(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")

        class FakeStrat:
            def generate_signals(self, df):
                return pd.Series([1, 0, -1], index=idx)

        d = backtest_decisions(FakeStrat(), df=None)
        assert list(d.values()) == ["BUY", "HOLD", "SELL"]
        assert idx[0].isoformat() in d


class TestCandleAlignment:
    def test_aligns_to_last_closed_candle_not_forming(self):
        # Decision logged at 11:10:05 → the bot decided on the 11:09 candle
        # (closed at 11:10), NOT the 11:10 candle (still forming). Aligning to the
        # forming candle is the off-by-one that faked a 6% divergence.
        from scripts.shadow_parity import _align_to_candles, _timeframe_delta
        candles = ["2026-08-11T11:08:00+00:00", "2026-08-11T11:09:00+00:00",
                   "2026-08-11T11:10:00+00:00"]
        rows = [{"symbol": "BTC/USDT", "timestamp": "2026-08-11T11:10:05+00:00",
                 "decision": "SELL"}]
        aligned = _align_to_candles(rows, candles, _timeframe_delta("1m"))
        assert aligned == {("BTC/USDT", "2026-08-11T11:09:00+00:00"): "SELL"}

    def test_timeframe_delta(self):
        import pandas as pd
        from scripts.shadow_parity import _timeframe_delta
        assert _timeframe_delta("1m") == pd.Timedelta(minutes=1)
        assert _timeframe_delta("5m") == pd.Timedelta(minutes=5)
        assert _timeframe_delta("4h") == pd.Timedelta(hours=4)
