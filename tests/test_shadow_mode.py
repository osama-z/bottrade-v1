"""Task 5.1 — shadow mode: decide + log expected fill, execute NOTHING."""
import pandas as pd

from backtesting.costs import CostModel
from config.constants import Signal
from execution.shadow import build_shadow_record
from storage.trade_logger import TradeLogger
from strategies.base import TradeSignal


def _df():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame({"close": [100.0, 101.0, 102.0], "volume": [1000.0] * 3}, index=idx)


class _MockTrader:
    def __init__(self):
        self.calls = []

    def process_candle(self, **kwargs):
        self.calls.append(kwargs)


# ─── Pure expected-fill math ───────────────────────────────────────────────────
class TestBuildShadowRecord:
    def test_buy_fills_higher_than_signal_price(self):
        r = build_shadow_record(Signal.BUY, 100.0, CostModel(enable_latency=False),
                                candle_volume=1000.0)
        assert r.decision == "BUY"
        assert r.expected_slippage > 0 and r.expected_fill > 100.0

    def test_sell_fills_lower(self):
        r = build_shadow_record(Signal.SELL, 100.0, CostModel(enable_latency=False),
                                candle_volume=1000.0)
        assert r.decision == "SELL" and r.expected_fill < 100.0

    def test_hold_has_no_fill_or_slippage(self):
        r = build_shadow_record(Signal.HOLD, 100.0, CostModel(), candle_volume=1000.0)
        assert r.decision == "HOLD"
        assert r.expected_slippage == 0.0 and r.expected_fill == 100.0


# ─── DB logging round-trip ─────────────────────────────────────────────────────
class TestShadowLogging:
    def test_log_and_read_back(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        db.log_shadow_decision(symbol="BTC/USDT", decision="BUY", price=100.0,
                               expected_slippage=0.001, expected_fill=100.1,
                               confidence=0.8, reason="filters passed", correlation_id="c1")
        rows = db.get_shadow_decisions()
        assert len(rows) == 1
        assert rows[0]["decision"] == "BUY"
        assert rows[0]["expected_fill"] == 100.1
        assert rows[0]["correlation_id"] == "c1"


# ─── Interception: shadow logs, never executes ─────────────────────────────────
class TestExecutionInterception:
    def _bot(self, db, *, shadow):
        from scripts.run_live import NeuronTradeBot
        bot = NeuronTradeBot.__new__(NeuronTradeBot)   # bypass heavy __init__
        bot.shadow_mode = shadow
        bot._cost_model = CostModel()
        bot.db = db
        bot.trader = _MockTrader()
        return bot

    def _signal(self):
        return TradeSignal(signal=Signal.BUY, confidence=0.9, pair="BTC/USDT",
                           price=102.0, reason="buy")

    def test_shadow_logs_and_places_no_order(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        bot = self._bot(db, shadow=True)
        bot._execute_or_shadow("BTC/USDT", _df(), ai_signal="AISIG",
                               trade_signal=self._signal(), corr="c1")
        assert bot.trader.calls == []                  # execution intercepted
        rows = db.get_shadow_decisions()
        assert len(rows) == 1 and rows[0]["decision"] == "BUY"
        assert rows[0]["expected_fill"] > rows[0]["price"]   # buy fills higher

    def test_non_shadow_executes_and_logs_no_shadow(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        bot = self._bot(db, shadow=False)
        bot._execute_or_shadow("BTC/USDT", _df(), ai_signal="AISIG",
                               trade_signal=self._signal(), corr="c1")
        assert len(bot.trader.calls) == 1              # normal execution
        assert bot.trader.calls[0]["pair"] == "BTC/USDT"
        assert db.get_shadow_decisions() == []          # no shadow row

    def test_shadow_hold_is_logged_without_order(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        bot = self._bot(db, shadow=True)
        hold = TradeSignal(signal=Signal.HOLD, confidence=0.0, pair="BTC/USDT",
                           price=102.0, reason="no signal")
        bot._execute_or_shadow("BTC/USDT", _df(), ai_signal=None,
                               trade_signal=hold, corr="c2")
        assert bot.trader.calls == []
        rows = db.get_shadow_decisions()
        assert len(rows) == 1 and rows[0]["decision"] == "HOLD"


class _MockTelegram:
    def __init__(self):
        self.shadow_calls = []

    def send_shadow_signal(self, **kwargs):
        self.shadow_calls.append(kwargs)


class TestShadowTelegramWiring:
    def _bot(self, db, *, shadow=True, telegram=None):
        from scripts.run_live import NeuronTradeBot
        bot = NeuronTradeBot.__new__(NeuronTradeBot)
        bot.shadow_mode = shadow
        bot._cost_model = CostModel()
        bot.db = db
        bot.trader = _MockTrader()
        bot.telegram = telegram
        return bot

    def test_shadow_buy_sends_marked_alert(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        tg = _MockTelegram()
        bot = self._bot(db, telegram=tg)
        ts = TradeSignal(signal=Signal.BUY, confidence=0.9, pair="BTC/USDT",
                         price=102.0, reason="buy")
        bot._execute_or_shadow("BTC/USDT", _df(), ai_signal=None, trade_signal=ts, corr="c1")
        assert len(tg.shadow_calls) == 1
        call = tg.shadow_calls[0]
        assert call["decision"] == "BUY" and call["expected_fill"] > call["price"]
        assert bot.trader.calls == []                 # still no execution

    def test_shadow_hold_sends_no_alert(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        tg = _MockTelegram()
        bot = self._bot(db, telegram=tg)
        hold = TradeSignal(signal=Signal.HOLD, confidence=0.0, pair="BTC/USDT",
                           price=102.0, reason="no signal")
        bot._execute_or_shadow("BTC/USDT", _df(), ai_signal=None, trade_signal=hold, corr="c2")
        assert tg.shadow_calls == []                   # no HOLD spam
        assert len(db.get_shadow_decisions()) == 1     # but still logged to DB


def test_run_live_uses_public_only_fetcher():
    # Paper/shadow trading needs only public market data — the fetcher must be
    # key-free so a testnet order key is never sent to production (-2008).
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert "DataFetcher(public_only=True)" in (root / "scripts" / "run_live.py").read_text()


class TestShadowDecisionTagging:
    def test_strategy_and_timeframe_are_stored_and_filterable(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        db.log_shadow_decision(symbol="BTC/USDT", decision="BUY", price=100.0,
                               expected_slippage=0.001, expected_fill=100.1,
                               strategy="rsi_reversal", timeframe="5m")
        db.log_shadow_decision(symbol="ETH/USDT", decision="SELL", price=50.0,
                               expected_slippage=0.001, expected_fill=49.9,
                               strategy="trend_following", timeframe="4h")
        # Filter to the stress config only.
        stress = db.get_shadow_decisions(strategy="rsi_reversal", timeframe="5m")
        assert len(stress) == 1 and stress[0]["symbol"] == "BTC/USDT"
        assert stress[0]["strategy"] == "rsi_reversal" and stress[0]["timeframe"] == "5m"
        # Timeframe-only filter.
        assert len(db.get_shadow_decisions(timeframe="4h")) == 1
        # No filter → both.
        assert len(db.get_shadow_decisions()) == 2

    def test_stress_runner_imports(self):
        import importlib
        m = importlib.import_module("scripts.run_shadow_stress")
        assert callable(m.main)
