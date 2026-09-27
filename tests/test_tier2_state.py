"""Tier-2 regression tests: Stage 0 state guarantees (audit V-9/V-10/V-12/V-13/V-24/V-25).

Covers breaker persistence wiring, balance recovery across restarts, the
atomic peak-equity update, duplicate-close protection, the edge-triggered
consecutive-loss breaker, and ZMQ signal idempotency.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from core.zmq_subscriber import SignalSubscriber
from execution.paper_trader import PaperTrader
from risk.manager import (
    CircuitBreakerState,
    SqliteCircuitBreakerStore,
    _BreakerRecord,
)
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Helpers ────────────────────────────────────────────────────────────────────

def make_df(price: float = 100.0, atr: float = 2.0) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 100.0,
        },
        index=idx,
    )
    df["ATR"] = atr
    return df


def buy_signal() -> AISignal:
    return AISignal(
        signal=Signal.BUY,
        score=0.8,
        confidence=0.9,
        is_actionable=True,
        llm_score=0.0,
        ml_score=0.0,
        sentiment_score=0.0,
    )


def make_trader(tmp_path, initial: float = 10_000.0) -> tuple[PaperTrader, TradeLogger, str]:
    db_path = str(tmp_path / "test.db")
    db = TradeLogger(db_path=db_path)
    trader = PaperTrader(initial_balance=initial, db=db, db_path=db_path)
    return trader, db, db_path


# ─── V-13: atomic peak-equity update ───────────────────────────────────────────

class TestPeakEquityUpdate:
    def test_update_never_touches_breaker_state(self, tmp_path):
        store = SqliteCircuitBreakerStore(tmp_path / "b.db")
        store.save(
            _BreakerRecord(
                state=CircuitBreakerState.TRIPPED,
                reason="daily loss",
                tripped_at_utc=datetime.now(UTC),
                peak_equity=10_000.0,
            )
        )
        store.update_peak_equity(12_000.0)
        record = store.load()
        assert record.state is CircuitBreakerState.TRIPPED  # trip survives
        assert record.reason == "daily loss"
        assert record.peak_equity == 12_000.0

    def test_update_is_monotonic(self, tmp_path):
        store = SqliteCircuitBreakerStore(tmp_path / "b.db")
        store.update_peak_equity(10_000.0)
        store.update_peak_equity(9_000.0)  # lower — must be ignored
        assert store.load().peak_equity == 10_000.0

    def test_update_creates_row_when_missing(self, tmp_path):
        store = SqliteCircuitBreakerStore(tmp_path / "b.db")
        store.update_peak_equity(5_000.0)
        record = store.load()
        assert record.state is CircuitBreakerState.ARMED
        assert record.peak_equity == 5_000.0

    def test_in_memory_store_same_contract(self):
        from risk.manager import InMemoryCircuitBreakerStore

        store = InMemoryCircuitBreakerStore()
        store.save(
            _BreakerRecord(state=CircuitBreakerState.TRIPPED, peak_equity=100.0)
        )
        store.update_peak_equity(200.0)
        assert store.load().state is CircuitBreakerState.TRIPPED
        assert store.load().peak_equity == 200.0
        store.update_peak_equity(150.0)
        assert store.load().peak_equity == 200.0


# ─── V-9: breaker persistence wiring ───────────────────────────────────────────

class TestBreakerPersistenceWiring:
    def test_run_live_passes_db_path(self):
        # The persistent breaker db_path is still wired through run_live. Post
        # Task-4.1 storage cutover it's read via getattr (the Postgres backend
        # has no db_path) and passed to PaperTrader — behaviour unchanged.
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert 'getattr(self.db, "db_path"' in src
        assert "db_path=str(_db_path)" in src

    def test_trip_survives_trader_restart(self, tmp_path):
        trader1, _, db_path = make_trader(tmp_path)
        trader1._risk.trip_circuit_breaker("test trip", datetime.now(UTC))

        trader2, _, _ = make_trader(tmp_path)
        assert trader2._risk.circuit_breaker_tripped()


# ─── V-10: balance recovery across restarts ────────────────────────────────────

class TestBalanceRecovery:
    def test_balance_survives_restart(self, tmp_path):
        trader1, db, _ = make_trader(tmp_path)
        trader1.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=buy_signal())
        assert len(db.get_open_trades()) == 1
        balance_after_open = trader1._balance
        assert balance_after_open < 10_000.0  # entry cost deducted

        # "Restart": new trader on the same DB must NOT reset to 10k
        trader2, _, _ = make_trader(tmp_path)
        assert trader2._balance == pytest.approx(balance_after_open)

    def test_fresh_db_uses_initial_balance(self, tmp_path):
        trader, _, _ = make_trader(tmp_path, initial=5_000.0)
        assert trader._balance == 5_000.0


# ─── V-24: duplicate close protection ──────────────────────────────────────────

class TestDuplicateCloseGuard:
    def test_double_close_credits_balance_once(self, tmp_path):
        trader, db, _ = make_trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=buy_signal())
        trade = db.get_open_trades()[0]

        trader._close_position(trade, 110.0, "take_profit")
        balance_after_first = trader._balance
        trader._close_position(trade, 110.0, "take_profit")  # duplicate

        assert trader._balance == balance_after_first
        assert len(db.get_open_trades()) == 0

    def test_log_trade_close_returns_false_when_already_closed(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        trade_id = db.log_trade_open(
            symbol="BTC/USDT", side="buy", quantity=1.0, entry_price=100.0,
            stop_loss=95.0, take_profit=110.0, ai_score=0.5,
            ai_confidence=0.8, ai_reasoning="test",
        )
        assert db.log_trade_close(
            trade_id=trade_id, exit_price=110.0, pnl=10.0, exit_reason="take_profit"
        ) is True
        assert db.log_trade_close(
            trade_id=trade_id, exit_price=110.0, pnl=10.0, exit_reason="take_profit"
        ) is False


# ─── V-25: consecutive-loss limit trips the real breaker ──────────────────────

class TestConsecutiveLossBreaker:
    # Small losses (~1% of position each) so the streak completes without
    # tripping the daily-loss breaker first.
    def _lose_once(self, trader, db, price_drop: float = 0.99):
        trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=buy_signal())
        trade = db.get_open_trades()[0]
        trader._close_position(trade, float(trade["entry_price"]) * price_drop, "stop_loss")

    def test_streak_trips_breaker_and_requires_manual_reset(self, tmp_path):
        trader, db, _ = make_trader(tmp_path)
        limit = trader._risk.config.consecutive_loss_limit

        for _ in range(limit):
            assert not trader._risk.circuit_breaker_tripped()
            self._lose_once(trader, db)

        assert trader._risk.circuit_breaker_tripped()
        assert "consecutive loss" in (trader._risk.store.load().reason or "")

        # Tripped breaker blocks new trades…
        allowed, reason = trader._risk.can_open_trade(
            current_balance=trader._balance, open_position_count=0,
            daily_pnl=0.0, db=db,
        )
        assert not allowed

        # …until (and only until) a manual reset.
        trader._risk.manual_reset()
        allowed, _ = trader._risk.can_open_trade(
            current_balance=trader._balance, open_position_count=0,
            daily_pnl=0.0, db=db,
        )
        assert allowed

    def test_winning_close_never_trips(self, tmp_path):
        trader, db, _ = make_trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=buy_signal())
        trade = db.get_open_trades()[0]
        trader._close_position(trade, float(trade["entry_price"]) * 1.1, "take_profit")
        assert not trader._risk.circuit_breaker_tripped()


# ─── V-12: signal idempotency ──────────────────────────────────────────────────

class TestSignalIdempotency:
    def test_publisher_includes_unique_message_id(self):
        src = (PROJECT_ROOT / "core" / "zmq_publisher.py").read_text()
        assert "message_id" in src and "uuid" in src

    def test_parse_extracts_message_id_and_excludes_from_extra(self):
        payload = json.dumps({
            "signal": "BUY", "pair": "BTC/USDT", "score": 0.5,
            "confidence": 0.7, "timestamp_utc": datetime.now(UTC).isoformat(),
            "message_id": "abc123", "ml_score": 0.2,
        }).encode()
        sig = SignalSubscriber._parse(payload)
        assert sig.message_id == "abc123"
        assert "message_id" not in sig.extra
        assert sig.extra["ml_score"] == 0.2

    def test_executor_deduplicates_on_message_id(self):
        src = (PROJECT_ROOT / "scripts" / "run_decoupled_execution.py").read_text()
        assert "last_acted_key" in src
        assert "sig.message_id" in src
