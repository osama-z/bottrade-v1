"""
Unit and integration tests for Phase 3.5 Institutional Upgrades.
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
import pytest

from storage.trade_logger import TradeLogger
from risk.manager import RiskManager
from ai.signal_combiner import SignalCombiner, LLMAnalysis
from config.constants import Signal


# ─── 1. TradeLogger Tests ──────────────────────────────────────────────────────

def test_logger_schema_and_migrations():
    """Test that TradeLogger correctly manages the new schema and database helpers."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize (which creates tables and applies migrations)
        db = TradeLogger(db_path)

        # Insert a trade and verify new columns
        trade_id = db.log_trade_open(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.5,
            entry_price=60000.0,
            stop_loss=58000.0,
            take_profit=64000.0,
            ai_score=0.8,
            ai_confidence=0.9,
            ai_reasoning="Oversold"
        )

        trade = db.get_trade(trade_id)
        assert trade is not None
        assert float(trade["highest_price"]) == 60000.0
        assert int(trade["scaled_out"]) == 0

        # Test update_trade_trailing_state
        db.update_trade_trailing_state(
            trade_id=trade_id,
            highest_price=62000.0,
            scaled_out=1,
            stop_loss=60000.0,
            quantity=0.75
        )

        updated_trade = db.get_trade(trade_id)
        assert float(updated_trade["highest_price"]) == 62000.0
        assert int(updated_trade["scaled_out"]) == 1
        assert float(updated_trade["stop_loss"]) == 60000.0
        assert float(updated_trade["quantity"]) == 0.75

        # Test consecutive losses query
        assert db.get_consecutive_losses() == 0

        # Close with loss
        db.log_trade_close(trade_id, exit_price=59000.0, pnl=-750.0, exit_reason="stop_loss")
        assert db.get_consecutive_losses() == 1

        # Add another loss
        t2 = db.log_trade_open("ETH/USDT", "buy", 10.0, 3000.0, 2900.0, 3200.0)
        db.log_trade_close(t2, exit_price=2900.0, pnl=-1000.0, exit_reason="stop_loss")
        assert db.get_consecutive_losses() == 2

        # Add a win (consecutive loss counter should reset to 0)
        t3 = db.log_trade_open("SOL/USDT", "buy", 10.0, 100.0, 95.0, 110.0)
        db.log_trade_close(t3, exit_price=105.0, pnl=50.0, exit_reason="take_profit")
        assert db.get_consecutive_losses() == 0

    finally:
        os.remove(db_path)


# ─── 2. SignalCombiner Dynamic Weighting Tests ──────────────────────────────────

def test_combiner_dynamic_weighting():
    """Test that SignalCombiner shifts weights correctly according to adx and atr regimes."""
    # Min confidence 0.60, high threshold 0.75
    combiner = SignalCombiner(
        min_confidence=0.60,
        buy_threshold=0.75,
        sell_threshold=-0.75
    )

    # Mock components
    llm = LLMAnalysis(score=0.9, signal=Signal.BUY, confidence=0.8, reasoning="Uptrend", key_factors=[], model_used="llama")

    class MockPred:
        score = 0.8
        confidence = 0.85
        error = None

    class MockSent:
        score = 0.7
        confidence = 0.75
        error = None

    # Trending regime: ADX > 25 (trust ML heavily: 60% ML, 25% LLM, 15% Sentiment)
    res_trending = combiner.combine(
        llm_analysis=llm,
        ml_prediction=MockPred(),
        sentiment_result=MockSent(),
        adx=30.0,
        atr=1.0,
        atr_sma=1.0
    )
    assert res_trending.breakdown["regime"] == "Trending"
    assert res_trending.breakdown["weights"]["ml"] == pytest.approx(0.60)
    assert res_trending.breakdown["weights"]["llm"] == pytest.approx(0.25)
    assert res_trending.breakdown["weights"]["sentiment"] == pytest.approx(0.15)

    # Choppy regime: ADX < 20 (trust LLM heavily: 25% ML, 55% LLM, 20% Sentiment)
    res_choppy = combiner.combine(
        llm_analysis=llm,
        ml_prediction=MockPred(),
        sentiment_result=MockSent(),
        adx=15.0,
        atr=1.0,
        atr_sma=1.0
    )
    assert res_choppy.breakdown["regime"] == "Choppy/Ranging"
    assert res_choppy.breakdown["weights"]["llm"] == pytest.approx(0.55)

    # High Volatility regime: ATR > 1.5 * ATR_SMA (trust Sentiment heavily: 30% ML, 30% LLM, 40% Sentiment)
    res_vol = combiner.combine(
        llm_analysis=llm,
        ml_prediction=MockPred(),
        sentiment_result=MockSent(),
        adx=30.0,
        atr=2.0,
        atr_sma=1.0
    )
    assert res_vol.breakdown["regime"] == "High Volatility/News"
    assert res_vol.breakdown["weights"]["sentiment"] == pytest.approx(0.40)


# ─── 3. RiskManager Sizing and Circuit Breakers ─────────────────────────────────

def test_risk_manager_circuit_breakers_and_sizing():
    """Test 1% position sizing and circuit breaker pauses."""
    risk = RiskManager(initial_balance=10000.0)

    # Position Sizing: 1% risk of $10,000 is $100 risk.
    # Entry=60000, ATR=1000, SL distance = 1.5 * ATR = 1500
    # Expected sl_pct = 1500 / 60000 = 2.5% (0.025)
    # Expected size = 100 / 0.025 = $4000. Qty = 4000 / 60000 = 0.06666...
    plan = risk.calculate_position(
        symbol="BTC/USDT",
        side="buy",
        entry_price=60000.0,
        current_balance=10000.0,
        atr=1000.0
    )
    assert plan is not None
    assert plan.risk_amount == 100.0  # Decimal("100") == 100.0 is exact
    assert float(plan.position_value) == pytest.approx(4000.0)
    assert float(plan.quantity) == pytest.approx(4000.0 / 60000.0)
    assert plan.stop_loss == 58500.0

    # Mock DB for circuit breakers
    class MockDB:
        def __init__(self, losses=0, exit_ago_seconds=0):
            self.losses = losses
            self.exit_ago_seconds = exit_ago_seconds

        def get_consecutive_losses(self):
            return self.losses

        def get_trade_history(self, limit=1):
            if self.losses > 0:
                t = datetime.now(timezone.utc) - timedelta(seconds=self.exit_ago_seconds)
                return [{"exit_time": t.isoformat()}]
            return []

    # Normal checks (no losses, no drawdown)
    allowed, reason = risk.can_open_trade(
        current_balance=10000.0,
        open_position_count=0,
        daily_pnl=0.0,
        db=MockDB()
    )
    assert allowed
    assert reason == "OK"

    # Consecutive loss limit (edge-triggered): a losing close that reaches
    # the limit trips the circuit breaker — manual reset only, no cooldown.
    risk.register_closed_trade(
        pnl=-50.0,
        db=MockDB(losses=3),
        timestamp_utc=datetime.now(timezone.utc),
    )
    assert risk.circuit_breaker_tripped()
    assert "consecutive loss" in (risk.store.load().reason or "").lower()

    allowed_loss, reason_loss = risk.can_open_trade(
        current_balance=10000.0,
        open_position_count=0,
        daily_pnl=0.0,
        db=MockDB(losses=3)
    )
    assert not allowed_loss
    assert "circuit breaker" in reason_loss.lower()

    # The breaker never auto-resumes — only a manual reset re-enables trading.
    risk.manual_reset()
    allowed_after_reset, _ = risk.can_open_trade(
        current_balance=10000.0,
        open_position_count=0,
        daily_pnl=0.0,
        db=MockDB(losses=3)
    )
    assert allowed_after_reset

    # A winning close never trips, regardless of prior streak
    risk.register_closed_trade(
        pnl=25.0,
        db=MockDB(losses=3),
        timestamp_utc=datetime.now(timezone.utc),
    )
    assert not risk.circuit_breaker_tripped()

    # Daily drawdown breaker: 4% daily loss, last exit 1 hour ago -> should block
    allowed_dd, reason_dd = risk.can_open_trade(
        current_balance=10000.0,
        open_position_count=0,
        daily_pnl=-400.0,  # 4% loss
        db=MockDB(losses=1, exit_ago_seconds=3600)
    )
    assert not allowed_dd
    assert "daily drawdown" in reason_dd.lower()
