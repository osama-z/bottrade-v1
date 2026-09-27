#!/usr/bin/env python3
"""Test script for Phase 3 components."""
import sys
sys.path.insert(0, "/home/osama/ai-workspace/projects/neurontrade")

import os

# ─── Test 1: TradeLogger ──────────────────────────────────────────────────────
print("=" * 50)
print("TEST 1: TradeLogger")
print("=" * 50)

from storage.trade_logger import TradeLogger

db = TradeLogger("/tmp/test_neurontrade.db")

# Open a trade
trade_id = db.log_trade_open(
    symbol="BTC/USDT",
    side="buy",
    quantity=0.01,
    entry_price=64000.0,
    stop_loss=62720.0,
    take_profit=67200.0,
    ai_score=0.75,
    ai_confidence=0.82,
    ai_reasoning="RSI oversold bounce",
)
print(f"  ✅ Trade opened: id={trade_id}")

# Log a signal
db.log_signal(
    symbol="BTC/USDT",
    ml_score=0.68,
    llm_score=0.75,
    sentiment_score=0.12,
    combined_score=0.596,
    confidence=0.78,
    decision="BUY",
)
print("  ✅ Signal logged")

# Get open trades
open_trades = db.get_open_trades()
assert len(open_trades) == 1, f"Expected 1 open trade, got {len(open_trades)}"
print(f"  ✅ Open trades: {len(open_trades)}")

# Close the trade
db.log_trade_close(trade_id, 67200.0, 32.0, "take_profit")
print("  ✅ Trade closed")

# Verify stats
stats = db.get_stats()
assert stats["total_trades"] == 1, "Expected 1 total trade"
assert stats["wins"] == 1, "Expected 1 win"
assert stats["total_pnl"] == 32.0, f"Expected pnl=32.0, got {stats['total_pnl']}"
print(f"  ✅ Stats: trades={stats['total_trades']}, wins={stats['wins']}, pnl=${stats['total_pnl']:.2f}")

# Daily PnL
daily = db.get_daily_pnl()
assert daily == 32.0, f"Expected daily pnl=32.0, got {daily}"
print(f"  ✅ Daily PnL: ${daily:.2f}")

# Cleanup
os.remove("/tmp/test_neurontrade.db")
print("\n  ALL TradeLogger TESTS PASSED ✅")

# ─── Test 2: RiskManager ─────────────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 2: RiskManager")
print("=" * 50)

from risk.manager import RiskManager

risk = RiskManager(initial_balance=10000.0)

# Test can_open_trade — should pass
allowed, reason = risk.can_open_trade(
    current_balance=10000.0,
    open_position_count=0,
    daily_pnl=0.0,
)
assert allowed, f"Expected trade allowed, got: {reason}"
print(f"  ✅ can_open_trade (normal): {allowed} — {reason}")

# Test max positions guard
allowed2, reason2 = risk.can_open_trade(
    current_balance=10000.0,
    open_position_count=3,
    daily_pnl=0.0,
)
assert not allowed2, "Expected trade blocked at max positions"
print(f"  ✅ Max positions guard: blocked = {not allowed2}")

# Test daily loss limit
allowed3, reason3 = risk.can_open_trade(
    current_balance=10000.0,
    open_position_count=0,
    daily_pnl=-600.0,  # 6% loss > 5% limit
)
assert not allowed3, "Expected trade blocked at daily loss limit"
print(f"  ✅ Daily loss limit guard: blocked = {not allowed3}")

# Test position sizing
plan = risk.calculate_position(
    symbol="BTC/USDT",
    side="buy",
    entry_price=64000.0,
    current_balance=10000.0,
    atr=850.0,
)
assert plan is not None, "Expected valid position plan"
assert plan.quantity > 0, "Expected positive quantity"
assert plan.stop_loss < plan.entry_price, "SL must be below entry for longs"
assert plan.take_profit > plan.entry_price, "TP must be above entry for longs"
assert plan.reward_risk_ratio >= 2.0, f"Expected R:R >= 2.0, got {plan.reward_risk_ratio}"
print(f"  ✅ Position plan: qty={plan.quantity:.6f} @ ${plan.entry_price:.2f}")
print(f"     SL=${plan.stop_loss:.2f} | TP=${plan.take_profit:.2f} | R:R={plan.reward_risk_ratio}")
print(f"     Risk=${plan.risk_amount:.2f} | Position=${plan.position_value:.2f}")

# Test SL/TP check
fake_trade = {
    "stop_loss": 62720.0,
    "take_profit": 67200.0,
    "side": "buy",
}
# Should trigger TP
result = risk.check_position_exits(fake_trade, candle_high=68000.0, candle_low=63000.0)
assert result is not None and result[0] == "take_profit", "Expected TP hit"
print(f"  ✅ TP detection: {result[0]} @ ${result[1]:.2f}")

# Should trigger SL
result2 = risk.check_position_exits(fake_trade, candle_high=64000.0, candle_low=62000.0)
assert result2 is not None and result2[0] == "stop_loss", "Expected SL hit"
print(f"  ✅ SL detection: {result2[0]} @ ${result2[1]:.2f}")

# Both hit same candle — SL priority
result3 = risk.check_position_exits(fake_trade, candle_high=68000.0, candle_low=62000.0)
assert result3 is not None and result3[0] == "stop_loss", "Expected SL priority over TP"
print(f"  ✅ SL priority over TP: {result3[0]}")

print("\n  ALL RiskManager TESTS PASSED ✅")

# ─── Test 3: PaperTrader basic init ──────────────────────────────────────────
print()
print("=" * 50)
print("TEST 3: PaperTrader Init")
print("=" * 50)

from execution.paper_trader import PaperTrader

db2 = TradeLogger("/tmp/test_papertrader.db")
trader = PaperTrader(initial_balance=10000.0, db=db2)

assert trader.balance == 10000.0, "Balance mismatch"
assert not trader.is_paused, "Should not be paused"
assert trader.is_running, "Should be running"
print(f"  ✅ Balance: ${trader.balance:,.2f}")
print(f"  ✅ is_paused: {trader.is_paused}")
print(f"  ✅ is_running: {trader.is_running}")

state = trader.get_portfolio_state()
assert state.balance == 10000.0
assert state.open_position_count == 0
assert state.total_return_pct == 0.0
print(f"  ✅ Portfolio state: balance=${state.balance:,.2f}, positions={state.open_position_count}")

os.remove("/tmp/test_papertrader.db")
print("\n  ALL PaperTrader TESTS PASSED ✅")

print()
print("=" * 50)
print("ALL PHASE 3 COMPONENT TESTS PASSED ✅")
print("=" * 50)
