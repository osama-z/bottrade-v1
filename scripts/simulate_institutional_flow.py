#!/usr/bin/env python3
"""
Simulate and audit the mathematical correctness of Phase 3.5 Institutional Upgrades.
Verifies position sizing, trailing stops, scale-out exits, and circuit breakers.
"""

import sys
sys.path.insert(0, "/home/osama/ai-workspace/projects/neurontrade")

import os
import tempfile

from storage.trade_logger import TradeLogger
from risk.manager import RiskManager
from execution.paper_trader import PaperTrader
import pytest


def run_math_audit():
    print("=" * 60)
    print("🧠 NEURONTRADE INSTITUTIONAL MATH & BUG AUDIT")
    print("=" * 60)

    # Setup database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = TradeLogger(db_path)
        # 1% risk for the math audit, injected via the constructor (the
        # old `RiskManager(risk_per_trade=...)` kwarg never existed and
        # the post-hoc `trader._risk = ...` bypassed encapsulation).
        from decimal import Decimal

        from risk.manager import RiskConfig
        config = RiskConfig(
            max_position_notional_usdt=Decimal("10000"),
            max_daily_loss_pct=Decimal("0.03"),
            max_drawdown_pct=Decimal("0.10"),
            risk_per_trade_pct=Decimal("0.01"),
        )
        risk = RiskManager(config=config, initial_balance=10000.0)
        trader = PaperTrader(initial_balance=10000.0, db=db, risk_manager=risk)

        print("\n--- 1. Position Sizing Math Verification ---")
        balance = 10000.0
        entry = 50000.0
        atr = 1000.0
        sl_distance = 1.5 * atr # 1500.0
        sl_pct = sl_distance / entry # 3.0% (0.03)

        # Risk amount is 1% of $10000 = $100
        # Position value should be 100 / 0.03 = $3333.33
        # Quantity should be 3333.33 / 50000 = 0.066667
        plan = risk.calculate_position(
            symbol="BTC/USDT",
            side="buy",
            entry_price=entry,
            current_balance=balance,
            atr=atr
        )

        expected_risk = balance * 0.01
        expected_position_val = expected_risk / sl_pct
        expected_qty = expected_position_val / entry

        print(f"Calculated Qty:          {plan.quantity:.6f}")
        print(f"Expected Qty:            {expected_qty:.6f}")
        print(f"Calculated Risk:         ${plan.risk_amount:.2f}")
        print(f"Expected Risk:           ${expected_risk:.2f}")
        print(f"Calculated Position Val: ${plan.position_value:.2f}")
        print(f"Expected Position Val:   ${expected_position_val:.2f}")

        assert abs(plan.quantity - expected_qty) < 1e-5
        assert abs(plan.risk_amount - expected_risk) < 1e-5
        assert abs(plan.position_value - round(expected_position_val, 4)) < 1e-5
        print("✅ POSITION SIZING MATH IS 100% CORRECT")

        print("\n--- 2. Scale-Out & Trailing Stop Simulation ---")
        # Let's open a trade manually in the database
        trade_id = db.log_trade_open(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.066667,
            entry_price=50000.0,
            stop_loss=48500.0,
            take_profit=53000.0,
            ai_score=0.8,
            ai_confidence=0.9
        )

        # Initial risk = (53000 - 50000) / 2 = 1500.0
        # 1.5R target = 50000 + 1.5 * 1500 = 52250.0
        # Price goes up to 51000. Stop loss should trail to 51000 - 1.5 * ATR (1000) = 49500
        # Let's simulate candle high = 51000, low = 50000
        trader._check_and_close_positions(candle_high=51000.0, candle_low=50000.0, current_price=51000.0, atr=1000.0)

        t_state = db.get_trade(trade_id)
        print(f"Price at $51,000 | New SL: ${float(t_state['stop_loss']):.2f} (Expected: $49,500.00)")
        assert float(t_state['stop_loss']) == 49500.0
        assert int(t_state['scaled_out']) == 0

        # Price spikes to 52300 (crosses 1.5R target of 52250)
        # Should scale out 50% (qty becomes 0.0333335)
        # SL should move to breakeven (50000.0)
        trader._check_and_close_positions(candle_high=52300.0, candle_low=51000.0, current_price=52300.0, atr=1000.0)

        t_state = db.get_trade(trade_id)
        print(f"Price at $52,300 | Scaled Out: {t_state['scaled_out']} (Expected: 1)")
        print(f"New Quantity:    {float(t_state['quantity']):.6f} (Expected: 0.033334)")
        print(f"New SL:          ${float(t_state['stop_loss']):.2f} (Expected: $50,800.00)")

        assert int(t_state['scaled_out']) == 1
        assert float(t_state['quantity']) == pytest.approx(0.066667 * 0.5)
        assert float(t_state['stop_loss']) == 50800.0

        # Price drops to 49800. Since we are scaled out, the SL is at $50,800.
        # The candle low of 49800 should trigger a full exit at the SL price ($50,800).
        trader._check_and_close_positions(candle_high=50500.0, candle_low=49800.0, current_price=50000.0, atr=1000.0)

        closed_trade = db.get_trade(trade_id)
        print(f"Price drops to $49,800 | Status: {closed_trade['status']} (Expected: closed)")
        print(f"Exit Price:      ${float(closed_trade['exit_price']):.2f} (Expected: $50,800.00)")
        assert closed_trade['status'] == 'closed'
        assert float(closed_trade['exit_price']) == pytest.approx(50800.0 * (1 - trader.slippage_pct))
        print("✅ SCALE-OUT & TRAILING STOP MATHEMATICS ARE 100% CORRECT")

        print("\n--- 3. Circuit Breaker Simulation ---")
        # Reset risk manager
        risk = RiskManager(initial_balance=10000.0, max_daily_loss_pct=0.03) # 3% limit

        # Scenario A: 3 consecutive losses
        # Let's log 3 losses
        db_consecutive = TradeLogger(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)

        for i in range(3):
            tid = db_consecutive.log_trade_open("BTC/USDT", "buy", 1.0, 50000.0, 48000.0, 52000.0)
            db_consecutive.log_trade_close(tid, 48000.0, -2000.0, "stop_loss")

        allowed, reason = risk.can_open_trade(
            current_balance=10000.0,
            open_position_count=0,
            daily_pnl=0.0,
            db=db_consecutive
        )
        print(f"3 Consecutive Losses | Allowed: {allowed} | Reason: {reason}")
        assert not allowed
        assert "consecutive loss" in reason.lower()

        # Scenario B: 3% daily drawdown
        db_drawdown = TradeLogger(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)
        tid = db_drawdown.log_trade_open("BTC/USDT", "buy", 1.0, 50000.0, 48000.0, 52000.0)
        db_drawdown.log_trade_close(tid, 48000.0, -350.0, "stop_loss") # $350 loss is 3.5% of $10,000

        allowed_dd, reason_dd = risk.can_open_trade(
            current_balance=10000.0,
            open_position_count=0,
            daily_pnl=-350.0,
            db=db_drawdown
        )
        print(f"3.5% Daily Loss | Allowed: {allowed_dd} | Reason: {reason_dd}")
        assert not allowed_dd
        assert "daily drawdown" in reason_dd.lower()
        print("✅ CIRCUIT BREAKERS OPERATE 100% CORRECTLY")

        print("\n" + "=" * 60)
        print("🎯 ALL MATHEMATICAL CHECKS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        os.remove(db_path)


if __name__ == "__main__":
    run_math_audit()
