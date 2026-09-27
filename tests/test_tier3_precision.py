"""Tier-3 regression tests: Decimal precision & short accounting (audit V-7/V-8).

Locks in: Decimal sizing with cautious rounding (quantity DOWN, losses UP),
a Decimal wallet, and the wallet-vs-recorded-PnL equality for BOTH sides
(the old code credited long-style proceeds on short exits, so a winning
short drained the wallet while the DB logged a profit).
"""

from decimal import Decimal

import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from execution.paper_trader import PaperTrader
from risk.manager import RiskManager
from storage.trade_logger import TradeLogger


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


def make_signal(signal: Signal) -> AISignal:
    return AISignal(
        signal=signal,
        score=0.8 if signal == Signal.BUY else -0.8,
        confidence=0.9,
        is_actionable=True,
        llm_score=0.0,
        ml_score=0.0,
        sentiment_score=0.0,
    )


def make_trader(tmp_path) -> tuple[PaperTrader, TradeLogger]:
    db_path = str(tmp_path / "test.db")
    db = TradeLogger(db_path=db_path)
    return PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path), db


def open_and_close(trader, db, signal: Signal, exit_mult: float):
    """Open one position via process_candle, close it at entry*exit_mult.

    Returns (recorded_pnl, entry_commission, balance_delta) as Decimals.
    """
    start_balance = trader._balance
    trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=make_signal(signal))
    open_trades = db.get_open_trades()
    assert len(open_trades) == 1, "position did not open"
    trade = open_trades[0]

    entry_cost = Decimal(str(trade["entry_price"])) * Decimal(str(trade["quantity"]))
    entry_commission = entry_cost * trader._commission

    trader._close_position(trade, float(trade["entry_price"]) * exit_mult, "signal")
    closed = db.get_trade_history(limit=1)[0]
    recorded_pnl = Decimal(str(closed["pnl"]))
    balance_delta = trader._balance - start_balance
    return recorded_pnl, entry_commission, balance_delta


# ─── V-8: short accounting ─────────────────────────────────────────────────────

class TestShortAccounting:
    def test_winning_short_increases_balance(self, tmp_path):
        trader, db = make_trader(tmp_path)
        pnl, entry_fee, delta = open_and_close(trader, db, Signal.SELL, 0.90)
        assert pnl > 0, "price fell 10% — the short must be a recorded win"
        assert delta > 0, "a winning short must INCREASE the wallet"

    def test_losing_short_decreases_balance(self, tmp_path):
        trader, db = make_trader(tmp_path)
        pnl, entry_fee, delta = open_and_close(trader, db, Signal.SELL, 1.05)
        assert pnl < 0
        assert delta < 0

    @pytest.mark.parametrize("signal,exit_mult", [
        (Signal.BUY, 1.10), (Signal.BUY, 0.95),
        (Signal.SELL, 0.90), (Signal.SELL, 1.05),
    ])
    def test_wallet_delta_equals_pnl_minus_entry_fee(self, tmp_path, signal, exit_mult):
        """The 1C-3 invariant: round-trip balance change == recorded PnL
        minus the entry commission, for both sides, wins and losses."""
        trader, db = make_trader(tmp_path)
        pnl, entry_fee, delta = open_and_close(trader, db, signal, exit_mult)
        # recorded pnl is quantized to 4 dp on persist — allow that much slack
        assert abs(delta - (pnl - entry_fee)) < Decimal("0.001"), (
            f"wallet delta {delta} != pnl {pnl} - entry fee {entry_fee}"
        )

    def test_short_exit_fee_adds_to_buyback_cost(self, tmp_path):
        """Flat price: a short round trip must lose exactly fees+slippage,
        never gain (the old formula subtracted the exit fee from the cost)."""
        trader, db = make_trader(tmp_path)
        pnl, entry_fee, delta = open_and_close(trader, db, Signal.SELL, 1.0)
        assert pnl < 0
        assert delta < 0


# ─── V-7: Decimal sizing & cautious rounding ───────────────────────────────────

class TestDecimalSizing:
    def test_plan_fields_are_decimal(self):
        rm = RiskManager(initial_balance=10_000.0)
        plan = rm.calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=60_000.0,
            current_balance=10_000.0, atr=1_000.0,
        )
        assert plan is not None
        for field in ("entry_price", "stop_loss", "take_profit",
                      "quantity", "position_value", "risk_amount"):
            assert isinstance(getattr(plan, field), Decimal), field

    def test_quantity_rounded_down_to_8dp(self):
        # 10000 balance, price 3, atr 1: qty = 200/3 = 66.666... repeating
        rm = RiskManager(initial_balance=10_000.0)
        plan = rm.calculate_position(
            symbol="X/USDT", side="buy", entry_price=3.0,
            current_balance=10_000.0, atr=1.0,
        )
        assert plan.quantity == Decimal("66.66666666")  # DOWN, not 66.66666667
        assert plan.position_value == plan.quantity * plan.entry_price

    def test_nan_atr_returns_none(self):
        rm = RiskManager(initial_balance=10_000.0)
        assert rm.calculate_position(
            symbol="X/USDT", side="buy", entry_price=100.0,
            current_balance=10_000.0, atr=float("nan"),
        ) is None

    def test_loss_pct_rounds_up(self):
        # 1/3 loss = 0.333... → must round UP at 8 dp (cautious direction)
        loss = RiskManager._loss_pct(Decimal("3"), Decimal("2"))
        assert loss == Decimal("0.33333334")

    def test_no_loss_is_exact_zero(self):
        assert RiskManager._loss_pct(Decimal("3"), Decimal("3")) == Decimal("0")

    def test_wallet_is_decimal_after_round_trip(self, tmp_path):
        trader, db = make_trader(tmp_path)
        open_and_close(trader, db, Signal.BUY, 1.05)
        assert isinstance(trader._balance, Decimal)
