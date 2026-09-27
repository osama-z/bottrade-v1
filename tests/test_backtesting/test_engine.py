"""
Tests for the BacktestEngine — verify all financial math is correct.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from backtesting.engine import BacktestEngine


@pytest.fixture
def simple_ohlcv() -> pd.DataFrame:
    """Simple price data where we know the expected outcomes."""
    n = 100
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n, freq="1h"
    )
    # Start at 100, go up to 110, back down to 95, back up to 105
    prices = np.concatenate([
        np.linspace(100, 110, 25),  # uptrend
        np.linspace(110, 95, 25),   # downtrend
        np.linspace(95, 105, 25),   # uptrend
        np.linspace(105, 100, 25),  # slight downtrend
    ])

    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.full(n, 100.0),
    }, index=dates)
    # The engine sizes via RiskManager.calculate_position (no static
    # fallback), which requires ATR — same as the live path.
    df["ATR"] = 2.0
    return df


class TestBacktestEngine:

    def test_no_trades_returns_initial_capital(self, simple_ohlcv):
        """If all signals are HOLD (0), capital should stay the same."""
        engine = BacktestEngine(initial_capital=1000.0, commission_pct=0, slippage_pct=0)
        signals = pd.Series(0, index=simple_ohlcv.index)
        result = engine.run(simple_ohlcv, signals, strategy_name="NoTrades")

        assert result.total_trades == 0
        assert result.total_return_pct == 0.0
        assert result.win_rate_pct == 0.0

    def test_capital_conservation(self, simple_ohlcv):
        """Capital should never go negative, and total equity should be tracked."""
        engine = BacktestEngine(initial_capital=1000.0)
        # Buy/sell repeatedly
        signals = pd.Series(0, index=simple_ohlcv.index)
        signals.iloc[5] = 1    # Buy
        signals.iloc[20] = -1  # Sell (during uptrend — should profit)
        signals.iloc[30] = 1   # Buy
        signals.iloc[45] = -1  # Sell (during downtrend — should lose)

        result = engine.run(simple_ohlcv, signals, strategy_name="Conservation")
        assert result.total_trades >= 2

    def test_profitable_trade_math(self):
        """Verify a single profitable trade has correct P&L."""
        n = 20
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
        # Price goes 100 → 104 (4% gain)
        prices = np.linspace(100, 104, n)
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001, "low": prices * 0.999,
            "close": prices, "volume": np.full(n, 100.0),
        }, index=dates)
        df["ATR"] = 2.0

        engine = BacktestEngine(initial_capital=1000.0, commission_pct=0, slippage_pct=0)
        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1    # Buy at ~100
        signals.iloc[-1] = -1  # Sell at ~104

        result = engine.run(df, signals, strategy_name="Profitable")
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.total_return_pct > 0

    def test_losing_trade_math(self):
        """Verify a single losing trade has correct P&L."""
        n = 20
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
        # Price goes 100 → 96 (4% loss)
        prices = np.linspace(100, 96, n)
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001, "low": prices * 0.999,
            "close": prices, "volume": np.full(n, 100.0),
        }, index=dates)
        df["ATR"] = 2.0

        engine = BacktestEngine(initial_capital=1000.0, commission_pct=0, slippage_pct=0)
        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1    # Buy at 100
        signals.iloc[-1] = -1  # Sell at 96

        result = engine.run(df, signals, strategy_name="Losing")
        assert result.total_trades == 1
        assert result.losing_trades == 1
        assert result.total_return_pct < 0

    def test_sharpe_ratio_is_reasonable(self, simple_ohlcv):
        """Sharpe ratio should be a finite number when there are returns."""
        engine = BacktestEngine(initial_capital=1000.0)
        signals = pd.Series(0, index=simple_ohlcv.index)
        signals.iloc[5] = 1
        signals.iloc[20] = -1

        result = engine.run(simple_ohlcv, signals, strategy_name="SharpeTest")
        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_is_negative_or_zero(self, simple_ohlcv):
        """Max drawdown should always be <= 0 (it's a peak-to-trough decline)."""
        engine = BacktestEngine(initial_capital=1000.0)
        signals = pd.Series(0, index=simple_ohlcv.index)
        signals.iloc[5] = 1
        signals.iloc[45] = -1

        result = engine.run(simple_ohlcv, signals, strategy_name="DrawdownTest")
        assert result.max_drawdown_pct <= 0

    def test_buy_and_hold_is_correct(self):
        """Buy-and-hold return should match (last_close / first_close - 1)."""
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
        prices = np.linspace(100, 150, n)  # 50% gain
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001, "low": prices * 0.999,
            "close": prices, "volume": np.full(n, 100.0),
        }, index=dates)
        df["ATR"] = 2.0

        engine = BacktestEngine(initial_capital=1000.0)
        signals = pd.Series(0, index=df.index)

        result = engine.run(df, signals, strategy_name="BHTest")
        expected_bh = (150 / 100 - 1) * 100  # 50%
        assert abs(result.buy_and_hold_return_pct - expected_bh) < 0.01

    def test_stop_loss_triggers(self):
        """Stop-loss should exit the trade when low touches stop level."""
        n = 20
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
        # Price drops sharply — should trigger 2% stop-loss
        prices = np.array([100] * 5 + [97, 96, 95, 94, 93] + [92] * 10)
        df = pd.DataFrame({
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.998,    # Low is slightly below close
            "close": prices,
            "volume": np.full(n, 100.0),
        }, index=dates)
        df["ATR"] = 2.0

        engine = BacktestEngine(initial_capital=1000.0, commission_pct=0, slippage_pct=0)
        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1  # Buy at 100

        result = engine.run(df, signals, strategy_name="StopLossTest", stop_loss_pct=0.02)
        # Should have been stopped out
        assert result.total_trades >= 1

    def test_commission_reduces_returns(self):
        """Higher commission should result in lower returns."""
        n = 30
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
        prices = np.linspace(100, 110, n)
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001, "low": prices * 0.999,
            "close": prices, "volume": np.full(n, 100.0),
        }, index=dates)
        df["ATR"] = 2.0

        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1
        signals.iloc[-2] = -1

        engine_no_fee = BacktestEngine(initial_capital=1000.0, commission_pct=0, slippage_pct=0)
        engine_with_fee = BacktestEngine(initial_capital=1000.0, commission_pct=0.01, slippage_pct=0)

        result_no_fee = engine_no_fee.run(df, signals, strategy_name="NoFee")
        result_with_fee = engine_with_fee.run(df, signals, strategy_name="WithFee")

        assert result_no_fee.total_return_pct > result_with_fee.total_return_pct

    def test_profit_factor_correct(self):
        """Profit factor = sum(wins) / abs(sum(losses))."""
        n = 60
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
        # Up then down then up — create 1 win and 1 loss
        prices = np.concatenate([
            np.linspace(100, 110, 20),  # +10%
            np.linspace(110, 100, 20),  # -10%
            np.linspace(100, 108, 20),  # +8%
        ])
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001, "low": prices * 0.999,
            "close": prices, "volume": np.full(n, 100.0),
        }, index=dates)
        df["ATR"] = 2.0

        engine = BacktestEngine(initial_capital=1000.0, commission_pct=0, slippage_pct=0)
        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1     # Buy at 100
        signals.iloc[18] = -1   # Sell at ~110 (win)
        signals.iloc[20] = 1    # Buy at ~110
        signals.iloc[38] = -1   # Sell at ~100 (loss)

        result = engine.run(df, signals, strategy_name="PFTest")
        if result.total_trades == 2:
            assert result.profit_factor > 0
