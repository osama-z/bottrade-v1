"""Tier-14 tests: backtest engine short support + configurable BUY gates.

The engine was long-only (`if signal == 1 and position == 0`) while
PaperTrader opens a SHORT on a SELL signal — so the validation harness
structurally could not evaluate half of what the live bot does. Over an
18-month bear market that meant 47 short signals scored as 0 trades.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, _equity
from config.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def frame(prices: list[float], atr: float = 2.0) -> pd.DataFrame:
    n = len(prices)
    arr = np.array(prices, dtype=float)
    df = pd.DataFrame(
        {"open": arr, "high": arr * 1.001, "low": arr * 0.999,
         "close": arr, "volume": np.full(n, 100.0)},
        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
    )
    df["ATR"] = atr
    return df


def run(prices, entry_signal, **kw):
    df = frame(prices)
    sig = pd.Series(0, index=df.index)
    sig.iloc[1] = entry_signal
    engine = BacktestEngine(initial_capital=1000.0,
                            commission_pct=kw.pop("commission_pct", 0.0),
                            slippage_pct=kw.pop("slippage_pct", 0.0))
    return engine.run(df, sig, **kw)


class TestTradeRecords:
    """entry_idx must be the real entry (was equal to exit_idx), so trades can
    be analysed by entry conditions; duration follows from it."""

    def test_entry_idx_is_the_open_not_the_close(self):
        r = run(list(np.linspace(100, 80, 40)), -1)  # short, held down to TP
        assert r.total_trades == 1
        t = r.trades[0]
        assert t["entry_idx"] == 1               # opened on candle 1
        assert t["exit_idx"] > t["entry_idx"]     # held more than one candle

    def test_avg_trade_duration_is_positive(self):
        r = run(list(np.linspace(100, 80, 40)), -1)
        assert r.avg_trade_duration_hours > 0


# ─── Directional correctness ───────────────────────────────────────────────────

class TestShortDirection:
    def test_short_profits_in_falling_market(self):
        r = run(list(np.linspace(100, 80, 40)), -1)
        assert r.total_trades == 1
        assert r.total_return_pct > 0, "a short must GAIN when price falls"

    def test_short_loses_in_rising_market(self):
        r = run(list(np.linspace(100, 120, 40)), -1)
        assert r.total_trades == 1
        assert r.total_return_pct < 0

    def test_long_still_profits_in_rising_market(self):
        r = run(list(np.linspace(100, 120, 40)), 1)
        assert r.total_trades == 1
        assert r.total_return_pct > 0

    def test_sell_signal_opens_a_trade_at_all(self):
        """The regression: SELL entries were silently ignored."""
        r = run(list(np.linspace(100, 85, 40)), -1)
        assert r.total_trades == 1, "SELL must OPEN a short, not be dropped"


# ─── Stop/target orientation is inverted for shorts ────────────────────────────

class TestShortStopsAndTargets:
    def test_short_stop_is_above_entry(self):
        """Price spikes up → the short's stop (above entry) must trigger."""
        prices = [100.0] * 3 + [130.0] * 10   # violent move against the short
        r = run(prices, -1)
        assert r.total_trades == 1
        assert r.total_return_pct < 0

    def test_short_take_profit_is_below_entry(self):
        prices = [100.0] * 3 + [80.0] * 10    # move in the short's favour
        r = run(prices, -1)
        assert r.total_trades == 1
        assert r.total_return_pct > 0

    def test_short_loss_is_bounded_by_the_stop(self):
        """Even in a runaway rally the loss is capped near the ATR stop —
        proof the stop fires rather than the position riding forever."""
        prices = [100.0] * 3 + list(np.linspace(101, 200, 40))
        r = run(prices, -1)
        assert r.total_trades == 1
        assert r.total_return_pct > -15, "stop must cap the loss"


# ─── Fee direction: the exit fee ADDS to a short's buy-back cost ───────────────

class TestShortFees:
    def test_flat_price_short_loses_only_fees(self):
        flat = [100.0] * 30
        r = run(flat, -1, commission_pct=0.001, slippage_pct=0.0005)
        if r.total_trades:
            assert r.total_return_pct < 0, "round trip at flat price must cost fees"

    def test_commission_reduces_short_profit(self):
        prices = list(np.linspace(100, 85, 40))
        free = run(prices, -1, commission_pct=0.0, slippage_pct=0.0)
        paid = run(prices, -1, commission_pct=0.01, slippage_pct=0.0)
        assert paid.total_return_pct < free.total_return_pct


# ─── Equity accounting parity with PaperTrader._mark_to_market ────────────────

class TestEquityHelper:
    def test_flat_equity_is_just_cash(self):
        assert _equity(1000.0, 0.0, 0.0, 50.0) == 1000.0

    def test_long_equity_rises_with_price(self):
        base = _equity(500.0, 5.0, 100.0, 100.0)
        assert _equity(500.0, 5.0, 100.0, 110.0) > base
        assert _equity(500.0, 5.0, 100.0, 90.0) < base

    def test_short_equity_rises_when_price_falls(self):
        base = _equity(500.0, -5.0, 100.0, 100.0)
        assert _equity(500.0, -5.0, 100.0, 90.0) > base
        assert _equity(500.0, -5.0, 100.0, 110.0) < base

    def test_short_equity_matches_paper_trader_formula(self):
        """cash + collateral + signed unrealized — same as _mark_to_market."""
        cash, qty, entry, price = 500.0, 5.0, 100.0, 90.0
        expected = cash + entry * qty + (entry - price) * qty
        assert _equity(cash, -qty, entry, price) == pytest.approx(expected)


# ─── Configurable BUY gates (the zero-trade fix) ───────────────────────────────

class TestGateSettings:
    def test_timing_defaults_to_or_not_and(self):
        """RSI<30 AND MACD-cross fired on 0.03% of candles (3 of 8,727) and
        produced zero trades in a full walk-forward."""
        assert Settings().buy_timing_require_both is False

    def test_regime_blocks_bearish_only_by_default(self):
        """The HMM labels 86% of candles 'Choppy'; blocking on it was a ~15x
        trade-count cut — effectively an off switch."""
        assert Settings().regime_block_bearish_only is True

    def test_gate_thresholds_are_validated(self):
        with pytest.raises(Exception):
            Settings(BUY_RSI_MAX=0)
        with pytest.raises(Exception):
            Settings(BUY_RSI_MAX=101)
        with pytest.raises(Exception):
            Settings(BUY_VOLUME_MULTIPLE=-1)

    def test_strategy_reads_gates_from_settings(self):
        """Audit V-65: these were hardcoded literals."""
        src = (PROJECT_ROOT / "strategies" / "ai_combined.py").read_text()
        assert "settings.buy_timing_require_both" in src
        assert "settings.buy_rsi_max" in src
        assert "settings.buy_volume_multiple" in src
        assert "settings.regime_block_bearish_only" in src
        assert "rsi_val < 30 and macd_cross_up" not in src  # old hardcoded AND


# ─── Engine/live parity guard ──────────────────────────────────────────────────

class TestParity:
    def test_engine_opens_on_both_signal_directions(self):
        src = (PROJECT_ROOT / "backtesting" / "engine.py").read_text()
        assert "signal in (1, -1) and position == 0" in src
        assert "if signal == 1 and position == 0" not in src  # long-only form

    def test_engine_short_close_adds_commission_to_buyback(self):
        src = (PROJECT_ROOT / "backtesting" / "engine.py").read_text()
        assert "buyback_cost = proceeds + commission" in src


# ─── Cross-system PnL convention pin ───────────────────────────────────────────
# Both systems report per-trade pnl as NET EXIT vs GROSS ENTRY (the entry
# fee is paid by the wallet at open but excluded from the stat). This pin
# exists because a well-intentioned "fix" to either side alone would
# silently break backtest/live parity.

def _paper_trade_pnl(tmp_path, signal, exit_price: float) -> float:
    """Open via PaperTrader at close=100 (slippage 0), close at exit_price."""
    from ai.signal_combiner import AISignal
    from execution.paper_trader import PaperTrader
    from storage.trade_logger import TradeLogger

    db_path = str(tmp_path / "parity.db")
    db = TradeLogger(db_path=db_path)
    trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path,
                         commission_pct=0.001, slippage_pct=0.0)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=5, freq="1h")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 100.0}, index=idx)
    df["ATR"] = 2.0
    trader.process_candle(df=df, pair="BTC/USDT", ai_signal=AISignal(
        signal=signal, score=0.8, confidence=0.9, is_actionable=True,
        llm_score=0.0, ml_score=0.0, sentiment_score=0.0))
    trade = db.get_open_trades()[0]
    trader._close_position(trade, exit_price, "signal")
    pnl = float(db.get_trade_history(limit=1)[0]["pnl"])
    db.close()
    return pnl


def _engine_trade_pnl(entry_signal: int, exit_close: float) -> float:
    """Same trade through the engine: open at close=100, close on the
    opposite signal at exit_close. Slippage 0, commission 0.1%."""
    n = 40
    prices = np.array([100.0] * (n // 2) + [exit_close] * (n - n // 2))
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices,
                       "close": prices, "volume": np.full(n, 100.0)},
                      index=pd.date_range("2024-01-01", periods=n,
                                          freq="1h", tz="UTC"))
    df["ATR"] = 2.0
    sig = pd.Series(0, index=df.index)
    sig.iloc[1] = entry_signal
    sig.iloc[n - 2] = -entry_signal
    result = BacktestEngine(initial_capital=10_000.0, commission_pct=0.001,
                            slippage_pct=0.0).run(df, sig)
    assert result.total_trades == 1, "parity scenario must produce one trade"
    return float(result.trades[0]["pnl"])


class TestCrossSystemPnlConvention:
    def test_long_pnl_matches_paper_trader(self, tmp_path):
        from config.constants import Signal
        engine_pnl = _engine_trade_pnl(1, 105.0)
        paper_pnl = _paper_trade_pnl(tmp_path, Signal.BUY, 105.0)
        # Decimal-vs-float sizing rounds qty slightly differently; the
        # CONVENTION (which fees are inside pnl) must agree to <0.1%.
        assert engine_pnl == pytest.approx(paper_pnl, rel=1e-3)

    def test_short_pnl_matches_paper_trader(self, tmp_path):
        from config.constants import Signal
        engine_pnl = _engine_trade_pnl(-1, 95.0)
        paper_pnl = _paper_trade_pnl(tmp_path, Signal.SELL, 95.0)
        assert engine_pnl == pytest.approx(paper_pnl, rel=1e-3)

    def test_convention_is_net_exit_vs_gross_entry(self):
        """Explicit statement of the shared convention: exit fee inside
        pnl, entry fee outside (paid by the wallet)."""
        from config.settings import settings

        engine_pnl = _engine_trade_pnl(1, 105.0)
        # qty = risk (settings %) / sl_pct (1.5*ATR/price = 3%) / price
        qty = 10_000 * settings.risk_per_trade / (3.0 / 100.0) / 100.0
        expected = (105.0 * qty) * (1 - 0.001) - 100.0 * qty  # exit fee only
        assert engine_pnl == pytest.approx(expected, rel=1e-6)

    def test_engine_risk_config_comes_from_settings(self):
        """The bug this class caught on its first run: the engine's bare
        RiskManager() fallback hardcodes 1% risk while live reads settings
        (2% default) — backtests sized at HALF the live risk."""
        src = (PROJECT_ROOT / "backtesting" / "engine.py").read_text()
        assert "RiskConfig.from_settings(capital)" in src
        paper = (PROJECT_ROOT / "execution" / "paper_trader.py").read_text()
        assert "RiskConfig.from_settings(initial_balance)" in paper
