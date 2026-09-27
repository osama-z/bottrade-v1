"""Tier-7 regression tests: metrics unification, logging, ops hygiene,
scale-out rework (audit V-59/V-17/V-39/V-55/V-56/V-58/V-50/V-71)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from config.settings import settings
from execution.paper_trader import PaperTrader
from risk import metrics
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── V-59: one metric library ──────────────────────────────────────────────────

class TestMetrics:
    def test_sharpe_per_trade_matches_reference(self):
        pnls = [10.0, -5.0, 20.0, -5.0, 8.0]
        n = len(pnls)
        mean = sum(pnls) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in pnls) / (n - 1))
        assert metrics.sharpe_per_trade(pnls) == pytest.approx(mean / std)

    def test_sharpe_per_trade_degenerate_cases(self):
        assert metrics.sharpe_per_trade([]) == 0.0
        assert metrics.sharpe_per_trade([5.0]) == 0.0
        assert metrics.sharpe_per_trade([3.0, 3.0, 3.0]) == 0.0  # zero variance

    def test_sharpe_annualized_matches_pandas_formula(self):
        returns = [0.01, -0.005, 0.02, 0.003, -0.01]
        s = pd.Series(returns)
        expected = float(s.mean() / s.std() * (8760 ** 0.5))
        assert metrics.sharpe_annualized(returns, 8760) == pytest.approx(expected)

    def test_zero_pnl_counts_as_loss(self):
        assert metrics.is_win(0.0) is False
        assert metrics.is_win(-1.0) is False
        assert metrics.is_win(0.01) is True
        assert metrics.win_rate([1.0, 0.0, -1.0, 2.0]) == pytest.approx(0.5)

    def test_consumers_import_the_shared_module(self):
        for rel in ("risk/walk_forward.py", "backtesting/engine.py",
                    "dashboard/analytics.py"):
            src = (PROJECT_ROOT / rel).read_text()
            assert "metrics" in src, rel
        # no residual inline sharpe formulas
        engine = (PROJECT_ROOT / "backtesting" / "engine.py").read_text()
        assert "returns.mean() / returns.std()" not in engine


# ─── V-17/V-39: single logging config, JSON + UTC ──────────────────────────────

class TestLogging:
    def test_file_sink_is_json_and_console_is_utc(self):
        src = (PROJECT_ROOT / "config" / "logging_config.py").read_text()
        assert "serialize=True" in src
        assert "!UTC" in src

    def test_entrypoints_use_central_setup_only(self):
        for rel in ("scripts/run_live.py", "scripts/run_decoupled_execution.py",
                    "scripts/run_decoupled_intelligence.py"):
            src = (PROJECT_ROOT / rel).read_text()
            assert "setup_logging()" in src, rel
            assert "logger.remove()" not in src, f"{rel} installs its own handlers"


# ─── V-71: single DB path ──────────────────────────────────────────────────────

class TestDbPath:
    def test_database_path_is_absolute_and_project_anchored(self):
        p = Path(settings.database_path)
        assert p.is_absolute()
        assert str(PROJECT_ROOT) in str(p)

    def test_dashboard_uses_configured_path(self):
        src = (PROJECT_ROOT / "dashboard" / "analytics.py").read_text()
        assert "settings.database_path" in src
        assert '"storage" / "neurontrade.db"' not in src


# ─── V-55/V-56/V-58: ops hygiene ──────────────────────────────────────────────

class TestOpsHygiene:
    def test_telegram_stop_uses_event_and_join(self):
        src = (PROJECT_ROOT / "notifications" / "telegram_bot.py").read_text()
        assert "_stop_event.set()" in src
        assert "self._thread.join" in src
        assert "await app.shutdown()" in src

    def test_telegram_send_has_plaintext_fallback(self):
        src = (PROJECT_ROOT / "notifications" / "telegram_bot.py").read_text()
        assert "delivered as plain text" in src

    def test_signal_handlers_do_not_sys_exit(self):
        for rel in ("scripts/run_live.py", "scripts/run_decoupled_execution.py"):
            src = (PROJECT_ROOT / rel).read_text()
            shutdown = src.split("def _shutdown", 1)[1].split("def ", 1)[0]
            assert "sys.exit(" not in shutdown, rel  # the call, not the docstring
        exec_src = (PROJECT_ROOT / "scripts" / "run_decoupled_execution.py").read_text()
        assert "self.scheduler.shutdown(wait=True)" in exec_src


# ─── V-50: scale-out uses configured RR and persists its PnL ──────────────────

def make_df(price: float, atr: float = 2.0) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=5, freq="1h")
    df = pd.DataFrame(
        {"open": price, "high": price * 1.01, "low": price * 0.99,
         "close": price, "volume": 100.0},
        index=idx,
    )
    df["ATR"] = atr
    return df


class TestScaleOut:
    def test_scale_out_triggers_at_config_rr_and_is_audited(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        db = TradeLogger(db_path=db_path)
        trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)

        buy = AISignal(signal=Signal.BUY, score=0.8, confidence=0.9,
                       is_actionable=True, llm_score=0.0, ml_score=0.0,
                       sentiment_score=0.0)
        hold = AISignal(signal=Signal.HOLD, score=0.0, confidence=0.0,
                        is_actionable=False, llm_score=0.0, ml_score=0.0,
                        sentiment_score=0.0)

        trader.process_candle(df=make_df(100.0), pair="BTC/USDT", ai_signal=buy)
        trade = db.get_open_trades()[0]
        rr = float(trader._risk.config.reward_risk_ratio)
        initial_risk = (trade["take_profit"] - trade["entry_price"]) / rr
        scale_out_price = trade["entry_price"] + 1.5 * initial_risk

        # candle whose high crosses 1.5R but not the take-profit
        price = scale_out_price / 1.005
        trader.process_candle(df=make_df(price), pair="BTC/USDT", ai_signal=hold)

        updated = db.get_open_trades()[0]
        assert updated["scaled_out"] == 1
        assert updated["quantity"] == pytest.approx(trade["quantity"] / 2, rel=1e-6)
        # SL floor is breakeven after scale-out; the ATR trailing stop may
        # legitimately sit above it when the candle high ran further.
        assert updated["stop_loss"] >= trade["entry_price"] - 1e-9

        with db._get_conn() as conn:
            rows = conn.execute(
                "SELECT event_type FROM risk_events WHERE event_type='scale_out'"
            ).fetchall()
        assert len(rows) == 1

    def test_short_positions_skip_scale_out_cleanly(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        db = TradeLogger(db_path=db_path)
        trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)

        sell = AISignal(signal=Signal.SELL, score=-0.8, confidence=0.9,
                        is_actionable=True, llm_score=0.0, ml_score=0.0,
                        sentiment_score=0.0)
        hold = AISignal(signal=Signal.HOLD, score=0.0, confidence=0.0,
                        is_actionable=False, llm_score=0.0, ml_score=0.0,
                        sentiment_score=0.0)

        trader.process_candle(df=make_df(100.0), pair="BTC/USDT", ai_signal=sell)
        assert len(db.get_open_trades()) == 1
        # price rises a little (against the short) — must not scale out,
        # must not fabricate risk, must not error
        trader.process_candle(df=make_df(101.0), pair="BTC/USDT", ai_signal=hold)
        trade = db.get_open_trades()
        if trade:  # may have hit its stop, which is a clean full exit
            assert trade[0]["scaled_out"] == 0
