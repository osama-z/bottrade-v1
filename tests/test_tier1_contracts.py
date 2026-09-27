"""Tier-1 regression tests: phantom cross-module references (audit V-1..V-6).

Locks in the contracts between the entry-point scripts and the modules they
drive, so a renamed method/field/column breaks CI instead of being swallowed
by a runtime ``except Exception``.
"""

import inspect
import re
from pathlib import Path

import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from data.fetcher import DataFetcher
from execution.paper_trader import PaperTrader
from risk.manager import RiskManager
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINTS = [
    PROJECT_ROOT / "scripts" / "run_live.py",
    PROJECT_ROOT / "scripts" / "run_decoupled_execution.py",
    PROJECT_ROOT / "scripts" / "run_decoupled_intelligence.py",
]


class TestPhantomReferences:
    """V-1, V-2, V-4, V-5: attributes the entrypoints call must exist."""

    def test_entrypoints_use_fetch_ohlcv_not_fetch(self):
        for path in ENTRYPOINTS:
            src = path.read_text()
            assert not re.search(r"\.fetcher\.fetch\(", src), (
                f"{path.name} calls the nonexistent DataFetcher.fetch()"
            )
            # Entrypoints fetch OHLCV directly OR via the shared pipeline helper
            # (data/pipeline.build_indicator_frame) — both are valid.
            assert (".fetcher.fetch_ohlcv(" in src
                    or "build_indicator_frame(" in src), path.name

    def test_fetch_ohlcv_accepts_entrypoint_kwargs(self):
        params = inspect.signature(DataFetcher.fetch_ohlcv).parameters
        assert {"pair", "timeframe", "limit"} <= set(params)

    def test_no_anthropic_api_key_references(self):
        for path in ENTRYPOINTS + [PROJECT_ROOT / "scripts" / "health_check.py"]:
            assert "anthropic_api_key" not in path.read_text(), (
                f"{path.name} references a Settings field that does not exist"
            )

    def test_settings_has_groq_api_key(self):
        from config.settings import Settings

        assert "groq_api_key" in Settings.model_fields

    def test_close_position_signature_matches_exit_job_call(self):
        params = list(inspect.signature(PaperTrader._close_position).parameters)
        assert params == ["self", "trade", "exit_price", "exit_reason"]

    def test_exit_job_does_not_call_calculate_pnl(self):
        src = (PROJECT_ROOT / "scripts" / "run_decoupled_execution.py").read_text()
        assert "_calculate_pnl" not in src

    def test_risk_manager_has_check_drawdown_limit(self):
        params = list(inspect.signature(RiskManager.check_drawdown_limit).parameters)
        assert params == ["self", "current_equity"]


class TestExecutorAISignalContract:
    """V-3: the executor must be able to rebuild AISignal from a ZMQ message."""

    def test_executor_kwargs_are_constructible(self):
        # Exact kwarg set built in run_decoupled_execution._run_polling_loop
        sig = AISignal(
            signal=Signal.BUY,
            score=0.5,
            confidence=0.7,
            is_actionable=True,
            llm_score=0.0,
            ml_score=0.2,
            sentiment_score=0.1,
        )
        assert sig.signal is Signal.BUY
        assert sig.is_actionable


class TestCheckDrawdownLimit:
    """V-5: behavior of the newly implemented drawdown monitoring helper."""

    def _rm(self) -> RiskManager:
        return RiskManager(initial_balance=10_000.0)  # max_drawdown_pct = 0.10

    def test_no_drawdown(self):
        breached, dd = self._rm().check_drawdown_limit(10_000.0)
        assert breached is False
        assert dd == 0.0

    def test_equity_above_peak_is_zero(self):
        breached, dd = self._rm().check_drawdown_limit(12_000.0)
        assert breached is False
        assert dd == 0.0

    def test_partial_drawdown_not_breached(self):
        breached, dd = self._rm().check_drawdown_limit(9_500.0)
        assert breached is False
        assert dd == pytest.approx(0.05)

    def test_breach_at_exact_limit(self):
        breached, dd = self._rm().check_drawdown_limit(9_000.0)
        assert breached is True
        assert dd == pytest.approx(0.10)

    def test_monitoring_is_read_only(self):
        rm = self._rm()
        rm.check_drawdown_limit(5_000.0)
        assert rm.circuit_breaker_tripped() is False


class TestATRColumnContract:
    """V-6: process_candle must read the "ATR" column compute_all() writes."""

    def _make_df(self, price: float = 100.0, atr: float = 2.0) -> pd.DataFrame:
        # Last candle stamped "now" so the stale-data check passes.
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

    def _make_trader(self, tmp_path) -> tuple[PaperTrader, TradeLogger]:
        db_path = str(tmp_path / "test.db")
        db = TradeLogger(db_path=db_path)
        trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)
        return trader, db

    def _buy_signal(self) -> AISignal:
        return AISignal(
            signal=Signal.BUY,
            score=0.8,
            confidence=0.9,
            is_actionable=True,
            llm_score=0.0,
            ml_score=0.0,
            sentiment_score=0.0,
        )

    def test_uppercase_atr_column_opens_position(self, tmp_path):
        trader, db = self._make_trader(tmp_path)
        trader.process_candle(
            df=self._make_df(), pair="BTC/USDT", ai_signal=self._buy_signal()
        )
        assert len(db.get_open_trades()) == 1

    def test_nan_atr_does_not_open_position(self, tmp_path):
        trader, db = self._make_trader(tmp_path)
        df = self._make_df()
        df.loc[df.index[-1], "ATR"] = float("nan")
        trader.process_candle(df=df, pair="BTC/USDT", ai_signal=self._buy_signal())
        assert len(db.get_open_trades()) == 0
