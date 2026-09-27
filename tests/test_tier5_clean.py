"""Tier-5 regression tests: DB layer, canonical equity, audit history, single
AI pipeline (audit V-22/V-23/V-26/V-34)."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from execution.paper_trader import PaperTrader
from risk.manager import (
    CircuitBreakerState,
    SqliteCircuitBreakerStore,
    _BreakerRecord,
)
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_df(price: float = 100.0, atr: float = 2.0) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=5, freq="1h")
    df = pd.DataFrame(
        {"open": price, "high": price * 1.01, "low": price * 0.99,
         "close": price, "volume": 100.0},
        index=idx,
    )
    df["ATR"] = atr
    return df


def make_signal(signal: Signal) -> AISignal:
    return AISignal(
        signal=signal, score=0.8, confidence=0.9, is_actionable=True,
        llm_score=0.0, ml_score=0.0, sentiment_score=0.0,
    )


def make_trader(tmp_path) -> tuple[PaperTrader, TradeLogger]:
    db_path = str(tmp_path / "test.db")
    db = TradeLogger(db_path=db_path)
    return PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path), db


# ─── V-22: DB layer — WAL + persistent connection ──────────────────────────────

class TestDbLayer:
    def test_wal_mode_enabled(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        with db._get_conn() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_connection_is_reused(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        with db._get_conn() as c1:
            pass
        with db._get_conn() as c2:
            pass
        assert c1 is c2, "each query opened a fresh connection"

    def test_close_and_reopen(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        db.save_balance(123.0)
        db.close()
        assert db.load_balance() == 123.0  # lazily reopens

    def test_breaker_store_shares_wal_file(self, tmp_path):
        path = tmp_path / "b.db"
        SqliteCircuitBreakerStore(path)
        import sqlite3
        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


# ─── V-23: breaker transition history ──────────────────────────────────────────

class TestBreakerHistory:
    def test_transitions_are_appended_not_overwritten(self, tmp_path):
        store = SqliteCircuitBreakerStore(tmp_path / "b.db")
        store.save(_BreakerRecord(
            state=CircuitBreakerState.TRIPPED, reason="daily loss",
            tripped_at_utc=datetime.now(UTC),
        ))
        store.save(_BreakerRecord(state=CircuitBreakerState.ARMED))  # manual reset

        import sqlite3
        with sqlite3.connect(tmp_path / "b.db") as conn:
            rows = conn.execute(
                "SELECT state, reason FROM circuit_breaker_history ORDER BY id"
            ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("tripped", "daily loss")  # trip survives the reset
        assert rows[1][0] == "armed"


# ─── V-26: canonical mark-to-market equity ─────────────────────────────────────

class TestCanonicalEquity:
    def test_long_marks_up_with_price(self, tmp_path):
        trader, db = make_trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT",
                              ai_signal=make_signal(Signal.BUY))
        assert trader.equity({"BTC/USDT": 110.0}) > trader.equity()
        assert trader.equity({"BTC/USDT": 90.0}) < trader.equity()

    def test_short_gains_when_price_falls(self, tmp_path):
        trader, db = make_trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT",
                              ai_signal=make_signal(Signal.SELL))
        assert trader.equity({"BTC/USDT": 90.0}) > trader.equity()
        assert trader.equity({"BTC/USDT": 110.0}) < trader.equity()

    def test_prices_are_not_borrowed_across_pairs(self, tmp_path):
        trader, db = make_trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT",
                              ai_signal=make_signal(Signal.BUY))
        trader.process_candle(df=make_df(), pair="ETH/USDT",
                              ai_signal=make_signal(Signal.BUY))
        btc = next(t for t in db.get_open_trades() if t["symbol"] == "BTC/USDT")
        expected_unrealized = (110.0 - btc["entry_price"]) * btc["quantity"]
        delta = trader.equity({"BTC/USDT": 110.0}) - trader.equity()
        assert delta == pytest.approx(expected_unrealized, rel=1e-9)

    def test_portfolio_state_uses_canonical_equity(self, tmp_path):
        trader, db = make_trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT",
                              ai_signal=make_signal(Signal.BUY))
        state = trader.get_portfolio_state()
        assert state.equity == pytest.approx(trader.equity())

    def test_marked_equity_feeds_risk_check(self):
        src = (PROJECT_ROOT / "execution" / "paper_trader.py").read_text()
        assert "total_equity = float(marked_equity)" in src


# ─── V-34: single AI pipeline per candle ───────────────────────────────────────

class TestSingleAIPipeline:
    def test_run_live_does_not_recombine(self):
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "strategy._combiner.combine" not in src
        assert "strategy._llm.analyze" not in src
        # getattr form: rule-based strategies (STRATEGY=trend_following)
        # have no last_ai_signal; the reuse invariant is unchanged.
        assert 'getattr(strategy, "last_ai_signal", None)' in src

    def test_strategy_exposes_last_ai_signal(self):
        src = (PROJECT_ROOT / "strategies" / "ai_combined.py").read_text()
        assert "self.last_ai_signal = ai_signal" in src
        assert "self.last_ai_signal = None" in src  # reset on entry
