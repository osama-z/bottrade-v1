"""Tier-9 regression tests: correlation IDs, latency instrumentation, and
the LOW-severity cleanup sweep (audit V-17-rest/V-38/V-73/75/76/78-88)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from config.settings import Settings, settings
from core.zmq_subscriber import SignalSubscriber
from execution.paper_trader import PaperTrader
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


def buy_signal() -> AISignal:
    return AISignal(signal=Signal.BUY, score=0.8, confidence=0.9,
                    is_actionable=True, llm_score=0.0, ml_score=0.0,
                    sentiment_score=0.0)


# ─── Correlation IDs (V-17 remainder) ─────────────────────────────────────────

class TestCorrelationIds:
    def test_cycle_id_joins_signal_and_trade(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        db = TradeLogger(db_path=db_path)
        trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)

        trader.process_candle(df=make_df(), pair="BTC/USDT",
                              ai_signal=buy_signal(), correlation_id="cycle-abc")

        with db._get_conn() as conn:
            sig_corr = conn.execute(
                "SELECT correlation_id FROM signals ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            trade_corr = conn.execute(
                "SELECT correlation_id FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
        assert sig_corr == "cycle-abc"
        assert trade_corr == "cycle-abc"

    def test_blocked_trade_event_carries_cycle_id(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        db = TradeLogger(db_path=db_path)
        trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)
        trader._risk.trip_circuit_breaker("test", datetime.now(UTC))

        trader.process_candle(df=make_df(), pair="BTC/USDT",
                              ai_signal=buy_signal(), correlation_id="cycle-x")
        with db._get_conn() as conn:
            row = conn.execute(
                "SELECT correlation_id FROM risk_events "
                "WHERE event_type='trade_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "cycle-x"

    def test_entrypoints_bind_and_pass_correlation(self):
        live = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "logger.contextualize(corr=" in live
        assert "correlation_id=corr" in live
        execd = (PROJECT_ROOT / "scripts" / "run_decoupled_execution.py").read_text()
        assert "logger.contextualize(corr=dedup_key" in execd
        assert "correlation_id=dedup_key" in execd


# ─── Latency instrumentation (V-38) ───────────────────────────────────────────

class TestLatency:
    def test_setting_exists_with_bounds(self):
        assert settings.latency_warn_ms > 0
        with pytest.raises(Exception):
            Settings(LATENCY_WARN_MS=0)

    def test_run_live_measures_pipeline_stages(self):
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "perf_counter" in src
        assert "settings.latency_warn_ms" in src
        for stage in ("fetch_ms", "indicators_ms", "signal_ms", "execute_ms"):
            assert stage in src, stage


# ─── LOW sweep ────────────────────────────────────────────────────────────────

class TestLowSweep:
    def test_order_book_repr_with_spread_does_not_raise(self):
        from data.order_book import OrderBookManager

        ob = OrderBookManager("BTC/USDT")
        ob.apply_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 2.0)],
                          last_update_id=1, timestamp_ms=0)
        text = repr(ob)  # used to raise ValueError whenever a spread existed
        assert "spread=" in text

    def test_preprocessor_drops_non_positive_closes(self):
        from data.preprocessor import DataPreprocessor

        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": [100.0, 0.0, 101.0, -5.0, 102.0, 103.0],
            "volume": 10.0,
        }, index=idx)
        out = DataPreprocessor().process(df, min_rows=1)
        assert (out["close"] > 0).all()
        assert len(out) == 4

    def test_preprocessor_ffill_is_bounded(self):
        from data.preprocessor import DataPreprocessor

        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        closes = [100.0] + [float("nan")] * 6 + [101.0, 102.0, 103.0]
        df = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": closes, "volume": 10.0,
        }, index=idx)
        out = DataPreprocessor().process(df, min_rows=1)
        # 1 real + 3 filled + tail 3 = 7; the 3 unfillable NaNs are dropped
        assert len(out) == 7

    def test_event_bus_rejects_async_handlers_without_crashing(self):
        from core.event_bus import EventBus

        bus = EventBus()
        ran = []

        async def async_handler(**kw):  # would silently never run before
            ran.append("async")

        def sync_handler(**kw):
            ran.append("sync")

        bus.subscribe("evt", async_handler)
        bus.subscribe("evt", sync_handler)
        bus.publish("evt", x=1)
        assert ran == ["sync"]

    def test_subscriber_ignores_foreign_topics(self):
        import zmq
        from datetime import timedelta

        class FakeSocket:
            def __init__(self, frames): self._frames = list(frames)
            def recv_multipart(self, flags=0):
                if not self._frames:
                    raise zmq.Again()
                return self._frames.pop(0)

        payload = json.dumps({
            "signal": "BUY", "pair": "BTC/USDT", "score": 0.5,
            "confidence": 0.7, "timestamp_utc": datetime.now(UTC).isoformat(),
        }).encode()
        sub = object.__new__(SignalSubscriber)
        sub._max_age = timedelta(seconds=30)
        sub._last = None
        sub._last_recv_monotonic = None
        sub._socket = FakeSocket([(b"signal_v2", payload)])
        assert sub.poll_signal().signal == "HOLD"  # foreign topic ignored

    def test_llm_agent_no_longer_claims_claude(self):
        src = (PROJECT_ROOT / "ai" / "llm_agent.py").read_text()
        assert "Claude" not in src  # audit-trail misattribution (V-88)

    def test_vestigial_package_removed(self):
        assert not (PROJECT_ROOT / "neurontrade" / "__init__.py").exists()

    def test_pagination_is_bounded(self):
        src = (PROJECT_ROOT / "data" / "fetcher.py").read_text()
        assert "max_batches" in src
        assert "while batches_done < max_batches" in src

    def test_health_check_inspects_database(self):
        src = (PROJECT_ROOT / "scripts" / "health_check.py").read_text()
        assert "def check_database" in src
        assert '("Database",    check_database)' in src
