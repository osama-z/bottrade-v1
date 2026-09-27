"""Tier-6 regression tests: config unification, risk-event audit, heartbeat
liveness (audit V-29/V-32/V-23-rest/V-31/V-57)."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from config.settings import Settings
from core.zmq_subscriber import SignalSubscriber
from execution.paper_trader import PaperTrader
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── V-29/V-44: settings are validated, frozen, and actually wired ────────────

class TestSettingsUnification:
    def test_settings_are_frozen(self):
        s = Settings()
        with pytest.raises(Exception):
            s.risk_per_trade = 0.99

    def test_absurd_risk_values_rejected(self):
        with pytest.raises(Exception):
            Settings(RISK_PER_TRADE=5.0)  # 500% per trade
        with pytest.raises(Exception):
            Settings(MAX_DRAWDOWN=0.9)
        with pytest.raises(Exception):
            Settings(MAX_OPEN_POSITIONS=0)

    def test_risk_config_comes_from_settings(self, tmp_path):
        from config.settings import settings

        db_path = str(tmp_path / "t.db")
        trader = PaperTrader(
            initial_balance=10_000.0,
            db=TradeLogger(db_path=db_path),
            db_path=db_path,
        )
        cfg = trader._risk.config
        assert cfg.risk_per_trade_pct == Decimal(str(settings.risk_per_trade))
        assert cfg.max_drawdown_pct == Decimal(str(settings.max_drawdown))
        assert cfg.max_daily_loss_pct == Decimal(str(settings.max_daily_loss))
        assert cfg.max_concurrent_positions == settings.max_open_positions

    def test_no_hardcoded_thresholds_in_entrypoints(self):
        exec_src = (PROJECT_ROOT / "scripts" / "run_decoupled_execution.py").read_text()
        assert "max_signal_age_seconds=settings.stale_signal_seconds" in exec_src
        assert "initial_balance=settings.initial_balance" in exec_src
        live_src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "initial_balance=settings.initial_balance" in live_src
        pt_src = (PROJECT_ROOT / "execution" / "paper_trader.py").read_text()
        # Timeframe-aware staleness: threshold is the configured grace plus one
        # candle period (the loop decides on the last CLOSED candle).
        assert "settings.stale_data_seconds + duration" in pt_src
        assert "time_diff > 300" not in pt_src


# ─── V-23 rest: risk events persisted ──────────────────────────────────────────

class TestRiskEventAudit:
    def test_log_and_read_risk_event(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        db.log_risk_event("trade_blocked", symbol="BTC/USDT", reason="max positions")
        db.log_risk_event("kill_switch", reason="SIGUSR1")
        with db._get_conn() as conn:
            rows = conn.execute(
                "SELECT event_type, symbol, reason FROM risk_events ORDER BY id"
            ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("trade_blocked", "BTC/USDT", "max positions"),
            ("kill_switch", "", "SIGUSR1"),
        ]

    def test_blocked_trade_writes_event(self, tmp_path):
        import pandas as pd
        from ai.signal_combiner import AISignal
        from config.constants import Signal

        db_path = str(tmp_path / "t.db")
        db = TradeLogger(db_path=db_path)
        trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)
        trader._risk.trip_circuit_breaker("test", datetime.now(UTC))

        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=5, freq="1h")
        df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": 100.0}, index=idx)
        df["ATR"] = 2.0
        sig = AISignal(signal=Signal.BUY, score=0.8, confidence=0.9,
                       is_actionable=True, llm_score=0.0, ml_score=0.0,
                       sentiment_score=0.0)
        trader.process_candle(df=df, pair="BTC/USDT", ai_signal=sig)

        with db._get_conn() as conn:
            rows = conn.execute(
                "SELECT event_type FROM risk_events"
            ).fetchall()
        assert ("trade_blocked",) in [tuple(r) for r in rows]


# ─── V-31/V-57: heartbeat liveness + per-message parse ─────────────────────────

def _payload(**overrides) -> bytes:
    base = {
        "signal": "BUY", "pair": "BTC/USDT", "score": 0.5, "confidence": 0.7,
        "timestamp_utc": datetime.now(UTC).isoformat(), "message_id": "m1",
    }
    base.update(overrides)
    return json.dumps(base).encode()


class TestHeartbeatLiveness:
    def _sub(self):
        """Subscriber with a stubbed socket (no network)."""
        from datetime import timedelta

        sub = object.__new__(SignalSubscriber)
        sub._max_age = timedelta(seconds=30)
        sub._last = None
        sub._last_recv_monotonic = None
        return sub

    class _FakeSocket:
        def __init__(self, frames):
            self._frames = list(frames)

        def recv_multipart(self, flags=0):
            import zmq
            if not self._frames:
                raise zmq.Again()
            return b"signal", self._frames.pop(0)

    def test_heartbeat_refreshes_liveness_without_overwriting_signal(self):
        sub = self._sub()
        sub._socket = self._FakeSocket([
            _payload(),                                        # real BUY
            _payload(signal="HOLD", heartbeat=True, message_id="hb"),
        ])
        sig = sub.poll_signal()
        assert sig.signal == "BUY", "heartbeat must not swallow the trade signal"
        assert sub.seconds_since_last_message is not None
        assert sub.seconds_since_last_message < 1.0

    def test_malformed_frame_discarded_but_valid_batch_kept(self):
        sub = self._sub()
        sub._socket = self._FakeSocket([
            _payload(),                # valid BUY
            b"garbage not json",       # malformed — discarded alone
        ])
        sig = sub.poll_signal()
        assert sig.signal == "BUY", "one bad frame destroyed the whole batch"

    def test_no_messages_means_no_liveness(self):
        sub = self._sub()
        sub._socket = self._FakeSocket([])
        sub.poll_signal()
        assert sub.seconds_since_last_message is None

    def test_intelligence_publishes_heartbeats(self):
        src = (PROJECT_ROOT / "scripts" / "run_decoupled_intelligence.py").read_text()
        assert '"heartbeat": True' in src
        assert "settings.heartbeat_interval_seconds" in src

    def test_executor_alarms_on_gap(self):
        src = (PROJECT_ROOT / "scripts" / "run_decoupled_execution.py").read_text()
        assert "seconds_since_last_message" in src
        assert "_gap_alarmed" in src
