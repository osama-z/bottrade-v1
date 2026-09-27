"""Tier-4 regression tests: kill switch & command-channel auth (audit V-19/V-16/V-53)."""

import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai.signal_combiner import AISignal
from config.constants import Signal
from execution.kill_switch import KillSwitch
from execution.paper_trader import PaperTrader
from notifications.telegram_bot import TelegramBot
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FakeFetcher:
    def __init__(self, price: float = 105.0):
        self.price = price

    def fetch_ticker(self, pair: str) -> dict:
        return {"last": self.price, "bid": self.price}


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
    return AISignal(
        signal=Signal.BUY, score=0.8, confidence=0.9, is_actionable=True,
        llm_score=0.0, ml_score=0.0, sentiment_score=0.0,
    )


def trader_with_position(tmp_path) -> tuple[PaperTrader, TradeLogger]:
    db_path = str(tmp_path / "test.db")
    db = TradeLogger(db_path=db_path)
    trader = PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path)
    trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=buy_signal())
    assert len(db.get_open_trades()) == 1
    return trader, db


# ─── V-19: kill switch ─────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_fire_flattens_trips_and_pauses(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(),
                        flag_path=tmp_path / "KILL_SWITCH")
        ks.fire("test trigger")

        assert ks.fired
        assert trader._risk.circuit_breaker_tripped()
        assert "kill switch" in (trader._risk.store.load().reason or "")
        assert trader.is_paused
        assert len(db.get_open_trades()) == 0
        closed = db.get_trade_history(limit=1)[0]
        assert closed["exit_reason"] == "kill_switch"

    def test_fire_is_idempotent(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(),
                        flag_path=tmp_path / "KILL_SWITCH")
        ks.fire("first")
        balance_after = trader._balance
        ks.fire("second")
        assert trader._balance == balance_after

    def test_flag_file_present_at_startup_fires_immediately(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        flag = tmp_path / "KILL_SWITCH"
        flag.touch()
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(), flag_path=flag)
        ks.start()
        assert ks.fired
        assert len(db.get_open_trades()) == 0

    def test_flag_file_created_later_fires_watch_loop(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        flag = tmp_path / "KILL_SWITCH"
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(),
                        flag_path=flag, poll_seconds=0.05)
        ks.start()
        assert not ks.fired
        flag.touch()
        # fired is set at the START of fire(); the flatten completes after.
        # Poll the outcome (no open trades), not the flag, to avoid racing
        # the in-progress flatten under suite load.
        deadline = time.time() + 8.0
        while len(db.get_open_trades()) > 0 and time.time() < deadline:
            time.sleep(0.02)
        assert ks.fired, "watch loop did not react to the flag file"
        assert len(db.get_open_trades()) == 0, "flatten did not complete in time"
        ks.stop()

    def test_flatten_without_fetcher_uses_entry_price(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        results = trader.flatten_all(fetcher=None)
        assert len(results) == 1
        assert len(db.get_open_trades()) == 0

    def test_trader_survives_breaker_after_fire(self, tmp_path):
        """After the kill switch fires, process_candle must not open trades."""
        trader, db = trader_with_position(tmp_path)
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(),
                        flag_path=tmp_path / "KILL_SWITCH")
        ks.fire("test")
        trader.process_candle(df=make_df(), pair="ETH/USDT", ai_signal=buy_signal())
        assert len(db.get_open_trades()) == 0


# ─── V-16: Telegram sender authorization ───────────────────────────────────────

def _update(chat_id, text="/pause"):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=42),
        message=SimpleNamespace(text=text),
    )


class TestTelegramAuth:
    def _bot(self) -> TelegramBot:
        bot = TelegramBot()
        bot.chat_id = "12345"
        return bot

    def test_configured_chat_is_authorized(self):
        assert self._bot()._is_authorized(_update(12345)) is True

    def test_other_chat_is_rejected(self):
        assert self._bot()._is_authorized(_update(99999)) is False

    def test_missing_chat_is_rejected(self):
        bot = self._bot()
        update = SimpleNamespace(effective_chat=None, effective_user=None, message=None)
        assert bot._is_authorized(update) is False

    def test_unconfigured_chat_id_rejects_everyone(self):
        bot = TelegramBot()
        bot.chat_id = ""
        assert bot._is_authorized(_update(12345)) is False

    def test_every_command_handler_checks_authorization(self):
        src = (PROJECT_ROOT / "notifications" / "telegram_bot.py").read_text()
        # start, status, trades, pause, resume, force_sell each guard
        # (help delegates to the guarded start handler)
        assert src.count("if not self._is_authorized(update):") >= 6


def _raise(*a, **k):
    raise RuntimeError("boom")


class TestKillSwitchErrorPaths:
    """The kill switch must still FLATTEN even when a non-critical step fails,
    and must never crash on a downstream error (safety code has no second try)."""

    def test_flatten_still_happens_if_breaker_trip_fails(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        trader._risk.trip_circuit_breaker = _raise           # halting fails
        KillSwitch(trader=trader, fetcher=FakeFetcher(),
                   flag_path=tmp_path / "K").fire("t")
        assert len(db.get_open_trades()) == 0                # flattened anyway

    def test_audit_failure_does_not_block_flatten(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        trader._db.log_risk_event = _raise                   # audit write fails
        KillSwitch(trader=trader, fetcher=FakeFetcher(),
                   flag_path=tmp_path / "K").fire("t")
        assert len(db.get_open_trades()) == 0

    def test_flatten_failure_is_survived(self, tmp_path):
        trader, db = trader_with_position(tmp_path)
        trader.flatten_all = _raise                          # flatten itself fails
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(), flag_path=tmp_path / "K")
        ks.fire("t")                                         # must NOT raise
        assert ks.fired

    def test_telegram_notify_failure_is_swallowed(self, tmp_path):
        trader, db = trader_with_position(tmp_path)

        class BadTG:
            def send_error(self, msg):
                raise RuntimeError("telegram down")

        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(),
                        flag_path=tmp_path / "K", telegram=BadTG())
        ks.fire("t")                                         # must NOT raise
        assert ks.fired and len(db.get_open_trades()) == 0

    def test_sigusr1_handler_fires_and_flattens(self, tmp_path):
        import signal as os_signal
        trader, db = trader_with_position(tmp_path)
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(), flag_path=tmp_path / "K")
        ks._on_signal(os_signal.SIGUSR1, None)
        assert ks.fired and len(db.get_open_trades()) == 0

    def test_sigusr1_registration_failure_still_arms_flag_watcher(self, tmp_path, monkeypatch):
        import execution.kill_switch as ksmod
        trader, db = trader_with_position(tmp_path)

        def _bad_signal(*a, **k):
            raise ValueError("not on main thread")

        monkeypatch.setattr(ksmod.os_signal, "signal", _bad_signal)
        ks = KillSwitch(trader=trader, fetcher=FakeFetcher(), flag_path=tmp_path / "K")
        ks.start()                                           # SIGUSR1 fails → flag watcher still starts
        assert ks._thread is not None and ks._thread.is_alive()
        ks.stop()
