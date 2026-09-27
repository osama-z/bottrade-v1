"""kill_switch.py — Out-of-band emergency flatten (Stage 0).

claude.md Kill Switch & Emergency Flatten: "A separate Kill Switch
capability must exist to flatten current positions and cancel working
orders. Triggerable out-of-band (e.g., via a local file flag, OS signal,
or HTTP endpoint) independent of the ZMQ intelligence core."

Two triggers, both independent of ZMQ and Telegram:

1. Flag file — ``touch KILL_SWITCH`` in the project root (path is
   configurable). Polled every ``poll_seconds`` by a daemon thread.
   The file is deliberately NOT deleted after firing: a restart with the
   flag still present fires again immediately, so trading cannot resume
   until an operator removes the file AND manually resets the breaker.
2. OS signal — ``kill -USR1 <pid>``.

On fire (in this order):
1. Trip the circuit breaker (persists; manual reset only) and pause the
   trader, so no new trading can interleave with the flatten.
2. Flatten every open position at the live market price.
3. Notify Telegram (best effort).
"""

from __future__ import annotations

import signal as os_signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from data.fetcher import DataFetcher
    from execution.paper_trader import PaperTrader
    from notifications.telegram_bot import TelegramBot


class KillSwitch:
    def __init__(
        self,
        *,
        trader: "PaperTrader",
        fetcher: Optional["DataFetcher"] = None,
        flag_path: str | Path = "KILL_SWITCH",
        poll_seconds: float = 2.0,
        telegram: Optional["TelegramBot"] = None,
    ) -> None:
        self._trader = trader
        self._fetcher = fetcher
        self._flag_path = Path(flag_path)
        self._poll_seconds = poll_seconds
        self._telegram = telegram
        self._fired = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Arm both triggers. Must be called from the main thread so the
        SIGUSR1 handler can be registered (the file flag works regardless)."""
        try:
            os_signal.signal(os_signal.SIGUSR1, self._on_signal)
        except ValueError:
            logger.warning(
                "KillSwitch: not on main thread — SIGUSR1 trigger unavailable; "
                "flag-file trigger remains active"
            )

        # A flag left over from a previous session must fire immediately:
        # the operator asked for a flatten and a restart must not undo that.
        if self._flag_path.exists():
            logger.critical(
                "KillSwitch: flag file {} present at startup", self._flag_path
            )
            self.fire(f"flag file {self._flag_path} present at startup")
            return

        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="kill-switch"
        )
        self._thread.start()
        logger.info(
            "KillSwitch armed | flag file: {} | OS signal: SIGUSR1 (pid check: "
            "kill -USR1 <pid>)",
            self._flag_path.resolve(),
        )

    def stop(self) -> None:
        self._stop.set()

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    # ─── Triggers ──────────────────────────────────────────────────────────────

    def _on_signal(self, signum: int, frame: object) -> None:
        logger.critical("KillSwitch: SIGUSR1 received")
        self.fire("SIGUSR1")

    def _watch_loop(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            if self._flag_path.exists():
                self.fire(f"flag file {self._flag_path}")
                return

    # ─── Action ────────────────────────────────────────────────────────────────

    def fire(self, trigger: str) -> None:
        """Halt trading, then flatten everything. Idempotent."""
        if self._fired.is_set():
            return
        self._fired.set()

        logger.critical("🔴 KILL SWITCH FIRED ({}) — halting and flattening", trigger)

        # 1. Halt FIRST: breaker latches TRIPPED (persisted, manual reset
        #    only) and the pause flag stops signal processing, so no new
        #    entry can interleave with the flatten below.
        try:
            self._trader._risk.trip_circuit_breaker(
                f"kill switch: {trigger}", datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.critical("KillSwitch: breaker trip FAILED: {}", e)
        self._trader.is_paused = True

        try:
            self._trader.database.log_risk_event("kill_switch", reason=trigger)
        except Exception as e:
            logger.error("KillSwitch: audit write failed: {}", e)

        # 2. Flatten all open positions at market.
        try:
            closed = self._trader.flatten_all(self._fetcher)
        except Exception as e:
            logger.critical("KillSwitch: flatten FAILED: {}", e)
            closed = []

        summary = (
            f"KILL SWITCH ({trigger})\n"
            f"Closed {len(closed)} position(s):\n" + "\n".join(closed)
            if closed
            else f"KILL SWITCH ({trigger})\nNo open positions to flatten."
        )
        logger.critical("{}", summary)

        # 3. Notify the operator (best effort — never blocks the flatten).
        if self._telegram:
            try:
                self._telegram.send_error(summary)
            except Exception:
                pass
