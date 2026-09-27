"""run_decoupled_execution.py — Decoupled Execution Core Node.

Connects to the ZeroMQ PUB socket from the intelligence process, polls
timestamped signals, applies the fail-safe stale-signal check, and forwards
valid actions to the PaperTrader and RiskManager execution core.

Also hosts the 24-hour Walk-Forward Performance Validator job (Stage 5)
and runs the Telegram bot interface.
"""

import sys
import time
import signal
from pathlib import Path

from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ── Add project root to path ──────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logging
from config.constants import Signal

setup_logging()
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from execution.paper_trader import PaperTrader
from storage.trade_logger import TradeLogger
from notifications.telegram_bot import TelegramBot
from core.zmq_subscriber import SignalSubscriber
from risk.walk_forward import WalkForwardValidator


class ExecutionNode:
    def __init__(self, address: str | None = None) -> None:
        address = address or settings.zmq_signal_address
        logger.info("Initializing Execution Node on address: {}", address)
        self.db = TradeLogger()
        self.fetcher = DataFetcher()
        self.preprocessor = DataPreprocessor()
        self.tech = TechnicalIndicators()

        # Initialize PaperTrader (which holds RiskManager internally)
        # Use our persistent SQLite database file for the circuit breaker store
        self.trader = PaperTrader(
            initial_balance=settings.initial_balance,
            db=self.db,
            db_path=str(self.db.db_path),
        )

        # Initialize subscriber with max age check (Stage 4).
        # Threshold comes from configuration, not a hardcoded literal.
        self.subscriber = SignalSubscriber(
            address=address,
            max_signal_age_seconds=settings.stale_signal_seconds,
        )
        # Signal-gap alarm state: True while the intelligence feed is silent
        # beyond the stale threshold (alarms once per outage, not per poll).
        self._gap_alarmed = False

        # Initialize Walk-Forward Validator (Stage 5)
        self.validator = WalkForwardValidator(risk_manager=self.trader.risk)

        # Telegram
        self.telegram = TelegramBot(
            trader=self.trader,
            db=self.db,
            fetcher=self.fetcher,   # live prices for /force_sell exits
        )
        self.trader._telegram = self.telegram

        # Kill Switch (Stage 0): out-of-band emergency flatten via
        # `touch KILL_SWITCH` in the project root or `kill -USR1 <pid>`,
        # independent of the ZMQ intelligence core and Telegram.
        from execution.kill_switch import KillSwitch
        self.kill_switch = KillSwitch(
            trader=self.trader,
            fetcher=self.fetcher,
            flag_path=project_root / "KILL_SWITCH",
            telegram=self.telegram,
        )

        # Scheduler for background jobs (SL/TP check & Walk-Forward check)
        self.scheduler = BackgroundScheduler(timezone="UTC")

        self.is_running = True

    def start(self) -> None:
        """Start execution processes, listeners, and jobs."""
        # 1. Start Telegram
        self.telegram.start()
        self.telegram.send_system_start(
            balance=self.trader.balance,
            pairs=settings.trading_pairs,
        )

        # 2. Schedule SL/TP exit checks (run every 10 seconds to check against current candle prices)
        self.scheduler.add_job(
            func=self._check_active_exits,
            trigger=IntervalTrigger(seconds=10),
            id="exit_checks",
            name="SL/TP Exit Checks",
            max_instances=1,
            coalesce=True,
        )

        # 3. Schedule Walk-Forward performance verification (Stage 5 — every 24h)
        self.scheduler.add_job(
            func=self._run_walk_forward,
            trigger=IntervalTrigger(days=1),
            id="walk_forward",
            name="Daily Walk-Forward Validation",
            max_instances=1,
            coalesce=True,
        )

        # 4. Graceful shutdown handler
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # 4b. Arm the kill switch (flag file + SIGUSR1) — main thread only
        self.kill_switch.start()

        self.scheduler.start()
        logger.info("Execution Node scheduler started.")

        # 5. Run ZMQ signal polling loop, then tear down cooperatively
        try:
            self._run_polling_loop()
        finally:
            self._cleanup()

    def _run_polling_loop(self) -> None:
        """Main thread loop: poll for signals from the intelligence node."""
        logger.info("Starting ZMQ polling loop. Monitoring signals...")

        # Dedup key of the last signal acted upon. poll_signal() keeps
        # returning the same cached message as "fresh" until it goes stale,
        # so without this guard one BUY would be re-executed every poll.
        last_acted_key: str | None = None

        while self.is_running:
            try:
                # Poll signal from socket (non-blocking, drains buffer to latest)
                sig = self.subscriber.poll_signal()

                # Liveness alarm: with heartbeats arriving every
                # heartbeat_interval_seconds, a gap beyond the stale
                # threshold means the intelligence process is down — alarm
                # loudly instead of HOLDing in silence.
                gap = self.subscriber.seconds_since_last_message
                if gap is not None:
                    if gap > settings.stale_signal_seconds and not self._gap_alarmed:
                        self._gap_alarmed = True
                        msg = (
                            f"Intelligence feed SILENT for {gap:.0f}s "
                            f"(threshold {settings.stale_signal_seconds}s) — "
                            "execution is failing safe (HOLD)"
                        )
                        logger.critical("{}", msg)
                        self.telegram.send_error(msg)
                    elif gap <= settings.stale_signal_seconds and self._gap_alarmed:
                        self._gap_alarmed = False
                        logger.info("Intelligence feed recovered")

                if sig.is_fresh:
                    dedup_key = (
                        sig.message_id
                        or f"{sig.pair}|{sig.timestamp_utc.isoformat()}"
                    )
                    if dedup_key == last_acted_key:
                        # Same message as last iteration — already handled.
                        time.sleep(1)
                        continue
                    # Mark consumed BEFORE acting: at-most-once execution is
                    # the safe direction for order placement.
                    last_acted_key = dedup_key

                    logger.info(
                        "Received fresh signal | {} | Action={} | Score={:.2f} | Confidence={:.1%}",
                        sig.pair, sig.signal, sig.score, sig.confidence
                    )

                    # Convert to our local Signal Enum
                    action = Signal(sig.signal)

                    # Build an AISignal object for paper trader.
                    # Component scores travel in the message's `extra` payload;
                    # default to 0.0 if the publisher omitted one.
                    from ai.signal_combiner import AISignal
                    ai_sig = AISignal(
                        signal=action,
                        score=sig.score,
                        confidence=sig.confidence,
                        is_actionable=True,
                        llm_score=float(sig.extra.get("llm_score", 0.0)),
                        ml_score=float(sig.extra.get("ml_score", 0.0)),
                        sentiment_score=float(sig.extra.get("sentiment_score", 0.0)),
                    )

                    # Fetch latest data for execution mapping.
                    # 500 candles so compute_all() has enough history for its
                    # indicators (ATR is required for position sizing).
                    df = self.fetcher.fetch_ohlcv(
                        pair=sig.pair,
                        timeframe=settings.default_timeframe,
                        limit=500,
                    )
                    if df is not None and not df.empty:
                        df = self.preprocessor.process(df)
                        df = self.tech.compute_all(df)
                        # Process candle through paper trader. The ZMQ
                        # message_id doubles as the correlation ID, joining
                        # this signal to its trade/risk events and log
                        # records across BOTH processes.
                        with logger.contextualize(corr=dedup_key, pair=sig.pair):
                            self.trader.process_candle(
                                df=df, pair=sig.pair, ai_signal=ai_sig,
                                correlation_id=dedup_key,
                            )

                # Small sleep to prevent CPU spinning
                time.sleep(1)

            except Exception as e:
                logger.error("Error in signal polling: {}", e)
                time.sleep(2)

    def _check_active_exits(self) -> None:
        """Fetch latest prices and check if any open position stop loss or take profit hit."""
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            pair = trade["symbol"]
            try:
                # Fetch recent candles to check high/low against SL/TP levels
                df = self.fetcher.fetch_ohlcv(
                    pair=pair,
                    timeframe=settings.default_timeframe,
                    limit=5,
                )
                if df is not None and not df.empty:
                    df = self.preprocessor.process(df)
                    latest_candle = df.iloc[-1]

                    # Check exits
                    exit_match = self.trader.risk.check_position_exits(
                        trade,
                        candle_high=float(latest_candle["high"]),
                        candle_low=float(latest_candle["low"]),
                        candle_open=float(latest_candle["open"]),
                    )
                    if exit_match:
                        reason, price = exit_match
                        logger.warning(
                            "Position exit hit! | Trade ID: {} | Pair: {} | Reason: {} @ ${:.2f}",
                            trade["id"], pair, reason, price
                        )
                        # Close position (PnL is computed and persisted inside)
                        self.trader._close_position(trade, price, reason)

            except Exception as e:
                logger.error("Error checking exits for {}: {}", pair, e)

    def _run_walk_forward(self) -> None:
        """Run daily performance check and trip breaker if underperforming (Stage 5)."""
        logger.info("Starting scheduled Walk-Forward performance validation pass...")
        try:
            # Retrieve last 100 closed trades for validation
            history = self.db.get_trade_history(limit=100)
            result = self.validator.evaluate(history)

            logger.info(
                "Walk-Forward result | Evaluated: {} | Win Rate: {:.1%} | Sharpe: {:.3f} | Breaker Tripped: {}",
                result.evaluated, result.win_rate, result.sharpe, result.breaker_tripped
            )

            if result.breaker_tripped:
                logger.critical("Walk-Forward validation FAILED: {} — tripping circuit breaker", result.reason)
                self.telegram.send_error(f"⚠️ Walk-Forward Validation Failed!\n{result.reason}\nBot is now paused.")

        except Exception as e:
            logger.error("Error in Walk-Forward evaluation job: {}", e)

    def _shutdown(self, signum, frame) -> None:
        """Signal handler: ONLY set flags — no teardown, no sys.exit.

        Tearing down inside the handler (the previous design) could
        interrupt an APScheduler worker mid-position-close, abandoning a
        half-written trade. The polling loop observes ``is_running`` and
        returns; ``_cleanup()`` then stops everything cooperatively.
        """
        logger.info("Shutdown signal received — stopping Execution Node...")
        self.is_running = False
        self.trader.is_running = False

    def _cleanup(self) -> None:
        """Cooperative teardown, run on the main thread after the polling
        loop exits: finish in-flight jobs, then release resources."""
        try:
            # wait=True: never abandon a worker mid-DB-write (persist-
            # before-complete would be violated by killing it here).
            self.scheduler.shutdown(wait=True)
        except Exception as e:
            logger.error("Scheduler shutdown error: {}", e)
        self.telegram.stop()
        self.subscriber.close()
        self.db.close()
        logger.info("Execution Node stopped cleanly.")


if __name__ == "__main__":
    node = ExecutionNode()
    node.start()
