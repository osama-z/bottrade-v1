"""
run_live.py — Master execution loop for NeuronTrade.

This is the script you run to start the bot:
    python scripts/run_live.py

What it does:
    1. Loads configuration from .env
    2. Initializes all components (DB, Trader, Telegram, Strategy)
    3. Sends Telegram startup message
    4. Schedules candle-close jobs for each trading pair
    5. Schedules a daily summary at midnight UTC
    6. Keeps running until Ctrl+C or /emergency_stop

Performance:
    - Uses APScheduler with async-compatible thread pool
    - Each trading pair runs independently
    - Errors in one pair never affect other pairs
    - Telegram runs in its own daemon thread
"""

import sys
import time
import signal
import uuid
from pathlib import Path

from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# ── Add project root to path ──────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from strategies.ai_combined import AICombinedStrategy
from strategies.base import BaseStrategy
from execution.paper_trader import PaperTrader
from notifications.telegram_bot import TelegramBot


# ─────────────────────────────────────────────────────────────────────────────
# Configure Logging — via the single central setup; entrypoints must not
# install their own handlers (JSON file sink + UTC timestamps live there).
# ─────────────────────────────────────────────────────────────────────────────
from config.logging_config import setup_logging

setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Candle-close timing (backtest/live parity)
# ─────────────────────────────────────────────────────────────────────────────
# The backtest treats every candle as CLOSED and acts at its close. Live must
# do the same, or paper results are not comparable to the validated backtest.
# Two pieces: (1) schedule ticks at candle-close boundaries, not free-running
# from process start (candle_close_cron below); (2) never act on the
# still-forming last candle ccxt returns (drop_forming_candle, shared with the
# decoupled node via core.candles).

from core.candles import drop_forming_candle  # noqa: E402  (after path bootstrap)


def candle_close_cron(timeframe: str, offset_seconds: int = 5) -> CronTrigger:
    """A CronTrigger aligned to candle-CLOSE boundaries in UTC.

    Fires ``offset_seconds`` after each boundary so the just-closed candle is
    final on the exchange before we fetch. Free-running IntervalTrigger fires
    at an arbitrary offset inside the candle (whenever the process happened to
    start); aligning to the close is what lets live decide on the same closed
    candle the backtest scored.
    """
    unit, n = timeframe[-1], int(timeframe[:-1])
    if unit == "m":
        return CronTrigger(minute=f"*/{n}", second=offset_seconds, timezone="UTC")
    if unit == "h":
        return CronTrigger(hour=f"*/{n}", minute=0, second=offset_seconds, timezone="UTC")
    if unit == "d":
        return CronTrigger(hour=0, minute=0, second=offset_seconds, timezone="UTC")
    if unit == "w":
        # Binance weekly candles open Monday 00:00 UTC.
        return CronTrigger(
            day_of_week="mon", hour=0, minute=0, second=offset_seconds, timezone="UTC"
        )
    raise ValueError(f"Unsupported timeframe for candle-close scheduling: {timeframe!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Global Component Registry
# ─────────────────────────────────────────────────────────────────────────────

class NeuronTradeBot:
    """
    Master orchestrator. Owns all components and the scheduling loop.
    """

    def __init__(self) -> None:
        logger.info("=" * 60)
        logger.info("  NeuronTrade — Starting up")
        logger.info(
            "  Mode: {}",
            "SHADOW (log-only, no execution)" if settings.shadow_mode
            else "PAPER TRADING",
        )
        logger.info("  Pairs: {}", ", ".join(settings.trading_pairs))
        logger.info("  Timeframe: {}", settings.default_timeframe)
        logger.info("=" * 60)

        # Fail loudly on an unsupported pair configuration rather than
        # discovering mid-run that equity math assumed a USDT quote.
        from config.pairs import validate_pairs
        for problem in validate_pairs(settings.trading_pairs):
            logger.warning("Pair config: {}", problem)

        # ── Core components ───────────────────────────────────────────────────
        # Backend driven by config (DATABASE_BACKEND): SQLite default, Postgres
        # opt-in (Task 4.1 storage factory).
        from storage.factory import get_trade_logger
        self.db = get_trade_logger()
        # Paper/shadow trading needs only PUBLIC market data — build the fetcher
        # key-free so a testnet order key is never sent to the production
        # market-data venue (which rejects it: -2008 Invalid Api-Key).
        self.fetcher = DataFetcher(public_only=True)
        self.preprocessor = DataPreprocessor()
        self.tech = TechnicalIndicators()

        # Shadow mode (Task 5.1): decide but execute nothing; log expected fills.
        self.shadow_mode = settings.shadow_mode
        from backtesting.costs import CostModel
        self._cost_model = CostModel()

        # ── Paper Trader ──────────────────────────────────────────────────────
        # db_path is SQLite-only; the Postgres backend has none (getattr → None).
        _db_path = getattr(self.db, "db_path", None)
        # Tax-lot ledger (Task 6.1): shares the SQLite file when available.
        from storage.tax_lots import TaxLotLedger
        tax_ledger = TaxLotLedger(str(_db_path)) if _db_path else None
        self.trader = PaperTrader(
            initial_balance=settings.initial_balance,
            db=self.db,
            # Persistent breaker store: a tripped circuit breaker must
            # survive restarts (Stage 0 — manual reset only).
            db_path=str(_db_path) if _db_path else None,
            tax_ledger=tax_ledger,
        )

        # ── Telegram ──────────────────────────────────────────────────────────
        self.telegram = TelegramBot(
            trader=self.trader,
            db=self.db,
            fetcher=self.fetcher,   # live prices for /force_sell exits
        )
        # Inject telegram back into trader for push alerts
        self.trader._telegram = self.telegram

        # ── Kill Switch (Stage 0) ────────────────────────────────────────────
        # Out-of-band emergency flatten: `touch KILL_SWITCH` in the project
        # root, or `kill -USR1 <pid>`. Independent of Telegram and ZMQ.
        from execution.kill_switch import KillSwitch
        self.kill_switch = KillSwitch(
            trader=self.trader,
            fetcher=self.fetcher,
            flag_path=project_root / "KILL_SWITCH",
            telegram=self.telegram,
        )

        # ── WS Order Book (Stage 2, opt-in via USE_WS_ORDER_BOOK) ────────────
        self.ws_feeds: dict = {}
        imbalance_source = None
        if settings.use_ws_order_book:
            from data.ws_depth_feed import BookImbalanceSource, WSDepthFeed
            for pair in settings.trading_pairs:
                self.ws_feeds[pair] = WSDepthFeed(pair, fetcher=self.fetcher)
            imbalance_source = BookImbalanceSource(self.ws_feeds)
            logger.info("WS order book enabled for {} pair(s)", len(self.ws_feeds))

        # ── News (feeds the LLM/sentiment components live headlines) ─────────
        news_fetcher = None
        if settings.news_api_key:
            from data.news_fetcher import NewsFetcher
            news_fetcher = NewsFetcher()

        # ── Strategies (one per pair, chosen via settings.strategy_name) ─────
        # Registry-driven so the paper run can trade the measured candidate
        # (STRATEGY=trend_following) without code changes. AICombined keeps
        # its injected collaborators; other strategies construct bare.
        from strategies.registry import get_strategy

        self.strategies: dict[str, BaseStrategy] = {}
        for pair in settings.trading_pairs:
            if settings.strategy_name == "ai_combined":
                self.strategies[pair] = AICombinedStrategy(
                    pair=pair,
                    use_llm=bool(settings.groq_api_key),  # Real LLM if key exists
                    min_confidence=0.60,
                    imbalance_source=imbalance_source,
                    news_fetcher=news_fetcher,
                )
            else:
                self.strategies[pair] = get_strategy(settings.strategy_name)
        logger.info("Live strategy: {}", settings.strategy_name)

        # ── Scheduler ────────────────────────────────────────────────────────
        self.scheduler = BlockingScheduler(timezone="UTC")

        logger.info("All components initialized")

    def start(self) -> None:
        """Start Telegram polling and schedule all jobs."""

        # 1. Start Telegram listener (daemon thread)
        self.telegram.start()

        # 2. Send startup notification
        self.telegram.send_system_start(
            balance=self.trader.balance,
            pairs=settings.trading_pairs,
        )

        # 3. Schedule candle-close job for each pair — aligned to the candle
        #    boundary (fires a few seconds after close), NOT free-running from
        #    process start, so live acts on the same closed candle the backtest
        #    scored (see candle_close_cron / drop_forming_candle).
        timeframe = settings.default_timeframe

        for pair in settings.trading_pairs:
            self.scheduler.add_job(
                func=self._process_pair,
                trigger=candle_close_cron(timeframe),
                args=[pair],
                id=f"tick_{pair.replace('/', '_')}",
                name=f"Candle tick: {pair}",
                max_instances=1,         # Never overlap jobs for same pair
                coalesce=True,           # Skip missed runs (e.g., if bot was down)
                misfire_grace_time=30,
            )
            logger.info(
                "Scheduled {} at every {} candle close (UTC-aligned)",
                pair, timeframe
            )

        # 4. Daily summary at midnight UTC
        self.scheduler.add_job(
            func=self._send_daily_summary,
            trigger=CronTrigger(hour=0, minute=0, second=0, timezone="UTC"),
            id="daily_summary",
            name="Daily PnL Summary",
        )

        # 5. Health check every 5 minutes
        self.scheduler.add_job(
            func=self._health_check,
            trigger=IntervalTrigger(minutes=5),
            id="health_check",
            name="System Health Check",
        )

        # 6. Graceful shutdown on Ctrl+C
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # 6b. Arm the kill switch (flag file + SIGUSR1) — main thread only
        self.kill_switch.start()

        # 6c. Start WS depth feeds (if enabled) — daemon threads
        for feed in self.ws_feeds.values():
            feed.start()

        logger.info("Scheduler starting — press Ctrl+C to stop")
        logger.info("=" * 60)

        # 7. Run the first tick immediately
        for pair in settings.trading_pairs:
            self._process_pair(pair)

        # 8. Enter the blocking scheduler loop
        self.scheduler.start()

    # ─── Scheduled Jobs ────────────────────────────────────────────────────────

    def _process_pair(self, pair: str) -> None:
        """
        Main job — runs once per candle close for each trading pair.

        Flow:
            Fetch → Preprocess → Indicators → AI Strategy → Paper Trader

        A per-cycle correlation ID is bound into every log record and
        persisted with the signal/trade/risk-event rows, so one cycle can
        be traced end-to-end through logs and the audit trail. Stage
        latencies are measured at each pipeline boundary.
        """
        corr = uuid.uuid4().hex[:12]
        t = {}
        t0 = time.perf_counter()
        with logger.contextualize(corr=corr, pair=pair):
            try:
                self._process_pair_inner(pair, corr, t, t0)
            except Exception as e:
                logger.error("Error processing {} candle: {}", pair, e, exc_info=True)
                self.telegram.send_error(f"Error on {pair}: {e}")

    def _process_pair_inner(self, pair: str, corr: str, t: dict, t0: float) -> None:
        logger.info("── Processing candle | {} ──", pair)

        # Step 1: Fetch latest candles
        df = self.fetcher.fetch_ohlcv(
            pair=pair,
            timeframe=settings.default_timeframe,
            limit=500,              # 500 candles is enough for all indicators
        )
        # Decide on the last CLOSED candle only — never the forming one
        # (backtest parity; the forming candle's indicators keep changing).
        df = drop_forming_candle(df, settings.default_timeframe)
        t["fetch_ms"] = (time.perf_counter() - t0) * 1000
        if df is None or len(df) < 100:
            logger.warning("Insufficient data for {} — skipping", pair)
            return

        # Step 2: Clean data
        df = self.preprocessor.process(df)

        # Step 3: Technical indicators
        df = self.tech.compute_all(df)
        t["indicators_ms"] = (time.perf_counter() - t0) * 1000 - t["fetch_ms"]

        # Step 4: Get AI signal
        strategy = self.strategies[pair]
        trade_signal = strategy.get_signal(df, pair)
        t["signal_ms"] = (
            (time.perf_counter() - t0) * 1000 - t["fetch_ms"] - t["indicators_ms"]
        )

        # Step 5: Reuse the AISignal the strategy just computed.
        # Re-running the LLM/sentiment pipeline here would double API
        # cost and latency per candle and — being non-deterministic —
        # persist component scores that differ from the actual decision.
        from ai.signal_combiner import AISignal
        from config.constants import Signal

        # getattr: only AICombined exposes component scores; rule-based
        # strategies log a neutral score row (decision still overrides below)
        ai_signal = getattr(strategy, "last_ai_signal", None)
        if ai_signal is None:
            # Strategy exited before the combiner ran (fetch failure /
            # early HOLD) — log a neutral signal, never a fabricated one.
            ai_signal = AISignal(
                signal=Signal.HOLD, score=0.0, confidence=0.0,
                is_actionable=False,
                llm_score=0.0, ml_score=0.0, sentiment_score=0.0,
            )

        # Override the signal with the strategy's final decision
        # (this ensures the filter gates and confidence gating are respected)
        from dataclasses import replace
        if trade_signal.signal == Signal.BUY:
            ai_signal = replace(ai_signal, signal=Signal.BUY, is_actionable=True)
        elif trade_signal.signal == Signal.SELL:
            ai_signal = replace(ai_signal, signal=Signal.SELL, is_actionable=True)
        else:
            ai_signal = replace(ai_signal, signal=Signal.HOLD, is_actionable=False)

        # Step 6: Execute through the paper trader — or, in shadow mode, record
        # the decision + expected fill and execute NOTHING (Task 5.1).
        self._execute_or_shadow(pair, df, ai_signal, trade_signal, corr)
        t["execute_ms"] = (
            (time.perf_counter() - t0) * 1000
            - t["fetch_ms"] - t["indicators_ms"] - t["signal_ms"]
        )

        # Step 7: record market structure (OI / taker flow / funding).
        # Binance keeps ~30 days of this history, so the bot accumulates
        # its own series for later feature testing (roadmap Step 3).
        # Best-effort: a failed fetch stores NULL and never blocks trading.
        try:
            self.db.log_market_structure(
                symbol=pair,
                open_interest=self.fetcher.fetch_open_interest(pair),
                taker_ratio=self.fetcher.fetch_taker_ratio(pair),
                funding_rate=self.fetcher.fetch_funding_rate(pair),
            )
        except Exception as e:
            logger.warning("market-structure record failed for {}: {}", pair, e)

        # Step 7: Check drawdown after every candle (marked equity —
        # an entry-valued figure is blind to unrealized losses)
        current_close = float(df["close"].iloc[-1])
        equity_now = self.trader.equity({pair: current_close})
        breached, dd = self.trader.risk.check_drawdown_limit(equity_now)
        if breached and not self.trader.is_paused:
            self.trader.is_paused = True
            self.telegram.send_drawdown_warning(abs(dd))
            logger.critical(
                "Drawdown limit breached ({:.1%}) — bot auto-paused", abs(dd)
            )

        total_ms = (time.perf_counter() - t0) * 1000
        latency_log = logger.warning if total_ms > settings.latency_warn_ms else logger.info
        latency_log(
            "Pipeline latency | total={:.0f}ms | fetch={:.0f} indicators={:.0f} "
            "signal={:.0f} execute={:.0f} (budget {}ms)",
            total_ms, t.get("fetch_ms", 0), t.get("indicators_ms", 0),
            t.get("signal_ms", 0), t.get("execute_ms", 0), settings.latency_warn_ms,
        )
        logger.info(
            "── Candle done | {} | Balance=${:.2f} | Signal={} ──",
            pair, self.trader.balance, trade_signal.signal.value
        )


    def _execute_or_shadow(self, pair, df, ai_signal, trade_signal, corr: str) -> None:
        """Shadow mode (Task 5.1): log the decision + expected fill and execute
        NOTHING — not even a paper trade. Otherwise run the paper trader."""
        if self.shadow_mode:
            from execution.shadow import build_shadow_record
            price = float(df["close"].iloc[-1])
            volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0
            rec = build_shadow_record(
                trade_signal.signal, price, self._cost_model, candle_volume=volume
            )
            self.db.log_shadow_decision(
                symbol=pair, decision=rec.decision, price=rec.price,
                expected_slippage=rec.expected_slippage, expected_fill=rec.expected_fill,
                confidence=trade_signal.confidence, reason=trade_signal.reason,
                correlation_id=corr,
                strategy=settings.strategy_name, timeframe=settings.default_timeframe,
            )
            logger.info(
                "SHADOW {} {} @ {:.6g} → expected fill {:.6g} (slippage {:.4%}) [no order]",
                pair, rec.decision, rec.price, rec.expected_fill, rec.expected_slippage,
            )
            # Telegram: alert on actionable signals only (never on every HOLD),
            # clearly marked SHADOW so it isn't mistaken for a real execution.
            if getattr(self, "telegram", None) and rec.decision in ("BUY", "SELL"):
                self.telegram.send_shadow_signal(
                    pair=pair, decision=rec.decision, price=rec.price,
                    expected_slippage=rec.expected_slippage, expected_fill=rec.expected_fill,
                    confidence=trade_signal.confidence, reason=trade_signal.reason,
                )
            return

        self.trader.process_candle(
            df=df, pair=pair, ai_signal=ai_signal, correlation_id=corr
        )

    def _send_daily_summary(self) -> None:
        """Job: send end-of-day summary via Telegram."""
        try:
            state = self.trader.get_portfolio_state()
            stats = self.db.get_stats()
            self.db.save_daily_summary(
                starting_balance=self.trader.initial_balance,
                ending_balance=state.balance,
            )
            self.telegram.send_daily_summary(state=state, stats=stats)
            logger.info("Daily summary sent")
        except Exception as e:
            logger.error("Daily summary error: {}", e)

    def _health_check(self) -> None:
        """Job: verify all components are alive every 5 minutes."""
        try:
            state = self.trader.get_portfolio_state()
            logger.debug(
                "Health check OK | Balance=${:.2f} | OpenPositions={} | Paused={}",
                state.balance, state.open_position_count, self.trader.is_paused
            )
        except Exception as e:
            logger.error("Health check failed: {}", e)
            self.telegram.send_error(f"Health check failed: {e}")

    def _shutdown(self, signum, frame) -> None:
        """Handle Ctrl+C / SIGTERM — cooperative shutdown.

        wait=True: a scheduler job may be mid-position-close; killing it
        would abandon a half-written trade (persist-before-complete).
        BlockingScheduler.start() returns once shutdown completes, so the
        main thread finishes cleanup after this handler — no sys.exit here.
        """
        logger.info("Shutdown signal received — stopping bot...")
        self.trader.is_running = False

        try:
            self.scheduler.shutdown(wait=True)
        except Exception as e:
            logger.error("Scheduler shutdown error: {}", e)
        for feed in self.ws_feeds.values():
            feed.stop()
        self.telegram.stop()
        self.db.close()
        logger.info("NeuronTrade stopped cleanly")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = NeuronTradeBot()
    bot.start()
