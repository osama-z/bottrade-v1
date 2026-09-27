"""run_decoupled_intelligence.py — Decoupled AI Intelligence Node.

Fetches data, runs the technical indicators + HMM regime filter + AI combiner,
and publishes the combined signal over ZeroMQ PUB socket.

Runs on a scheduled interval (e.g. every candle close, or continuously polling).
"""

import sys
import time
from pathlib import Path

from loguru import logger

# ── Add project root to path ──────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logging
from config.constants import TIMEFRAME_SECONDS

setup_logging()
from data.fetcher import DataFetcher
from data.pipeline import build_indicator_frame
from strategies.ai_combined import AICombinedStrategy
from strategies.base import BaseStrategy
from strategies.registry import get_strategy
from core.zmq_publisher import SignalPublisher
from core.candles import seconds_to_next_close


class IntelligenceNode:
    def __init__(self, address: str | None = None) -> None:
        address = address or settings.zmq_signal_address
        logger.info("Initializing Intelligence Node on address: {}", address)
        self.publisher = SignalPublisher(address=address)
        self.fetcher = DataFetcher()

        # News source for live headlines (LLM/sentiment components)
        news_fetcher = None
        if settings.news_api_key:
            from data.news_fetcher import NewsFetcher
            news_fetcher = NewsFetcher()

        # Strategy per pair, chosen via settings.strategy_name (registry) so
        # the decoupled node honors STRATEGY exactly like run_live.py.
        # Previously this hardcoded AICombinedStrategy, so STRATEGY=trend_following
        # was silently ignored in the decoupled deployment. Only ai_combined
        # needs the injected collaborators (news/models); rule-based strategies
        # (e.g. trend_following) construct bare and need no trained models.
        self.strategies: dict[str, BaseStrategy] = {}
        for pair in settings.trading_pairs:
            if settings.strategy_name == "ai_combined":
                self.strategies[pair] = AICombinedStrategy(
                    pair=pair,
                    use_llm=bool(settings.groq_api_key),
                    min_confidence=0.60,
                    news_fetcher=news_fetcher,
                )
            else:
                self.strategies[pair] = get_strategy(settings.strategy_name)
        logger.info("Intelligence strategy: {}", settings.strategy_name)

    def process_and_publish(self) -> None:
        """Run one iteration: fetch, strategy, publish."""
        for pair in settings.trading_pairs:
            try:
                logger.info("Intelligence Node: Analyzing {}", pair)

                # 1. Fetch — then decide on the last CLOSED candle only (never
                #    the forming one ccxt returns; backtest parity).
                df = build_indicator_frame(
                    self.fetcher, pair, settings.default_timeframe, limit=500,
                )
                if df is None:
                    continue

                # 4. Strategy Signal
                strategy = self.strategies[pair]
                trade_signal = strategy.get_signal(df, pair)

                # 5. Publish the component scores from the ACTUAL decision
                # (re-running sentiment/ML here would duplicate work and,
                # being separately computed, could disagree with the scores
                # that produced the signal — same class of bug as audit V-34)
                # getattr: only ai_combined exposes component scores; rule-based
                # strategies (trend_following) have no last_ai_signal.
                ai = getattr(strategy, "last_ai_signal", None)
                ml_score = ai.ml_score if ai else 0.0
                sentiment_score = ai.sentiment_score if ai else 0.0
                llm_score = ai.llm_score if ai else 0.0

                # 6. Publish via ZMQ. `score` is the combined AI score in
                # [-1, +1] per the publisher contract — previously this
                # field carried confidence ([0, 1]), so the audit trail
                # recorded bullish scores for bearish trades (audit V-30).
                self.publisher.publish(
                    signal=trade_signal.signal.value,
                    pair=pair,
                    score=ai.score if ai else 0.0,
                    confidence=trade_signal.confidence,
                    extra={
                        "llm_score": llm_score,
                        "ml_score": ml_score,
                        "sentiment_score": sentiment_score,
                        "reason": trade_signal.reason,
                    }
                )
                logger.info(
                    "Published signal | {} | Signal={} | Confidence={:.1%}",
                    pair, trade_signal.signal.value, trade_signal.confidence
                )

            except Exception as e:
                logger.error("Error in intelligence pass for {}: {}", pair, e, exc_info=True)

    def run(self) -> None:
        """Continuous execution loop matching default timeframe interval.

        Between analysis passes the node publishes a heartbeat every
        ``settings.heartbeat_interval_seconds`` so the execution core can
        tell "intelligence alive, market says HOLD" from "intelligence
        dead" — and so a restarted subscriber (PUB/SUB slow-joiner) is
        signal-aware within seconds instead of a full candle interval.
        """
        timeframe = settings.default_timeframe
        interval = TIMEFRAME_SECONDS.get(timeframe, 3600)
        heartbeat = settings.heartbeat_interval_seconds

        logger.info(
            "Intelligence Node running. Timeframe: {} ({}s, candle-close aligned) "
            "| heartbeat: {}s",
            timeframe, interval, heartbeat,
        )
        try:
            while True:
                self.process_and_publish()
                # Sleep until just after the NEXT candle close (aligned to UTC
                # boundaries), not `interval` from now — so passes land at
                # candle close like run_live's cron, not at process-start drift.
                next_pass = time.monotonic() + seconds_to_next_close(timeframe)
                while (remaining := next_pass - time.monotonic()) > 0:
                    time.sleep(min(heartbeat, remaining))
                    self.publisher.publish(
                        signal="HOLD", pair="", score=0.0, confidence=0.0,
                        extra={"heartbeat": True},
                    )
        except KeyboardInterrupt:
            logger.info("Intelligence Node shutting down cleanly.")
        finally:
            self.publisher.close()


if __name__ == "__main__":
    node = IntelligenceNode()
    node.run()
