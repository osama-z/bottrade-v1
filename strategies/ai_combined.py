"""
AI Combined Strategy — the main production strategy for Phase 2.

Uses all three AI components (Claude + XGBoost + Sentiment) to generate
trading signals. Inherits from BaseStrategy for full compatibility with
the Phase 1 backtesting engine.

This is the strategy you run in both backtesting AND live trading.
"""

import pandas as pd
from loguru import logger
from typing import Optional

from strategies.base import BaseStrategy, TradeSignal
from strategies.context import MarketContext, build_replay_context
from config.constants import Signal, MIN_SIGNAL_CONFIDENCE
from config.settings import settings

from ai.llm_agent import LLMAgent
from ai.ml_predictor import MLPredictor
from ai.regime_detector import RegimeDetector
from ai.sentiment_analyzer import SentimentAnalyzer
from ai.signal_combiner import SignalCombiner, AISignal
from indicators.technical import TechnicalIndicators


class AICombinedStrategy(BaseStrategy):
    """
    The flagship NeuronTrade strategy — combines Claude, XGBoost, and Sentiment.

    For backtesting: Uses XGBoost + Sentiment only (no Claude API calls per candle)
    For live trading: Uses all three components

    Parameters:
        use_llm: Whether to call Claude API (True for live, False for backtesting)
        target_periods: Periods ahead for XGBoost to predict (default: 1)
        min_confidence: Minimum combined confidence to act on a signal
    """

    name = "ai_combined"

    def __init__(
        self,
        pair: str = "BTC/USDT",
        use_llm: bool = False,          # Disable LLM for backtesting (too many API calls)
        target_periods: int = 1,
        min_confidence: float = MIN_SIGNAL_CONFIDENCE,
        news_headlines: Optional[list[str]] = None,
        imbalance_source=None,   # optional live WS book (BookImbalanceSource)
        news_fetcher=None,       # optional NewsFetcher for LIVE headlines
        *,
        # Optional collaborator injection (tests / alternative components).
        # None → construct the production default. Keyword-only so casual
        # positional use can't silently swap a collaborator.
        sentiment=None,
        predictor=None,
        llm=None,
        combiner=None,
        fetcher=None,
        preprocessor=None,
    ) -> None:
        self.pair = pair
        self.use_llm = use_llm
        self.target_periods = target_periods
        self.min_confidence = min_confidence
        self.news_headlines = news_headlines or []
        # Injected order-book imbalance source (Stage 2 WS book). None →
        # fall back to the REST snapshot. Both use base-quantity units.
        self._imbalance_source = imbalance_source
        # Injected news source: without it the LLM/sentiment components
        # reason over the static constructor list (usually EMPTY) — i.e.
        # the LLM leg duplicates the indicator analysis the ML already
        # does instead of contributing what it's actually good at.
        self._news_fetcher = news_fetcher

        # AI components — injected or production defaults
        self._sentiment = sentiment or SentimentAnalyzer()
        self._predictor = predictor or MLPredictor(pair=pair)
        self._llm = llm or LLMAgent(mock=not use_llm)
        # Roadmap Task 1.1 — Decouple Noise from Execution Path.
        # The production combiner runs MATH-ONLY: LLM and VADER sentiment are
        # decoupled from live execution (neither waited on nor weighed).
        # Inject a full-fusion SignalCombiner(math_only=False) for showcase.
        self._combiner = combiner or SignalCombiner(
            min_confidence=min_confidence,
            buy_threshold=0.25,
            sell_threshold=-0.25,
            math_only=True,
        )
        self._ti = TechnicalIndicators()

        from data.fetcher import DataFetcher
        from data.preprocessor import DataPreprocessor
        self._fetcher = fetcher or DataFetcher()
        self._preprocessor = preprocessor or DataPreprocessor()

        # The combined AISignal from the most recent get_signal() call.
        # Callers that need the component scores (e.g. run_live's audit
        # logging) read this instead of re-running the LLM/sentiment
        # pipeline — a second run doubles API cost and, being
        # non-deterministic, logs scores that differ from the decision.
        self.last_ai_signal: Optional[AISignal] = None

        # Try to load saved ML model
        self._model_loaded = self._predictor.load()
        if not self._model_loaded:
            math_only = getattr(self._combiner, "math_only", False)
            if math_only:
                logger.warning(
                    "No trained XGBoost model found for {} — run scripts/train_model.py first. "
                    "In math-only execution (Roadmap Task 1.1) the combiner has no validated "
                    "signal and will HOLD until a model is trained.",
                    pair
                )
            else:
                logger.warning(
                    "No trained XGBoost model found for {} — run scripts/train_model.py first. "
                    "Strategy will use LLM + Sentiment only until model is trained.",
                    pair
                )

        # HMM Regime Detector — acts as a FILTER ONLY, never initiates trades.
        # Blocks BUY entries when market is Bearish (state 0) or Choppy (state 1).
        self._regime = RegimeDetector(pair=pair)
        self._regime_loaded = self._regime.load()
        if not self._regime_loaded:
            logger.info(
                "No HMM regime model found for {} — regime filter disabled. "
                "Run scripts/train_regime.py to enable.",
                pair
            )

    # ── Public collaborator access (read-only) ────────────────────────────
    # Scripts (walk_forward, run_live) need these; exposing properties
    # keeps the `_`-prefixed storage private-by-convention without callers
    # reaching through it.

    @property
    def regime(self):
        """The HMM regime detector (read-only)."""
        return self._regime

    @property
    def predictor(self):
        """The ML predictor (read-only)."""
        return self._predictor

    def train_regime(self, df_train: pd.DataFrame) -> None:
        """Fit + persist the HMM regime detector and mark it active.

        Mirrors train_model(); walk-forward retrains per window. Without
        this method callers had to mutate _regime/_regime_loaded directly.
        """
        self._regime.fit(df_train)
        self._regime.save()
        self._regime_loaded = True

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals for the full backtest DataFrame.

        For backtesting efficiency, uses XGBoost predictions + Sentiment only.
        Claude is NOT called per candle (would be thousands of API calls).

        Returns:
            Series of +1 (buy), -1 (sell), 0 (hold) aligned with df.index
        """
        from indicators.features import FeatureEngineer

        signals = pd.Series(0, index=df.index, dtype=int)

        # ── Sentiment (static across backtest — use current headlines) ────────
        sentiment_result = self._sentiment.analyze(self.news_headlines)

        # ── XGBoost predictions (if model is trained) ─────────────────────────
        if self._model_loaded:
            fe = FeatureEngineer()
            try:
                X, _ = fe.build_features(df, target_periods=self.target_periods)
            except Exception as e:
                logger.warning("Feature engineering failed: {} — using zero signals", e)
                return signals

            # Predict for each row using the model
            for i in range(len(X)):
                try:
                    row_X = X.iloc[:i+1]  # Up to current row (no lookahead)
                    if len(row_X) < 2:
                        continue

                    ml_pred = self._predictor.predict(row_X)
                    combined = self._combiner.combine(
                        llm_analysis=None,
                        ml_prediction=ml_pred,
                        sentiment_result=sentiment_result,
                    )

                    # Map to signal series using original df index
                    original_idx = X.index[i]
                    if original_idx in signals.index:
                        if combined.is_actionable:
                            signals.loc[original_idx] = 1 if combined.signal == Signal.BUY else -1

                except Exception as e:
                    logger.debug("Signal generation error at row {}: {}", i, e)
                    continue

        else:
            # Fallback: use composite_score from technical indicators as a proxy
            logger.info("Using composite_score fallback for backtesting (no trained model)")
            if "composite_score" in df.columns:
                buy_mask = df["composite_score"] > 0.3
                sell_mask = df["composite_score"] < -0.3
                signals[buy_mask] = 1
                signals[sell_mask] = -1

        n_buy = (signals == 1).sum()
        n_sell = (signals == -1).sum()
        logger.info("AI Combined signals: {} buys, {} sells", n_buy, n_sell)

        return signals

    def _build_live_context(self, df: pd.DataFrame, pair: str) -> MarketContext:
        """Fetch everything decide() needs. ALL I/O lives here.

        Each piece degrades to None on failure so decide() can distinguish
        "filter data unavailable" from a genuine market reading.
        """
        df_1d = df_4h = None
        try:
            # Macro-context frames are slow-changing and re-fetched every cycle
            # → cache them (a faster decision loop reuses them across cycles).
            df_1d = self._ti.compute_all(
                self._preprocessor.process(
                    self._fetcher.fetch_ohlcv(pair, timeframe="1d", limit=100,
                                              use_cache=True)
                )
            )
            df_4h = self._ti.compute_all(
                self._preprocessor.process(
                    self._fetcher.fetch_ohlcv(pair, timeframe="4h", limit=100,
                                              use_cache=True)
                )
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch macro trend frames: {} — macro filter will block BUYs", e
            )
            df_1d = df_4h = None

        try:
            funding_rate = self._fetcher.fetch_funding_rate(pair)
        except Exception:
            funding_rate = None
        try:
            if self._imbalance_source is not None:
                # Live WS book (Stage 2): returns None while unsynced/stale
                imbalance = self._imbalance_source.get_imbalance(pair)
            else:
                imbalance = self._fetcher.fetch_order_book_imbalance(pair, percentage=0.01)
        except Exception:
            imbalance = None

        # Fresh headlines for the LLM/sentiment components. Failure falls
        # back to the static constructor list — never fabricated content.
        headlines = tuple(self.news_headlines)
        if self._news_fetcher is not None:
            try:
                articles = self._news_fetcher.fetch_pair_news(pair, hours_back=12)
                fetched = tuple(a["title"] for a in articles if a.get("title"))
                if fetched:
                    headlines = fetched
            except Exception as e:
                logger.warning("News fetch failed: {} — using static headlines", e)

        return MarketContext(
            df=df,
            df_1d=df_1d,
            df_4h=df_4h,
            funding_rate=funding_rate,
            imbalance=imbalance,
            news_headlines=headlines,
        )

    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """
        Get the latest signal for live trading — uses all three AI components.

        Thin wrapper: builds a live MarketContext (all fetching), then runs
        the pure decision path. Backtests run the SAME decide() via
        generate_signals_via_decide() — that is the parity guarantee.
        """
        self.last_ai_signal = None  # reset; set once the combiner has run
        try:
            ctx = self._build_live_context(df, pair)
            return self.decide(ctx, pair)
        except Exception as e:
            logger.error("AICombinedStrategy.get_signal failed: {}", e, exc_info=True)
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                pair=pair,
                price=df["close"].iloc[-1],
                reason=f"Strategy error: {e}",
            )

    def decide(self, ctx: MarketContext, pair: str, verbose: bool = True) -> TradeSignal:
        """Pure decision function: no network, no filesystem, no training.

        Every filter gate lives here so live and replay share one code
        path. Filters whose context field is None (funding, imbalance in
        replay) pass through — they are parity exceptions, reported by the
        backtest runner. Missing 1d/4h frames BLOCK buys (same as a live
        macro-fetch failure).
        """
        df = ctx.df
        price = df["close"].iloc[-1]
        self.last_ai_signal = None  # reset; set once the combiner has run

        # ── 1. Daily (1D) and 4-Hour (4H) Macro Trends ────────────────────────
        try:
            if ctx.df_1d is None or ctx.df_4h is None or ctx.df_1d.empty or ctx.df_4h.empty:
                macro_trend_ok = False
            else:
                price_1d = ctx.df_1d["close"].iloc[-1]
                ema_50_1d = ctx.df_1d["EMA_50"].iloc[-1]
                supertrend_dir_1d = ctx.df_1d["Supertrend_dir"].iloc[-1]

                price_4h = ctx.df_4h["close"].iloc[-1]
                ema_50_4h = ctx.df_4h["EMA_50"].iloc[-1]
                supertrend_dir_4h = ctx.df_4h["Supertrend_dir"].iloc[-1]

                macro_trend_ok = (
                    price_1d > ema_50_1d and supertrend_dir_1d == 1 and
                    price_4h > ema_50_4h and supertrend_dir_4h == 1
                )
        except Exception as e:
            logger.warning("Failed to calculate macro trend indicators: {} — skipping macro checks", e)
            macro_trend_ok = False

        # ── 2. Micro Timing (RSI oversold and/or MACD bullish cross) ──────────
        # `buy_timing_require_both` defaults to False (OR). With AND, this
        # gate fired on 0.03% of candles and the strategy never traded.
        rsi_val = df["RSI"].iloc[-1]

        macd_curr = df["MACD"].iloc[-1]
        macd_sig_curr = df["MACD_signal"].iloc[-1]
        macd_prev = df["MACD"].iloc[-2]
        macd_sig_prev = df["MACD_signal"].iloc[-2]
        macd_cross_up = macd_curr > macd_sig_curr and macd_prev <= macd_sig_prev

        rsi_oversold = rsi_val < settings.buy_rsi_max
        timing_ok = (
            (rsi_oversold and macd_cross_up)
            if settings.buy_timing_require_both
            else (rsi_oversold or macd_cross_up)
        )

        # ── 3. Volume Validation (> configured multiple of SMA20) ─────────────
        vol_curr = df["volume"].iloc[-1]
        vol_sma = df["volume"].rolling(20).mean().iloc[-1]
        volume_ok = vol_curr > settings.buy_volume_multiple * vol_sma

        # ── 4. Perpetual Futures Funding Rate (None → parity exception) ───────
        funding_rate = ctx.funding_rate
        funding_ok = funding_rate is None or funding_rate <= 0.0005  # Max 0.05%

        # ── 5. Order Book Imbalance (None → parity exception) ─────────────────
        imbalance = ctx.imbalance
        imbalance_ok = imbalance is None or imbalance <= 1.5

        # ── Run AI Intelligence components ────────────────────────────────────
        # Roadmap Task 1.1 — Decouple Noise from Execution Path. In math-only
        # mode the execution path must not WAIT ON the LLM or VADER sentiment,
        # so their (network / NLP) calls are skipped entirely — the combiner
        # would drop them anyway. They remain available in showcase mode
        # (full-fusion combiner injected → math_only=False).
        math_only = getattr(self._combiner, "math_only", False)

        ml_prediction = None
        if self._model_loaded:
            from indicators.features import FeatureEngineer
            fe = FeatureEngineer()
            X, _ = fe.build_features(df, target_periods=self.target_periods)
            ml_prediction = self._predictor.predict(X)

        sentiment_result = None
        llm_analysis = None
        if not math_only:
            indicators = self._ti.get_latest_signals(df)
            sentiment_result = self._sentiment.analyze(list(ctx.news_headlines))
            if self.use_llm:
                llm_analysis = self._llm.analyze(
                    indicators=indicators,
                    news_headlines=list(ctx.news_headlines),
                    pair=pair,
                )
            else:
                llm_analysis = self._llm.analyze(indicators=indicators, pair=pair)

        # Get ADX/ATR parameters from latest 1H candle for dynamic weighting
        adx_val = float(df["ADX"].iloc[-1]) if "ADX" in df.columns else None
        atr_val = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else None
        atr_sma_val = float(df["ATR"].rolling(20).mean().iloc[-1]) if "ATR" in df.columns else None

        # ── Combine all signals with Dynamic Weighting ────────────────────────
        ai_signal: AISignal = self._combiner.combine(
            llm_analysis=llm_analysis,
            ml_prediction=ml_prediction,
            sentiment_result=sentiment_result,
            adx=adx_val,
            atr=atr_val,
            atr_sma=atr_sma_val,
        )
        self.last_ai_signal = ai_signal

        if verbose:
            ai_signal.print_summary()

        # ── HMM Regime Detection (shared by BUY and SELL filters) ────────
        regime_ok = True
        regime_reason = ""
        if self._regime_loaded:
            regime = self._regime.predict(df)
            # Measured: the HMM labels 86% of hourly candles "Choppy", so
            # blocking on Choppy is a ~15x trade-count cut — effectively an
            # off switch. Default blocks only Bearish (7.5% of candles).
            blocked_states = (0,) if settings.regime_block_bearish_only else (0, 1)
            if regime["state"] in blocked_states:
                regime_ok = False
                regime_reason = (
                    f"HMM Regime={regime['name']} "
                    f"(confidence={regime['confidence']:.0%})"
                )

        # ── Apply Validation Filters to BUY Actions ──────────────────────
        if ai_signal.signal == Signal.BUY and ai_signal.is_actionable:
            if not regime_ok:
                logger.info("BUY blocked by HMM regime filter: {}", regime_reason)

            filters_ok = macro_trend_ok and timing_ok and volume_ok and funding_ok and imbalance_ok and regime_ok
            if not filters_ok:
                reasons = []
                if not macro_trend_ok:
                    reasons.append("Macro Trend (1D or 4H EMA/Supertrend DOWN)")
                if not timing_ok:
                    reasons.append(
                        f"Micro Timing (RSI={rsi_val:.1f} >= {settings.buy_rsi_max:.0f} or MACD no cross)"
                    )
                if not volume_ok:
                    reasons.append(
                        f"Volume (Curr={vol_curr:.1f} <= {settings.buy_volume_multiple:.1f}x SMA={vol_sma:.1f})"
                    )
                if not funding_ok:
                    reasons.append(f"Funding Rate too positive ({funding_rate*100:.4f}% > 0.05%)")
                if not imbalance_ok:
                    reasons.append(f"Order book imbalance ask/bid ({imbalance:.2f} > 1.5)")
                if not regime_ok:
                    reasons.append(regime_reason)

                logger.info("BUY signal skipped due to filter checks: {}", ", ".join(reasons))
                return TradeSignal(
                    signal=Signal.HOLD,
                    confidence=ai_signal.confidence,
                    pair=pair,
                    price=price,
                    reason=f"BUY filters failed: {', '.join(reasons)}",
                )

        # ── Apply Validation Filters to SELL Actions ─────────────────────
        elif ai_signal.signal == Signal.SELL and ai_signal.is_actionable:
            # SELL gets a lighter filter gate: regime + volume only
            # (we want to exit quickly, but not on noise in dead markets)
            if not regime_ok and not volume_ok:
                logger.info("SELL signal skipped: weak regime + low volume")
                return TradeSignal(
                    signal=Signal.HOLD,
                    confidence=ai_signal.confidence,
                    pair=pair,
                    price=price,
                    reason="SELL filters failed: weak regime + low volume",
                )

        # Convert to TradeSignal
        if ai_signal.signal == Signal.BUY and ai_signal.is_actionable:
            reason = (
                llm_analysis.reasoning
                if (llm_analysis and not llm_analysis.mock)
                else f"AI score={ai_signal.score:+.2f}, ML p_up={ml_prediction.p_up:.0%}" if ml_prediction else f"AI score={ai_signal.score:+.2f}"
            )
            return TradeSignal(
                signal=Signal.BUY,
                confidence=ai_signal.confidence,
                pair=pair,
                price=price,
                reason=reason,
            )
        elif ai_signal.signal == Signal.SELL and ai_signal.is_actionable:
            reason = (
                llm_analysis.reasoning
                if (llm_analysis and not llm_analysis.mock)
                else f"AI score={ai_signal.score:+.2f}"
            )
            return TradeSignal(
                signal=Signal.SELL,
                confidence=ai_signal.confidence,
                pair=pair,
                price=price,
                reason=reason,
            )
        else:
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=ai_signal.confidence,
                pair=pair,
                price=price,
                reason=f"AI HOLD — score={ai_signal.score:+.2f}, confidence={ai_signal.confidence:.0%}",
            )


    def generate_signals_via_decide(
        self,
        history_1h: pd.DataFrame,
        *,
        df_1d: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        decide_index: Optional[pd.Index] = None,
        window: int = 500,
        min_history: int = 50,
    ) -> pd.Series:
        """Parity backtest path: run the SAME decide() live uses over
        history, one replay context per candle (no lookahead slicing).

        Args:
            history_1h: full indicator-computed 1h frame (may include the
                training period — it is legitimate PAST context).
            df_1d/df_4h: indicator-computed higher-timeframe frames; sliced
                to fully-closed candles per decision. None blocks BUYs,
                same as a live macro-fetch failure.
            decide_index: candles to decide on (default: all of history_1h).
                Pass the out-of-sample window in walk-forward runs.
            window: 1h context rows per decision (matches live limit=500).

        Funding rate and order-book imbalance have no stored history —
        they are None in every replay context (parity exceptions; the
        corresponding filters pass through).

        Returns:
            Series of +1/-1/0 indexed by decide_index.
        """
        idx = decide_index if decide_index is not None else history_1h.index
        signals = pd.Series(0, index=idx, dtype=int)

        for ts in idx:
            ctx = build_replay_context(
                history_1h=history_1h,
                as_of=ts,
                df_1d=df_1d,
                df_4h=df_4h,
                news_headlines=tuple(self.news_headlines),
                window=window,
            )
            if len(ctx.df) < min_history:
                continue
            try:
                sig = self.decide(ctx, self.pair, verbose=False)
            except Exception as e:
                logger.debug("decide() failed at {}: {}", ts, e)
                continue
            if sig.signal == Signal.BUY:
                signals.loc[ts] = 1
            elif sig.signal == Signal.SELL:
                signals.loc[ts] = -1

        n_buy = int((signals == 1).sum())
        n_sell = int((signals == -1).sum())
        logger.info(
            "Replay decide() signals: {} buys, {} sells over {} candles "
            "(parity exceptions: funding, imbalance)",
            n_buy, n_sell, len(idx),
        )
        return signals

    def train_model(self, df: pd.DataFrame) -> None:
        """
        Train the XGBoost model on historical data.
        Call this before running backtests or live trading.

        Args:
            df: Full OHLCV + indicators DataFrame
        """
        from indicators.features import FeatureEngineer
        fe = FeatureEngineer()
        X, y = fe.build_features(df, target_periods=self.target_periods)

        result = self._predictor.train(X, y)
        result.print_report()

        self._model_loaded = True
        logger.info("Model training complete — strategy is ready")

    def get_params(self) -> dict:
        return {
            "pair": self.pair,
            "use_llm": self.use_llm,
            "target_periods": self.target_periods,
            "min_confidence": self.min_confidence,
            "model_loaded": self._model_loaded,
        }
