"""
Signal Combiner — fuses AI signals into one final trade decision.

Two modes:
    Full fusion (math_only=False)  — showcase / offline analysis. Weighs all
        three components:
            Claude LLM  = 40%  (reasoning-based, high quality but slower)
            XGBoost ML  = 40%  (pattern-based, fast, data-driven)
            Sentiment   = 20%  (news-based, leading indicator)
    Math-only (math_only=True)     — LIVE EXECUTION PATH (Roadmap Task 1.1).
        The LLM (llm_agent) and VADER sentiment (sentiment_analyzer) are
        decoupled: they are neither waited on nor weighed. Live trades route
        purely through the validated mathematical (ML) signal. The fusion code
        is fully retained for showcase; it just no longer drives execution.

Final decision rules:
    score > +0.20 → BUY
    score < -0.20 → SELL
    else          → HOLD

Minimum confidence threshold: MIN_SIGNAL_CONFIDENCE (0.60, configurable)
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config.constants import Signal, MIN_SIGNAL_CONFIDENCE
from ai.llm_agent import LLMAnalysis
from ai.ml_predictor import PredictionResult
from ai.sentiment_analyzer import SentimentResult


@dataclass
class AISignal:
    """
    The final fused AI trading signal with full breakdown.

    This is what gets passed to the strategy / executor.
    """
    signal: Signal
    score: float                    # -1.0 to +1.0 (final weighted score)
    confidence: float               # 0.0 to 1.0
    is_actionable: bool             # True if confidence >= threshold

    # Component scores (for transparency / logging)
    llm_score: float                # Claude's contribution
    ml_score: float                 # XGBoost's contribution
    sentiment_score: float          # Sentiment's contribution

    # Full source objects
    llm_analysis: Optional[LLMAnalysis] = None
    ml_prediction: Optional[PredictionResult] = None
    sentiment_result: Optional[SentimentResult] = None

    # Breakdown for display
    breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal": self.signal.value,
            "score": self.score,
            "confidence": self.confidence,
            "is_actionable": self.is_actionable,
            "breakdown": {
                "llm_score": self.llm_score,
                "ml_score": self.ml_score,
                "sentiment_score": self.sentiment_score,
                "llm_signal": self.llm_analysis.signal.value if self.llm_analysis else "N/A",
                "ml_signal": self.ml_prediction.signal.value if self.ml_prediction else "N/A",
                "sentiment_signal": self.sentiment_result.signal.value if self.sentiment_result else "N/A",
                "llm_mock": self.llm_analysis.mock if self.llm_analysis else True,
            },
        }

    def print_summary(self) -> None:
        """Pretty-print the combined signal summary."""
        action_mark = "✅ ACTIONABLE" if self.is_actionable else "⏸️  HOLD (low confidence)"
        mock_note = " [MOCK LLM]" if (self.llm_analysis and self.llm_analysis.mock) else ""

        print(f"""
┌─────────────────────────────────────────────────────┐
│           AI Signal Combiner — Final Decision        │
├─────────────────────────────────────────────────────┤
│  Signal:      {self.signal.value:<10}  Score: {self.score:+.3f}          │
│  Confidence:  {self.confidence:.0%}                                  │
│  Status:      {action_mark}{mock_note}         
├─────────────────────────────────────────────────────┤
│  Component Breakdown                                 │
│  • Claude LLM   (40%): {self.llm_score:+.3f}                    │
│  • XGBoost ML   (40%): {self.ml_score:+.3f}                    │
│  • Sentiment    (20%): {self.sentiment_score:+.3f}                    │
└─────────────────────────────────────────────────────┘""")

        if self.llm_analysis and not self.llm_analysis.mock:
            print(f"\n🤖 Claude Reasoning:\n   {self.llm_analysis.reasoning}")
            if self.llm_analysis.key_factors:
                print("\n   Key Factors:")
                for factor in self.llm_analysis.key_factors:
                    print(f"   • {factor}")


class SignalCombiner:
    """
    Fuses Claude, XGBoost, and Sentiment signals into one final AI decision.

    Usage:
        combiner = SignalCombiner()
        signal = combiner.combine(
            llm_analysis=llm_result,
            ml_prediction=xgb_result,
            sentiment_result=sent_result,
        )
        print(signal.signal, signal.confidence)
    """

    # Component weights — must sum to 1.0
    WEIGHT_LLM = 0.40
    WEIGHT_ML = 0.40
    WEIGHT_SENTIMENT = 0.20

    # Thresholds
    BUY_THRESHOLD = 0.20
    SELL_THRESHOLD = -0.20
    MIN_CONFIDENCE = MIN_SIGNAL_CONFIDENCE  # From constants.py (0.60)

    def __init__(
        self,
        weight_llm: float = 0.40,
        weight_ml: float = 0.40,
        weight_sentiment: float = 0.20,
        min_confidence: float = MIN_SIGNAL_CONFIDENCE,
        buy_threshold: float = 0.20,
        sell_threshold: float = -0.20,
        math_only: bool = False,
    ) -> None:
        # Normalize weights in case they don't sum to exactly 1.0
        total = weight_llm + weight_ml + weight_sentiment
        self.weight_llm = weight_llm / total
        self.weight_ml = weight_ml / total
        self.weight_sentiment = weight_sentiment / total
        self.min_confidence = min_confidence
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Roadmap Task 1.1 — Decouple Noise from Execution Path.
        # In math-only mode the LLM (llm_agent) and VADER sentiment
        # (sentiment_analyzer) are NOT weighed for live trade execution; the
        # combiner routes purely through the validated mathematical (ML) signal.
        # The fusion code below is fully retained and usable for showcase /
        # offline analysis via math_only=False.
        self.math_only = math_only

        if math_only:
            logger.info(
                "SignalCombiner in MATH-ONLY mode — LLM & VADER sentiment "
                "decoupled from the execution path (Roadmap Task 1.1)"
            )
        else:
            logger.debug(
                "SignalCombiner default weights — LLM: {:.0%}, ML: {:.0%}, Sentiment: {:.0%}",
                self.weight_llm, self.weight_ml, self.weight_sentiment
            )

    def combine(
        self,
        llm_analysis: Optional[LLMAnalysis] = None,
        ml_prediction: Optional[PredictionResult] = None,
        sentiment_result: Optional[SentimentResult] = None,
        adx: Optional[float] = None,
        atr: Optional[float] = None,
        atr_sma: Optional[float] = None,
    ) -> AISignal:
        """
        Combine all three AI signals into one final decision.

        Any or all components can be None — the combiner adjusts weights
        dynamically to use only available signals.

        Args:
            llm_analysis: Result from LLMAgent.analyze()
            ml_prediction: Result from MLPredictor.predict()
            sentiment_result: Result from SentimentAnalyzer.analyze()
            adx: Average Directional Index (regime filter)
            atr: Average True Range (volatility filter)
            atr_sma: SMA of ATR (historical volatility benchmark)

        Returns:
            AISignal with final signal, score, confidence, and breakdown
        """
        # Roadmap Task 1.1 — Decouple Noise from Execution Path.
        # In math-only mode the LLM and VADER sentiment are dropped before any
        # weighting: the execution decision routes purely through the validated
        # mathematical (ML) signal. With no ML signal available this yields a
        # safe HOLD (fail-closed) — an honest "no validated edge" rather than a
        # decision manufactured from unvalidated language/sentiment noise.
        if self.math_only:
            llm_analysis = None
            sentiment_result = None

        # 1. Determine dynamic weights based on market regime
        w_ml, w_llm, w_sent = self.weight_ml, self.weight_llm, self.weight_sentiment
        regime = "Normal"

        if atr is not None and atr_sma is not None and atr_sma > 0 and atr > 1.5 * atr_sma:
            regime = "High Volatility/News"
            w_ml, w_llm, w_sent = 0.30, 0.30, 0.40
        elif adx is not None:
            if adx > 25:
                regime = "Trending"
                w_ml, w_llm, w_sent = 0.60, 0.25, 0.15
            elif adx < 20:
                regime = "Choppy/Ranging"
                w_ml, w_llm, w_sent = 0.25, 0.55, 0.20

        # Re-normalize to sum to exactly 1.0
        total_w = w_ml + w_llm + w_sent
        w_ml /= total_w
        w_llm /= total_w
        w_sent /= total_w

        components: list[tuple[float, float, float]] = []
        # (score, confidence, weight)

        # ── Collect available components ───────────────────────────────────
        llm_score = 0.0
        ml_score = 0.0
        sentiment_score = 0.0

        available_weight = 0.0

        if llm_analysis is not None:
            llm_score = llm_analysis.score
            llm_conf = llm_analysis.confidence
            components.append((llm_score, llm_conf, w_llm))
            available_weight += w_llm

        if ml_prediction is not None and ml_prediction.error is None:
            ml_score = ml_prediction.score  # p_up - p_down, already in [-1, 1]
            ml_conf = ml_prediction.confidence
            components.append((ml_score, ml_conf, w_ml))
            available_weight += w_ml

        if sentiment_result is not None and sentiment_result.error is None:
            sentiment_score = sentiment_result.score
            sent_conf = sentiment_result.confidence
            components.append((sentiment_score, sent_conf, w_sent))
            available_weight += w_sent

        if not components:
            logger.warning("No AI components available — returning HOLD")
            return self._hold_signal(llm_analysis, ml_prediction, sentiment_result,
                                     llm_score, ml_score, sentiment_score)

        # ── Normalize weights for available components ──────────────────────
        normalized = [
            (score, conf, weight / available_weight)
            for score, conf, weight in components
        ]

        # ── Compute weighted score and confidence ───────────────────────────
        final_score = sum(score * weight for score, conf, weight in normalized)
        final_confidence = sum(conf * weight for score, conf, weight in normalized)

        final_score = round(max(-1.0, min(1.0, final_score)), 4)
        final_confidence = round(max(0.0, min(1.0, final_confidence)), 4)

        # ── Determine signal ────────────────────────────────────────────────
        if final_score > self.buy_threshold:
            signal = Signal.BUY
        elif final_score < self.sell_threshold:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        is_actionable = (
            signal != Signal.HOLD
            and final_confidence >= self.min_confidence
        )

        logger.info(
            "AI Combined (Regime: {}): {} | score={:+.3f} | confidence={:.0%} | actionable={}",
            regime, signal.value, final_score, final_confidence, is_actionable
        )

        return AISignal(
            signal=signal,
            score=final_score,
            confidence=final_confidence,
            is_actionable=is_actionable,
            llm_score=round(llm_score, 4),
            ml_score=round(ml_score, 4),
            sentiment_score=round(sentiment_score, 4),
            llm_analysis=llm_analysis,
            ml_prediction=ml_prediction,
            sentiment_result=sentiment_result,
            breakdown={
                "weights": {
                    "llm": w_llm,
                    "ml": w_ml,
                    "sentiment": w_sent,
                },
                "scores": {
                    "llm": llm_score,
                    "ml": ml_score,
                    "sentiment": sentiment_score,
                },
                "available_weight": available_weight,
                "regime": regime,
            },
        )

    def _hold_signal(
        self,
        llm: Optional[LLMAnalysis],
        ml: Optional[PredictionResult],
        sent: Optional[SentimentResult],
        llm_score: float,
        ml_score: float,
        sentiment_score: float,
    ) -> AISignal:
        return AISignal(
            signal=Signal.HOLD,
            score=0.0,
            confidence=0.0,
            is_actionable=False,
            llm_score=llm_score,
            ml_score=ml_score,
            sentiment_score=sentiment_score,
            llm_analysis=llm,
            ml_prediction=ml,
            sentiment_result=sent,
        )
