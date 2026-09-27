"""
Tests for SignalCombiner — verifies weighted fusion math, threshold logic,
signal selection, and partial-signal handling.
"""

import pytest
from ai.signal_combiner import SignalCombiner, AISignal
from ai.llm_agent import LLMAnalysis
from ai.ml_predictor import PredictionResult
from ai.sentiment_analyzer import SentimentResult
from config.constants import Signal, MarketSentiment


# ─── Fixtures: Mock AI component results ─────────────────────────────────────

def make_llm(score: float, signal: Signal = None, confidence: float = 0.7) -> LLMAnalysis:
    if signal is None:
        signal = Signal.BUY if score > 0 else (Signal.SELL if score < 0 else Signal.HOLD)
    return LLMAnalysis(
        score=score,
        signal=signal,
        confidence=confidence,
        reasoning="Mock LLM analysis",
        key_factors=["factor1", "factor2"],
        model_used="mock",
        mock=True,
    )


def make_ml(score: float, signal: Signal = None, confidence: float = 0.65) -> PredictionResult:
    p_up = (score + 1) / 2   # Convert [-1,1] to [0,1]
    p_down = 1 - p_up
    if signal is None:
        signal = Signal.BUY if score > 0 else (Signal.SELL if score < 0 else Signal.HOLD)
    return PredictionResult(
        p_up=round(p_up, 4),
        p_down=round(p_down, 4),
        score=score,
        signal=signal,
        confidence=confidence,
    )


def make_sentiment(score: float, signal: Signal = None, confidence: float = 0.6) -> SentimentResult:
    if signal is None:
        signal = Signal.BUY if score > 0.15 else (Signal.SELL if score < -0.15 else Signal.HOLD)
    return SentimentResult(
        score=score,
        signal=signal,
        market_sentiment=MarketSentiment.NEUTRAL,
        headlines_analyzed=5,
        confidence=confidence,
    )


@pytest.fixture
def combiner() -> SignalCombiner:
    return SignalCombiner(min_confidence=0.50)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestSignalCombinerWeights:
    """Test that weights sum correctly and scores are combined properly."""

    def test_weights_sum_to_one(self) -> None:
        c = SignalCombiner(weight_llm=0.4, weight_ml=0.4, weight_sentiment=0.2)
        total = c.weight_llm + c.weight_ml + c.weight_sentiment
        assert abs(total - 1.0) < 1e-9

    def test_unequal_weights_normalized(self) -> None:
        c = SignalCombiner(weight_llm=2, weight_ml=2, weight_sentiment=1)
        total = c.weight_llm + c.weight_ml + c.weight_sentiment
        assert abs(total - 1.0) < 1e-9

    def test_equal_strong_buy_signals_produce_buy(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(0.8),
            ml_prediction=make_ml(0.6),
            sentiment_result=make_sentiment(0.5),
        )
        assert result.signal == Signal.BUY

    def test_equal_strong_sell_signals_produce_sell(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(-0.8),
            ml_prediction=make_ml(-0.6),
            sentiment_result=make_sentiment(-0.5),
        )
        assert result.signal == Signal.SELL

    def test_conflicting_signals_produce_hold(self, combiner: SignalCombiner) -> None:
        """Strong buy + strong sell should cancel out to HOLD."""
        result = combiner.combine(
            llm_analysis=make_llm(0.9),
            ml_prediction=make_ml(-0.9),
            sentiment_result=make_sentiment(0.0),
        )
        # The net score should be near 0 (cancel out)
        assert -0.4 < result.score < 0.4


class TestSignalCombinerThresholds:
    """Test BUY/SELL/HOLD threshold boundaries."""

    def test_score_just_above_buy_threshold_is_buy(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(0.3),
            ml_prediction=make_ml(0.3),
            sentiment_result=make_sentiment(0.3),
        )
        assert result.signal == Signal.BUY

    def test_score_just_below_sell_threshold_is_sell(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(-0.3),
            ml_prediction=make_ml(-0.3),
            sentiment_result=make_sentiment(-0.3),
        )
        assert result.signal == Signal.SELL

    def test_score_within_band_is_hold(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(0.05),
            ml_prediction=make_ml(0.05),
            sentiment_result=make_sentiment(0.05),
        )
        assert result.signal == Signal.HOLD

    def test_score_always_clamped_to_minus1_plus1(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(1.0),
            ml_prediction=make_ml(1.0),
            sentiment_result=make_sentiment(1.0),
        )
        assert -1.0 <= result.score <= 1.0


class TestSignalCombinerPartialInputs:
    """Test behaviour when only some components are available."""

    def test_only_llm_works(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(llm_analysis=make_llm(0.7))
        assert isinstance(result, AISignal)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_only_ml_works(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(ml_prediction=make_ml(0.7))
        assert isinstance(result, AISignal)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_only_sentiment_works(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(sentiment_result=make_sentiment(0.5))
        assert isinstance(result, AISignal)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_no_components_returns_hold(self, combiner: SignalCombiner) -> None:
        result = combiner.combine()
        assert result.signal == Signal.HOLD
        assert result.score == 0.0
        assert result.is_actionable is False


class TestSignalCombinerOutput:
    """Test output structure and actionability flag."""

    def test_returns_ai_signal_type(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(0.5),
            ml_prediction=make_ml(0.5),
        )
        assert isinstance(result, AISignal)

    def test_to_dict_has_required_keys(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(0.5),
            ml_prediction=make_ml(0.5),
            sentiment_result=make_sentiment(0.3),
        )
        d = result.to_dict()
        assert "signal" in d
        assert "score" in d
        assert "confidence" in d
        assert "is_actionable" in d
        assert "breakdown" in d

    def test_high_confidence_buy_is_actionable(self) -> None:
        combiner = SignalCombiner(min_confidence=0.50)
        result = combiner.combine(
            llm_analysis=make_llm(0.8, confidence=0.85),
            ml_prediction=make_ml(0.7, confidence=0.80),
            sentiment_result=make_sentiment(0.5, confidence=0.70),
        )
        assert result.is_actionable is True

    def test_low_confidence_is_not_actionable(self) -> None:
        combiner = SignalCombiner(min_confidence=0.90)  # Very high threshold
        result = combiner.combine(
            llm_analysis=make_llm(0.5, confidence=0.30),
            ml_prediction=make_ml(0.5, confidence=0.30),
        )
        # Even with BUY score, low confidence should not be actionable
        if result.signal != Signal.HOLD:
            assert result.is_actionable is False

    def test_hold_signal_is_never_actionable(self, combiner: SignalCombiner) -> None:
        result = combiner.combine(
            llm_analysis=make_llm(0.0),
            ml_prediction=make_ml(0.0),
        )
        if result.signal == Signal.HOLD:
            assert result.is_actionable is False

    def test_component_scores_stored_correctly(self, combiner: SignalCombiner) -> None:
        llm = make_llm(0.6)
        ml = make_ml(0.4)
        sent = make_sentiment(0.2)
        result = combiner.combine(
            llm_analysis=llm,
            ml_prediction=ml,
            sentiment_result=sent,
        )
        assert result.llm_score == 0.6
        assert result.ml_score == 0.4
        assert result.sentiment_score == 0.2


class TestMathOnlyExecutionDecoupling:
    """Roadmap Task 1.1 — Decouple Noise from Execution Path.

    In math_only mode the combiner must not weigh the LLM or VADER sentiment;
    live execution routes purely through the validated mathematical (ML) signal.
    """

    def test_llm_and_sentiment_are_not_weighed(self) -> None:
        c = SignalCombiner(min_confidence=0.50, math_only=True)
        # ML says BUY; LLM and sentiment both scream SELL. Math-only follows ML.
        r = c.combine(
            llm_analysis=make_llm(-0.9),
            ml_prediction=make_ml(0.8),
            sentiment_result=make_sentiment(-0.9),
        )
        assert r.signal == Signal.BUY
        assert r.score == pytest.approx(0.8)          # ML score, undiluted by noise
        assert r.llm_score == 0.0 and r.sentiment_score == 0.0

    def test_only_ml_routes_the_decision(self) -> None:
        c = SignalCombiner(min_confidence=0.50, math_only=True)
        r = c.combine(ml_prediction=make_ml(-0.7))
        assert r.signal == Signal.SELL
        assert r.score == pytest.approx(-0.7)

    def test_no_ml_signal_holds_despite_strong_llm_and_sentiment(self) -> None:
        # Decoupled noise cannot manufacture a trade — fail-closed to HOLD.
        c = SignalCombiner(min_confidence=0.50, math_only=True)
        r = c.combine(llm_analysis=make_llm(0.9), sentiment_result=make_sentiment(0.9))
        assert r.signal == Signal.HOLD
        assert r.is_actionable is False
        assert r.score == 0.0

    def test_math_only_flag_defaults_off_and_is_exposed(self) -> None:
        assert SignalCombiner(math_only=True).math_only is True
        assert SignalCombiner().math_only is False   # default keeps full fusion


class TestFullFusionShowcaseUnchanged:
    """The retained showcase path (math_only=False) must be untouched."""

    def test_default_still_weighs_all_three_components(self) -> None:
        c = SignalCombiner(min_confidence=0.50)      # math_only=False default
        r = c.combine(
            llm_analysis=make_llm(-0.9),
            ml_prediction=make_ml(0.8),
            sentiment_result=make_sentiment(-0.9),
        )
        # All component scores are recorded (weighed), unlike math-only mode.
        assert r.llm_score == -0.9
        assert r.ml_score == 0.8
        assert r.sentiment_score == -0.9
