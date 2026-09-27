"""
Tests for SentimentAnalyzer — verifies VADER scoring, aggregation math, and edge cases.
"""

import pytest
from ai.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from config.constants import Signal, MarketSentiment


@pytest.fixture
def analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


class TestSentimentScoring:
    """Test individual headline scoring."""

    def test_positive_headline_scores_positive(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze(["Bitcoin surges to new all-time high as adoption grows"])
        assert result.score > 0, "Bullish headline should score positive"

    def test_negative_headline_scores_negative(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze(["Crypto market crashes — Bitcoin loses 30% in massive selloff"])
        assert result.score < 0, "Bearish headline should score negative"

    def test_neutral_headline_near_zero(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze(["Bitcoin price remains stable at current levels"])
        assert -0.5 < result.score < 0.5, "Neutral headline should score near zero"

    def test_single_analyze_returns_float(self, analyzer: SentimentAnalyzer) -> None:
        score = analyzer.analyze_single("Bitcoin is rising")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


class TestSentimentAggregation:
    """Test multi-headline aggregation logic."""

    def test_mixed_headlines_produce_intermediate_score(self, analyzer: SentimentAnalyzer) -> None:
        headlines = [
            "Bitcoin hits record high — investors celebrate",  # Positive
            "SEC investigation causes concern in crypto market",  # Negative
        ]
        result = analyzer.analyze(headlines)
        assert -1.0 <= result.score <= 1.0, "Score must stay in [-1, 1]"
        assert result.headlines_analyzed == 2

    def test_all_bullish_produces_buy_signal(self, analyzer: SentimentAnalyzer) -> None:
        headlines = [
            "Bitcoin surges past $100k milestone",
            "Crypto market rally continues with massive gains",
            "Institutional adoption drives Bitcoin to new ATH",
        ]
        result = analyzer.analyze(headlines)
        assert result.signal == Signal.BUY, f"All bullish headlines should → BUY, got {result.score}"

    def test_all_bearish_produces_sell_signal(self, analyzer: SentimentAnalyzer) -> None:
        headlines = [
            "Crypto crash wipes out billions as market collapses",
            "Bitcoin plunges 40% amid regulatory crackdown",
            "Major exchange hack triggers panic selloff",
        ]
        result = analyzer.analyze(headlines)
        assert result.signal == Signal.SELL, f"All bearish headlines should → SELL, got {result.score}"

    def test_confidence_between_0_and_1(self, analyzer: SentimentAnalyzer) -> None:
        headlines = ["Bitcoin rises", "Ethereum falls", "Market neutral"]
        result = analyzer.analyze(headlines)
        assert 0.0 <= result.confidence <= 1.0

    def test_headlines_analyzed_count(self, analyzer: SentimentAnalyzer) -> None:
        headlines = ["headline one", "headline two", "headline three"]
        result = analyzer.analyze(headlines)
        assert result.headlines_analyzed == 3

    def test_breakdown_matches_headline_count(self, analyzer: SentimentAnalyzer) -> None:
        headlines = ["Bitcoin up", "Ethereum down"]
        result = analyzer.analyze(headlines)
        assert len(result.breakdown) == 2

    def test_breakdown_scores_in_range(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze(["Bitcoin is the future", "Crypto crash incoming"])
        for hs in result.breakdown:
            assert -1.0 <= hs.score <= 1.0


class TestSentimentEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_headlines_returns_neutral(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze([])
        assert result.score == 0.0
        assert result.signal == Signal.HOLD
        assert result.market_sentiment == MarketSentiment.NEUTRAL

    def test_returns_sentiment_result_type(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze(["Bitcoin"])
        assert isinstance(result, SentimentResult)

    def test_to_dict_has_required_keys(self, analyzer: SentimentAnalyzer) -> None:
        result = analyzer.analyze(["Bitcoin is bullish"])
        d = result.to_dict()
        assert "score" in d
        assert "signal" in d
        assert "market_sentiment" in d
        assert "headlines_analyzed" in d
        assert "confidence" in d

    def test_score_always_clamped(self, analyzer: SentimentAnalyzer) -> None:
        """Score must always be within [-1, 1] regardless of input."""
        very_positive = ["amazing incredible fantastic bullish moon ATH pump rally surge soar"] * 5
        result = analyzer.analyze(very_positive)
        assert -1.0 <= result.score <= 1.0

    def test_market_sentiment_extreme_greed(self, analyzer: SentimentAnalyzer) -> None:
        extremely_bullish = [
            "Bitcoin hits $200k — crypto market in extreme euphoria",
            "Massive institutional buying drives BTC to all-time highs",
            "Crypto adoption reaches record levels as prices surge",
            "Bitcoin ATH broken — bulls dominate the market",
        ]
        result = analyzer.analyze(extremely_bullish)
        assert result.market_sentiment in (MarketSentiment.GREED, MarketSentiment.EXTREME_GREED)
