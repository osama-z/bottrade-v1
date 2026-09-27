"""
Sentiment Analyzer — scores crypto news headlines using VADER NLP.

Pipeline:
    1. Receive list of news headlines (strings)
    2. Score each headline with VADER compound score (-1 to +1)
    3. Aggregate into a single composite sentiment score
    4. Return structured SentimentResult

No external API key needed — VADER runs fully locally.
"""

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed — sentiment will return neutral scores")

from config.constants import Signal, MarketSentiment


# ─── Crypto-specific word boosters ───────────────────────────────────────────
# VADER is trained on general English — these words boost accuracy for crypto
CRYPTO_LEXICON: dict[str, float] = {
    # Bullish words
    "bullish": 2.5,
    "moon": 2.0,
    "pump": 1.5,
    "rally": 2.0,
    "breakout": 1.8,
    "surge": 1.8,
    "soar": 2.0,
    "ath": 2.5,       # All-time high
    "adoption": 1.5,
    "institutional": 1.2,
    "etf": 1.0,
    "halving": 1.5,
    "accumulation": 1.2,

    # Bearish words
    "bearish": -2.5,
    "dump": -2.0,
    "crash": -3.0,
    "collapse": -3.0,
    "plunge": -2.5,
    "rekt": -2.5,
    "liquidation": -2.0,
    "ban": -2.0,
    "hack": -3.0,
    "scam": -3.0,
    "fraud": -3.0,
    "regulation": -1.0,
    "sec": -0.8,
    "lawsuit": -2.0,
    "fud": -1.5,       # Fear Uncertainty Doubt
    "bearmarket": -2.5,
}


@dataclass
class HeadlineSentiment:
    """Sentiment result for a single headline."""
    headline: str
    score: float          # VADER compound score -1.0 to +1.0
    positive: float
    negative: float
    neutral: float


@dataclass
class SentimentResult:
    """Aggregated sentiment result across all analyzed headlines."""
    score: float                              # -1.0 (extreme fear) to +1.0 (extreme greed)
    signal: Signal                            # BUY / SELL / HOLD
    market_sentiment: MarketSentiment         # Enum label
    headlines_analyzed: int
    confidence: float                         # 0.0–1.0 — how many headlines agree
    breakdown: list[HeadlineSentiment] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "signal": self.signal.value,
            "market_sentiment": self.market_sentiment.value,
            "headlines_analyzed": self.headlines_analyzed,
            "confidence": self.confidence,
        }


class SentimentAnalyzer:
    """
    Scores a list of crypto news headlines and returns a composite sentiment signal.

    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(["Bitcoin surges to new ATH", "SEC sues Binance"])
        print(result.score, result.signal)
    """

    # Signal thresholds
    BUY_THRESHOLD = 0.15     # score > +0.15 = bullish
    SELL_THRESHOLD = -0.15   # score < -0.15 = bearish

    def __init__(self) -> None:
        if VADER_AVAILABLE:
            self._analyzer = SentimentIntensityAnalyzer()
            # Inject crypto-specific words into VADER's lexicon
            self._analyzer.lexicon.update(CRYPTO_LEXICON)
            logger.debug("SentimentAnalyzer initialized with {} crypto lexicon entries", len(CRYPTO_LEXICON))
        else:
            self._analyzer = None

    def analyze(self, headlines: list[str]) -> SentimentResult:
        """
        Analyze a list of headlines and return composite sentiment.

        Args:
            headlines: List of news headline strings

        Returns:
            SentimentResult with score, signal, and per-headline breakdown
        """
        if not headlines:
            logger.debug("No headlines provided — returning neutral sentiment")
            return self._neutral_result(headlines_analyzed=0)

        if not VADER_AVAILABLE or self._analyzer is None:
            logger.warning("VADER unavailable — returning neutral sentiment")
            return self._neutral_result(
                headlines_analyzed=len(headlines),
                error="vaderSentiment not installed"
            )

        breakdown: list[HeadlineSentiment] = []
        scores: list[float] = []

        for headline in headlines:
            if not headline or not headline.strip():
                continue

            vs = self._analyzer.polarity_scores(headline.lower())

            hs = HeadlineSentiment(
                headline=headline,
                score=round(vs["compound"], 4),
                positive=round(vs["pos"], 4),
                negative=round(vs["neg"], 4),
                neutral=round(vs["neu"], 4),
            )
            breakdown.append(hs)
            scores.append(vs["compound"])

        if not scores:
            return self._neutral_result(headlines_analyzed=0)

        # Weighted average — more extreme scores get higher weight
        abs_scores = [abs(s) for s in scores]
        total_weight = sum(abs_scores) or 1.0
        composite = sum(s * w for s, w in zip(scores, abs_scores)) / total_weight

        # Confidence: proportion of headlines that agree with the composite direction
        agreeing = sum(1 for s in scores if (s >= 0) == (composite >= 0))
        confidence = round(agreeing / len(scores), 3)

        composite = round(composite, 4)
        signal = self._score_to_signal(composite)
        sentiment_label = self._score_to_label(composite)

        logger.info(
            "Sentiment: score={:.3f} signal={} headlines={} confidence={:.0%}",
            composite, signal.value, len(scores), confidence
        )

        return SentimentResult(
            score=composite,
            signal=signal,
            market_sentiment=sentiment_label,
            headlines_analyzed=len(scores),
            confidence=confidence,
            breakdown=breakdown,
        )

    def analyze_single(self, text: str) -> float:
        """Score a single piece of text. Returns compound score -1 to +1."""
        if not VADER_AVAILABLE or self._analyzer is None:
            return 0.0
        return self._analyzer.polarity_scores(text.lower())["compound"]

    def _score_to_signal(self, score: float) -> Signal:
        """Convert composite score to BUY/SELL/HOLD signal."""
        if score > self.BUY_THRESHOLD:
            return Signal.BUY
        elif score < self.SELL_THRESHOLD:
            return Signal.SELL
        return Signal.HOLD

    def _score_to_label(self, score: float) -> MarketSentiment:
        """Convert score to human-readable sentiment label."""
        if score >= 0.5:
            return MarketSentiment.EXTREME_GREED
        elif score >= 0.15:
            return MarketSentiment.GREED
        elif score <= -0.5:
            return MarketSentiment.EXTREME_FEAR
        elif score <= -0.15:
            return MarketSentiment.FEAR
        return MarketSentiment.NEUTRAL

    def _neutral_result(
        self,
        headlines_analyzed: int = 0,
        error: Optional[str] = None
    ) -> SentimentResult:
        return SentimentResult(
            score=0.0,
            signal=Signal.HOLD,
            market_sentiment=MarketSentiment.NEUTRAL,
            headlines_analyzed=headlines_analyzed,
            confidence=0.0,
            error=error,
        )
