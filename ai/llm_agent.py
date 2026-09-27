"""
Groq LLM Agent — uses the Groq API (Llama 3.3 70B) for market analysis.

Given current indicator values and recent news, the model reasons about:
  - Market trend and momentum
  - Key support/resistance levels
  - News sentiment impact
  - Overall trade recommendation

Returns a structured score + reasoning explanation.

Requires: GROQ_API_KEY in .env  (free at console.groq.com)
"""

import json
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed — LLM agent will use mock mode")

from config.constants import Signal
from config.settings import settings


@dataclass
class LLMAnalysis:
    """Result from Groq Llama market analysis."""
    score: float              # -1.0 (strong sell) to +1.0 (strong buy)
    signal: Signal            # BUY / SELL / HOLD
    confidence: float         # 0.0–1.0
    reasoning: str            # Model's explanation (human-readable)
    key_factors: list[str]    # Bullet-point factors influencing the decision
    model_used: str
    mock: bool = False        # True if no API key / mock mode
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors,
            "model_used": self.model_used,
            "mock": self.mock,
        }


# ─── Prompt template ─────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """You are an expert crypto trading analyst. Analyze the following market data and provide a trading recommendation.

## Current Market Data for {pair}

### Price Action
- Current Price: ${price:,.2f}
- Price Change (last candle): {price_change:+.3f}%

### Technical Indicators
- RSI (14): {rsi:.1f} {rsi_note}
- MACD: {macd:.6f} | Signal: {macd_signal:.6f} | Histogram: {macd_hist:.6f}
- MACD Cross: {macd_cross}
- Bollinger Band %B: {bb_pct:.3f} (0=lower band, 1=upper band)
- BB Squeeze: {bb_squeeze}
- ATR%: {atr_pct:.3f}%

### Trend
- Trend vs 200 SMA: {trend_200}
- Trend Score (-4 to +4): {trend_score}
- SMA Cross: {sma_cross}

### Volume
- Volume vs 20-period avg: {volume_ratio:.2f}x
- OBV Trend: {obv_trend}
- MFI: {mfi:.1f}

### Composite Signal
- Composite Score (-1 to +1): {composite_score:.3f}

### Recent News Headlines
{news_section}

## Your Task

Based on ALL the above data, provide a JSON response with:
1. `score`: A float between -1.0 (strong sell) and +1.0 (strong buy)
2. `signal`: One of "BUY", "SELL", or "HOLD"  
3. `confidence`: Float 0.0–1.0 (how confident you are)
4. `reasoning`: 2–3 sentence explanation of your recommendation
5. `key_factors`: List of 3–5 bullet points (the most important factors)

IMPORTANT: Respond with ONLY valid JSON. No extra text before or after.

Example format:
{{
  "score": 0.65,
  "signal": "BUY",
  "confidence": 0.72,
  "reasoning": "RSI is recovering from oversold territory while MACD shows a bullish crossover. Price is above the 200 SMA indicating an uptrend, and volume confirms buying interest.",
  "key_factors": [
    "RSI bouncing from oversold at {rsi:.0f}",
    "MACD bullish crossover confirmed",
    "Price above 200 SMA — uptrend intact",
    "Volume {volume_ratio:.1f}x above average confirms move"
  ]
}}"""


class LLMAgent:
    """
    Groq-powered market analyst agent (Llama 3.3 70B).

    When GROQ_API_KEY is set: Makes real API calls to Groq.
    When no key / mock=True: Returns a mock analysis based on composite_score.

    Usage:
        agent = LLMAgent()
        analysis = agent.analyze(indicator_data, news_headlines, pair="BTC/USDT")
        print(analysis.reasoning)
        print(analysis.score)
    """

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        self._client = None

        if not mock and GROQ_AVAILABLE and settings.groq_api_key:
            self._client = Groq(api_key=settings.groq_api_key)
            logger.info("LLM Agent initialized with Groq API (model={})", settings.ai_model)
        else:
            if not settings.groq_api_key:
                logger.warning(
                    "GROQ_API_KEY not set — LLM agent running in mock mode. "
                    "Add your key to .env to enable real AI analysis."
                )
            self.mock = True

    def analyze(
        self,
        indicators: dict,
        news_headlines: Optional[list[str]] = None,
        pair: str = "BTC/USDT",
    ) -> LLMAnalysis:
        """
        Analyze market conditions and return a trading recommendation.

        Args:
            indicators: Dict from TechnicalIndicators.get_latest_signals()
            news_headlines: Optional list of recent news headline strings
            pair: Trading pair (e.g., "BTC/USDT")

        Returns:
            LLMAnalysis with score, signal, reasoning, and key factors
        """
        if self.mock or self._client is None:
            return self._mock_analysis(indicators, pair)

        try:
            prompt = self._build_prompt(indicators, news_headlines or [], pair)
            response_text = self._call_groq(prompt)
            return self._parse_response(response_text, pair)
        except Exception as e:
            logger.error("LLM analysis failed: {} — falling back to mock", e)
            return self._mock_analysis(indicators, pair, error=str(e))

    # ─── Private methods ──────────────────────────────────────────────────────

    def _build_prompt(
        self,
        indicators: dict,
        headlines: list[str],
        pair: str,
    ) -> str:
        """Build the structured prompt for the LLM (Groq-hosted)."""
        # RSI interpretation note
        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            rsi_note = "⚠️ OVERSOLD"
        elif rsi > 70:
            rsi_note = "⚠️ OVERBOUGHT"
        else:
            rsi_note = "(normal)"

        # Trend labels
        trend_200_val = indicators.get("trend_200", 0)
        trend_200 = "🟢 ABOVE (bullish)" if trend_200_val == 1 else "🔴 BELOW (bearish)"

        sma_cross_val = indicators.get("sma_cross", 0)
        sma_cross = "🟢 Fast above Slow" if sma_cross_val == 1 else "🔴 Fast below Slow"

        macd_cross_val = indicators.get("macd_cross", 0)
        macd_cross = "🟢 Bullish" if macd_cross_val == 1 else "🔴 Bearish"

        obv_trend_val = indicators.get("obv_trend", 0)
        obv_trend = "🟢 Rising (buying pressure)" if obv_trend_val == 1 else "🔴 Falling (selling pressure)"

        bb_squeeze_val = indicators.get("bb_squeeze", False)
        bb_squeeze = "⚡ YES — volatility compression, breakout possible" if bb_squeeze_val else "No"

        # News section
        if headlines:
            news_lines = "\n".join(f"  - {h}" for h in headlines[:10])
            news_section = f"Recent headlines:\n{news_lines}"
        else:
            news_section = "No recent news available."

        return ANALYSIS_PROMPT.format(
            pair=pair,
            price=indicators.get("price", 0),
            price_change=indicators.get("price_change", 0),
            rsi=rsi,
            rsi_note=rsi_note,
            macd=indicators.get("macd", 0),
            macd_signal=indicators.get("macd_signal", 0),
            macd_hist=indicators.get("macd_hist", 0),
            macd_cross=macd_cross,
            bb_pct=indicators.get("bb_pct", 0.5),
            bb_squeeze=bb_squeeze,
            atr_pct=indicators.get("atr_pct", 0),
            trend_200=trend_200,
            trend_score=indicators.get("trend_score", 0),
            sma_cross=sma_cross,
            volume_ratio=indicators.get("volume_ratio", 1),
            obv_trend=obv_trend,
            mfi=indicators.get("mfi", 50),
            composite_score=indicators.get("composite_score", 0),
            news_section=news_section,
        )

    def _call_groq(self, prompt: str) -> str:
        """Make the API call to Groq and return the raw response text."""
        logger.debug("Calling Groq API ({})...", settings.ai_model)

        response = self._client.chat.completions.create(
            model=settings.ai_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert crypto trading analyst. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,   # Low temperature = more consistent/deterministic JSON
        )
        response_text = response.choices[0].message.content
        logger.debug("Groq response received ({} chars)", len(response_text))
        return response_text

    def _parse_response(self, response_text: str, pair: str) -> LLMAnalysis:
        """Parse the LLM JSON response into LLMAnalysis."""
        try:
            # Strip markdown code blocks if present
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)

            score = float(data.get("score", 0.0))
            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

            signal_str = data.get("signal", "HOLD").upper()
            signal = Signal(signal_str) if signal_str in ("BUY", "SELL", "HOLD") else Signal.HOLD

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(data.get("reasoning", ""))
            key_factors = list(data.get("key_factors", []))

            logger.info(
                "LLM analysis: {} (score={:.2f}, confidence={:.0%})",
                signal.value, score, confidence
            )

            return LLMAnalysis(
                score=round(score, 4),
                signal=signal,
                confidence=round(confidence, 4),
                reasoning=reasoning,
                key_factors=key_factors,
                model_used=settings.ai_model,
                mock=False,
            )

        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
            # TypeError/AttributeError cover valid JSON with wrong-typed fields
            # (e.g. {"score": null} → float(None), {"signal": 5} → .upper()).
            # Without them these escape to analyze()'s catch-all, which returns a
            # MOCK analysis that can be actionable — a malformed response must
            # never fabricate a tradeable signal. Fail safe to HOLD instead.
            logger.error("Failed to parse LLM response: {} | Response: {}", e, response_text[:200])
            return LLMAnalysis(
                score=0.0,
                signal=Signal.HOLD,
                confidence=0.0,
                reasoning="Failed to parse LLM response",
                key_factors=[],
                model_used=settings.ai_model,
                mock=False,
                error=str(e),
            )

    def _mock_analysis(
        self,
        indicators: dict,
        pair: str,
        error: Optional[str] = None,
    ) -> LLMAnalysis:
        """
        Generate a mock analysis based on composite_score from Phase 1 indicators.
        Used when no API key is set or in testing.
        """
        composite = indicators.get("composite_score", 0.0)
        rsi = indicators.get("rsi", 50)
        trend = indicators.get("trend_200", 0)

        # Mock score: weighted combination of available indicators
        mock_score = (composite * 0.5) + ((rsi - 50) / 100) + (trend * 0.1)
        mock_score = round(max(-1.0, min(1.0, mock_score)), 4)

        if mock_score > 0.15:
            signal = Signal.BUY
            reasoning = (
                f"[MOCK] Composite score is positive ({composite:.2f}), "
                f"RSI at {rsi:.0f} shows momentum. Trend is {'bullish' if trend == 1 else 'bearish'}."
            )
        elif mock_score < -0.15:
            signal = Signal.SELL
            reasoning = (
                f"[MOCK] Composite score is negative ({composite:.2f}), "
                f"RSI at {rsi:.0f}. Trend is {'bullish' if trend == 1 else 'bearish'}."
            )
        else:
            signal = Signal.HOLD
            reasoning = (
                f"[MOCK] Composite score near zero ({composite:.2f}), "
                f"RSI at {rsi:.0f}. No clear directional bias."
            )

        confidence = min(abs(mock_score) + 0.3, 0.75)

        return LLMAnalysis(
            score=mock_score,
            signal=signal,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            key_factors=[
                f"Composite score: {composite:.3f}",
                f"RSI: {rsi:.1f}",
                f"Trend vs 200 SMA: {'bullish' if trend == 1 else 'bearish'}",
            ],
            model_used="mock",
            mock=True,
            error=error,
        )
