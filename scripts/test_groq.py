#!/usr/bin/env python3
"""Quick live test of Groq LLM integration."""
import sys
sys.path.insert(0, "/home/osama/ai-workspace/projects/neurontrade")

print("Testing Groq LLM Agent...")
print("=" * 50)

from ai.llm_agent import LLMAgent

# Create agent — will auto-load key from .env
agent = LLMAgent(mock=False)

print(f"  Client connected: {agent._client is not None}")
print(f"  Mock mode: {agent.mock}")

if agent._client is None:
    print("  ❌ No client — check GROQ_API_KEY in .env")
    sys.exit(1)

# Fake indicator data (same structure the strategy sends)
fake_indicators = {
    "price": 64250.0,
    "price_change": 1.23,
    "rsi": 52.4,
    "macd": 0.000123,
    "macd_signal": 0.000098,
    "macd_hist": 0.000025,
    "macd_cross": 1,
    "bb_pct": 0.62,
    "bb_squeeze": False,
    "atr_pct": 1.8,
    "trend_200": 1,
    "trend_score": 2,
    "sma_cross": 1,
    "volume_ratio": 1.35,
    "obv_trend": 1,
    "mfi": 58.2,
    "composite_score": 0.41,
}

fake_news = [
    "Bitcoin ETF sees record inflows as institutional demand surges",
    "Fed signals potential rate cuts in Q3 2024",
]

print("\nCalling Groq API (Llama 3.3 70B)...")
result = agent.analyze(
    indicators=fake_indicators,
    news_headlines=fake_news,
    pair="BTC/USDT",
)

print()
print(f"  Signal:     {result.signal.value}")
print(f"  Score:      {result.score:+.3f}")
print(f"  Confidence: {result.confidence:.0%}")
print(f"  Mock:       {result.mock}")
print(f"  Model:      {result.model_used}")
print(f"\n  Reasoning:\n  {result.reasoning}")
print("\n  Key Factors:")
for f in result.key_factors:
    print(f"    • {f}")

if result.mock:
    print("\n  ❌ FAILED — still in mock mode")
elif result.error:
    print(f"\n  ❌ FAILED — error: {result.error}")
else:
    print("\n  ✅ GROQ LIVE API TEST PASSED")
