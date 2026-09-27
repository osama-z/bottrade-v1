"""
AI Layer — Phase 2 Intelligence Components.

Components:
    LLMAgent         — Claude API market analyst
    MLPredictor      — XGBoost price direction predictor
    SentimentAnalyzer— VADER crypto news sentiment scorer
    SignalCombiner   — Fuses all AI signals into one decision
"""

from ai.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from ai.ml_predictor import MLPredictor, PredictionResult, TrainingResult
from ai.llm_agent import LLMAgent, LLMAnalysis
from ai.signal_combiner import SignalCombiner, AISignal

__all__ = [
    "SentimentAnalyzer",
    "SentimentResult",
    "MLPredictor",
    "PredictionResult",
    "TrainingResult",
    "LLMAgent",
    "LLMAnalysis",
    "SignalCombiner",
    "AISignal",
]
