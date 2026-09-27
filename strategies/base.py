"""
Abstract base strategy class.
All strategies must inherit from this and implement generate_signals().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from config.constants import Signal


@dataclass
class TradeSignal:
    """A trading signal with full context."""
    signal: Signal          # BUY, SELL, or HOLD
    confidence: float       # 0.0 → 1.0
    pair: str
    price: float
    reason: str             # Human-readable explanation
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self) -> None:
        # Enforce the documented [0, 1] confidence contract at the value-object
        # boundary. Some strategy formulas can otherwise emit out-of-range
        # values — e.g. bollinger_bounce/rsi_reversal produce a NEGATIVE
        # confidence when the band/RSI reading is on the unexpected side (or a
        # source column is missing). Clamping here fixes every strategy, present
        # and future, in one place instead of per-strategy `max(0, min(1, …))`.
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def is_actionable(self, min_confidence: float = 0.60) -> bool:
        """Returns True if signal meets the minimum confidence threshold."""
        return self.signal != Signal.HOLD and self.confidence >= min_confidence

    def __str__(self) -> str:
        return (
            f"[{self.signal.value}] {self.pair} @ {self.price:.4f} "
            f"(confidence={self.confidence:.0%}) — {self.reason}"
        )


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Every strategy must implement:
    - generate_signals(): Vectorized signals for backtesting
    - get_signal(): Latest single signal for live trading
    - get_params(): Parameter dict for optimization
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal series for entire DataFrame (for backtesting).

        Args:
            df: OHLCV + indicators DataFrame

        Returns:
            Series of: +1 (buy), -1 (sell), 0 (hold)
            aligned with df index
        """
        pass

    @abstractmethod
    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """
        Get the latest signal for live trading.

        Args:
            df: Recent OHLCV + indicators DataFrame
            pair: Trading pair symbol

        Returns:
            TradeSignal with signal, confidence, and context
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return strategy parameters (for optimization and logging)."""
        pass

    def describe(self) -> str:
        """Short description of the strategy."""
        return f"{self.__class__.__name__}({self.get_params()})"
