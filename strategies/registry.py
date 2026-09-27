"""
Strategy Registry — register, list, and load strategies by name.
Makes it easy to swap strategies via config without changing code.
"""

from typing import Type
from strategies.base import BaseStrategy
from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_reversal import RSIReversalStrategy
from strategies.bollinger_bounce import BollingerBounceStrategy
from strategies.trend_following import (
    TrendFollowingStrategy,
    TrendFollowingFilteredStrategy,
)
from strategies.funding_carry import FundingCarryStrategy
from strategies.ai_combined import AICombinedStrategy
from core.exceptions import StrategyNotFoundError


# ─── Strategy registry ────────────────────────────────────────────────────────
# Add new strategies here — that's all you need to do to register them
_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "ma_crossover": MACrossoverStrategy,
    "rsi_reversal": RSIReversalStrategy,
    "bollinger_bounce": BollingerBounceStrategy,
    "trend_following": TrendFollowingStrategy,  # single-idea control strategy
    "trend_following_filtered": TrendFollowingFilteredStrategy,  # + 200-SMA regime gate
    "funding_carry": FundingCarryStrategy,  # Phase 1 — delta-neutral structural edge
    "ai_combined": AICombinedStrategy,   # Phase 2 — AI-powered strategy
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Instantiate a strategy by name.

    Args:
        name: Strategy name (e.g., "ma_crossover")
        **kwargs: Strategy constructor parameters

    Returns:
        Instantiated strategy object

    Raises:
        StrategyNotFoundError: If name not in registry
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise StrategyNotFoundError(
            f"Strategy '{name}' not found. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(_REGISTRY.keys())


def register_strategy(name: str, strategy_cls: Type[BaseStrategy]) -> None:
    """Register a new strategy class at runtime."""
    _REGISTRY[name] = strategy_cls
