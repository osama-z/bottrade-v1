"""
Custom exceptions for NeuronTrade.
All exceptions inherit from NeuronTradeError for easy catching.
"""


class NeuronTradeError(Exception):
    """Base exception for all NeuronTrade errors."""
    pass


# ─── Data Exceptions ──────────────────────────────────────────────────────────

class DataFetchError(NeuronTradeError):
    """Failed to fetch market data from exchange."""
    pass


class InsufficientDataError(NeuronTradeError):
    """Not enough data to calculate indicators or make a decision."""
    pass


class WebSocketError(NeuronTradeError):
    """WebSocket connection error."""
    pass


# ─── Exchange Exceptions ──────────────────────────────────────────────────────

class ExchangeError(NeuronTradeError):
    """Exchange API error."""
    pass


class OrderError(NeuronTradeError):
    """Order placement or management error."""
    pass


class InsufficientBalanceError(NeuronTradeError):
    """Not enough balance to place order."""
    pass


# ─── Strategy Exceptions ──────────────────────────────────────────────────────

class StrategyError(NeuronTradeError):
    """Strategy computation error."""
    pass


class StrategyNotFoundError(NeuronTradeError):
    """Strategy not found in registry."""
    pass


# ─── Risk Exceptions ──────────────────────────────────────────────────────────

class RiskLimitExceeded(NeuronTradeError):
    """Risk management limit exceeded — trade blocked."""
    pass


class MaxDrawdownReached(NeuronTradeError):
    """Maximum drawdown limit reached — bot stopped."""
    pass


# ─── AI Exceptions ────────────────────────────────────────────────────────────

class AIAnalysisError(NeuronTradeError):
    """AI analysis failed."""
    pass


class ModelNotTrainedError(NeuronTradeError):
    """ML model not trained yet."""
    pass


# ─── Config Exceptions ────────────────────────────────────────────────────────

class ConfigurationError(NeuronTradeError):
    """Invalid or missing configuration."""
    pass
