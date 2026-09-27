"""
Trading constants and enumerations.
"""

from enum import Enum


class Signal(str, Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class PositionStatus(str, Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"


class Timeframe(str, Enum):
    """Supported timeframes (OHLCV candle sizes)."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class MarketSentiment(str, Enum):
    """Market sentiment levels."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class BotMode(str, Enum):
    """Bot operating modes."""
    PAPER = "paper"       # Simulated trading (no real money)
    LIVE = "live"         # Real trading (real money ⚠️)
    BACKTEST = "backtest"  # Historical testing


# ─── Timeframe to seconds mapping ────────────────────────────────────────────
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# ─── Indicator defaults ───────────────────────────────────────────────────────
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_SMA_FAST = 20
DEFAULT_SMA_SLOW = 50
DEFAULT_EMA_FAST = 12
DEFAULT_EMA_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_ATR_PERIOD = 14

# ─── Risk defaults ────────────────────────────────────────────────────────────
DEFAULT_STOP_LOSS_PCT = 0.02      # 2%
DEFAULT_TAKE_PROFIT_PCT = 0.04    # 4% (2:1 RR ratio)
DEFAULT_RISK_PER_TRADE = 0.02     # 2% of portfolio
DEFAULT_MAX_DRAWDOWN = 0.10       # 10% — stop trading
DEFAULT_MAX_POSITIONS = 3

# ─── Minimum confidence for trade execution ────────────────────────────────────
MIN_SIGNAL_CONFIDENCE = 0.60      # 60% minimum AI confidence to trade
