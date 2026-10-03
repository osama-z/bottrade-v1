"""
Global settings — loaded from .env file via pydantic-settings.
All configuration lives here. No hardcoded values anywhere else.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # ─── Exchange ──────────────────────────────────────────────────────────────
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=True, alias="BINANCE_TESTNET")
    # Market data (candles, tickers, book, funding) is PUBLIC and free —
    # take it from production even in testnet mode. Testnet history is only
    # a few weeks deep with synthetic prices, which trains coin-flip models
    # and makes paper results meaningless. Flip only to test the feed itself.
    market_data_testnet: bool = Field(default=False, alias="MARKET_DATA_TESTNET")

    # ─── AI / LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    ai_model: str = Field(default="llama-3.3-70b-versatile", alias="AI_MODEL")
    ai_analysis_interval: int = Field(default=300, alias="AI_ANALYSIS_INTERVAL")  # seconds

    # ─── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # ─── Trading ───────────────────────────────────────────────────────────────
    # Risk fields carry hard range constraints: a typo like RISK_PER_TRADE=5.0
    # (500% per trade) must fail at startup, not silently trade.
    trading_pairs_raw: str = Field(default="BTC/USDT,ETH/USDT", alias="TRADING_PAIRS")
    default_timeframe: str = Field(default="1h", alias="DEFAULT_TIMEFRAME")
    # Which registered strategy the live loop trades (strategies/registry.py).
    # The lab measured trend_following as the current best candidate; the
    # AI ensemble must beat it in walk-forward to earn this slot back.
    strategy_name: str = Field(default="ai_combined", alias="STRATEGY")
    # Portfolio heat cap: total open risk (Σ |entry−stop|×qty / equity)
    # across ALL positions. Correlated pairs (ADA/BTC/XRP move together)
    # make N full positions ≈ N× one risk, not diversification — the cap
    # bounds that. 0.06 = e.g. 3 concurrent positions at 2% risk each.
    portfolio_max_heat: float = Field(
        default=0.06, alias="PORTFOLIO_MAX_HEAT", gt=0, le=0.5
    )
    max_open_positions: int = Field(default=3, alias="MAX_OPEN_POSITIONS", ge=1, le=20)
    risk_per_trade: float = Field(default=0.02, alias="RISK_PER_TRADE", gt=0, le=0.1)
    max_drawdown: float = Field(default=0.10, alias="MAX_DRAWDOWN", gt=0, le=0.5)
    max_daily_loss: float = Field(default=0.03, alias="MAX_DAILY_LOSS", gt=0, le=0.2)
    initial_balance: float = Field(default=10_000.0, alias="INITIAL_BALANCE", gt=0)
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    # Shadow mode (Task 5.1): run live — fetch data, decide() — but execute
    # NOTHING (not even paper). Log the signal + expected fill for backtest-parity
    # validation. Overrides paper/live execution when true.
    shadow_mode: bool = Field(default=False, alias="SHADOW_MODE")

    # ─── Storage stack (Task 4.1) ──────────────────────────────────────────────
    # sqlite (default, single-node) or postgres (pooled, ACID, scalable). The
    # Postgres logger reads these DSNs; sqlite ignores them.
    database_backend: str = Field(default="sqlite", alias="DATABASE_BACKEND")   # sqlite | postgres
    postgres_dsn: str = Field(default="", alias="POSTGRES_DSN")   # postgresql://user:pass@host:5432/db
    # Market-data (OHLCV / order-book snapshots) time-series store. Falls back to
    # postgres_dsn when blank (TimescaleDB is a Postgres extension).
    market_data_backend: str = Field(default="none", alias="MARKET_DATA_BACKEND")  # none | timescale | clickhouse
    timescale_dsn: str = Field(default="", alias="TIMESCALE_DSN")
    db_pool_min: int = Field(default=1, alias="DB_POOL_MIN", ge=1, le=100)
    db_pool_max: int = Field(default=10, alias="DB_POOL_MAX", ge=1, le=200)

    # ─── BUY entry gates ───────────────────────────────────────────────────────
    # Measured over 8,727 hourly BTC candles (1 year): requiring RSI<30 AND a
    # MACD cross on the SAME candle fired 3 times — 0.03%. The two conditions
    # are anti-correlated (RSI<30 = falling hard; MACD cross = momentum just
    # turned), so ANDing them, then ANDing four more filters, produced ZERO
    # trades in a full walk-forward. These are now tunable rather than
    # hardcoded (audit V-65) so gate calibration is a config change.
    buy_timing_require_both: bool = Field(
        default=False, alias="BUY_TIMING_REQUIRE_BOTH"
    )  # False = RSI-oversold OR MACD-cross; True = the old (untradeable) AND
    buy_rsi_max: float = Field(default=30.0, alias="BUY_RSI_MAX", gt=0, le=100)
    buy_volume_multiple: float = Field(
        default=1.0, alias="BUY_VOLUME_MULTIPLE", ge=0
    )  # candle volume must exceed this × SMA20 (was 1.5)
    regime_block_bearish_only: bool = Field(
        default=True, alias="REGIME_BLOCK_BEARISH_ONLY"
    )  # True = block only Bearish; False = also block Choppy (86% of candles)

    # ─── Data / signal staleness ───────────────────────────────────────────────
    stale_data_seconds: int = Field(default=300, alias="STALE_DATA_SECONDS", gt=0)
    # Pipeline latency budget (fetch → indicators → decide → execute);
    # exceeding it logs a WARNING (claude.md Logging & Observability).
    latency_warn_ms: int = Field(default=10_000, alias="LATENCY_WARN_MS", gt=0)

    # ─── Order book (Stage 2) ──────────────────────────────────────────────────
    # Opt-in: feed the strategy's imbalance filter from the live WS depth
    # book instead of per-candle REST snapshots.
    use_ws_order_book: bool = Field(default=False, alias="USE_WS_ORDER_BOOK")

    # ─── ZMQ (Stage 4 decoupling) ──────────────────────────────────────────────
    zmq_signal_address: str = Field(
        default="tcp://127.0.0.1:5555", alias="ZMQ_SIGNAL_ADDRESS"
    )
    # The signal channel is UNAUTHENTICATED — publishing to it commands the
    # executor. Binding beyond loopback is refused unless this is set,
    # which should only happen behind CURVE auth or network isolation.
    zmq_allow_nonlocal: bool = Field(default=False, alias="ZMQ_ALLOW_NONLOCAL")
    stale_signal_seconds: int = Field(default=30, alias="STALE_SIGNAL_SECONDS", gt=0)
    heartbeat_interval_seconds: int = Field(
        default=10, alias="HEARTBEAT_INTERVAL_SECONDS", gt=0
    )

    # ─── News ──────────────────────────────────────────────────────────────────
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    # ─── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///storage/neurontrade.db", alias="DATABASE_URL"
    )

    # ─── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/neurontrade.log", alias="LOG_FILE")

    @field_validator("paper_trading")
    @classmethod
    def require_paper_demo(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Public demo requires PAPER_TRADING=true; real-money execution is disabled")
        return value

    @property
    def trading_pairs(self) -> List[str]:
        """Parse comma-separated trading pairs into a list."""
        return [p.strip() for p in self.trading_pairs_raw.split(",") if p.strip()]

    @property
    def database_path(self) -> str:
        """Absolute SQLite path derived from DATABASE_URL.

        Always anchored at the project root: a CWD-relative DB path means
        launching from a different directory silently creates a fresh,
        empty state database — a de facto full state reset.
        """
        from pathlib import Path

        raw = self.database_url.replace("sqlite:///", "", 1)
        p = Path(raw)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        return str(p)

    # Fields whose values must NEVER appear in repr/str output. Without
    # this, any traceback or log line that renders the settings object
    # (pydantic includes every field in repr) dumps live API keys into
    # logs, CI output, and error reports — found when a failing test
    # printed real credentials (OWASP A09).
    _SECRET_FIELDS = frozenset({
        "binance_api_key", "binance_api_secret", "groq_api_key",
        "telegram_bot_token", "telegram_chat_id", "news_api_key",
    })

    def __repr_args__(self):
        for key, value in super().__repr_args__():
            if key in self._SECRET_FIELDS and value:
                yield key, "***REDACTED***"
            else:
                yield key, value

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
        # Risk limits must not be mutable from arbitrary code — the Risk
        # Manager owns runtime risk state; settings are read-only inputs.
        "frozen": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance. Call this everywhere."""
    return Settings()


# Global settings instance — import this directly
settings = get_settings()
