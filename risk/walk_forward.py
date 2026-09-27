"""Stage 5 — Walk-Forward Performance Validator.

Background job that runs every 24 hours (or on-demand) to evaluate live
trading performance. If the bot is losing money, it trips the EXISTING
Stage 0 circuit breaker — no second halt mechanism is created.

Design rules
------------
1. **No lookahead / no leakage**: every evaluation window uses only trades
   whose ``exit_time`` is strictly BEFORE the evaluation timestamp.
2. **Wired to Stage 0 circuit breaker**: uses ``RiskManager.trip_circuit_breaker()``
   directly — the same breaker that stops live order execution.
3. **Two gates before tripping**:
   - Win rate over the last ``lookback_trades`` closed trades.
   - Sharpe ratio over the same window.
4. **Stays quiet on good performance**: does NOT reset a tripped breaker
   automatically — manual reset (``RiskManager.manual_reset()``) is always
   required. Only human intervention can rearm the system.
5. **Data-leakage guard**: validates that the trade window is truly
   time-ordered (ascending exit_time) before computing metrics.

Known limitations
-----------------
- Sharpe on 20 trades has huge estimation variance. Use it as a coarse
  filter (threshold is deliberately lenient, e.g. -0.5) not a precise metric.
- Trades must have pnl denominated in the same currency (USDT notional).

Usage (production)::

    from risk.manager import RiskManager, SqliteCircuitBreakerStore
    from risk.walk_forward import WalkForwardValidator

    store   = SqliteCircuitBreakerStore("storage/neurontrade.db")
    risk    = RiskManager(config=cfg, store=store)
    validator = WalkForwardValidator(risk_manager=risk)

    # Call from APScheduler every 24h:
    validator.evaluate(trade_history=db.get_trade_history(limit=200))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Sequence

from risk import metrics


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for the walk-forward validator.

    Attributes:
        lookback_trades:    Number of most-recent CLOSED trades to evaluate.
        min_trades:         Minimum trades required before any evaluation.
        min_win_rate:       Below this → trip the breaker. [0, 1]
        min_sharpe:         Below this → trip the breaker. (lenient default: -0.5)
    """
    lookback_trades: int = 20
    min_trades: int = 10
    min_win_rate: float = 0.35   # 35% minimum — very lenient, but non-trivial
    min_sharpe: float = -0.5     # Negative Sharpe = losing money persistently


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single walk-forward evaluation pass."""
    evaluated: bool          # False if insufficient data
    n_trades: int
    win_rate: float          # [0, 1]
    sharpe: float            # Annualised over trades, not calendar time
    breaker_tripped: bool
    reason: str


class WalkForwardValidator:
    """Evaluates recent live performance and trips Stage 0 circuit breaker if failing.

    Args:
        risk_manager:   Stage 0 RiskManager instance (already configured with a store).
        config:         Walk-forward thresholds.
    """

    def __init__(
        self,
        risk_manager,   # RiskManager — avoid circular import, type-check at runtime
        config: WalkForwardConfig | None = None,
    ) -> None:
        from risk.manager import RiskManager  # Local import to avoid circularity
        if not isinstance(risk_manager, RiskManager):
            raise TypeError("risk_manager must be a RiskManager instance")
        self._risk = risk_manager
        self.config = config or WalkForwardConfig()

    def evaluate(
        self,
        trade_history: Sequence[dict],
        *,
        evaluation_time: datetime | None = None,
    ) -> ValidationResult:
        """Run one walk-forward evaluation pass.

        Args:
            trade_history:    List of closed trade dicts (from TradeLogger.get_trade_history()).
                              Must be ordered by exit_time DESCENDING (newest first) — same
                              as TradeLogger returns.
            evaluation_time:  UTC datetime of evaluation (default: now). Used for the
                              circuit-breaker trip timestamp.

        Returns:
            ValidationResult describing the outcome.
        """
        now = evaluation_time or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        cfg = self.config

        # ── Filter: only fully-closed trades with a valid exit_time ───────────
        closed = [
            t for t in trade_history
            if t.get("status") == "closed"
            and t.get("exit_time") is not None
            and t.get("pnl") is not None
        ]

        # ── Leakage guard: ensure no future trades sneak in ───────────────────
        closed = [t for t in closed if self._parse_utc(t["exit_time"]) <= now]

        # ── Take the N most recent (history is DESC, so take from the front) ──
        window = closed[:cfg.lookback_trades]

        if len(window) < cfg.min_trades:
            return ValidationResult(
                evaluated=False,
                n_trades=len(window),
                win_rate=0.0,
                sharpe=0.0,
                breaker_tripped=False,
                reason=f"Insufficient data: {len(window)} trades < min {cfg.min_trades}",
            )

        # ── Time-order integrity check (data-leakage guard) ───────────────────
        exit_times = [self._parse_utc(t["exit_time"]) for t in window]
        if not self._is_descending(exit_times):
            return ValidationResult(
                evaluated=False,
                n_trades=len(window),
                win_rate=0.0,
                sharpe=0.0,
                breaker_tripped=False,
                reason="Trade history is not properly ordered by exit_time (potential leakage)",
            )

        pnl_series = [float(t["pnl"]) for t in window]

        win_rate = metrics.win_rate(pnl_series)
        sharpe = self._sharpe(pnl_series)

        # ── Gate 1: win rate ──────────────────────────────────────────────────
        if win_rate < cfg.min_win_rate:
            reason = (
                f"Win rate {win_rate:.1%} < threshold {cfg.min_win_rate:.1%} "
                f"over last {len(window)} trades"
            )
            self._risk.trip_circuit_breaker(reason, now)
            return ValidationResult(
                evaluated=True,
                n_trades=len(window),
                win_rate=win_rate,
                sharpe=sharpe,
                breaker_tripped=True,
                reason=reason,
            )

        # ── Gate 2: Sharpe ratio ──────────────────────────────────────────────
        if sharpe < cfg.min_sharpe:
            reason = (
                f"Sharpe {sharpe:.3f} < threshold {cfg.min_sharpe:.3f} "
                f"over last {len(window)} trades"
            )
            self._risk.trip_circuit_breaker(reason, now)
            return ValidationResult(
                evaluated=True,
                n_trades=len(window),
                win_rate=win_rate,
                sharpe=sharpe,
                breaker_tripped=True,
                reason=reason,
            )

        return ValidationResult(
            evaluated=True,
            n_trades=len(window),
            win_rate=win_rate,
            sharpe=sharpe,
            breaker_tripped=False,
            reason="Performance within acceptable range",
        )

    # ── Metric helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(pnl_series: list[float]) -> float:
        """Per-trade Sharpe (NOT annualised) — see risk/metrics.py, the
        single home for metric definitions."""
        return metrics.sharpe_per_trade(pnl_series)

    @staticmethod
    def _is_descending(times: list[datetime]) -> bool:
        """Return True if times are monotonically non-increasing (newest-first order)."""
        return all(times[i] >= times[i + 1] for i in range(len(times) - 1))

    @staticmethod
    def _parse_utc(value: object) -> datetime:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
