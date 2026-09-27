"""Stage 5 tests — WalkForwardValidator + Stage 0 circuit breaker integration.

Key contracts verified:
1. Synthetic LOSING streak (win rate < 35%) → circuit breaker trips.
2. Synthetic WINNING streak → circuit breaker stays ARMED (quiet).
3. Insufficient trades → evaluator skips (does not trip).
4. Leakage guard: out-of-order exit_times → evaluator skips with error reason.
5. Sharpe gate: positive win rate but very negative Sharpe → trips.
6. Already-tripped breaker: stays tripped after second evaluation.
7. Manual reset + re-evaluation on good performance → stays ARMED.

All tests use InMemoryCircuitBreakerStore so no disk I/O is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from risk.manager import (
    InMemoryCircuitBreakerStore,
    RiskConfig,
    RiskManager,
    CircuitBreakerState,
)
from risk.walk_forward import WalkForwardConfig, WalkForwardValidator


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_risk_manager() -> RiskManager:
    cfg = RiskConfig(
        max_position_notional_usdt=Decimal("10000"),
        max_daily_loss_pct=Decimal("0.05"),
        max_drawdown_pct=Decimal("0.10"),
    )
    store = InMemoryCircuitBreakerStore()
    return RiskManager(config=cfg, store=store)


def _make_validator(
    risk: RiskManager,
    min_win_rate: float = 0.35,
    min_sharpe: float = -0.5,
    lookback_trades: int = 20,
    min_trades: int = 10,
) -> WalkForwardValidator:
    cfg = WalkForwardConfig(
        lookback_trades=lookback_trades,
        min_trades=min_trades,
        min_win_rate=min_win_rate,
        min_sharpe=min_sharpe,
    )
    return WalkForwardValidator(risk_manager=risk, config=cfg)


def _make_trade(pnl: float, exit_offset_seconds: int = 0) -> dict:
    """Build a synthetic closed trade dict."""
    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    exit_time = now - timedelta(seconds=exit_offset_seconds)
    return {
        "status": "closed",
        "pnl": pnl,
        "exit_time": exit_time.isoformat(),
    }


def _make_losing_history(n: int = 20) -> list[dict]:
    """n trades, all losses, newest-first (DESC by exit_time)."""
    return [_make_trade(-100.0, exit_offset_seconds=i * 3600) for i in range(n)]


def _make_winning_history(n: int = 20) -> list[dict]:
    """n trades, all wins, newest-first."""
    return [_make_trade(100.0, exit_offset_seconds=i * 3600) for i in range(n)]


def _make_mixed_history(wins: int, losses: int) -> list[dict]:
    """Alternating wins and losses, newest-first."""
    trades = []
    offset = 0
    for i in range(wins + losses):
        pnl = 100.0 if i < wins else -100.0
        trades.append(_make_trade(pnl, exit_offset_seconds=offset))
        offset += 3600
    return trades


# ─── Gate 1: Win Rate ─────────────────────────────────────────────────────────

class TestWinRateGate:
    def test_losing_streak_trips_circuit_breaker(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk)
        history = _make_losing_history(20)

        result = validator.evaluate(history)

        assert result.evaluated is True
        assert result.breaker_tripped is True
        assert result.win_rate == 0.0
        assert risk.circuit_breaker_tripped() is True
        assert risk.breaker_state == CircuitBreakerState.TRIPPED

    def test_winning_streak_does_not_trip(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk)
        history = _make_winning_history(20)

        result = validator.evaluate(history)

        assert result.evaluated is True
        assert result.breaker_tripped is False
        assert result.win_rate == 1.0
        assert risk.circuit_breaker_tripped() is False

    def test_borderline_win_rate_below_threshold_trips(self) -> None:
        """33% win rate < 35% threshold → trips."""
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_win_rate=0.35)
        # 6 wins, 12 losses = 33% win rate
        history = _make_mixed_history(wins=6, losses=12)

        result = validator.evaluate(history)

        assert result.breaker_tripped is True
        assert result.win_rate == pytest.approx(6 / 18)

    def test_borderline_win_rate_above_threshold_does_not_trip(self) -> None:
        """40% win rate > 35% threshold → does not trip."""
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_win_rate=0.35)
        # 8 wins, 12 losses = 40% win rate
        history = _make_mixed_history(wins=8, losses=12)

        result = validator.evaluate(history)

        assert result.breaker_tripped is False
        assert result.win_rate == pytest.approx(8 / 20)


# ─── Gate 2: Sharpe Ratio ─────────────────────────────────────────────────────

class TestSharpeGate:
    def test_negative_sharpe_with_passing_win_rate_trips(self) -> None:
        """Win rate 60% but wildly variable PnL → very negative Sharpe → trips."""
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_win_rate=0.35, min_sharpe=0.0)

        # 12 small wins (+10) vs 8 large losses (-200): win rate=60% but avg loss dominates
        trades = []
        offset = 0
        for _ in range(12):
            trades.append(_make_trade(10.0, exit_offset_seconds=offset))
            offset += 3600
        for _ in range(8):
            trades.append(_make_trade(-200.0, exit_offset_seconds=offset))
            offset += 3600

        result = validator.evaluate(trades)

        assert result.evaluated is True
        assert result.win_rate >= 0.35        # passes win rate gate
        assert result.sharpe < 0.0           # fails Sharpe gate
        assert result.breaker_tripped is True

    def test_positive_sharpe_does_not_trip(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_win_rate=0.35, min_sharpe=-0.5)

        # Win history with variance so std dev > 0 and Sharpe is positive and defined
        history = []
        for i in range(20):
            pnl = 80.0 if i % 2 == 0 else 120.0
            history.append(_make_trade(pnl, exit_offset_seconds=i * 3600))

        result = validator.evaluate(history)

        assert result.sharpe > 0
        assert result.breaker_tripped is False


# ─── Insufficient Data Guard ──────────────────────────────────────────────────

class TestInsufficientData:
    def test_too_few_trades_does_not_evaluate(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_trades=10)
        history = _make_losing_history(5)  # Only 5, need 10

        result = validator.evaluate(history)

        assert result.evaluated is False
        assert result.breaker_tripped is False
        assert risk.circuit_breaker_tripped() is False
        assert "Insufficient" in result.reason

    def test_exactly_min_trades_does_evaluate(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_trades=10, lookback_trades=10)
        history = _make_losing_history(10)

        result = validator.evaluate(history)

        assert result.evaluated is True
        assert result.n_trades == 10


# ─── Leakage Guard ───────────────────────────────────────────────────────────

class TestLeakageGuard:
    def test_out_of_order_history_does_not_evaluate(self) -> None:
        """TradeLogger returns DESC order; reversed = ASC = wrong → skip."""
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_trades=5, lookback_trades=10)

        # Build trades in ascending order (oldest first) — this is WRONG for our contract
        history = list(reversed(_make_losing_history(10)))

        result = validator.evaluate(history)

        assert result.evaluated is False
        assert "ordered" in result.reason.lower() or "leakage" in result.reason.lower()
        assert risk.circuit_breaker_tripped() is False

    def test_future_trades_are_excluded(self) -> None:
        """Trades with exit_time AFTER evaluation_time must be excluded."""
        risk = _make_risk_manager()
        validator = _make_validator(risk, min_trades=10, lookback_trades=20)  # min_trades=10 is key here

        evaluation_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

        # 10 future trades (exit_time > evaluation_time) = should be filtered out
        future = [
            _make_trade(-500.0, exit_offset_seconds=-(i + 1) * 3600)  # negative offset = future
            for i in range(10)
        ]
        # 8 historical losses
        historical = _make_losing_history(8)

        result = validator.evaluate(future + historical, evaluation_time=evaluation_time)

        # Only historical trades qualify, and 8 < 10 min_trades → no evaluation
        assert result.evaluated is False


# ─── Circuit Breaker Persistence ─────────────────────────────────────────────

class TestCircuitBreakerIntegration:
    def test_already_tripped_breaker_stays_tripped(self) -> None:
        """After tripping, a second passing evaluation does NOT reset the breaker."""
        risk = _make_risk_manager()
        validator = _make_validator(risk)

        # First: trip it
        result1 = validator.evaluate(_make_losing_history(20))
        assert result1.breaker_tripped is True
        assert risk.circuit_breaker_tripped() is True

        # Second: good performance — but breaker was never manually reset
        result2 = validator.evaluate(_make_winning_history(20))
        assert result2.breaker_tripped is False        # Validator didn't re-trip
        assert risk.circuit_breaker_tripped() is True  # Breaker still tripped

    def test_manual_reset_allows_good_performance_to_pass(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk)

        # Trip then reset
        validator.evaluate(_make_losing_history(20))
        risk.manual_reset()
        assert risk.circuit_breaker_tripped() is False

        # Good performance after reset → stays armed
        result = validator.evaluate(_make_winning_history(20))
        assert result.breaker_tripped is False
        assert risk.circuit_breaker_tripped() is False

    def test_trip_message_contains_meaningful_reason(self) -> None:
        risk = _make_risk_manager()
        validator = _make_validator(risk)

        result = validator.evaluate(_make_losing_history(20))

        assert result.breaker_tripped is True
        assert "win rate" in result.reason.lower() or "sharpe" in result.reason.lower()
        assert len(result.reason) > 10


# ─── Sharpe Math ─────────────────────────────────────────────────────────────

class TestSharpeCalc:
    def test_all_identical_pnl_returns_zero(self) -> None:
        from risk.walk_forward import WalkForwardValidator as WFV
        pnl = [100.0] * 10
        assert WFV._sharpe(pnl) == 0.0

    def test_all_positive_returns_positive_sharpe(self) -> None:
        from risk.walk_forward import WalkForwardValidator as WFV
        pnl = [50.0, 60.0, 55.0, 70.0, 65.0]
        assert WFV._sharpe(pnl) > 0

    def test_all_negative_returns_negative_sharpe(self) -> None:
        from risk.walk_forward import WalkForwardValidator as WFV
        pnl = [-50.0, -60.0, -55.0, -70.0, -65.0]
        assert WFV._sharpe(pnl) < 0

    def test_single_value_returns_zero(self) -> None:
        from risk.walk_forward import WalkForwardValidator as WFV
        assert WFV._sharpe([100.0]) == 0.0
