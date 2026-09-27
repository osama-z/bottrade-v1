import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from risk.manager import (
    CircuitBreakerState,
    InMemoryCircuitBreakerStore,
    SqliteCircuitBreakerStore,
    RiskConfig,
    RiskManager,
    RiskSnapshot,
)


def config(paper_trading: bool = True) -> RiskConfig:
    return RiskConfig(
        max_position_notional_usdt=Decimal("1000"),
        max_daily_loss_pct=Decimal("0.03"),
        max_drawdown_pct=Decimal("0.10"),
        max_concurrent_positions=3,
        paper_trading=paper_trading,
    )


def snapshot(
    *,
    equity: str = "10000",
    day_start: str = "10000",
    peak: str = "10000",
    open_positions: int = 0,
) -> RiskSnapshot:
    return RiskSnapshot(
        equity_usdt=Decimal(equity),
        day_start_equity_usdt=Decimal(day_start),
        peak_equity_usdt=Decimal(peak),
        open_positions=open_positions,
        timestamp_utc=datetime(2026, 7, 5, 12, 0, tzinfo=UTC),
    )


def test_blocks_order_when_resulting_position_exceeds_limit() -> None:
    manager = RiskManager(config())

    decision = manager.approve_order(
        order_notional_usdt=Decimal("250"),
        resulting_position_notional_usdt=Decimal("1000.01"),
        snapshot=snapshot(),
    )

    assert not decision.accepted
    assert decision.paper_order
    assert "exceeds max position" in decision.reason
    assert manager.breaker_state == CircuitBreakerState.ARMED


def test_trips_daily_loss_on_mark_to_market_equity_drop() -> None:
    manager = RiskManager(config())

    decision = manager.evaluate_equity_risk(snapshot(equity="9699.99"))

    assert not decision.accepted
    assert "daily loss limit breached" in decision.reason
    assert manager.breaker_state == CircuitBreakerState.TRIPPED


def test_trips_max_drawdown_on_peak_to_current_equity_drop() -> None:
    manager = RiskManager(config())

    decision = manager.evaluate_equity_risk(
        snapshot(equity="9000", day_start="9000", peak="10000")
    )

    assert not decision.accepted
    assert "max drawdown breached" in decision.reason
    assert manager.breaker_state == CircuitBreakerState.TRIPPED


def test_tripped_breaker_blocks_orders_until_manual_reset() -> None:
    store = InMemoryCircuitBreakerStore()
    manager = RiskManager(config(), store=store)
    manager.trip_circuit_breaker("operator test", datetime(2026, 7, 5, 12, 0, tzinfo=UTC))

    blocked = manager.approve_order(
        order_notional_usdt=Decimal("100"),
        resulting_position_notional_usdt=Decimal("100"),
        snapshot=snapshot(),
    )
    assert not blocked.accepted
    assert blocked.reason == "circuit breaker is tripped"

    restarted_manager = RiskManager(config(), store=store)
    still_blocked = restarted_manager.approve_order(
        order_notional_usdt=Decimal("100"),
        resulting_position_notional_usdt=Decimal("100"),
        snapshot=snapshot(),
    )
    assert not still_blocked.accepted

    restarted_manager.manual_reset()
    accepted = restarted_manager.approve_order(
        order_notional_usdt=Decimal("100"),
        resulting_position_notional_usdt=Decimal("100"),
        snapshot=snapshot(),
    )
    assert accepted.accepted
    assert accepted.reason == "paper order accepted"


def test_paper_trading_accepts_only_when_risk_allows() -> None:
    manager = RiskManager(config(paper_trading=True))

    decision = manager.approve_order(
        order_notional_usdt=Decimal("100"),
        resulting_position_notional_usdt=Decimal("100"),
        snapshot=snapshot(),
    )

    assert decision.accepted
    assert decision.paper_order
    assert decision.reason == "paper order accepted"


def test_live_mode_marks_accepted_order_as_live() -> None:
    manager = RiskManager(config(paper_trading=False))

    decision = manager.approve_order(
        order_notional_usdt=Decimal("100"),
        resulting_position_notional_usdt=Decimal("100"),
        snapshot=snapshot(),
    )

    assert decision.accepted
    assert not decision.paper_order
    assert decision.reason == "live order accepted"


def test_blocks_when_max_concurrent_positions_reached() -> None:
    manager = RiskManager(config())

    decision = manager.approve_order(
        order_notional_usdt=Decimal("100"),
        resulting_position_notional_usdt=Decimal("100"),
        snapshot=snapshot(open_positions=3),
    )

    assert not decision.accepted
    assert decision.reason == "max concurrent positions reached"


def test_rejects_non_utc_snapshot_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        RiskSnapshot(
            equity_usdt=Decimal("10000"),
            day_start_equity_usdt=Decimal("10000"),
            peak_equity_usdt=Decimal("10000"),
            open_positions=0,
            timestamp_utc=datetime(2026, 7, 5, 12, 0),
        )


def test_sqlite_store_persists_across_restart() -> None:
    """A tripped circuit breaker must survive process restart (new RiskManager instance).

    This test simulates a restart by constructing a second RiskManager from the
    same SQLite file after tripping the breaker via the first instance.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # --- Process 1: trip the breaker ---
        store1 = SqliteCircuitBreakerStore(db_path)
        manager1 = RiskManager(config(), store=store1)
        assert manager1.breaker_state == CircuitBreakerState.ARMED

        manager1.trip_circuit_breaker(
            "daily loss limit breached: 0.031",
            datetime(2026, 7, 7, 10, 0, tzinfo=UTC),
        )
        assert manager1.breaker_state == CircuitBreakerState.TRIPPED

        # --- Process 2: new instance, same SQLite file (simulates restart) ---
        store2 = SqliteCircuitBreakerStore(db_path)
        manager2 = RiskManager(config(), store=store2)

        # Must still be tripped — disk persists through the "restart"
        assert manager2.breaker_state == CircuitBreakerState.TRIPPED
        blocked = manager2.approve_order(
            order_notional_usdt=Decimal("100"),
            resulting_position_notional_usdt=Decimal("100"),
            snapshot=snapshot(),
        )
        assert not blocked.accepted
        assert blocked.reason == "circuit breaker is tripped"

        # --- Manual reset clears it ---
        manager2.manual_reset()
        assert manager2.breaker_state == CircuitBreakerState.ARMED

        # A third instance (another "restart") sees the reset state
        store3 = SqliteCircuitBreakerStore(db_path)
        manager3 = RiskManager(config(), store=store3)
        assert manager3.breaker_state == CircuitBreakerState.ARMED
        accepted = manager3.approve_order(
            order_notional_usdt=Decimal("100"),
            resulting_position_notional_usdt=Decimal("100"),
            snapshot=snapshot(),
        )
        assert accepted.accepted

    finally:
        os.unlink(db_path)


# ─── Task 2.1: fractional Kelly + volatility-targeted sizing ───────────────────
from decimal import Decimal as D  # noqa: E402

from risk.manager import kelly_fraction, vol_target_leverage  # noqa: E402


class TestKellyFraction:
    def test_full_kelly_formula(self):
        # p=0.6, b=2 → f* = p - q/b = 0.6 - 0.4/2 = 0.4; quarter = 0.10.
        assert kelly_fraction(0.6, 2.0, multiplier=D("1")) == D("0.4")
        assert kelly_fraction(0.6, 2.0, multiplier=D("0.25")) == D("0.1")

    def test_no_edge_returns_zero_dont_bet(self):
        # p=0.4, b=1 → f* = 0.4 - 0.6 = -0.2 < 0 → 0 (never bet a losing game).
        assert kelly_fraction(0.4, 1.0) == D("0")
        # Break-even p=0.5, b=1 → f* = 0 → 0.
        assert kelly_fraction(0.5, 1.0) == D("0")

    def test_cap_limits_aggressive_edges(self):
        # Strong edge would size big; the cap holds it down.
        assert kelly_fraction(0.9, 3.0, multiplier=D("0.25"), cap=D("0.05")) == D("0.05")

    def test_invalid_inputs_are_safe(self):
        assert kelly_fraction(1.5, 2.0) == D("0")     # prob out of range
        assert kelly_fraction(0.6, 0.0) == D("0")     # non-positive odds
        assert kelly_fraction(0.6, -1.0) == D("0")

    def test_returns_decimal(self):
        assert isinstance(kelly_fraction(0.7, 2.0), Decimal)


class TestVolTargetLeverage:
    def test_scales_inversely_with_asset_vol(self):
        # target 0.5% daily; asset vol 4% → leverage 0.125 (size down a hot asset).
        assert vol_target_leverage(0.005, 0.04) == D("0.005") / D("0.04")
        # A calm asset (0.25% vol) is levered up toward the target.
        assert vol_target_leverage(0.005, 0.0025) == D("2")

    def test_cap_and_bad_inputs(self):
        assert vol_target_leverage(0.005, 0.0025, cap=D("1.0")) == D("1.0")  # capped
        assert vol_target_leverage(0.005, 0.0) == D("0")                     # zero vol
        assert vol_target_leverage(0.0, 0.04) == D("0")
        assert isinstance(vol_target_leverage(0.005, 0.02), Decimal)


class TestPositionSizingIntegration:
    def _mgr(self):
        return RiskManager(config())

    def test_default_path_is_unchanged_fixed_fractional(self):
        # No edge/vol args → same fixed-fractional plan as before Task 2.1.
        plan = self._mgr().calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=100.0,
            current_balance=10_000.0, atr=2.0,
        )
        assert plan is not None
        # risk_amount = balance * risk_per_trade_pct (config default 0.01) = 100.
        assert plan.risk_amount == D("10000") * D("0.01")

    def test_kelly_edge_sizes_the_risk_budget(self):
        # Quarter-Kelly at p=0.6,b=2 → 0.10 risk fraction (max_kelly raised so the
        # cap doesn't bind) → 10× the 0.01 fixed default.
        cfg = RiskConfig(
            max_position_notional_usdt=Decimal("100000"),   # don't let the notional cap bind
            max_daily_loss_pct=Decimal("0.03"), max_drawdown_pct=Decimal("0.10"),
            max_kelly_fraction=Decimal("0.20"),
        )
        plan = RiskManager(cfg).calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=100.0, current_balance=10_000.0,
            atr=2.0, win_prob=0.6, payoff_ratio=2.0,
        )
        assert plan is not None
        assert plan.risk_amount == D("10000") * D("0.1")

    def test_max_kelly_fraction_caps_the_budget(self):
        # Same edge, default cap 0.05 → risk fraction held to 0.05 (not 0.10).
        plan = self._mgr().calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=100.0, current_balance=10_000.0,
            atr=2.0, win_prob=0.6, payoff_ratio=2.0,
        )
        assert plan is not None
        assert plan.risk_amount == D("10000") * D("0.05")

    def test_no_edge_declines_the_trade(self):
        plan = self._mgr().calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=100.0, current_balance=10_000.0,
            atr=2.0, win_prob=0.45, payoff_ratio=1.0,   # negative edge
        )
        assert plan is None

    def test_vol_target_caps_notional(self):
        # asset_vol 10% vs target 0.5% → leverage 0.05 → notional cap = 500.
        plan = self._mgr().calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=100.0, current_balance=10_000.0,
            atr=2.0, target_vol=0.005, asset_vol=0.10,
        )
        assert plan is not None
        assert plan.position_value <= D("500")        # vol-target binds below the fixed size


# ─── Task 2.2: correlation-aware heat caps ─────────────────────────────────────
import pandas as pd  # noqa: E402

from risk.manager import average_pairwise_correlation  # noqa: E402
from risk.correlation import CorrelationTracker  # noqa: E402


class TestAveragePairwiseCorrelation:
    def test_mean_of_off_diagonal_pairs(self):
        corr = {
            "A": {"A": 1.0, "B": 0.8, "C": 0.2},
            "B": {"A": 0.8, "B": 1.0, "C": 0.4},
            "C": {"A": 0.2, "B": 0.4, "C": 1.0},
        }
        # mean(AB, AC, BC) = mean(0.8, 0.2, 0.4) = 0.4667
        assert average_pairwise_correlation(corr, ["A", "B", "C"]) == pytest.approx((0.8 + 0.2 + 0.4) / 3)

    def test_single_or_empty_symbol_is_zero(self):
        assert average_pairwise_correlation({"A": {"A": 1.0}}, ["A"]) == 0.0
        assert average_pairwise_correlation({}, []) == 0.0

    def test_missing_and_nan_pairs_are_skipped(self):
        corr = {"A": {"B": float("nan")}, "B": {}}
        assert average_pairwise_correlation(corr, ["A", "B", "C"]) == 0.0

    def test_values_are_clamped(self):
        corr = {"A": {"B": 5.0}, "B": {"A": 5.0}}   # nonsense > 1
        assert average_pairwise_correlation(corr, ["A", "B"]) == 1.0


class TestCorrelationTracker:
    def _series(self, values):
        idx = pd.date_range(end=datetime(2026, 8, 1, tzinfo=UTC), periods=len(values), freq="D")
        return pd.Series(values, index=idx)

    def test_perfectly_correlated_and_anticorrelated(self):
        import numpy as np
        r = np.random.default_rng(0).normal(0, 0.01, 30)
        t = CorrelationTracker(window_days=60, min_periods=10)
        t.update_returns("A", self._series(r))
        t.update_returns("B", self._series(r))        # identical → corr +1
        t.update_returns("C", self._series(-r))        # opposite → corr -1
        assert t.average_correlation(["A", "B"]) == pytest.approx(1.0, abs=1e-6)
        assert t.average_correlation(["A", "C"]) == pytest.approx(-1.0, abs=1e-6)

    def test_independent_series_near_zero(self):
        import numpy as np
        rng = np.random.default_rng(1)
        t = CorrelationTracker(window_days=365, min_periods=10)
        t.update_returns("A", self._series(rng.normal(0, 0.01, 300)))
        t.update_returns("B", self._series(rng.normal(0, 0.01, 300)))
        assert abs(t.average_correlation(["A", "B"])) < 0.2

    def test_rolling_window_uses_only_recent_data(self):
        # 60 daily points: A and B are ANTI-correlated in the old half but
        # IDENTICAL in the recent 30 days. A 30-day window must see only the
        # recent half → correlation ≈ +1, not ~0.
        import numpy as np
        dates = pd.date_range(end=datetime(2026, 8, 1, tzinfo=UTC), periods=60, freq="D")
        r = np.random.default_rng(2).normal(0, 0.01, 60)
        a = pd.Series(r, index=dates)
        b = pd.Series(np.concatenate([-r[:30], r[30:]]), index=dates)
        recent = CorrelationTracker(window_days=30, min_periods=10)
        allhist = CorrelationTracker(window_days=365, min_periods=10)
        for t in (recent, allhist):
            t.update_returns("A", a)
            t.update_returns("B", b)
        # 30-day window sees the identical recent half → strongly positive;
        # full history mixes in the anti-correlated old half → near zero.
        assert recent.average_correlation(["A", "B"]) > 0.7
        assert recent.average_correlation(["A", "B"]) > allhist.average_correlation(["A", "B"]) + 0.5


class _FakeDB:
    def __init__(self, trades):
        self._trades = trades

    def get_open_trades(self):
        return self._trades


class TestCorrelationAwareHeatCap:
    def _mgr(self):
        return RiskManager(config())

    # One open BTC long: heat = |100-97|*100 = 300 on 10k equity = 3%.
    # Nominal projected = 3% + 1% (new) = 4% < 6% cap.
    _BTC = {"entry_price": 100.0, "stop_loss": 97.0, "quantity": 100.0,
            "symbol": "BTC/USDT", "status": "open"}

    def test_correlated_second_trade_is_rejected(self):
        corr = {"BTC/USDT": {"ETH/USDT": 0.9}, "ETH/USDT": {"BTC/USDT": 0.9}}
        ok, reason = self._mgr().can_open_trade(
            current_balance=10_000.0, open_position_count=1, daily_pnl=0.0,
            db=_FakeDB([self._BTC]), symbol="ETH/USDT", correlation=corr,
        )
        # 4% × (1 + 0.9) = 7.6% > 6% → rejected on correlation-adjusted heat.
        assert ok is False and "heat" in reason.lower()

    def test_uncorrelated_second_trade_is_allowed(self):
        corr = {"BTC/USDT": {"ETH/USDT": 0.0}, "ETH/USDT": {"BTC/USDT": 0.0}}
        ok, _ = self._mgr().can_open_trade(
            current_balance=10_000.0, open_position_count=1, daily_pnl=0.0,
            db=_FakeDB([self._BTC]), symbol="ETH/USDT", correlation=corr,
        )
        assert ok is True                    # 4% nominal, unchanged → under cap

    def test_without_correlation_matches_nominal_behaviour(self):
        ok, _ = self._mgr().can_open_trade(
            current_balance=10_000.0, open_position_count=1, daily_pnl=0.0,
            db=_FakeDB([self._BTC]), symbol="ETH/USDT",
        )
        assert ok is True                    # byte-identical to pre-Task-2.2


# ─── Task 2.3: automated deleveraging drawdown schedule ────────────────────────
from datetime import timedelta  # noqa: E402

from risk.manager import deleverage_multiplier  # noqa: E402


class TestDeleverageMultiplier:
    def test_schedule_thresholds(self):
        assert deleverage_multiplier(0.00) == D("1")      # full size
        assert deleverage_multiplier(0.049) == D("1")     # just under −5%
        assert deleverage_multiplier(0.05) == D("0.5")    # −5% → half
        assert deleverage_multiplier(0.099) == D("0.5")
        assert deleverage_multiplier(0.10) == D("0.25")   # −10% → quarter
        assert deleverage_multiplier(0.149) == D("0.25")
        assert deleverage_multiplier(0.15) == D("0")      # −15% → halt
        assert deleverage_multiplier(0.50) == D("0")

    def test_returns_decimal(self):
        assert isinstance(deleverage_multiplier(0.07), Decimal)


class TestDeleverageSizing:
    def _mgr(self):
        # initial_balance = peak baseline; 10k so drawdown is measured off 10k.
        cfg = RiskConfig(max_position_notional_usdt=Decimal("100000"),
                         max_daily_loss_pct=Decimal("0.03"), max_drawdown_pct=Decimal("0.50"))
        return RiskManager(cfg, initial_balance=10_000.0)

    def _size(self, drawdown):
        plan = self._mgr().calculate_position(
            symbol="BTC/USDT", side="buy", entry_price=100.0,
            current_balance=10_000.0, atr=2.0, drawdown=drawdown,
        )
        return plan.position_value if plan else None

    def test_size_halves_at_5pct_and_quarters_at_10pct(self):
        # Ratios (not exact equality) — quantity floors to 8 dp independently.
        base = self._size(0.0)
        assert float(self._size(0.06) / base) == pytest.approx(0.5, abs=1e-4)
        assert float(self._size(0.11) / base) == pytest.approx(0.25, abs=1e-4)

    def test_halt_drawdown_declines_the_trade(self):
        assert self._size(0.16) is None


class TestDrawdownDeleverageAndCooldown:
    def _mgr(self, store=None):
        cfg = RiskConfig(max_position_notional_usdt=Decimal("100000"),
                         max_daily_loss_pct=Decimal("0.03"), max_drawdown_pct=Decimal("0.50"))
        return RiskManager(cfg, store=store or InMemoryCircuitBreakerStore(),
                           initial_balance=10_000.0)

    _T0 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    def test_soft_tiers_do_not_halt(self):
        m = self._mgr()
        d = m.drawdown_deleverage(9_300.0, self._T0)   # −7% DD
        assert d.multiplier == D("0.5") and d.halted is False
        assert not m.circuit_breaker_tripped()

    def test_halt_trips_breaker(self):
        m = self._mgr()
        d = m.drawdown_deleverage(8_400.0, self._T0)   # −16% DD
        assert d.halted is True and d.multiplier == D("0")
        assert m.circuit_breaker_tripped()

    def test_reset_refused_within_24h_cooldown(self):
        m = self._mgr()
        m.drawdown_deleverage(8_000.0, self._T0)       # −20% → halt
        # 23h later: still within cooldown → refuse.
        assert m.manual_reset(now=self._T0 + timedelta(hours=23)) is False
        assert m.circuit_breaker_tripped()             # still tripped

    def test_reset_allowed_after_24h_cooldown(self):
        m = self._mgr()
        m.drawdown_deleverage(8_000.0, self._T0)       # halt
        assert m.manual_reset(now=self._T0 + timedelta(hours=24)) is True
        assert not m.circuit_breaker_tripped()

    def test_reset_without_now_bypasses_cooldown_backward_compat(self):
        m = self._mgr()
        m.drawdown_deleverage(8_000.0, self._T0)
        assert m.manual_reset() is True                # internal/backtest path
        assert not m.circuit_breaker_tripped()
