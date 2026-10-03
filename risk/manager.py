"""Risk and safety controls for NeuronTrade.

Stage 0 rules live here and are intentionally independent of exchange code.
All account-risk inputs are UTC-timestamped and denominated in USDT notional or
account-equity percentages. The execution layer can keep using the legacy
position-sizing API while the persistent circuit breaker remains the source of
truth for halts.

Circuit-breaker persistence
---------------------------
* ``InMemoryCircuitBreakerStore`` — in-process only; use for unit tests.
* ``SqliteCircuitBreakerStore``   — disk-backed; survives process restarts;
  suitable for paper and live trading. Pass its path to ``RiskManager``
  via the ``store=`` argument.

Example (production wiring)::

    from risk.manager import RiskManager, SqliteCircuitBreakerStore
    store = SqliteCircuitBreakerStore("/path/to/trades.db")
    risk  = RiskManager(config=cfg, store=store)
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Protocol


# Quantization steps (crypto convention: 8 decimal places)
_QTY_STEP = Decimal("0.00000001")
_PCT_STEP = Decimal("0.00000001")


# ─── Volatility-targeted / fractional-Kelly sizing (Roadmap Task 2.1) ──────────
# Pure Decimal math (precision policy: no float drift in the sizing path).
def kelly_fraction(
    win_prob,
    payoff_ratio,
    *,
    multiplier: Decimal = Decimal("0.25"),
    cap: Decimal | None = None,
) -> Decimal:
    """Fractional Kelly bet size from a strategy's edge and odds.

    Full Kelly f* = (b·p − q) / b = p − q/b, where
        p = win probability, q = 1 − p, b = payoff ratio (avg win / avg loss).
    Returns ``multiplier · f*`` (Quarter-Kelly by default), floored at 0 — a
    non-positive edge means DON'T BET — and optionally capped for safety.
    """
    p = Decimal(str(win_prob))
    b = Decimal(str(payoff_ratio))
    if not (Decimal("0") <= p <= Decimal("1")) or b <= 0:
        return Decimal("0")
    q = Decimal("1") - p
    f_star = p - q / b                       # = (b·p − q) / b
    f = Decimal(str(multiplier)) * f_star
    if f <= 0:
        return Decimal("0")
    if cap is not None and f > Decimal(str(cap)):
        return Decimal(str(cap))
    return f


def _corr_lookup(corr, a: str, b: str):
    """Read corr[a][b] from a pandas DataFrame or a dict-of-dicts; None if absent."""
    try:
        loc = getattr(corr, "loc", None)
        if loc is not None and a in corr.index and b in corr.columns:
            return corr.loc[a, b]
    except Exception:
        pass
    try:
        return corr.get(a, {}).get(b)
    except AttributeError:
        return None


def average_pairwise_correlation(corr, symbols) -> float:
    """Mean of the OFF-DIAGONAL pairwise correlations among ``symbols``.

    ``corr`` is a correlation matrix (pandas DataFrame or dict-of-dicts). With
    fewer than two symbols (or no usable pairs) there is no diversification
    signal, so it returns 0.0 (heat multiplier 1 + 0 = 1 → nominal heat).
    Values are clamped to [-1, 1].
    """
    syms = list(dict.fromkeys(symbols))          # de-dupe, preserve order
    if len(syms) < 2:
        return 0.0
    vals = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = _corr_lookup(corr, syms[i], syms[j])
            if c is None or c != c:              # missing / NaN
                continue
            vals.append(max(-1.0, min(1.0, float(c))))
    return sum(vals) / len(vals) if vals else 0.0


def _avg_correlation(correlation, symbols) -> float:
    """Average pairwise correlation from either a CorrelationTracker
    (has .average_correlation) or a raw matrix (DataFrame / dict-of-dicts)."""
    fn = getattr(correlation, "average_correlation", None)
    if callable(fn):
        return float(fn(symbols))
    return average_pairwise_correlation(correlation, symbols)


def vol_target_leverage(
    target_vol,
    asset_vol,
    *,
    cap: Decimal | None = None,
) -> Decimal:
    """Leverage so a position's volatility hits the target: L = target_vol / asset_vol.

    A calmer asset (low ``asset_vol``) is sized UP toward the target; a volatile
    one is sized DOWN. Non-positive inputs → 0. Optionally capped (max leverage).
    """
    tv = Decimal(str(target_vol))
    av = Decimal(str(asset_vol))
    if tv <= 0 or av <= 0:
        return Decimal("0")
    lev = tv / av
    if cap is not None and lev > Decimal(str(cap)):
        return Decimal(str(cap))
    return lev


def deleverage_multiplier(
    drawdown,
    *,
    dd_50: Decimal = Decimal("0.05"),
    dd_25: Decimal = Decimal("0.10"),
    dd_halt: Decimal = Decimal("0.15"),
) -> Decimal:
    """Position-size multiplier from the drawdown deleveraging schedule (Task 2.3).

        drawdown < 5%   → 1.00  (full size)
        5% ≤ dd < 10%   → 0.50
        10% ≤ dd < 15%  → 0.25
        dd ≥ 15%        → 0.00  (halt — trip the breaker, flatten)

    ``drawdown`` is a positive fraction (peak-to-current loss). Thresholds are
    configurable so the schedule can be tuned without touching this logic.
    """
    dd = Decimal(str(drawdown))
    if dd >= dd_halt:
        return Decimal("0")
    if dd >= dd_25:
        return Decimal("0.25")
    if dd >= dd_50:
        return Decimal("0.5")
    return Decimal("1")


class CircuitBreakerState(str, Enum):
    ARMED = "armed"
    TRIPPED = "tripped"


@dataclass(frozen=True)
class RiskConfig:
    max_position_notional_usdt: Decimal
    max_daily_loss_pct: Decimal
    max_drawdown_pct: Decimal
    max_concurrent_positions: int = 3
    paper_trading: bool = True
    risk_per_trade_pct: Decimal = Decimal("0.01")
    # Total open risk cap across ALL positions (Σ |entry−stop|·qty ÷ equity).
    # Correlated pairs make N positions ≈ N× one risk, not diversification.
    max_portfolio_heat_pct: Decimal = Decimal("0.06")
    atr_stop_multiplier: Decimal = Decimal("1.5")
    reward_risk_ratio: Decimal = Decimal("2")
    consecutive_loss_limit: int = 3
    loss_cooldown_hours: int = 12
    # Volatility-targeted / fractional-Kelly sizing (Task 2.1).
    kelly_multiplier: Decimal = Decimal("0.25")     # Quarter-Kelly
    max_kelly_fraction: Decimal = Decimal("0.05")   # hard cap on the Kelly risk fraction
    target_daily_vol: Decimal = Decimal("0.005")    # 0.5% vol target per position
    # Drawdown deleveraging schedule (Task 2.3).
    deleverage_dd_50: Decimal = Decimal("0.05")     # −5% DD → size ×0.50
    deleverage_dd_25: Decimal = Decimal("0.10")     # −10% DD → size ×0.25
    deleverage_halt_dd: Decimal = Decimal("0.15")   # −15% DD → halt + flatten
    deleverage_cooldown_hours: int = 24             # mandatory cooldown before manual reset

    def __post_init__(self) -> None:
        if self.max_position_notional_usdt <= 0:
            raise ValueError("max_position_notional_usdt must be positive")
        if not Decimal("0") < self.max_daily_loss_pct < Decimal("1"):
            raise ValueError("max_daily_loss_pct must be between 0 and 1")
        if not Decimal("0") < self.max_drawdown_pct < Decimal("1"):
            raise ValueError("max_drawdown_pct must be between 0 and 1")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if not Decimal("0") < self.risk_per_trade_pct < Decimal("1"):
            raise ValueError("risk_per_trade_pct must be between 0 and 1")
        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be positive")
        if self.reward_risk_ratio <= 0:
            raise ValueError("reward_risk_ratio must be positive")
        if self.consecutive_loss_limit <= 0:
            raise ValueError("consecutive_loss_limit must be positive")
        if self.loss_cooldown_hours <= 0:
            raise ValueError("loss_cooldown_hours must be positive")
        if not Decimal("0") < self.kelly_multiplier <= Decimal("1"):
            raise ValueError("kelly_multiplier must be in (0, 1]")
        if not Decimal("0") < self.max_kelly_fraction < Decimal("1"):
            raise ValueError("max_kelly_fraction must be in (0, 1)")
        if not Decimal("0") < self.target_daily_vol < Decimal("1"):
            raise ValueError("target_daily_vol must be in (0, 1)")
        if not (Decimal("0") < self.deleverage_dd_50 < self.deleverage_dd_25
                < self.deleverage_halt_dd < Decimal("1")):
            raise ValueError("deleverage thresholds must satisfy 0 < dd_50 < dd_25 < halt < 1")
        if self.deleverage_cooldown_hours <= 0:
            raise ValueError("deleverage_cooldown_hours must be positive")

    @classmethod
    def from_settings(cls, initial_balance: float) -> "RiskConfig":
        """The ONE way to build a settings-derived risk config.

        Both PaperTrader (live/paper) and BacktestEngine must use this —
        RiskManager's no-config fallback hardcodes its own defaults (1%
        risk vs the settings default 2%), so an engine built without this
        sized every backtest at HALF the live risk and silently ignored
        the operator's .env tuning (found by the tier-14 cross-system
        parity test).
        """
        from config.settings import settings

        return cls(
            max_position_notional_usdt=Decimal(str(initial_balance)),
            max_daily_loss_pct=Decimal(str(settings.max_daily_loss)),
            max_drawdown_pct=Decimal(str(settings.max_drawdown)),
            max_concurrent_positions=settings.max_open_positions,
            paper_trading=settings.paper_trading,
            risk_per_trade_pct=Decimal(str(settings.risk_per_trade)),
            max_portfolio_heat_pct=Decimal(str(settings.portfolio_max_heat)),
        )


@dataclass(frozen=True)
class RiskSnapshot:
    equity_usdt: Decimal
    day_start_equity_usdt: Decimal
    peak_equity_usdt: Decimal
    open_positions: int
    timestamp_utc: datetime

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware UTC")
        if self.timestamp_utc.utcoffset() != UTC.utcoffset(self.timestamp_utc):
            raise ValueError("timestamp_utc must be UTC")
        if self.equity_usdt <= 0:
            raise ValueError("equity_usdt must be positive")
        if self.day_start_equity_usdt <= 0:
            raise ValueError("day_start_equity_usdt must be positive")
        if self.peak_equity_usdt <= 0:
            raise ValueError("peak_equity_usdt must be positive")
        if self.open_positions < 0:
            raise ValueError("open_positions cannot be negative")


@dataclass(frozen=True)
class OrderDecision:
    accepted: bool
    paper_order: bool
    reason: str


@dataclass(frozen=True)
class DeleverageDecision:
    """Drawdown deleveraging outcome (Task 2.3)."""
    multiplier: Decimal      # position-size multiplier ∈ {1.0, 0.5, 0.25, 0.0}
    drawdown: float          # current peak-to-equity drawdown fraction
    halted: bool             # True when ≥ halt threshold → breaker tripped
    reason: str


@dataclass(frozen=True)
class PositionPlan:
    """Sized order plan. All monetary fields are Decimal (precision policy):

    quantity is rounded DOWN to 8 dp so the plan never risks more than
    budgeted; consumers convert to float only at non-order boundaries
    (backtest math, SQLite REAL persistence).
    """

    symbol: str
    side: str
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    quantity: Decimal
    position_value: Decimal
    risk_amount: Decimal
    reward_risk_ratio: Decimal


@dataclass(frozen=True)
class _BreakerRecord:
    state: CircuitBreakerState
    reason: str | None = None
    tripped_at_utc: datetime | None = None
    peak_equity: float | None = None


class CircuitBreakerStore(Protocol):
    def load(self) -> _BreakerRecord:
        """Return the latest persisted circuit breaker state."""

    def save(self, record: _BreakerRecord) -> None:
        """Persist the latest circuit breaker state."""

    def update_peak_equity(self, peak_equity: float) -> None:
        """Raise (never lower) peak equity without touching breaker state.

        Must be atomic with respect to concurrent ``save`` calls so a
        peak update can never resurrect a stale ARMED state over a trip.
        """


class InMemoryCircuitBreakerStore:
    """In-process store — resets on every restart. Use only in unit tests."""

    def __init__(self) -> None:
        self._record = _BreakerRecord(state=CircuitBreakerState.ARMED)
        self._lock = threading.Lock()

    def load(self) -> _BreakerRecord:
        return self._record

    def save(self, record: _BreakerRecord) -> None:
        with self._lock:
            self._record = record

    def update_peak_equity(self, peak_equity: float) -> None:
        with self._lock:
            current = self._record.peak_equity
            if current is None or peak_equity > current:
                self._record = replace(self._record, peak_equity=peak_equity)


class SqliteCircuitBreakerStore:
    """Disk-backed circuit breaker store.

    Persists the circuit breaker state to a SQLite database so that a tripped
    breaker survives process restarts. Uses the same database file as
    ``TradeLogger`` by default — no second file is created.

    Thread-safety: each call opens, uses, and closes its own connection inside a
    single ``with`` block; SQLite's WAL mode ensures concurrent readers are safe.
    """

    _TABLE = "circuit_breaker_state"
    _DDL = f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            id          INTEGER PRIMARY KEY,
            state       TEXT    NOT NULL,
            reason      TEXT,
            tripped_at  TEXT
        )
    """

    _DDL_HISTORY = """
        CREATE TABLE IF NOT EXISTS circuit_breaker_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            state       TEXT NOT NULL,
            reason      TEXT,
            tripped_at  TEXT,
            recorded_at TEXT NOT NULL
        )
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        # busy_timeout: wait for a contended write lock instead of raising
        # "database is locked" — a failed save here can lose a breaker trip.
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_table(self) -> None:
        with self._connect() as conn:
            # WAL is a persistent property of the DB file: readers no longer
            # block the writer (this store shares a file with TradeLogger).
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(self._DDL)
            conn.execute(self._DDL_HISTORY)

            # Migration
            cursor = conn.execute(f"PRAGMA table_info({self._TABLE})")
            columns = [row[1] for row in cursor.fetchall()]
            if "peak_equity" not in columns:
                conn.execute(f"ALTER TABLE {self._TABLE} ADD COLUMN peak_equity REAL")

            conn.commit()

    def load(self) -> _BreakerRecord:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT state, reason, tripped_at, peak_equity FROM {self._TABLE} WHERE id = 1"
            ).fetchone()
        if row is None:
            return _BreakerRecord(state=CircuitBreakerState.ARMED)
        state = CircuitBreakerState(row[0])
        reason = row[1]
        tripped_at_raw = row[2]
        peak_equity = row[3] if len(row) > 3 else None
        tripped_at: datetime | None = None
        if tripped_at_raw:
            tripped_at = datetime.fromisoformat(tripped_at_raw)
            if tripped_at.tzinfo is None:
                tripped_at = tripped_at.replace(tzinfo=UTC)
        return _BreakerRecord(state=state, reason=reason, tripped_at_utc=tripped_at, peak_equity=peak_equity)

    def save(self, record: _BreakerRecord) -> None:
        tripped_at_iso = (
            record.tripped_at_utc.isoformat() if record.tripped_at_utc else None
        )
        with self._connect() as conn:
            conn.execute(
                f"""INSERT OR REPLACE INTO {self._TABLE}
                    (id, state, reason, tripped_at, peak_equity) VALUES (1, ?, ?, ?, ?)""",
                (record.state.value, record.reason, tripped_at_iso, record.peak_equity),
            )
            # Append-only audit trail: the single-row latest-state table
            # loses history on every save; the mandated "every circuit
            # breaker state transition persisted" lives here.
            conn.execute(
                """INSERT INTO circuit_breaker_history
                   (state, reason, tripped_at, recorded_at) VALUES (?, ?, ?, ?)""",
                (
                    record.state.value,
                    record.reason,
                    tripped_at_iso,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()

    def update_peak_equity(self, peak_equity: float) -> None:
        # Single UPDATE that only touches peak_equity: cannot overwrite a
        # concurrent breaker trip with a stale ARMED state, and MAX() keeps
        # the update monotonic even under concurrent writers.
        with self._connect() as conn:
            conn.execute(
                f"""INSERT OR IGNORE INTO {self._TABLE} (id, state, peak_equity)
                    VALUES (1, ?, ?)""",
                (CircuitBreakerState.ARMED.value, peak_equity),
            )
            conn.execute(
                f"""UPDATE {self._TABLE}
                    SET peak_equity = MAX(COALESCE(peak_equity, 0), ?)
                    WHERE id = 1""",
                (peak_equity,),
            )
            conn.commit()

class RiskManager:
    """Evaluate pre-trade, position-sizing, and account-risk controls.

    Assumptions made explicit:
    - order/position values are USDT notional, not base quantity or contracts;
    - equity snapshots include realized plus unrealized mark-to-market PnL;
    - UTC is the only clock accepted for risk state;
    - a tripped breaker never auto-resumes after process restart.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        store: CircuitBreakerStore | None = None,
        initial_balance: float | None = None,
    ) -> None:
        if config is None:
            base_equity = Decimal(str(initial_balance or 10_000.0))
            config = RiskConfig(
                max_position_notional_usdt=base_equity,
                max_daily_loss_pct=Decimal("0.03"),
                max_drawdown_pct=Decimal("0.10"),
                max_concurrent_positions=3,
                paper_trading=True,
                risk_per_trade_pct=Decimal("0.01"),
            )
        self.config = config
        self.store = store or InMemoryCircuitBreakerStore()
        self.initial_balance = float(initial_balance or self.config.max_position_notional_usdt)
        # Serializes breaker-state transitions within this process so a
        # trip from one thread can't race a peak update from another.
        self._breaker_lock = threading.Lock()

    @property
    def breaker_state(self) -> CircuitBreakerState:
        return self.store.load().state

    def circuit_breaker_tripped(self) -> bool:
        return self.breaker_state == CircuitBreakerState.TRIPPED

    def trip_circuit_breaker(self, reason: str, timestamp_utc: datetime) -> None:
        self._require_utc(timestamp_utc)
        with self._breaker_lock:
            record = self.store.load()
            self.store.save(
                _BreakerRecord(
                    state=CircuitBreakerState.TRIPPED,
                    reason=reason,
                    tripped_at_utc=timestamp_utc,
                    peak_equity=record.peak_equity,
                )
            )

    def manual_reset(self, now: datetime | None = None) -> bool:
        """Re-arm a tripped breaker. Returns True if reset, False if refused.

        Mandatory cooldown (Task 2.3): when ``now`` is supplied, a reset within
        ``deleverage_cooldown_hours`` of the trip is REFUSED — the operator
        cannot clear a drawdown halt before the cooldown elapses. Called with no
        ``now`` (internal / backtest simulated-operator), it resets immediately,
        preserving prior behaviour.
        """
        with self._breaker_lock:
            record = self.store.load()
            if (
                now is not None
                and record.state == CircuitBreakerState.TRIPPED
                and record.tripped_at_utc is not None
            ):
                self._require_utc(now)
                elapsed = now - record.tripped_at_utc
                if elapsed < timedelta(hours=self.config.deleverage_cooldown_hours):
                    return False
            self.store.save(
                _BreakerRecord(state=CircuitBreakerState.ARMED, peak_equity=None)
            )
            return True

    def drawdown_deleverage(
        self, current_equity: float, timestamp_utc: datetime
    ) -> DeleverageDecision:
        """Apply the drawdown deleveraging schedule (Task 2.3).

        Computes drawdown vs the stored peak and returns the position-size
        multiplier (1.0 / 0.5 / 0.25). At/above the halt threshold it TRIPS the
        circuit breaker (multiplier 0.0, halted=True) — the caller then flattens
        and cannot manually reset until the cooldown elapses.
        """
        self._require_utc(timestamp_utc)
        record = self.store.load()
        peak = record.peak_equity or self.initial_balance
        drawdown = float(self._loss_pct(Decimal(str(peak)), Decimal(str(current_equity))))
        mult = deleverage_multiplier(
            drawdown,
            dd_50=self.config.deleverage_dd_50,
            dd_25=self.config.deleverage_dd_25,
            dd_halt=self.config.deleverage_halt_dd,
        )
        if mult <= 0:
            reason = f"deleverage HALT: drawdown {drawdown:.1%} ≥ {float(self.config.deleverage_halt_dd):.0%}"
            if not self.circuit_breaker_tripped():
                self.trip_circuit_breaker(reason, timestamp_utc)
            return DeleverageDecision(Decimal("0"), drawdown, True, reason)
        return DeleverageDecision(
            mult, drawdown, False,
            f"drawdown {drawdown:.1%} → size ×{float(mult):.2f}",
        )

    def register_closed_trade(self, *, pnl: float, db, timestamp_utc: datetime) -> None:
        """Stage 0 integration for the consecutive-loss limit (edge-triggered).

        Call after every closed trade. A losing close that brings the
        consecutive-loss streak to the configured limit trips the circuit
        breaker, which then requires a manual reset — it never auto-resumes.
        Edge-triggering (per losing close, not polled pre-trade) means a
        manual reset genuinely re-enables trading until the *next* loss.
        """
        if pnl >= 0 or db is None:
            return
        losses = int(db.get_consecutive_losses())
        if losses >= self.config.consecutive_loss_limit and not self.circuit_breaker_tripped():
            self.trip_circuit_breaker(
                f"consecutive loss limit reached ({losses} losses)", timestamp_utc
            )

    def check_drawdown_limit(self, current_equity: float) -> tuple[bool, float]:
        """Return (breached, drawdown_pct) of current equity vs the stored peak.

        Read-only monitoring helper for post-candle checks: it does not
        update peak equity and does not trip the breaker (the pre-trade
        path in ``evaluate_equity_risk`` owns those transitions).
        """
        record = self.store.load()
        peak = record.peak_equity or self.initial_balance
        if peak <= 0:
            return False, 0.0
        drawdown = float(
            self._loss_pct(Decimal(str(peak)), Decimal(str(current_equity)))
        )
        return drawdown >= float(self.config.max_drawdown_pct), drawdown

    def evaluate_equity_risk(self, snapshot: RiskSnapshot) -> OrderDecision:
        """Trip the breaker when daily loss or drawdown thresholds are breached."""

        if self.circuit_breaker_tripped():
            return OrderDecision(False, self.config.paper_trading, "circuit breaker is tripped")

        daily_loss_pct = self._loss_pct(snapshot.day_start_equity_usdt, snapshot.equity_usdt)
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            reason = f"daily loss limit breached: {daily_loss_pct:.6f}"
            self.trip_circuit_breaker(reason, snapshot.timestamp_utc)
            return OrderDecision(False, self.config.paper_trading, reason)

        drawdown_pct = self._loss_pct(snapshot.peak_equity_usdt, snapshot.equity_usdt)
        if drawdown_pct >= self.config.max_drawdown_pct:
            reason = f"max drawdown breached: {drawdown_pct:.6f}"
            self.trip_circuit_breaker(reason, snapshot.timestamp_utc)
            return OrderDecision(False, self.config.paper_trading, reason)

        return OrderDecision(True, self.config.paper_trading, "equity risk within limits")

    def approve_order(
        self,
        *,
        order_notional_usdt: Decimal,
        resulting_position_notional_usdt: Decimal,
        snapshot: RiskSnapshot,
    ) -> OrderDecision:
        """Approve, paper-accept, or reject an order before execution."""

        equity_decision = self.evaluate_equity_risk(snapshot)
        if not equity_decision.accepted:
            return equity_decision

        if snapshot.open_positions >= self.config.max_concurrent_positions:
            return OrderDecision(False, self.config.paper_trading, "max concurrent positions reached")

        if order_notional_usdt <= 0:
            return OrderDecision(False, self.config.paper_trading, "order notional must be positive")

        if resulting_position_notional_usdt <= 0:
            return OrderDecision(False, self.config.paper_trading, "resulting position notional must be positive")

        if resulting_position_notional_usdt > self.config.max_position_notional_usdt:
            return OrderDecision(False, self.config.paper_trading, "resulting position exceeds max position notional")

        if self.config.paper_trading:
            return OrderDecision(True, True, "paper order accepted")

        return OrderDecision(True, False, "live order accepted")

    def can_open_trade(
        self,
        *,
        current_balance: float,
        open_position_count: int,
        daily_pnl: float,
        db=None,
        symbol: str | None = None,
        correlation=None,
    ) -> tuple[bool, str]:
        """Legacy execution guard used by PaperTrader.

        daily_pnl is assumed to include realized plus unrealized mark-to-market
        PnL when called from live execution.

        ``correlation`` (optional, Task 2.2): a CorrelationTracker or correlation
        matrix. When given, the portfolio-heat check is inflated by
        (1 + average pairwise correlation) across the open book plus ``symbol`` —
        so a second highly-correlated position is rejected sooner than nominal
        heat alone would. Omitted → the check is byte-identical to before.
        """

        now = datetime.now(UTC)
        # Subtract in Decimal — a float subtraction here would launder float
        # representation error into the day-start baseline of the risk check.
        day_start_equity = Decimal(str(current_balance)) - Decimal(str(daily_pnl))

        # Load and update peak equity. update_peak_equity() only raises the
        # peak and never rewrites breaker state, so this cannot overwrite a
        # concurrent trip from another thread.
        with self._breaker_lock:
            record = self.store.load()
            stored_peak = record.peak_equity or self.initial_balance
            new_peak = max(stored_peak, current_balance)
            if new_peak > stored_peak:
                self.store.update_peak_equity(new_peak)

        snapshot = RiskSnapshot(
            equity_usdt=Decimal(str(current_balance)),
            day_start_equity_usdt=day_start_equity if day_start_equity > 0 else Decimal("0.01"),
            peak_equity_usdt=Decimal(str(new_peak)),
            open_positions=open_position_count,
            timestamp_utc=now,
        )
        decision = self.evaluate_equity_risk(snapshot)
        if not decision.accepted:
            if "daily loss" in decision.reason:
                return False, "Daily drawdown limit breached"
            return False, decision.reason

        if open_position_count >= self.config.max_concurrent_positions:
            return False, "Max open positions reached"

        # Portfolio heat: refuse a new position when existing open risk plus
        # this trade's risk budget would exceed the cap. Heat is measured as
        # distance-to-stop, i.e. what the stops actually put at risk today —
        # not notional, which overstates hedged/tight-stop books.
        if db is not None and hasattr(db, "get_open_trades"):
            open_heat = Decimal("0")
            open_symbols: list[str] = []
            for trade in db.get_open_trades():
                try:
                    entry = Decimal(str(trade["entry_price"]))
                    stop = Decimal(str(trade["stop_loss"]))
                    qty = Decimal(str(trade["quantity"]))
                except (KeyError, TypeError):
                    continue  # rows without stops contribute no measurable heat
                open_heat += abs(entry - stop) * qty
                try:
                    sym = trade["symbol"]
                    if sym:
                        open_symbols.append(str(sym))
                except (KeyError, TypeError):
                    pass
            equity = Decimal(str(current_balance))
            if equity > 0:
                nominal = open_heat / equity + self.config.risk_per_trade_pct
                # Correlation-aware heat (Task 2.2): correlated positions carry
                # more real risk than their nominal sum, so inflate the heat by
                # (1 + average pairwise correlation) across the open book plus
                # the incoming symbol. avg_corr is clamped to [-1, 1] → the
                # multiplier is in [0, 2]. No correlation supplied → factor 1.
                avg_corr = 0.0
                factor = Decimal("1")
                if correlation is not None:
                    avg_corr = _avg_correlation(
                        correlation, open_symbols + ([symbol] if symbol else [])
                    )
                    factor = Decimal("1") + Decimal(str(avg_corr))
                projected = nominal * factor
                if projected > self.config.max_portfolio_heat_pct:
                    corr_note = f" ×(1+ρ̄ {avg_corr:+.2f})" if correlation is not None else ""
                    return False, (
                        f"Portfolio heat cap: {float(nominal):.1%}{corr_note} "
                        f"= {float(projected):.1%} exceeds "
                        f"{float(self.config.max_portfolio_heat_pct):.1%}"
                    )

        # Consecutive-loss enforcement moved to register_closed_trade():
        # a losing close that reaches the limit trips the circuit breaker
        # (checked above via evaluate_equity_risk), which requires a manual
        # reset. The previous time-based cooldown here auto-resumed after
        # loss_cooldown_hours, violating the never-auto-resume rule.

        return True, "OK"

    def calculate_position(
        self,
        *,
        symbol: str,
        side: str,
        entry_price: float,
        current_balance: float,
        atr: float | None = None,
        win_prob: float | None = None,
        payoff_ratio: float | None = None,
        target_vol: float | None = None,
        asset_vol: float | None = None,
        drawdown: float | None = None,
    ) -> PositionPlan | None:
        """Calculate a risk-based ATR stop position plan.

        Sizing (all Decimal — precision policy; quantity rounds DOWN to 8 dp so
        the plan never risks more than budget):

        - Risk fraction: fractional (Quarter-)Kelly when ``win_prob`` and
          ``payoff_ratio`` are supplied — a non-positive edge sizes to zero
          (no trade); otherwise the fixed ``risk_per_trade_pct`` (Task 2.1).
        - Volatility targeting: when ``target_vol`` is supplied, the notional is
          also capped so the position's volatility ≈ target — ``asset_vol``
          defaults to ATR/price. The final notional is the MIN of the Kelly
          budget, the vol-target, and the hard ``max_position_notional`` cap.
        """

        # atr != atr guards NaN, which would otherwise pass every <= check
        if entry_price <= 0 or current_balance <= 0 or atr is None or atr != atr or atr <= 0:
            return None
        if side not in {"buy", "sell"}:
            return None

        price = Decimal(str(entry_price))
        balance = Decimal(str(current_balance))
        atr_d = Decimal(str(atr))

        # Risk fraction: fractional Kelly (edge-driven) or the fixed fallback.
        if win_prob is not None and payoff_ratio is not None:
            risk_frac = kelly_fraction(
                win_prob, payoff_ratio,
                multiplier=self.config.kelly_multiplier,
                cap=self.config.max_kelly_fraction,
            )
            if risk_frac <= 0:
                return None                      # no edge → don't bet (Kelly)
        else:
            risk_frac = self.config.risk_per_trade_pct

        risk_amount = balance * risk_frac
        stop_distance = atr_d * self.config.atr_stop_multiplier
        stop_pct = stop_distance / price
        if stop_pct <= 0:
            return None

        candidates = [
            risk_amount / stop_pct,
            self.config.max_position_notional_usdt,
        ]

        # Volatility targeting: cap notional so position vol ≈ target_vol.
        if target_vol is not None:
            av = Decimal(str(asset_vol)) if asset_vol is not None else (atr_d / price)
            leverage = vol_target_leverage(target_vol, av)
            if leverage > 0:
                candidates.append(balance * leverage)

        position_value = min(candidates)

        # Drawdown deleveraging (Task 2.3): scale the notional by the schedule
        # multiplier; a halt-level drawdown (multiplier 0) declines the trade.
        if drawdown is not None:
            mult = deleverage_multiplier(
                drawdown,
                dd_50=self.config.deleverage_dd_50,
                dd_25=self.config.deleverage_dd_25,
                dd_halt=self.config.deleverage_halt_dd,
            )
            if mult <= 0:
                return None
            position_value = position_value * mult

        quantity = (position_value / price).quantize(_QTY_STEP, rounding=ROUND_DOWN)
        if quantity <= 0:
            return None
        # Recompute notional from the quantized quantity so the plan is
        # internally consistent (value = quantity * price exactly).
        position_value = quantity * price

        if side == "buy":
            stop_loss = price - stop_distance
            take_profit = price + stop_distance * self.config.reward_risk_ratio
        else:
            stop_loss = price + stop_distance
            take_profit = price - stop_distance * self.config.reward_risk_ratio

        if stop_loss <= 0 or take_profit <= 0:
            return None

        return PositionPlan(
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            position_value=position_value,
            risk_amount=risk_amount,
            reward_risk_ratio=self.config.reward_risk_ratio,
        )

    def check_position_exits(
        self,
        trade: dict,
        *,
        candle_high: float,
        candle_low: float,
        candle_open: float | None = None,
    ) -> tuple[str, float] | None:
        """Return an exit reason and FILL price when SL/TP is touched.

        Gap-through modeling: when ``candle_open`` is supplied and the
        candle opened beyond the trigger level, the fill is the open —
        assuming a fill exactly at the stop understates tail losses on
        precisely the trades where tail risk lives.
        """

        side = trade.get("side")
        stop_loss = float(trade["stop_loss"])
        take_profit = float(trade["take_profit"])

        if side == "buy":
            if candle_low <= stop_loss:
                fill = stop_loss if candle_open is None else min(stop_loss, candle_open)
                return "stop_loss", fill
            if candle_high >= take_profit:
                fill = take_profit if candle_open is None else max(take_profit, candle_open)
                return "take_profit", fill
        elif side == "sell":
            if candle_high >= stop_loss:
                fill = stop_loss if candle_open is None else max(stop_loss, candle_open)
                return "stop_loss", fill
            if candle_low <= take_profit:
                fill = take_profit if candle_open is None else min(take_profit, candle_open)
                return "take_profit", fill
        return None

    @staticmethod
    def _loss_pct(reference_equity: Decimal, current_equity: Decimal) -> Decimal:
        if current_equity >= reference_equity:
            return Decimal("0")
        loss = (reference_equity - current_equity) / reference_equity
        # Cautious rounding: round calculated losses UP (precision policy),
        # so threshold comparisons can only err toward tripping the breaker.
        return loss.quantize(_PCT_STEP, rounding=ROUND_UP)

    @staticmethod
    def _require_utc(timestamp_utc: datetime) -> None:
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware UTC")
        if timestamp_utc.utcoffset() != UTC.utcoffset(timestamp_utc):
            raise ValueError("timestamp_utc must be UTC")

