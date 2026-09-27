"""
Backtesting engine — tests trading strategies on historical data.
Uses vectorbt for high-performance vectorized backtesting.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from loguru import logger
from datetime import timezone
from risk import metrics
from risk.manager import (
    CircuitBreakerState,
    InMemoryCircuitBreakerStore,
    RiskConfig,
    RiskManager,
    _BreakerRecord,
)

from config.constants import DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT, TIMEFRAME_SECONDS
from backtesting.costs import CostModel


@dataclass
class BacktestResult:
    """Backtesting result with all key performance metrics."""

    strategy_name: str
    pair: str
    timeframe: str
    start_date: str
    end_date: str

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    buy_and_hold_return_pct: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_trade_duration_hours: float

    # Additional
    best_trade_pct: float
    worst_trade_pct: float

    # Circuit-breaker trips during the run (each = a simulated operator
    # review; see BacktestEngine.run breaker_review_hours). A strategy
    # that trips often is telling you its loss clustering exceeds policy.
    n_breaker_trips: int = 0

    # Raw per-trade records (entry/exit/pnl/side/…) — lets tests and
    # analysis tools inspect individual fills instead of only aggregates.
    trades: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__

    def print_summary(self) -> None:
        """Print a formatted performance summary."""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║          BACKTEST RESULTS — {self.strategy_name:<29}║
╠══════════════════════════════════════════════════════════╣
║  Pair: {self.pair:<20} Timeframe: {self.timeframe:<14}║
║  Period: {self.start_date} → {self.end_date}          
╠══════════════════════════════════════════════════════════╣
║  RETURNS                                                 ║
║  Total Return:       {self.total_return_pct:>8.2f}%                      ║
║  Annualized Return:  {self.annualized_return_pct:>8.2f}%                      ║
║  Buy & Hold:         {self.buy_and_hold_return_pct:>8.2f}%                      ║
╠══════════════════════════════════════════════════════════╣
║  RISK METRICS                                            ║
║  Sharpe Ratio:       {self.sharpe_ratio:>8.2f}                       ║
║  Max Drawdown:       {self.max_drawdown_pct:>8.2f}%                      ║
║  Calmar Ratio:       {self.calmar_ratio:>8.2f}                       ║
╠══════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                        ║
║  Total Trades:       {self.total_trades:>8}                       ║
║  Win Rate:           {self.win_rate_pct:>8.2f}%                      ║
║  Profit Factor:      {self.profit_factor:>8.2f}                       ║
║  Avg Win:            {self.avg_win_pct:>8.2f}%                      ║
║  Avg Loss:           {self.avg_loss_pct:>8.2f}%                      ║
║  Best Trade:         {self.best_trade_pct:>8.2f}%                      ║
║  Worst Trade:        {self.worst_trade_pct:>8.2f}%                      ║
╚══════════════════════════════════════════════════════════╝
        """)


def _equity(capital: float, position: float, entry_price: float, price: float) -> float:
    """Cash + collateral + SIGNED unrealized PnL.

    Mirrors PaperTrader._mark_to_market so backtest and live agree on what
    "equity" means (shorts gain when price falls). `position` is signed:
    >0 long, <0 short, 0 flat.
    """
    if position == 0:
        return capital
    qty = abs(position)
    unrealized = (price - entry_price) * qty if position > 0 else (entry_price - price) * qty
    return capital + entry_price * qty + unrealized


class MockDB:
    def __init__(self):
        self.consecutive_losses = 0
        self.last_exit_time = None
        self.trades = []

    def get_consecutive_losses(self):
        return self.consecutive_losses

    def get_trade_history(self, limit=1):
        if not self.trades:
            return []
        return self.trades[-limit:]

    def get_open_trades(self):
        return [t for t in self.trades if t.get("status") == "open"]


class BacktestEngine:
    """
    Pure Python backtesting engine.

    Simulates trading signals on historical data with:
    - Stop-loss and take-profit
    - Commission fees
    - Position sizing
    - Performance metrics calculation
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        commission_pct: float | None = None,   # legacy flat fee (both sides)
        slippage_pct: float | None = None,     # legacy flat slippage
        cost_model: CostModel | None = None,
    ) -> None:
        """Execution costs (Roadmap Task 1.2).

        - cost_model given → use it (realistic depth/latency/fee-by-type model).
        - legacy commission_pct/slippage_pct given → flat model (backward
          compatible; commission_pct=slippage_pct=0 → frictionless).
        - nothing given → realistic Binance VIP0 defaults (CostModel()).
        """
        self.initial_capital = initial_capital
        if cost_model is not None:
            self.cost_model = cost_model
        elif commission_pct is not None or slippage_pct is not None:
            self.cost_model = CostModel.flat(
                commission_pct=commission_pct if commission_pct is not None else 0.001,
                slippage_pct=slippage_pct if slippage_pct is not None else 0.0005,
            )
        else:
            self.cost_model = CostModel()   # realistic Task 1.2 defaults
        # Back-compat scalar views (legacy callers / reporting).
        self.commission_pct = self.cost_model.taker_fee_pct
        self.slippage_pct = self.cost_model.flat_slippage_pct or 0.0

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,  # +1=buy, -1=sell, 0=hold
        strategy_name: str = "Strategy",
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        risk_per_trade: float = 0.02,
        verbose: bool = True,   # False → no console banner (batch runs)
        breaker_review_hours: float | None = 24.0,
    ) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            df: OHLCV DataFrame
            signals: Series aligned with df — +1=buy, -1=sell, 0=hold
            strategy_name: Name for reporting
            pair: Trading pair
            timeframe: Candle timeframe
            stop_loss_pct: Stop-loss percentage from entry
            take_profit_pct: Take-profit percentage from entry
            risk_per_trade: Fraction of capital to risk per trade
            breaker_review_hours: SIMULATED OPERATOR. Live, a tripped
                circuit breaker requires a human review + manual reset
                (never-auto-resume rule). An unattended multi-year
                backtest has no human, so a trip would silently block
                every remaining candle — measured: 4,756 blocked entries
                across one lab run, i.e. results became "performance
                until the first losing streak". This models the operator
                reviewing and resetting after N hours. Trips are counted
                in the result (n_breaker_trips) so a strategy that trips
                constantly is visible, not hidden. None → strict live
                semantics (first trip halts the rest of the backtest).

        Returns:
            BacktestResult with all metrics
        """
        logger.info(
            "Running backtest: {} on {} ({}) — {} candles",
            strategy_name, pair, timeframe, len(df)
        )

        capital = self.initial_capital
        # Seeded RNG for the latency draws in the cost model → reproducible.
        cm = self.cost_model
        rng = np.random.default_rng(cm.seed)
        position = 0.0
        entry_price = 0.0
        entry_cost = 0.0
        entry_i = 0            # candle index where the open position was entered
        stop_loss = 0.0
        take_profit = 0.0

        trades: list[dict] = []
        equity_curve: list[float] = [capital]

        # Risk manager with the SAME settings-derived config live uses
        # (RiskConfig.from_settings) — the bare RiskManager() fallback
        # hardcodes 1% risk vs the configured 2%, which sized every
        # backtest at half the live risk. InMemory store keeps backtests
        # out of the live breaker DB.
        risk_manager = RiskManager(
            config=RiskConfig.from_settings(capital),
            initial_balance=capital,
            store=InMemoryCircuitBreakerStore()
        )
        mock_db = MockDB()

        current_day = None
        day_start_equity = capital
        n_breaker_trips = 0
        was_tripped = False
        trip_seen_ts = None

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            signal = signals.iloc[i]
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            volume = float(df["volume"].iloc[i]) if "volume" in df.columns else 0.0
            atr = df["ATR"].iloc[i] if "ATR" in df.columns else None

            ts = df.index[i]
            if not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if current_day is None or ts.date() != current_day:
                current_day = ts.date()
                day_start_equity = _equity(capital, position, entry_price, price)

            # ── Simulated operator: review + reset a tripped breaker ──────────
            # (see breaker_review_hours in the docstring; live semantics are
            # manual-reset-only and remain untouched in RiskManager).
            # Trip time is tracked in BACKTEST time from the candle where the
            # trip is first observed — the store's tripped_at_utc is wall
            # clock (datetime.now), which never elapses against 2024 candles.
            tripped_now = risk_manager.circuit_breaker_tripped()
            if tripped_now and not was_tripped:
                n_breaker_trips += 1
                trip_seen_ts = ts
            was_tripped = tripped_now
            if tripped_now and breaker_review_hours is not None:
                elapsed_h = (ts - trip_seen_ts).total_seconds() / 3600.0
                if elapsed_h >= breaker_review_hours:
                    logger.debug(
                        "Simulated operator reset after {:.0f}h (trip: {})",
                        elapsed_h, risk_manager.store.load().reason,
                    )
                    risk_manager.manual_reset()
                    # The review covers the loss streak too — otherwise the
                    # stale counter re-trips on the next candle forever.
                    mock_db.consecutive_losses = 0
                    was_tripped = False

            # ── Check exit conditions for open position (long OR short) ───────
            if position != 0:
                open_px = df["open"].iloc[i]
                qty = abs(position)
                is_long = position > 0

                if is_long:
                    hit_stop, hit_tp = low <= stop_loss, high >= take_profit
                else:
                    # Short: stop is ABOVE entry, target BELOW
                    hit_stop, hit_tp = high >= stop_loss, low <= take_profit

                if hit_stop or hit_tp:
                    # Gap-through: a candle that OPENS beyond the trigger
                    # fills at the open, not the trigger price — assuming a
                    # clean stop fill understates tail losses.
                    if is_long:
                        fill = min(stop_loss, open_px) if hit_stop else max(take_profit, open_px)
                        exit_price = cm.fill_price(fill, qty, volume, is_buy=False, rng=rng)
                    else:
                        fill = max(stop_loss, open_px) if hit_stop else min(take_profit, open_px)
                        # buying back
                        exit_price = cm.fill_price(fill, qty, volume, is_buy=True, rng=rng)
                    exit_reason = "stop_loss" if hit_stop else "take_profit"

                    proceeds = exit_price * qty
                    # Stop-loss crosses the book (taker); take-profit rests as a
                    # limit order (maker) — Task 1.2 fee-by-order-type.
                    commission = cm.fee(proceeds, "taker" if hit_stop else "maker")
                    entry_notional = entry_price * qty

                    # PnL CONVENTION (pinned by tier-14 parity tests): the
                    # per-trade stat is net exit vs GROSS entry — the entry
                    # fee is paid by capital at open but excluded from pnl,
                    # exactly like PaperTrader._close_position (which uses
                    # entry_cost = entry_price * quantity). Equity curves
                    # carry both fees; "fixing" either side alone would
                    # silently break backtest/live parity.
                    if is_long:
                        net_proceeds = proceeds - commission
                        capital += net_proceeds
                        pnl = net_proceeds - entry_notional
                        net_return_pct = ((net_proceeds / qty - entry_price) / entry_price) * 100
                    else:
                        # Short buy-back: the exit fee ADDS to the cost.
                        # Collateral (entry_notional) returns with the PnL —
                        # identical to PaperTrader._close_position.
                        buyback_cost = proceeds + commission
                        pnl = entry_notional - buyback_cost
                        capital += entry_notional + pnl
                        net_return_pct = ((entry_price - buyback_cost / qty) / entry_price) * 100

                    trade = {
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position": position,
                        "side": "buy" if is_long else "sell",
                        "pnl": pnl,
                        "return_pct": net_return_pct,
                        "exit_reason": exit_reason,
                        "entry_idx": entry_i,
                        "exit_idx": i,
                        "exit_time": ts.isoformat()
                    }
                    trades.append(trade)
                    mock_db.trades.append(trade)

                    if not metrics.is_win(pnl):
                        mock_db.consecutive_losses += 1
                    else:
                        mock_db.consecutive_losses = 0
                    # Same edge-triggered consecutive-loss breaker live uses
                    # (engine previously never called this — a parity gap:
                    # backtests under-enforced the loss-streak trip)
                    risk_manager.register_closed_trade(
                        pnl=pnl, db=mock_db, timestamp_utc=ts)

                    position = 0.0
                    entry_cost = 0.0

            # Determine daily PnL
            current_equity = _equity(capital, position, entry_price, price)
            daily_pnl = current_equity - day_start_equity

            # ── Open new position on BUY signal ───────────────────────────────
            # ── Open a position (BUY → long, SELL → short) ────────────────────
            # PaperTrader opens a SHORT on a SELL signal, so the engine must
            # too — a long-only backtest cannot validate half of what the
            # live bot does (it silently ignored every SELL entry signal).
            if signal in (1, -1) and position == 0 and capital > 0:
                going_long = signal == 1
                allowed, reason = risk_manager.can_open_trade(
                    current_balance=current_equity,
                    open_position_count=0,
                    daily_pnl=daily_pnl,
                    db=mock_db
                )

                if allowed:
                    # Sizing basis: flat mode sizes on the slipped price (legacy
                    # parity); the depth model needs qty first, so it sizes on
                    # the signal price and applies depth+latency slippage after.
                    if cm.is_flat:
                        sizing_price = cm.fill_price(price, 0.0, volume, is_buy=going_long, rng=rng)
                    else:
                        sizing_price = price

                    # Position sizing via risk manager — the SAME sizing
                    # model live uses. No static fallback: a backtest that
                    # sizes differently from production certifies a
                    # different bot (parity rule). No ATR → no trade.
                    plan = None
                    if atr is not None and atr == atr and atr > 0:
                        plan = risk_manager.calculate_position(
                            symbol=pair,
                            side="buy" if going_long else "sell",
                            entry_price=sizing_price,
                            current_balance=capital,
                            atr=atr
                        )
                    if plan:
                        # Plan fields are Decimal; the backtest engine's
                        # vectorized math stays in float at its boundary.
                        qty = float(plan.quantity)
                        stop_loss = float(plan.stop_loss)
                        take_profit = float(plan.take_profit)

                        # Entries cross the book → taker (Task 1.2).
                        entry_price = (
                            sizing_price if cm.is_flat
                            else cm.fill_price(price, qty, volume, is_buy=going_long, rng=rng)
                        )
                        entry_notional = entry_price * qty
                        commission = cm.fee(entry_notional, "taker")
                        # Collateral model (mirrors PaperTrader): BOTH sides
                        # reserve notional + entry fee.
                        entry_cost = entry_notional + commission
                        if entry_cost > capital:
                            # Affordability invariant: a fill may never take
                            # cash below zero (the paper trader rejects the
                            # same way — parity, no unmodeled leverage).
                            logger.debug(
                                "Backtest entry skipped: cost {:.2f} > capital {:.2f}",
                                entry_cost, capital,
                            )
                            entry_cost = 0.0
                        else:
                            capital -= entry_cost
                            position = qty if going_long else -qty
                            entry_i = i   # remember where this position opened
                else:
                    logger.debug("Trade blocked by RiskManager in backtest: {}", reason)

            # ── Close on an OPPOSITE signal ───────────────────────────────────
            elif (signal == -1 and position > 0) or (signal == 1 and position < 0):
                qty = abs(position)
                is_long = position > 0
                # Opposite-signal close is a market order → taker (Task 1.2).
                exit_price = cm.fill_price(price, qty, volume, is_buy=not is_long, rng=rng)
                proceeds = exit_price * qty
                commission = cm.fee(proceeds, "taker")
                entry_notional = entry_price * qty

                # Same pnl convention as the stop/TP close above (net exit
                # vs gross entry) — see the comment there.
                if is_long:
                    net_proceeds = proceeds - commission
                    capital += net_proceeds
                    pnl = net_proceeds - entry_notional
                    net_return_pct = ((net_proceeds / qty - entry_price) / entry_price) * 100
                else:
                    buyback_cost = proceeds + commission
                    pnl = entry_notional - buyback_cost
                    capital += entry_notional + pnl
                    net_return_pct = ((entry_price - buyback_cost / qty) / entry_price) * 100

                trade = {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position": position,
                    "side": "buy" if is_long else "sell",
                    "pnl": pnl,
                    "return_pct": net_return_pct,
                    "exit_reason": "signal",
                    "entry_idx": entry_i,
                    "exit_idx": i,
                    "exit_time": ts.isoformat()
                }
                trades.append(trade)
                mock_db.trades.append(trade)

                if not metrics.is_win(pnl):
                    mock_db.consecutive_losses += 1
                else:
                    mock_db.consecutive_losses = 0
                risk_manager.register_closed_trade(
                    pnl=pnl, db=mock_db, timestamp_utc=ts)

                position = 0.0
                entry_cost = 0.0

            # Update circuit breaker tracking for Max Drawdown
            current_equity = _equity(capital, position, entry_price, price)
            if risk_manager.store:
                record = risk_manager.store.load()
                peak = record.peak_equity or capital
                if current_equity > peak:
                    risk_manager.store.save(
                        _BreakerRecord(
                            state=record.state,
                            reason=record.reason,
                            tripped_at_utc=record.tripped_at_utc,
                            peak_equity=current_equity
                        )
                    )
                elif peak > 0:
                    dd = float((peak - current_equity) / peak)
                    if dd >= float(risk_manager.config.max_drawdown_pct):
                        if record.state != CircuitBreakerState.TRIPPED:
                            risk_manager.trip_circuit_breaker(f"Max Drawdown Limit Reached: {dd*100:.1f}%", ts)

            equity_curve.append(current_equity)

        result = self._compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            df=df,
            strategy_name=strategy_name,
            pair=pair,
            timeframe=timeframe,
        )

        result.n_breaker_trips = n_breaker_trips
        if n_breaker_trips:
            logger.info("Breaker tripped {}x during backtest (simulated operator resets)", n_breaker_trips)
        if verbose:
            result.print_summary()
        return result

    def _compute_metrics(
        self,
        trades: list[dict],
        equity_curve: list[float],
        df: pd.DataFrame,
        strategy_name: str,
        pair: str,
        timeframe: str,
    ) -> BacktestResult:
        """Compute all performance metrics from trade history and equity curve."""

        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        # ── Basic returns ──────────────────────────────────────────────────────
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        n_periods = len(returns)  # Number of return periods, not candle count
        periods_per_year = metrics.PERIODS_PER_YEAR.get(timeframe, 8760)
        annualized_return = (
            ((1 + total_return / 100) ** (periods_per_year / n_periods) - 1) * 100
            if n_periods > 0 else 0.0  # degenerate input: report 0, don't crash
        )
        buy_and_hold = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

        # ── Risk metrics (single definitions in risk/metrics.py) ─────────────
        sharpe = metrics.sharpe_annualized(returns.tolist(), periods_per_year)
        sortino = metrics.sortino_annualized(returns.tolist(), periods_per_year)

        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()

        calmar = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0

        # ── Trade stats ────────────────────────────────────────────────────────
        if trades:
            returns_list = [t["return_pct"] for t in trades]
            wins = [r for r in returns_list if r > 0]
            losses = [r for r in returns_list if r <= 0]

            win_rate = len(wins) / len(trades) * 100 if trades else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = (
                sum(wins) / abs(sum(losses))
                if losses and sum(losses) != 0 else float("inf")
            )
            best_trade = max(returns_list)
            worst_trade = min(returns_list)
            # Holding period per trade = (exit_idx − entry_idx) candles × the
            # candle length. Now meaningful since entry_idx is the real entry.
            period_h = TIMEFRAME_SECONDS.get(timeframe, 3600) / 3600.0
            durations = [(t["exit_idx"] - t["entry_idx"]) * period_h for t in trades]
            avg_duration = float(np.mean(durations)) if durations else 0.0
        else:
            win_rate = avg_win = avg_loss = best_trade = worst_trade = 0
            profit_factor = 0
            avg_duration = 0.0

        return BacktestResult(
            strategy_name=strategy_name,
            pair=pair,
            timeframe=timeframe,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annualized_return, 2),
            buy_and_hold_return_pct=round(buy_and_hold, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=round(max_drawdown, 2),
            calmar_ratio=round(calmar, 3),
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t["return_pct"] > 0]),
            losing_trades=len([t for t in trades if t["return_pct"] <= 0]),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 3),
            avg_win_pct=round(avg_win, 3),
            avg_loss_pct=round(avg_loss, 3),
            avg_trade_duration_hours=round(avg_duration, 1),
            best_trade_pct=round(best_trade, 3),
            worst_trade_pct=round(worst_trade, 3),
            trades=trades,
        )
