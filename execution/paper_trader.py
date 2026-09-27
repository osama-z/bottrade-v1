"""
Paper Trader — Simulated trade execution engine for NeuronTrade.

Manages a virtual USDT wallet and executes simulated orders using
real-time market prices from Binance. No real money is used.

Uses:
    - RiskManager for position sizing + safety checks
    - TradeLogger for all persistence
    - TelegramBot (injected) for push alerts

This is the main component that connects the AI brain (Phase 2)
to the execution layer (Phase 3).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Optional

import pandas as pd
from loguru import logger

from config.settings import settings
from config.constants import TIMEFRAME_SECONDS, Signal
from risk.manager import RiskConfig, RiskManager, SqliteCircuitBreakerStore, PositionPlan
from storage.trade_logger import TradeLogger

if TYPE_CHECKING:
    from notifications.telegram_bot import TelegramBot


@dataclass
class PortfolioState:
    """Snapshot of the current portfolio."""
    balance: float                          # Available USDT
    initial_balance: float                  # Starting USDT
    open_trades: list[dict] = field(default_factory=list)
    total_pnl: float = 0.0                  # Lifetime PnL
    daily_pnl: float = 0.0                  # Today's PnL
    open_position_count: int = 0

    # Canonical equity, computed once by PaperTrader._mark_to_market()
    # (cash + collateral + signed unrealized PnL — shorts included).
    equity: float = 0.0

    @property
    def total_return_pct(self) -> float:
        """Return percentage since start."""
        if self.initial_balance == 0:
            return 0.0
        return (self.equity - self.initial_balance) / self.initial_balance * 100


class PaperTrader:
    """
    Simulated trading engine — executes the full trading loop using
    paper (virtual) money, real market prices.

    Flow per candle:
        1. Check open positions for SL/TP hits
        2. Get AI signal from AICombinedStrategy
        3. Log signal to database (every candle, regardless of action)
        4. If signal is actionable → RiskManager validates → open position
        5. Notify Telegram (async, non-blocking)

    Usage:
        trader = PaperTrader(initial_balance=10000.0)
        trader.process_candle(df=df, pair="BTC/USDT", ai_signal=signal)
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        db: Optional[TradeLogger] = None,
        telegram: Optional["TelegramBot"] = None,
        commission_pct: float = 0.001,      # 0.1% Binance fee
        slippage_pct: float = 0.0005,       # 0.05% slippage
        db_path: Optional[str] = None,      # Path to SQLite file for persistent breaker
        risk_manager: Optional[RiskManager] = None,  # injected for tests/sims
        tax_ledger=None,                    # optional TaxLotLedger (Task 6.1)
    ) -> None:
        self.initial_balance = initial_balance
        # Wallet arithmetic runs in Decimal (precision policy): float
        # accumulation error must not drift the balance over thousands of
        # trades. Converted to float only at DB/reporting boundaries.
        self._balance = Decimal(str(initial_balance))
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self._commission = Decimal(str(commission_pct))
        self._slippage = Decimal(str(slippage_pct))
        self._lock = threading.RLock()      # Protects _balance and state flags

        # Injected dependencies
        self._db = db or TradeLogger()
        self._telegram = telegram  # Optional — injected after init
        self._tax = tax_ledger     # Optional tax-lot ledger (Task 6.1); None → skipped

        # Restore the persisted cash balance so a restart cannot reset the
        # risk envelope while open trades survive in the database.
        persisted_balance = self._db.load_balance()
        if persisted_balance is not None:
            self._balance = Decimal(str(persisted_balance))
            logger.info(
                "Restored persisted balance ${:.2f} from database "
                "(constructor initial_balance ${:.2f} not applied)",
                persisted_balance, initial_balance,
            )
        else:
            self._db.save_balance(float(self._balance))

        # Risk manager wired to persistent SQLite store so tripped breakers
        # survive process restarts. Falls back to in-memory for tests that
        # don't provide a db_path.
        _store = SqliteCircuitBreakerStore(db_path) if db_path else None
        if _store is None:
            logger.warning(
                "No db_path provided — circuit breaker state is IN-MEMORY ONLY "
                "and will NOT survive a restart. Acceptable in tests only."
            )
        # Risk limits come from settings (single source of truth) via the
        # shared factory — the backtest engine builds its config the same
        # way, so both systems size identically (tier-14 parity pin).
        # An injected risk_manager (tests/simulations) takes precedence.
        self._risk = risk_manager or RiskManager(
            config=RiskConfig.from_settings(initial_balance),
            store=_store, initial_balance=initial_balance
        )

        # System state flags
        self.is_paused: bool = False        # Set by /pause command
        self.is_running: bool = True        # Set False by /emergency_stop

        logger.info(
            "PaperTrader initialized | balance=${:.2f} | commission={:.2%} | slippage={:.2%}",
            initial_balance, commission_pct, slippage_pct
        )

    # ─── Equity (single canonical definition) ──────────────────────────────────

    def _mark_to_market(
        self,
        open_trades: list[dict],
        price_lookup: Optional[dict[str, float]] = None,
    ) -> Decimal:
        """Canonical equity: cash + collateral + signed unrealized PnL.

        Shorts gain when price falls. Pairs without a live price in
        ``price_lookup`` are valued at entry (unrealized = 0) — a price is
        never borrowed from another pair.
        """
        total = self._balance
        lookup = price_lookup or {}
        for t in open_trades:
            entry = Decimal(str(t["entry_price"]))
            qty = Decimal(str(t["quantity"]))
            price = Decimal(str(lookup.get(t["symbol"], t["entry_price"])))
            if t.get("side") == "buy":
                unrealized = (price - entry) * qty
            else:
                unrealized = (entry - price) * qty
            total += entry * qty + unrealized
        return total

    def equity(self, price_lookup: Optional[dict[str, float]] = None) -> float:
        """Current equity; pass live prices to mark open positions."""
        with self._lock:
            return float(self._mark_to_market(self._db.get_open_trades(), price_lookup))

    # ─── Main Entry Point ──────────────────────────────────────────────────────

    def process_candle(
        self,
        df: pd.DataFrame,
        pair: str,
        ai_signal,          # AISignal from signal_combiner.py
        correlation_id: str = "",
    ) -> None:
        """
        Main method — called once per candle close.

        Args:
            df: Full OHLCV + indicators DataFrame (latest candle is last row)
            pair: Trading pair symbol (e.g., "BTC/USDT")
            ai_signal: AISignal dataclass from SignalCombiner.combine()
            correlation_id: joins this cycle's signal, trade, and risk
                events in the audit trail (ZMQ message_id in decoupled
                mode; a per-cycle UUID in run_live)
        """
        if not self.is_running:
            return

        # ── Step 0: Check for stale data (data integrity fail-safe) ───────────
        from datetime import datetime, timezone

        latest_time = df.index[-1]
        now_time = datetime.now(timezone.utc)
        timeframe = settings.default_timeframe
        duration = TIMEFRAME_SECONDS.get(timeframe, 3600)

        expected_close_time = latest_time + pd.Timedelta(seconds=duration)
        time_diff = (now_time - expected_close_time).total_seconds()

        # The live loop decides on the last CLOSED candle (the forming candle is
        # dropped upstream), so at startup / just after a restart the newest
        # candle is legitimately up to one full period old — that is NOT a dead
        # feed. Allow one period plus the configured grace before aborting; a
        # genuinely stale feed is more than one period behind and still caught.
        stale_threshold = settings.stale_data_seconds + duration
        if time_diff > stale_threshold:
            logger.error(
                "Data integrity check FAILED: latest candle is stale for {}! "
                "Expected close: {}, Current time: {}, Difference: {:.1f}s "
                "(threshold {:.0f}s). Aborting cycle.",
                pair, expected_close_time, now_time, time_diff, stale_threshold
            )
            if self._telegram:
                self._telegram.send_error(
                    f"Data integrity check FAILED for {pair}!\n"
                    f"Latest candle open: {latest_time}\n"
                    f"Stale by {time_diff/60:.1f} minutes. Cycle aborted."
                )
            return

        latest = df.iloc[-1]
        current_price = float(latest["close"])
        candle_high = float(latest["high"])
        candle_low = float(latest["low"])
        candle_open = float(latest["open"])
        # compute_all() writes the column as "ATR"; guard against NaN warm-up
        # rows so sizing never sees a non-finite stop distance.
        atr: Optional[float] = None
        if "ATR" in df.columns and pd.notna(latest["ATR"]):
            atr = float(latest["ATR"])

        # ── Step 1: Check existing positions for SL/TP ────────────────────────
        self._check_and_close_positions(
            candle_high, candle_low, current_price, atr, candle_open=candle_open
        )

        # ── Step 2: Log the AI signal (always, even if HOLD) ─────────────────
        self._db.log_signal(
            symbol=pair,
            ml_score=ai_signal.ml_score,
            llm_score=ai_signal.llm_score,
            sentiment_score=ai_signal.sentiment_score,
            combined_score=ai_signal.score,
            confidence=ai_signal.confidence,
            decision=ai_signal.signal.value,
            correlation_id=correlation_id,
        )

        # ── Step 3: Skip if paused or no actionable signal ────────────────────
        if self.is_paused:
            logger.debug("Bot is paused — skipping signal processing")
            return

        if not ai_signal.is_actionable:
            logger.debug(
                "Signal not actionable | {} | score={:+.3f} | confidence={:.0%}",
                ai_signal.signal.value, ai_signal.score, ai_signal.confidence
            )
            return

        if ai_signal.signal == Signal.HOLD:
            return

        # ── Step 4: Validate with Risk Manager ────────────────────────────────
        # Use total equity (cash + mark-to-market of open positions) so that
        # unrealized losses on open trades count toward the daily loss limit.
        # Steps 4-6 run under the lock so the risk check and the order that
        # follows it are atomic with respect to concurrent scheduler threads
        # (otherwise two threads can both pass the position-count check).
        with self._lock:
            open_trades = self._db.get_open_trades()
            open_count = len(open_trades)
            # Canonical mark-to-market: only THIS pair has a live price;
            # other pairs are valued at entry rather than borrowing this
            # pair's price (the previous inline math applied current_price
            # to every open trade regardless of symbol).
            marked_equity = self._mark_to_market(open_trades, {pair: current_price})
            entry_valued = self._mark_to_market(open_trades, None)
            unrealized_pnl = float(marked_equity - entry_valued)
            realized_pnl = self._db.get_daily_pnl()
            # Marked equity feeds the risk check, so drawdown and daily-loss
            # limits see unrealized losses instead of entry-price fiction.
            total_equity = float(marked_equity)
            daily_pnl = realized_pnl + unrealized_pnl

            allowed, reason = self._risk.can_open_trade(
                current_balance=total_equity,
                open_position_count=open_count,
                daily_pnl=daily_pnl,
                db=self._db,
            )

            if not allowed:
                logger.warning("Trade blocked by RiskManager: {}", reason)
                self._db.log_risk_event(
                    "trade_blocked", symbol=pair, reason=reason,
                    correlation_id=correlation_id,
                )
                return

            # ── Step 5: Calculate position plan ───────────────────────────────
            side = "buy" if ai_signal.signal == Signal.BUY else "sell"
            entry_price = current_price * (
                1 + self.slippage_pct if side == "buy" else 1 - self.slippage_pct
            )

            plan = self._risk.calculate_position(
                symbol=pair,
                side=side,
                entry_price=entry_price,
                current_balance=float(self._balance),
                atr=atr,
            )

            if plan is None:
                self._db.log_risk_event(
                    "sizing_failed", symbol=pair,
                    reason=f"no position plan (atr={atr})",
                    correlation_id=correlation_id,
                )
                return

            # ── Step 6: Execute virtual order ─────────────────────────────────
            self._open_position(plan, ai_signal, correlation_id=correlation_id)

    # ─── Position Management ──────────────────────────────────────────────────

    def _open_position(
        self, plan: PositionPlan, ai_signal, correlation_id: str = ""
    ) -> None:
        """Open a new simulated position. Cost math in Decimal (plan fields
        are Decimal); converted to float only at the DB/Telegram boundary."""
        commission = plan.position_value * self._commission
        total_cost = plan.position_value + commission

        if total_cost > self._balance:
            logger.warning(
                "Insufficient balance: need ${:.2f}, have ${:.2f}",
                total_cost, self._balance
            )
            return

        # Persist the trade before mutating the balance (persist-then-act)
        trade_id = self._db.log_trade_open(
            symbol=plan.symbol,
            side=plan.side,
            quantity=float(plan.quantity),
            entry_price=float(plan.entry_price),
            stop_loss=float(plan.stop_loss),
            take_profit=float(plan.take_profit),
            correlation_id=correlation_id,
            ai_score=ai_signal.score,
            ai_confidence=ai_signal.confidence,
            ai_reasoning=(
                ai_signal.llm_analysis.reasoning
                if ai_signal.llm_analysis and not ai_signal.llm_analysis.mock
                else f"AI score={ai_signal.score:+.3f}"
            ),
        )

        # Deduct cost from virtual balance and persist it
        self._balance -= total_cost
        self._db.save_balance(float(self._balance))

        # Tax-lot ledger (Task 6.1): a long entry acquires a lot.
        self._tax_acquire(plan, correlation_id)

        logger.info(
            "📈 PAPER TRADE OPENED | id={} | {} {} {} @ ${:.2f} | "
            "SL=${:.2f} TP=${:.2f} | Balance=${:.2f}",
            trade_id, plan.side.upper(), plan.quantity, plan.symbol,
            plan.entry_price, plan.stop_loss, plan.take_profit, self._balance
        )

        # Notify Telegram (non-blocking)
        if self._telegram:
            self._telegram.send_trade_opened(
                pair=plan.symbol,
                side=plan.side,
                price=float(plan.entry_price),
                quantity=float(plan.quantity),
                stop_loss=float(plan.stop_loss),
                take_profit=float(plan.take_profit),
                ai_score=ai_signal.score,
                ai_confidence=ai_signal.confidence,
                reasoning=ai_signal.llm_analysis.reasoning if (
                    ai_signal.llm_analysis and not ai_signal.llm_analysis.mock
                ) else f"Score={ai_signal.score:+.3f}",
                trade_id=trade_id,
            )

    def _close_position(
        self,
        trade: dict,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Close an open position and calculate PnL.

        PnL math runs in Decimal. Wallet model: opening either side reserves
        the full notional plus entry fee as collateral, so closing returns:
        - buy:  the sale proceeds net of exit fee;
        - sell: the reserved collateral plus the short's PnL (the exit is a
          BUY-back, so the exit fee ADDS to its cost).
        Round-trip balance change equals recorded PnL minus the entry fee
        for both sides.
        """
        entry_price = Decimal(str(trade["entry_price"]))
        quantity = Decimal(str(trade["quantity"]))
        side = trade["side"]
        one = Decimal("1")

        # Apply slippage on exit (adverse direction for each side)
        if side == "buy":
            actual_exit = Decimal(str(exit_price)) * (one - self._slippage)
        else:
            actual_exit = Decimal(str(exit_price)) * (one + self._slippage)

        proceeds = actual_exit * quantity
        commission = proceeds * self._commission

        # Gross cost (what we originally spent as collateral, ex entry fee)
        entry_cost = entry_price * quantity

        if side == "buy":
            net_proceeds = proceeds - commission
            pnl = net_proceeds - entry_cost
            balance_credit = net_proceeds
        else:
            # Short buy-back: commission increases the cost of exiting
            buyback_cost = proceeds + commission
            pnl = entry_cost - buyback_cost
            balance_credit = entry_cost + pnl

        with self._lock:
            # Persist first: the status='open' guard in log_trade_close makes
            # the close atomic, so two threads racing to close the same trade
            # credit the balance exactly once.
            closed = self._db.log_trade_close(
                trade_id=trade["id"],
                exit_price=float(actual_exit),
                pnl=float(pnl.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)),
                exit_reason=exit_reason,
            )
            if not closed:
                return

            # Return proceeds/collateral to virtual balance
            self._balance += balance_credit
            self._db.save_balance(float(self._balance))

        # Tax-lot ledger (Task 6.1): closing a long disposes matched lots (FIFO).
        self._tax_dispose(trade, float(actual_exit))

        # Stage 0: a losing close that reaches the consecutive-loss limit
        # trips the circuit breaker (manual reset required — no auto-resume).
        self._risk.register_closed_trade(
            pnl=float(pnl), db=self._db, timestamp_utc=datetime.now(timezone.utc)
        )

        emoji = "✅" if pnl >= 0 else "❌"
        logger.info(
            "{} PAPER TRADE CLOSED | id={} | {} | exit=${:.2f} | "
            "PnL=${:+.2f} | reason={} | Balance=${:.2f}",
            emoji, trade["id"], trade["symbol"], actual_exit,
            pnl, exit_reason, self._balance
        )

        # Notify Telegram (non-blocking)
        if self._telegram:
            self._telegram.send_trade_closed(
                pair=trade["symbol"],
                side=trade["side"],
                entry_price=float(entry_price),
                exit_price=float(actual_exit),
                pnl=float(pnl),
                exit_reason=exit_reason,
                trade_id=trade["id"],
            )

    # ── Tax-lot ledger hooks (Task 6.1) — long-only spot; best-effort ──────────
    def _tax_acquire(self, plan, correlation_id: str) -> None:
        """A long entry acquires a lot; the correlation_id is the external lot
        reference (the candle's clientOrderId). Never breaks trading."""
        if self._tax is None or plan.side != "buy":
            return
        try:
            self._tax.acquire(plan.symbol, float(plan.quantity),
                              float(plan.entry_price), correlation_id=correlation_id)
        except Exception as e:
            logger.error("tax-lot acquire failed for {}: {}", plan.symbol, e)

    def _tax_dispose(self, trade: dict, exit_price: float) -> None:
        """Closing a long disposes matched lots (FIFO), booking realized P&L."""
        if self._tax is None or trade.get("side") != "buy":
            return
        try:
            self._tax.dispose(str(trade["symbol"]), float(trade["quantity"]),
                             float(exit_price), method="fifo",
                             correlation_id=str(trade.get("correlation_id", "")))
        except Exception as e:
            logger.error("tax-lot dispose failed for {}: {}", trade.get("symbol"), e)

    def _check_and_close_positions(
        self,
        candle_high: float,
        candle_low: float,
        current_price: float,
        atr: Optional[float] = None,
        candle_open: Optional[float] = None,
    ) -> None:
        """
        Scan all open positions and:
        1. Check for 50% scale-out trigger (at 1.5R).
        2. Update highest price reached and trailing SL.
        3. Check for full exit (stop loss or take profit).
        """
        open_trades = self._db.get_open_trades()

        for trade in open_trades:
            side = trade["side"]
            highest_price = float(trade.get("highest_price") or trade["entry_price"])
            current_sl = float(trade["stop_loss"])
            take_profit = float(trade["take_profit"])
            entry_price = float(trade["entry_price"])
            scaled_out = int(trade.get("scaled_out") or 0)
            quantity = float(trade["quantity"])

            # Initial risk derived from the CONFIGURED reward:risk ratio —
            # the take-profit was placed at entry + RR * risk, so risk =
            # (TP - entry) / RR. (Previously hardcoded to RR=2, silently
            # wrong whenever config.reward_risk_ratio differed, and sells
            # got a fabricated flat 2% "risk".)
            rr = float(self._risk.config.reward_risk_ratio)
            initial_risk = (take_profit - entry_price) / rr if side == "buy" else 0.0
            scale_out_price = entry_price + 1.5 * initial_risk

            # ── Check Scale-Out (longs only; short scale-out/trailing is
            # not modeled — sells skip cleanly instead of using fake risk) ──
            if (
                side == "buy" and scaled_out == 0
                and initial_risk > 0 and candle_high >= scale_out_price
            ):
                # Execution price is scale_out_price (with slippage)
                partial_qty = quantity * 0.5
                exit_price = scale_out_price * (1 - self.slippage_pct)
                proceeds = exit_price * partial_qty
                commission = proceeds * self.commission_pct
                net_proceeds = proceeds - commission

                entry_cost = entry_price * partial_qty
                pnl = net_proceeds - entry_cost

                quantity = quantity - partial_qty
                scaled_out = 1
                current_sl = entry_price  # Move SL to breakeven
                highest_price = max(highest_price, candle_high)

                with self._lock:
                    # Persist the trade mutation before crediting the balance
                    self._db.update_trade_trailing_state(
                        trade_id=trade["id"],
                        highest_price=highest_price,
                        scaled_out=1,
                        stop_loss=current_sl,
                        quantity=quantity
                    )
                    self._balance += Decimal(str(net_proceeds))
                    self._db.save_balance(float(self._balance))

                # Partial-exit PnL was previously only a log line — persist
                # it so the audit trail accounts for every cash movement.
                self._db.log_risk_event(
                    "scale_out", symbol=trade["symbol"],
                    reason=f"sold 50% ({partial_qty:.6f}) @ {exit_price:.2f}, pnl={pnl:+.4f}",
                )

                logger.info(
                    "🎯 PARTIAL SCALE-OUT | id={} | Sold 50% ({:.6f}) @ ${:.2f} | PnL=${:+.2f} | New SL=Breakeven (${:.2f})",
                    trade["id"], partial_qty, exit_price, pnl, entry_price
                )

                if self._telegram:
                    msg = (
                        f"🎯 *Partial Scale-Out* #{trade['id']}\n"
                        f"{'─' * 28}\n"
                        f"Pair:   `{trade['symbol']}`\n"
                        f"Sold:   `50% ({partial_qty:.6f})` @ `${exit_price:,.2f}`\n"
                        f"PnL:    `${pnl:+.2f}`\n"
                        f"SL:     Moved to Breakeven `${entry_price:,.2f}`"
                    )
                    self._telegram._send(msg)

            # ── Update Trailing Stop Price ──────────────────────────────────
            if side == "buy":
                if candle_high > highest_price:
                    highest_price = candle_high

                if atr is not None and atr > 0:
                    potential_sl = highest_price - 1.5 * atr
                    # If scaled out, SL cannot drop below entry price (breakeven)
                    floor_sl = entry_price if scaled_out == 1 else current_sl
                    potential_sl = max(potential_sl, floor_sl)

                    if potential_sl > current_sl:
                        current_sl = potential_sl
                        self._db.update_trade_trailing_state(
                            trade_id=trade["id"],
                            highest_price=highest_price,
                            scaled_out=scaled_out,
                            stop_loss=current_sl
                        )

            # ── Check Exit Conditions ───────────────────────────────────────
            # Update trade object values for final exit check
            updated_trade = {
                **trade,
                "quantity": quantity,
                "stop_loss": current_sl,
                "take_profit": take_profit,
                "scaled_out": scaled_out,
            }

            result = self._risk.check_position_exits(
                trade=updated_trade,
                candle_high=candle_high,
                candle_low=candle_low,
                candle_open=candle_open,  # gap-through fill modeling
            )
            if result is not None:
                exit_reason, exit_price = result
                # Close remaining quantity
                self._close_position(updated_trade, exit_price, exit_reason)

    # ─── Portfolio State ───────────────────────────────────────────────────────

    def get_portfolio_state(self) -> PortfolioState:
        """Return a full snapshot of the current portfolio."""
        open_trades = self._db.get_open_trades()
        daily_pnl = self._db.get_daily_pnl()
        stats = self._db.get_stats()

        return PortfolioState(
            balance=float(self._balance),
            initial_balance=self.initial_balance,
            open_trades=open_trades,
            total_pnl=stats.get("total_pnl", 0.0),
            daily_pnl=daily_pnl,
            open_position_count=len(open_trades),
            equity=float(self._mark_to_market(open_trades)),
        )

    def force_close_position(self, pair: str, current_price: float) -> Optional[str]:
        """
        Emergency close: immediately close any open position for a pair.
        Called by Telegram /force_sell command.

        Returns:
            Summary string for Telegram reply, or None if no position found.
        """
        open_trades = self._db.get_open_trades()
        closed = []

        for trade in open_trades:
            if trade["symbol"].upper() == pair.upper():
                self._close_position(trade, current_price, "force_sell")
                closed.append(trade)

        if not closed:
            return f"No open position found for {pair}"

        return f"Force closed {len(closed)} position(s) for {pair} @ ${current_price:.2f}"

    def flatten_all(self, fetcher=None) -> list[str]:
        """Kill-switch flatten: close EVERY open position at market.

        Prices come from the injected fetcher; if a live price is
        unavailable the position is still closed (flattening takes
        priority over PnL accuracy) at its entry price, loudly flagged.
        """
        results: list[str] = []
        for trade in self._db.get_open_trades():
            pair = trade["symbol"]
            price = 0.0
            if fetcher is not None:
                try:
                    ticker = fetcher.fetch_ticker(pair)
                    price = float(ticker.get("last") or ticker.get("bid") or 0)
                except Exception as e:
                    logger.error(
                        "flatten_all: live price fetch failed for {}: {}", pair, e
                    )
            if price <= 0:
                price = float(trade["entry_price"])
                logger.critical(
                    "flatten_all: no live price for {} — closing at ENTRY price; "
                    "recorded PnL for this exit is unreliable", pair
                )
            self._close_position(trade, price, "kill_switch")
            results.append(f"{pair} {trade['side']} qty={trade['quantity']} @ ${price:,.2f}")
        return results

    @property
    def risk(self) -> RiskManager:
        """The risk manager (read-only) — drawdown checks, exits, breaker."""
        return self._risk

    @property
    def database(self) -> TradeLogger:
        """The trade logger (read-only) — audit trail and state store."""
        return self._db

    @property
    def balance(self) -> float:
        """Current available cash balance."""
        return float(self._balance)
