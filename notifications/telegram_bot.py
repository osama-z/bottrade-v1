"""
Telegram Bot — Two-way gateway for NeuronTrade remote control.

PUSH (bot → you):
    - New trade opened
    - Trade closed (win/loss)
    - Daily summary
    - System errors

PULL (you → bot):
    /start          — Welcome + show commands
    /status         — Portfolio state + today's PnL
    /trades         — Last 5 closed trades
    /pause          — Stop taking new trades
    /resume         — Resume trading
    /force_sell     — Emergency exit a pair
    /help           — Show all commands

Architecture:
    Runs in a background thread. Trade alerts are sent via
    send_*() methods (called from PaperTrader). Commands
    from the user are processed in the polling thread.

    All sends are fire-and-forget (non-blocking).
    A failed Telegram message NEVER halts trading.
"""

import threading
from typing import TYPE_CHECKING, Optional

from loguru import logger

try:
    import telegram
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed — Telegram disabled")

from config.settings import settings
from storage.trade_logger import TradeLogger

if TYPE_CHECKING:
    from execution.paper_trader import PaperTrader


class TelegramBot:
    """
    Async Telegram gateway with push alerts and command handling.

    Usage:
        bot = TelegramBot(trader=paper_trader, db=trade_logger)
        bot.start()              # Start background polling thread
        bot.send_trade_opened(…) # Push an alert
        bot.stop()               # Clean shutdown
    """

    def __init__(
        self,
        trader: Optional["PaperTrader"] = None,
        db: Optional[TradeLogger] = None,
        fetcher=None,
    ) -> None:
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self._trader = trader
        self._db = db or TradeLogger()
        self._fetcher = fetcher  # Optional DataFetcher for live exit prices

        self._app: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._enabled = bool(self.token and self.chat_id and TELEGRAM_AVAILABLE)

        if not self._enabled:
            logger.warning(
                "Telegram disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )
        else:
            logger.info("TelegramBot initialized — polling enabled")

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Telegram polling loop in a background daemon thread."""
        if not self._enabled:
            return

        self._thread = threading.Thread(
            target=self._run_polling,
            daemon=True,
            name="TelegramPoller",
        )
        self._thread.start()
        logger.info("Telegram polling started in background thread")

    def stop(self) -> None:
        """Stop the polling loop and wait for the thread to exit.

        The previous implementation called the coroutine ``app.stop()``
        without awaiting it (a no-op), swallowed the resulting warning,
        and logged success while the poller kept running.
        """
        if not self._enabled or self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            logger.warning("Telegram poller did not stop within 10s")
        else:
            logger.info("Telegram bot stopped")

    def _run_polling(self) -> None:
        """Internal: build and run the telegram Application."""
        import asyncio

        async def _main():
            app = (
                Application.builder()
                .token(self.token)
                .build()
            )
            self._app = app

            # Register command handlers
            app.add_handler(CommandHandler("start", self._cmd_start))
            app.add_handler(CommandHandler("help", self._cmd_help))
            app.add_handler(CommandHandler("status", self._cmd_status))
            app.add_handler(CommandHandler("trades", self._cmd_trades))
            app.add_handler(CommandHandler("pause", self._cmd_pause))
            app.add_handler(CommandHandler("resume", self._cmd_resume))
            app.add_handler(CommandHandler("force_sell", self._cmd_force_sell))

            logger.info("Telegram application built — starting polling")
            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)

            # Keep running until stop() sets the event
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)

            await app.updater.stop()
            await app.stop()
            await app.shutdown()

        # Reconnect loop: a TRANSIENT network error reaching api.telegram.org
        # (e.g. httpx ReadTimeout at startup) must not permanently kill the
        # poller — on an unattended run that would silently disable alerts and
        # remote control (/pause, /force_sell) for the rest of the session.
        # Retry with backoff until stop() is requested.
        while not self._stop_event.is_set():
            try:
                asyncio.run(_main())
                break  # clean shutdown — stop was requested inside _main()
            except Exception as e:
                logger.error(
                    "Telegram poller error ({}) — reconnecting in 30s", e
                )
                self._stop_event.wait(30)

    # ─── Command Handlers ──────────────────────────────────────────────────────

    def _is_authorized(self, update: "Update") -> bool:
        """Only the configured TELEGRAM_CHAT_ID may issue commands.

        Without this, ANY Telegram user who discovers the bot's username
        could pause the bot or force-close positions. Unauthorized commands
        are logged (with sender identity) and silently ignored — no reply,
        so the bot does not confirm its existence to strangers.
        """
        chat = update.effective_chat
        if chat is None or not self.chat_id:
            return False
        if str(chat.id) == str(self.chat_id):
            return True
        user = update.effective_user
        logger.warning(
            "Telegram UNAUTHORIZED command rejected | chat_id={} | user_id={} | text={!r}",
            chat.id,
            user.id if user else "?",
            update.message.text if update.message else "?",
        )
        return False

    async def _cmd_start(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text(
            "🧠 *NeuronTrade Bot Active*\n\n"
            "Your AI crypto trading bot is running.\n\n"
            "Commands:\n"
            "/status — Portfolio & PnL\n"
            "/trades — Recent trade history\n"
            "/pause — Stop new entries\n"
            "/resume — Resume trading\n"
            "/force\\_sell BTC/USDT — Emergency exit\n"
            "/help — All commands",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        await self._cmd_start(update, context)

    async def _cmd_status(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        if not self._is_authorized(update):
            return
        if not self._trader:
            await update.message.reply_text("⚠️ Trader not connected.")
            return

        state = self._trader.get_portfolio_state()
        open_trades = state.open_trades

        positions_text = ""
        for t in open_trades:
            positions_text += (
                f"\n  • {t['symbol']} {t['side'].upper()} "
                f"| Entry=${t['entry_price']:.2f} "
                f"| SL=${t.get('stop_loss', 0):.2f}"
            )

        status_icon = "⏸️" if self._trader.is_paused else "🟢"
        msg = (
            f"{status_icon} *NeuronTrade Status*\n"
            f"{'─' * 30}\n"
            f"💰 Balance: `${state.balance:,.2f}`\n"
            f"📊 Equity:  `${state.equity:,.2f}`\n"
            f"📈 Total PnL: `${state.total_pnl:+,.2f}`\n"
            f"📅 Daily PnL: `${state.daily_pnl:+,.2f}`\n"
            f"📉 Return: `{state.total_return_pct:+.2f}%`\n"
            f"{'─' * 30}\n"
            f"🔓 Open Positions: {state.open_position_count}/{settings.max_open_positions}\n"
            f"{positions_text if positions_text else '  None'}\n"
            f"{'─' * 30}\n"
            f"Status: {'PAUSED ⏸️' if self._trader.is_paused else 'RUNNING ✅'}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_trades(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        if not self._is_authorized(update):
            return
        trades = self._db.get_trade_history(limit=5)
        if not trades:
            await update.message.reply_text("No closed trades yet.")
            return

        lines = ["📋 *Last 5 Trades*\n"]
        for t in trades:
            emoji = "✅" if (t.get("pnl") or 0) >= 0 else "❌"
            lines.append(
                f"{emoji} {t['symbol']} {t['side'].upper()} "
                f"| PnL: `${t.get('pnl', 0):+.2f}` "
                f"| {t.get('exit_reason', '—')}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_pause(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        if not self._is_authorized(update):
            return
        if self._trader:
            self._trader.is_paused = True
            logger.warning("Bot PAUSED by Telegram command")
            self._db.log_risk_event("pause", reason="Telegram /pause")
            await update.message.reply_text(
                "⏸️ *Bot Paused*\n\nNew trades are suspended. "
                "Existing positions are still monitored.\n"
                "Send /resume to restart.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("⚠️ Trader not connected.")

    async def _cmd_resume(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        if not self._is_authorized(update):
            return
        if self._trader:
            self._trader.is_paused = False
            logger.info("Bot RESUMED by Telegram command")
            self._db.log_risk_event("resume", reason="Telegram /resume")
            await update.message.reply_text(
                "✅ *Bot Resumed*\n\nTrading is active again.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("⚠️ Trader not connected.")

    async def _cmd_force_sell(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        if not self._is_authorized(update):
            return
        if not self._trader:
            await update.message.reply_text("⚠️ Trader not connected.")
            return

        args = context.args
        if not args:
            await update.message.reply_text(
                "⚠️ Usage: `/force_sell BTC/USDT`", parse_mode="Markdown"
            )
            return

        pair = args[0].upper()
        open_trades = self._db.get_open_trades()
        if not any(t["symbol"] == pair for t in open_trades):
            await update.message.reply_text(f"No open position found for {pair}")
            return

        # Close at the LIVE market price. Falling back to the entry price
        # records PnL ≈ 0 for the emergency exit — exactly when price has
        # moved against us — so the fallback is loudly flagged.
        current_price = 0.0
        if self._fetcher is not None:
            try:
                ticker = self._fetcher.fetch_ticker(pair)
                current_price = float(ticker.get("last") or ticker.get("bid") or 0)
            except Exception as e:
                logger.error("force_sell: live price fetch failed for {}: {}", pair, e)
        if current_price <= 0:
            current_price = float(next(
                t["entry_price"] for t in open_trades if t["symbol"] == pair
            ))
            logger.critical(
                "force_sell: no live price for {} — closing at ENTRY price; "
                "recorded PnL for this exit is unreliable", pair
            )

        result = self._trader.force_close_position(pair, current_price)
        logger.warning("Force sell executed by Telegram: {}", result)
        await update.message.reply_text(
            f"🚨 *Force Sell Executed*\n{result}", parse_mode="Markdown"
        )

    # ─── Push Notifications ────────────────────────────────────────────────────

    def send_trade_opened(
        self,
        pair: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        ai_score: float,
        ai_confidence: float,
        reasoning: str,
        trade_id: int,
    ) -> None:
        """Push: new trade opened alert."""
        direction = "📈 LONG" if side == "buy" else "📉 SHORT"
        msg = (
            f"🟢 *New Trade Opened* #{trade_id}\n"
            f"{'─' * 28}\n"
            f"{direction} `{pair}`\n"
            f"Entry:  `${price:,.2f}`\n"
            f"Size:   `{quantity:.6f}`\n"
            f"SL:     `${stop_loss:,.2f}`\n"
            f"TP:     `${take_profit:,.2f}`\n"
            f"{'─' * 28}\n"
            f"🧠 AI Score: `{ai_score:+.2f}` | Confidence: `{ai_confidence:.0%}`\n"
            f"💬 _{reasoning}_"
        )
        self._send(msg)

    def send_shadow_signal(
        self,
        pair: str,
        decision: str,
        price: float,
        expected_slippage: float,
        expected_fill: float,
        confidence: float,
        reason: str = "",
    ) -> None:
        """Push: a SHADOW-MODE signal — decision logged, NO order placed (Task 5.1).

        Clearly distinct from a real fill so shadow runs aren't mistaken for
        executions; includes the expected entry price + modelled slippage and
        confirms the row was written to shadow_decisions.
        """
        direction = "📈 BUY" if decision == "BUY" else "📉 SELL"
        msg = (
            f"🔍 *SHADOW SIGNAL* — no order placed\n"
            f"{'─' * 28}\n"
            f"{direction} `{pair}`\n"
            f"Signal price:   `${price:,.2f}`\n"
            f"Expected fill:  `${expected_fill:,.2f}`  (slippage `{expected_slippage:.4%}`)\n"
            f"Confidence:     `{confidence:.0%}`\n"
            f"{'─' * 28}\n"
            + (f"💬 _{reason}_\n" if reason else "")
            + "📝 Logged to `shadow_decisions` (backtest-parity validation)"
        )
        self._send(msg)

    def send_trade_closed(
        self,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        trade_id: int,
    ) -> None:
        """Push: trade closed alert (win or loss)."""
        emoji = "✅ WIN" if pnl >= 0 else "❌ LOSS"
        return_pct = (exit_price - entry_price) / entry_price * 100
        if side == "sell":
            return_pct = -return_pct

        reason_map = {
            "stop_loss": "🛑 Stop Loss",
            "take_profit": "🎯 Take Profit",
            "signal": "📊 Strategy Signal",
            "force_sell": "🚨 Force Sell",
        }
        reason_label = reason_map.get(exit_reason, exit_reason)

        msg = (
            f"{emoji} *Trade Closed* #{trade_id}\n"
            f"{'─' * 28}\n"
            f"Pair:   `{pair}`\n"
            f"Entry:  `${entry_price:,.2f}`\n"
            f"Exit:   `${exit_price:,.2f}`\n"
            f"Return: `{return_pct:+.2f}%`\n"
            f"PnL:    `${pnl:+.2f}`\n"
            f"{'─' * 28}\n"
            f"Reason: {reason_label}"
        )
        self._send(msg)

    def send_daily_summary(self, state, stats: dict) -> None:
        """Push: end-of-day summary."""
        win_rate = stats.get("win_rate", 0)
        total = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        daily_pnl = state.daily_pnl

        emoji = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{emoji} *Daily Summary*\n"
            f"{'─' * 28}\n"
            f"💰 Balance: `${state.balance:,.2f}`\n"
            f"📊 Today's PnL: `${daily_pnl:+.2f}`\n"
            f"{'─' * 28}\n"
            f"📋 Trades Today:\n"
            f"  Total:  {total}\n"
            f"  Wins:   {wins} ✅\n"
            f"  Losses: {losses} ❌\n"
            f"  Win Rate: {win_rate:.1f}%"
        )
        self._send(msg)

    def send_error(self, message: str) -> None:
        """Push: system error alert."""
        msg = f"⚠️ *System Error*\n\n`{message[:500]}`"
        self._send(msg)

    def send_system_start(self, balance: float, pairs: list[str],
                          mode: str | None = None) -> None:
        """Push: bot startup notification. ``mode`` reflects the actual run mode
        (shadow / paper / live) so the alert never mislabels a shadow run."""
        if mode is None:
            from config.settings import settings
            mode = ("🔍 SHADOW (log-only, no orders)" if settings.shadow_mode
                    else ("PAPER TRADING" if settings.paper_trading else "⚠️ LIVE TRADING"))
        msg = (
            f"🚀 *NeuronTrade Started*\n"
            f"{'─' * 28}\n"
            f"Mode: `{mode}`\n"
            f"Balance: `${balance:,.2f}`\n"
            f"Pairs: `{', '.join(pairs)}`\n"
            f"{'─' * 28}\n"
            f"Send /help for commands."
        )
        self._send(msg)

    def send_drawdown_warning(self, drawdown_pct: float) -> None:
        """Push: max drawdown warning."""
        msg = (
            f"🚨 *DRAWDOWN ALERT*\n\n"
            f"Current drawdown: `{drawdown_pct:.1%}`\n"
            f"Bot has been automatically paused.\n"
            f"Send /resume when ready."
        )
        self._send(msg)

    # ─── Internal Send ─────────────────────────────────────────────────────────

    def _send(self, message: str) -> None:
        """
        Fire-and-forget message send. Never raises. Never blocks trading.
        """
        if not self._enabled:
            logger.debug("Telegram disabled — message not sent: {}", message[:80])
            return

        thread = threading.Thread(
            target=self._send_sync,
            args=(message,),
            daemon=True,
        )
        thread.start()

    def _send_sync(self, message: str) -> None:
        """Synchronous send executed in a background thread."""
        import asyncio
        try:
            bot = telegram.Bot(token=self.token)
            asyncio.run(
                bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="Markdown",
                )
            )
        except Exception as e:
            # Markdown failures are content-dependent (underscores/asterisks
            # in error text reliably break parse_mode="Markdown") — exactly
            # the alerts that matter most. Retry once as plain text before
            # giving up; NEVER let Telegram errors crash the trading system.
            try:
                bot = telegram.Bot(token=self.token)
                asyncio.run(
                    bot.send_message(chat_id=self.chat_id, text=message)
                )
                logger.warning(
                    "Telegram Markdown send failed ({}); delivered as plain text", e
                )
            except Exception as e2:
                logger.error(
                    "Telegram send FAILED (Markdown and plain text): {}", e2
                )
