#!/usr/bin/env python3
"""paper_status.py — one-page, READ-ONLY snapshot of the running paper bot.

Reads only the SQLite state the bot writes; never mutates anything. Safe to
run anytime, including while the bot is live (WAL lets readers and the writer
coexist). Live prices are best-effort — without them, open positions are
marked at entry (unrealized = 0).

    python scripts/paper_status.py

Companion to scripts/health_check.py (which is a PRE-FLIGHT checker); this is
the DURING-RUN dashboard you glance at while babysitting the 30-day run.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from storage.trade_logger import TradeLogger
from risk.manager import SqliteCircuitBreakerStore
from risk import metrics


def _fmt_age(iso_ts: str | None) -> str:
    if not iso_ts:
        return "—"
    try:
        ts = datetime.fromisoformat(iso_ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - ts).total_seconds()
    except (ValueError, TypeError):
        return "—"
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f}m ago"
    return f"{secs / 3600:.1f}h ago"


def _live_prices(symbols: list[str]) -> dict[str, float]:
    """Best-effort live prices; empty dict if the exchange is unreachable."""
    prices: dict[str, float] = {}
    try:
        from data.fetcher import DataFetcher

        fetcher = DataFetcher()
        for sym in symbols:
            try:
                prices[sym] = float(fetcher.fetch_ticker(sym)["price"])
            except Exception:
                pass
    except Exception:
        pass
    return prices


def main() -> int:
    db = TradeLogger()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 64)
    print("  NeuronTrade — Paper Run Status")
    print(f"  {now}")
    print(f"  Strategy: {settings.strategy_name} | Timeframe: {settings.default_timeframe}")
    print(f"  Pairs: {', '.join(settings.trading_pairs)}")
    print("=" * 64)

    # ── Account & open positions ──────────────────────────────────────────────
    balance = db.load_balance()
    balance = float(balance) if balance is not None else settings.initial_balance
    open_trades = db.get_open_trades()
    prices = _live_prices([t["symbol"] for t in open_trades]) if open_trades else {}

    equity = balance
    open_heat = 0.0
    print(f"\n  Cash balance : ${balance:,.2f}")
    if open_trades:
        print(f"\n  Open positions ({len(open_trades)}):")
        print(f"    {'PAIR':<11}{'SIDE':<5}{'QTY':>12}{'ENTRY':>12}{'PRICE':>12}{'UNREAL':>12}")
        for t in open_trades:
            entry = float(t["entry_price"])
            qty = float(t["quantity"])
            px = prices.get(t["symbol"], entry)
            unreal = (px - entry) * qty if t["side"] == "buy" else (entry - px) * qty
            equity += entry * qty + unreal
            try:
                open_heat += abs(entry - float(t["stop_loss"])) * qty
            except (KeyError, TypeError):
                pass
            flag = "" if t["symbol"] in prices else "  (entry px — no live)"
            print(
                f"    {t['symbol']:<11}{t['side']:<5}{qty:>12.6f}{entry:>12.2f}"
                f"{px:>12.2f}{unreal:>+12.2f}{flag}"
            )
    else:
        print("\n  Open positions : none")

    ret_pct = (equity - settings.initial_balance) / settings.initial_balance * 100
    heat_pct = (open_heat / equity * 100) if equity > 0 else 0.0
    cap_pct = settings.portfolio_max_heat * 100
    print(f"\n  Marked equity: ${equity:,.2f}  ({ret_pct:+.2f}% vs start)")
    print(f"  Portfolio heat: {heat_pct:.1f}%  (cap {cap_pct:.0f}%)")

    # ── Risk / breaker state ──────────────────────────────────────────────────
    try:
        rec = SqliteCircuitBreakerStore(db.db_path).load()
        state = rec.state.value.upper()
        mark = "🔴" if state == "TRIPPED" else "🟢"
        print(f"\n  Circuit breaker: {mark} {state}")
        if rec.reason:
            print(f"    reason: {rec.reason}")
    except Exception as e:
        print(f"\n  Circuit breaker: (unreadable: {e})")
    print(f"  Consecutive losses: {db.get_consecutive_losses()}")
    print(f"  Today's PnL: ${db.get_daily_pnl():+,.2f}")

    # ── Performance (all closed trades) ───────────────────────────────────────
    history = db.get_trade_history(limit=10_000)
    print("\n  Performance (closed trades):")
    if history:
        pnls = [float(h["pnl"]) for h in history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        sharpe = metrics.sharpe_per_trade(pnls)
        wr = len(wins) / len(pnls) * 100 if pnls else 0.0
        print(f"    trades={len(pnls)}  win_rate={wr:.1f}%  PF={pf:.2f}  "
              f"sharpe/trade={sharpe:.2f}")
        print(f"    total PnL=${sum(pnls):+,.2f}  best=${max(pnls):+,.2f}  "
              f"worst=${min(pnls):+,.2f}")
        print("\n  Last 5 closed:")
        for h in history[:5]:
            print(f"    {h['symbol']:<11}{h['side']:<5} pnl=${float(h['pnl']):+9.2f}  "
                  f"{h['exit_reason']:<12} {_fmt_age(h.get('exit_time'))}")
    else:
        print("    no closed trades yet")

    # ── Latest signal per pair (freshness of the pipeline) ────────────────────
    print("\n  Latest signal per pair:")
    try:
        from dashboard import analytics

        sig = analytics.get_signal_log(limit=500)
        if not sig.empty:
            for pair in settings.trading_pairs:
                rows = sig[sig["symbol"] == pair]
                if rows.empty:
                    print(f"    {pair:<11} —")
                    continue
                last = rows.iloc[0]
                ts = last["timestamp"]
                age = _fmt_age(ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
                print(f"    {pair:<11}{str(last['decision']):<6} "
                      f"score={float(last['combined_score']):+.2f} "
                      f"conf={float(last['confidence']):.0%}  {age}")
        else:
            print("    no signals logged yet")
    except Exception as e:
        print(f"    (signal log unreadable: {e})")

    print("\n" + "=" * 64)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
