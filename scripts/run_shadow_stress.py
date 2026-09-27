#!/usr/bin/env python3
"""scripts/run_shadow_stress.py — short SHADOW-MODE stress test.

Exercises the BUY/SELL parity path: overrides config to a FAST timeframe + an
ACTIVE strategy (rsi_reversal fires far more than trend_following) across many
pairs, so it generates real BUY/SELL signals quickly. Every decision is logged
to the SAME SQLite DB, TAGGED with strategy+timeframe so shadow_parity can filter
it apart from the 4h run. No orders, no Telegram, no kill switch — so it runs
cleanly alongside (or instead of) the normal bot.

    python scripts/run_shadow_stress.py                     # 5m / rsi_reversal, run until Ctrl-C
    python scripts/run_shadow_stress.py --minutes 30        # auto-stop after 30 min

Afterwards:
    python scripts/shadow_parity.py --strategy rsi_reversal --timeframe 5m
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_DEFAULT_PAIRS = ("BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,"
                  "ADA/USDT,LINK/USDT,DOGE/USDT,AVAX/USDT,DOT/USDT")


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow-mode stress test (Task 5.1 BUY/SELL parity)")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--strategy", default="rsi_reversal")
    ap.add_argument("--pairs", default=_DEFAULT_PAIRS)
    ap.add_argument("--minutes", type=float, default=0.0, help="auto-stop after N minutes (0 = until killed)")
    args = ap.parse_args()

    # Override config BEFORE any settings import (env vars beat .env).
    os.environ["SHADOW_MODE"] = "true"
    os.environ["STRATEGY"] = args.strategy
    os.environ["DEFAULT_TIMEFRAME"] = args.timeframe

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from config.logging_config import setup_logging
    setup_logging()

    from loguru import logger
    from backtesting.costs import CostModel
    from core.candles import seconds_to_next_close
    from data.fetcher import DataFetcher
    from data.pipeline import build_indicator_frame
    from execution.shadow import build_shadow_record
    from storage.trade_logger import TradeLogger
    from strategies.registry import get_strategy

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    tf, strat_name = args.timeframe, args.strategy
    fetcher = DataFetcher(public_only=True)     # public market data only — no keys
    db = TradeLogger()                          # same default DB (storage/neurontrade.db)
    cost = CostModel()
    strategies = {p: get_strategy(strat_name) for p in pairs}

    logger.warning(
        "🔬 SHADOW STRESS TEST — {} @ {} | {} pairs | shadow-only (NO orders/Telegram)",
        strat_name, tf, len(pairs),
    )

    def one_pass() -> int:
        actionable = 0
        for pair in pairs:
            try:
                df = build_indicator_frame(fetcher, pair, tf, limit=500)  # drops forming candle
                if df is None or len(df) < 60:
                    continue
                sig = strategies[pair].get_signal(df, pair)
                price = float(df["close"].iloc[-1])
                vol = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0
                rec = build_shadow_record(sig.signal, price, cost, candle_volume=vol)
                db.log_shadow_decision(
                    symbol=pair, decision=rec.decision, price=rec.price,
                    expected_slippage=rec.expected_slippage, expected_fill=rec.expected_fill,
                    confidence=sig.confidence, reason=sig.reason,
                    strategy=strat_name, timeframe=tf,
                )
                if rec.decision in ("BUY", "SELL"):
                    actionable += 1
                logger.info("SHADOW[{}] {:10} {:4} @ {:.6g}", tf, pair, rec.decision, price)
            except Exception as e:
                logger.warning("{}: {}", pair, e)
        return actionable

    deadline = (time.time() + args.minutes * 60) if args.minutes > 0 else None
    total = one_pass()                          # immediate pass so you see signals now
    try:
        while deadline is None or time.time() < deadline:
            wait = seconds_to_next_close(tf)
            if deadline is not None:
                wait = min(wait, max(1.0, deadline - time.time()))
            time.sleep(max(1.0, wait))
            if deadline is not None and time.time() >= deadline:
                break
            total += one_pass()
    except KeyboardInterrupt:
        logger.info("stress test interrupted")
    logger.warning("🔬 Stress test done — actionable BUY/SELL signals logged: {}", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
