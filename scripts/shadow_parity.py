"""scripts/shadow_parity.py — close the Task 5.1 loop.

Reads the shadow_decisions table (live signals logged by SHADOW_MODE), recomputes
what the BACKTEST strategy would have decided on the exact same candles, aligns
them, and reports the signal-divergence percentage. Gate: divergence < 5% means
live and backtest agree (data parity); higher points to a live-vs-replay data bug.

The comparison core (signal_to_decision / signal_divergence) is pure so it is
unit-tested without a DB or network.

Usage:
    python scripts/shadow_parity.py [--limit 1000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

_DECISION = {1: "BUY", -1: "SELL", 0: "HOLD"}


def signal_to_decision(sig) -> str:
    """Map a backtest signal (+1/-1/0) to a decision label (BUY/SELL/HOLD)."""
    try:
        return _DECISION.get(int(sig), "HOLD")
    except (TypeError, ValueError):
        return "HOLD"


def signal_divergence(shadow: dict, backtest: dict) -> dict:
    """Compare two {key: decision} maps on their shared keys.

    Returns compared count, mismatch count, divergence percentage, and the
    mismatching (key, live, backtest) tuples. No shared keys → 0% (nothing to
    compare), reported via ``compared``.
    """
    keys = set(shadow) & set(backtest)
    n = len(keys)
    if n == 0:
        return {"compared": 0, "mismatches": 0, "divergence_pct": 0.0, "details": []}
    details = [(k, shadow[k], backtest[k]) for k in sorted(keys, key=str)
               if shadow[k] != backtest[k]]
    return {
        "compared": n,
        "mismatches": len(details),
        "divergence_pct": 100.0 * len(details) / n,
        "details": details,
    }


def backtest_decisions(strategy, df: pd.DataFrame) -> dict:
    """{candle-timestamp-iso: decision} from the strategy over an indicator frame."""
    signals = strategy.generate_signals(df)
    return {ts.isoformat(): signal_to_decision(v) for ts, v in signals.items()}


def _timeframe_delta(timeframe: str) -> "pd.Timedelta":
    n, unit = int(timeframe[:-1]), timeframe[-1]
    return {"m": pd.Timedelta(minutes=n), "h": pd.Timedelta(hours=n),
            "d": pd.Timedelta(days=n)}[unit]


def _align_to_candles(shadow_rows: list[dict], candle_ts_iso: list[str],
                      period: "pd.Timedelta") -> dict:
    """Bucket each shadow row to the candle it ACTUALLY decided on: the last
    CLOSED candle at log time — i.e. the latest candle whose CLOSE
    (open + period) is <= the shadow log time. The bot drops the still-forming
    candle, so aligning to `open <= log_time` would be one candle too late.
    Returns {(symbol, candle_iso): decision}."""
    ts_sorted = sorted(candle_ts_iso)
    out: dict = {}
    for row in shadow_rows:
        try:
            logged = pd.Timestamp(row["timestamp"])
        except Exception:
            continue
        candle = None
        for c in ts_sorted:
            if pd.Timestamp(c) + period <= logged:   # candle fully closed by log time
                candle = c
            else:
                break
        if candle is not None:
            out[(row["symbol"], candle)] = row["decision"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow-mode vs backtest signal parity (Task 5.1)")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--timeframe", default=None, help="filter shadow rows + backtest at this timeframe")
    ap.add_argument("--strategy", default=None, help="filter shadow rows + backtest with this strategy")
    args = ap.parse_args()

    from config.settings import settings
    from data.pipeline import build_indicator_frame
    from data.fetcher import DataFetcher
    from storage.factory import get_trade_logger
    from strategies.registry import get_strategy

    db = get_trade_logger()
    rows = db.get_shadow_decisions(limit=args.limit, strategy=args.strategy,
                                   timeframe=args.timeframe)
    if not rows:
        print("\nNo matching shadow decisions — run the bot in SHADOW_MODE (or the stress "
              "runner) first. Check your --strategy / --timeframe filters.\n")
        return 0

    # Prefer the tag on the rows; fall back to the flags / live settings.
    timeframe = args.timeframe or rows[0].get("timeframe") or settings.default_timeframe
    strategy_name = args.strategy or rows[0].get("strategy") or settings.strategy_name
    print(f"\nComparing {len(rows)} shadow decisions | strategy={strategy_name} | timeframe={timeframe}")
    fetcher = DataFetcher(public_only=True)
    shadow_map: dict = {}
    backtest_map: dict = {}

    for symbol in sorted({r["symbol"] for r in rows}):
        sym_rows = [r for r in rows if r["symbol"] == symbol]
        try:
            df = build_indicator_frame(fetcher, symbol, timeframe, limit=500)
        except Exception as e:
            logger.warning("skip {} — data fetch failed: {}", symbol, e)
            continue
        if df is None or df.empty:
            continue
        bt = backtest_decisions(get_strategy(strategy_name), df)
        candle_iso = list(bt.keys())
        aligned_shadow = _align_to_candles(sym_rows, candle_iso, _timeframe_delta(timeframe))
        shadow_map.update(aligned_shadow)
        for (sym, ts), _ in aligned_shadow.items():
            if ts in bt:
                backtest_map[(sym, ts)] = bt[ts]

    result = signal_divergence(shadow_map, backtest_map)
    print(f"\n{'='*60}\n  SHADOW-MODE PARITY (Task 5.1)\n{'='*60}")
    print(f"Compared candles : {result['compared']}")
    print(f"Signal mismatches: {result['mismatches']}")
    print(f"Divergence       : {result['divergence_pct']:.2f}%")
    if result["compared"] == 0:
        print("  (no overlapping candles — need more shadow history or fresher data)")
        return 0
    passed = result["divergence_pct"] < 5.0
    print(f"{'='*60}")
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} — divergence "
          f"{'<' if passed else '>='} 5% gate")
    if result["details"][:10]:
        print("  Sample mismatches (key, live, backtest):")
        for k, live, bt_dec in result["details"][:10]:
            print(f"    {k}: live={live} backtest={bt_dec}")
    print(f"{'='*60}\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
