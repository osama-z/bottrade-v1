"""strategy_lab.py — screen every rule-based strategy across pairs and
timeframes on real production data, through the parity-verified engine.

    python scripts/strategy_lab.py                       # defaults below
    python scripts/strategy_lab.py --days 730 --timeframes 1h,4h
    python scripts/strategy_lab.py --strategies trend_following

Purpose: the AI-combined pipeline produced ~2 trades/year in walk-forward
— an untestable rate. Before adding intelligence, we need a baseline that
TRADES. This lab answers, with one command: which simple idea, on which
timeframe, generates a statistically meaningful sample, and does anything
beat sitting out?

Reading the table honestly:
- n_trades < ~30 → Sharpe is noise for that row, whatever its value.
- This screens ~30 combinations, so the single best cell is partly luck
  (multiple-comparisons bias). Treat the table as a FILTER; confirm any
  candidate on data it hasn't seen (later period / other pairs) before
  believing it.
- Rule-based strategies have no trained parameters, so there is no
  train/test leakage — but strategy SELECTION is itself a fit to this
  sample.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from backtesting.engine import BacktestEngine
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from strategies.registry import get_strategy

CACHE_DIR = project_root / "data" / "cache"

DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_TIMEFRAMES = ["1h", "4h"]
# ai_combined is excluded by default: it needs trained models per
# pair/timeframe and a walk-forward (its scores are fitted) — this lab is
# for rule-based baselines. Add it explicitly via --strategies if models
# exist for every combination you request.
DEFAULT_STRATEGIES = ["trend_following", "ma_crossover",
                      "rsi_reversal", "bollinger_bounce"]


def load_frame(fetcher: DataFetcher, pair: str, timeframe: str,
               days: int) -> pd.DataFrame:
    """Fetch (or reuse cached) OHLCV and compute the full indicator set."""
    safe = pair.replace("/", "_")
    cache = CACHE_DIR / f"{safe}_{timeframe}_{days}d.csv"
    if cache.exists():
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        logger.info("cache hit: {} ({} candles)", cache.name, len(raw))
    else:
        raw = fetcher.get_historical_data(pair=pair, timeframe=timeframe,
                                          days=days)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache)
    clean = DataPreprocessor().process(raw)
    return TechnicalIndicators().compute_all(clean)


def run_lab(pairs: list[str], timeframes: list[str], strategies: list[str],
            days: int) -> pd.DataFrame:
    fetcher = DataFetcher()
    rows = []
    total = len(pairs) * len(timeframes) * len(strategies)
    done = 0
    for pair in pairs:
        for timeframe in timeframes:
            try:
                df = load_frame(fetcher, pair, timeframe, days)
            except Exception as e:
                logger.error("data failed for {} {}: {}", pair, timeframe, e)
                continue
            for name in strategies:
                done += 1
                try:
                    strategy = get_strategy(name)
                    signals = strategy.generate_signals(df)
                    result = BacktestEngine(initial_capital=10_000.0).run(
                        df, signals, strategy_name=name, pair=pair,
                        timeframe=timeframe, verbose=False,
                    )
                    rows.append({
                        "strategy": name,
                        "pair": pair,
                        "tf": timeframe,
                        "n_trades": result.total_trades,
                        "return_pct": result.total_return_pct,
                        "bh_pct": result.buy_and_hold_return_pct,
                        "sharpe": result.sharpe_ratio,
                        "max_dd_pct": result.max_drawdown_pct,
                        "win_rate": result.win_rate_pct,
                        "profit_factor": result.profit_factor,
                    })
                    logger.info("[{}/{}] {} {} {} → {} trades, {:+.1f}%",
                                done, total, name, pair, timeframe,
                                result.total_trades, result.total_return_pct)
                except Exception as e:
                    logger.error("[{}/{}] {} {} {} failed: {}",
                                 done, total, name, pair, timeframe, e)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    args = parser.parse_args()

    report = run_lab(
        pairs=[p.strip() for p in args.pairs.split(",") if p.strip()],
        timeframes=[t.strip() for t in args.timeframes.split(",") if t.strip()],
        strategies=[s.strip() for s in args.strategies.split(",") if s.strip()],
        days=args.days,
    )
    if report.empty:
        print("No results — every combination failed; see log above.")
        return

    out = CACHE_DIR / "strategy_lab_results.csv"
    report.to_csv(out, index=False)

    ranked = report.sort_values("sharpe", ascending=False)
    print(f"\n══ Strategy lab — {args.days}d, ranked by Sharpe "
          f"(n_trades < 30 ⇒ Sharpe is noise) ══\n")
    print(ranked.to_string(index=False,
                           float_format=lambda x: f"{x:.2f}"))

    print("\n── Per-strategy aggregate (mean across pairs/timeframes) ──")
    agg = report.groupby("strategy").agg(
        combos=("pair", "count"),
        trades_total=("n_trades", "sum"),
        sharpe_mean=("sharpe", "mean"),
        return_mean=("return_pct", "mean"),
        bh_mean=("bh_pct", "mean"),
    ).sort_values("sharpe_mean", ascending=False)
    print(agg.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
