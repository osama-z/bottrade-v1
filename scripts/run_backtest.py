"""
Script: Run backtesting for a specific strategy on historical data.

Usage:
    python scripts/run_backtest.py --strategy ma_crossover --pair BTC/USDT --days 365
    python scripts/run_backtest.py --strategy rsi_reversal --pair ETH/USDT --timeframe 4h
    python scripts/run_backtest.py --compare  # Run all strategies and compare
"""

import argparse
from loguru import logger

from config.logging_config import setup_logging
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from backtesting.engine import BacktestEngine, BacktestResult
from strategies.registry import get_strategy, list_strategies


def run_single(
    strategy_name: str,
    pair: str,
    timeframe: str,
    days: int,
    capital: float,
) -> BacktestResult:
    """Run a single strategy backtest."""
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    indicators = TechnicalIndicators()
    engine = BacktestEngine(initial_capital=capital)

    # Fetch and prepare data
    logger.info("Fetching data for backtest...")
    df = fetcher.get_historical_data(pair=pair, timeframe=timeframe, days=days)
    df = preprocessor.process(df)
    df = indicators.compute_all(df)

    # Get strategy and generate signals
    strategy = get_strategy(strategy_name)
    signals = strategy.generate_signals(df)

    # Run backtest
    result = engine.run(
        df=df,
        signals=signals,
        strategy_name=strategy_name,
        pair=pair,
        timeframe=timeframe,
    )

    return result


def compare_all(pair: str, timeframe: str, days: int, capital: float) -> None:
    """Run all strategies and compare results."""
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    indicators = TechnicalIndicators()
    engine = BacktestEngine(initial_capital=capital)

    # Fetch data once, reuse for all strategies
    logger.info("Fetching data for comparison...")
    df = fetcher.get_historical_data(pair=pair, timeframe=timeframe, days=days)
    df = preprocessor.process(df)
    df = indicators.compute_all(df)

    results = []
    for strategy_name in list_strategies():
        try:
            strategy = get_strategy(strategy_name)
            signals = strategy.generate_signals(df)
            result = engine.run(
                df=df,
                signals=signals,
                strategy_name=strategy_name,
                pair=pair,
                timeframe=timeframe,
            )
            results.append(result)
        except Exception as e:
            logger.warning("Strategy '{}' failed: {}", strategy_name, e)

    # Print comparison table
    print("\n" + "═" * 70)
    print(f"  STRATEGY COMPARISON — {pair} {timeframe} ({days} days)")
    print("═" * 70)
    print(f"  {'Strategy':<22} {'Return':>8} {'Sharpe':>7} {'Drawdown':>10} {'WinRate':>8} {'Trades':>7}")
    print("─" * 70)
    for r in sorted(results, key=lambda x: x.total_return_pct, reverse=True):
        print(
            f"  {r.strategy_name:<22} "
            f"{r.total_return_pct:>7.1f}% "
            f"{r.sharpe_ratio:>7.2f} "
            f"{r.max_drawdown_pct:>9.1f}% "
            f"{r.win_rate_pct:>7.1f}% "
            f"{r.total_trades:>7}"
        )
    print("═" * 70)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run strategy backtesting")
    parser.add_argument("--strategy", default="ma_crossover", help="Strategy name")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital (USD)")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    args = parser.parse_args()

    if args.compare:
        compare_all(args.pair, args.timeframe, args.days, args.capital)
    else:
        run_single(args.strategy, args.pair, args.timeframe, args.days, args.capital)


if __name__ == "__main__":
    main()
