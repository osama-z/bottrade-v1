"""
Script: Download historical OHLCV data and save to CSV.

Usage:
    python scripts/download_data.py --pair BTC/USDT --timeframe 1h --days 365
"""

import argparse
from pathlib import Path
from loguru import logger

from config.logging_config import setup_logging
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (e.g., 1h, 15m)")
    parser.add_argument("--days", type=int, default=365, help="Days of history to download")
    parser.add_argument("--output", default="data/historical", help="Output directory")
    args = parser.parse_args()

    logger.info(
        "Downloading {} days of {} data for {}",
        args.days, args.timeframe, args.pair
    )

    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()

    # Fetch raw data
    df = fetcher.get_historical_data(
        pair=args.pair,
        timeframe=args.timeframe,
        days=args.days,
    )

    # Clean and preprocess
    df = preprocessor.process(df)

    # Save to CSV
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{args.pair.replace('/', '_')}_{args.timeframe}_{args.days}d.csv"
    output_path = output_dir / filename
    df.to_csv(output_path)

    logger.info("Saved {} rows to {}", len(df), output_path)
    print(f"\n✅ Data saved to: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Period: {df.index[0]} → {df.index[-1]}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
