"""
Model training script — downloads data, engineers features, trains XGBoost, saves model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --pair ETH/USDT --timeframe 1h --days 180

This script must be run BEFORE running backtests with the ai_combined strategy.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.logging_config import setup_logging

setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NeuronTrade XGBoost price predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py
  python scripts/train_model.py --pair ETH/USDT --days 365
  python scripts/train_model.py --pair BTC/USDT --timeframe 4h --days 730
        """
    )
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    parser.add_argument("--target-periods", type=int, default=1, help="Candles ahead to predict (default: 1)")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split fraction (default: 0.20)")
    parser.add_argument("--no-download", action="store_true", help="Skip data download, use cached CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NeuronTrade — XGBoost Model Training")
    logger.info("  Pair:          {}", args.pair)
    logger.info("  Timeframe:     {}", args.timeframe)
    logger.info("  History:       {} days", args.days)
    logger.info("  Target:        {} period(s) ahead", args.target_periods)
    logger.info("=" * 60)

    # ── Step 1: Load or download data ─────────────────────────────────────
    import pandas as pd

    safe_pair = args.pair.replace("/", "_")
    csv_path = Path(f"data/cache/{safe_pair}_{args.timeframe}_{args.days}d.csv")

    if not args.no_download or not csv_path.exists():
        logger.info("Downloading {} days of {} {} data...", args.days, args.pair, args.timeframe)
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            # get_historical_data PAGINATES. fetch_ohlcv is a single request
            # capped at 1000 candles by the exchange, so `limit=days*24`
            # silently yielded ~41 days for any --days above ~41 — every
            # model was trained on 1000 candles regardless of the flag.
            df_raw = fetcher.get_historical_data(
                pair=args.pair,
                timeframe=args.timeframe,
                days=args.days,
            )
            # Save to cache
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(csv_path)
            logger.info("Data saved to {}", csv_path)
        except Exception as e:
            logger.warning("Download failed: {} — trying to load cached CSV", e)
            if csv_path.exists():
                df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            else:
                logger.error("No cached data and download failed. Exiting.")
                sys.exit(1)
    else:
        logger.info("Loading cached data from {}", csv_path)
        df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    logger.info("Loaded {} candles", len(df_raw))

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    logger.info("Preprocessing data...")
    from data.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.process(df_raw)
    logger.info("After preprocessing: {} rows", len(df_clean))

    # ── Step 3: Calculate indicators ──────────────────────────────────────
    logger.info("Computing technical indicators...")
    from indicators.technical import TechnicalIndicators
    ti = TechnicalIndicators()
    df_indicators = ti.compute_all(df_clean)
    logger.info("After indicators: {} rows", len(df_indicators))

    # ── Step 4: Feature engineering ───────────────────────────────────────
    logger.info("Engineering ML features (target = {} periods ahead)...", args.target_periods)
    from indicators.features import FeatureEngineer
    fe = FeatureEngineer()
    X, y = fe.build_features(df_indicators, target_periods=args.target_periods)

    logger.info("Feature matrix: {} samples × {} features", len(X), len(X.columns))
    logger.info("Target distribution: {:.1%} UP, {:.1%} DOWN",
                y.mean(), 1 - y.mean())

    # ── Step 5: Train model ───────────────────────────────────────────────
    logger.info("Training XGBoost classifier...")
    from ai.ml_predictor import MLPredictor
    predictor = MLPredictor(pair=args.pair)
    result = predictor.train(X, y, test_size=args.test_size)

    # ── Step 6: Print results ─────────────────────────────────────────────
    result.print_report()

    # ── Step 7: Feature importance ────────────────────────────────────────
    importance = predictor.get_feature_importance(top_n=10)
    if importance:
        print("\n📊 Top 10 Most Important Features:")
        for i, (feat, imp) in enumerate(importance.items(), 1):
            bar = "█" * int(imp * 200)
            print(f"  {i:2}. {feat:<30} {imp:.4f}  {bar}")

    logger.info("✅ Training complete! Model saved to ai/models/")
    logger.info("You can now run: python scripts/run_backtest.py --strategy ai_combined")


if __name__ == "__main__":
    main()
