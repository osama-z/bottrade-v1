"""
scripts/walk_forward.py — Stage 5 Walk-Forward Validation

Rigorously tests the AI strategy by simulating real-world conditions:
1. Trains the ML and Regime models on N days of historical data.
2. Generates signals on the next M days of unseen (out-of-sample) data.
3. Rolls forward by M days, retrains the models, and repeats.
4. Stitches all out-of-sample predictions together.
5. Runs the integrated BacktestEngine to enforce circuit breakers and risk rules.

Usage:
    python scripts/walk_forward.py --pair BTC/USDT --train-days 180 --test-days 30 --total-days 365
"""

import argparse
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from strategies.ai_combined import AICombinedStrategy
from backtesting.engine import BacktestEngine


def run_walk_forward(
    pair: str,
    timeframe: str,
    train_days: int,
    test_days: int,
    total_days: int,
    capital: float,
) -> None:
    logger.info(
        "Starting Walk-Forward Validation for {} | Total History: {}d | Train Window: {}d | Test Window: {}d",
        pair, total_days, train_days, test_days
    )

    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    indicators = TechnicalIndicators()
    engine = BacktestEngine(initial_capital=capital)

    # 1. Fetch full contiguous dataset
    logger.info("Fetching {} days of historical data...", total_days)
    df = fetcher.get_historical_data(pair=pair, timeframe=timeframe, days=total_days)
    df = preprocessor.process(df)

    # 2. Compute all indicators continuously to avoid warm-up periods per window
    logger.info("Computing technical indicators across entire dataset...")
    df = indicators.compute_all(df)

    # 2b. Higher-timeframe frames for the macro-trend filter (parity: the
    # live decide() path gates BUYs on 1D/4H trend — the replay must too).
    # Extra history covers the EMA_50/Supertrend warm-up on the 1D frame.
    logger.info("Fetching 1D and 4H frames for the macro-trend filter...")
    df_1d = indicators.compute_all(preprocessor.process(
        fetcher.get_historical_data(pair=pair, timeframe="1d", days=total_days + 120)
    ))
    df_4h = indicators.compute_all(preprocessor.process(
        fetcher.get_historical_data(pair=pair, timeframe="4h", days=total_days + 30)
    ))

    # Drop NaNs created by long moving averages (e.g. 200 SMA)
    df = df.dropna(subset=["SMA_200", "ATR"])
    if len(df) < (train_days + test_days) * 24: # Roughly checking hours
        logger.error("Insufficient data left after dropping NaNs to form even one walk-forward window.")
        return

    start_date = df.index[0]
    end_date = df.index[-1]
    logger.info("Usable dataset ranges from {} to {}", start_date, end_date)

    # Strategy instance for generating signals
    strategy = AICombinedStrategy(pair=pair, use_llm=False)

    master_signals = pd.Series(dtype=int)
    master_test_df_pieces = []

    current_train_start = start_date
    window_idx = 1

    while True:
        current_train_end = current_train_start + timedelta(days=train_days)
        current_test_end = current_train_end + timedelta(days=test_days)

        if current_test_end > end_date:
            # If we don't have enough data for a full test window at the end, run what's left
            if current_train_end >= end_date:
                break
            current_test_end = end_date

        logger.info(
            "--- Window {} | Train: {} to {} | Test: {} to {} ---",
            window_idx,
            current_train_start.date(), current_train_end.date(),
            current_train_end.date(), current_test_end.date()
        )

        df_train = df.loc[current_train_start:current_train_end]
        df_test = df.loc[current_train_end:current_test_end].iloc[1:] # exclusive of overlap

        if len(df_train) < 500:
            logger.warning("Train window too small ({} candles). Stopping.", len(df_train))
            break
        if len(df_test) < 10:
            break

        # 3. Train models on this window
        logger.info("Training Regime Filter (HMM)...")
        try:
            strategy.train_regime(df_train)
        except Exception as e:
            logger.warning("HMM training failed for window {}: {}", window_idx, e)

        logger.info("Training ML Predictor (XGBoost)...")
        strategy.train_model(df_train)

        # 4. Generate signals on unseen test data via the SAME decide()
        # path live trading runs (macro trend, timing, volume, and HMM
        # regime gates all applied inside decide — no hand-replicated
        # filter approximations). Funding rate and order-book imbalance
        # have no stored history: they are parity exceptions (filters
        # pass through; decide() receives None).
        logger.info("Generating out-of-sample signals via decide() replay...")
        window_signals = strategy.generate_signals_via_decide(
            history_1h=df,               # full past context (train data is
                                         # legitimate history, not leakage)
            df_1d=df_1d,
            df_4h=df_4h,
            decide_index=df_test.index,
        )

        master_signals = pd.concat([master_signals, window_signals])
        master_test_df_pieces.append(df_test)

        # Shift forward
        current_train_start = current_train_start + timedelta(days=test_days)
        window_idx += 1

    if master_signals.empty:
        logger.error("No out-of-sample signals generated. Check date math.")
        return

    # 5. Run integrated BacktestEngine on the continuous out-of-sample signals
    df_full_test = pd.concat(master_test_df_pieces)

    # Ensure index alignment — use label-based alignment, not positional
    df_full_test = df_full_test[~df_full_test.index.duplicated(keep='first')]
    master_signals = master_signals[~master_signals.index.duplicated(keep='first')]

    # Re-align signals to match DataFrame index exactly (label-based safety)
    master_signals = master_signals.reindex(df_full_test.index, fill_value=0)

    logger.info("=== Walk-Forward Validation Complete ===")
    logger.info("Running integrated BacktestEngine on {} out-of-sample candles...", len(df_full_test))

    engine.run(
        df=df_full_test,
        signals=master_signals,
        strategy_name="WalkForward_AI_Combined",
        pair=pair,
        timeframe=timeframe,
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run out-of-sample walk-forward validation")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--train-days", type=int, default=180, help="Days per training window")
    parser.add_argument("--test-days", type=int, default=30, help="Days per testing window")
    parser.add_argument("--total-days", type=int, default=365, help="Total days of history to fetch")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Starting capital (USD)")
    args = parser.parse_args()

    run_walk_forward(
        pair=args.pair,
        timeframe=args.timeframe,
        train_days=args.train_days,
        test_days=args.test_days,
        total_days=args.total_days,
        capital=args.capital,
    )


if __name__ == "__main__":
    main()
