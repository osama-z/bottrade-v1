"""Script to train the Hidden Markov Model (HMM) regime detector on historical data.

Usage:
    python scripts/train_regime.py --pair BTC/USDT --days 365 --timeframe 1h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from ai.regime_detector import RegimeDetector
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HMM Regime Detector")
    parser.add_argument("--pair", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument(
        "--days", type=int, default=365, help="Days of historical data to train on"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Candle timeframe"
    )
    args = parser.parse_args()

    logger.info(
        "Starting HMM training for {} | timeframe={} | days={}",
        args.pair,
        args.timeframe,
        args.days,
    )

    try:
        # 1. Fetch data
        fetcher = DataFetcher()
        df = fetcher.get_historical_data(
            pair=args.pair,
            timeframe=args.timeframe,
            days=args.days,
        )

        # 2. Preprocess data
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.process(df)

        # 3. Train regime detector
        detector = RegimeDetector(pair=args.pair)
        detector.fit(df_clean)

        # 4. Save model
        detector.save()
        logger.info("Successfully trained and saved HMM regime model for {}", args.pair)

        # Print some summary statistics
        if detector.model is not None:
            print("\n" + "=" * 60)
            print(f" HMM MODEL SUMMARY: {args.pair}")
            print("=" * 60)
            print(f"Model path: {detector.model_path}")
            print(f"Number of states: {detector.n_components}")
            print(f"Converged: {detector.model.monitor_.converged}")
            print(f"Iterations: {detector.model.monitor_.iter}")

            print("\nState Parameters (Sorted: 0=Bear, 1=Choppy, 2=Bull):")
            # Map sorted states back to original model states for printing
            inv_map = {v: k for k, v in detector._state_map.items()}
            for sorted_state in range(detector.n_components):
                orig = inv_map[sorted_state]
                mean_ret = detector.model.means_[orig, 0]
                mean_vol = detector.model.means_[orig, 1]
                cov_ret = detector.model.covars_[orig, 0, 0]
                cov_vol = detector.model.covars_[orig, 1, 1]
                state_name = detector._state_names[sorted_state]
                print(
                    f"State {sorted_state} ({state_name.upper()}):"
                )
                print(f"  Mean Log Return: {mean_ret:+.6f} | Volatility: {mean_vol:.6f}")
                print(f"  Var  Log Return: {cov_ret:.6f} | Volatility: {cov_vol:.6f}")

            print("\nTransition Matrix (Probability of state_t -> state_{t+1}):")
            trans_matrix = detector.model.transmat_
            # Print transition matrix using sorted state order
            header = "From / To".ljust(15) + " ".join(
                f"State {i}".rjust(10) for i in range(detector.n_components)
            )
            print(header)
            for i in range(detector.n_components):
                row_str = f"State {i}".ljust(15)
                for j in range(detector.n_components):
                    orig_i = inv_map[i]
                    orig_j = inv_map[j]
                    prob = trans_matrix[orig_i, orig_j]
                    row_str += f"{prob:.4f}".rjust(10)
                print(row_str)
            print("=" * 60 + "\n")

        return 0

    except Exception as e:
        logger.exception("Failed to train HMM model: {}", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
