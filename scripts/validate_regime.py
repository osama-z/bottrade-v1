"""Script to perform out-of-sample validation on HMM regime detection.

Splits data 80/20 time-serially, trains on train portion, evaluates on out-of-sample
portion. Checks the three safety gates:
1. State ordering (Bullish > Neutral > Bearish)
2. State distribution (each state >= 5% of OOS data)
3. State persistence (average state duration >= 3 bars)

Usage:
    python scripts/validate_regime.py --pair BTC/USDT --days 365 --timeframe 1h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from ai.regime_detector import RegimeDetector
from data.fetcher import DataFetcher
from data.preprocessor import DataPreprocessor


def calculate_persistence(series: pd.Series) -> dict[int, float]:
    """Calculate the average duration (consecutive bars) for each state."""
    # Find groups of consecutive values
    change = series != series.shift(1)
    group_ids = change.cumsum()

    durations: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for _, group in series.groupby(group_ids):
        state = int(group.iloc[0])
        if state in durations:
            durations[state].append(len(group))

    avg_durations = {}
    for state, d_list in durations.items():
        avg_durations[state] = float(sum(d_list) / len(d_list)) if d_list else 0.0
    return avg_durations


def validate_regime(
    pair: str = "BTC/USDT",
    days: int = 365,
    timeframe: str = "1h",
) -> bool:
    """Run time-serial out-of-sample validation for the HMM model.

    Returns:
        True if all safety gates pass, False otherwise.
    """
    logger.info("Fetching {} days of {} for HMM validation...", days, pair)
    fetcher = DataFetcher()
    df = fetcher.get_historical_data(pair=pair, timeframe=timeframe, days=days)

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.process(df)

    # 1. Strict time-serial split (80% train, 20% test)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    logger.info(
        "Split data: train_len={}, test_len={} (OOS)", len(train_df), len(test_df)
    )

    # 2. Fit HMM on training data
    detector = RegimeDetector(pair=pair)
    detector.fit(train_df)

    # 3. Predict on OOS test data
    oos_states = detector.predict_series(test_df)

    # Calculate distributions & persistence
    counts = oos_states.value_counts(normalize=True).to_dict()
    persistence = calculate_persistence(oos_states)

    # Extract test-set stats per state
    features_df = detector.extract_features(test_df)
    features_df["state"] = oos_states

    # ─── GATES CHECK ───
    # Gate 1: Label ordering check (Bullish mean return > Neutral > Bearish)
    inv_map = {v: k for k, v in detector._state_map.items()}
    means = [
        detector.model.means_[inv_map[i], 0]
        for i in range(detector.n_components)
    ]
    gate1_passed = means[2] > means[1] > means[0]

    # Gate 2: Distribution check (each state >= 5% of OOS bars)
    gate2_passed = all(counts.get(i, 0.0) >= 0.05 for i in range(3))

    # Gate 3: State persistence check (average duration >= 3 bars)
    gate3_passed = all(persistence.get(i, 0.0) >= 3.0 for i in range(3))

    overall_passed = gate1_passed and gate2_passed and gate3_passed

    print("\n" + "=" * 60)
    print(f" OUT-OF-SAMPLE (OOS) HMM VALIDATION REPORT: {pair}")
    print("=" * 60)
    print(f"OOS period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Total OOS bars: {len(test_df)}")

    print("\n--- SAFETY GATES check ---")
    print(
        f"Gate 1: State Sorting (Bull > Neutral > Bear): "
        f"{'PASSED ✅' if gate1_passed else 'FAILED ❌'}"
    )
    print(
        f"  Bear (State 0) Mean Ret: {means[0]:+.6f}"
    )
    print(
        f"  Neut (State 1) Mean Ret: {means[1]:+.6f}"
    )
    print(
        f"  Bull (State 2) Mean Ret: {means[2]:+.6f}"
    )

    print(
        f"Gate 2: State Distribution (>= 5% each):       "
        f"{'PASSED ✅' if gate2_passed else 'FAILED ❌'}"
    )
    for i in range(3):
        name = detector._state_names[i]
        pct = counts.get(i, 0.0) * 100
        print(f"  State {i} ({name}): {pct:.2f}%")

    print(
        f"Gate 3: State Persistence (>= 3.0 bars):       "
        f"{'PASSED ✅' if gate3_passed else 'FAILED ❌'}"
    )
    for i in range(3):
        name = detector._state_names[i]
        dur = persistence.get(i, 0.0)
        print(f"  State {i} ({name}): {dur:.2f} bars avg duration")

    print("\n--- OOS Realized Returns by State ---")
    for i in range(3):
        state_df = features_df[features_df["state"] == i]
        name = detector._state_names[i]
        if not state_df.empty:
            realized_mean = state_df["log_return"].mean()
            realized_vol = state_df["log_range"].mean()
            print(
                f"  State {i} ({name}): mean_return={realized_mean:+.6f}, volatility={realized_vol:.6f}"
            )
        else:
            print(f"  State {i} ({name}): NO OBSERVATIONS")

    print("-" * 60)
    if overall_passed:
        print(" OVERALL VALIDATION STATUS: PASSED ✅")
    else:
        print(" OVERALL VALIDATION STATUS: FAILED ❌")
    print("=" * 60 + "\n")

    return overall_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HMM Out-Of-Sample Validation"
    )
    parser.add_argument("--pair", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument(
        "--days", type=int, default=365, help="Days of historical data"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Candle timeframe"
    )
    args = parser.parse_args()

    passed = validate_regime(
        pair=args.pair,
        days=args.days,
        timeframe=args.timeframe,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
