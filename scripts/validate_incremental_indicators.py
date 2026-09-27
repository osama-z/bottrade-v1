"""Validate Stage 1 incremental indicators against pandas-ta.

Usage:
    .venv/bin/python scripts/validate_incremental_indicators.py

The script generates a deterministic 6,000-candle OHLCV series, computes the
batch pandas-ta EMA/RSI/ATR references, streams the same candles through the
incremental indicators, and reports max absolute deviation after warmup.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pandas_ta as ta

from config.constants import DEFAULT_ATR_PERIOD, DEFAULT_EMA_FAST, DEFAULT_RSI_PERIOD
from indicators.incremental import IncrementalATR, IncrementalEMA, IncrementalRSI


@dataclass(frozen=True)
class ValidationResult:
    name: str
    max_deviation: float
    compared_points: int


def generate_ohlcv(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.00005, scale=0.004, size=rows)
    close = 50_000.0 * np.cumprod(1.0 + returns)
    spread = rng.uniform(0.0005, 0.006, size=rows)
    high = close * (1.0 + spread)
    low = close * (1.0 - spread * rng.uniform(0.6, 1.2, size=rows))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(100, 5_000, size=rows)
    index = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def max_abs_deviation(reference: pd.Series, observed: list[float | None]) -> tuple[float, int]:
    observed_series = pd.Series(observed, index=reference.index, dtype="float64")
    aligned = pd.concat([reference, observed_series], axis=1).dropna()
    aligned.columns = ["reference", "observed"]
    if aligned.empty:
        return float("nan"), 0
    return float((aligned["reference"] - aligned["observed"]).abs().max()), len(aligned)


def validate(rows: int = 6_000, resync_interval: int = 1_000) -> list[ValidationResult]:
    if rows < 5_000:
        raise ValueError("Stage 1 validation requires at least 5,000 ticks")

    df = generate_ohlcv(rows)

    ema_ref = ta.ema(df["close"], length=DEFAULT_EMA_FAST)
    rsi_ref = ta.rsi(df["close"], length=DEFAULT_RSI_PERIOD)
    atr_ref = ta.atr(df["high"], df["low"], df["close"], length=DEFAULT_ATR_PERIOD)

    ema = IncrementalEMA(period=DEFAULT_EMA_FAST, resync_interval=resync_interval,
                         window_size=resync_interval if resync_interval > 0 else None)
    rsi = IncrementalRSI(period=DEFAULT_RSI_PERIOD, resync_interval=resync_interval,
                         window_size=resync_interval if resync_interval > 0 else None)
    atr = IncrementalATR(period=DEFAULT_ATR_PERIOD, resync_interval=resync_interval,
                         window_size=resync_interval if resync_interval > 0 else None)

    ema_values: list[float | None] = []
    rsi_values: list[float | None] = []
    atr_values: list[float | None] = []
    for row in df.itertuples(index=False):
        ema_values.append(ema.update(float(row.close)))
        rsi_values.append(rsi.update(float(row.close)))
        atr_values.append(atr.update(float(row.high), float(row.low), float(row.close)))

    results = []
    for name, reference, values in [
        ("EMA", ema_ref, ema_values),
        ("RSI", rsi_ref, rsi_values),
        ("ATR", atr_ref, atr_values),
    ]:
        deviation, compared = max_abs_deviation(reference, values)
        results.append(ValidationResult(name, deviation, compared))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6_000)
    parser.add_argument("--resync-interval", type=int, default=1_000)
    args = parser.parse_args()

    print("Stage 1 incremental indicator drift validation")
    print(f"ticks={args.rows} resync_interval={args.resync_interval}")
    for result in validate(args.rows, args.resync_interval):
        print(
            f"{result.name}: max_deviation={result.max_deviation:.12f} "
            f"compared_points={result.compared_points}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
