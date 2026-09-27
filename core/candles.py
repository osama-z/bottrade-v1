"""Shared candle-timing helpers for the live entrypoints.

Backtest/live parity: the backtest treats every candle as CLOSED and acts at
its close. Both live entrypoints (scripts/run_live.py and
scripts/run_decoupled_intelligence.py) must do the same, or paper results are
not comparable to the validated backtest. Keeping these here means both use
the identical rule instead of one drifting from the other.
"""

from __future__ import annotations

import time

import pandas as pd

from config.constants import TIMEFRAME_SECONDS


def drop_forming_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Return ``df`` without a still-forming last candle.

    ccxt/Binance return the CURRENT (incomplete) candle as the last row. Acting
    on it means deciding on partial data that keeps changing as the candle
    forms — and it diverges from the backtest, which treats every candle as
    closed. A candle opened at T for period P is closed once ``T + P <= now``;
    if the last row is not closed yet, drop it. Robust regardless of when the
    caller happened to run.
    """
    if df is None or len(df) == 0:
        return df
    period = pd.Timedelta(seconds=TIMEFRAME_SECONDS.get(timeframe, 3600))
    if df.index[-1] + period > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1]
    return df


def seconds_to_next_close(timeframe: str, offset_seconds: int = 5) -> float:
    """Seconds from now until just after the next candle-close boundary.

    Candle boundaries for minute/hour/day timeframes align to the Unix epoch,
    which coincides with UTC boundaries (e.g. 4h → 00:00/04:00/08:00 UTC). Used
    by the decoupled intelligence node's sleep loop to align passes to candle
    close instead of drifting from process-start time. ``offset_seconds`` lets
    the just-closed candle finalise on the exchange before we fetch.

    (Weekly alignment is handled by CronTrigger in run_live, not here — the
    epoch's Thursday origin doesn't match Binance's Monday weekly open.)
    """
    period = TIMEFRAME_SECONDS.get(timeframe, 3600)
    now = time.time()
    next_boundary = (now // period + 1) * period + offset_seconds
    return next_boundary - now
