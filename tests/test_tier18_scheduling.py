"""Tier 18 — live candle-close timing (backtest/live parity).

The backtest acts on CLOSED candles; live must too. These pin the two
helpers that make that true: aligned scheduling and forming-candle drop.
"""

import pandas as pd
import pytest

from scripts.run_live import candle_close_cron
from core.candles import drop_forming_candle, seconds_to_next_close


def _frame(last_open: pd.Timestamp, timeframe_seconds: int, rows: int = 5) -> pd.DataFrame:
    idx = pd.date_range(
        end=last_open, periods=rows, freq=pd.Timedelta(seconds=timeframe_seconds)
    )
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=idx,
    )


class TestDropFormingCandle:
    def test_forming_last_candle_is_dropped(self):
        now = pd.Timestamp.now(tz="UTC")
        # Last candle opened 1h ago on a 4h timeframe → closes 3h from now → forming.
        df = _frame(now - pd.Timedelta(hours=1), timeframe_seconds=14400)
        out = drop_forming_candle(df, "4h")
        assert len(out) == len(df) - 1
        assert out.index[-1] == df.index[-2]

    def test_closed_last_candle_is_kept(self):
        now = pd.Timestamp.now(tz="UTC")
        # Last candle opened 5h ago on a 4h timeframe → closed 1h ago → keep it.
        df = _frame(now - pd.Timedelta(hours=5), timeframe_seconds=14400)
        out = drop_forming_candle(df, "4h")
        assert len(out) == len(df)
        assert out.index[-1] == df.index[-1]

    def test_empty_frame_is_returned_unchanged(self):
        empty = pd.DataFrame()
        assert drop_forming_candle(empty, "4h") is empty


class TestCandleCloseCron:
    @pytest.mark.parametrize("tf", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"])
    def test_supported_timeframes_build_a_trigger(self, tf):
        trig = candle_close_cron(tf)
        # Fires a few seconds after the boundary so the candle is final.
        assert "second='5'" in str(trig)

    def test_unsupported_timeframe_unit_raises(self):
        # 'M' (month) is not a scheduling unit this bot supports.
        with pytest.raises(ValueError):
            candle_close_cron("1M")


class TestSecondsToNextClose:
    @pytest.mark.parametrize("tf,period", [("1m", 60), ("1h", 3600), ("4h", 14400)])
    def test_within_one_period_and_positive(self, tf, period):
        # Next boundary is always ahead, and no further than one period away
        # (plus the small finalise offset).
        secs = seconds_to_next_close(tf, offset_seconds=5)
        assert 0 < secs <= period + 5

    def test_boundary_is_epoch_aligned(self):
        import time

        secs = seconds_to_next_close("4h", offset_seconds=0)
        boundary = time.time() + secs
        # 4h boundaries align to the Unix epoch (== 00/04/08... UTC).
        assert round(boundary) % 14400 == 0
