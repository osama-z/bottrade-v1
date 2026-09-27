"""MarketContext — the replayable input to AICombinedStrategy.decide().

Backtest/live parity (claude.md, docs/backtest_live_parity_design.md):
the decision function must be a pure function of its inputs so the same
code path runs live and in walk-forward validation. Live, a context is
built from fetchers; in replay, from pre-downloaded frames sliced as-of
each candle with no lookahead.

Fields that cannot be replayed (funding rate, order-book imbalance — no
history is stored) are ``None`` in replay contexts. ``decide()`` treats
None as "filter unavailable → pass-through", and backtest reports must
list them as parity exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

_PERIODS: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


@dataclass(frozen=True)
class MarketContext:
    """Everything decide() may look at. No fetchers, no filesystem."""

    df: pd.DataFrame                       # 1h OHLCV + indicators (windowed)
    df_1d: Optional[pd.DataFrame] = None   # daily frame, indicator-computed
    df_4h: Optional[pd.DataFrame] = None   # 4h frame, indicator-computed
    funding_rate: Optional[float] = None   # None = unavailable (parity exception)
    imbalance: Optional[float] = None      # None = unavailable (parity exception)
    news_headlines: tuple[str, ...] = ()


def closed_candles_as_of(
    frame: Optional[pd.DataFrame], period: str, as_of_close: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """Only candles fully CLOSED by ``as_of_close``.

    A candle indexed at its open time T (period P) is closed once
    T + P <= as_of_close. Including a still-forming higher-timeframe
    candle would leak future information into the macro-trend filter.
    """
    if frame is None:
        return None
    return frame.loc[frame.index + _PERIODS[period] <= as_of_close]


def build_replay_context(
    *,
    history_1h: pd.DataFrame,
    as_of: pd.Timestamp,
    df_1d: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    news_headlines: tuple[str, ...] = (),
    window: int = 500,
) -> MarketContext:
    """Context exactly as an execution at the close of candle ``as_of``
    would have seen it.

    ``history_1h`` must already be indicator-computed (rolling indicators
    are backward-looking, so computing them once over the full history
    introduces no lookahead). The 1h window is capped at ``window`` rows
    to match the live fetch limit.
    """
    as_of_close = as_of + _PERIODS["1h"]
    return MarketContext(
        df=history_1h.loc[:as_of].tail(window),
        df_1d=closed_candles_as_of(df_1d, "1d", as_of_close),
        df_4h=closed_candles_as_of(df_4h, "4h", as_of_close),
        funding_rate=None,
        imbalance=None,
        news_headlines=tuple(news_headlines),
    )
