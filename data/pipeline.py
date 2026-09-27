"""Shared market-data → indicator-frame pipeline.

The sequence *fetch OHLCV → drop the still-forming candle → preprocess → compute
indicators* was copy-pasted into every live entrypoint (run_testnet,
run_decoupled_intelligence, …). Centralising it here removes that duplication
and guarantees every entrypoint prepares data identically.

Preprocessor and indicator computation are stateless, so module-level singletons
are reused rather than re-instantiated per call.

(``scripts/run_live.py`` deliberately keeps its pipeline inline — it measures
per-stage latency (fetch vs indicators separately), which a single call would
collapse.)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from core.candles import drop_forming_candle
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators

_preprocessor = DataPreprocessor()
_tech = TechnicalIndicators()


def build_indicator_frame(
    fetcher,
    pair: str,
    timeframe: str,
    limit: int = 500,
    *,
    drop_forming: bool = True,
    min_rows: int = 100,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV and return a cleaned, indicator-computed frame.

    Returns None (with a warning) when there isn't enough data. With
    ``drop_forming=True`` the decision is made on the last CLOSED candle
    (backtest parity); execution-side callers that act on the current price
    pass ``drop_forming=False``.
    """
    df = fetcher.fetch_ohlcv(pair=pair, timeframe=timeframe, limit=limit)
    if drop_forming:
        df = drop_forming_candle(df, timeframe)
    if df is None or len(df) < min_rows:
        logger.warning(
            "Insufficient data for {} ({}): {} rows",
            pair, timeframe, 0 if df is None else len(df),
        )
        return None
    return _tech.compute_all(_preprocessor.process(df))
