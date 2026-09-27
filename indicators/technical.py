"""
Technical indicators — wraps pandas-ta to compute all indicators.
Returns a unified DataFrame with all indicator columns attached.

Usage:
    from indicators.technical import TechnicalIndicators

    ti = TechnicalIndicators()
    df_with_indicators = ti.compute_all(df)
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from loguru import logger

from config.constants import (
    DEFAULT_RSI_PERIOD, DEFAULT_RSI_OVERBOUGHT, DEFAULT_RSI_OVERSOLD,
    DEFAULT_SMA_FAST, DEFAULT_SMA_SLOW,
    DEFAULT_EMA_FAST, DEFAULT_EMA_SLOW, DEFAULT_MACD_SIGNAL,
    DEFAULT_BB_PERIOD, DEFAULT_BB_STD, DEFAULT_ATR_PERIOD,
)


class TechnicalIndicators:
    """
    Computes all technical indicators using pandas-ta.

    Indicators computed:
    - Trend: SMA, EMA, VWAP, Supertrend
    - Momentum: RSI, MACD, Stochastic, CCI, Williams %R
    - Volatility: Bollinger Bands, ATR, Keltner Channels
    - Volume: OBV, MFI, VWAP
    - Custom: Candle patterns, trend strength
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and attach to DataFrame.

        Args:
            df: OHLCV DataFrame with index as datetime

        Returns:
            DataFrame with all indicator columns added
        """
        df = df.copy()

        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_custom_indicators(df)

        # Drop rows where indicators are NaN (beginning of data)
        # Keep enough for ML features — drop only the first 50 rows max
        df.dropna(subset=["RSI", "MACD", "BB_upper"], inplace=True)

        logger.debug(
            "Computed {} indicators — {} rows remaining",
            len([c for c in df.columns if c not in ["open","high","low","close","volume"]]),
            len(df)
        )

        return df

    @staticmethod
    def _ma_or_nan(result, index: pd.Index) -> pd.Series:
        """pandas-ta returns None (not NaN) when the frame is shorter than
        the MA length; assigning None creates an object column that makes
        float comparisons raise. This crashed compute_all() on any frame
        under 200 rows — including the live 1D/4H macro-trend fetches
        (limit=100), which silently blocked every BUY."""
        if result is None:
            return pd.Series(np.nan, index=index)
        return result

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""

        # ── Simple Moving Averages ──────────────────────────────────────────────
        df[f"SMA_{DEFAULT_SMA_FAST}"] = self._ma_or_nan(
            ta.sma(df["close"], length=DEFAULT_SMA_FAST), df.index)
        df[f"SMA_{DEFAULT_SMA_SLOW}"] = self._ma_or_nan(
            ta.sma(df["close"], length=DEFAULT_SMA_SLOW), df.index)
        df["SMA_200"] = self._ma_or_nan(ta.sma(df["close"], length=200), df.index)

        # ── Exponential Moving Averages ─────────────────────────────────────────
        df[f"EMA_{DEFAULT_EMA_FAST}"] = self._ma_or_nan(
            ta.ema(df["close"], length=DEFAULT_EMA_FAST), df.index)
        df[f"EMA_{DEFAULT_EMA_SLOW}"] = self._ma_or_nan(
            ta.ema(df["close"], length=DEFAULT_EMA_SLOW), df.index)
        df["EMA_9"] = self._ma_or_nan(ta.ema(df["close"], length=9), df.index)
        df["EMA_21"] = self._ma_or_nan(ta.ema(df["close"], length=21), df.index)

        # ── SMA Crossover Signal ────────────────────────────────────────────────
        df["SMA_cross"] = (
            df[f"SMA_{DEFAULT_SMA_FAST}"] > df[f"SMA_{DEFAULT_SMA_SLOW}"]
        ).astype(int)

        # ── Trend Direction ─────────────────────────────────────────────────────
        # 1 = above 200 SMA (uptrend), -1 = below (downtrend)
        df["trend_200"] = np.where(df["close"] > df["SMA_200"], 1, -1)

        # ── Supertrend ──────────────────────────────────────────────────────────
        # Always create the columns (NaN on a too-short frame), never omit
        # them. A vanished Supertrend_dir makes trend_following silently
        # return zero signals — the same class of bug _ma_or_nan guards for
        # the EMAs. Downstream code already treats NaN as "warming up".
        supertrend = ta.supertrend(df["high"], df["low"], df["close"])
        if supertrend is not None and not supertrend.empty:
            df["Supertrend"] = supertrend.iloc[:, 0]       # Supertrend line
            df["Supertrend_dir"] = supertrend.iloc[:, 1]   # Direction: 1=up, -1=down
        else:
            df["Supertrend"] = np.nan
            df["Supertrend_dir"] = np.nan

        # ── EMA 50 ──────────────────────────────────────────────────────────────
        df["EMA_50"] = self._ma_or_nan(ta.ema(df["close"], length=50), df.index)

        # ── ADX (Average Directional Index) ─────────────────────────────────────
        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None and not adx.empty:
            df["ADX"] = adx.iloc[:, 0]
        else:
            df["ADX"] = np.nan

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators."""

        # ── RSI (Relative Strength Index) ───────────────────────────────────────
        df["RSI"] = ta.rsi(df["close"], length=DEFAULT_RSI_PERIOD)
        df["RSI_signal"] = 0
        df.loc[df["RSI"] < DEFAULT_RSI_OVERSOLD, "RSI_signal"] = 1    # Oversold = bullish
        df.loc[df["RSI"] > DEFAULT_RSI_OVERBOUGHT, "RSI_signal"] = -1  # Overbought = bearish

        # ── MACD ────────────────────────────────────────────────────────────────
        macd = ta.macd(
            df["close"],
            fast=DEFAULT_EMA_FAST,
            slow=DEFAULT_EMA_SLOW,
            signal=DEFAULT_MACD_SIGNAL,
        )
        if macd is not None and not macd.empty:
            # Use column name matching — pandas-ta column order varies by version.
            # Column names follow the pattern: MACD_F_S_Sig, MACDh_F_S_Sig, MACDs_F_S_Sig
            macd_cols = macd.columns.tolist()
            macd_line_col = [c for c in macd_cols if c.startswith("MACD_") and not c.startswith("MACDh") and not c.startswith("MACDs")]
            macd_hist_col = [c for c in macd_cols if c.startswith("MACDh")]
            macd_signal_col = [c for c in macd_cols if c.startswith("MACDs")]

            df["MACD"] = macd[macd_line_col[0]] if macd_line_col else macd.iloc[:, 0]
            df["MACD_hist"] = macd[macd_hist_col[0]] if macd_hist_col else macd.iloc[:, 1]
            df["MACD_signal"] = macd[macd_signal_col[0]] if macd_signal_col else macd.iloc[:, 2]
        else:
            # Always create the columns (NaN on a too-short frame) so the
            # MACD_cross line below and downstream code never KeyError on a
            # vanished column — same guarantee _ma_or_nan gives the EMAs.
            df["MACD"] = np.nan
            df["MACD_hist"] = np.nan
            df["MACD_signal"] = np.nan

        # MACD cross signal: 1 = bullish cross, -1 = bearish cross
        df["MACD_cross"] = np.where(df["MACD"] > df["MACD_signal"], 1, -1)

        # ── Stochastic Oscillator ───────────────────────────────────────────────
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df["Stoch_K"] = stoch.iloc[:, 0]
            df["Stoch_D"] = stoch.iloc[:, 1]
        else:
            df["Stoch_K"] = np.nan
            df["Stoch_D"] = np.nan

        # ── CCI (Commodity Channel Index) ────────────────────────────────────────
        df["CCI"] = ta.cci(df["high"], df["low"], df["close"])

        # ── Williams %R ─────────────────────────────────────────────────────────
        df["Williams_R"] = ta.willr(df["high"], df["low"], df["close"])

        # ── ROC (Rate of Change) ─────────────────────────────────────────────────
        df["ROC"] = ta.roc(df["close"], length=10)

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""

        # ── Bollinger Bands ─────────────────────────────────────────────────────
        bb = ta.bbands(df["close"], length=DEFAULT_BB_PERIOD, std=DEFAULT_BB_STD)
        if bb is not None and not bb.empty:
            df["BB_lower"] = bb.iloc[:, 0]
            df["BB_mid"] = bb.iloc[:, 1]
            df["BB_upper"] = bb.iloc[:, 2]
            df["BB_width"] = bb.iloc[:, 3]    # Band width (volatility measure)
            df["BB_pct"] = bb.iloc[:, 4]      # %B: position within bands
        else:
            # Always create the columns (NaN on a too-short frame): the
            # BB_squeeze line below reads BB_width, and compute_all dropna's
            # on BB_upper — a vanished column would KeyError, not warm up.
            df["BB_lower"] = np.nan
            df["BB_mid"] = np.nan
            df["BB_upper"] = np.nan
            df["BB_width"] = np.nan
            df["BB_pct"] = np.nan

        # BB squeeze: tight bands = low volatility = potential breakout incoming
        df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(20).mean() * 0.8).astype(int)

        # ── ATR (Average True Range) ─────────────────────────────────────────────
        df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=DEFAULT_ATR_PERIOD)

        # Normalized ATR (as % of price) — for position sizing
        df["ATR_pct"] = df["ATR"] / df["close"] * 100

        # ── Keltner Channels ─────────────────────────────────────────────────────
        kc = ta.kc(df["high"], df["low"], df["close"])
        if kc is not None and not kc.empty:
            df["KC_lower"] = kc.iloc[:, 0]
            df["KC_mid"] = kc.iloc[:, 1]
            df["KC_upper"] = kc.iloc[:, 2]
        else:
            df["KC_lower"] = np.nan
            df["KC_mid"] = np.nan
            df["KC_upper"] = np.nan

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""

        # ── OBV (On-Balance Volume) ──────────────────────────────────────────────
        df["OBV"] = ta.obv(df["close"], df["volume"])

        # OBV trend (rising OBV = buying pressure)
        df["OBV_trend"] = np.where(
            df["OBV"] > df["OBV"].rolling(20).mean(), 1, -1
        )

        # ── MFI (Money Flow Index) ───────────────────────────────────────────────
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"])

        # ── VWAP (Volume Weighted Average Price) ─────────────────────────────────
        # VWAP resets daily — using rolling approximation for multi-day data
        df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # Price vs VWAP: 1 = above (bullish), -1 = below (bearish)
        if "VWAP" in df.columns:
            df["price_vs_vwap"] = np.where(df["close"] > df["VWAP"], 1, -1)

        # ── Volume SMA ───────────────────────────────────────────────────────────
        df["Volume_SMA_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["Volume_SMA_20"]  # >1 = above avg volume

        return df

    def add_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom composite indicators and signals."""

        # ── Overall Trend Score (-3 to +3) ──────────────────────────────────────
        # Sum of multiple trend signals — more positive = stronger uptrend
        trend_signals = []

        if "SMA_cross" in df.columns:
            trend_signals.append(df["SMA_cross"].map({1: 1, 0: -1}))
        if "MACD_cross" in df.columns:
            trend_signals.append(df["MACD_cross"])
        if "trend_200" in df.columns:
            trend_signals.append(df["trend_200"])
        if "OBV_trend" in df.columns:
            trend_signals.append(df["OBV_trend"])

        if trend_signals:
            df["trend_score"] = sum(trend_signals)

        # ── RSI Divergence Flag ──────────────────────────────────────────────────
        # Simple version: price makes new high but RSI doesn't (bearish divergence)
        if "RSI" in df.columns:
            price_new_high = df["close"] > df["close"].rolling(14).max().shift(1)
            rsi_not_new_high = df["RSI"] < df["RSI"].rolling(14).max().shift(1)
            df["rsi_divergence"] = (price_new_high & rsi_not_new_high).astype(int)

        # ── Composite Signal Score ───────────────────────────────────────────────
        # A quick combined signal for visualization (-1 strong sell, +1 strong buy)
        score_components = []

        if "RSI_signal" in df.columns:
            score_components.append(df["RSI_signal"] * 0.3)
        if "MACD_cross" in df.columns:
            score_components.append(df["MACD_cross"] * 0.3)
        if "trend_score" in df.columns:
            # Normalize trend_score to -1..1 range
            score_components.append(df["trend_score"] / 4 * 0.4)

        if score_components:
            df["composite_score"] = sum(score_components).clip(-1, 1)

        return df

    def get_latest_signals(self, df: pd.DataFrame) -> dict:
        """
        Extract the latest indicator values as a clean summary dict.
        Useful for AI analysis and decision making.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        return {
            # Price
            "price": round(latest["close"], 4),
            "price_change": round(
                ((latest["close"] - prev["close"]) / prev["close"]) * 100, 3
            ),

            # RSI
            "rsi": round(latest.get("RSI", 50), 2),
            "rsi_signal": int(latest.get("RSI_signal", 0)),

            # MACD
            "macd": round(latest.get("MACD", 0), 6),
            "macd_signal": round(latest.get("MACD_signal", 0), 6),
            "macd_hist": round(latest.get("MACD_hist", 0), 6),
            "macd_cross": int(latest.get("MACD_cross", 0)),

            # Bollinger Bands
            "bb_upper": round(latest.get("BB_upper", 0), 4),
            "bb_mid": round(latest.get("BB_mid", 0), 4),
            "bb_lower": round(latest.get("BB_lower", 0), 4),
            "bb_pct": round(latest.get("BB_pct", 0.5), 4),
            "bb_squeeze": bool(latest.get("BB_squeeze", False)),

            # Trend
            "sma_fast": round(latest.get(f"SMA_{DEFAULT_SMA_FAST}", 0), 4),
            "sma_slow": round(latest.get(f"SMA_{DEFAULT_SMA_SLOW}", 0), 4),
            "trend_200": int(latest.get("trend_200", 0)),
            "trend_score": int(latest.get("trend_score", 0)),

            # Volume
            "volume_ratio": round(latest.get("volume_ratio", 1), 2),
            "obv_trend": int(latest.get("OBV_trend", 0)),
            "mfi": round(latest.get("MFI", 50), 2),

            # Volatility
            "atr_pct": round(latest.get("ATR_pct", 0), 3),

            # Composite
            "composite_score": round(latest.get("composite_score", 0), 3),
        }
