"""PaperTrader safety guards: stale-data abort and emergency force-close.

Two money-relevant paths that were untested:
- process_candle must REFUSE to act on a stale feed (trading on a dead price
  is how a bot bleeds out silently).
- force_close_position (Telegram /force_sell) must close exactly the matching
  pair and report honestly.
"""
import pandas as pd

from ai.signal_combiner import AISignal
from config.constants import Signal
from execution.paper_trader import PaperTrader
from storage.trade_logger import TradeLogger


class FakeFetcher:
    def __init__(self, price: float = 105.0):
        self.price = price

    def fetch_ticker(self, pair: str) -> dict:
        return {"last": self.price, "bid": self.price}


def make_df(hours_old: float = 0.0, price: float = 100.0, atr: float = 2.0) -> pd.DataFrame:
    """5x 1h candles ending `hours_old` hours before now."""
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_old)
    idx = pd.date_range(end=end, periods=5, freq="1h")
    df = pd.DataFrame(
        {"open": price, "high": price * 1.01, "low": price * 0.99,
         "close": price, "volume": 100.0},
        index=idx,
    )
    df["ATR"] = atr
    return df


def buy_signal() -> AISignal:
    return AISignal(signal=Signal.BUY, score=0.8, confidence=0.9, is_actionable=True,
                    llm_score=0.0, ml_score=0.0, sentiment_score=0.0)


def _trader(tmp_path) -> tuple[PaperTrader, TradeLogger]:
    db_path = str(tmp_path / "t.db")
    db = TradeLogger(db_path=db_path)
    return PaperTrader(initial_balance=10_000.0, db=db, db_path=db_path), db


class TestStaleDataGuard:
    def test_fresh_data_opens_a_trade(self, tmp_path):
        trader, db = _trader(tmp_path)
        trader.process_candle(df=make_df(hours_old=0.0), pair="BTC/USDT", ai_signal=buy_signal())
        assert len(db.get_open_trades()) == 1     # baseline: fresh feed trades

    def test_stale_data_aborts_the_cycle(self, tmp_path):
        # Default tf=1h, stale threshold = 300s + 3600s ≈ 65min; 3 days old is
        # unambiguously stale → the cycle must abort with no position opened.
        trader, db = _trader(tmp_path)
        trader.process_candle(df=make_df(hours_old=72.0), pair="BTC/USDT", ai_signal=buy_signal())
        assert len(db.get_open_trades()) == 0     # refused to trade on a dead feed


class TestForceClose:
    def _open(self, tmp_path):
        trader, db = _trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=buy_signal())
        assert len(db.get_open_trades()) == 1
        return trader, db

    def test_force_close_closes_matching_position(self, tmp_path):
        trader, db = self._open(tmp_path)
        msg = trader.force_close_position("BTC/USDT", current_price=110.0)
        assert "Force closed 1 position" in msg
        assert len(db.get_open_trades()) == 0

    def test_force_close_reports_when_nothing_to_close(self, tmp_path):
        trader, _ = self._open(tmp_path)
        msg = trader.force_close_position("ETH/USDT", current_price=110.0)
        assert "No open position found" in msg

    def test_force_close_leaves_other_pairs_untouched(self, tmp_path):
        trader, db = self._open(tmp_path)                 # BTC open
        trader.force_close_position("ETH/USDT", current_price=110.0)
        assert len(db.get_open_trades()) == 1             # BTC still open
