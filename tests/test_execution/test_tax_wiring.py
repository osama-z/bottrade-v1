"""Final integration — tax-lot ledger wired into PaperTrader (open→acquire, close→dispose)."""
import pandas as pd

from ai.signal_combiner import AISignal
from config.constants import Signal
from execution.paper_trader import PaperTrader
from storage.tax_lots import TaxLotLedger
from storage.trade_logger import TradeLogger


def make_df(price=100.0, atr=2.0):
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=5, freq="1h")
    df = pd.DataFrame({"open": price, "high": price * 1.01, "low": price * 0.99,
                       "close": price, "volume": 100.0}, index=idx)
    df["ATR"] = atr
    return df


def _buy():
    return AISignal(signal=Signal.BUY, score=0.8, confidence=0.9, is_actionable=True,
                    llm_score=0.0, ml_score=0.0, sentiment_score=0.0)


def _trader(tmp_path):
    dbp = str(tmp_path / "t.db")
    tax = TaxLotLedger(db_path=str(tmp_path / "tax.db"))
    trader = PaperTrader(initial_balance=10_000.0, db=TradeLogger(db_path=dbp),
                         db_path=dbp, tax_ledger=tax)
    return trader, tax


class TestTaxWiring:
    def test_open_acquires_a_lot(self, tmp_path):
        trader, tax = _trader(tmp_path)
        trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=_buy())
        lots = tax.open_lots("BTC/USDT")
        assert len(lots) == 1 and lots[0].remaining > 0

    def test_close_disposes_and_books_realized_pnl(self, tmp_path):
        trader, tax = _trader(tmp_path)
        trader.process_candle(df=make_df(price=100.0), pair="BTC/USDT", ai_signal=_buy())
        assert tax.open_lots("BTC/USDT")                      # lot exists
        trader.force_close_position("BTC/USDT", current_price=110.0)
        assert tax.open_lots("BTC/USDT") == []                # fully disposed
        assert tax.realized_pnl("BTC/USDT") > 0               # sold at 110 > ~100 basis

    def test_no_ledger_is_a_noop(self, tmp_path):
        # Without a tax ledger, trading is unchanged (backward compatible).
        dbp = str(tmp_path / "t.db")
        trader = PaperTrader(initial_balance=10_000.0, db=TradeLogger(db_path=dbp), db_path=dbp)
        trader.process_candle(df=make_df(), pair="BTC/USDT", ai_signal=_buy())
        assert len(trader._db.get_open_trades()) == 1         # trades normally
