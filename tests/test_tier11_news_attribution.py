"""Tier-11 tests: live news wiring into the LLM/sentiment leg, and the
component-attribution measurement tool."""

import sys
from pathlib import Path

import pandas as pd

from strategies.ai_combined import AICombinedStrategy
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from component_attribution import attribution, load_joined  # noqa: E402


# ─── News wiring ───────────────────────────────────────────────────────────────

class FakeNewsFetcher:
    def __init__(self, titles):
        self.titles = titles
        self.calls = []

    def fetch_pair_news(self, pair, hours_back=12):
        self.calls.append(pair)
        return [{"title": t} for t in self.titles]


class FailingNewsFetcher:
    def fetch_pair_news(self, pair, hours_back=12):
        raise ConnectionError("news API down")


class NoMarketData:
    """Market-data fetcher stub: every fetch fails → those ctx fields None."""
    def fetch_ohlcv(self, *a, **k): raise ConnectionError("offline")
    def fetch_funding_rate(self, *a, **k): raise ConnectionError("offline")
    def fetch_order_book_imbalance(self, *a, **k): raise ConnectionError("offline")


def make_strategy(news_fetcher) -> AICombinedStrategy:
    strat = AICombinedStrategy(pair="BTC/USDT", use_llm=False,
                               news_fetcher=news_fetcher)
    strat._fetcher = NoMarketData()
    return strat


def tiny_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)


class TestNewsWiring:
    def test_live_context_carries_fetched_headlines(self):
        fetcher = FakeNewsFetcher(["BTC ETF approved", "Fed cuts rates"])
        strat = make_strategy(fetcher)
        ctx = strat._build_live_context(tiny_df(), "BTC/USDT")
        assert ctx.news_headlines == ("BTC ETF approved", "Fed cuts rates")
        assert fetcher.calls == ["BTC/USDT"]

    def test_news_failure_falls_back_to_static_list(self):
        strat = make_strategy(FailingNewsFetcher())
        strat.news_headlines = ["static headline"]
        ctx = strat._build_live_context(tiny_df(), "BTC/USDT")
        assert ctx.news_headlines == ("static headline",)

    def test_empty_fetch_keeps_static_list(self):
        strat = make_strategy(FakeNewsFetcher([]))
        strat.news_headlines = ["static headline"]
        ctx = strat._build_live_context(tiny_df(), "BTC/USDT")
        assert ctx.news_headlines == ("static headline",)

    def test_entrypoints_wire_news_fetcher(self):
        for rel in ("scripts/run_live.py", "scripts/run_decoupled_intelligence.py"):
            src = (PROJECT_ROOT / rel).read_text()
            assert "news_fetcher=news_fetcher" in src, rel
            assert "settings.news_api_key" in src, rel

    def test_intelligence_publishes_decision_scores(self):
        """Published component scores must come from the decision itself
        (last_ai_signal), and `score` must be the combined score — the
        publisher contract is [-1, +1], not confidence (audit V-30)."""
        src = (PROJECT_ROOT / "scripts" / "run_decoupled_intelligence.py").read_text()
        # Read via getattr so rule-based strategies (no last_ai_signal) don't
        # crash the node; still sourced from the decision, not a re-run.
        assert 'getattr(strategy, "last_ai_signal", None)' in src
        assert "score=ai.score if ai else 0.0" in src
        assert "score=trade_signal.confidence" not in src
        assert '"llm_score": llm_score' in src


# ─── Component attribution ─────────────────────────────────────────────────────

def seed_db(tmp_path) -> str:
    """Synthetic history: llm_score predicts outcomes, sentiment is noise."""
    db_path = str(tmp_path / "t.db")
    db = TradeLogger(db_path=db_path)
    cases = [
        # (corr_id, llm, sentiment, pnl)  — llm sign matches pnl sign
        ("c1", 0.8, 0.1, 50.0),
        ("c2", 0.6, -0.2, 30.0),
        ("c3", -0.7, 0.3, -40.0),
        ("c4", -0.5, -0.1, -20.0),
        ("c5", 0.9, 0.2, 60.0),
    ]
    for corr, llm, sent, pnl in cases:
        db.log_signal(symbol="BTC/USDT", ml_score=0.0, llm_score=llm,
                      sentiment_score=sent, combined_score=llm,
                      confidence=0.8, decision="BUY", correlation_id=corr)
        tid = db.log_trade_open(symbol="BTC/USDT", side="buy", quantity=1.0,
                                entry_price=100.0, stop_loss=95.0,
                                take_profit=110.0, correlation_id=corr)
        db.log_trade_close(trade_id=tid, exit_price=100.0 + pnl, pnl=pnl,
                           exit_reason="signal")
    db.close()
    return db_path


class TestAttribution:
    def test_join_and_report(self, tmp_path):
        db_path = seed_db(tmp_path)
        df = load_joined(db_path)
        assert len(df) == 5

        report = attribution(df).set_index("component")
        # llm_score perfectly predicts outcome direction in the seed data
        assert report.loc["llm_score", "pnl_correlation"] > 0.8
        assert report.loc["llm_score", "directional_hit_rate"] == 1.0
        # sentiment is uncorrelated noise — hit rate well below llm's
        assert report.loc["sentiment_score", "directional_hit_rate"] < 1.0

    def test_unlinked_trades_are_excluded(self, tmp_path):
        db_path = str(tmp_path / "t.db")
        db = TradeLogger(db_path=db_path)
        tid = db.log_trade_open(symbol="BTC/USDT", side="buy", quantity=1.0,
                                entry_price=100.0, stop_loss=95.0,
                                take_profit=110.0)  # no correlation_id
        db.log_trade_close(trade_id=tid, exit_price=105.0, pnl=5.0,
                           exit_reason="signal")
        db.close()
        assert load_joined(db_path).empty
