"""Tier-17 tests: paper-run readiness — strategy-agnostic live loop,
portfolio heat cap, market-structure recording."""

from decimal import Decimal
from pathlib import Path

from config.settings import Settings
from risk.manager import RiskConfig, RiskManager
from storage.trade_logger import TradeLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Strategy-agnostic live loop ───────────────────────────────────────────────

class TestStrategySelection:
    def test_settings_field_exists_with_ai_default(self):
        assert Settings().strategy_name == "ai_combined"

    def test_run_live_uses_registry(self):
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "get_strategy(settings.strategy_name)" in src
        assert 'getattr(strategy, "last_ai_signal", None)' in src, (
            "rule-based strategies have no last_ai_signal — direct attribute "
            "access would crash the live loop"
        )


# ─── Portfolio heat cap ────────────────────────────────────────────────────────

def _config(heat: str = "0.06", risk: str = "0.02") -> RiskConfig:
    return RiskConfig(
        max_position_notional_usdt=Decimal("10000"),
        max_daily_loss_pct=Decimal("0.03"),
        max_drawdown_pct=Decimal("0.10"),
        risk_per_trade_pct=Decimal(risk),
        max_portfolio_heat_pct=Decimal(heat),
    )


class HeatDB:
    """Open trades worth a configurable amount of stop-distance risk."""
    def __init__(self, open_trades):
        self._open = open_trades

    def get_open_trades(self):
        return self._open

    def get_consecutive_losses(self):
        return 0

    def get_trade_history(self, limit=1):
        return []


def _trade(entry, stop, qty):
    return {"entry_price": entry, "stop_loss": stop, "quantity": qty}


class TestPortfolioHeat:
    def test_open_risk_below_cap_allows(self):
        risk = RiskManager(config=_config(), initial_balance=10_000.0)
        # one open position risking 2% (200/10000); +2% new = 4% < 6% cap
        db = HeatDB([_trade(100.0, 98.0, 100.0)])
        allowed, reason = risk.can_open_trade(
            current_balance=10_000.0, open_position_count=1,
            daily_pnl=0.0, db=db)
        assert allowed, reason

    def test_open_risk_at_cap_blocks(self):
        risk = RiskManager(config=_config(), initial_balance=10_000.0)
        # two open positions risking 2% each; +2% new = 6.0%+ > 6% cap?
        # exactly-at-cap passes; push slightly over with 2.1% open each
        db = HeatDB([_trade(100.0, 97.9, 100.0), _trade(50.0, 48.95, 200.0)])
        allowed, reason = risk.can_open_trade(
            current_balance=10_000.0, open_position_count=2,
            daily_pnl=0.0, db=db)
        assert not allowed
        assert "heat" in reason.lower()

    def test_rows_without_stops_contribute_nothing(self):
        risk = RiskManager(config=_config(), initial_balance=10_000.0)
        db = HeatDB([{"entry_price": 100.0, "quantity": 100.0}])  # no stop key
        allowed, _ = risk.can_open_trade(
            current_balance=10_000.0, open_position_count=1,
            daily_pnl=0.0, db=db)
        assert allowed

    def test_short_heat_counts_absolute_distance(self):
        """A short's stop is ABOVE entry — heat must use |entry − stop|."""
        risk = RiskManager(config=_config(heat="0.05"), initial_balance=10_000.0)
        # short: entry 100, stop 103.5 → risk 350 = 3.5%; +2% new > 5% cap
        db = HeatDB([_trade(100.0, 103.5, 100.0)])
        allowed, reason = risk.can_open_trade(
            current_balance=10_000.0, open_position_count=1,
            daily_pnl=0.0, db=db)
        assert not allowed
        assert "heat" in reason.lower()

    def test_factory_wires_heat_from_settings(self):
        cfg = RiskConfig.from_settings(10_000.0)
        assert cfg.max_portfolio_heat_pct == Decimal(str(Settings().portfolio_max_heat))


# ─── Market-structure recording ────────────────────────────────────────────────

class TestMarketStructure:
    def test_log_and_read_back(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "m.db"))
        db.log_market_structure(symbol="BTC/USDT", open_interest=81234.5,
                                taker_ratio=1.07, funding_rate=0.0001)
        db.log_market_structure(symbol="BTC/USDT", open_interest=None,
                                taker_ratio=None, funding_rate=0.0002)
        with db._get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol, open_interest, taker_ratio, funding_rate "
                "FROM market_structure ORDER BY id").fetchall()
        db.close()
        assert len(rows) == 2
        assert tuple(rows[0]) == ("BTC/USDT", 81234.5, 1.07, 0.0001)
        # absence is stored as NULL, never fabricated
        assert rows[1][1] is None and rows[1][2] is None

    def test_run_live_records_each_cycle(self):
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "log_market_structure" in src
        assert "fetch_open_interest" in src
        assert "fetch_taker_ratio" in src

    def test_fetchers_return_none_on_failure(self):
        from data.fetcher import DataFetcher

        f = DataFetcher.__new__(DataFetcher)

        class Boom:
            def fetch_open_interest(self, *a, **k):
                raise ConnectionError("down")

            def fapiDataGetTakerlongshortRatio(self, *a, **k):
                raise ConnectionError("down")

        f._exchange = Boom()
        assert f.fetch_open_interest("BTC/USDT") is None
        assert f.fetch_taker_ratio("BTC/USDT") is None

    def test_fetchers_parse_success_payloads(self):
        from data.fetcher import DataFetcher

        f = DataFetcher.__new__(DataFetcher)

        class Fake:
            def fetch_open_interest(self, symbol):
                assert symbol.endswith(":USDT")
                return {"openInterestAmount": 81234.5}

            def fapiDataGetTakerlongshortRatio(self, params):
                assert params["symbol"] == "BTCUSDT"
                return [{"buySellRatio": "1.0700"}]

        f._exchange = Fake()
        assert f.fetch_open_interest("BTC/USDT") == 81234.5
        assert f.fetch_taker_ratio("BTC/USDT") == 1.07
