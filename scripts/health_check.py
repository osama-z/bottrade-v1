"""
Script: Health check — verify all modules are working correctly.
Run this after setup to confirm everything is connected.

Usage:
    python scripts/health_check.py
"""

import sys
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
# Must precede any project import: without it `python scripts/health_check.py`
# dies with ModuleNotFoundError before running a single check.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging  # noqa: E402


def check_imports() -> bool:
    """Verify all core imports work."""
    try:
        from config.settings import settings
        from config.constants import Signal, BotMode
        from core.exceptions import NeuronTradeError
        from core.event_bus import event_bus
        from data.fetcher import DataFetcher
        from data.preprocessor import DataPreprocessor
        from data.cache import cache
        from data.news_fetcher import NewsFetcher
        from data.sentiment_fetcher import SentimentFetcher
        from indicators.technical import TechnicalIndicators
        from indicators.features import FeatureEngineer
        from strategies.registry import list_strategies, get_strategy
        from backtesting.engine import BacktestEngine
        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def check_settings() -> bool:
    """Verify settings load correctly."""
    try:
        from config.settings import settings
        print("  ✅ Settings loaded")
        print(f"     Testnet: {settings.binance_testnet}")
        print(f"     Paper trading: {settings.paper_trading}")
        print(f"     Trading pairs: {settings.trading_pairs}")
        print(f"     Log level: {settings.log_level}")

        warnings = []
        if not settings.binance_api_key:
            warnings.append("BINANCE_API_KEY not set")
        if not settings.groq_api_key:
            warnings.append("GROQ_API_KEY not set")
        if not settings.telegram_bot_token:
            warnings.append("TELEGRAM_BOT_TOKEN not set")
        if not settings.news_api_key:
            warnings.append("NEWS_API_KEY not set")

        if warnings:
            print("  ⚠️  Missing API keys (set in .env):")
            for w in warnings:
                print(f"     - {w}")

        return True
    except Exception as e:
        print(f"  ❌ Settings error: {e}")
        return False


def check_strategies() -> bool:
    """Verify strategies load correctly."""
    try:
        from strategies.registry import list_strategies, get_strategy
        strategies = list_strategies()
        print(f"  ✅ Strategies registered: {strategies}")
        for name in strategies:
            s = get_strategy(name)
            print(f"     - {name}: {s.get_params()}")
        return True
    except Exception as e:
        print(f"  ❌ Strategy error: {e}")
        return False


def check_sentiment() -> bool:
    """Verify Fear & Greed Index fetch works (no API key needed)."""
    try:
        from data.sentiment_fetcher import SentimentFetcher
        fetcher = SentimentFetcher()
        sentiment = fetcher.get_current_sentiment()
        print(f"  ✅ Fear & Greed Index: {sentiment['value']}/100 — {sentiment['classification']}")
        return True
    except Exception as e:
        print(f"  ❌ Sentiment fetch error: {e}")
        return False


def check_exchange() -> bool:
    """Try connecting to Binance (public endpoints only)."""
    try:
        from data.fetcher import DataFetcher
        from config.settings import settings
        if not settings.binance_api_key:
            print("  ⚠️  Skipping exchange check — no API key")
            return True
        fetcher = DataFetcher()
        ticker = fetcher.fetch_ticker("BTC/USDT")
        print(f"  ✅ Binance connected — BTC/USDT: ${ticker['price']:,.2f}")
        return True
    except Exception as e:
        print(f"  ❌ Exchange error: {e}")
        return False


def check_pairs() -> bool:
    """Validate configured pairs and report which have trained models.

    The trap this catches: adding a pair to TRADING_PAIRS without training
    its XGBoost/HMM models. The bot doesn't crash — it silently falls back
    to the LLM+sentiment path for that pair, so the ML and regime legs are
    quietly absent and you only notice in the results weeks later.
    """
    try:
        from pathlib import Path as _Path

        from config.pairs import get_pair_spec, validate_pairs
        from config.settings import settings

        pairs = settings.trading_pairs
        print(f"  Configured pairs: {', '.join(pairs)}")

        problems = validate_pairs(pairs)
        if problems:
            print("  ⚠️  Pair configuration warnings:")
            for p in problems:
                print(f"     - {p}")
        else:
            print("  ✅ Pair configuration valid")

        model_dir = _Path(__file__).resolve().parent.parent / "ai" / "models"
        untrained: list[str] = []
        for pair in pairs:
            safe = pair.replace("/", "_")
            has_ml = (model_dir / f"xgb_{safe}.joblib").exists()
            has_scaler = (model_dir / f"scaler_{safe}.joblib").exists()
            has_regime = (model_dir / f"regime_hmm_{safe}.joblib").exists()
            spec = get_pair_spec(pair)

            marks = (
                f"ML={'✅' if has_ml and has_scaler else '❌'} "
                f"HMM={'✅' if has_regime else '❌'}"
            )
            print(f"     • {pair:<12} {marks}   ({spec.notes})")
            if not (has_ml and has_scaler and has_regime):
                untrained.append(pair)

        if untrained:
            print("  ⚠️  Missing models — these pairs trade on LLM+sentiment only:")
            for pair in untrained:
                print(f"     python scripts/train_model.py  --pair {pair}")
                print(f"     python scripts/train_regime.py --pair {pair}")
        # Warnings, not failures: the bot runs (degraded) without models.
        return True
    except Exception as e:
        print(f"  ❌ Pairs check error: {e}")
        return False


def check_database() -> bool:
    """The check that matters most after a crash: is the state DB sane?

    Verifies the DB opens at the configured path, the circuit-breaker
    state is readable (and reports it — a TRIPPED breaker means an
    operator reset is required before trading resumes), and lists any
    persisted open positions plus the saved balance.
    """
    try:
        from config.settings import settings
        from storage.trade_logger import TradeLogger
        from risk.manager import SqliteCircuitBreakerStore, CircuitBreakerState

        db = TradeLogger()  # configured absolute path
        print(f"  ✅ Database opens: {settings.database_path}")

        store = SqliteCircuitBreakerStore(settings.database_path)
        record = store.load()
        if record.state is CircuitBreakerState.TRIPPED:
            print(f"  ⚠️  Circuit breaker TRIPPED: {record.reason!r} "
                  f"(tripped at {record.tripped_at_utc}) — manual reset required")
        else:
            print("  ✅ Circuit breaker ARMED")

        open_trades = db.get_open_trades()
        print(f"  {'⚠️ ' if open_trades else '✅'} Open positions in DB: {len(open_trades)}")
        for t in open_trades:
            print(f"      • {t['symbol']} {t['side']} qty={t['quantity']} "
                  f"entry=${t['entry_price']:.2f}")

        balance = db.load_balance()
        if balance is None:
            print("  ✅ No persisted balance (fresh start)")
        else:
            print(f"  ✅ Persisted balance: ${balance:,.2f}")
        db.close()
        # A tripped breaker is surfaced but is not a FAILED check — the
        # operator must see it, not be told the system is broken.
        return True
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False


def main() -> None:
    setup_logging()

    print("\n" + "═" * 50)
    print("  🧠 NeuronTrade — Health Check")
    print("═" * 50)

    checks = [
        ("Imports",     check_imports),
        ("Settings",    check_settings),
        ("Pairs",       check_pairs),
        ("Database",    check_database),
        ("Strategies",  check_strategies),
        ("Sentiment",   check_sentiment),
        ("Exchange",    check_exchange),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        results.append(check_fn())

    passed = sum(results)
    total = len(results)
    print(f"\n{'═' * 50}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  🟢 All systems go! Ready to trade.")
    else:
        print("  🔴 Some checks failed — fix issues above.")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
