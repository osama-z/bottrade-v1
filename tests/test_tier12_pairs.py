"""Tier-12 tests: multi-pair configuration and validation."""

import dataclasses
from pathlib import Path

from config.pairs import (
    PAIR_SPECS,
    SUPPORTED_QUOTE,
    PairSpec,
    get_pair_spec,
    validate_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPairSpecs:
    def test_supported_pairs_present(self):
        for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"):
            assert pair in PAIR_SPECS, pair

    def test_spec_is_immutable(self):
        """Audit finding: the old dict returned a live reference into shared
        module state, so any caller could rewrite config for everyone."""
        spec = get_pair_spec("BTC/USDT")
        with_error = dataclasses.FrozenInstanceError
        try:
            spec.min_volume_usdt = 1.0  # type: ignore[misc]
            raised = None
        except with_error as e:
            raised = e
        assert raised is not None, "PairSpec must be frozen"

    def test_unknown_pair_gets_defaults_not_exception(self):
        spec = get_pair_spec("PEPE/USDT")
        assert isinstance(spec, PairSpec)
        assert spec.symbol == "PEPE/USDT"
        assert spec.min_volume_usdt > 0

    def test_risk_params_are_not_defined_here(self):
        """Risk lives in RiskConfig (single source of truth) — per-pair risk
        fields here would silently contradict it."""
        src = (PROJECT_ROOT / "config" / "pairs.py").read_text()
        for banned in ("risk_per_trade", "stop_loss_pct", "take_profit_pct"):
            assert f"{banned}:" not in src and f'"{banned}"' not in src, banned


class TestValidatePairs:
    def test_valid_list_has_no_problems(self):
        assert validate_pairs(["BTC/USDT", "ETH/USDT", "SOL/USDT"]) == []

    def test_non_usdt_quote_is_flagged(self):
        problems = validate_pairs(["ETH/BTC"])
        assert any("not supported" in p for p in problems)
        assert any(SUPPORTED_QUOTE in p for p in problems)

    def test_malformed_pair_is_flagged(self):
        assert any("BASE/QUOTE" in p for p in validate_pairs(["BTCUSDT"]))

    def test_duplicates_are_flagged(self):
        problems = validate_pairs(["BTC/USDT", "BTC/USDT"])
        assert any("duplicate" in p.lower() for p in problems)

    def test_unspecced_pair_warns_but_is_usable(self):
        problems = validate_pairs(["PEPE/USDT"])
        assert any("PAIR_SPECS" in p for p in problems)
        # only a warning — no exception, and the spec still resolves
        assert get_pair_spec("PEPE/USDT").symbol == "PEPE/USDT"


class TestWiring:
    def test_health_check_reports_model_coverage(self):
        src = (PROJECT_ROOT / "scripts" / "health_check.py").read_text()
        assert "def check_pairs" in src
        assert '("Pairs",       check_pairs)' in src
        assert "train_model.py" in src  # tells the operator how to fix it

    def test_run_live_validates_pairs_at_startup(self):
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "validate_pairs(settings.trading_pairs)" in src

    def test_env_example_documents_usdt_only(self):
        src = (PROJECT_ROOT / ".env.example").read_text()
        assert "USDT quote only" in src
