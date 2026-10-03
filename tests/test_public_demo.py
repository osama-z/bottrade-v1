"""Public-demo execution guards and fresh-copy onboarding checks."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from config.settings import Settings

from execution.live_executor import LiveExecutor, OrderExecutor, RealMoneyRefused
from execution.testnet_trader import TestnetTrader

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [False, "false", "0"])
def test_public_settings_reject_nonpaper_mode(value):
    with pytest.raises(ValidationError, match="Public demo requires PAPER_TRADING=true"):
        Settings(_env_file=None, PAPER_TRADING=value)


def test_public_settings_accept_paper_mode():
    assert Settings(_env_file=None, PAPER_TRADING="true").paper_trading is True


@pytest.mark.parametrize("executor", [LiveExecutor, TestnetTrader])
def test_exchange_executors_refuse_construction(executor, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING", "false")
    with pytest.raises(RealMoneyRefused):
        executor()


def test_placeholder_refuses_order_submission():
    with pytest.raises(RealMoneyRefused):
        OrderExecutor().place_market_order("BTC/USDT", "buy", 1)


def _run(script, tmp_path, *args, bad_database=False):
    # Exercise the shipped example from a separate cwd, without local secrets.
    (tmp_path / ".env").write_bytes((ROOT / ".env.example").read_bytes())
    env = os.environ.copy()
    env.update({
        "BINANCE_API_KEY": "", "BINANCE_API_SECRET": "",
        "GROQ_API_KEY": "", "NEWS_API_KEY": "",
        "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "",
        "DATABASE_URL": f"sqlite:///{tmp_path if bad_database else tmp_path / 'demo.db'}",
        "LOG_FILE": str(tmp_path / "demo.log"),
        "STRATEGY": "trend_following", "PAPER_TRADING": "true",
    })
    # Prevent the caller's import path from hiding a missing entrypoint bootstrap.
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("script", ["run_testnet.py", "testnet_smoke.py"])
def test_disabled_commands_exit_nonzero(script, tmp_path):
    result = _run(script, tmp_path)
    assert result.returncode == 2
    assert "Disabled" in result.stdout


def test_documented_backtest_command_imports_from_another_cwd(tmp_path):
    result = _run("run_backtest.py", tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    assert "--strategy" in result.stdout


def test_offline_health_check_with_example_config(tmp_path):
    result = _run("health_check.py", tmp_path, "--offline")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5/5 checks passed" in result.stdout
    assert "[Exchange]" not in result.stdout
    assert "[Sentiment]" not in result.stdout


def test_health_check_failure_returns_nonzero(tmp_path):
    result = _run("health_check.py", tmp_path, "--offline", bad_database=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Database error" in result.stdout
