"""testnet_trader.py — DISABLED in the public Bottrade-v1 paper-only demo.

Testnet/live trading is stripped from this public copy by design.
Use PaperTrader (execution/paper_trader.py) and scripts/run_live.py.
"""

from __future__ import annotations

from execution.live_executor import RealMoneyRefused


class TestnetTrader:
    """Disabled stub — raises on any use."""

    __test__ = False  # not a pytest test class despite the "Test" prefix

    def __init__(self, *args, **kwargs):
        raise RealMoneyRefused(
            "TestnetTrader is disabled in Bottrade-v1 (paper-only demo). "
            "Use PaperTrader instead."
        )

    def __getattr__(self, name):  # pragma: no cover
        raise RealMoneyRefused(
            f"TestnetTrader.{name} is disabled in Bottrade-v1 (paper-only demo)."
        )
