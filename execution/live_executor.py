"""live_executor.py — DISABLED in the public Bottrade-v1 paper-only demo.

This public copy cannot place real orders by design. The private repo keeps the
full testnet-guarded LiveExecutor; here only the *names* are preserved so that
imports keep working and failures are loud and obvious.

Any attempt to instantiate or use LiveExecutor raises RealMoneyRefused.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class RealMoneyRefused(RuntimeError):
    """Raised whenever live execution is attempted in the paper-only demo."""


@dataclass(frozen=True)
class OrderResult:
    """Normalized result shape (kept for type-compat with PaperTrader paths)."""

    id: str = ""
    symbol: str = ""
    side: str = ""
    amount: float = 0.0
    filled: float = 0.0
    avg_price: float = 0.0
    cost: float = 0.0
    status: str = "disabled"
    ok: bool = False
    reason: str = "live execution disabled in public paper-only demo"


class OrderExecutor:
    """Protocol placeholder — never implemented in the public demo."""

    def place_market_order(self, *args, **kwargs):  # pragma: no cover
        raise RealMoneyRefused(
            "Live execution is disabled in Bottrade-v1 (paper-only demo). "
            "Use PaperTrader / scripts/run_live.py instead."
        )


class LiveExecutor:
    """Disabled stub — raises on any use."""

    def __init__(self, *args, **kwargs):
        raise RealMoneyRefused(
            "LiveExecutor is disabled in Bottrade-v1 (paper-only demo). "
            "This copy cannot place real orders by design."
        )

    def __getattr__(self, name):  # pragma: no cover
        raise RealMoneyRefused(
            f"LiveExecutor.{name} is disabled in Bottrade-v1 (paper-only demo)."
        )
