"""Per-pair metadata.

WHICH pairs the bot trades comes from ``settings.trading_pairs``
(``TRADING_PAIRS`` in .env) — that is the single source of truth. This
module holds only metadata *about* pairs that the env var can't express.

Risk parameters deliberately do NOT live here. Position sizing, stop
distance, and the loss limits all come from ``RiskConfig``, built from
``Settings`` (claude.md: risk limits exist in exactly one place). The
previous per-pair ``risk_per_trade`` / ``stop_loss_pct`` fields here were
dead code that contradicted the real risk config — a trap for anyone who
edited them expecting the bot to obey.
"""

from __future__ import annotations

from dataclasses import dataclass

# Quote currency the whole system is denominated in. Equity, the risk
# limits, and the circuit breaker are all USDT-valued, so a non-USDT
# quote (e.g. ETH/BTC) would need multi-currency equity accounting that
# does not exist yet — validated at startup rather than failing subtly.
SUPPORTED_QUOTE = "USDT"


@dataclass(frozen=True)
class PairSpec:
    """Immutable per-pair metadata (frozen: no caller can mutate shared config)."""

    symbol: str
    min_volume_usdt: float   # sanity floor for 24h volume
    notes: str = ""


PAIR_SPECS: dict[str, PairSpec] = {
    "BTC/USDT": PairSpec("BTC/USDT", 1_000_000, "Deepest book; the reference pair"),
    "ETH/USDT": PairSpec("ETH/USDT", 500_000, "Deep book, high correlation with BTC"),
    "SOL/USDT": PairSpec("SOL/USDT", 250_000, "Liquid major; higher volatility than BTC/ETH"),
    "BNB/USDT": PairSpec("BNB/USDT", 100_000, "Liquid; moves partly on exchange-specific news"),
    # Screening tier — in the strategy lab's robustness set; promote to
    # TRADING_PAIRS only if the candidate strategy holds up on them.
    "XRP/USDT": PairSpec("XRP/USDT", 250_000, "Liquid; strong idiosyncratic (legal/news) moves"),
    "DOGE/USDT": PairSpec("DOGE/USDT", 250_000, "Liquid; sentiment-driven, trend-prone bursts"),
    "ADA/USDT": PairSpec("ADA/USDT", 100_000, "Liquid major"),
    "LINK/USDT": PairSpec("LINK/USDT", 100_000, "Liquid; DeFi bellwether"),
    "AVAX/USDT": PairSpec("AVAX/USDT", 100_000, "Liquid; higher beta"),
    "LTC/USDT": PairSpec("LTC/USDT", 100_000, "Old major; lower beta, steady book"),
}

_DEFAULT_MIN_VOLUME = 100_000.0


def get_pair_spec(pair: str) -> PairSpec:
    """Metadata for a pair. Unknown pairs get conservative defaults rather
    than raising — an operator adding a pair to TRADING_PAIRS should not
    crash the bot; ``validate_pairs`` surfaces it at startup instead."""
    spec = PAIR_SPECS.get(pair)
    if spec is not None:
        return spec
    return PairSpec(symbol=pair, min_volume_usdt=_DEFAULT_MIN_VOLUME,
                    notes="No spec entry — using conservative defaults")


def validate_pairs(pairs: list[str]) -> list[str]:
    """Return human-readable problems with a configured pair list.

    Empty list means the configuration is sound. Checks the invariants the
    rest of the system assumes rather than merely that the string parses.
    """
    problems: list[str] = []
    for pair in pairs:
        if "/" not in pair:
            problems.append(f"{pair!r}: not in BASE/QUOTE form (e.g. BTC/USDT)")
            continue
        base, quote = pair.split("/", 1)
        if quote != SUPPORTED_QUOTE:
            problems.append(
                f"{pair!r}: quote currency {quote!r} is not supported — equity, "
                f"risk limits, and the circuit breaker are all {SUPPORTED_QUOTE}-denominated"
            )
        if not base:
            problems.append(f"{pair!r}: empty base currency")
        if pair not in PAIR_SPECS:
            problems.append(
                f"{pair!r}: no entry in PAIR_SPECS — trading with default metadata"
            )
    if len(set(pairs)) != len(pairs):
        problems.append("duplicate pairs in TRADING_PAIRS")
    return problems
