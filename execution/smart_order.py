"""Smart-order primitives (Roadmap Task 3.1) — pure, exchange-free helpers.

These compute the *decisions* of smart execution so they can be unit-tested
without any network: the passive post-only price, the market-impact depth cap,
and the TWAP slice plan. `execution/live_executor.py` calls them and turns the
decisions into real (testnet) orders.

Order-book format is ccxt's: {"bids": [[price, qty], ...], "asks": [...]},
bids descending / asks ascending by price.
"""

from __future__ import annotations

import math


def top_n_liquidity(levels, n: int = 5) -> float:
    """Sum of base quantities across the top-``n`` [price, qty] levels."""
    return float(sum(float(q) for _, q in list(levels)[:n]))


def depth_capped_quantity(
    quantity: float,
    side: str,
    order_book: dict,
    *,
    levels: int = 5,
    pct: float = 0.25,
) -> float:
    """Cap ``quantity`` at ``pct`` of the top-``levels`` liquidity you'd trade
    against (asks for a buy, bids for a sell) — a market-impact guard so a single
    order never eats more than a quarter of visible depth. 0 when no depth."""
    book_side = order_book.get("asks" if side == "buy" else "bids", []) or []
    cap = top_n_liquidity(book_side, levels) * pct
    if cap <= 0:
        return 0.0
    return min(float(quantity), cap)


def post_only_price(
    side: str,
    best_bid: float,
    best_ask: float,
    *,
    requested_price: float | None = None,
) -> tuple[float, bool]:
    """Passive maker price that will NOT cross the book (post-only / GTX).

    A buy joins the bid, a sell joins the ask. If a ``requested_price`` would
    cross (buy ≥ best_ask, sell ≤ best_bid) it is REJECTED and re-priced to the
    passive side. Returns (price, was_repriced).
    """
    if side == "buy":
        crosses = requested_price is not None and requested_price >= best_ask
        if requested_price is None or crosses:
            price = best_bid
        else:
            price = min(requested_price, best_bid)   # never above the bid
        return float(price), bool(crosses)
    if side == "sell":
        crosses = requested_price is not None and requested_price <= best_bid
        if requested_price is None or crosses:
            price = best_ask
        else:
            price = max(requested_price, best_ask)    # never below the ask
        return float(price), bool(crosses)
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def twap_plan(
    total_qty: float,
    notional: float,
    *,
    threshold_usd: float = 500.0,
    max_slices: int = 10,
    slice_seconds: float = 60.0,
) -> tuple[list[float], float]:
    """Split an order into equal TWAP chunks when it exceeds ``threshold_usd``.

    At/below the threshold → a single chunk, no delay. Above it → ``n`` equal
    chunks, ``n = min(max_slices, ceil(notional / threshold_usd))`` (so each
    chunk is ≈ threshold-sized), with ``slice_seconds`` between them — total
    duration ≤ ``max_slices · slice_seconds`` (10 min at the defaults). Returns
    (chunks, delay_seconds); the last chunk absorbs any rounding remainder.
    """
    total_qty = float(total_qty)
    if total_qty <= 0:
        return [], 0.0
    if notional <= threshold_usd:
        return [total_qty], 0.0
    n = min(max_slices, max(2, math.ceil(notional / threshold_usd)))
    base = total_qty / n
    chunks = [base] * (n - 1)
    chunks.append(total_qty - base * (n - 1))
    return chunks, float(slice_seconds)
