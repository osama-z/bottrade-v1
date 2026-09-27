"""Triple-Barrier labelling + Meta-Labelling (López de Prado, AFML ch. 3).

Roadmap Phase 1, Task 1.5 — replace the AUC-0.50 next-candle up/down target with
a path-dependent label that asks a trading question: from each bar, does price
hit a PROFIT barrier, a STOP barrier, or run out of time first?

    label = +1  upper (profit-take) barrier hit first
    label = -1  lower (stop-loss)   barrier hit first
    label =  0  vertical (time)     barrier hit first  (neither horizontal touched)

Barriers are scaled by a volatility estimate so they adapt to regime: the upper
barrier is ``pt_mult * vol`` above entry, the lower ``sl_mult * vol`` below.

Meta-labelling (ch. 3.6): a PRIMARY model proposes a side; a SECONDARY model
predicts whether that specific bet will win (meta-label 1) or lose (0). You take
— and size — a trade only when the meta-label says 1. Meta-labels are computed
here from the primary side vs. the realised triple-barrier outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """EWMA volatility of simple returns — the barrier-width scale.

    Early bars (before enough history) are NaN; the labeller leaves those
    unlabelled rather than guessing a barrier width.
    """
    returns = close.pct_change()
    return returns.ewm(span=span).std()


def triple_barrier_labels(
    close: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 10,
    vol: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Label each bar by which barrier price touches first.

    Args:
        close: price series.
        pt_mult / sl_mult: profit / stop barrier widths in units of ``vol``.
        max_holding: vertical (time) barrier, in bars.
        vol: per-bar volatility (Series), a constant, or None → daily_volatility.

    Returns:
        DataFrame aligned to ``close`` with columns:
            label : +1 / -1 / 0, or NaN when the bar can't be labelled
                    (no vol estimate, or < max_holding bars of future remain).
            ret   : realised return at the touch (NaN when unlabelled).
            t1    : integer position of the touched barrier (−1 when unlabelled).
    """
    n = len(close)
    px = close.to_numpy(dtype=float)
    if vol is None:
        vol = daily_volatility(close)
    if np.isscalar(vol):
        vol_arr = np.full(n, float(vol))
    else:
        vol_arr = pd.Series(vol, index=close.index).to_numpy(dtype=float)

    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    t1 = np.full(n, -1, dtype=int)

    for i in range(n):
        v = vol_arr[i]
        # Can't label without a barrier width or without a full forward horizon.
        if not np.isfinite(v) or v <= 0 or i + max_holding >= n:
            continue
        upper, lower = pt_mult * v, -sl_mult * v
        horizon = i + max_holding
        touched = False
        for j in range(i + 1, horizon + 1):
            r = px[j] / px[i] - 1.0
            if r >= upper:
                labels[i], rets[i], t1[i] = 1.0, r, j
                touched = True
                break
            if r <= lower:
                labels[i], rets[i], t1[i] = -1.0, r, j
                touched = True
                break
        if not touched:                       # vertical (time) barrier
            labels[i], rets[i], t1[i] = 0.0, px[horizon] / px[i] - 1.0, horizon

    return pd.DataFrame({"label": labels, "ret": rets, "t1": t1}, index=close.index)


def meta_labels(side: pd.Series, tb_label: pd.Series) -> pd.Series:
    """Meta-label a proposed side against the realised triple-barrier outcome.

    For each bar the primary model proposes a side (+1 long / −1 short / 0 no
    bet), the meta-label is:
        1  the bet won  — the side matches the barrier that was hit
        0  the bet lost — opposite barrier, or the time barrier (no profit)
      NaN  no bet (side == 0) or the bar is unlabelled

    So a +1 side is a WIN only if the profit barrier (+1) was hit; hitting the
    stop (−1) or timing out (0) is a loss.
    """
    side = pd.Series(side).astype(float)
    tb = pd.Series(tb_label).astype(float)
    out = pd.Series(np.nan, index=side.index)
    trades = (side != 0) & tb.notna()
    out[trades] = (np.sign(side[trades]) == np.sign(tb[trades])).astype(float)
    return out
