"""scripts/validate_funding_carry.py — Task 1.4 gate.

Run the delta-neutral funding-carry strategy through the CPCV rig (Task 1.3) on
REAL Binance data, penalising every trade with VIP0 fees + slippage (Task 1.2),
and report the Deflated Sharpe Ratio. The roadmap gate: funding carry must yield
a POSITIVE Deflated Sharpe after friction before we build on it.

Pipeline
--------
1. Fetch real historical funding rates + 8h spot OHLCV (data/fetcher.py).
2. Align them per 8h funding epoch.
3. Grid-search entry/exit thresholds (these are the "trials" the DSR deflates by).
4. For each trial: FundingCarryStrategy.carry_returns() with an effective per-leg
   cost = VIP0 taker fee + a depth/latency slippage estimate (backtesting/costs.py).
5. Select the best trial; run it through the CPCV rig for the OOS path Sharpe
   distribution; compute the Deflated Sharpe Ratio deflated by the number of
   trials tested.

Usage:
    python scripts/validate_funding_carry.py --pair BTC/USDT --epochs 3000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtesting.cpcv import (  # noqa: E402
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    run_cpcv,
)
from backtesting.costs import CostModel  # noqa: E402
from data.fetcher import DataFetcher  # noqa: E402
from strategies.funding_carry import DAYS_PER_YEAR, EPOCHS_PER_DAY, FundingCarryStrategy  # noqa: E402

EPOCHS_PER_YEAR = EPOCHS_PER_DAY * DAYS_PER_YEAR   # 1095


# ─── Data fetching (paginated) ─────────────────────────────────────────────────
def fetch_funding_history(fetcher: DataFetcher, pair: str, target_epochs: int) -> pd.DataFrame:
    """Page backwards through funding history until ~target_epochs are collected."""
    frames: list[pd.DataFrame] = []
    since_ms = int((datetime.now(timezone.utc) - timedelta(hours=8 * target_epochs)).timestamp() * 1000)
    collected = 0
    while collected < target_epochs:
        batch = fetcher.fetch_funding_rate_history(pair, limit=1000, since=since_ms)
        if batch.empty:
            break
        frames.append(batch)
        collected += len(batch)
        last_ms = int(batch.index[-1].timestamp() * 1000)
        if last_ms <= since_ms:
            break
        since_ms = last_ms + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)   # be polite to the public endpoint
    if not frames:
        return pd.DataFrame({"funding_rate": []})
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def fetch_spot_8h(fetcher: DataFetcher, pair: str, start: datetime) -> pd.DataFrame:
    """Page forward through 8h OHLCV from ``start`` to now."""
    frames: list[pd.DataFrame] = []
    since = start
    now = datetime.now(timezone.utc)
    while since < now:
        batch = fetcher.fetch_ohlcv(pair, timeframe="8h", limit=1000, since=since)
        if batch is None or batch.empty:
            break
        frames.append(batch)
        last = batch.index[-1].to_pydatetime()
        if last <= since:
            break
        since = last + timedelta(hours=8)
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def build_aligned_frame(funding: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """Align funding epochs with the nearest 8h candle (close + volume)."""
    f = funding.sort_index()
    s = spot.sort_index()[["close", "volume"]]
    merged = pd.merge_asof(
        f, s, left_index=True, right_index=True,
        direction="nearest", tolerance=pd.Timedelta("4h"),
    ).dropna()
    return merged


# ─── Cost model → effective per-leg cost (fee + slippage) ──────────────────────
def effective_cost_per_leg(df: pd.DataFrame, notional_usd: float) -> tuple[float, float, float]:
    """VIP0 taker fee + a representative depth/latency slippage per leg."""
    cm = CostModel()
    rng = np.random.default_rng(0)
    slips = [
        cm.slippage_pct_for(order_qty=notional_usd / px, candle_volume=vol, rng=rng)
        for px, vol in zip(df["close"].to_numpy(), df["volume"].to_numpy())
        if px > 0 and vol > 0
    ]
    slippage = float(np.median(slips)) if slips else 0.0
    return cm.taker_fee_pct + slippage, cm.taker_fee_pct, slippage


# ─── Sharpe helpers ────────────────────────────────────────────────────────────
def per_obs_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualized(sharpe_per_epoch: float) -> float:
    return sharpe_per_epoch * np.sqrt(EPOCHS_PER_YEAR)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="CPCV/DSR validation of the funding-carry strategy")
    ap.add_argument("--pair", default="BTC/USDT")
    ap.add_argument("--epochs", type=int, default=3000, help="target funding epochs (~8h each)")
    ap.add_argument("--notional", type=float, default=10_000.0, help="carry notional for slippage sizing")
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--test-groups", type=int, default=2)
    ap.add_argument("--embargo", type=float, default=0.02)
    args = ap.parse_args()

    print(f"\n{'='*70}\n  FUNDING-CARRY VALIDATION (Task 1.4 gate) — {args.pair}\n{'='*70}")

    # ── 1. Fetch real data ────────────────────────────────────────────────────
    logger.info("Fetching funding history + 8h spot for {} …", args.pair)
    fetcher = DataFetcher(public_only=True)
    funding = fetch_funding_history(fetcher, args.pair, args.epochs)
    if funding.empty:
        print("\n❌ No funding history returned (network / symbol / endpoint issue). Aborting.")
        return 1
    spot = fetch_spot_8h(fetcher, args.pair, funding.index[0].to_pydatetime())
    if spot.empty:
        print("\n❌ No spot OHLCV returned. Aborting.")
        return 1

    df = build_aligned_frame(funding, spot)
    n = len(df)
    if n < args.groups * 10:
        print(f"\n❌ Only {n} aligned epochs — too few for CPCV ({args.groups} groups). Aborting.")
        return 1

    apr = df["funding_rate"] * EPOCHS_PER_YEAR
    span_days = (df.index[-1] - df.index[0]).days
    print(f"\nData: {n} funding epochs over ~{span_days} days "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print(f"Funding APR: mean={apr.mean():.1%}  median={apr.median():.1%}  "
          f"p95={apr.quantile(0.95):.1%}  max={apr.max():.1%}")

    # ── 2. Effective per-leg cost (VIP0 fee + slippage) ───────────────────────
    cost_leg, taker, slip = effective_cost_per_leg(df, args.notional)
    print(f"\nFriction per leg: VIP0 taker {taker:.4%} + slippage {slip:.4%} "
          f"= {cost_leg:.4%}  (round-trip both legs = {4*cost_leg:.4%})")

    # ── 3. Grid search over thresholds (these are the DSR "trials") ────────────
    entry_grid = [0.10, 0.15, 0.20, 0.30, 0.50]
    exit_grid = [0.05, 0.10]
    trials = [(e, x) for e in entry_grid for x in exit_grid if x < e]
    n_trials = len(trials)

    print(f"\nGrid: {n_trials} threshold variations (entry×exit). Deflating by n_trials={n_trials}.")
    print(f"{'entry_apr':>10} {'exit_apr':>9} {'trades':>7} {'time_in':>8} "
          f"{'Sharpe/ep':>10} {'Sharpe/yr':>10}")

    results = []
    for entry, exit_ in trials:
        strat = FundingCarryStrategy(entry_apr=entry, exit_apr=exit_)
        r = strat.carry_returns(df, cost_per_leg=cost_leg).to_numpy()
        state = strat.in_carry_state(df)
        n_entries = int(((state) & (~state.shift(1, fill_value=False))).sum())
        sr = per_obs_sharpe(r)
        results.append({"entry": entry, "exit": exit_, "returns": r,
                        "sharpe": sr, "entries": n_entries, "time_in": float(state.mean())})
        print(f"{entry:>10.2f} {exit_:>9.2f} {n_entries:>7d} {state.mean():>7.1%} "
              f"{sr:>10.4f} {annualized(sr):>10.3f}")

    trial_sharpes = np.array([x["sharpe"] for x in results])
    best = max(results, key=lambda x: x["sharpe"])
    print(f"\nBest trial: entry={best['entry']:.2f} exit={best['exit']:.2f} "
          f"→ Sharpe/epoch={best['sharpe']:.4f} (annualised {annualized(best['sharpe']):.3f})")

    # ── 4. CPCV OOS path distribution for the best trial ──────────────────────
    best_returns = best["returns"]
    cv = CombinatorialPurgedCV(
        n_groups=args.groups, n_test_groups=args.test_groups,
        embargo_pct=args.embargo, label_horizon=1,
    )
    cpcv_res = run_cpcv(
        len(best_returns),
        evaluate=lambda tr, te: best_returns[te],
        cv=cv, n_trials=n_trials,
    )
    print(f"\nCPCV: {cpcv_res.n_paths} OOS paths | per-path Sharpe/epoch: "
          f"{[round(s, 4) for s in cpcv_res.path_sharpes]}")
    print(f"      mean={cpcv_res.observed_sharpe:.4f}  std={cpcv_res.sharpe_std:.4f}  "
          f"[{cpcv_res.sharpe_min:.4f}, {cpcv_res.sharpe_max:.4f}]")
    if cpcv_res.sharpe_std == 0:
        print("      (paths coincide — a fixed rule-based strategy doesn't refit, "
              "so CPCV gives one OOS estimate, not a spread. Deflation below uses the "
              "threshold-grid variance instead.)")

    # ── 5. Deflated Sharpe Ratio (deflated by the grid of trials) ─────────────
    sr_variance = float(np.var(trial_sharpes, ddof=1)) if n_trials > 1 else 0.0
    sk = float(_skew(best_returns))
    kt = float(_kurtosis(best_returns, fisher=False))
    dsr = deflated_sharpe_ratio(
        observed_sharpe=best["sharpe"],
        sharpe_variance=sr_variance if sr_variance > 0 else 1e-6,
        n_trials=n_trials,
        n_obs=len(best_returns),
        skew=sk, kurtosis=kt,
    )

    # Effective sample = number of independent carry EPISODES (entries), not 8h
    # epochs: epochs within one episode are the same position, highly
    # autocorrelated, so n_obs overstates the statistical evidence.
    n_episodes = best["entries"]
    MIN_EPISODES = 30       # below this the DSR's n_obs is not credible
    KURT_WARN = 10.0        # carry's fat tail: Sharpe hides crash risk above this

    print(f"\n{'─'*70}")
    print(f"Deflated Sharpe Ratio: {dsr:.4f}   (n_trials={n_trials}, n_obs={len(best_returns)}, "
          f"skew={sk:.2f}, kurt={kt:.1f})")
    print(f"Effective sample: {n_episodes} carry episodes (independent bets), "
          f"not {len(best_returns)} epochs")

    positive_edge = best["sharpe"] > 0
    dsr_pass = dsr > 0.95
    thin_sample = n_episodes < MIN_EPISODES
    fat_tail = kt > KURT_WARN

    warnings = []
    if thin_sample:
        warnings.append(
            f"only {n_episodes} independent carry episodes (< {MIN_EPISODES}) — the "
            f"DSR's n_obs={len(best_returns)} treats correlated in-episode epochs as "
            f"independent, overstating confidence")
    if fat_tail:
        warnings.append(
            f"kurtosis {kt:.0f} (>> normal 3): the classic carry 'smooth-premium-then-"
            f"crash' tail that Sharpe cannot see")
    warnings.append(
        "return model omits margin/collateral cost of the short-perp leg, basis "
        "risk, and liquidation tail — the real risks of the carry")

    print(f"{'='*70}")
    if positive_edge and dsr_pass and not (thin_sample or fat_tail):
        print(f"  ✅ VERDICT: PASS — positive edge survives friction (DSR {dsr:.3f} > 0.95) "
              f"on a credible sample.")
        rc = 0
    elif positive_edge and dsr_pass:
        print(f"  ⚠️  VERDICT: INCONCLUSIVE — mechanically PASS (positive Sharpe, DSR "
              f"{dsr:.3f} > 0.95) but NOT trustworthy yet:")
        for w in warnings:
            print(f"       • {w}")
        rc = 2
    elif positive_edge:
        print(f"  ⚠️  VERDICT: INCONCLUSIVE — positive raw Sharpe but DSR {dsr:.3f} ≤ 0.95 "
              f"after deflating for {n_trials} trials.")
        rc = 2
    else:
        print(f"  ❌ VERDICT: FAIL — no positive edge after fees & slippage "
              f"(best Sharpe/epoch {best['sharpe']:.4f}). Do not proceed.")
        rc = 2
    print(f"{'='*70}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
