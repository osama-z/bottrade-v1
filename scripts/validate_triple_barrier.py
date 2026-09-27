"""scripts/validate_triple_barrier.py — validate the Task 1.5 ML edge.

Train the triple-barrier + meta-labelling model (ai/ml_predictor.py) on REAL
OHLCV, run it through the CPCV rig (Task 1.3) — REFITTING per split so the OOS
Sharpe distribution is genuine — penalise every bet with VIP0 fees + slippage
(Task 1.2), trade ONLY when the meta-label is 1, and report the Deflated Sharpe.

Because the model refits on each CPCV training set, the reconstructed paths
differ (unlike the rule-based strategies), so this produces a real distribution
of out-of-sample Sharpes.

Usage:
    python scripts/validate_triple_barrier.py --pair BTC/USDT --timeframe 1h --bars 5000
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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtesting.cpcv import CombinatorialPurgedCV, deflated_sharpe_ratio, generate_paths  # noqa: E402
from backtesting.costs import CostModel  # noqa: E402
from data.fetcher import DataFetcher  # noqa: E402
from data.preprocessor import DataPreprocessor  # noqa: E402
from indicators.features import FeatureEngineer  # noqa: E402
from indicators.technical import TechnicalIndicators  # noqa: E402
from ai.ml_predictor import TripleBarrierPredictor  # noqa: E402
from ai.triple_barrier import triple_barrier_labels  # noqa: E402

_TF_HOURS = {"1h": 1, "2h": 2, "4h": 4, "1d": 24}


def fetch_ohlcv_paginated(fetcher: DataFetcher, pair: str, timeframe: str, target_bars: int) -> pd.DataFrame:
    step = timedelta(hours=_TF_HOURS.get(timeframe, 1))
    since = datetime.now(timezone.utc) - step * target_bars
    frames, now = [], datetime.now(timezone.utc)
    while since < now and sum(len(f) for f in frames) < target_bars:
        batch = fetcher.fetch_ohlcv(pair, timeframe=timeframe, limit=1000, since=since)
        if batch is None or batch.empty:
            break
        frames.append(batch)
        last = batch.index[-1].to_pydatetime()
        if last <= since:
            break
        since = last + step
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="first")].sort_index()


def per_trade_sharpe(path: np.ndarray) -> float:
    """Sharpe over the ACTUAL bets in a path (non-zero returns) — flat bars,
    which carry no risk and no return, don't count as observations."""
    bets = path[path != 0.0]
    if bets.size < 2:
        return 0.0
    sd = bets.std(ddof=1)
    return float(bets.mean() / sd) if sd > 0 else 0.0


def annualized(sr_per_trade: float, trades_per_year: float) -> float:
    return sr_per_trade * np.sqrt(max(trades_per_year, 1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description="CPCV/DSR validation of the triple-barrier ML model")
    ap.add_argument("--pair", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--notional", type=float, default=10_000.0)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--test-groups", type=int, default=2)
    ap.add_argument("--embargo", type=float, default=0.02)
    args = ap.parse_args()

    print(f"\n{'='*72}\n  TRIPLE-BARRIER ML VALIDATION (Task 1.5) — {args.pair} {args.timeframe}\n{'='*72}")

    # ── 1. Real data → features + aligned close ───────────────────────────────
    logger.info("Fetching {} {} bars for {} …", args.bars, args.timeframe, args.pair)
    fetcher = DataFetcher(public_only=True)
    raw = fetch_ohlcv_paginated(fetcher, args.pair, args.timeframe, args.bars)
    if raw.empty or len(raw) < 800:
        print(f"\n❌ Only {len(raw)} candles fetched — need ≥800. Network/symbol issue. Aborting.")
        return 1
    ind = TechnicalIndicators().compute_all(DataPreprocessor().process(raw))
    X, _ = FeatureEngineer().build_features(ind, target_periods=1)   # features only; ignore old target
    close_full = ind["close"]
    span_days = (raw.index[-1] - raw.index[0]).days
    print(f"\nData: {len(raw)} candles over ~{span_days} days ({raw.index[0].date()} → {raw.index[-1].date()})")
    print(f"Features: {X.shape[1]} columns, {len(X)} rows after cleaning")

    # ── 2. Effective per-fill cost = VIP0 taker + slippage ────────────────────
    cm = CostModel()
    rng = np.random.default_rng(0)
    slips = [cm.slippage_pct_for(order_qty=args.notional / px, candle_volume=vol, rng=rng)
             for px, vol in zip(ind["close"].to_numpy(), ind["volume"].to_numpy()) if px > 0 and vol > 0]
    cost_per_fill = cm.taker_fee_pct + (float(np.median(slips)) if slips else 0.0)
    print(f"\nFriction: VIP0 taker {cm.taker_fee_pct:.4%} + slippage {cost_per_fill-cm.taker_fee_pct:.4%} "
          f"= {cost_per_fill:.4%} per fill  (round-trip {2*cost_per_fill:.4%})")

    trades_per_year = _TF_HOURS[args.timeframe] and (365 * 24 / _TF_HOURS[args.timeframe])

    # ── 3. Grid of barrier configs = the DSR "trials" ─────────────────────────
    configs = [
        {"pt": 2.0, "sl": 2.0, "h": 12},
        {"pt": 1.5, "sl": 1.5, "h": 8},
        {"pt": 3.0, "sl": 2.0, "h": 24},
    ]
    cv = CombinatorialPurgedCV(n_groups=args.groups, n_test_groups=args.test_groups,
                               embargo_pct=args.embargo, label_horizon=max(c["h"] for c in configs))
    print(f"\nCPCV: {args.groups} groups, {args.test_groups} test → {cv.num_paths} OOS paths per config; "
          f"deflating by n_trials={len(configs)} barrier configs.\n")
    print(f"{'pt':>4} {'sl':>4} {'hold':>5} {'events':>7} {'paths_meanSR/trade':>19} {'annualised':>11}")

    results = []
    for c in configs:
        tb = triple_barrier_labels(close_full, c["pt"], c["sl"], c["h"]).reindex(X.index)
        keep = tb["label"].notna() & X.notna().all(axis=1)
        Xev = X[keep]
        yev = tb.loc[keep, "label"].astype(int)
        retev = tb.loc[keep, "ret"].to_numpy()
        n = len(Xev)

        def evaluate(tr, te, Xev=Xev, yev=yev, retev=retev):
            try:
                model = TripleBarrierPredictor(pair="_cpcv")
                model.fit_labeled(Xev.iloc[tr], yev.iloc[tr], save=False)
                pb = model.predict_batch(Xev.iloc[te])
                signed = pb["signed_size"].to_numpy()
            except Exception:
                signed = np.zeros(len(te))
            bet = signed * retev[te] - np.abs(signed) * (2 * cost_per_fill)
            return bet

        paths = generate_paths(n, cv, evaluate)
        path_sr = [per_trade_sharpe(p) for p in paths]
        mean_sr = float(np.mean(path_sr))
        results.append({**c, "n": n, "paths": paths, "path_sr": path_sr, "mean_sr": mean_sr})
        print(f"{c['pt']:>4.1f} {c['sl']:>4.1f} {c['h']:>5d} {n:>7d} "
              f"{mean_sr:>19.4f} {annualized(mean_sr, trades_per_year):>11.2f}")

    # ── 4. Select best config; genuine OOS path distribution ──────────────────
    best = max(results, key=lambda r: r["mean_sr"])
    all_bets = np.concatenate([p[p != 0.0] for p in best["paths"]]) if best["paths"] else np.array([])
    n_bets = int(all_bets.size / max(1, cv.num_paths))     # bets per path ≈ effective independent trades
    print(f"\nBest config: pt={best['pt']} sl={best['sl']} hold={best['h']}")
    print(f"CPCV OOS path Sharpes/trade: {[round(s, 3) for s in best['path_sr']]}")
    print(f"  mean={best['mean_sr']:.4f}  std={np.std(best['path_sr']):.4f}  "
          f"[{min(best['path_sr']):.4f}, {max(best['path_sr']):.4f}]   ← a REAL distribution (model refits per split)")

    # ── 5. Deflated Sharpe Ratio ──────────────────────────────────────────────
    trial_sr = np.array([r["mean_sr"] for r in results])
    sr_var = float(np.var(trial_sr, ddof=1)) if len(trial_sr) > 1 else 1e-6
    sk = float(_skew(all_bets)) if all_bets.size > 2 else 0.0
    kt = float(_kurtosis(all_bets, fisher=False)) if all_bets.size > 3 else 3.0
    dsr = deflated_sharpe_ratio(
        observed_sharpe=best["mean_sr"], sharpe_variance=max(sr_var, 1e-6),
        n_trials=len(configs), n_obs=max(n_bets, 2), skew=sk, kurtosis=kt,
    )

    print(f"\n{'─'*72}")
    print(f"Deflated Sharpe Ratio: {dsr:.4f}   (n_trials={len(configs)}, "
          f"effective trades/path≈{n_bets}, skew={sk:.2f}, kurt={kt:.1f})")

    positive = best["mean_sr"] > 0
    dsr_pass = dsr > 0.95
    thin = n_bets < 50
    fat = kt > 10.0
    print(f"{'='*72}")
    if positive and dsr_pass and not (thin or fat):
        print(f"  ✅ VERDICT: PASS — ML edge survives friction + multiple testing "
              f"(DSR {dsr:.3f} > 0.95) on a credible sample.")
        rc = 0
    elif positive and dsr_pass:
        print("  ⚠️  VERDICT: INCONCLUSIVE — mechanically PASS but not trustworthy:")
        if thin:
            print(f"       • only ~{n_bets} effective trades/path (< 50) — evidence too thin")
        if fat:
            print(f"       • kurtosis {kt:.0f} — fat-tailed returns Sharpe can't see")
        rc = 2
    elif positive:
        print(f"  ⚠️  VERDICT: INCONCLUSIVE — positive OOS Sharpe ({best['mean_sr']:.3f}) but "
              f"DSR {dsr:.3f} ≤ 0.95 after deflating for {len(configs)} trials.")
        rc = 2
    else:
        print(f"  ❌ VERDICT: FAIL — no positive OOS edge after fees & slippage "
              f"(best mean Sharpe/trade {best['mean_sr']:.4f}). The ML target is not tradeable as-is.")
        rc = 2
    print(f"{'='*72}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
