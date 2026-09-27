# Paper-Run Pre-Registration — success criteria fixed BEFORE results

*Purpose: commit, in writing and dated, to the exact go/no-go thresholds that
decide whether `trend_following` earns a real-money trial — **before** any
paper results exist. This is the guardrail against post-hoc rationalisation
("if I exclude that week it looks fine"). Once the run starts, these numbers
do not move. Changing them requires a dated amendment below, with a reason,
written before looking at the affected metric.*

---

## Run definition (fill in when you start)

| Field | Value |
|---|---|
| Start date (UTC) | `__________` |
| Planned end date (UTC) | start + 30 days |
| Strategy | `trend_following` (ADX_min = 20, symmetric long/short) |
| Timeframe | `4h`, decided on the last **closed** candle |
| Pairs | `ADA/USDT, BTC/USDT, XRP/USDT, LINK/USDT` |
| Starting balance | `10,000` USDT (paper) |
| Risk per trade | 2% · Portfolio heat cap 6% |
| Venue | Binance production market data; paper execution (no real orders) |
| Commit pinned | `__________` (git SHA the run is deployed from) |

## The gate — ALL must hold to advance toward live money

These extend the standing `claude.md` gate (≥100 paper trades, PF > 1.3, drills
passed). Numbers below are the **pre-registered bar**; adjust *before* starting
if you disagree, not after.

| # | Metric | Threshold | Measured by |
|---|---|---|---|
| G1 | Total closed trades | **≥ 40** over the window (enough to be non-anecdotal; the backtest saw 42–52/pair/2y, so 4 pairs × 30d should clear this) | `scripts/paper_status.py` / DB |
| G2 | Profit factor | **> 1.3** | `risk.metrics` (Σwins / \|Σlosses\|) |
| G3 | Sharpe (per-trade) | **> 0.15** | `risk.metrics.sharpe_per_trade` |
| G4 | Max drawdown | **< 12%** (below the 10% soft-pause + margin) | equity curve |
| G5 | Circuit-breaker trips | **≤ 2** operator-review-worthy trips | `risk_events` / breaker history |
| G6 | Live vs backtest sanity | realised win-rate and avg-trade within a **factor of ~2** of the backtest for the same period — a gross divergence means an execution/parity bug, not alpha | `component_attribution.py` + backtest rerun |
| G7 | Ops | kill-switch drill passed, no unhandled crashes, no stale-data trading incidents | logs / drill record |

## Decision rule (write the outcome here at end of run)

- **All of G1–G7 pass** → proceed to the next stage (extended paper run or a
  minimal real-money trial per `claude.md`), not before.
- **Any fail** → do **not** advance. The honest default outcome is "flat or
  worse out-of-sample" (the backtest was best-of-selection); treat a miss as
  expected data, not failure. Options: extend the window, or return to
  Roadmap Step 3 (market-structure features) — never widen the gate to fit.

## Explicitly pre-committed guards against self-deception
- No excluding "bad weeks" after the fact.
- No switching the headline metric after seeing results.
- No adding pairs mid-run to dilute a losing pair.
- No lowering a threshold because the result is "close".

## Amendments (dated, reason, written before viewing the affected metric)

- _none yet_

---

*Companion: `docs/STRATEGY_NOTES.md` (measurement chronology + roadmap),
`claude.md` (standing go-live gate).*
