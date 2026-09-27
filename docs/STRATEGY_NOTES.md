# Strategy Notes — measurements, conclusions, and the road to smarter trading

*2026-07-23 · covers walk-forward runs, the gate analysis, and the first
strategy-lab screening. Every number here is measured, not assumed; rerun
instructions are included so results can be reproduced after any change.*

---

## 1. What we measured, in order

| Experiment | Result | Lesson |
|---|---|---|
| Walk-forward, AI strategy, 1y | **0 trades** | `RSI<30 AND MACD-cross` fired on 0.03% of candles; 6 ANDed gates ⇒ never trades |
| Gate sensitivity (8,727 candles) | timing gate alone blocks 100% | AND of anti-correlated conditions; HMM says "Choppy" 86% of the time |
| Walk-forward after gate fix, 2y | 2→18 sell signals, still ~0 executed | Test period was a bear market; strategy long-biased AND engine was long-only |
| Engine short support added | 23 trades, Sharpe −0.40 | First real number — but computed at **half risk** (see next) |
| Cross-system parity test | engine risked 1%, live 2% | `RiskManager()` fallback ≠ settings config; fixed via `RiskConfig.from_settings` |
| Corrected walk-forward, 2y | **3 trades** | Breaker deadlock discovered (next row) |
| Strategy lab, first run | 4,756 entries blocked: "circuit breaker is tripped" | Manual-reset-only breaker + no operator in backtests ⇒ every backtest stopped at its first losing streak |
| Engine: simulated operator + `register_closed_trade` parity | trade counts become meaningful | Backtests now measure the strategy, not the time-to-first-streak |
| **Strategy lab, corrected (32 combos, 2y)** | table below | 4h ≫ 1h; one genuine candidate |

**Meta-lesson:** every "the strategy is bad" number so far was actually a
*harness* bug (long-only engine, half-risk sizing, breaker deadlock). The
measurement infrastructure had to be debugged before the strategy could be
judged at all. That is normal — and it is why parity tests now pin each of
these behaviors.

## 2. Strategy-lab results (730 days, 4 pairs × 2 timeframes × 4 strategies)

Market context: avg buy & hold across pairs = **−26.9%** (bear-heavy period).

Top of the table (full CSV: `data/cache/strategy_lab_results.csv`):

| strategy | pair | tf | trades | return | B&H | Sharpe | PF |
|---|---|---|---|---|---|---|---|
| **trend_following** | BTC/USDT | 4h | 52 | **+7.8%** | −4.3% | **0.34** | 1.12 |
| ma_crossover | ETH/USDT | 4h | 87 | +3.6% | −41.8% | 0.19 | 1.03 |
| bollinger_bounce | BNB/USDT | 4h | 60 | −0.1% | −2.6% | 0.06 | 1.27 |

Bottom third: almost entirely **1h rows** (Sharpe −0.5 to −1.8).

### What the table says

1. **The timeframe hypothesis is confirmed with data.** Every top-5 row is
   4h; 1h rows dominate the bottom. At 1h, fees (~0.3%/round trip) eat the
   thin edge and noise triggers stops. **1h should be abandoned as the
   trading timeframe.**
2. **One genuine candidate:** `trend_following` on BTC/USDT 4h — 52 trades
   (statistically meaningful), +7.8% in a market that fell 4.3%, positive
   Sharpe. Modest, but real and human-explainable.
3. **Multiple-comparisons caveat:** best-of-32 is partly luck. The
   candidate must be confirmed on data it hasn't seen (e.g. rerun the lab
   on `--days 365` vs the older half, or on new months as they arrive)
   before it earns a paper run.
4. **Capital preservation works everywhere:** worst strategy row lost 16%
   while the worst market fell 58%. The risk engine — sizing, ATR stops,
   breaker — is doing its job regardless of signal quality.
5. **The AI ensemble is now the benchmark's challenger, not the default.**
   `trend_following` is one idea with zero trained parts. `ai_combined`
   must *beat* it in walk-forward to justify its complexity and API costs.

## 2b. Pair-robustness screen (10 pairs × 4h × 4 strategies, 730d)

Added XRP, DOGE, ADA, LINK, AVAX, LTC to the screening set (avg B&H −27.5%).

**trend_following holds the top 3 rows and wins the fair aggregate** —
mean Sharpe −0.18 vs −0.55/−0.92/−1.01 for the other three strategies,
mean return −4.0% vs the market's −27.5%:

| pair | trades | return | B&H | Sharpe | PF |
|---|---|---|---|---|---|
| ADA/USDT | 42 | **+13.4%** | −59.2% | **0.57** | 1.32 |
| BTC/USDT | 52 | +7.8% | −4.3% | 0.34 | 1.12 |
| XRP/USDT | 45 | +5.7% | +86.8% | 0.28 | 1.27 |
| LINK/USDT | 51 | −0.1% | −37.3% | 0.07 | 0.97 |
| …negative on the other 6 | 24–42 | −10 to −14% | | −0.4 to −0.6 | |

Honest read:
- **Not BTC-only luck**: positive with 40+ trades on 3 independent pairs,
  and it is the best strategy on the *unfiltered* 10-pair aggregate — a
  statistic free of pair cherry-picking.
- **Not a universal edge either**: it loses on 6 of 10 pairs. This is a
  "works in trend-prone conditions" strategy, not magic.
- XRP made +5.7% while its B&H made +86.8% — trend-following gives up
  most of a monster bull run (it exits on every flip). Known cost of the
  style.
- Paper-run candidate set: **ADA, BTC, XRP, LINK** — but picking winners
  from the same sample is selection bias; the paper run itself is the
  out-of-sample test, and dropping to flat-or-worse there must be treated
  as the expected outcome, not a surprise.

## 2c. Paper-run readiness (implemented 2026-07-23)

- **Strategy-agnostic live loop**: `STRATEGY=trend_following` in .env runs
  the candidate through the identical execution path (risk manager,
  breaker, Telegram, audit trail). ai_combined remains the default in
  code but the paper run uses the measured winner.
- **Portfolio heat cap** (`PORTFOLIO_MAX_HEAT`, default 6%): total open
  risk Σ|entry−stop|·qty ÷ equity across ALL positions. ADA/BTC/XRP are
  heavily correlated — four "independent" 2% positions are ≈ one 8% bet;
  the cap bounds that to 3 concurrent stop-distances.
- **Market-structure recorder**: every cycle stores open interest, taker
  buy/sell ratio, and funding per pair (`market_structure` table).
  Binance serves only ~30 days of this history, so the paper run doubles
  as data collection for Step 3's feature tests.

## 2d. Out-of-sample confirmation (2026-07-27) — the edge does NOT survive

Roadmap Step 1, finally run. The 730d numbers in §2b are **in-sample** (the
candidate was *selected* on that window). Splitting each pair's 730d 4h data
into first half (older) vs second half (newer), `trend_following` (adx_min=20):

| pair | 1st half | 2nd half |
|---|---|---|
| ADA | +20.1% (PF 1.73) | **−12.1%** (PF 0.00, 0% win) |
| BTC | +11.9% (PF 1.28) | **−0.9%** (PF 0.97) |
| XRP | +15.2% (PF 1.65) | **−12.4%** (PF 0.00, 0% win) |
| LINK | +32.3% (PF 1.52) | **−12.5%** (PF 0.32) |

**All four pairs positive in the older half, negative in the recent half.** The
headline returns were carried by the first year; in the recent regime the
strategy loses. Cause: trend-following is regime-dependent and the recent
crypto market has been choppier (same mechanism that made SOL a −11.6% loser).
Parameter sweep showed `adx_min=20` sits on a **peak** (BTC: 20→+7.8%, 15→
−10.8%) — a sign the selection is partly luck, not a robust plateau.

**Regime filter tested (200-SMA alignment — take longs only above SMA_200,
shorts only below).** One *principled, a-priori* filter, judged on BOTH halves:

| window | base (mean of 4) | filtered |
|---|---|---|
| 1st | +19.9% | +11.3% |
| 2nd | −9.5% | **−3.6%** |
| full | +6.7% | **+8.5%** |

Verdict: the filter is a **genuine full-period improvement** (rescues LINK
−0.1%→+10.4%, boosts BTC +7.8%→+14.0%, and — the anti-overfitting tell — helps
the *first* half too for BTC/LINK, not only the known-bad second half). But it
**does not make the recent regime profitable** (2nd half still −3.6%; only ADA
flips positive). It cuts trade count ~half. Shipped as the
`trend_following_filtered` registered variant — a **candidate for the next run
/ an A/B**, NOT a mid-run swap of the frozen live strategy.

**Honest takeaway:** expect the live paper run to be **flat-to-negative** in the
current regime — that is now the *pre-registered expectation*, not a surprise.
The strategic open question: trade trend-following only when a trend exists
(regime timing), or accept it sits out choppy markets. Live data decides.

## 2e. Entry-regime diagnostic (2026-07-27) — the edge is ALL shorts

With the backtest engine's `entry_idx` fixed (it previously equalled `exit_idx`,
so per-trade entry conditions were unrecoverable), every `trend_following`
trade was tagged by its state AT ENTRY, pooled across the 4 pairs (n=183):

| split | trades | win% | total return |
|---|---|---|---|
| **long (buy)** | 91 | 32% | **−41.0%** |
| **short (sell)** | 92 | 47% | **+109.5%** |
| ADX 20-25 | 94 | 41% | +63.3% |
| ADX 25-30 | 51 | 31% | −24.8% |
| ADX 30+ | 38 | 45% | +29.9% |
| aligned w/ 200-SMA | 90 | 41% | +38.5% |
| counter to 200-SMA | 93 | 38% | +29.9% |

**The strategy's entire profit is the SHORT side.** Longs are a net loser;
shorts carry everything. Neither ADX level nor 200-SMA alignment cleanly
separates winners (explains why raising `adx_min` and the SMA filter only
helped modestly).

**Decisive implication for real money:** live SPOT cannot short, so a real
spot deployment trades only the losing long side (−41%). The paper trader
simulates shorts freely, so **the paper baseline OVERSTATES real-spot
viability** — it will show short-driven profit that is unrealizable on spot.
Real-money options: (1) futures/margin (shorting works, adds leverage/funding/
liquidation risk), (2) fix the long side (a research project, currently −41%),
or (3) accept this strategy is not spot-viable and reconsider the approach.
This is the single most important measured result to date.

## 2f. Derivatives data — funding-rate filter (2026-07-28) — no robust edge

Roadmap Step 3, first alt-data feature tested properly. Fetched real historical
funding rates (Binance futures, 8h intervals, 730d, 2190 obs/pair) and tested a
principled contrarian filter: block a LONG when funding is elevated positive
(long-crowded) and a SHORT when elevated negative (short-crowded).

Funding distribution is tame (p90 ≈ 0.01%/8h; |funding|>0.05% happens <0.3% of
the time), so a meaningful threshold is ~0.01–0.02%, not 0.05%.

| threshold | mean base | mean +funding |
|---|---|---|
| block >0.01%/8h | +6.7% | +6.3% (worse) |
| block >0.02%/8h | +6.7% | +7.2% (better) |

**Verdict: no robust edge.** The effect is marginal (mean moves <1%), mixed per
pair (LINK −0.1%→+6.2% but ADA 13.4%→3.3–9.1%), drops only 0–4 of ~50 trades,
and — the decisive tell — **flips direction with the threshold** (0.01% hurts,
0.02% helps). A real signal is robust to the exact cutoff; one that flips on a
tiny parameter change is curve-fitting. Not shipped.

This is the expected base rate for a single alt-data signal (most are noise).
The value was the *method*: fetch real data, test both halves + multiple
thresholds + per-pair, and reject honestly. Funding history fetch lives in the
scratchpad experiment; re-add to DataFetcher only if a future test earns it.

## 3. How to rerun everything

```bash
python scripts/strategy_lab.py                       # 4 pairs × 1h,4h × 4 strategies
python scripts/strategy_lab.py --timeframes 4h       # focus the good timeframe
python scripts/walk_forward.py --pair BTC/USDT --total-days 730   # AI strategy
python scripts/component_attribution.py              # after paper trades exist
```

Backtest semantics to know: `breaker_review_hours=24` models the human
who reviews a tripped breaker (live: manual reset only). `n_breaker_trips`
in results shows how often policy halted the strategy — a high count is a
red flag *by itself*.

## 4. Roadmap: how to develop smarter trading (in this order)

Each step is gated on the previous one *measuring* positive. Skipping the
measurement is how projects end up with a 25–55%-weighted LLM reading an
empty headline list for six months.

### Step 1 — Confirm the candidate (now, free)
Switch default timeframe focus to 4h. Validate `trend_following` BTC 4h on
unseen data; try the obvious 1-parameter variations (ADX 15/25, EMA-200
side filter) — *one at a time*. Keep whatever survives out-of-sample.

### Step 2 — 30-day paper run on the winner (the gate to everything else)
Run the confirmed rule-based strategy live-paper on the VPS. This produces
real fills, real audit rows, and a baseline Sharpe that everything later
must beat. It also finally exercises the ops stack (breaker drills,
Telegram, recovery) against live data.

### Step 3 — Add market-structure features, measured (cheap, high value)
The data layer already fetches more than the strategies use. In order of
expected value: **open interest + its delta**, **taker buy/sell ratio**
(complements the book-imbalance filter), funding-rate *extremes* as a
contrarian flag, large-liquidation events. Add ONE as a filter on the
candidate; keep it only if walk-forward improves.

### Step 4 — Re-measure the ML leg on 4h — **DONE, verdict: drop it**
Measured 2026-07-23 on 730d of production 4h data: BTC AUC **0.498**,
ADA AUC **0.495** (1h was 0.52). Coin flip on both timeframes ⇒ this
feature set has no ML-extractable directional edge. The honest move is
dropping/zero-weighting the ML leg, not tuning it. Revisit only with
genuinely new inputs (the recorded market-structure series). Note: model
filenames lack a timeframe suffix, so 4h training overwrites 1h models —
irrelevant now (1h abandoned) but a trap to fix if both ever coexist.

### Step 5 — LLM where LLMs help: event risk, not entry timing
Attribution showed nothing yet justifies per-candle LLM votes. The shape
that plausibly earns its cost: a *risk overlay* — classify fresh headlines
for high-impact events (ETF rulings, exchange failures, macro prints) and
tighten/flatten exposure around them. Structured outputs, cheap model
first (Haiku-class), measured via `component_attribution.py` before any
weight increase. Upgrade the model only after the *role* proves itself.

### Step 6 — Portfolio-level intelligence (later)
Only meaningful once 2+ strategies/pairs run concurrently: correlation-aware
sizing (BTC/ETH move together — two full positions ≈ 2× one risk),
volatility-targeted allocation, and regime switching *between strategies*
(the HMM may be more useful choosing trend-vs-range mode than blocking
trades).

### Explicitly not on the roadmap
- Bigger LLMs / more AI legs before Step 4–5 measurements justify them.
- More indicators on 1h. The timeframe was the problem, not indicator count.
- Live money before the claude.md gate (100 paper trades, PF > 1.3,
  drills passed) — unchanged.

---

*Companion docs: `ANALYSIS_REPORT.md` (audit), `docs/backtest_live_parity_design.md`
(parity), `claude.md` (rules). Architecture guide: claude.ai artifact
"NeuronTrade — Code Review Guide".*
