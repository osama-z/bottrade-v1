# Backtest/Live Parity Redesign — Proposal (awaiting approval)

**Problem (audit V-46/V-43/V-45):** the walk-forward gate certifies a bot
that never trades. Live `get_signal` gates BUYs through six filters (1D/4H
macro trend, RSI+MACD timing, volume, funding rate, order-book imbalance,
HMM regime); backtest `generate_signals` applies none of them. The ML model
is trained on the same DataFrame it is then backtested on (80% in-sample
for the model, 100% for the scaler). So "walk-forward Sharpe > 1.0" — the
go-live criterion — is computed on a different signal stream than
production, on leaked data.

## Design

### 1. One decision function, two data sources

Extract the decision logic from `AICombinedStrategy.get_signal` into a pure
function:

```python
@dataclass(frozen=True)
class MarketContext:
    df: pd.DataFrame              # 1h OHLCV + indicators
    df_1d: pd.DataFrame | None    # daily, indicator-computed
    df_4h: pd.DataFrame | None
    funding_rate: float | None    # None = unavailable (NOT 0.0)
    imbalance: float | None       # None = unavailable (NOT 1.0)
    news_headlines: list[str]

def decide(ctx: MarketContext, models: ModelBundle, cfg: StrategyConfig) -> TradeSignal:
    ...  # every filter gate lives here; NO network, NO filesystem
```

- **Live:** a `LiveContextBuilder` performs the fetches (moved out of the
  strategy) and calls `decide()`.
- **Backtest:** a `ReplayContextBuilder` slices pre-downloaded 1d/4h/1h
  frames as of each candle timestamp (no lookahead: only rows with
  `ts < candle_close`) and calls the same `decide()`.
- Filters whose inputs cannot be replayed (funding, imbalance — we have no
  history) receive `None`; `decide()` must handle `None` as
  "filter unavailable → pass-through", and the backtest report lists them
  as **parity exceptions** with their live pass rates.

This also resolves the god-object finding (V-45): the strategy becomes a
pure function + a context builder, unit-testable without a network.

### 2. Leakage-free walk-forward

- `train/validate` split by time window, anchored: train on
  `[t0, t_k)`, evaluate on `[t_k, t_k+1)`, roll forward. The
  `StandardScaler` is fit inside each training window only.
- `MLPredictor.train` gains a `train_until: datetime` cut; the walk-forward
  runner retrains per fold instead of once on the full frame.
- Regime (HMM) models follow the same fold boundaries.

### 3. Engine parity

- Engine drops its fallback static sizing branch and `open_position_count=0`;
  it maintains the real open-position count and calls the same
  `RiskManager.calculate_position` + affordability check the paper trader
  uses (already Decimal). No fill without cash.
- Fees/slippage already match; keep bar-based fills, add gap-through stop
  fills (`fill = min(open, stop)` for longs) in both engine and paper
  trader in the same change.

### 4. Acceptance criteria

- A replay of N historical candles through `decide()` (backtest path) and
  through the live path with mocked fetchers returns **identical**
  TradeSignals.
- Walk-forward fold boundaries verified by a leakage test: shifting the
  evaluation window into the training set must measurably change results.
- The run report lists parity exceptions (funding/imbalance) explicitly.

## Effort & sequencing

1. `MarketContext` + `decide()` extraction (mechanical, ~1 session) —
   no behavior change, golden-file test against current live path.
2. `ReplayContextBuilder` + engine parity (~1 session).
3. Fold-based retraining in walk-forward (~1 session).

**Decision needed:** approve the `MarketContext`/`decide()` split (step 1)
— it restructures `strategies/ai_combined.py` and touches every entrypoint.
