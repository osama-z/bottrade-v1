# Bottrade-v1 — Paper-Only Trading Demo

> **PUBLIC DEMO — PAPER TRADING ONLY — NOT FOR REAL MONEY.**
>
> This is a **portfolio/educational project** built to demonstrate trading
> infrastructure engineering and quantitative research methodology. It is
> **not** a production system. Live/testnet executors are disabled by design.
> No real money. The private development repo (`osama-z/bottrade`) stays
> private.

A production-grade crypto **trading system** in Python — data pipeline, technical
& AI signal generation, a Decimal-precise risk engine, backtesting with
enforced backtest/live parity, and a fault-tolerant execution engine.

**What this project demonstrates** isn't a bot that prints money — it's
two things employers actually value: the ability to **build robust trading
infrastructure**, and the **quantitative discipline to prove, with data, when a
strategy has no edge and refuse to trade it.** Both are documented below.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env          # no keys needed for paper mode

python scripts/health_check.py           # pre-flight
python scripts/run_live.py               # paper trading (simulated)
python scripts/paper_status.py           # one-page status snapshot
python scripts/strategy_lab.py           # backtest strategies × pairs × timeframes
```

## Running Tests

```bash
pip install -r requirements.lock
pytest tests/ -q          # 460+ tests, all offline/hermetic
pytest tests/ -q --cov    # with coverage report
```

## Highlights

### Fault-tolerant execution engine
The order path handles the messy realities that separate a demo from a system
you'd trust with orders — **stripped/disabled in this public demo**
(`execution/live_executor.py`, `execution/testnet_trader.py` are disabled stubs
here; full versions live in the private repo):

- **Idempotent retries** — a network error can occur *after* the exchange
  accepts an order; every order carries a `clientOrderId` and retries check
  whether it already landed, so a blip can never double-place a position.
- **Partial-fill handling** — a market sell that under-fills books PnL on the
  filled portion and keeps the remainder open.
- **Startup reconciliation** — on restart, DB positions are verified against
  real exchange holdings; phantom positions are orphaned rather than traded.
- **Min-notional guard, backoff retry on transient errors, graceful failure** —
  a bad order returns a result, never crashes the loop.
- **Append-only order audit log** — every attempt (accepted or rejected) persisted.
- **Hard safety guard** — the executor refuses to run against a production
  account; it is testnet-only by construction.

### Rigorous, self-skeptical quant research
Every strategy claim here is *measured*, and the headline finding is a
**negative** one, honestly reported (`docs/STRATEGY_NOTES.md`):

- **Out-of-sample validation** caught overfitting: the candidate strategy was
  positive in-sample but **negative out-of-sample on all pairs** — the
  in-sample returns were a selection artifact.
- **Multiple signals tested and rejected** — an XGBoost model (AUC ~0.50, a
  coin flip), a 200-SMA regime filter (marginal), and a funding-rate filter
  (flipped sign with the threshold = curve-fitting). None shipped.
- **A decisive structural finding:** the strategy's only edge was in *shorts*,
  which a spot account can't trade — so it isn't spot-viable, documented before
  a cent of real money was risked.

*The value is the method: form a hypothesis, test it on data it wasn't chosen
on, and reject what doesn't hold.*

### Risk engineering
- **Decimal** arithmetic throughout the money path (documented rounding policy:
  quantities round down, losses round up).
- **Circuit breaker** (daily-loss / drawdown / consecutive-loss), persistent
  across restarts, **manual-reset-only** — never auto-resumes.
- **Portfolio heat cap** (Σ stop-distance risk across correlated positions).
- **Out-of-band kill switch** (flag file + `SIGUSR1`) independent of the app.
- **Enforced backtest/live parity** — the same decision function runs in
  backtest and live, pinned by golden-file tests.

---

## Architecture

**See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** for the layer diagram,
a start-to-finish trace of one trade cycle (with a flow chart), and the
extension points for adding strategies, indicators, data sources, or venues.

```
data/         market-data fetch (ccxt), preprocessing, order-book sync
indicators/   technical indicators, feature engineering
strategies/   pluggable strategies via a registry (trend_following, ai_combined, ...)
ai/           XGBoost predictor, HMM regime detector, LLM/sentiment (measured, mostly dropped)
risk/         RiskManager, position sizing, circuit breaker, metrics, walk-forward
backtesting/  parity-verified engine (shorts, gap-through fills, simulated operator)
execution/    PaperTrader (simulation) · LiveExecutor + TestnetTrader (disabled in demo)
storage/      SQLite persistence (WAL), trades / signals / orders / audit tables
core/         ZMQ pub/sub for decoupled intelligence<->execution, candle timing
notifications/ Telegram control + alerts (authorized-chat only)
scripts/      run_live (paper), backtests, health_check, paper_status, strategy_lab
```

## Tech Stack

Python 3.12 · pandas / numpy · ccxt · XGBoost · hmmlearn · pandas-ta · SQLite ·
ZeroMQ · APScheduler · pytest · ruff · GitHub Actions

## Engineering Practices

- **460+ tests** including cross-system parity and no-lookahead pins; **CI on
  every PR** installs from a pinned lockfile and imports the live entrypoint.
- Single-source config (frozen pydantic settings; secrets redacted from `repr`).
- Correlation IDs end-to-end; structured JSON logs with rotation.
- Security pass (OWASP-mapped): ZMQ loopback guard, dependency pinning,
  systemd sandboxing, no secrets in logs.

---

## Disclaimer

**This is a portfolio/educational project — not for real money.** It has
**no proven trading edge** (see `docs/STRATEGY_NOTES.md`) and is **not**
financial advice. Do not trade real money with it. Live execution is
deliberately restricted to the testnet sandbox in the private repo.
