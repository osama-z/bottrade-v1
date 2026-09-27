# Skills writeup — NeuronTrade (for CV / LinkedIn / interviews)

*Adapt the phrasing below for a resume bullet list, a LinkedIn "Projects"
entry, or as talking points in an interview. It is written to be honest — the
rigor and the "I proved it had no edge" story are the parts that impress
technical interviewers, so keep them.*

---

## One-line summary
Built a ~20k-line production-grade crypto trading system in Python — data
pipeline, backtesting, a Decimal-precise risk engine, and a fault-tolerant
live-execution engine against Binance's API — with rigorous, out-of-sample
quantitative validation.

## Resume bullets (pick 3–5)
- Designed and built a **fault-tolerant order-execution engine** against the
  Binance API with **idempotent retries (clientOrderId), partial-fill handling,
  startup position reconciliation, and min-notional/rate-limit guards** — 27
  execution tests, verified live on testnet.
- Implemented a **Decimal-precise risk engine**: ATR position sizing, a
  restart-persistent circuit breaker, portfolio heat limits, and an out-of-band
  kill switch.
- Built a **backtesting engine with enforced backtest/live parity** (the same
  decision function runs in both, pinned by golden-file tests) and used
  **out-of-sample validation to detect and reject overfitting** — measured that
  the candidate strategy had no edge on unseen data and did not deploy it.
- Tested and **rejected multiple signals** (an XGBoost classifier at ~0.50 AUC,
  regime and funding-rate filters) using proper walk-forward methodology.
- Engineered for production: **460+ tests, CI on every PR**, pinned dependency
  lockfile, structured JSON logging with correlation IDs, an OWASP-mapped
  security pass, and systemd deployment.

## Skills demonstrated
**Software engineering** — Python, clean layered architecture, dependency
injection, extensive testing (pytest), CI/CD (GitHub Actions), git/PR workflow,
SQLite (WAL), concurrency, ZeroMQ.

**Quantitative / data** — pandas/numpy, feature engineering, backtesting,
walk-forward & out-of-sample validation, Sharpe/Sortino/profit-factor, XGBoost,
HMM regime detection; the judgment to distinguish signal from noise and reject
overfit results.

**Trading systems / fintech** — exchange API integration (ccxt), order lifecycle
management, fill reconciliation, idempotency, risk controls, precise monetary
arithmetic.

## Interview talking points
- *"Tell me about a hard bug/edge case."* → The double-order trap: a network
  error after the exchange accepts an order; a naive retry doubles the position.
  Solved with clientOrderId idempotency + an existence check before retry.
- *"Tell me about a time the data changed your conclusion."* → The strategy
  looked profitable in-sample; out-of-sample it was negative on every pair, and
  its only edge was in shorts a spot account can't trade. I reported that and
  refused to trade real money — the discipline mattered more than the result.
- *"How do you ensure correctness with money on the line?"* → Decimal math with
  a documented rounding policy, enforced backtest/live parity via golden tests,
  a persistent manual-reset circuit breaker, and an order audit trail.

## What roles this fits
Python developer · backend engineer · data engineer · junior/associate quant
developer · fintech/trading-systems engineer · freelance automation & data work.
