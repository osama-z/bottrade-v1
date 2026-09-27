# Compliance Posture

_NeuronTrade — Roadmap Task 6.2 (Regulatory & Anti-Manipulation Audit)_

This document records how the bot is designed to operate within legal and
exchange-terms bounds. It is a good-faith engineering statement, **not legal
advice**. Operators are responsible for their own jurisdiction's laws and their
exchange account's terms.

---

## 1. Real-money guardrail (current status)

**The bot does not trade real money.** The live-execution path (`LiveExecutor`)
is **hard-guarded to the Binance testnet sandbox** and raises `RealMoneyRefused`
if `BINANCE_TESTNET` is false — it will not place a production order even if a
production key is configured. This is a code invariant, not a setting, verified
by tests.

Rationale: the strategy has **no proven out-of-sample edge** (Phase 1 validation
rejected both candidates), so deploying real capital is a separate, deliberate
decision that must first clear the Phase 5 gates — never a flag flip.

## 2. Anti-spoofing / anti-layering (Task 6.2)

**Spoofing/layering** = placing orders with the intent to cancel them, to
project false liquidity. The bot is designed **not** to do this:

- **Post-only entries rest to fill.** `execute_smart_order` places GTX
  (post-only) maker slices intended to be filled, and **never cancels them**.
- **Cancellations are legitimate cleanup only.** `cancel_order` is invoked from
  exactly two places, both non-manipulative:
  1. **Crash recovery** (`execution/recovery.py`) — on restart, an order left
     unfilled/orphaned by a crash is cancelled so it can't act on stale intent.
  2. **Reconciliation** — cancelling an orphan detected against local state.
- **TWAP slicing** spreads a large order over time to *reduce* market impact —
  the opposite of layering; each slice is a genuine order, capped at ≤25% of the
  top-5 visible depth.

**Defense-in-depth guardrail** (`execution/spoofing_guard.py`): the executor
tracks place→cancel timing. If orders are repeatedly placed and cancelled within
seconds (the spoofing signature: `max_fast_cancels` fast cancels within
`window_seconds`), the guard **trips and halts all further order placement**
until an operator reviews and resets it (mirroring the manual-reset circuit
breaker). This ensures the bot can never even *appear* to spoof, regardless of
future cancel-replace logic.

## 3. Exchange rate limits & API terms

- ccxt is initialised with **`enableRateLimit: True`**, which throttles requests
  to stay within Binance's published limits.
- Market-data and order calls are **batched where possible** (e.g. one
  `fetch_tickers` call for account equity rather than one per asset) to minimise
  request volume.
- Public market data uses a **key-free** client (`DataFetcher(public_only=True)`);
  credentials are only used for account/order endpoints.
- Orders are **idempotent** (unique `clientOrderId`, write-ahead to SQLite,
  startup recovery) so retries after a network drop never duplicate an order —
  avoiding accidental order-flooding.

## 4. KYC / account

- The bot does **not** bypass, automate, or interfere with KYC. Running against a
  **live** exchange account is the operator's responsibility and requires that
  account's KYC to be **fully cleared** under the exchange's terms.
- Testnet (the only path enabled here) uses separate sandbox credentials and
  involves no real funds or KYC.

## 5. Banned / prohibited strategies

The bot does not implement and must not be configured to perform:

- **Spoofing / layering** (see §2 — actively guarded).
- **Wash trading** — it holds at most one position per symbol and never trades
  against itself.
- **Market manipulation / momentum ignition** — TWAP + the ≤25%-of-top-5 depth
  cap explicitly limit market impact.
- **Front-running / abuse of non-public information** — the bot trades only on
  public market data.

## 6. Records & retention

- Every order attempt (accepted or rejected), signal, and risk event is written
  to an **append-only audit trail** with correlation IDs.
- Tax lots use an **immutable, append-only** ledger (Task 6.1): rows are never
  deleted or rewritten; corrections supersede via new rows. **7-year retention.**

## 7. Operator responsibilities

- Confirm **personal algorithmic crypto trading is lawful** in your jurisdiction.
- Confirm your use complies with the **exchange's API terms of service**.
- Keep **KYC cleared** on any live account.
- Rotate and protect API keys; never commit them (`.env` is gitignored).
