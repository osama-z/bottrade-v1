# NeuronTrade — Project Rules

## Role
Act as a Quantitative Trading Systems Engineer and Principal Python Engineer. This is a production-grade crypto trading bot built incrementally across defined stages, with rigorous testing at each stage. When auditing or reviewing, act as a Lead Quantitative Auditor: find bugs, verify correctness, don't just describe what the code does.

## Global Rules
- Build and modify one stage/area at a time. Do not start the next stage, or fix a new category of bug, until the current one is confirmed working.
- Every new or changed component needs a test or validation step before being "done" (unit test, backtest, or comparison against a known-good reference implementation).
- Flag any assumption about data format, timing, or state — never assume silently.
- Prioritize correctness and safety over cleverness. If a technique (incremental indicators, HMM regime detection) has known failure modes, explain them BEFORE implementing or approving, not after.
- No component that can affect live order execution gets built or modified before its corresponding risk control exists and is verified.
- Database: SQLite. The path comes from configuration (`settings.database_path`, derived from `DATABASE_URL`), resolved to an absolute path anchored at the project root. No component may declare its own path default; `sqlite3.connect` on the trading DB may appear only in `storage/trade_logger.py` and the breaker store. CWD-relative DB paths are forbidden (launching from a different directory must not silently create a fresh, empty state database).
- Static contract gate: CI must run a static attribute/arity check (mypy/pyright, or at minimum a smoke test that imports and constructs every entrypoint's object graph) on `risk/`, `execution/`, `core/`, `scripts/run_*`. Any unresolved attribute in code that can reach order execution is a merge blocker. (Rationale: seven phantom references shipped across three entrypoints, all hidden by broad `except Exception`.)
- For any audit, review, or bug-finding task: cite exact file and line number for every finding. If you can't locate the relevant code, say so explicitly — never infer that something exists.
- Don't rewrite working code from scratch during an audit. Fix only what's broken; show diffs, not full rewrites, unless a full rewrite is explicitly requested.

### Numerical Precision Policy
- Prices, quantities, and P&L: `Decimal` — always. No exceptions for order construction, risk limit checks, or database persistence.
- Risk limit comparisons: `Decimal` — explicitly round in favor of caution (round position size down, round calculated losses up).
- Indicator calculations: `float` — acceptable and standard for EMA/RSI/ATR math, but cast back to `Decimal` when passed to the Risk Manager.
- Flag any use of `float` in the order execution or risk check path as a blocking bug.
- `Decimal` values are constructed from strings (`Decimal(str(x))`), never directly from floats; `quantize()` always specifies an explicit rounding mode.
- Exchange quantization: every order quantity is quantized DOWN to the symbol's `stepSize`, every price to `tickSize`, and orders below `minNotional` rejected — using exchange filters fetched at boot (required before live deployment; paper mode quantizes to 8 dp).
- Position sizing must budget the ALL-IN loss at stop: price risk plus round-trip commission plus modeled slippage.
- Short selling: until margin/borrow mechanics are fully modeled, short cash-flow accounting (collateral reserved at entry; buy-back cost including exit fee at close) must be unit-tested for the invariant `wallet Δ == recorded PnL − entry fee` on both sides.

### State Recovery & Reconciliation
- No component may assume its in-memory state is the source of truth upon restart.
- Upon boot, the system must reconcile internal position state with the exchange's actual position via REST API before trading is enabled.
- Open orders whose state is ambiguous (submitted but no ack received due to network failure) must be queried and explicitly resolved (canceled or confirmed) before new trading begins.
- Critical state transitions (order submitted, order filled, circuit breaker tripped) must be written to the SQLite database synchronously (or via WAL) before the action is considered complete. Persist-then-act ordering: the DB write happens BEFORE the in-memory mutation it records.
- Startup recovery sequence (every entrypoint, before any trading loop is scheduled): (1) open DB and verify schema, (2) load breaker state, (3) load open positions, (4) restore cash/equity from persisted state (paper) or exchange REST snapshot (live), (5) resolve in-flight orders, (6) log recovery complete.
- Order identity: every order gets a UUID client order ID persisted with status `SUBMITTED` before the network call, transitioned to `ACKED/FILLED/REJECTED/UNKNOWN` after; retries reuse the same ID. (Required before live; the paper `trades` schema must grow these states before a live executor is built.)
- Non-SQLite state (models, flags) is written atomically (temp file + `os.replace`); multi-file model artifacts are bundled and versioned (schema version, training window, feature hash); loads hard-fail on mismatch — never degrade to unscaled inputs or silent neutral output.
- Crash-recovery testing: each stage gate includes a kill-and-restart test — SIGKILL at critical transitions, assert restart converges with no duplicated or lost balance/positions.
- SQLite access: exactly one connection owner per component, `journal_mode=WAL`, `busy_timeout ≥ 10s`. Circuit-breaker transitions are ALSO appended to `circuit_breaker_history` (append-only); a locked DB must never crash the bot uncaught, and a failed breaker-trip write must fail the trading action, not be swallowed.

### Logging & Observability
- All logging must be structured (JSON format) with a correlation ID for every signal and order lifecycle.
- Exactly one logging setup (`config/logging_config.setup_logging`) may install handlers; entrypoints must not call `logger.remove()/add()` themselves. All timestamps UTC (`!UTC` + explicit `Z` on human-readable sinks; tz-aware ISO 8601 in JSON sinks), matching the SQLite audit trail.
- A complete trade audit trail is mandatory: every order submission, rejection, risk check pass/fail, sizing failure, scale-out, pause/resume, kill-switch firing, and circuit breaker state transition must be persisted to SQLite (`trades`, `signals`, `risk_events`, `circuit_breaker_history`) with a UTC timestamp.
- Secrets: API keys/tokens never appear in log output, exception messages, or outbound notifications. Secrets go in HTTP headers, not URL query params, wherever supported; exception strings derived from HTTP requests are scrubbed of query strings before logging or re-raising.
- Alerting reliability: operator alerts for breaker trips/drawdown/errors get at least one retry with a plain-text fallback (Markdown parse failures are content-dependent); a failed critical alert is itself logged at ERROR.
- Latency must be measured at pipeline boundaries (WS receive → indicator update → signal generation → risk check → order submission), with thresholds as named config keys (`latency_warn_ms`), not literals. Clock sanity: compare local time against the exchange server-time endpoint at startup; sustained drift beyond a configured threshold flags data stale.

### Data Validation & Boundary Checks
- Incoming WebSocket data is untrusted. The system must explicitly handle and reject:
  - NaN, zero, or negative prices/quantities.
  - Out-of-order sequence numbers (gaps must trigger a snapshot resync).
- Order-book sync must implement the exchange's documented algorithm (Binance: buffer diffs, fetch snapshot, drop events with `u ≤ lastUpdateId`, first applied event brackets `lastUpdateId`, every subsequent event's `U == previous u + 1` — otherwise discard the book and re-snapshot). The consumed schema must include the first-update-ID field.
- A "stale data" timer must exist: if no ticks are received for N seconds (`settings.stale_data_seconds`), the system must enter a fail-safe state (halt trading, flag stale data) until the feed recovers.
- Fetch retry standard: bounded retries with exponential backoff and jitter, honoring 429/`Retry-After`. Exhausted retries raise a typed `DataFetchError` — NEVER return a fabricated neutral value (0.0 funding, 1.0 imbalance, "Neutral" sentiment). Data-quality failures must be distinguishable from neutral market states.
- Malformed message frames are discarded individually with a warning — one bad frame must never poison a batch or crash a consumer.

### Kill Switch & Emergency Flatten
- The circuit breaker halts new trading. A separate Kill Switch capability must exist to flatten current positions and cancel working orders. (Implemented: `execution/kill_switch.py` — `KILL_SWITCH` flag file in project root, or `SIGUSR1`.)
- The Kill Switch must be triggerable out-of-band (e.g., via a local file flag, OS signal, or HTTP endpoint) independent of the ZMQ intelligence core.
- Firing order: halt FIRST (trip the persistent breaker + pause), then flatten — no new entry may interleave with the flatten. The flag file is not auto-deleted: a restart with the flag present fires again; trading resumes only after the operator removes the file AND manually resets the breaker.
- Command-channel authentication: any out-of-band control that can pause, resume, flatten, or force-close (Telegram, HTTP) must verify the sender against a configured allowlist; rejected commands are logged with sender identity.
- Kill-switch drill: before paper trading begins, a documented drill must pass on the running system — trigger the flatten while positions are open and the intelligence core is deliberately hung; verify all positions flatten and the breaker latches TRIPPED (see `deploy/README.md`).

## Stage Roadmap
0. **Risk & Safety Layer** — max position size per trade, max daily loss limit, max drawdown halt. Circuit breaker halts all trading when tripped and requires manual reset (never auto-resume). Dry-run/paper-trading flag respected system-wide. Deliverable: standalone risk module + tests showing it blocks oversized orders and trips correctly on simulated loss sequences.
1. **Incremental Indicators (O(1) per tick)** — stateful EMA, RSI, ATR classes. Periodic resync (recompute from rolling window every N ticks) to correct floating-point/smoothing drift. Validated against pandas/ta-lib on 5,000+ ticks with max deviation reported.
2. **Local Order Book & Live Data Sync** — WebSocket-based order book manager (no pandas), efficient dict/deque structures. Efficient tick append to cached historical data without full re-fetch. Deliverable: replay test against a recorded WS feed confirming state stays consistent.
3. **Regime Detection (HMM)** — known failure modes (state-label instability across retrains, overfitting on short windows) explained and mitigated before implementation. Retraining cadence defined. Out-of-sample validation required before the regime signal is allowed to gate any live signal.
4. **Process Decoupling (ZeroMQ)** — separate execution core and intelligence core. Heartbeat/timestamp on every signal message; execution ignores signals older than X seconds. Fail-safe (does nothing) if the intelligence process crashes or hangs, rather than acting on stale data.
5. **Walk-Forward Validation** — background job evaluating live performance every 24h (Sharpe, win rate over last 20 trades), wired into the Stage 0 circuit breaker (not a second, separate halt mechanism). Includes a data-leakage check in the walk-forward split.

## OOP & Architecture Standards
When reviewing or writing classes, check against these:
- **Single Responsibility** — a class should have one reason to change. Flag "god objects" (e.g. a `TradingBot` class that fetches data, computes indicators, manages risk, and executes orders all in one place).
- **Composition over inheritance** — indicator classes, risk checks, and execution logic should be composed into the pipeline, not built as deep inheritance chains.
- **Dependency injection** — components (exchange client, risk manager, order book) should be passed into constructors, not hardcoded/instantiated inside other classes. This is required for testability (mocking the exchange client in tests).
- **Interfaces/protocols over concrete coupling** — e.g. the Signal Generator should depend on an abstract `OrderBookInterface`, not directly on the WebSocket implementation, so the data source can be swapped or mocked.
- **Type hints everywhere** — every function signature and class attribute should be typed. Flag any `Any` type used to paper over an unclear contract between modules.
- **Immutability where it matters** — risk limits and circuit breaker thresholds should not be mutable from arbitrary parts of the codebase; flag any place that can silently modify them outside the Risk Manager.
- **No circular imports** — flag any module that imports from a module that (directly or transitively) imports it back.
- **Explicit state ownership** — for any piece of shared state (circuit breaker status, order book snapshot, position state), identify exactly one class that owns writes to it. Flag any other class that mutates it directly instead of going through the owner.
- **Position sizing methodology** — the Risk Manager must implement a clearly defined sizing model (e.g., Fixed Fractional, ATR-scaled) before applying the max position size limit. Sizing logic must not be hardcoded inside the Signal Generator.
- **Strategy purity** — `generate_signals`/`get_signal` should be pure functions of their inputs: no network calls, no filesystem access, no model training inside a strategy. External context (higher-timeframe data, funding, imbalance) arrives via the pipeline; collaborators are constructor-injected against Protocols. (Known violation: `AICombinedStrategy` — scheduled for the parity redesign.)
- **No module-level side effects** — importing a project module must not read `.env`, create directories, or construct singletons; composition happens in entrypoints.
- **Typed boundaries** — data crossing a package boundary is a frozen dataclass/TypedDict, never a bare `dict`; `.get(key, default)` on a cross-module payload is a review-blocking smell.
- **Indicator warm-up contract** — never emit a value computed from fewer than `period` observations; constructors validate `window_size ≥ warm-up length`; batch indicators never coerce NaN warm-up rows into directional values (`np.where(NaN > x, 1, -1)` is banned).
- **Config is validated once, then frozen** — pydantic `frozen=True` with range constraints on every risk-relevant field; risk parameters exist in exactly one place (`Settings`), and `RiskConfig` is constructed FROM settings in the execution layer, never from library defaults.

## Connections & Data Flow Standards
When reviewing structural connections between modules, verify:
- The ZMQ Execution Core correctly deserializes the JSON schema published by the Intelligence Core (matching field names/types on both ends).
- The Order Book Manager correctly feeds Bid/Ask imbalance data to the Signal Generator (check the actual data contract, not just that a function is called).
- The Risk Manager wraps every path to order execution — no code path can send an order without passing through position sizing and circuit breaker checks first.
- No mismatched data types across module boundaries (e.g. float vs Decimal for prices/quantities, naive vs timezone-aware timestamps).
- No missing state handoffs — e.g. confirm the Walk-Forward job and Risk Manager read/write the same circuit breaker state, not two separate copies.
- All timestamps must be UTC, timezone-aware (ISO 8601). Flag any naive datetimes.
- Stale signal threshold (X seconds) for ZMQ execution must be explicitly defined in configuration (`settings.stale_signal_seconds`), not hardcoded.
- Exactly-once signal consumption: every published message carries a unique `message_id`; the execution core records the last acted-upon ID and never acts twice on the same one. Freshness (staleness) and consumption (idempotency) are separate mechanisms.
- Heartbeats: the intelligence core publishes a heartbeat at an interval ≤ stale_threshold/3; the execution core ALARMS (not just HOLDs) when no message of any kind arrives past the threshold.
- Inter-process messages are defined once as a typed schema shared by producer and consumer; semantic contracts (value ranges, sign conventions) are part of the schema and tested. Unknown/missing required fields are rejected and counted, never `.get()`-defaulted into silent HOLDs.
- One metric library: Sharpe, win rate, and drawdown definitions live only in `risk/metrics.py`; every threshold names the variant it gates (per-trade vs annualized). Win = `pnl > 0` project-wide; zero-PnL trades count as losses.
- Backtest/live parity: the signal path used by backtest/walk-forward and live must be the same function; any live-only filter is either replayable in backtest or explicitly listed as a parity exception in the run report; the engine enforces the same RiskManager checks (position cap, affordability) with the same sizing function. Walk-forward training data excludes the evaluation window entirely, including scaler fitting. (See `docs/backtest_live_parity_design.md`.)

## Performance Standards
- Indicator updates (EMA/RSI/ATR) must be O(1) per tick — flag any implementation that recomputes over a window on every update instead of updating incrementally.
- No blocking calls inside the asyncio event loop — flag any `time.sleep()`, synchronous `requests.get()`, or other blocking I/O inside async functions. Handlers on an asyncio loop that serve operator commands must dispatch blocking work via `asyncio.to_thread`/executor.
- Shared mutable state accessed from more than one THREAD, PROCESS, or async task (balance, open-trade set, breaker record, pause flags) must be guarded by a named lock or a single-writer design. Every load-modify-save of the breaker record must be atomic (peak-equity updates use `update_peak_equity`, which never rewrites state). A declared lock that is never acquired is a blocking finding.
- Staleness math uses `time.monotonic()` at receipt — embedded wall-clock timestamps are for logging only (an NTP step must not resurrect a stale signal or fake liveness).
- Graceful shutdown: signal handlers only set flags; the main loop exits cooperatively, then schedulers stop with bounded `wait=True`, ZMQ sockets close with LINGER=0, Telegram stops via its async API. `sys.exit()` inside a signal handler while workers may be mid-write is forbidden.
- Process supervision: both nodes run under a supervisor (systemd/supervisord) with restart-on-exit (see `deploy/`); the execution node exposes "seconds since last intelligence message" and alarms past the stale threshold.
- Bounded memory — the order book / tick history structures must have a fixed max size (deque with maxlen, or explicit eviction). Flag any structure that can grow unbounded under sustained high-frequency updates (e.g. 10,000 updates/sec).
- Fail-safe behavior over silent degradation — if the WebSocket drops, the system must stop trading (or clearly flag stale data), never silently continue on stale state. If the SQLite database is locked, the bot must handle the exception, never crash uncaught.

## Current Status & Stage Gating
Stage: 0-5 built. The comprehensive Stage 0-5 gate review is complete — see `ANALYSIS_REPORT.md` (2026-07). Fix tiers 1-6 are merged (all BLOCKING + major HIGH findings; regression tests in `tests/test_tier1_*` … `test_tier6_*`). Known open items: correlation-ID threading through logs, live latency instrumentation, order-book wiring (Stage 2 manager is currently unused), and the backtest/live parity redesign (`docs/backtest_live_parity_design.md`).

### Stage Dependency Map
- Stage 1 (Indicators): No dependencies.
- Stage 2 (Order Book): No dependencies.
- Stage 3 (HMM Regime): Depends on Stage 1.
- Stage 4 (ZMQ Decoupling): Depends on Stages 0, 1, 2, 3.
- Stage 5 (Walk-Forward): Depends on Stage 0 (Circuit Breaker) and Stage 4 (Signal Pipeline).

### Paper-Trading → Live Transition Criteria
Before live capital is deployed, paper trading must meet ALL of the following:
- [ ] 30 consecutive days of unattended uptime without uncaught exceptions.
- [ ] Minimum of 100 executed paper trades.
- [ ] Walk-forward Sharpe ratio > 1.0.
- [ ] Zero state desync incidents between bot and exchange upon restarts.
- [ ] Manual review of trade audit trail confirming expected behavior.

Not yet added (planned incrementally before paper trading): testing specifics, config management standards.