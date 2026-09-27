# NeuronTrade — Comprehensive Audit Report (Stage 0–5 Gate Review)

**Date:** 2026-07-15 · **Branch:** `version1` · **Scope:** all project code (~13,300 lines Python) audited against `claude.md` by six parallel audit passes (Risk/Precision, State Recovery, Logging/Validation, Concurrency/ZMQ, OOP/Architecture, Cross-Module Data Flow). Every cited line number was verified against source; a sample of 14 findings was independently re-verified in the main session. Behavioral claims marked "(verified by execution)" were reproduced in the project venv. **No files were modified.**

**Headline:** the system as currently written **cannot complete a single trading cycle in any runtime mode**. Five separate phantom references (`DataFetcher.fetch`, `settings.anthropic_api_key`, `RiskManager.check_drawdown_limit`, `PaperTrader._calculate_pnl`, a 4-arg call to a 3-arg `_close_position`) plus an `AISignal` constructor crash and an `"atr"`/`"ATR"` column mismatch kill every pipeline at or before its first hop — all hidden behind broad `except Exception` handlers. Independently of those, the safety layer's core guarantees (Decimal precision, persistent circuit breaker, kill switch, boot reconciliation, audit trail) are either absent or void. The paper→live transition criteria in claude.md cannot currently be met.

---

# Section 1 — CLAUDE.md improvements

## 1A. Corrections to existing rules

### 1A-1. Fix the database-path rule and make it binding on constructors

Current rule (`claude.md:12`) says `./data/neurontrade.db`; the code uses **three other variants**: CWD-relative `storage/neurontrade.db` (`storage/trade_logger.py:83`), `sqlite:///storage/neurontrade.db` in settings that nothing reads (`config/settings.py:39-41`), and an absolute repo-anchored path (`dashboard/analytics.py:19`).
**Proposed replacement:** "The SQLite path comes from configuration (`settings.database_url`), resolved to an absolute path anchored at the project root. No component may declare its own path default; `sqlite3.connect` on the trading DB may appear in exactly one module. CWD-relative DB paths are forbidden (launching from a different directory must not silently create a fresh, empty state database)."
**Reasoning:** a CWD-relative path makes "accidental full state reset" a one-command operator mistake, and the current three-way split means the bot, the dashboard, and the documentation can each be looking at a different file — a fork of the very state (trades, breaker) the design assumes is shared.

### 1A-2. Extend the concurrency rule from "async tasks" to threads

Current rule (`claude.md:76`) says shared mutable state accessed from multiple **async tasks** must use `asyncio.Lock()`. The production Stage 4 topology is **thread-based** (ZMQ polling main thread + APScheduler worker threads + Telegram event-loop thread), and the codebase took the rule literally: zero locks are acquired anywhere (two `RLock`s exist; neither is ever used in a `with` block).
**Proposed replacement:** "Shared mutable state accessed from more than one thread, process, or async task (balance, open-trade set, breaker record, pause flags) must be guarded by a named lock or a single-writer design. Every load-modify-save of the circuit-breaker record must be atomic. A declared lock that is never acquired is a blocking finding."
**Reasoning:** the wording gap produced a real lost-update hazard where a concurrent peak-equity save can overwrite `TRIPPED` with `ARMED` (`risk/manager.py:337-349`) — the single most safety-critical state in the system.

### 1A-3. Specify the order-book synchronization algorithm, not just "gaps must trigger a resync"

Current rule (`claude.md:36`) demands gap-triggered resync but doesn't say how gaps are detected. The implementation blindly overwrites `last_update_id` (`data/order_book.py:134`) — gaps are _undetectable_, so the rule is unfalsifiable as written.
**Proposed replacement:** "The local book must implement the exchange's documented sync algorithm (for Binance: buffer diffs, fetch snapshot, discard events with `u ≤ lastUpdateId`, require the first applied event to bracket `lastUpdateId`, and require every subsequent event's `U` to equal previous `u + 1` — otherwise discard the book and re-snapshot). The consumed diff schema must include the first-update-ID field, not just the final one."
**Reasoning:** this is the documented, load-bearing algorithm ([Binance local order book guide](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly)); anything less silently serves a corrupted book to the signal generator.

### 1A-4. Replace "X ms" / "N seconds" placeholders with named config keys

`claude.md:31` ("latency exceeding X ms") and `claude.md:37` ("no ticks for N seconds") leave thresholds undefined; the code answered with hardcoded magic numbers (`if time_diff > 300` at `execution/paper_trader.py:142`, `max_signal_age_seconds=30` at `scripts/run_decoupled_execution.py:54`).
**Proposed replacement:** every threshold the rules reference (`latency_warn_ms`, `stale_data_halt_seconds`, `stale_signal_seconds`, `heartbeat_interval_seconds`) must exist as a named field in `Settings`, with a startup assertion on cross-field coherence (see 1B-4).

### 1A-5. Strengthen the Stage 4 heartbeat rule: cadence must be coupled to the staleness window

`claude.md:48` requires "heartbeat/timestamp on every signal message." Timestamps exist; a heartbeat does not — the publisher sends once per hour (`scripts/run_decoupled_intelligence.py:111-114`) while the subscriber invalidates after 30 s, so "intelligence crashed" is indistinguishable from "intelligence idle" for up to 59 minutes.
**Proposed replacement:** "The intelligence core must publish a heartbeat (or re-publish the last signal) at an interval ≤ stale_threshold/3. The execution core must alarm — not just HOLD — when no message of any kind has arrived for > stale_threshold, and must expose 'seconds since last intelligence message' as a monitored metric. Publisher startup must mitigate the PUB/SUB slow-joiner (settling delay or last-value re-publication)." ([ZeroMQ Guide, advanced pub-sub patterns](https://zguide.zeromq.org/docs/chapter5/))

## 1B. New rules — order lifecycle & state

### 1B-1. Ordered startup recovery sequence (mandatory, before trading is enabled)

**Proposed rule:** "Every entrypoint must run an explicit recovery sequence before scheduling any trading loop: (1) open DB and verify schema version; (2) load breaker state; (3) load open positions; (4) reconstruct cash/equity from persisted state (paper) or an exchange REST snapshot (live); (5) resolve any in-flight orders; (6) write a `RECOVERY_COMPLETE` audit row."
**Reasoning:** claude.md states individual reconciliation requirements (`:23-26`) but no rule forces a single ordered boot procedure — which is why `run_live.py` goes straight from construction to the first tick, the balance silently resets to `initial_balance` while open trades persist in SQLite, and a restart _resets the risk envelope_. Best practice is to query venues for all open orders on recovery and reconcile the local log against venue reality ([quant.engineering on execution systems](https://quant.engineering/build-execution-systems-crypto-trading-at-scale.html)).

### 1B-2. Idempotent order identity: client order IDs, persisted submit-before-send

**Proposed rule:** "Every order gets a UUID client order ID, persisted with status `SUBMITTED` _before_ the network call, then transitioned to `ACKED`/`FILLED`/`REJECTED`/`UNKNOWN`. Retries reuse the same ID. The trades schema must be able to represent in-flight and rejected orders, not only `open`/`closed`."
**Reasoning:** the current schema (`storage/trade_logger.py:34-55`) has only `open/closed` — a trade is born already-filled, so claude.md's ambiguous-order rule (`:25`) is _unimplementable_ on this schema. Client-assigned IDs are the standard mechanism that makes retries idempotent and crash-windows recoverable ([idempotency keys for order placement](https://www.tokenmetrics.com/blog/idempotency-keys-order-placement)).

### 1B-3. Exactly-once signal consumption

**Proposed rule:** "Every published signal carries a unique `message_id` (or publish sequence number). The execution core records the last acted-upon ID and never acts twice on the same one; the dedup key is persisted with the trade row. Freshness (staleness control) and consumption (idempotency control) are separate mechanisms."
**Reasoning:** the subscriber returns the same cached signal as "fresh" for the whole 30 s window (`core/zmq_subscriber.py:157`) while the executor polls every second — one BUY deterministically pyramids to `max_concurrent_positions`. The staleness window answers "is intelligence alive?", not "should I act again?".

### 1B-4. Versioned message schema, validated on both ends, defined once

**Proposed rule:** "Every inter-process message is defined once as a typed schema (dataclass/pydantic) in a shared module imported by both producer and consumer, carries `schema_version` + `message_id`, and is validated on receipt: unknown/missing required fields are rejected and _counted_, never `.get()`-defaulted. Semantic contracts (value ranges, sign conventions) are part of the schema docstring and tested."
**Reasoning:** the publisher ships `confidence` in the `score` field ("Combined AI score in [-1,+1]") at `scripts/run_decoupled_intelligence.py:88`; the subscriber masks missing fields into silent HOLDs (`core/zmq_subscriber.py:181-190`); the executor reconstructs `AISignal` missing three required fields and crashes (`run_decoupled_execution.py:131-136`). All three stem from the schema existing only informally.

### 1B-5. Monotonic clocks for staleness; exchange-clock drift check

**Proposed rule:** "Staleness is computed from `time.monotonic()` at receipt, never by subtracting sender wall-clock timestamps; embedded UTC timestamps are for logging only. At startup and periodically, compare local time against the exchange server-time endpoint; sustained drift beyond a configured threshold flags data stale."
**Reasoning:** both ZMQ ends use `datetime.now(UTC)` arithmetic — an NTP step backward on the execution host makes an old cached signal look _young again_, re-opening the acting-on-stale-data hole Stage 4 exists to close. Latency measurement (`claude.md:31`) is meaningless if the two clocks disagree.

### 1B-6. Single DB access layer with explicit concurrency settings; append-only safety history; backups

**Proposed rule:** "Exactly one module owns SQLite connections; it sets `PRAGMA journal_mode=WAL` and `busy_timeout ≥ 10s` and wraps writes with retry-on-locked. Circuit-breaker transitions are append-only history rows, never `INSERT OR REPLACE` of a single row. The DB is snapshotted (SQLite `.backup`) daily and before any schema migration; schema changes go through a versioned migration table."
**Reasoning:** today three independent connection factories share one file with three different timeout/lock policies, the breaker store — the most critical writer — has the weakest settings and a docstring claiming a WAL mode that is never enabled (`risk/manager.py:152-153`, grep: no `PRAGMA journal_mode` anywhere), and every trip/reset destroys the previous record (`manager.py:204-210`), so the mandated transition audit trail cannot exist.

### 1B-7. Atomic writes and versioned bundles for all non-SQLite state

**Proposed rule:** "Non-SQLite state (models, scalers, flags) is written to a temp file and `os.replace()`d; multi-file model artifacts are bundled into one versioned artifact containing schema version, training-data time range, feature-name hash, and library versions. Loading hard-fails on any mismatch or missing component — never degrades to neutral output or unscaled inputs."
**Reasoning:** `ml_predictor.py` demonstrates both halves of the gap: three sequential non-transactional `joblib.dump` calls (`:214-216`), an unguarded load on the boot path (`:246`), and a _silent-garbage_ mode where a missing scaler file lets the model predict on unscaled features (`:246-248`, `:288-291`). Torn model/scaler pairs produce silently wrong predictions — worse than a crash.

### 1B-8. Crash-recovery testing as a stage gate

**Proposed rule:** "Every stage gate includes a kill-and-restart test: SIGKILL the process at each critical transition (post-submit/pre-persist, post-persist/pre-act, mid-scale-out) and assert restart converges to a consistent state with no duplicated or lost balance/positions."
**Reasoning:** the tests verify breaker persistence but nothing tests process death; every act-then-write bug found (balance mutated before the DB write at `paper_trader.py:257→260`, `330→333`, `403→415`) would be caught mechanically.

## 1C. New rules — risk & precision

### 1C-1. Exchange lot-size / tick-size / min-notional quantization

**Proposed rule:** "Every order quantity is quantized DOWN to the symbol's `stepSize`, every price to `tickSize`, and orders below `minNotional` are rejected — using exchange filters fetched at boot. `Decimal` values are constructed from strings, never floats; `quantize()` always specifies an explicit rounding mode."
**Reasoning:** the codebase emits raw float quantities with 15+ digits (`risk/manager.py:405`); on real Binance these are rejected by the `LOT_SIZE` filter, so the first live deployment fails on every order. Quantizing down is also the natural implementation site for claude.md's "round position size down" mandate, which currently has no implementation anywhere (grep: no `quantize`/`ROUND_*` in the repo). Generic `round(x, N)` is a documented trading-bot failure pattern ([production bot failure patterns](https://florinelchis.medium.com/production-trading-bots-15-failure-patterns-nobody-warns-you-about-af917d263c35)).

### 1C-2. Fee- and slippage-inclusive risk budgeting

**Proposed rule:** "Position sizing solves for expected loss at stop _including_ round-trip commission and modeled slippage; risk checks use the same all-in loss figure."
**Reasoning:** `calculate_position` budgets pure price risk (`manager.py:395-405`) and execution bolts fees on afterwards, so every stopped-out trade overspends its budget by ~20-50% of the risk amount — the daily-loss breaker trips earlier than the operator's model predicts, in the optimistic-assumption direction.

### 1C-3. Short selling: model it or forbid it

**Proposed rule:** "Until margin/borrow mechanics are implemented, the Risk Manager rejects `side='sell'` entries on spot accounts. When implemented, short cash-flow accounting (credit proceeds at entry, debit buy-back at exit) must be unit-tested for equality against recorded PnL."
**Reasoning:** strategies freely emit SELL entries, sizing plans them, and the wallet accounts for them as longs — producing the inverted-PnL BLOCKING bug (V-8 below) where the DB says a short won while the wallet lost money. A one-line rejection converts silent capital corruption into an obvious blocked-trade log.

### 1C-4. One canonical equity definition, one peak-equity writer, one metric library

**Proposed rule:** "Equity = cash + mark-to-market of open positions (shorts signed correctly), defined in exactly one function. Peak equity and drawdown have exactly one writer (RiskManager); `_BreakerRecord` and the store are private to the risk package. Sharpe, win rate, and drawdown live in one module (`risk/metrics.py`); every threshold names the metric variant it gates (per-trade vs annualized)."
**Reasoning:** three code sites write peak equity against two different equity definitions, and three incompatible Sharpe implementations coexist (`risk/walk_forward.py:203-217` per-trade, `backtesting/engine.py:373` annualized per-candle, `dashboard/analytics.py:132-147` hourly equity curve) — so the Stage-5 breaker gate and the go-live "Sharpe > 1.0" criterion are numerically incomparable.

### 1C-5. Backtest/live parity requirement

**Proposed rule:** "The signal path used by backtest/walk-forward and the live path must be the same function. Any live-only filter (funding, order-book imbalance, macro trend, HMM gate) must be replayable in backtest or explicitly listed as a parity exception in the run report. The engine must enforce the same RiskManager checks (including `max_concurrent_positions` and affordability) with the same sizing function — no fallback sizing branch. Walk-forward training data must exclude the evaluation window entirely, including scaler fitting."
**Reasoning:** live gates BUYs through six filters that the backtest path never runs (`strategies/ai_combined.py:156-313` vs `:91-154`), the engine has its own sizing fallback and lets capital go negative, and the documented workflow trains the model on the same DataFrame it then backtests — so the go-live gate certifies a different bot than the one that will trade, on leaked data ([lookahead bias and leakage in crypto backtesting](https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/)).

### 1C-6. Kill-switch drill as a go-live criterion

**Proposed rule:** "Before paper trading begins, a documented kill-switch drill must pass: trigger the out-of-band flatten (flag file / OS signal / HTTP) while positions are open and the intelligence core is deliberately hung; verify all positions flatten, working orders cancel, the breaker latches TRIPPED, and the drill result is persisted to the audit trail. The kill switch runs in (or is reachable from) a separate process from the main bot."
**Reasoning:** claude.md requires the capability to exist (`:40-41`) but nothing forces it to be _proven_; this audit found the adjacent emergency machinery consists of three references to code that was never written. Separate-process kill switches are the standard recommendation ([algorithmic trading risk management guides](https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/)).

## 1D. New rules — engineering hygiene

### 1D-1. Static contract gate on the execution perimeter

**Proposed rule:** "CI runs mypy/pyright strict on `risk/`, `execution/`, `core/`, `scripts/run_*`; any unresolved attribute or arity mismatch in code that can reach order execution is a merge blocker. Additionally, a smoke test constructs every entrypoint's object graph (mocked exchange) and drives one full cycle."
**Reasoning:** seven separate runtime-fatal phantom references shipped across three entry points, all hidden behind broad `except Exception`. Every one is caught by mypy or by a single construction-plus-one-cycle smoke test. This is the highest-leverage single rule this audit can propose.

### 1D-2. Secret redaction

**Proposed rule:** "API keys/tokens never appear in log output, exception messages, or outbound notifications. Secrets go in headers, not URL query params, wherever supported; exception strings derived from HTTP requests are scrubbed of query strings before logging or re-raising."
**Reasoning:** the NewsAPI key rides in a URL query param and lands in `DataFetchError` messages via `HTTPStatusError` stringification (`data/news_fetcher.py:68-75, 111-112`), then persists in 30-day-retained log archives and can be pushed to Telegram.

### 1D-3. Command-channel authentication and command auditing

**Proposed rule:** "Any out-of-band control channel that can pause, resume, flatten, or force-close must verify the sender against a configured allowlist before executing; every command received (including rejected ones) is persisted to the audit trail with sender ID and UTC timestamp."
**Reasoning:** the Telegram bot executes `/pause`, `/resume`, `/force_sell` for _any_ Telegram user who finds it (`notifications/telegram_bot.py:144-262`; grep: zero `effective_chat`/`effective_user` checks) — remote unauthenticated control of order execution.

### 1D-4. Fetch retry/backoff standard; fabricated defaults forbidden

**Proposed rule:** "External fetchers implement bounded retries with exponential backoff and jitter, honoring 429/`Retry-After`. Exhausted retries raise a typed `DataFetchError` — never return a fabricated neutral value (0.0 funding, 1.0 imbalance, 'Neutral' sentiment). Data-quality failures must be distinguishable from neutral market states by every consumer."
**Reasoning:** `except Exception → return 0.0/1.0/neutral` appears in three fetch paths (`data/fetcher.py:228-230, 274-276`; `data/sentiment_fetcher.py:83-89`), converting programming bugs into plausible-looking market data the strategy trades on forever.

### 1D-5. Config validated once at startup, then frozen; single source of truth

**Proposed rule:** "One `validate_config()` runs at process start: pydantic `frozen=True`, range constraints on every risk-relevant field (`0 < risk_per_trade ≤ 0.05` etc.), live-mode cross-checks (non-empty API keys when `paper_trading=False`). Risk parameters exist in exactly one place; `RiskConfig` is constructed _from settings_ in entrypoints, never from library defaults. Per-pair config is returned as immutable objects."
**Reasoning:** verified by execution: `settings.risk_per_trade = 0.99` succeeds silently, `RISK_PER_TRADE=5.0` is accepted from the environment, and risk limits exist in four disagreeing places — the `.env` values an operator sets are dead config while hardcoded `RiskConfig` defaults actually govern.

### 1D-6. Strategy interface is pure and I/O-free; no module-level side effects; typed boundaries

**Proposed rule:** "`generate_signals`/`get_signal` are pure functions of their inputs: no network, filesystem, or training calls; external context arrives via a typed context object; collaborators are constructor-injected against Protocols. Importing any project module must not read `.env`, create directories, or construct singletons — composition happens only in entrypoints. Data crossing a package boundary is a frozen dataclass/TypedDict, never a bare `dict`; `.get(key, default)` on a cross-module payload is a review-blocking smell."
**Reasoning:** `AICombinedStrategy` constructs seven collaborators internally and performs 4+ synchronous REST calls per signal, making it untestable and unbacktestable; the always-bearish `sma_cross` LLM input existed precisely because producer and consumer shared only a stringly-typed dict.

### 1D-7. Indicator warm-up/readiness contract

**Proposed rule:** "Every indicator defines and tests three states: NOT_READY (never emit a value computed from fewer than `period` observations), READY, and DEGRADED (post-resync with insufficient window → raise, don't return None forever). Constructors validate `window_size ≥ warmup length`. Batch indicators must never coerce NaN warm-up rows into directional values — the `np.where(NaN > x, 1, -1)` pattern is banned."
**Reasoning:** three verified bugs share this root cause: RSI reports `ready` after one delta, `trend_200` labels all 174 warm-up rows (and, at live's `limit=100` fetch, _every_ row) as downtrend, and a bad `window_size` silently bricks an indicator mid-session.

### 1D-8. Logging: one config, JSON, UTC; alerting reliability; supervision

**Proposed rule:** "Exactly one logging setup installs handlers (entrypoints may not call `logger.remove()/add()`); all handlers use `serialize=True` and UTC (`!UTC` + explicit `Z`), matching the DB's ISO-8601 UTC. Operator alerts for breaker trips/drawdown/errors get ≥1 retry and a plain-text fallback; a failed critical alert is itself persisted. Error-rate budget: > N ERRORs per M minutes trips the Stage 0 breaker. Both nodes run under a supervisor with restart-on-exit and duplicate-instance protection; signal handlers only set flags (no `sys.exit` while workers may be mid-write); shutdown stops schedulers with bounded `wait=True`, closes ZMQ sockets with LINGER=0, and stops Telegram via its async API. Async handlers for operator commands never do direct blocking I/O (dispatch via `asyncio.to_thread`)."
**Reasoning:** two competing logging configs exist (the live entrypoint silently bypasses the central one), log lines are local-time while the DB is UTC, Markdown-parse failures silently drop exactly the alerts that matter, `TelegramBot.stop()` is a no-op on an un-awaited coroutine, and the current per-candle `AttributeError` spam means an error-rate budget would have surfaced every BLOCKING bug in this report on day one.

**Sources:** [Binance order-book sync](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly) · [Idempotency keys for orders](https://www.tokenmetrics.com/blog/idempotency-keys-order-placement) · [Execution systems at scale](https://quant.engineering/build-execution-systems-crypto-trading-at-scale.html) · [Production bot failure patterns](https://florinelchis.medium.com/production-trading-bots-15-failure-patterns-nobody-warns-you-about-af917d263c35) · [Backtesting leakage/lookahead](https://www.blockchain-council.org/cryptocurrency/backtesting-ai-crypto-trading-strategies-avoiding-overfitting-lookahead-bias-data-leakage/) · [ZeroMQ Guide ch.5](https://zguide.zeromq.org/docs/chapter5/) · [Algo risk management](https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/)

---

# Section 2 — Codebase violations

Ordered by severity. Each entry: rule violated → location(s) → failure mode. Findings confirmed independently by 2+ audit passes are marked ✱.

## BLOCKING — system cannot run, trades wrong, or safety guarantee void

**V-1. Phantom `DataFetcher.fetch()` — every runtime pipeline dies at the first hop.** _(Rule: claude.md:65-66 data contracts; :13 never infer existence)_ Callers: `scripts/run_live.py:192`, `scripts/run_decoupled_execution.py:139` and `:166`, `scripts/run_decoupled_intelligence.py:52`. `data/fetcher.py` defines only `fetch_ohlcv` (`:60`), and its keyword is `pair`, not `symbol`. Every candle cycle raises `AttributeError`, swallowed by broad handlers (`run_live.py:254`) → no data ever flows, no trade ever opens or exits, Telegram gets an error per cycle forever.

**V-2. `settings.anthropic_api_key` doesn't exist — both intelligence entrypoints crash at construction.** _(Rule: same)_ `scripts/run_live.py:108`, `scripts/run_decoupled_intelligence.py:41`, `scripts/health_check.py:49` vs `config/settings.py` (only `binance_api_key`, `groq_api_key`, `news_api_key`). Raised before any try/except — `run_live.py` cannot boot. The name is also semantically wrong: the LLM used is Groq (`ai/llm_agent.py:133-134`).

**V-3. Executor rebuilds `AISignal` missing 3 required fields — every fresh ZMQ signal raises `TypeError`.** _(Rule: claude.md:65 schema match)_ `scripts/run_decoupled_execution.py:131-136` passes 4 kwargs; `ai/signal_combiner.py` requires `llm_score`, `ml_score`, `sentiment_score` (no defaults). The publisher even ships those values in `extra`, but the executor never reads `sig.extra`. The decoupled pipeline can never trade.

**V-4. ✱ SL/TP exit job calls phantom `_calculate_pnl` and miscalls `_close_position` — stops never execute in decoupled mode.** _(Rule: claude.md:11 risk control verified before execution component; :78 silent degradation)_ `scripts/run_decoupled_execution.py:188-189`: `_calculate_pnl` is defined nowhere in the repo; `PaperTrader._close_position` (`execution/paper_trader.py:300-305`) takes 3 args, not 4. The 10-second exit job — the _only_ exit mechanism between hourly candles — throws per-trade `AttributeError`, swallowed at `:191-192`. Positions under water stay open indefinitely; the stop-loss is decorative. Confirmed by four of six audit passes.

**V-5. ✱ Phantom `RiskManager.check_drawdown_limit` — the drawdown halt never runs, error swallowed every candle.** _(Rule: Stage 0 max-drawdown halt)_ `scripts/run_live.py:241`; no such method exists in `risk/manager.py`. The `AttributeError` fires _after_ the candle has already traded (`:237`) and is caught at `:254`. The auto-pause path (`:242-247`) is dead code.

**V-6. ✱ `"atr"` vs `"ATR"` column mismatch — the live path can never open a position.** _(Rule: claude.md:68 no mismatched data across boundaries)_ Producer `indicators/technical.py:170` writes `df["ATR"]`; consumer `execution/paper_trader.py:160` reads lowercase `"atr"` → always `None` → `calculate_position` returns `None` (`risk/manager.py:390`) → silent no-trade, and ATR trailing stops are permanently disabled (`paper_trader.py:438`). `backtesting/engine.py:182` reads the correct case, so backtests trade while live silently does nothing — false parity. The decoupled node additionally never calls `compute_all` at all (`run_decoupled_execution.py:144-147`).

**V-7. ✱ Entire production order/risk path runs on `float`; the Decimal API is dead code.** _(Rule: Numerical Precision Policy, claude.md:17-20 — pre-declared blocking)_ `risk/manager.py:103-114` (`PositionPlan` all-float), `:395-405` (sizing math in float, `float(self.config.risk_per_trade_pct)` down-casting the Decimal config), `:334` (float subtraction _before_ the Decimal cast), `execution/paper_trader.py:86-89, 246-257, 317-330` (balance/commission/PnL float), `storage/trade_logger.py:39-52` (REAL columns), `strategies/base.py:19-22` (`TradeSignal` float prices). The Decimal-typed `approve_order`/`RiskSnapshot` API has zero production callers (tests only). No cautious rounding exists anywhere (grep: no `quantize`/`ROUND_*`).

**V-8. Short-position wallet accounting inverted — a winning short _reduces_ the balance.** _(Rule: Stage 0 correctness; proposed 1C-3)_ `execution/paper_trader.py:244-257` debits `position_value + commission` for both sides at entry; `:317-330` computes correct short PnL (`:327`) but then credits `self._balance += net_proceeds` (`:330`) — long-only accounting. Short entry 100 → exit 90: DB logs +10 PnL, wallet nets −10 − fees. `get_daily_pnl()` and `_balance` permanently diverge; the daily-loss breaker input is wrong in the dangerous direction for short-heavy sessions.

**V-9. ✱ `run_live` wires the circuit breaker to the in-memory store — a tripped breaker is erased by restart.** _(Rule: claude.md:26 persist-before-complete; Stage 0 never auto-resume; :69 same breaker state)_ `scripts/run_live.py:90-93` omits `db_path`; `execution/paper_trader.py:99-100` then falls back to `InMemoryCircuitBreakerStore` — documented "Use only in unit tests" (`risk/manager.py:133`) yet the default (`manager.py:244`). Daily-loss trip at 14:00 + restart at 14:05 = trading resumes the same day. `peak_equity` drawdown history is also lost. Only `run_decoupled_execution.py:48` wires the persistent store. The unsafe default itself is the root violation.

**V-10. ✱ Balance silently resets to `initial_balance` on restart while open trades persist in the DB.** _(Rule: claude.md:23 in-memory state not source of truth; :95 zero desync criterion)_ `execution/paper_trader.py:87-88` (never reloaded anywhere), `run_live.py:91` (hardcoded `10_000.0`). DB is authoritative for positions, memory for cash: restart with a $5,000 open position → balance snaps to $10,000 → position closes → equity fabricated at ~$15,100. Restarting the bot resets the daily-loss/drawdown envelope.

**V-11. No boot reconciliation exists anywhere in the repo.** _(Rule: claude.md:24 — verbatim requirement)_ Both entrypoints go straight from construction to trading (`run_live.py:172-177`, `run_decoupled_execution.py:37-109`). `DataFetcher.fetch_balance` has zero callers — and is itself broken (V-15). Grep: no `reconcil*`, `fetch_open_orders`, `fetch_positions` anywhere.

**V-12. One fresh ZMQ signal is re-executed once per second for 30 seconds — one BUY opens three positions.** _(Rule: proposed 1B-3)_ `core/zmq_subscriber.py:157` returns cached `self._last` as fresh for the whole window; `scripts/run_decoupled_execution.py:115-150` polls with `time.sleep(1)` and acts on every fresh poll (`is_actionable=True` hardcoded at `:135`). One signal deterministically pyramids to `max_concurrent_positions` — triple the intended risk per signal.

**V-13. Circuit-breaker load-modify-save race — a concurrent peak-equity update can overwrite TRIPPED with ARMED.** _(Rule: claude.md:76; :60 state ownership; Stage 0 never auto-resume)_ `risk/manager.py:337-349` re-saves the _whole_ record (including stale `state`) with no lock, racing `trip_circuit_breaker` (`:254-264`) called from the walk-forward APScheduler thread (`run_decoupled_execution.py:92-99`). Interleaving: load ARMED → other thread trips → save ARMED with new peak → the trip silently evaporates.

**V-14. No sequence-number validation or gap-triggered resync in the order book.** _(Rule: claude.md:36 — verbatim requirement)_ `data/order_book.py:115-137`: `last_update_id` is blindly overwritten (`:134`); the first-update-ID (`U`) field isn't even part of the API, so gaps are undetectable and `reset()` (`:139`) has no caller. A dropped WS message silently corrupts the book; corrupted imbalance/mid feed the signal path with no error.

**V-15. `fetch_balance()` crashes with uncaught `TypeError` on any funded account.** _(Rule: claude.md:78 never crash uncaught)_ `data/fetcher.py:180-188`: iterates `balance["total"]` (currency → float), filters `isinstance(info, (int, float))`, then subscripts `info["free"]`. First non-zero balance → `TypeError`, not caught by the `ccxt`-only except at `:190`. The account-balance path can never have worked — evidence it is untested.

**V-16. Telegram command handlers have zero sender authorization.** _(Rule: proposed 1D-3)_ `notifications/telegram_bot.py:144-262`: no handler checks `update.effective_chat.id`/user against config (grep: zero hits). Any Telegram user who finds the bot can `/pause`, `/resume`, and `/force_sell` positions — remote unauthenticated control of order execution.

**V-17. Logging is plain-text with no correlation IDs; the live entrypoint bypasses the central config.** _(Rule: claude.md:29 — verbatim)_ `config/logging_config.py:18-47` (no `serialize=True`), and `scripts/run_live.py:50-63` calls `logger.remove()` and installs its own divergent handlers (`level="DEBUG"`, `retention=3`). Grep: no `correlation|trace_id|bind(` usage anywhere. A signal cannot be traced through indicator → risk → order; post-incident reconstruction is impossible.

**V-18. Missing scaler file silently produces garbage ML predictions on unscaled features.** _(Rule: claude.md:78 silent degradation)_ `ai/ml_predictor.py:246-248` treats the scaler as optional at load; `:288-291` falls through to raw features. Model trained exclusively on StandardScaler output (`:170-172`) then emits numerically-valid, meaningless probabilities into the combiner at 40-60% weight, with no error flag.

## HIGH — mandated capability absent, safety math wrong, or latent race

**V-19. Kill Switch does not exist anywhere.** _(Rule: claude.md:40-41 — verbatim)_ Absence grep-verified. `paper_trader.py:104` references an `/emergency_stop` command that `telegram_bot.py` never registers; SIGINT/SIGTERM handlers (`run_decoupled_execution.py:214-228`, `run_live.py:284-296`) exit _without flattening or cancelling_. No file-flag/OS-signal/HTTP flatten path exists; the closest thing (per-pair `/force_sell`) is in-band and unauthenticated (V-16).

**V-20. Ambiguous-order resolution and order identity are unrepresentable in the schema.** _(Rule: claude.md:25)_ `storage/trade_logger.py:34-55`: `status IN ('open','closed')` only — no `SUBMITTED/REJECTED/UNKNOWN`, no exchange order ID, no client order ID, no correlation ID column. A trade is born already-filled; the rule cannot be implemented on this schema.

**V-21. Act-then-write ordering on every balance transition.** _(Rule: claude.md:26 persist-before-complete)_ `execution/paper_trader.py:257→260` (open), `:330→333` (close), `:403→415` (scale-out). A locked DB or crash between mutation and persist loses cash or double-credits scale-out proceeds on restart replay.

**V-22. ✱ WAL claimed but never enabled; breaker store has no lock, no busy_timeout, no locked-DB handling.** _(Rule: claude.md:26, :78)_ `risk/manager.py:152-153` docstring claims WAL; grep: no `PRAGMA journal_mode` anywhere. `save()` (`:204`) uses default 5 s timeout on the same file `TradeLogger` locks with different settings; no `sqlite3.OperationalError` handling exists repo-wide. A locked DB during `trip_circuit_breaker` means **the breaker trip itself is lost** — logged and skipped by the blanket job handler.

**V-23. ✱ Audit trail incomplete: no rejections, no risk-check outcomes, breaker history destroyed on every save.** _(Rule: claude.md:30 — verbatim)_ Only `trades`, `signals`, `daily_summary`, and a single-row breaker state are persisted. `INSERT OR REPLACE ... id=1` (`risk/manager.py:204-210`) destroys the previous transition on every save and `manual_reset()` wipes `peak_equity`. Risk rejections (`paper_trader.py:219`) and `/pause`//`/resume` leave no DB trace. The go-live "manual review of trade audit trail" criterion cannot be satisfied.

**V-24. ✱ `PaperTrader._lock` declared but never acquired — cross-thread balance corruption and double-close.** _(Rule: claude.md:76)_ `execution/paper_trader.py:90` is the only occurrence of `_lock` in the file. Concurrent writers: ZMQ polling thread, APScheduler workers (10 s exit job, walk-forward), Telegram loop (`/force_sell`). `_check_active_exits` and `process_candle` Step 1 can both close the same trade → proceeds credited twice. `+=`/`-=` on `_balance` are non-atomic. Check-then-act across `can_open_trade` → `_open_position` (`paper_trader.py:194-240`) similarly lets concurrent pair jobs exceed `max_concurrent_positions`.

**V-25. Consecutive-loss "circuit breaker" auto-resumes on a 12 h timer.** _(Rule: Stage 0 "never auto-resume")_ `risk/manager.py:367-375`: pure time-based re-enable, no manual reset, and it checks the most recent trade's timestamp _regardless of whether it was a loss_. In backtests `now = datetime.now(UTC)` vs historical exit times means the cooldown is always already expired — the limit never binds in backtesting, so live and backtest diverge.

**V-26. ✱ Equity for drawdown/peak tracking values open positions at entry price, not market.** _(Rule: RiskManager's own contract, `manager.py:222`; Stage 0)_ `execution/paper_trader.py:47-50` and `:206-208`: `quantity * entry_price` — a position down 40% mark-to-market shows zero drawdown; shorts are counted as long assets. Max-drawdown enforcement is blind to unrealized losses; dashboard/Telegram equity is fiction.

**V-27. Backtest engine lets capital go negative and carries its own sizing fallback.** _(Rule: Stage 0 sizing; proposed 1C-5)_ `backtesting/engine.py:275-280`: no affordability check (`paper_trader.py:249` has one — duplicated, divergent); `:268-273` fallback sizing bypasses `calculate_position`; `:246` passes `open_position_count=0` unconditionally, disabling the concurrency cap live enforces. Backtest metrics gating go-live are computed on unmodeled leverage.

**V-28. Backtest engine mutates the breaker store directly and re-implements drawdown.** _(Rule: claude.md:60 single writer; Stage 5 no second halt mechanism)_ `backtesting/engine.py:12` imports private `_BreakerRecord`; `:316-334` writes peak/trips around the RiskManager in float, parallel to `evaluate_equity_risk` (Decimal). Three peak-equity writers exist (`manager.py:341-349`, `engine.py:318-329`, seeding at `manager.py:245`).

**V-29. ✱ Risk limits are dead config in four disagreeing places.** _(Rule: claude.md:58 immutability/ownership; settings.py's own header)_ Enforced values are the hardcoded `RiskConfig` defaults (`risk/manager.py:233-242`: 1%/trade, 3% daily, notional = 100% of equity); `config/settings.py:30-33` (2%/10%), `config/constants.py:97-102`, and `config/pairs.py:13-14` all have zero consumers in the risk path (grep-verified). Setting `MAX_DRAWDOWN=0.05` in `.env` changes nothing.

**V-30. ✱ Publisher sends `confidence` in the `score` field — audit trail records bullish scores for bearish trades.** _(Rule: claude.md:65 schema semantics)_ `scripts/run_decoupled_intelligence.py:88` maps `trade_signal.confidence` ([0,1]) into `score` (documented [-1,+1] at `core/zmq_publisher.py:74-75`); persisted downstream as `combined_score`. A SELL at confidence 0.8 is recorded as score +0.8.

**V-31. ✱ Slow-joiner + hourly publish + 30 s staleness: a restarted executor is signal-blind for up to 59 minutes, indistinguishable from a dead intelligence core.** _(Rule: claude.md:48; proposed 1A-5)_ `core/zmq_publisher.py:57-59` (bind, no sync, no HWM — grep: zero HWM/CONFLATE config), `run_decoupled_intelligence.py:111-114` (`time.sleep(3600)`), `zmq_subscriber.py:52`. Also `ReceivedSignal.is_fresh` conflates freshness with actionability (`zmq_subscriber.py:67-70`): a fresh HOLD reports not-fresh, so liveness monitoring is impossible through this API.

**V-32. ✱ Stale-signal and stale-data thresholds hardcoded, not configured.** _(Rule: claude.md:71 — verbatim)_ `run_decoupled_execution.py:54` (30 s), `zmq_subscriber.py:52`, `paper_trader.py:142` (300 s), plus ZMQ address duplicated as module constants in both `zmq_publisher.py:32` and `zmq_subscriber.py:50`. `Settings` has none of these fields. The stale-data check also keys off `settings.default_timeframe` rather than the DataFrame passed (`paper_trader.py:136`), and only runs per trading cycle — it is not a timer; the order book has no staleness concept at all (`is_ready` stays True forever, `order_book.py:152-154`).

**V-33. ✱ Stage 2 Order Book Manager feeds nothing; the imbalance actually used has different units.** _(Rule: claude.md:66 — verbatim)_ `data/order_book.py` has zero production imports (grep). The strategy uses REST `fetch_order_book_imbalance` (`strategies/ai_combined.py:218`) — quantity-weighted (`fetcher.py:261-262`) vs the manager's notional-weighted (`order_book.py:249-250`) — same threshold `1.5` would mean different things when the manager is wired in.

**V-34. `run_live` re-runs LLM+sentiment and recombines without ML — the audit trail logs a different signal than the one traded.** _(Rule: claude.md:29-30)_ `scripts/run_live.py:217-224`: second, non-deterministic `_llm.analyze` call, `ml_prediction=None`, no ADX/ATR weights, output logged as if it were the decision from `strategies/ai_combined.py:248-255`. Doubles Groq cost and falsifies component scores in the `signals` table.

**V-35. Order book and tick aggregator accept NaN/negative/zero prices and quantities.** _(Rule: claude.md:34-35 — verbatim)_ `data/order_book.py:214-221` (`qty == 0.0` removal test passes NaN and negatives into the book; NaN price becomes a dict key and can become best bid via `max()` at `:223-226`), snapshot path `:107-108` (accepts NaN/negative _prices_). `data/tick_aggregator.py:123-141`: no validation at all — one NaN tick poisons the candle (`max(high, nan)` → NaN), then the historical DataFrame and every indicator.

**V-36. NewsAPI key leaks into exception messages, logs, and potentially Telegram.** _(Rule: proposed 1D-2)_ `data/news_fetcher.py:68-75` (key as URL query param) + `:111-112` (`f"...{e}"` wraps the full URL from `HTTPStatusError`). Logs are retained 30 days compressed.

**V-37. Silent fabricated market data on any exception.** _(Rule: claude.md:78 — verbatim)_ `data/fetcher.py:228-230` (funding → 0.0), `:274-276` (imbalance → 1.0), `data/sentiment_fetcher.py:83-89` (sentiment → Neutral 50). `except Exception` converts programming bugs into plausible neutral market data the strategy trades on indefinitely.

**V-38. No latency measurement exists at any pipeline boundary.** _(Rule: claude.md:31 — verbatim)_ Grep: no `latency|perf_counter|monotonic|elapsed` instrumentation in any non-test file. Exchange timestamps are stored but never compared to local time.

**V-39. Log timestamps are naive local time.** _(Rule: claude.md:70 UTC everywhere)_ `config/logging_config.py:22, 38` and `run_live.py:53`: `{time:...}` without `!UTC`/offset — log lines and the UTC SQLite audit trail disagree by the host's UTC offset and are ambiguous across DST.

**V-40. `trend_200` labels every NaN warm-up row — and at live's `limit=100` fetch, every row, always — as downtrend.** _(Rule: claude.md:9 never assume silently; proposed 1D-7)_ `indicators/technical.py:88` (`np.where(df["close"] > df["SMA_200"], 1, -1)`; NaN > x is False); the dropna at `:57` doesn't cover SMA_200. Verified by execution: 174 of 260 rising-market rows labeled -1. Live fetches 100 candles (`ai_combined.py:172-173`) so SMA_200 is always NaN → the macro-trend BUY gate is permanently bearish. Same NaN→-1 pattern at `:135` (MACD_cross) and `:191-193` (OBV_trend).

**V-41. LLM prompt's `sma_cross` input is permanently "Fast below Slow".** _(Rule: claude.md:66 actual data contract)_ Producer `technical.py:255-306` never emits an `sma_cross` key; consumer `ai/llm_agent.py:194-195` defaults `indicators.get("sma_cross", 0)` → every prompt asserts a bearish cross regardless of market (verified by execution). Silently biases the 25-55%-weighted LLM component.

**V-42. `IncrementalRSI` reports `ready` and emits pinned 100/0 after a single delta.** _(Rule: Stage 1 warm-up)_ `indicators/incremental.py:141-146` seeds averages from the first delta; `:126-128` defines `ready` as "value exists". Verified: `update(100); update(101)` → RSI 100.0, ready=True. Also flat-market RSI returns 50.0 (`:154-157`) where pandas-ta returns NaN — a formula divergence between `incremental.py` and `technical.py` masked in tests by `.dropna()`.

**V-43. Documented workflow trains the ML model on the same DataFrame it then backtests.** _(Rule: Stage 5 data-leakage check)_ `strategies/ai_combined.py:361-368` ("Call this before running backtests") + `generate_signals` (`:109-139`): 80% of the backtest window is in-sample for the model, 100% for the scaler. The walk-forward Sharpe gate can be passed on leakage.

**V-44. Settings are mutable from anywhere and accept absurd risk values from the environment.** _(Rule: claude.md:58 immutability)_ `config/settings.py:52-57` sets neither `frozen=True` nor validation constraints. Verified by execution: `settings.risk_per_trade = 0.99` succeeds; `RISK_PER_TRADE=5.0` accepted from env. `config/pairs.py:46-50` returns live references into mutable global per-pair risk params (verified mutation propagates).

**V-45. `AICombinedStrategy` is a god object hard-coupled to live infrastructure.** _(Rule: claude.md:53-56 SRP/DI/interfaces)_ `strategies/ai_combined.py:56-69` constructs 7 collaborators internally (including `DataFetcher()` at `:68` via a deferred import hiding the layering violation); `get_signal` performs 4+ synchronous REST calls (`:172-179, 214-218`) plus a Groq HTTP call, all swallowed to HOLD by the blanket except at `:351`; `train_model` (`:361-377`) trains models inside a strategy. Untestable without a live exchange; unusable from an async loop.

**V-46. Backtest and live run different strategies.** _(Rule: proposed 1C-5)_ Live `get_signal` (`ai_combined.py:156-313`) gates through macro trend, RSI+MACD timing, volume, funding, imbalance, and HMM regime; backtest `generate_signals` (`:91-154`) applies none of those filters (HMM only externally in `scripts/walk_forward.py:127-133`). Walk-forward certifies a signal stream production will never emit.

**V-47. The risk test suite tests only the dead Decimal API; the production path is untested.** _(Rule: Stage 0 deliverable; claude.md:8)_ `tests/test_risk_manager.py` (all 9 tests) exercises `approve_order`/`evaluate_equity_risk` — zero production callers. No test covers `can_open_trade`, `calculate_position`, `check_position_exits`, the cooldown, or any `PaperTrader` behavior. The short inversion (V-8), ATR mismatch (V-6), and both phantom methods (V-4, V-5) would each have been caught by one integration test. Green CI with three BLOCKING bugs in the deployed path.

## MEDIUM

**V-48.** Peak equity persisted as REAL and seeded from `max_position_notional_usdt` when `initial_balance` is omitted — config 50k vs equity 10k → instant spurious 80% "drawdown" trip; `manual_reset()` erases drawdown history. `risk/manager.py:178, 245, 338, 267`. _(Numerical Precision; Stage 0)_

**V-49.** No cautious rounding anywhere: `_loss_pct` divides at default `ROUND_HALF_EVEN` (`risk/manager.py:451-455`); threshold decisions depend on banker's rounding rather than deliberate quantize. _(claude.md:18)_

**V-50.** Scale-out reverse-engineers risk from TP distance assuming RR=2 hardcoded (`execution/paper_trader.py:384-389`); RR is config (`manager.py:48`). Sells get a fabricated flat 2% "risk"; the whole scale-out/trailing block is long-only. _(claude.md:61 sizing ownership)_

**V-51.** Sizing ignores fees/slippage — realized loss at stop systematically ~20-50% over the risk budget (`risk/manager.py:395-405` vs fees applied at `paper_trader.py:246, 318, 313-315`). _(proposed 1C-2)_

**V-52.** Stop fills modeled exactly at stop price — no gap-through (`risk/manager.py:439-441`, `backtesting/engine.py:200-202`). Tail-risk losses and max-drawdown stats understated exactly where it matters. _(proposed 1C-5)_

**V-53.** `/force_sell` closes at _entry_ price ("approximation" comment) — the emergency exit records PnL ≈ 0 regardless of reality, blinding daily-loss/drawdown checks (`notifications/telegram_bot.py:248-258`). _(claude.md:30 audit accuracy)_

**V-54.** Blocking SQLite/execution calls inside async Telegram handlers — emergency commands can stall 10 s behind `TradeLogger._lock` (`telegram_bot.py:166, 195, 249, 258`; grep: no `to_thread`/`run_in_executor` anywhere). _(claude.md:75)_

**V-55.** `TelegramBot.stop()` is a silent no-op: un-awaited coroutine + bare `except: pass` + `while True` keep-alive with no stop event; logs "stopped" anyway (`telegram_bot.py:100-107, 136-138`). Shutdown ordering in the logs is a lie. _(claude.md:78)_

**V-56.** Alert delivery single-shot with content-dependent failure: unescaped dynamic Markdown → Telegram 400 → downgraded to a warning, alert lost — most likely for `send_error` messages full of underscores (`telegram_bot.py:400-414, 290-292, 355-358`). _(proposed 1D-8)_

**V-57.** `poll_signal` raises on malformed frames instead of failing safe, and the drain loop discards the whole batch of earlier _valid_ messages first (`core/zmq_subscriber.py:131-137, 172, 184-185`); the test enshrines the crash (`tests/test_core/test_zmq.py:264-281`, named `test_invalid_json_does_not_crash` yet asserting `pytest.raises`). Conversely `.get()` defaults elsewhere mask missing fields into silent HOLDs. _(claude.md:34, :78)_

**V-58.** Graceful shutdown: `sys.exit(0)` in the signal handler while APScheduler workers may be mid-`_close_position` and `scheduler.shutdown(wait=False)` (`run_decoupled_execution.py:214-228`) — a half-written trade close violates persist-before-complete. Publisher bind failure (`zmq_publisher.py:59`) and intelligence SIGTERM (`run_decoupled_intelligence.py:115-118`) unhandled. _(claude.md:26, :78)_

**V-59.** Three divergent Sharpe implementations and three win/loss classifications (`risk/walk_forward.py:203-217` vs `backtesting/engine.py:373` vs `dashboard/analytics.py:132-147`; `pnl <= 0` vs `pnl < 0` vs `p > 0` at `trade_logger.py:348, 381`, `walk_forward.py:157`). The Stage 5 gate and the go-live criterion are numerically incomparable. _(proposed 1C-4)_

**V-60.** `MLPredictor.predict` silently drops missing trained features and silently predicts on a stale (older) row when the newest has NaNs (`ai/ml_predictor.py:281-286`); non-atomic 3-file dumps + unguarded `joblib.load` on the boot path (`:214-216, 246`); `Optional[object]` type hints erase the contract (`:132-133`). _(claude.md:78; :57)_

**V-61.** Ad-hoc `PRAGMA table_info` + `ALTER TABLE` migrations duplicated in two components with no version table; two processes racing the same file's schema at startup can abort boot (`storage/trade_logger.py:106-114`, `risk/manager.py:174-178`). _(proposed 1B-6)_

**V-62.** Strategy pair identity split: constructor-bound models vs per-call `pair` argument — `get_signal(df_eth, "ETH/USDT")` on a BTC-constructed strategy scores ETH with BTC models, no check (`strategies/ai_combined.py:41-57` vs `:156, 172-173`). _(claude.md:9)_

**V-63.** Incremental indicators: `window_size < period` silently bricks the indicator forever after first resync — no constructor validation (verified by execution; `indicators/incremental.py:34-37, 80-84`; constraint documented only in a test comment). Resync also resets `_ticks` to the window length, so the configured resync cadence silently drifts (`:45-57`; verified). _(Stage 1)_

**V-64.** Module-level side effects and mutable singletons: `.env` read at import (`config/settings.py:67`), `MODEL_DIR.mkdir` on import (`ai/ml_predictor.py:46-47`), global `event_bus` (`core/event_bus.py:58`), registry silently overwrites duplicate names (`strategies/registry.py:52-54`), vestigial `neurontrade/__init__.py:5` importing settings. _(proposed 1D-6)_

**V-65.** Live trade-gating thresholds hardcoded and drifting: RSI<30, 1.5×vol, funding ≤0.0005, imbalance ≤1.5, buy threshold 0.25-vs-0.20 duplication, dead shadowed class constants (`strategies/ai_combined.py:61-62, 206-219`; `ai/signal_combiner.py:111-119, 174-183`). _(claude.md:71 principle)_

**V-66.** Blocking I/O + bare `print()` in the decision path: 4+ REST calls per `get_signal`, `print_summary()` multi-line prints bypassing structured logging (`ai_combined.py:172-234, 257`; `signal_combiner.py:70-94`; `ml_predictor.py:62-75`). _(claude.md:75; :29)_

**V-67.** `close_candle` timestamp bugs: `ts = open_ts_ms or now_ms` treats epoch-0 as absent, and the documented fall-back to first-trade timestamp never happens — candles stamped with close-time wall clock can be dropped/misplaced by dedup-by-index merging (`data/tick_aggregator.py:145-172, 252`). Synthesized candles can carry price 0.0 into history → `-inf` log-returns (`:173-182`; `preprocessor.py:99-100`). _(claude.md:9, :34)_

**V-68.** Preprocessor: unlimited `ffill` silently manufactures hours of flat prices after an outage; no zero-close guard before `log(close/close.shift())`; docstring advertises outlier removal that doesn't exist (`data/preprocessor.py:74-113`, docstring `:21`). _(claude.md:78)_

**V-69.** Malformed API payloads escape as raw `KeyError`/`ValueError`/`JSONDecodeError` — only `httpx.HTTPError`/ccxt errors are translated (`data/sentiment_fetcher.py:46-58`, `data/news_fetcher.py:81`, `data/fetcher.py:137-147`; `fetch_ticker` also passes `None` prices downstream unvalidated). _(claude.md:34)_

**V-70.** Dashboard: connections never closed (`with conn` commits, doesn't close — one leak per Streamlit rerun starving the writer; `dashboard/analytics.py:22-25` + call sites), `check_same_thread=False` with no locked-DB handling. _(claude.md:78)_

**V-71.** ✱ DB path split three ways + CWD-relative default: `storage/trade_logger.py:83`, `config/settings.py:39-41` (read by nobody), `dashboard/analytics.py:19`, vs claude.md's `./data/`. Launching from a different directory silently creates a fresh empty DB — a de facto state reset (`trade_logger.py:90-91` auto-mkdirs it). _(claude.md:12)_

**V-72.** ZMQ test quality: hardcoded port 15555, 20-50 ms real-socket sleeps (flaky/parallel-unsafe), `object.__new__` bypasses constructors so `_connect`/SUBSCRIBE logic is never executed, leaked contexts, and no test drives the polling loop or exit job — the suite could not have caught V-3, V-4, or V-12 (`tests/test_core/test_zmq.py:114-120, 145-168, 199, 213-298`). _(claude.md:8)_

## LOW

**V-73.** `health_check.py` never inspects the DB, breaker state, or open-trade consistency; prints "All systems go!" regardless (`scripts/health_check.py:119-137`). — **V-74.** `SimpleCache`: expired entries evicted only on same-key `get`, no max size, stores live DataFrame references, and is dead code — `DataFetcher` never uses it despite the rate-limit-protection docstring (`data/cache.py:39-41, 99`). — **V-75.** `OrderBookManager.__repr__` raises `ValueError` whenever a spread exists — f-string format-spec bug `{snap.spread:.4f if ...}` (`data/order_book.py:203-210`; verified). — **V-76.** Dashboard `pd.to_datetime` without `utc=True` at `analytics.py:195` (naive), unlike `:62-63, 179`. — **V-77.** `published_at` returned as raw unparsed string, breaking the tz-aware-datetime contract every other fetcher honors (`data/news_fetcher.py:97`). — **V-78.** `merge_into` is advertised in the class usage example but raises `NotImplementedError` (`data/tick_aggregator.py:94, 187-213`). — **V-79.** `get_historical_data`: unbounded `while True` pagination with blocking `time.sleep`, no per-batch error translation (`data/fetcher.py:307-338`). — **V-80.** Walk-forward doc/code contradictions: "annualised" vs "not annualised" (`risk/walk_forward.py:71` vs `:205`); zero-variance → Sharpe 0.0 even with negative mean (`:216-217`); `<= now` vs documented "strictly before" (`:128`). — **V-81.** Backtest metric defects: `entry_idx` recorded at exit so durations are impossible (`engine.py:223, 422`), `ZeroDivisionError` on <2 rows (`:369`), float `==` on signal (`:243`). — **V-82.** SUB topic filter is prefix-match and the received topic frame is discarded unchecked — a future `signal_v2` topic silently becomes trade signals (`core/zmq_subscriber.py:116, 134`). — **V-83.** Event bus: async handlers silently no-op (un-awaited coroutine), `handler.__name__` breaks on partials, unsynchronized subscriber list — and the bus is dead code (`core/event_bus.py:27, 44, 48, 58`). — **V-84.** `MACrossoverStrategy` confidence columns never exist for EMA mode/custom periods → hardcoded 0.60 always (`strategies/ma_crossover.py:72-75` vs `technical.py:71-79`). — **V-85.** `BollingerBounce` can emit confidence −4.0 (no lower clamp, `BB_pct` default 0.5) (`strategies/bollinger_bounce.py:96-100`). — **V-86.** Bare-`dict` contracts at every module boundary (`strategies/base.py:76`, `technical.py:255`, `llm_agent.py:146`, `event_bus.py:34`, `manager.py:325` untyped `db=None`). — **V-87.** Vestigial nested `neurontrade/` package: imported by nothing, triggers `.env` read on import, breaks if ever installed as a distribution (`neurontrade/__init__.py:5`). — **V-88.** Audit-trail misattribution: "Claude analysis" log lines and docstrings for what is actually Groq/Llama (`ai/llm_agent.py:180, 259, 285`; `ai_combined.py:2-9`). — **V-89.** Test coverage gaps: no tests for `LLMAgent`, `AICombinedStrategy` (the production strategy), `BollingerBounce`, `FeatureEngineer`, `event_bus`, settings validation; indicator tests never assert warm-up semantics (which would have caught V-40/V-42).

---

## Verified positives (no violation found)

- Timestamps in the persistence layer are consistently tz-aware UTC ISO-8601 (`storage/trade_logger.py:435-438`; `risk/manager.py:457-462` enforces and coerces).
- `TradeLogger` correctly serializes its own access (RLock + 10 s timeout + rollback-on-error, `trade_logger.py:119-136`).
- **No circular imports** anywhere (full import graph checked; dependency direction is strictly config/core ← indicators/ai/risk ← strategies).
- Incremental EMA/RSI/ATR/ADX are genuinely O(1) per tick with the mandated periodic resync; values match pandas-ta on the validation harness (16/16 indicator tests pass).
- The walk-forward validator itself uses the same `RiskManager` instance as execution in the decoupled node (`run_decoupled_execution.py:58`) and its test suite genuinely covers both gates, the leakage guard, and no-auto-reset.
- Fees/slippage constants match between backtest and paper trader (0.001/0.0005); both are bar-based.
- The ZMQ fail-safe default (HOLD on no/stale signal) is the right _direction_ — the findings above are about it being silent and re-triggerable, not about the default itself.

## Suggested fix order

1. **V-1–V-6** (phantom references + AISignal + ATR case): the system cannot run at all until these are fixed; then add the 1D-1 static-contract gate so this class of bug cannot ship again.
2. **V-9/V-10/V-13/V-24/V-25** (breaker persistence, balance recovery, races, auto-resume): restore the Stage 0 guarantees.
3. **V-7/V-8** (Decimal path, short accounting): correctness of every number the system records.
4. **V-12/V-19/V-16** (signal idempotency, kill switch, Telegram auth): close the live-execution safety holes.
5. Everything else in severity order, with the claude.md rule additions from Section 1 adopted first so fixes land against the corrected standard.
