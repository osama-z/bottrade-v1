# Public demo review — 2026-10-03

Scope: the local public repository `osama-z/bottrade-v1`, its supported paper
entrypoint, risk persistence, setup commands, tests and portfolio-facing docs.
The private development repository was not modified. This is a code and local
validation review, not a profitability assessment or a complete formal audit.

## Fixes made

| Finding | Change | Code / evidence |
| --- | --- | --- |
| Circuit-breaker initialization referenced nonexistent `self._table`, preventing SQLite-backed paper startup | Use the declared `_TABLE` constant consistently for migration, load, save and peak updates | [risk/manager.py:361](../risk/manager.py#L361), [migration:401](../risk/manager.py#L401); existing persistence tests now pass |
| Direct `run_backtest.py` command failed with `ModuleNotFoundError: config` | Add the project root before project imports; use credential-free fetching | [scripts/run_backtest.py:14](../scripts/run_backtest.py#L14), [fetcher:34](../scripts/run_backtest.py#L34); subprocess command test from a separate directory |
| Copying the example config enabled integrations with fake tokens | Leave credentials blank; identify optional integrations | [.env.example:1](../.env.example#L1), [health check:55](../scripts/health_check.py#L55); fresh-copy health-check test |
| Health checks skipped public exchange access without a key, always contacted sentiment services, and returned success even when checks failed | Credential-free exchange check, `--offline`, and nonzero exit on failed checks | [scripts/health_check.py:107](../scripts/health_check.py#L107), [CLI:214](../scripts/health_check.py#L214), [exit:251](../scripts/health_check.py#L251) |
| Deployment bootstrap selected the private repo/branch; the local shell copy had CRLF line endings and failed Bash parsing | Default to public `bottrade-v1/main`, require Python >=3.12, remove key requirement, normalize this shell file to LF | [deploy/setup_server.sh:15](../deploy/setup_server.sh#L15), [version check:28](../deploy/setup_server.sh#L28); `bash -n` passes |
| Architecture docs described real testnet orders even though executors are stubs | Document disabled paths and the actual public scope | [docs/ARCHITECTURE.md:1](ARCHITECTURE.md#L1); [executor refusal:49](../execution/live_executor.py#L49), [testnet refusal:18](../execution/testnet_trader.py#L18) |
| Two unused imports failed the configured lint rules | Remove the unused imports without changing execution behavior | [execution/live_executor.py:12](../execution/live_executor.py#L12), [smart-order tests:1](../tests/test_execution/test_smart_order.py#L1) |
| README mixed production/private execution claims with the public demo and overstated precision | Rebuild around supported behavior, runnable onboarding, code-linked evidence and explicit limitations | [README](../README.md) |

Twelve public-demo checks in [tests/test_public_demo.py](../tests/test_public_demo.py#L16)
cover executor refusal, disabled command exit statuses, direct backtest imports,
example-config offline setup and failed health-check exit status.

The public configuration now rejects `PAPER_TRADING=false` via
[config/settings.py:127](../config/settings.py#L127), and inherited live-mode
labels were removed from startup logs and Telegram messages. Regression tests
cover false boolean/string settings and accepted paper configuration. Static
search found no built-in exchange order creation, withdrawal or transfer calls.
The optional recovery utility has injected-executor cancellation calls, but is
not invoked by either paper entrypoint and has no usable exchange executor in
this demo.

## Remaining issues

### High: paper trades and account balance do not commit together

Opening a position commits the trade at
[execution/paper_trader.py:351](../execution/paper_trader.py#L351), then changes and
saves cash at [line 369](../execution/paper_trader.py#L369). Closing commits the
trade at [line 446](../execution/paper_trader.py#L446), then credits and saves cash
at [line 456](../execution/paper_trader.py#L456). The storage methods commit
independently: [trade open:313](../storage/trade_logger.py#L313),
[trade close:394](../storage/trade_logger.py#L394),
[balance:422](../storage/trade_logger.py#L422).

A crash or failed balance write between these commits can leave a durable
position change with stale cash. Startup loads that stale cash at
[execution/paper_trader.py:104](../execution/paper_trader.py#L104).

**Reproduced locally:** create a 1-unit long at 100 with persisted cash 9,900,
then inject an exception in `save_balance` during a close at 110. The trade is
persisted as closed; persisted cash remains 9,900. A new PaperTrader restores
9,900 and has no open position from which to recover the missing credit.
This fault injection used a temporary database, not the demo account.

**Recommended next change:** give the storage owner a transaction that writes
trade state and account balance together, then mutate in-memory cash only after
that transaction succeeds. Cover entry, exit and partial scale-out paths with
failure-injection/restart tests. This needs a focused storage/execution change,
including consideration of the optional Postgres backend; it is not fixed by
README wording. Do not describe the current system as crash-atomic accounting.

### Precision: persisted monetary values use binary floats

The trade schema uses SQLite `REAL` for quantities, prices and PnL at
[storage/trade_logger.py:39](../storage/trade_logger.py#L39). Paper execution
converts calculated amounts to floats at
[execution/paper_trader.py:354](../execution/paper_trader.py#L354) and
[balance persistence:370](../execution/paper_trader.py#L370).
Decimal-based calculation is useful, but does not imply exact decimal persistence.
The README now states this limitation. A future precision migration should use
an explicit decimal-string or scaled-integer storage contract with migration tests.

### Optional backend: ClickHouse is an interface stub

[storage/market_data.py:143](../storage/market_data.py#L143) declares
`ClickHouseMarketDataStore`; its write methods raise `NotImplementedError` at
[line 159](../storage/market_data.py#L159). Avoid presenting this as a working
backend. SQLite is the supported quickstart path.

### Research: the historical time split is not an untouched holdout

[docs/STRATEGY_NOTES.md:110](STRATEGY_NOTES.md#L110) says the strategy was selected
on the full 730-day window before splitting it. The weaker newer-half performance
is a useful negative result, but the split cannot prove unbiased prospective
out-of-sample performance. The README labels the results as recorded historical
experiments and preserves that qualification.

## Verification

- **709 passed, 4 deselected**, with five pandas-related warnings:

  ```bash
  python -m pytest tests/ -q -p no:cacheprovider -k 'not TestRoundTrip'
  ```

- The four excluded tests belong to
  [tests/test_core/test_zmq.py:195](../tests/test_core/test_zmq.py#L195).
  A prior run aborted inside the native ZeroMQ TCP bind at
  [line 205](../tests/test_core/test_zmq.py#L205) in this restricted environment.
  The other ZeroMQ tests passed in the final run. This does not establish why
  the native abort occurs on unrestricted machines; full socket validation remains open.
- `ruff check .` passes.
- `bash -n deploy/setup_server.sh` passes; the bootstrap was not executed.
- The paper bot object graph constructs using `STRATEGY=trend_following`, blank
  Telegram credentials and a temporary SQLite path. The scheduler/trading loop
  was not started.
- New subprocess tests exercise the copied example configuration and direct
  command imports without relying on `PYTHONPATH`.

Python checks reused the existing Python 3.12 project environment. A clean install
from the lockfile was not performed. External provider connectivity, historical
results, optional database services and an unattended paper run were not verified.

The working copy already contained widespread CRLF-only Git differences. Those
were preserved; only the edited bootstrap shell script was normalized for Bash.
No commits, pushes, deployment, or real orders were performed by this review.
