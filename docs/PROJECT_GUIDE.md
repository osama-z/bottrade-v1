# Project map

Start with the [README](../README.md) for the supported paper demo. This guide
explains which files are part of that path and which are optional experiments.

## Repository layout

```text
bottrade-v1/
├── README.md                 Demo overview and quickstart
├── LICENSE                   MIT license for project-owned code
├── requirements.txt          Direct dependency ranges
├── requirements.lock         Pinned environment snapshot used by CI
├── config/                   Settings, symbols and logging
├── data/                     Market data, cache, preprocessing and book utilities
├── indicators/               Batch and incremental indicators
├── strategies/               Strategy contracts, implementations and registry
├── risk/                     Sizing, limits, breaker and metrics
├── execution/                Paper trading, controls and disabled exchange stubs
├── storage/                  SQLite plus optional storage experiments
├── core/                     Candle timing and optional ZeroMQ messaging
├── notifications/            Optional Telegram integration
├── ai/                       Optional model, regime and sentiment research
├── backtesting/              Historical simulation and validation helpers
├── dashboard/                Analytics helpers; no bundled web dashboard
├── scripts/                  Demo commands and explicit research tools
├── tests/                    Unit and regression tests
├── deploy/                   Optional Linux service templates/bootstrap
├── docs/                     Architecture, experiments and review notes
│   └── archive/              Historical illustration, labeled as historical
└── .github/workflows/        CI
```

Logs, databases, downloaded candles, caches, environment files and trained models
are local outputs, ignored by Git. They do not belong in a portfolio source
commit. `.env.example` is the intentionally tracked configuration template.

## Main demo commands

| File | Purpose |
| --- | --- |
| [health_check.py](../scripts/health_check.py) | Local setup validation; optional network checks |
| [run_live.py](../scripts/run_live.py) | Primary paper loop; historical filename, no live orders |
| [paper_status.py](../scripts/paper_status.py) | Inspect the simulated account |
| [run_backtest.py](../scripts/run_backtest.py) | One historical simulation |
| [strategy_lab.py](../scripts/strategy_lab.py) | Compare strategies, pairs and timeframes |

`run_testnet.py` and `testnet_smoke.py` intentionally remain as refusal stubs.
They make old commands fail clearly rather than accidentally suggesting that
exchange execution is available.

## Optional tools

| Area | Files | Requirements / scope |
| --- | --- | --- |
| Data and models | `download_data.py`, `train_model.py`, `train_regime.py` | Historical data, network access and local model artifacts |
| Research validation | `walk_forward.py`, `validate_regime.py`, `validate_triple_barrier.py`, `validate_funding_carry.py`, `component_attribution.py` | Experiment-specific inputs; not the default demo path |
| Indicator validation | `validate_incremental_indicators.py` | Numerical comparison utility |
| Shadow comparison | `run_shadow_stress.py`, `shadow_parity.py` | Data access and shadow records |
| Decoupled simulation | `run_decoupled_intelligence.py`, `run_decoupled_execution.py` | Two processes and local ZeroMQ sockets; alternative to the main loop |
| Feed smoke test | `ws_book_smoke.py` | Network/WebSocket access |
| Optional LLM check | `test_groq.py` | Explicit Groq API check; needs a key and may consume API quota; never executes orders |

The SQLite implementation is the default. Postgres/Timescale paths need their
own driver and services. ClickHouse is an interface stub. Keep these examples
clearly qualified rather than describing every optional backend as deployed.
The generic `recovery.py` helpers are tested using injected fakes; they do not
provide a usable live exchange executor in this repository.

## Cleanup performed

- Removed `scripts/test_phase3.py` and `scripts/simulate_institutional_flow.py`:
  old one-off checks used a hardcoded private-checkout import path and were
  superseded by automated tests. See the original
  [phase check at line 4](https://github.com/osama-z/bottrade-v1/blob/de502f67880eb57ceb1eb3aa365ff03cc6ad7e4b/scripts/test_phase3.py#L4)
  and [manual audit at line 8](https://github.com/osama-z/bottrade-v1/blob/de502f67880eb57ceb1eb3aa365ff03cc6ad7e4b/scripts/simulate_institutional_flow.py#L8).
  Existing coverage remains in [institutional tests](../tests/test_institutional.py#L1)
  and [paper guard tests](../tests/test_paper_trader_guards.py#L1).
- Made the retained Groq check use its own checkout, with execution under a
  `main` guard so importing it does not contact the API.
- Moved the unlinked root `architecture.html` to
  [docs/archive/architecture.html](archive/architecture.html), with a historical
  banner. The README diagram and [architecture guide](ARCHITECTURE.md) are current.
- Removed unused direct requirements for Plotly, mplfinance, Streamlit,
  TextBlob, newspaper3k, pytest-asyncio and mypy. Added SciPy explicitly because
  source code imports it. The dashboard package is analytics, not a web UI.
- Replaced personal paths/users in service templates with `@PROJECT_ROOT@`
  and `@RUN_USER@`, rendered by the bootstrap script.

The numbered `test_tier*` files remain because they are regression coverage,
not unused deliverables. Renaming them provides little value and would obscure
references in the review history. Empty package markers and model `.gitkeep`
files establish package/output locations and are intentional.

## Remaining cleanup boundary

`requirements.lock` is the previously tested environment snapshot and includes
packages beyond the direct manifest. It was retained to avoid silently changing
CI's dependency graph without a clean-install test. A future dependency reduction
should regenerate that snapshot in a fresh Python 3.12 environment and run the
whole suite; import scanning alone cannot establish every transitive dependency.

See the [public review](PUBLIC_DEMO_REVIEW.md) for the reproduced paper-account
transaction gap and other limitations. Structural cleanup does not resolve that
storage correctness issue.

## License

The project's [LICENSE](../LICENSE) contains the standard MIT text and the
copyright notice `2026 osama-z`. It is linked from the README; no license terms
were changed. MIT permits commercial reuse and requires preservation of its
notice; the paper-only design is a software capability limit rather than an
extra license condition. [Canonical MIT text](https://opensource.org/license/mit).

See [third-party notices](THIRD_PARTY_NOTICES.md) for the dependency inventory
and the boundary of this license check.

## Cleanup verification

The public-demo, deployment-security and analytics checks returned **42 passed**.
Lint, bootstrap shell parsing, service-template rendering and 74 local documentation
link targets passed. The earlier runtime review returned **709 passed, 4 deselected**;
that full run preceded this structural cleanup. No bootstrap, API request, push or
deployment was performed during the cleanup.
