# Paper demo deployment and operation

This public repository supports **paper simulation only**. No exchange
credentials are needed, and its live/testnet executors are disabled.
Start with the local [quickstart](../README.md#quickstart); an unattended server
is optional and was not deployed or validated by the public-demo review.

## Optional Linux service

The bootstrap script installs packages and systemd units. Review it before
running it on a machine you administer. It targets Ubuntu with Python 3.12
available and uses the public repository's `main` branch.

```bash
git clone https://github.com/osama-z/bottrade-v1.git
cd bottrade-v1
# Inspect deploy/setup_server.sh, then run if you want a system service.
bash deploy/setup_server.sh
```

By default, the script installs into `~/neurontrade`. `REPO_URL`, `BRANCH`
and `INSTALL_DIR` can override the source and destination. It installs pinned
dependencies, runs tests and templates unit paths for the current user.

1. Inspect `~/neurontrade/.env`. Leave Binance credentials blank; keep
   `PAPER_TRADING=true`, `STRATEGY=trend_following`, `DEFAULT_TIMEFRAME=4h`.
2. Run `.venv/bin/python scripts/health_check.py --offline` from the install
   directory. Omit `--offline` to also check external data services.
3. Start the single-process simulation: `sudo systemctl enable --now neurontrade`.
4. Inspect logs: `journalctl -fu neurontrade`.
5. Inspect account state: `.venv/bin/python scripts/paper_status.py`.

The running simulation requires access to public market-data providers; availability
varies by network and provider restrictions. Optional Telegram controls require
both a bot token and an authorized chat ID. The rule-based default needs no
ML model artifacts, Groq key or news key.

The `neurontrade-intelligence` and `neurontrade-execution` units describe the
advanced two-process ZeroMQ alternative. Do not enable them alongside the
single-process unit: both paths would act on the same simulated account.
Validate this alternative separately before relying on it.

## Kill-switch drill for paper positions

Run this against a simulation with an open position. These commands affect only
paper state in this public demo.

1. Confirm an open position using `paper_status.py` or authorized Telegram `/status`.
2. From the project root, create the flag: `touch KILL_SWITCH`. On Linux, sending
   `SIGUSR1` to the paper process is the alternative.
3. Verify that paper positions close with `exit_reason='kill_switch'`, the
   breaker reads `tripped`, and a kill-switch risk event is recorded.
4. Restart the simulation and verify the halt persists. The flag stays present
   until removed by the operator.
5. To recover, remove the flag and deliberately reset the breaker using the
   operator controls. Restarting alone does not reset it.

For the two-process alternative, stop the intelligence service first and trigger
its execution service to test independence from the intelligence feed.

## Operator controls

| Action | Mechanism |
| --- | --- |
| Halt new simulated entries | Daily-loss, drawdown or consecutive-loss circuit breaker |
| Pause entries | Authorized Telegram `/pause` |
| Flatten paper positions | `KILL_SWITCH` flag or Linux `SIGUSR1` |
| Resume after a breaker trip | Explicit operator reset, after removing any kill-switch flag |

Keep the SQLite database, including its WAL state, when preserving a paper run.
Changing or deleting state files resets the experiment; record such changes in
your research notes. The local review does not establish unattended uptime or
exact decimal persistence across all storage paths.
