# Deployment & Operations

## Deploying to a VPS (recommended for the 30-day paper run)

**Server:** any small Ubuntu 22.04/24.04 VPS — 2 vCPU / 2–4 GB RAM is plenty
(4h candles; inference is light — Hetzner CX22, DigitalOcean basic, or similar,
~€5/month). Pick an **EU region (never US — Binance geoblocks US IPs)**; latency
is irrelevant at 4h cadence. A VPS beats running at home: static IP for the
Binance whitelist, no sleep, no ISP resets — the 30-day uptime criterion is
realistic there.

**One-command bootstrap** (fresh server, as a sudo-capable user):

```bash
curl -fsSL https://raw.githubusercontent.com/osama-z/bottrade/version1/deploy/setup_server.sh | bash
```

The script is idempotent (re-run it to update): installs system deps, clones
the repo to `~/neurontrade`, builds the venv from `requirements.lock`, runs the
full test suite, templates the systemd units for the server's paths/user, and
prints the remaining manual steps.

### Paper run — the default path (`neurontrade`, single process)

`neurontrade.service` runs `scripts/run_live.py`: one process that fetches,
runs the strategy chosen by `STRATEGY`, and executes through the PaperTrader —
honoring `STRATEGY=trend_following` and deciding on **closed** candles at
candle-close boundaries.

1. Fill `~/neurontrade/.env` — Binance key (Reading-only, **IP-whitelisted to
   the server's IP**, which the script prints), plus `STRATEGY=trend_following`,
   `DEFAULT_TIMEFRAME=4h`, Telegram, (Groq/NewsAPI optional). `chmod 600 .env`.
2. `.venv/bin/python scripts/health_check.py` → all green.
3. `sudo systemctl enable --now neurontrade`
4. `journalctl -fu neurontrade` — watch it live.
5. `.venv/bin/python scripts/paper_status.py` — one-page status, anytime.
6. Run the kill-switch drill (below) **before leaving it unattended**.

`trend_following` needs **no trained models**. Only `STRATEGY=ai_combined`
requires `scripts/train_model.py` + `scripts/train_regime.py` first.

**Operating it from your phone:** Telegram gives you `/status`, `/pause`,
`/force_sell` and push alerts (trades, breaker trips, feed outages); GitHub
holds the code/issues; for logs, any SSH app (e.g. Termius) into the VPS —
`journalctl -fu neurontrade`.

**Updating the running bot:** merge to `version1` on GitHub, then on the
server: `bash ~/neurontrade/deploy/setup_server.sh && sudo systemctl restart
neurontrade`. SIGTERM is the cooperative shutdown path — in-flight position
writes complete before exit.

### Advanced: two-process ZMQ path (optional, use INSTEAD of `neurontrade`)

`neurontrade-intelligence` (signals → ZMQ) + `neurontrade-execution`
(ZMQ → PaperTrader) split intelligence and execution into separate processes
for isolation/resilience. Both honor `STRATEGY` and closed-candle timing too.
Enable **one path or the other, never both** (they share the same DB/breaker):

```bash
sudo systemctl enable --now neurontrade-intelligence neurontrade-execution
journalctl -fu neurontrade-execution
```

## Process supervision (systemd)

All units restart on failure. SIGTERM triggers the cooperative shutdown path
(flags only in the handler; in-flight position writes finish before exit), so
`systemctl stop`/`restart` never abandons a half-written trade.

Note: on WSL2, enable systemd in `/etc/wsl.conf` (`[boot] systemd=true`)
or use `supervisord` with equivalent `autorestart=true` programs.

## Kill-switch drill (required before extended paper runs — claude.md)

Run this against the ACTUALLY RUNNING system, not the test suite. Steps below
use the single-process unit `neurontrade`; on the two-process path substitute
`neurontrade-execution` and additionally `systemctl stop neurontrade-intelligence`
first to simulate a hung intelligence core.

1. Confirm at least one open position (`/status` in Telegram, or `paper_status.py`).
2. Fire the switch out-of-band (pick one):
   - `touch KILL_SWITCH` in the project root, or
   - `systemctl kill -s SIGUSR1 neurontrade`
3. Verify, within ~5 seconds:
   - all open positions closed with `exit_reason='kill_switch'`
     (`SELECT * FROM trades ORDER BY id DESC LIMIT 5;`),
   - breaker latched: `SELECT * FROM circuit_breaker_state;` → `tripped`,
   - a `kill_switch` row in `risk_events`,
   - Telegram received the flatten summary.
4. Verify the latch survives restart: `systemctl restart neurontrade`
   → breaker still `tripped`; with the flag file still present the switch
   re-fires on boot (expected).
5. Recover: remove the flag file, manually reset the breaker (deliberate,
   operator-only action), restart the service.

Record the drill date/result in the trade journal; the paper→live
transition checklist requires it.

## Out-of-band controls summary

| Action | Mechanism |
|---|---|
| Halt new trading | circuit breaker (auto: daily loss / drawdown / loss streak) |
| Pause entries | Telegram `/pause` (authorized chat only) |
| Flatten everything | `touch KILL_SWITCH` or `SIGUSR1` — independent of ZMQ & Telegram |
| Resume after trip | operator: remove flag, manual breaker reset |
