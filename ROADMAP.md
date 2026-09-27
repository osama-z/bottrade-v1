Phase 1: Edge Discovery & Validation (The "Engine")
Objective: Strip away unvalidated intelligence, build a backtest that reflects harsh market reality, and prove a structural edge exists.

Task 1.1: Decouple Noise from Execution Path
What to do: Remove the LLM agent (llm_agent.py) and VADER sentiment analyzer (sentiment_analyzer.py) from the live signal_combiner.py execution path.
Details: Do not delete the code; keep it for showcase. Modify signal_combiner.py so it no longer waits on or weighs these signals for live trades. The combiner must now only route validated mathematical signals.
Task 1.2: Implement Realistic Transaction Cost & Slippage Models
What to do: Upgrade the backtest execution simulator to penalize every trade with real-world friction.
Details:
Exchange Fees: Apply Binance VIP0 taker fees (0.04%) and maker fees (0.02%) based on the order type used.
Slippage Model: Do not assume fills at the close price. Calculate slippage using your order_book.py data: fill_price = signal_price + (order_size / top_N_levels_liquidity) * impact_factor.
Latency Penalty: Add a mandatory 200ms to 2000ms delay between signal generation and simulated fill price.
Funding Rates: If using perpetuals, accumulate or deduct funding payments every 8h based on position direction and funding rate at that epoch.
Task 1.3: Implement Combinatorial Purged Cross-Validation (CPCV)
What to do: Replace basic walk-forward validation with CPCV to eliminate data leakage and overlapping labels.
Details: Implement López de Prado's CPCV. Purge observations around training/test boundaries. Generate multiple out-of-sample paths to produce a distribution of Sharpe ratios, not a single point estimate. Calculate the Deflated Sharpe Ratio to penalize the result based on how many strategy variations were tested.
Task 1.4: Develop a Structural-Edge Strategy (Funding Rate Carry)
What to do: Implement a delta-neutral funding rate carry strategy in strategies/.
Details:
Monitor perpetual funding rates.
When annualized funding exceeds a threshold (e.g., >30% APR), execute a delta-neutral position: long spot, short equivalent perp.
Hold for the funding epoch (or until funding normalizes).
Run this through the CPCV backtest. Requirement: It must yield a positive Deflated Sharpe after fees and slippage before proceeding.
Task 1.5: Refactor ML Target to Triple-Barrier Method
What to do: Replace the AUC 0.50 next-candle direction predictor in ml_predictor.py.
Details:
Implement the Triple-Barrier method: label outcomes as +1 (hit upper profit barrier), -1 (hit lower stop barrier), or 0 (hit time stop).
Train a new XGBoost model on this target.
Implement a secondary Meta-Labeling model: the first model predicts direction, the second model predicts whether to take the trade. Size positions only when the meta-label is 1.
Phase 2: Risk Management Hardening
Objective: Transition from hobby-bot risk controls to institutional-grade capital protection.

Task 2.1: Implement Volatility-Targeted Position Sizing
What to do: Replace fixed-fractional sizing with fractional Kelly + volatility targeting.
Details:
Calculate the strategy's edge and odds to determine Kelly fraction. Multiply by 0.25 (Quarter-Kelly).
Calculate target volatility (e.g., 0.5% daily vol per position). Use ATR or realized vol to scale position size: size = target_vol / asset_vol.
Ensure risk/manager.py uses Decimal math for these calculations to prevent float drift.
Task 2.2: Build Correlation-Aware Heat Caps
What to do: Upgrade the 6% portfolio heat cap to account for asset correlation.
Details:
Maintain a rolling 30-day correlation matrix of traded assets.
Calculate portfolio heat as sum(individual_heats) * (1 + average_correlation).
If two highly correlated assets (e.g., BTC and ETH) are both long, the risk/manager.py must reject the second trade or reduce its size to keep portfolio heat under the cap, not just nominal heat.
Task 2.3: Automate Deleveraging Drawdown Schedules
What to do: Program automatic size reductions based on equity drawdowns.
Details:
Track rolling peak equity.
At -5% DD: Cut position sizing multiplier to 50%.
At -10% DD: Cut position sizing multiplier to 25%.
At -15% DD: Trigger circuit breaker, halt all new trades, flatten existing positions. Enforce a mandatory 24-hour cooldown in code before the system can be manually reset.
Phase 3: Execution & Reliability Hardening
Objective: Ensure the bot interacts with the exchange flawlessly, recoverable from any crash, network drop, or API error.

Task 3.1: Implement Smart Order Execution
What to do: Upgrade live_executor.py to minimize market impact and capture maker fees.
Details:
Implement Post-Only (GTX) orders for entries to capture maker rebates. If the order is at risk of crossing, reject and re-price.
For orders > $500, implement a basic TWAP (Time-Weighted Average Price) slicing algorithm to split the order into chunks over 1–10 minutes.
Cap individual order sizes at 25% of the top-5 levels of order book liquidity.
Task 3.2: Build Dead-Man's Switch & Reconciliation Loop
What to do: Protect against process crashes leaving orphaned orders on the exchange.
Details:
Implement a heartbeat mechanism that pings Binance every 30 seconds. If the bot process dies, Binance should auto-cancel all open orders.
Build a reconciliation thread that runs every 60 seconds: fetches live exchange positions and open orders, compares them to the bot's internal SQLite state. If divergence occurs, halt trading and push a critical alert.
Task 3.3: Upgrade Idempotency and State Recovery
What to do: Ensure the bot can restart mid-trade without duplicating or missing orders.
Details:
Every action (entry, scale-out, stop-move) must generate a unique clientOrderId, written to SQLite before the API call is made.
On startup, run_live.py must check for pending actions. If found, query exchange for status. If filled, update local DB. If unfilled, cancel or evaluate based on current market state. Never blindly re-send an order.
Phase 4: Infrastructure & Observability
Objective: Move from local scripts to a monitored, resilient, cloud-native deployment.

Task 4.1: Migrate Storage Stack
What to do: Replace SQLite with a scalable time-series stack.
Details:
Migrate trade/signal/risk logs to PostgreSQL (using WAL mode equivalent via WAL-G for backups).
Migrate market data (ticks, OHLCV, order book snapshots) to TimescaleDB or ClickHouse.
Ensure all DB writes are ACID-compliant and use connection pooling.
Task 4.2: Containerization & Cloud Deployment
What to do: Package the bot for resilient cloud execution.
Details:
Create a Dockerfile with health checks.
Deploy to a cloud region physically close to Binance servers (e.g., AWS ap-northeast-1, Tokyo) to minimize API latency.
Use systemd or Kubernetes for auto-restart on failure (Restart=on-failure).
Remove all API keys from .env files. Store secrets in AWS Secrets Manager or HashiCorp Vault.
Task 4.3: Implement Telemetry & Alerting Pipeline
What to do: Build dashboards and high-priority alerting.
Details:
Instrument the Python code to expose Prometheus metrics: live P&L, open exposure, API latency, WebSocket health, order rate.
Build Grafana dashboards visualizing these metrics in real-time.
Integrate PagerDuty or OpsGenie for critical alerts (circuit breaker tripped, reconciliation failed, API key invalid). Telegram is reserved for informational alerts only.
Phase 5: The Live Ramp Protocol
Objective: Systematically deploy real capital, using strict statistical gates to proceed.

Task 5.1: Shadow Mode Validation
What to do: Run the finalized strategy without executing real orders.
Details: Run the bot pulling live data. Compare the signals generated against what the backtest would have generated on the exact same live data. Gate to pass: Signal divergence must be < 5%. If higher, fix data parity issues before proceeding.
Task 5.2: Real-Money Micro-Capital Run
What to do: Execute with $100–$500 real capital.
Details: The goal is strictly to measure execution friction, not to make money.
Track expected fill price (from backtest slippage model) vs. actual live fill price.
Track API reliability.
Gate to pass: Run 50+ trades. Average slippage must be within 2x of modeled slippage.
Task 5.3: Scaled Execution & Kill Criteria
What to do: Scale capital while strictly monitoring live vs. backtest performance.
Details:
Double position sizes only if live Sharpe ratio is within 30% of backtest Sharpe ratio.
Hard Kill Criteria: If drawdown exceeds 10% before reaching full size, OR if live underperforms backtest by >50%, halt the bot. Do not restart until root cause is fully diagnosed and fixed.
Cap at 1/3 of ultimate intended size until 3 months of stable live performance is recorded.
Phase 6: Compliance & Accounting
Objective: Ensure legality and tax readiness.

Task 6.1: Implement Trade-Lot Accounting
What to do: Build a tax-compliant ledger into the storage layer.
Details: Every trade must be assigned a lot ID. Support FIFO (First-In-First-Out) and Specific ID accounting methods. Calculate realized and unrealized P&L per trade for tax reporting. Ensure logs are immutable and retained for 7 years.
Task 6.2: Regulatory & Anti-Manipulation Audit
What to do: Verify the bot operates within legal bounds.
Details:
Review the execution logic to ensure it does not inadvertently engage in spoofing (placing orders with intent to cancel) or layering.
Review local jurisdiction laws regarding personal algorithmic crypto trading and exchange API usage terms. Ensure KYC is fully cleared on the live exchange account.