# Damico Capital Management

Multi-agent US equities trading system. Design and cost estimate: see the project's technical design doc.

## What is here

A complete long-only trading loop that runs with **no paid services**: simulated market data, the real agents, the Risk engine and a simulated broker, all driven by the orchestrator. Vendor (Alpaca) and IBKR adapters plug into the same interfaces later.

| Path | What it does |
| --- | --- |
| `src/damico/messages.py` | Typed messages (signals, proposals, risk decisions, executions, overrides). Every message carries run, cycle, timestamp, agent, version and the ids of its inputs. |
| `src/damico/orchestrator.py` | Runs one cycle a minute in market hours: analysis stages (parallel within a stage), Strategy, freshness check, Risk, human overrides, Execution, Portfolio. |
| `src/damico/agents/base.py` | Interfaces: `Agent`, `RiskAgent`, `ExecutionAgent`. |
| `src/damico/agents/core.py` | Portfolio, Market (universe filter, breadth), Technical (trend score, ATR stop) and Strategy (breakout entries, trailing-stop exits) agents. |
| `src/damico/risk.py` | Risk engine: risk per trade, position, sector, gross, liquidity and position-count limits; daily, weekly and drawdown circuit breakers. Limits are percentages of a $500,000 `capital_base`. |
| `src/damico/broker.py` | Simulated broker with IBKR Pro Tiered commissions, slippage and order sanity checks. |
| `src/damico/portfolio.py` | Cash, positions, equity marks, closed trades. |
| `src/damico/data/` | Vendor-neutral data interface, synthetic market generator, CSV loader. |
| `src/damico/sim/` | Day-by-day replay through the orchestrator, with a performance report. |
| `src/damico/agents/demo.py` | Stub agents for tests and the demo. |
| `src/damico/overrides.py` | Pause entries, flatten, block symbol, approve-each-trade. |
| `src/damico/store.py` | Message store (in-memory now, Postgres next). |

## Safety rules the orchestrator enforces

- Risk fails closed: an error, timeout, missing decision, or an approval larger than proposed is a veto.
- Proposals built on inputs older than 2 cycles (or degraded fallbacks) are vetoed before Risk sees them.
- A failing or slow agent gets a neutral fallback and the cycle continues.
- Pause and blocked symbols stop new entries only; exits still go out.
- Flatten closes everything, then pauses entries until a human resumes.
- Human-approved held trades are re-checked by Risk before they are sent; unapproved ones expire after 5 cycles.
- Every override is recorded with who and why.

## Run

```bash
pip install -e ".[dev]"
python -m damico.demo         # three cycles with stub agents
python -m damico.sim.runner   # full system on synthetic data, $500,000
python -m damico.sim.runner --csv path/to/daily_csvs   # your own data, no vendor needed
pytest
```

Synthetic prices are random: simulation results on them test the machinery, not whether the strategy makes money.
