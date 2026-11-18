# Damico Capital Management

Multi-agent US equities trading system. Design and cost estimate: see the project's technical design doc.

## What is here

The orchestrator and the message contracts every agent uses. Real agents (market data, technical, news, strategy, risk, IBKR execution) plug in next; stub agents stand in for them.

| Path | What it does |
| --- | --- |
| `src/damico/messages.py` | Typed messages (signals, proposals, risk decisions, executions, overrides). Every message carries run, cycle, timestamp, agent, version and the ids of its inputs. |
| `src/damico/orchestrator.py` | Runs one cycle a minute in market hours: analysis stages (parallel within a stage), Strategy, freshness check, Risk, human overrides, Execution, Portfolio. |
| `src/damico/agents/base.py` | Interfaces: `Agent`, `RiskAgent`, `ExecutionAgent`. |
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
python -m damico.demo   # three cycles with stub agents
pytest
```
