"""Run three orchestrator cycles with stub agents: `python -m damico.demo`."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from damico.agents.demo import (
    PaperExecutionStub,
    SimpleRisk,
    StubMarket,
    StubNews,
    StubTechnical,
    ThresholdStrategy,
)
from damico.messages import HoldNotice, RiskDecision
from damico.orchestrator import Orchestrator


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    execution = PaperExecutionStub()
    orch = Orchestrator(
        analysis_stages=[[StubMarket()], [StubTechnical(), StubNews()]],
        strategy=ThresholdStrategy(),
        risk=SimpleRisk(max_risk_usd=500),
        execution=execution,
    )
    t0 = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)  # 10:00 New York

    print("Cycle 1: normal")
    await show(orch, t0)

    orch.set_approval_threshold(2_000, who="founder", why="demo: approve trades above $2k")
    print("Cycle 2: approve-each-trade above $2,000")
    await show(orch, t0 + timedelta(minutes=1))

    held = orch.pending_approvals[0]
    orch.approve(held.message_id, who="founder", why="demo approval")
    print("Cycle 3: after human approval")
    await show(orch, t0 + timedelta(minutes=2))


async def show(orch: Orchestrator, now: datetime) -> None:
    report = await orch.run_cycle(now)
    for m in orch.store.by_cycle(orch.run_id, report.cycle):
        if isinstance(m, RiskDecision):
            print(f"  risk: {m.verdict} qty={m.approved_quantity} {list(m.reasons)}")
        elif isinstance(m, HoldNotice):
            print(f"  held: {m.symbol} ({m.reason})")
    for e in report.executions:
        print(f"  sent: {e.symbol} {e.detail}")
    print(f"  {report.proposals} proposals, {report.vetoed} vetoed, {report.held} held, "
          f"{report.sent} sent, {report.seconds * 1000:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
