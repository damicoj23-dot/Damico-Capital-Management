import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from damico.agents.base import Agent, AgentContext
from damico.agents.demo import (
    PaperExecutionStub,
    SimpleRisk,
    StubMarket,
    StubNews,
    StubTechnical,
    ThresholdStrategy,
)
from damico.clock import MarketClock
from damico.messages import (
    Action,
    HoldNotice,
    MacroRegime,
    OverrideEvent,
    RiskDecision,
    Side,
    Signal,
    TradeProposal,
)
from damico.orchestrator import Orchestrator, OrchestratorConfig

T0 = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)  # Wed 10:00 New York


def make(**kw):
    execution = kw.pop("execution", PaperExecutionStub())
    orch = Orchestrator(
        analysis_stages=kw.pop("stages", [[StubMarket()], [StubTechnical(), StubNews()]]),
        strategy=kw.pop("strategy", ThresholdStrategy()),
        risk=kw.pop("risk", SimpleRisk(max_risk_usd=500)),
        execution=execution,
        **kw,
    )
    return orch, execution


def run(orch, now=T0):
    return asyncio.run(orch.run_cycle(now))


def of(orch, cycle, kind):
    return [m for m in orch.store.by_cycle(orch.run_id, cycle) if isinstance(m, kind)]


def test_happy_path_sends_approved_trade_and_records_everything():
    orch, ex = make()
    report = run(orch)
    assert report.proposals == 1 and report.sent == 1 and report.vetoed == 0
    assert ex.sent[0][0].symbol == "AAPL"
    kinds = [m.kind for m in orch.store.by_cycle(orch.run_id, 1)]
    assert kinds.index("market_state") < kinds.index("signal") < kinds.index("trade_proposal")
    assert kinds.index("trade_proposal") < kinds.index("risk_decision") < kinds.index("execution_report")


def test_risk_resize_is_respected():
    orch, ex = make(risk=SimpleRisk(max_risk_usd=200))  # $10/share risk -> 20 shares
    run(orch)
    (_, decision), = ex.sent
    assert decision.verdict == "resize" and decision.approved_quantity == 20


def test_risk_veto_blocks_execution():
    orch, ex = make(stages=[[StubMarket()], [StubTechnical({"AAPL": (0.9, 199.99)})]],
                    risk=SimpleRisk(max_risk_usd=0.001))
    report = run(orch)
    assert report.vetoed == 1 and report.sent == 0 and ex.sent == []


class Boom(Agent):
    name = "news"

    async def run(self, ctx):
        raise RuntimeError("vendor down")


class Slow(Agent):
    name = "news"
    timeout_s = 0.05

    async def run(self, ctx):
        await asyncio.sleep(1)
        return []


@pytest.mark.parametrize("bad", [Boom(), Slow()])
def test_failing_agent_does_not_stop_the_cycle(bad):
    orch, ex = make(stages=[[StubMarket()], [StubTechnical(), bad]])
    report = run(orch)
    failed = [t for t in report.timings if t.agent == "news"]
    assert failed and not failed[0].ok
    assert report.sent == 1


class BrokenRisk(SimpleRisk):
    async def evaluate(self, ctx, proposals):
        raise RuntimeError("bug")


class SilentRisk(SimpleRisk):
    async def evaluate(self, ctx, proposals):
        return []


class GreedyRisk(SimpleRisk):
    async def evaluate(self, ctx, proposals):
        return [RiskDecision(**ctx.envelope(self, [p.message_id]), proposal_id=p.message_id,
                             verdict="approve", approved_quantity=p.quantity * 10) for p in proposals]


@pytest.mark.parametrize("risk", [BrokenRisk(), SilentRisk(), GreedyRisk()])
def test_risk_fails_closed(risk):
    orch, ex = make(risk=risk)
    report = run(orch)
    assert report.sent == 0 and ex.sent == []
    (decision,) = of(orch, 1, RiskDecision)
    assert decision.verdict == "veto" and decision.agent == "orchestrator"


class StaleStrategy(Agent):
    """Builds a proposal on a signal from an earlier cycle."""

    name = "strategy"

    def __init__(self):
        self.first_signal = None

    async def run(self, ctx):
        if self.first_signal is None:
            self.first_signal = ctx.of_kind(Signal)[0]
            return []
        return [TradeProposal(**ctx.envelope(self, [self.first_signal.message_id]), symbol="AAPL",
                              side=Side.LONG, action=Action.OPEN, quantity=10, entry=200, stop=190)]


def test_stale_inputs_are_vetoed_before_risk():
    orch, ex = make(strategy=StaleStrategy())
    run(orch, T0)
    run(orch, T0 + timedelta(minutes=1))  # 1 minute old: fresh enough
    assert len(ex.sent) == 1
    report = run(orch, T0 + timedelta(minutes=5))  # 5 minutes old: stale
    assert report.sent == 0
    (decision,) = of(orch, 3, RiskDecision)
    assert decision.verdict == "veto" and "stale signal" in decision.reasons[0]


def test_pause_blocks_entries_but_not_exits():
    class EntryAndExit(Agent):
        name = "strategy"

        async def run(self, ctx):
            env = ctx.envelope(self)
            return [
                TradeProposal(**env, symbol="AAPL", side=Side.LONG, action=Action.OPEN, quantity=10, entry=200, stop=190),
                TradeProposal(**env, symbol="MSFT", side=Side.LONG, action=Action.CLOSE, quantity=10, entry=420, stop=400),
            ]

    orch, ex = make(strategy=EntryAndExit())
    orch.pause_entries(who="founder", why="test")
    report = run(orch)
    assert [p.symbol for p, _ in ex.sent] == ["MSFT"]
    assert report.held == 1 and of(orch, 1, HoldNotice)[0].reason == "paused"
    assert any(e.change == "pause_entries" and e.who == "founder" for e in orch.store.all()
               if isinstance(e, OverrideEvent))


def test_blocked_symbol():
    orch, ex = make()
    orch.block_symbol("aapl", who="founder", why="earnings")
    report = run(orch)
    assert report.sent == 0 and of(orch, 1, HoldNotice)[0].reason == "blocked_symbol"


def test_approve_each_trade_holds_then_releases_after_approval():
    orch, ex = make(strategy=ThresholdStrategy(threshold=0.5))
    orch.set_approval_threshold(5_000, who="founder", why="test")  # 50 x $200 = $10k
    run(orch, T0)
    assert ex.sent == [] and len(orch.pending_approvals) == 1
    held_id = orch.pending_approvals[0].message_id
    orch.approve(held_id, who="founder", why="looks good")
    run(orch, T0 + timedelta(minutes=1))
    approved = [p for p, _ in ex.sent if p.agent == "orchestrator"]
    assert len(approved) == 1 and approved[0].input_ids == (held_id,)


def test_unapproved_proposals_expire():
    orch, ex = make(config=OrchestratorConfig(approval_ttl_cycles=2),
                    strategy=ThresholdStrategy(threshold=2.0))  # proposes nothing...
    orch.set_approval_threshold(1, who="founder", why="test")
    orch.strategy = ThresholdStrategy()  # ...until now
    run(orch, T0)
    orch.strategy = ThresholdStrategy(threshold=2.0)
    run(orch, T0 + timedelta(minutes=1))
    run(orch, T0 + timedelta(minutes=2))
    assert orch.pending_approvals == []
    assert of(orch, 3, HoldNotice)[0].reason == "approval_expired"


def test_flatten_closes_everything_then_pauses():
    orch, ex = make()
    orch.request_flatten(who="founder", why="test")
    report = run(orch)
    assert report.flattened and ex.flatten_calls == 1 and ex.sent == []
    assert orch.overrides.paused_entries and not orch.overrides.flatten_requested
    run(orch, T0 + timedelta(minutes=1))
    assert ex.sent == []  # still paused


class DailyMacro(Agent):
    name = "macro"
    run_every = timedelta(hours=24)
    calls = 0

    async def run(self, ctx):
        DailyMacro.calls += 1
        return [MacroRegime(**ctx.envelope(self), regime="neutral")]


def test_scheduled_agent_runs_only_when_due_and_is_readable_later():
    DailyMacro.calls = 0
    seen = []

    class ReadsMacro(Agent):
        name = "strategy"

        async def run(self, ctx: AgentContext):
            seen.append(ctx.latest(MacroRegime))
            return []

    orch, _ = make(stages=[[DailyMacro()]], strategy=ReadsMacro())
    for i in range(3):
        run(orch, T0 + timedelta(minutes=i))
    assert DailyMacro.calls == 1 and all(m is not None for m in seen)


def test_agent_with_wrong_envelope_is_rejected():
    class Liar(Agent):
        name = "technical"

        async def run(self, ctx):
            env = {**ctx.envelope(self), "agent": "risk"}
            return [Signal(**env, symbol="AAPL", score=1, stop=190)]

    orch, ex = make(stages=[[Liar()]])
    report = run(orch)
    assert report.proposals == 0
    assert not [t for t in report.timings if t.agent == "technical"][0].ok


def test_parallel_agents_run_concurrently():
    class Sleepy(Agent):
        timeout_s = 2

        def __init__(self, name):
            self.name = name

        async def run(self, ctx):
            await asyncio.sleep(0.2)
            return []

    orch, _ = make(stages=[[Sleepy("a"), Sleepy("b"), Sleepy("c")]])
    report = run(orch)
    assert report.seconds < 0.5


def test_market_clock():
    clock = MarketClock()
    assert clock.is_open(T0)
    assert not clock.is_open(T0.replace(hour=13))  # 08:00 New York
    assert not clock.is_open(datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc))  # Saturday


def test_run_forever_skips_closed_market_and_stops():
    orch, ex = make(clock=MarketClock(), now=lambda: datetime(2026, 11, 21, 15, tzinfo=timezone.utc),
                    config=OrchestratorConfig(closed_market_poll=timedelta(seconds=0.01)))

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(orch.run_forever(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await task

    asyncio.run(go())
    assert orch.cycle == 0
