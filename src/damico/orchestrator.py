"""The orchestrator: runs one decision cycle every minute during market hours.

Cycle order:
    1. flatten, if a human asked for it (then stop the cycle)
    2. analysis stages, agents within a stage in parallel (Market; Technical + News; ...)
       scheduled agents (Fundamental, Macro) run only when due
    3. Strategy -> trade proposals (plus any held proposals a human approved)
    4. input freshness check -> Risk agent -> decisions (fail closed: no decision = veto)
    5. human overrides (pause, blocked symbols, approve-each-trade)
    6. Execution
    7. Portfolio

Every message is written to the store before the next stage reads it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TypeVar

from damico.agents.base import Agent, AgentContext, BaseAgent, ExecutionAgent, RiskAgent
from damico.clock import MarketClock
from damico.messages import (
    Action,
    ExecutionReport,
    FundamentalScore,
    HoldNotice,
    MacroRegime,
    Message,
    OverrideEvent,
    RiskDecision,
    TradeProposal,
    new_id,
)
from damico.overrides import OverrideState
from damico.store import InMemoryStore, MessageStore

log = logging.getLogger(__name__)

ORCHESTRATOR = "orchestrator"
VERSION = "0.1.0"

T = TypeVar("T")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class OrchestratorConfig:
    cycle_interval: timedelta = timedelta(seconds=60)
    stale_after_cycles: int = 2
    """Per-cycle inputs older than this many cycles make a proposal stale."""
    max_input_age: dict[str, timedelta] = field(
        default_factory=lambda: {
            FundamentalScore.kind: timedelta(days=8),
            MacroRegime.kind: timedelta(hours=26),
        }
    )
    """Age limits for scheduled inputs, by message kind."""
    approval_ttl_cycles: int = 5
    """A held proposal not approved within this many cycles expires."""
    closed_market_poll: timedelta = timedelta(seconds=30)

    def input_age_limit(self, kind: str) -> timedelta:
        return self.max_input_age.get(kind, self.cycle_interval * self.stale_after_cycles)


@dataclass
class AgentTiming:
    agent: str
    seconds: float
    ok: bool
    error: str = ""


@dataclass
class CycleReport:
    run_id: str
    cycle: int
    as_of: datetime
    seconds: float = 0.0
    timings: list[AgentTiming] = field(default_factory=list)
    proposals: int = 0
    vetoed: int = 0
    held: int = 0
    sent: int = 0
    executions: list[ExecutionReport] = field(default_factory=list)
    flattened: bool = False


class Orchestrator:
    def __init__(
        self,
        *,
        analysis_stages: Sequence[Sequence[Agent]],
        strategy: Agent,
        risk: RiskAgent,
        execution: ExecutionAgent,
        portfolio: Agent | None = None,
        store: MessageStore | None = None,
        clock: MarketClock | None = None,
        config: OrchestratorConfig | None = None,
        overrides: OverrideState | None = None,
        run_id: str | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self.analysis_stages = [list(stage) for stage in analysis_stages]
        self.strategy = strategy
        self.risk = risk
        self.execution = execution
        self.portfolio = portfolio
        self.store: MessageStore = store if store is not None else InMemoryStore()
        self.clock = clock or MarketClock()
        self.config = config or OrchestratorConfig()
        self.overrides = overrides or OverrideState()
        self.run_id = run_id or new_id()
        self._now = now
        self.cycle = 0
        self._last_run: dict[str, datetime] = {}
        self._pending: dict[str, tuple[TradeProposal, int]] = {}
        """Proposals waiting for human approval: id -> (proposal, cycle held)."""

    # ------------------------------------------------------------------ overrides

    def _record_override(self, change: str, who: str, why: str) -> None:
        self.store.append(
            OverrideEvent(
                run_id=self.run_id,
                cycle=self.cycle,
                as_of=self._now(),
                agent=ORCHESTRATOR,
                agent_version=VERSION,
                change=change,
                who=who,
                why=why,
            )
        )
        log.warning("override: %s by %s (%s)", change, who, why)

    def pause_entries(self, who: str, why: str) -> None:
        self.overrides.paused_entries = True
        self._record_override("pause_entries", who, why)

    def resume_entries(self, who: str, why: str) -> None:
        self.overrides.paused_entries = False
        self._record_override("resume_entries", who, why)

    def request_flatten(self, who: str, why: str) -> None:
        self.overrides.flatten_requested = True
        self._record_override("flatten", who, why)

    def block_symbol(self, symbol: str, who: str, why: str) -> None:
        self.overrides.blocked_symbols.add(symbol.upper())
        self._record_override(f"block_symbol:{symbol.upper()}", who, why)

    def unblock_symbol(self, symbol: str, who: str, why: str) -> None:
        self.overrides.blocked_symbols.discard(symbol.upper())
        self._record_override(f"unblock_symbol:{symbol.upper()}", who, why)

    def set_approval_threshold(self, usd: float | None, who: str, why: str) -> None:
        self.overrides.approve_each_trade_above = usd
        self._record_override(f"approve_each_trade_above:{usd}", who, why)

    def approve(self, proposal_id: str, who: str, why: str) -> None:
        if proposal_id not in self._pending:
            raise KeyError(f"no held proposal {proposal_id}")
        self.overrides.approved_proposals.add(proposal_id)
        self._record_override(f"approve:{proposal_id}", who, why)

    @property
    def pending_approvals(self) -> list[TradeProposal]:
        return [proposal for proposal, _ in self._pending.values()]

    # ------------------------------------------------------------------ helpers

    def _record(self, ctx: AgentContext, messages: Sequence[Message]) -> None:
        for message in messages:
            self.store.append(message)
            ctx.cycle_messages.append(message)

    def _validate(self, agent: BaseAgent, ctx: AgentContext, outputs: object) -> list[Message]:
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"{agent.name} returned {type(outputs).__name__}, not a sequence")
        for message in outputs:
            if not isinstance(message, Message):
                raise TypeError(f"{agent.name} returned a non-message: {message!r}")
            if (message.run_id, message.cycle, message.agent) != (ctx.run_id, ctx.cycle, agent.name):
                raise ValueError(f"{agent.name} returned a message with a wrong envelope")
        return list(outputs)

    async def _timed(
        self, agent: BaseAgent, report: CycleReport, call: Awaitable[T]
    ) -> tuple[T | None, str]:
        start = time.perf_counter()
        error = ""
        result: T | None = None
        try:
            result = await asyncio.wait_for(call, timeout=agent.timeout_s)
        except asyncio.TimeoutError:
            error = f"timed out after {agent.timeout_s}s"
        except Exception as exc:  # an agent bug must not stop the cycle
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start
        report.timings.append(AgentTiming(agent.name, seconds, ok=not error, error=error))
        if error:
            log.error("agent %s failed in cycle %s: %s", agent.name, report.cycle, error)
        return result, error

    async def _run_agent(self, agent: Agent, ctx: AgentContext, report: CycleReport) -> list[Message]:
        outputs, error = await self._timed(agent, report, agent.run(ctx))
        if not error:
            try:
                messages = self._validate(agent, ctx, outputs)
                self._last_run[agent.name] = ctx.as_of
                return messages
            except (TypeError, ValueError) as exc:
                report.timings[-1].ok = False
                report.timings[-1].error = str(exc)
                log.error("agent %s output rejected: %s", agent.name, exc)
        try:
            fallback = self._validate(agent, ctx, list(agent.fallback(ctx)))
        except Exception as exc:
            log.error("agent %s fallback failed: %s", agent.name, exc)
            return []
        return [m.model_copy(update={"degraded": True}) for m in fallback]

    def _due(self, agent: Agent, now: datetime) -> bool:
        if agent.run_every is None:
            return True
        last = self._last_run.get(agent.name)
        return last is None or now - last >= agent.run_every

    def _stale_reasons(self, proposal: TradeProposal, now: datetime) -> list[str]:
        if proposal.agent == ORCHESTRATOR:
            return []  # re-issued after human approval; Risk still re-checks it
        reasons = []
        for input_id in proposal.input_ids:
            source = self.store.get(input_id)
            if source is None:
                reasons.append(f"input {input_id} not found")
                continue
            age = now - source.as_of
            limit = self.config.input_age_limit(source.kind)
            if age > limit:
                reasons.append(f"stale {source.kind} from {source.agent} ({age} old, limit {limit})")
            if source.degraded:
                reasons.append(f"degraded {source.kind} from {source.agent}")
        return reasons

    def _decision(
        self, ctx: AgentContext, proposal: TradeProposal, reasons: Sequence[str]
    ) -> RiskDecision:
        """A veto issued by the orchestrator itself (fail closed)."""
        return RiskDecision(
            run_id=ctx.run_id,
            cycle=ctx.cycle,
            as_of=ctx.as_of,
            agent=ORCHESTRATOR,
            agent_version=VERSION,
            input_ids=(proposal.message_id,),
            proposal_id=proposal.message_id,
            verdict="veto",
            approved_quantity=0,
            reasons=tuple(reasons),
        )

    def _hold(self, ctx: AgentContext, proposal: TradeProposal, reason: str) -> HoldNotice:
        return HoldNotice(
            run_id=ctx.run_id,
            cycle=ctx.cycle,
            as_of=ctx.as_of,
            agent=ORCHESTRATOR,
            agent_version=VERSION,
            input_ids=(proposal.message_id,),
            proposal_id=proposal.message_id,
            symbol=proposal.symbol,
            reason=reason,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ the cycle

    async def run_cycle(self, now: datetime | None = None) -> CycleReport:
        now = now or self._now()
        self.cycle += 1
        started = time.perf_counter()
        ctx = AgentContext(run_id=self.run_id, cycle=self.cycle, as_of=now, store=self.store)
        report = CycleReport(run_id=self.run_id, cycle=self.cycle, as_of=now)

        if self.overrides.flatten_requested:
            await self._flatten(ctx, report)
            report.seconds = time.perf_counter() - started
            return report

        # 2. analysis stages
        for stage in self.analysis_stages:
            due = [agent for agent in stage if self._due(agent, now)]
            results = await asyncio.gather(*(self._run_agent(a, ctx, report) for a in due))
            for messages in results:
                self._record(ctx, messages)

        # 3. strategy, plus held proposals a human approved
        strategy_out = await self._run_agent(self.strategy, ctx, report)
        self._record(ctx, strategy_out)
        proposals = [m for m in strategy_out if isinstance(m, TradeProposal)]
        proposals += self._release_approved(ctx)
        report.proposals = len(proposals)

        # 4. freshness check, then Risk
        decisions = await self._risk(ctx, proposals, now, report)

        # 5. overrides, 6. execution
        orders = self._apply_overrides(ctx, proposals, decisions, report)
        if orders:
            executions, error = await self._timed(
                self.execution, report, self.execution.execute(ctx, orders)
            )
            if error:
                log.critical("execution failed; orders may be partially sent: %s", error)
            else:
                try:
                    valid = self._validate(self.execution, ctx, executions)
                except (TypeError, ValueError) as exc:
                    log.critical("execution returned invalid reports: %s", exc)
                else:
                    self._record(ctx, valid)
                    report.executions = [m for m in valid if isinstance(m, ExecutionReport)]
        report.sent = len(orders)

        # 7. portfolio
        if self.portfolio is not None:
            self._record(ctx, await self._run_agent(self.portfolio, ctx, report))

        self._expire_pending(ctx, report)
        report.seconds = time.perf_counter() - started
        if report.seconds > self.config.cycle_interval.total_seconds():
            log.warning("cycle %s overran: %.1fs", self.cycle, report.seconds)
        return report

    async def _flatten(self, ctx: AgentContext, report: CycleReport) -> None:
        reports, error = await self._timed(self.execution, report, self.execution.flatten(ctx))
        if error:
            log.critical("FLATTEN FAILED, will retry next cycle: %s", error)
            return
        try:
            valid = self._validate(self.execution, ctx, reports)
        except (TypeError, ValueError) as exc:
            log.critical("FLATTEN returned invalid reports, will retry next cycle: %s", exc)
            return
        self._record(ctx, valid)
        report.executions = [m for m in valid if isinstance(m, ExecutionReport)]
        report.flattened = True
        self.overrides.flatten_requested = False
        self.overrides.paused_entries = True
        self._pending.clear()
        self._record_override("pause_entries", ORCHESTRATOR, "automatic pause after flatten")

    async def _risk(
        self,
        ctx: AgentContext,
        proposals: list[TradeProposal],
        now: datetime,
        report: CycleReport,
    ) -> dict[str, RiskDecision]:
        decisions: dict[str, RiskDecision] = {}
        to_risk: list[TradeProposal] = []
        for proposal in proposals:
            reasons = self._stale_reasons(proposal, now)
            if reasons:
                decisions[proposal.message_id] = self._decision(ctx, proposal, reasons)
            else:
                to_risk.append(proposal)

        if to_risk:
            result, error = await self._timed(self.risk, report, self.risk.evaluate(ctx, to_risk))
            by_id = {p.message_id: p for p in to_risk}
            returned: list[RiskDecision] = []
            if not error:
                try:
                    returned = [
                        d
                        for d in self._validate(self.risk, ctx, result)
                        if isinstance(d, RiskDecision)
                    ]
                except (TypeError, ValueError) as exc:
                    error = str(exc)
            for decision in returned:
                proposal = by_id.get(decision.proposal_id)
                if proposal is None or decision.proposal_id in decisions:
                    continue
                if decision.approved_quantity > proposal.quantity:
                    decisions[proposal.message_id] = self._decision(
                        ctx, proposal, ["risk approved more than proposed"]
                    )
                else:
                    decisions[proposal.message_id] = decision
            for proposal in to_risk:
                if proposal.message_id not in decisions:
                    why = f"risk agent error: {error}" if error else "no risk decision"
                    decisions[proposal.message_id] = self._decision(ctx, proposal, [why])

        self._record(ctx, [decisions[p.message_id] for p in proposals])
        report.vetoed = sum(not d.allows_trade for d in decisions.values())
        return decisions

    def _apply_overrides(
        self,
        ctx: AgentContext,
        proposals: list[TradeProposal],
        decisions: dict[str, RiskDecision],
        report: CycleReport,
    ) -> list[tuple[TradeProposal, RiskDecision]]:
        orders: list[tuple[TradeProposal, RiskDecision]] = []
        holds: list[HoldNotice] = []
        o = self.overrides
        for proposal in proposals:
            decision = decisions[proposal.message_id]
            if not decision.allows_trade:
                continue
            if proposal.action is Action.OPEN:
                human_approved = proposal.agent == ORCHESTRATOR
                notional = decision.approved_quantity * proposal.entry
                if o.paused_entries:
                    holds.append(self._hold(ctx, proposal, "paused"))
                    continue
                if proposal.symbol.upper() in o.blocked_symbols:
                    holds.append(self._hold(ctx, proposal, "blocked_symbol"))
                    continue
                if (
                    not human_approved
                    and o.approve_each_trade_above is not None
                    and notional > o.approve_each_trade_above
                ):
                    self._pending[proposal.message_id] = (proposal, ctx.cycle)
                    holds.append(self._hold(ctx, proposal, "awaiting_approval"))
                    continue
            orders.append((proposal, decision))
        self._record(ctx, holds)
        report.held = len(holds)
        return orders

    def _release_approved(self, ctx: AgentContext) -> list[TradeProposal]:
        """Re-issue human-approved proposals so Risk checks them against today's book."""
        released = []
        for proposal_id in sorted(self.overrides.approved_proposals):
            held = self._pending.pop(proposal_id, None)
            if held is None:
                continue
            original, _ = held
            reissued = original.model_copy(
                update={
                    "message_id": new_id(),
                    "cycle": ctx.cycle,
                    "as_of": ctx.as_of,
                    "agent": ORCHESTRATOR,
                    "agent_version": VERSION,
                    "input_ids": (original.message_id,),
                    "rationale": f"human-approved: {original.rationale}",
                }
            )
            released.append(reissued)
        self.overrides.approved_proposals.clear()
        self._record(ctx, released)
        return released

    def _expire_pending(self, ctx: AgentContext, report: CycleReport) -> None:
        ttl = self.config.approval_ttl_cycles
        expired = [pid for pid, (_, held_at) in self._pending.items() if ctx.cycle - held_at >= ttl]
        notices = []
        for pid in expired:
            proposal, _ = self._pending.pop(pid)
            notices.append(self._hold(ctx, proposal, "approval_expired"))
        self._record(ctx, notices)

    # ------------------------------------------------------------------ the loop

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """Run a cycle every `cycle_interval` while the market is open."""
        stop = stop or asyncio.Event()
        interval = self.config.cycle_interval.total_seconds()
        while not stop.is_set():
            now = self._now()
            if self.clock.is_open(now):
                started = time.monotonic()
                report = await self.run_cycle(now)
                log.info(
                    "cycle %s: %d proposals, %d vetoed, %d held, %d sent in %.2fs",
                    report.cycle, report.proposals, report.vetoed, report.held,
                    report.sent, report.seconds,
                )
                wait = max(0.0, interval - (time.monotonic() - started))
            else:
                wait = self.config.closed_market_poll.total_seconds()
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass
