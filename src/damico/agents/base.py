"""Agent interfaces the orchestrator drives.

Agents are plain async classes with typed outputs. The orchestrator owns
ordering, timeouts, persistence and the Risk veto; agents only compute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar, TypeVar

from damico.messages import ExecutionReport, Message, RiskDecision, TradeProposal
from damico.store import MessageStore

M = TypeVar("M", bound=Message)


@dataclass
class AgentContext:
    """What an agent sees when it runs."""

    run_id: str
    cycle: int
    as_of: datetime
    store: MessageStore
    cycle_messages: list[Message] = field(default_factory=list)
    """Messages produced earlier in this cycle, in order."""

    def of_kind(self, kind: type[M]) -> list[M]:
        return [m for m in self.cycle_messages if isinstance(m, kind)]

    def latest(self, kind: type[M], symbol: str | None = None) -> M | None:
        """Latest message of a kind, from this cycle first, then the store."""
        for message in reversed(self.cycle_messages):
            if isinstance(message, kind) and (
                symbol is None or getattr(message, "symbol", None) == symbol
            ):
                return message
        return self.store.latest(kind, symbol)

    def envelope(self, agent: "BaseAgent", input_ids: Sequence[str] = ()) -> dict[str, Any]:
        """Envelope fields for a new message from `agent`."""
        return {
            "run_id": self.run_id,
            "cycle": self.cycle,
            "as_of": self.as_of,
            "agent": agent.name,
            "agent_version": agent.version,
            "input_ids": tuple(input_ids),
        }


class BaseAgent(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    timeout_s: ClassVar[float] = 10.0


class Agent(BaseAgent):
    """An analysis or strategy agent: reads context, returns messages."""

    run_every: ClassVar[timedelta | None] = None
    """None = every cycle. Otherwise the agent runs only when its last run is older
    than this (e.g. Fundamental nightly, Macro daily); others read its latest output."""

    @abstractmethod
    async def run(self, ctx: AgentContext) -> Sequence[Message]: ...

    def fallback(self, ctx: AgentContext) -> Sequence[Message]:
        """Neutral output when run() fails or times out. Default: nothing."""
        return ()


class RiskAgent(BaseAgent):
    """Deterministic gatekeeper. Its veto cannot be overridden by another agent."""

    @abstractmethod
    async def evaluate(
        self, ctx: AgentContext, proposals: Sequence[TradeProposal]
    ) -> Sequence[RiskDecision]: ...


class ExecutionAgent(BaseAgent):
    @abstractmethod
    async def execute(
        self, ctx: AgentContext, orders: Sequence[tuple[TradeProposal, RiskDecision]]
    ) -> Sequence[ExecutionReport]: ...

    @abstractmethod
    async def flatten(self, ctx: AgentContext) -> Sequence[ExecutionReport]:
        """Cancel open orders and close every position."""
