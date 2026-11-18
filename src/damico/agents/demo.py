"""Stub agents with fixed outputs, for tests and the demo run.

They stand in for the real Market, Technical, News, Strategy, Risk and
Execution agents until those are built, and show the contract each must meet.
"""

from __future__ import annotations

from collections.abc import Sequence

from damico.agents.base import Agent, AgentContext, ExecutionAgent, RiskAgent
from damico.messages import (
    Action,
    ExecutionReport,
    MarketState,
    Message,
    NewsEvent,
    RiskDecision,
    Side,
    Signal,
    TradeProposal,
)


class StubMarket(Agent):
    name = "market"

    def __init__(self, symbols: Sequence[str] = ("AAPL", "MSFT")) -> None:
        self.symbols = tuple(symbols)

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        return [MarketState(**ctx.envelope(self), regime="trending", tradable_symbols=self.symbols)]


class StubTechnical(Agent):
    name = "technical"

    def __init__(self, scores: dict[str, tuple[float, float]] | None = None) -> None:
        """scores: symbol -> (score, stop price)."""
        self.scores = scores or {"AAPL": (0.8, 190.0), "MSFT": (-0.2, 400.0)}

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        market = ctx.latest(MarketState)
        inputs = [market.message_id] if market else []
        return [
            Signal(**ctx.envelope(self, inputs), symbol=s, score=score, setup="trend", stop=stop)
            for s, (score, stop) in self.scores.items()
        ]


class StubNews(Agent):
    name = "news"

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        return [NewsEvent(**ctx.envelope(self), symbol="AAPL", sentiment=0.3, materiality=0.2)]

    def fallback(self, ctx: AgentContext) -> Sequence[Message]:
        return []


class ThresholdStrategy(Agent):
    """Propose a long entry for every signal above a threshold."""

    name = "strategy"

    def __init__(self, threshold: float = 0.5, quantity: int = 50, prices: dict[str, float] | None = None):
        self.threshold = threshold
        self.quantity = quantity
        self.prices = prices or {"AAPL": 200.0, "MSFT": 420.0}

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        proposals = []
        for signal in ctx.of_kind(Signal):
            if signal.score < self.threshold:
                continue
            proposals.append(
                TradeProposal(
                    **ctx.envelope(self, [signal.message_id]),
                    symbol=signal.symbol,
                    side=Side.LONG,
                    action=Action.OPEN,
                    quantity=self.quantity,
                    entry=self.prices[signal.symbol],
                    stop=signal.stop,
                    rationale=f"{signal.setup} score {signal.score:+.2f}",
                )
            )
        return proposals


class SimpleRisk(RiskAgent):
    """Placeholder rules: stop required; cap risk per trade at a dollar amount."""

    name = "risk"

    def __init__(self, max_risk_usd: float = 500.0) -> None:
        self.max_risk_usd = max_risk_usd

    async def evaluate(self, ctx: AgentContext, proposals: Sequence[TradeProposal]) -> Sequence[RiskDecision]:
        decisions = []
        for p in proposals:
            base = {**ctx.envelope(self, [p.message_id]), "proposal_id": p.message_id}
            if p.stop is None:
                decisions.append(RiskDecision(**base, verdict="veto", reasons=("no stop",)))
                continue
            per_share = abs(p.entry - p.stop)
            allowed = int(self.max_risk_usd // per_share) if per_share > 0 else 0
            if allowed <= 0:
                decisions.append(RiskDecision(**base, verdict="veto", reasons=("stop too wide",)))
            elif allowed < p.quantity:
                decisions.append(
                    RiskDecision(**base, verdict="resize", approved_quantity=allowed,
                                 reasons=(f"risk per trade capped at ${self.max_risk_usd:.0f}",))
                )
            else:
                decisions.append(RiskDecision(**base, verdict="approve", approved_quantity=p.quantity))
        return decisions


class PaperExecutionStub(ExecutionAgent):
    """Records what it would send; the IBKR adapter replaces it."""

    name = "execution"

    def __init__(self) -> None:
        self.sent: list[tuple[TradeProposal, RiskDecision]] = []
        self.flatten_calls = 0

    async def execute(self, ctx, orders):
        self.sent.extend(orders)
        return [
            ExecutionReport(
                **ctx.envelope(self, [p.message_id, d.message_id]),
                proposal_id=p.message_id,
                symbol=p.symbol,
                status="submitted",
                detail=f"{p.action.value} {p.side.value} {d.approved_quantity} @ {p.entry}",
            )
            for p, d in orders
        ]

    async def flatten(self, ctx):
        self.flatten_calls += 1
        return [ExecutionReport(**ctx.envelope(self), status="flatten_submitted", detail="close all")]
