"""Simulated broker: fills orders against market data with IBKR-like costs.

Stands in for the IBKR adapter so the whole system runs with no account.
Commission model: IBKR Pro Tiered, first tier ($0.0035/share, $0.35 minimum,
1% of trade value maximum). Slippage: a fixed number of basis points against us.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from damico.agents.base import AgentContext, ExecutionAgent
from damico.agents.core import last_prices
from damico.data.source import MarketDataSource
from damico.messages import Action, ExecutionReport, RiskDecision, Side, TradeProposal, new_id
from damico.portfolio import PortfolioBook


def ibkr_tiered_commission(shares: int, price: float) -> float:
    return min(max(0.35, 0.0035 * shares), 0.01 * shares * price)


@dataclass(frozen=True)
class BrokerConfig:
    slippage_bps: float = 5.0
    max_order_fraction: float = 0.15
    """Second line of defense: reject any entry above this share of equity."""
    max_price_deviation: float = 0.03
    """Reject if the proposal's price is this far from the last trade."""


class SimulatedBroker(ExecutionAgent):
    name = "execution"

    def __init__(self, book: PortfolioBook, source: MarketDataSource, config: BrokerConfig | None = None):
        self.book = book
        self.source = source
        self.config = config or BrokerConfig()
        self.sectors = {i.symbol: i.sector for i in source.symbols()}

    def _fill(self, ctx: AgentContext, p: TradeProposal, qty: int, last: float, ids: list[str]) -> ExecutionReport:
        buying = (p.side is Side.LONG) == (p.action is Action.OPEN)
        slip = self.config.slippage_bps / 10_000
        price = last * (1 + slip) if buying else last * (1 - slip)
        commission = ibkr_tiered_commission(qty, price)
        self.book.apply_fill(
            day=ctx.as_of.date(), symbol=p.symbol, side=p.side, action=p.action, quantity=qty,
            price=price, commission=commission, stop=p.stop, sector=self.sectors.get(p.symbol, "Unknown"),
        )
        return ExecutionReport(
            **ctx.envelope(self, ids), proposal_id=p.message_id, symbol=p.symbol, status="filled",
            broker_order_id=f"SIM-{new_id()[:10]}",
            detail=f"{p.action.value} {p.side.value} {qty} @ {price:.2f}, commission {commission:.2f}",
        )

    async def execute(
        self, ctx: AgentContext, orders: Sequence[tuple[TradeProposal, RiskDecision]]
    ) -> Sequence[ExecutionReport]:
        prices = last_prices(self.source, sorted({p.symbol for p, _ in orders} | set(self.book.positions)), ctx.as_of)
        equity = self.book.equity(prices)
        reports = []
        for p, d in orders:
            ids = [p.message_id, d.message_id]
            last = prices.get(p.symbol)
            qty = d.approved_quantity
            problem = None
            if last is None:
                problem = "no price"
            elif abs(p.entry - last) / last > self.config.max_price_deviation:
                problem = f"price {p.entry:.2f} is more than {self.config.max_price_deviation:.0%} from last {last:.2f}"
            elif p.action is Action.OPEN and qty * last > self.config.max_order_fraction * equity:
                problem = f"order above {self.config.max_order_fraction:.0%} of equity"
            if problem:
                reports.append(ExecutionReport(**ctx.envelope(self, ids), proposal_id=p.message_id,
                                               symbol=p.symbol, status="rejected", detail=problem))
                continue
            try:
                reports.append(self._fill(ctx, p, qty, last, ids))
            except ValueError as exc:
                reports.append(ExecutionReport(**ctx.envelope(self, ids), proposal_id=p.message_id,
                                               symbol=p.symbol, status="rejected", detail=str(exc)))
        return reports

    async def flatten(self, ctx: AgentContext) -> Sequence[ExecutionReport]:
        prices = last_prices(self.source, list(self.book.positions), ctx.as_of)
        reports = []
        for symbol, pos in list(self.book.positions.items()):
            p = TradeProposal(
                **ctx.envelope(self), symbol=symbol, side=pos.side, action=Action.CLOSE,
                quantity=abs(pos.quantity), entry=prices.get(symbol, pos.avg_price), rationale="flatten",
            )
            reports.append(self._fill(ctx, p, abs(pos.quantity), p.entry, []))
        if not reports:
            reports.append(ExecutionReport(**ctx.envelope(self), status="flatten_submitted", detail="no positions"))
        return reports
