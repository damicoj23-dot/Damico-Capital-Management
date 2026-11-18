"""Risk engine: the deterministic veto.

Limits are percentages of `capital_base` / current equity, so the same rules
work at $500,000 or $20,000. Starting values come from the design doc and are
meant to be tuned in backtests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from damico import indicators as ind
from damico.agents.base import AgentContext, RiskAgent
from damico.agents.core import last_prices
from damico.data.source import MarketDataSource
from damico.messages import Action, RiskDecision, Side, TradeProposal
from damico.portfolio import PortfolioBook


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = 0.005
    max_position: float = 0.10
    max_sector: float = 0.25
    max_gross: float = 1.00
    """No margin for longs in the MVP."""
    max_positions: int = 20
    max_adv_fraction: float = 0.01
    """An order may be at most this share of 20-day average dollar volume."""
    min_price: float = 5.0
    daily_loss_halt: float = 0.02
    weekly_loss_halt: float = 0.04
    drawdown_caution: float = 0.08
    """At or beyond this drawdown, new position risk is halved."""
    drawdown_flatten: float = 0.12
    allow_shorts: bool = False
    """Off until the Short Book phase."""


class RiskEngine(RiskAgent):
    name = "risk"
    version = "0.1.0"

    def __init__(
        self,
        book: PortfolioBook,
        source: MarketDataSource,
        limits: RiskLimits | None = None,
        sectors: dict[str, str] | None = None,
    ) -> None:
        self.book = book
        self.source = source
        self.limits = limits or RiskLimits()
        self.sectors = sectors or {i.symbol: i.sector for i in source.symbols()}

    def wants_flatten(self, ctx: AgentContext) -> str | None:
        if not self.book.positions:
            return None
        prices = last_prices(self.source, list(self.book.positions), ctx.as_of)
        drawdown = self.book.drawdown(prices)
        if drawdown >= self.limits.drawdown_flatten:
            return f"drawdown {drawdown:.1%} reached the {self.limits.drawdown_flatten:.0%} limit"
        return None

    async def evaluate(self, ctx: AgentContext, proposals: Sequence[TradeProposal]) -> Sequence[RiskDecision]:
        lim = self.limits
        symbols = set(self.book.positions) | {p.symbol for p in proposals}
        prices = last_prices(self.source, sorted(symbols), ctx.as_of)
        equity = self.book.equity(prices)
        drawdown = self.book.drawdown(prices)

        halt = None
        if drawdown >= lim.drawdown_flatten:
            halt = f"drawdown {drawdown:.1%} reached the {lim.drawdown_flatten:.0%} limit"
        elif self.book.day_loss(prices) >= lim.daily_loss_halt:
            halt = f"daily loss limit {lim.daily_loss_halt:.0%} reached"
        elif self.book.week_loss(prices) >= lim.weekly_loss_halt:
            halt = f"weekly loss limit {lim.weekly_loss_halt:.0%} reached"
        caution = 0.5 if drawdown >= lim.drawdown_caution else 1.0

        gross = self.book.gross_exposure(prices)
        open_positions = len(self.book.positions)
        sector_used: dict[str, float] = {}
        decisions = []

        # Exits first: they free room for entries in the same cycle.
        ordered = sorted(proposals, key=lambda p: p.action is not Action.CLOSE)
        for p in ordered:
            env = {**ctx.envelope(self, [p.message_id]), "proposal_id": p.message_id}

            def veto(*reasons: str) -> RiskDecision:
                return RiskDecision(**env, verdict="veto", reasons=reasons)

            if p.action is Action.CLOSE:
                pos = self.book.positions.get(p.symbol)
                if pos is None or pos.side is not p.side:
                    decisions.append(veto("nothing to close"))
                    continue
                qty = min(p.quantity, abs(pos.quantity))
                verdict = "approve" if qty == p.quantity else "resize"
                decisions.append(RiskDecision(**env, verdict=verdict, approved_quantity=qty,
                                              reasons=() if qty == p.quantity else ("capped at held quantity",)))
                gross -= qty * prices.get(p.symbol, p.entry)
                if qty == abs(pos.quantity):
                    open_positions -= 1
                continue

            # ---- entries
            if halt:
                decisions.append(veto(halt))
                continue
            if p.side is Side.SHORT and not lim.allow_shorts:
                decisions.append(veto("short selling not enabled"))
                continue
            if p.stop is None:
                decisions.append(veto("no stop price"))
                continue
            price = prices.get(p.symbol)
            if price is None:
                decisions.append(veto("no price data"))
                continue
            if price < lim.min_price:
                decisions.append(veto(f"price under ${lim.min_price:.0f}"))
                continue
            if open_positions >= lim.max_positions:
                decisions.append(veto(f"already at {lim.max_positions} positions"))
                continue
            per_share = abs(p.entry - p.stop)
            if per_share <= 0:
                decisions.append(veto("stop on wrong side of entry"))
                continue

            adv = ind.avg_dollar_volume(self.source.history(p.symbol, ctx.as_of, 20), 20) or 0.0
            sector = self.sectors.get(p.symbol, "Unknown")
            used = sector_used.get(sector, self.book.sector_exposure(sector, prices))
            caps = {
                "risk per trade": equity * lim.risk_per_trade * caution / per_share,
                "position size": equity * lim.max_position / p.entry,
                "gross exposure": (equity * lim.max_gross - gross) / p.entry,
                "sector exposure": (equity * lim.max_sector - used) / p.entry,
                "liquidity": adv * lim.max_adv_fraction / p.entry,
            }
            binding, cap = min(caps.items(), key=lambda kv: kv[1])
            qty = min(p.quantity, math.floor(cap))
            if qty <= 0:
                decisions.append(veto(f"{binding} limit leaves no room"))
                continue
            reasons = []
            if qty < p.quantity:
                reasons.append(f"resized by {binding} limit")
            if caution < 1:
                reasons.append(f"drawdown {drawdown:.1%}: position risk halved")
            decisions.append(RiskDecision(**env, verdict="approve" if qty == p.quantity else "resize",
                                          approved_quantity=qty, reasons=tuple(reasons)))
            notional = qty * p.entry
            gross += notional
            sector_used[sector] = used + notional
            open_positions += 1
        return decisions
