"""The first real agents: Portfolio, Market, Technical and a trend-following Strategy.

All read prices only through `MarketDataSource`, as of the cycle time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from damico import indicators as ind
from damico.agents.base import Agent, AgentContext
from damico.data.source import MarketDataSource
from damico.messages import (
    Action,
    MarketState,
    Message,
    PortfolioSnapshot,
    Side,
    Signal,
    TradeProposal,
)
from damico.portfolio import PortfolioBook


def last_prices(source: MarketDataSource, symbols: Sequence[str], as_of: datetime) -> dict[str, float]:
    prices = {}
    for s in symbols:
        bars = source.history(s, as_of, 1)
        if bars:
            prices[s] = bars[-1].close
    return prices


class PortfolioAgent(Agent):
    """Marks the book to market and publishes a snapshot.

    Runs first in each cycle (so Risk sees today's equity and loss limits) and
    again after Execution (so the record shows the book after fills).
    """

    name = "portfolio"

    def __init__(self, book: PortfolioBook, source: MarketDataSource) -> None:
        self.book = book
        self.source = source

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        prices = last_prices(self.source, list(self.book.positions), ctx.as_of)
        start = self.book.day_start_equity
        equity = self.book.mark(ctx.as_of.date(), prices)
        return [
            PortfolioSnapshot(
                **ctx.envelope(self),
                equity=equity,
                cash=self.book.cash,
                gross_exposure=self.book.gross_exposure(prices),
                net_exposure=self.book.net_exposure(prices),
                positions=len(self.book.positions),
                drawdown=self.book.drawdown(prices),
                day_pnl=equity - start if self.book._day == ctx.as_of.date() else 0.0,
            )
        ]


class MarketAgent(Agent):
    """Universe filter and market breadth."""

    name = "market"

    def __init__(self, source: MarketDataSource, min_price: float = 5.0, min_dollar_volume: float = 20e6):
        self.source = source
        self.min_price = min_price
        self.min_dollar_volume = min_dollar_volume

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        tradable, above = [], 0
        for info in self.source.symbols():
            bars = self.source.history(info.symbol, ctx.as_of, 60)
            closes = [b.close for b in bars]
            adv = ind.avg_dollar_volume(bars, 20)
            ma50 = ind.sma(closes, 50)
            if not bars or adv is None or ma50 is None:
                continue
            if bars[-1].close >= self.min_price and adv >= self.min_dollar_volume:
                tradable.append(info.symbol)
                above += bars[-1].close > ma50
        breadth = above / len(tradable) if tradable else 0.0
        regime = "trending" if breadth >= 0.6 else "volatile" if breadth <= 0.35 else "ranging"
        return [MarketState(**ctx.envelope(self), regime=regime, tradable_symbols=tuple(tradable), breadth=breadth)]


class TechnicalAgent(Agent):
    """Trend score per symbol from three up and three down conditions.

    Up: close above 50-day average, 50-day above 200-day, close at a 20-day high.
    Down: the mirror images. Score = (up - down) / 3, from -1 to +1.
    Stop: 2 ATR below the close for longs.
    """

    name = "technical"

    def __init__(self, source: MarketDataSource, atr_mult: float = 2.0) -> None:
        self.source = source
        self.atr_mult = atr_mult

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        market = ctx.latest(MarketState)
        if market is None:
            return []
        out = []
        for symbol in market.tradable_symbols:
            bars = self.source.history(symbol, ctx.as_of, 210)
            closes = [b.close for b in bars]
            ma50, ma200, atr = ind.sma(closes, 50), ind.sma(closes, 200), ind.atr(bars, 14)
            hi20, lo20 = ind.highest(closes, 20), (min(closes[-20:]) if len(closes) >= 20 else None)
            if None in (ma50, ma200, atr, hi20, lo20):
                continue
            close = closes[-1]
            up = (close > ma50) + (ma50 > ma200) + (close >= hi20)
            down = (close < ma50) + (ma50 < ma200) + (close <= lo20)
            score = (up - down) / 3
            setup = "breakout" if up == 3 else "breakdown" if down == 3 else "trend"
            out.append(
                Signal(
                    **ctx.envelope(self, [market.message_id]),
                    symbol=symbol,
                    score=score,
                    setup=setup,
                    stop=round(close - self.atr_mult * atr, 2),
                )
            )
        return out


@dataclass
class StrategyConfig:
    entry_score: float = 1.0
    exit_score: float = -1 / 3
    trail_atr: float = 3.0
    max_positions: int = 15
    risk_per_trade: float = 0.005
    """Sizing target; Risk enforces its own cap independently."""


class TrendStrategy(Agent):
    """Buy breakouts in uptrends; exit on a trailing stop or a broken trend. Long only."""

    name = "strategy"

    def __init__(self, source: MarketDataSource, book: PortfolioBook, config: StrategyConfig | None = None):
        self.source = source
        self.book = book
        self.config = config or StrategyConfig()

    async def run(self, ctx: AgentContext) -> Sequence[Message]:
        cfg = self.config
        signals = {s.symbol: s for s in ctx.of_kind(Signal)}
        held = dict(self.book.positions)
        prices = last_prices(self.source, list(set(held) | set(signals)), ctx.as_of)
        equity = self.book.equity(prices)
        proposals: list[TradeProposal] = []

        for symbol, pos in held.items():
            if pos.quantity <= 0 or symbol not in prices:
                continue
            price = prices[symbol]
            atr = ind.atr(self.source.history(symbol, ctx.as_of, 20), 14) or 0.0
            trail = max(pos.stop or 0.0, pos.high_water - cfg.trail_atr * atr)
            signal = signals.get(symbol)
            reason = None
            if price < trail:
                reason = f"trailing stop {trail:.2f} hit"
            elif signal is not None and signal.score <= cfg.exit_score:
                reason = f"trend broken, score {signal.score:+.2f}"
            if reason:
                proposals.append(
                    TradeProposal(
                        **ctx.envelope(self, [signal.message_id] if signal else []),
                        symbol=symbol, side=Side.LONG, action=Action.CLOSE,
                        quantity=pos.quantity, entry=price, stop=None, rationale=reason,
                    )
                )

        slots = cfg.max_positions - len(held) + len([p for p in proposals if p.action is Action.CLOSE])
        def momentum(symbol: str) -> float:
            bars = self.source.history(symbol, ctx.as_of, 21)
            return bars[-1].close / bars[0].close - 1 if len(bars) == 21 else 0.0

        candidates = sorted(
            (s for s in signals.values() if s.score >= cfg.entry_score and s.symbol not in held),
            key=lambda s: -momentum(s.symbol),  # strongest 20-day move first
        )
        for signal in candidates[: max(0, slots)]:
            price = prices[signal.symbol]
            per_share = price - (signal.stop or 0)
            if per_share <= 0:
                continue
            quantity = math.floor(equity * cfg.risk_per_trade / per_share)
            if quantity <= 0:
                continue
            proposals.append(
                TradeProposal(
                    **ctx.envelope(self, [signal.message_id]),
                    symbol=signal.symbol, side=Side.LONG, action=Action.OPEN,
                    quantity=quantity, entry=price, stop=signal.stop,
                    rationale=f"{signal.setup}, score {signal.score:+.2f}",
                )
            )
        return proposals
