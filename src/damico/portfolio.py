"""Portfolio book: cash, positions, equity marks and closed trades.

Quantities are signed: positive = long, negative = short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from damico.messages import Action, Side


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    stop: float | None
    sector: str
    opened: date
    high_water: float
    """Highest close since entry (longs) / lowest (shorts), for trailing stops."""

    @property
    def side(self) -> Side:
        return Side.LONG if self.quantity > 0 else Side.SHORT


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    side: Side
    quantity: int
    entry: float
    exit: float
    pnl: float
    """Net of commissions on both sides."""
    opened: date
    closed: date


@dataclass(frozen=True)
class Fill:
    day: date
    symbol: str
    side: Side
    action: Action
    quantity: int
    price: float
    commission: float


@dataclass
class PortfolioBook:
    capital_base: float
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[ClosedTrade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    _day: date | None = None
    _week: tuple[int, int] | None = None
    _open_commission: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cash:
            self.cash = self.capital_base
        self.peak_equity = self.day_start_equity = self.week_start_equity = self.cash

    # ---------------------------------------------------------------- valuation

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.quantity * prices.get(s, p.avg_price) for s, p in self.positions.items())

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(abs(p.quantity) * prices.get(s, p.avg_price) for s, p in self.positions.items())

    def net_exposure(self, prices: dict[str, float]) -> float:
        return sum(p.quantity * prices.get(s, p.avg_price) for s, p in self.positions.items())

    def sector_exposure(self, sector: str, prices: dict[str, float]) -> float:
        return sum(
            abs(p.quantity) * prices.get(s, p.avg_price)
            for s, p in self.positions.items()
            if p.sector == sector
        )

    def drawdown(self, prices: dict[str, float]) -> float:
        return max(0.0, 1 - self.equity(prices) / self.peak_equity)

    def day_loss(self, prices: dict[str, float]) -> float:
        return max(0.0, 1 - self.equity(prices) / self.day_start_equity)

    def week_loss(self, prices: dict[str, float]) -> float:
        return max(0.0, 1 - self.equity(prices) / self.week_start_equity)

    def mark(self, day: date, prices: dict[str, float]) -> float:
        """Roll day/week start equity, update peak and trailing high-water marks."""
        equity = self.equity(prices)
        if self._day != day:
            self._day = day
            self.day_start_equity = equity
        week = day.isocalendar()[:2]
        if self._week != week:
            self._week = week
            self.week_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        for s, p in self.positions.items():
            if s in prices:
                price = prices[s]
                p.high_water = max(p.high_water, price) if p.quantity > 0 else min(p.high_water, price)
        return equity

    # ---------------------------------------------------------------- fills

    def apply_fill(
        self,
        *,
        day: date,
        symbol: str,
        side: Side,
        action: Action,
        quantity: int,
        price: float,
        commission: float,
        stop: float | None = None,
        sector: str = "Unknown",
    ) -> ClosedTrade | None:
        sign = 1 if side is Side.LONG else -1
        self.fills.append(Fill(day, symbol, side, action, quantity, price, commission))
        self.cash -= commission

        if action is Action.OPEN:
            self.cash -= sign * quantity * price
            pos = self.positions.get(symbol)
            if pos is None:
                self.positions[symbol] = Position(symbol, sign * quantity, price, stop, sector, day, price)
            else:
                if (pos.quantity > 0) != (sign > 0):
                    raise ValueError(f"cannot open {side.value} against existing {pos.side.value} {symbol}")
                total = abs(pos.quantity) + quantity
                pos.avg_price = (pos.avg_price * abs(pos.quantity) + price * quantity) / total
                pos.quantity = sign * total
                pos.stop = stop if stop is not None else pos.stop
            self._open_commission[symbol] = self._open_commission.get(symbol, 0.0) + commission
            return None

        pos = self.positions.get(symbol)
        if pos is None or pos.side is not side or quantity > abs(pos.quantity):
            raise ValueError(f"cannot close {quantity} {side.value} {symbol}: not held")
        self.cash += sign * quantity * price
        share = quantity / abs(pos.quantity)
        entry_commission = self._open_commission.get(symbol, 0.0) * share
        self._open_commission[symbol] = self._open_commission.get(symbol, 0.0) - entry_commission
        pnl = sign * quantity * (price - pos.avg_price) - commission - entry_commission
        trade = ClosedTrade(symbol, side, quantity, pos.avg_price, price, pnl, pos.opened, day)
        self.trades.append(trade)
        pos.quantity -= sign * quantity
        if pos.quantity == 0:
            del self.positions[symbol]
            self._open_commission.pop(symbol, None)
        return trade
