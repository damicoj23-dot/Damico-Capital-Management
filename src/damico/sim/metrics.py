"""Performance metrics matching the targets in the design doc."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from damico.portfolio import ClosedTrade

TRADING_DAYS = 252


@dataclass(frozen=True)
class Performance:
    start_equity: float
    end_equity: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    trades: int
    win_rate: float
    profit_factor: float
    commissions: float

    def lines(self) -> list[str]:
        pf = "n/a" if math.isinf(self.profit_factor) else f"{self.profit_factor:.2f}"
        return [
            f"Equity           ${self.start_equity:,.0f} -> ${self.end_equity:,.0f}",
            f"Total return     {self.total_return:+.1%}",
            f"Annual return    {self.annual_return:+.1%}   (target 12% to 20%)",
            f"Max drawdown     {self.max_drawdown:.1%}    (target under 15%)",
            f"Sharpe           {self.sharpe:.2f}     (target above 1.5)",
            f"Trades           {self.trades}",
            f"Win rate         {self.win_rate:.0%}     (target above 50%)",
            f"Profit factor    {pf}     (target above 1.5)",
            f"Commissions      ${self.commissions:,.0f}",
        ]


def max_drawdown(curve: Sequence[float]) -> float:
    peak, worst = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        worst = max(worst, 1 - v / peak)
    return worst


def performance(curve: Sequence[float], trades: Sequence[ClosedTrade], commissions: float) -> Performance:
    if len(curve) < 2:
        raise ValueError("need at least two equity points")
    rets = [b / a - 1 for a, b in zip(curve[:-1], curve[1:])]
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1))
    years = len(rets) / TRADING_DAYS
    total = curve[-1] / curve[0] - 1
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl <= 0]
    return Performance(
        start_equity=curve[0],
        end_equity=curve[-1],
        total_return=total,
        annual_return=(1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0,
        max_drawdown=max_drawdown(curve),
        sharpe=mean / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0,
        trades=len(trades),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=sum(wins) / sum(losses) if losses and sum(losses) > 0 else math.inf,
        commissions=commissions,
    )
