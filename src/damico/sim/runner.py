"""Run the full agent system day by day through the real orchestrator.

This is the event-driven replay from the design doc: the same Market,
Technical, Strategy, Risk, Execution and Portfolio code that will run live,
fed from a data source one day at a time with no lookahead.

    python -m damico.sim.runner              # synthetic data, $500,000
    python -m damico.sim.runner --csv DIR    # your own daily CSV files
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from damico.agents.core import MarketAgent, PortfolioAgent, StrategyConfig, TechnicalAgent, TrendStrategy
from damico.broker import BrokerConfig, SimulatedBroker
from damico.data.source import MarketDataSource, load_csv_dir
from damico.data.synthetic import generate
from damico.messages import PortfolioSnapshot
from damico.orchestrator import Orchestrator, OrchestratorConfig
from damico.portfolio import PortfolioBook
from damico.risk import RiskEngine, RiskLimits
from damico.sim.metrics import Performance, performance

NEW_YORK = ZoneInfo("America/New_York")
DECISION_TIME = time(15, 50)  # one decision per day, ten minutes before the close
WARMUP_DAYS = 210  # the 200-day average needs history before trading starts


@dataclass
class SimResult:
    performance: Performance
    equity_curve: list[float]
    orchestrator: Orchestrator
    book: PortfolioBook


def build(
    source: MarketDataSource,
    capital_base: float = 500_000,
    limits: RiskLimits | None = None,
    strategy: StrategyConfig | None = None,
    broker: BrokerConfig | None = None,
) -> tuple[Orchestrator, PortfolioBook]:
    book = PortfolioBook(capital_base=capital_base)
    orch = Orchestrator(
        analysis_stages=[[PortfolioAgent(book, source)], [MarketAgent(source)], [TechnicalAgent(source)]],
        strategy=TrendStrategy(source, book, strategy),
        risk=RiskEngine(book, source, limits),
        execution=SimulatedBroker(book, source, broker),
        portfolio=PortfolioAgent(book, source),
        # Daily replay: inputs are fresh within the same simulated day.
        config=OrchestratorConfig(),
    )
    return orch, book


async def run(source: MarketDataSource, capital_base: float = 500_000, **kw) -> SimResult:
    orch, book = build(source, capital_base, **kw)
    curve = [capital_base]
    for day in source.trading_days()[WARMUP_DAYS:]:
        now = datetime.combine(day, DECISION_TIME, NEW_YORK)
        report = await orch.run_cycle(now)
        snaps = [m for m in orch.store.by_cycle(orch.run_id, report.cycle) if isinstance(m, PortfolioSnapshot)]
        if snaps:
            curve.append(snaps[-1].equity)
    commissions = sum(f.commission for f in book.fills)
    return SimResult(performance(curve, book.trades, commissions), curve, orch, book)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="folder of SYMBOL.csv daily files (date,open,high,low,close,volume)")
    parser.add_argument("--capital", type=float, default=500_000)
    parser.add_argument("--seed", type=int, default=7, help="synthetic data seed")
    parser.add_argument("--days", type=int, default=756, help="synthetic trading days")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    source = load_csv_dir(args.csv) if args.csv else generate(n_days=args.days, seed=args.seed)
    result = asyncio.run(run(source, args.capital))
    label = f"CSV data from {args.csv}" if args.csv else f"synthetic data, seed {args.seed}"
    print(f"Simulation on {label}: {len(result.equity_curve) - 1} trading days\n")
    print("\n".join(result.performance.lines()))
    halts = [m for m in result.orchestrator.store.all()
             if getattr(m, "change", None) == "flatten" and getattr(m, "who", None) == "risk"]
    if halts:
        print(f"\nRisk flattened the book on {halts[0].as_of.date()} ({halts[0].why}); "
              "new entries stay paused until a human resumes them.")
    if not args.csv:
        print("\nSynthetic prices are random: these numbers test the machinery, not the strategy.")


if __name__ == "__main__":
    main()
