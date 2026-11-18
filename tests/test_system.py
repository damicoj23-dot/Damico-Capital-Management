import asyncio
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from damico import indicators as ind
from damico.agents.base import AgentContext
from damico.agents.core import last_prices
from damico.broker import SimulatedBroker, ibkr_tiered_commission
from damico.data.source import Bar, BarStore, SymbolInfo, load_csv_dir
from damico.data.synthetic import generate
from damico.messages import Action, PortfolioSnapshot, RiskDecision, Side, TradeProposal
from damico.portfolio import PortfolioBook
from damico.risk import RiskEngine, RiskLimits
from damico.sim.runner import WARMUP_DAYS, run
from damico.store import InMemoryStore

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def source():
    return generate(n_symbols=12, n_days=300, seed=3)


def ctx_at(day: date) -> AgentContext:
    return AgentContext(run_id="t", cycle=1, as_of=datetime.combine(day, time(15, 50), NY), store=InMemoryStore())


def proposal(ctx, symbol, qty, entry, stop, action=Action.OPEN, side=Side.LONG):
    return TradeProposal(**{**ctx.envelope(type("S", (), {"name": "strategy", "version": "0"})())},
                         symbol=symbol, side=side, action=action, quantity=qty, entry=entry, stop=stop)


# ---------------------------------------------------------------- data and indicators

def test_history_never_returns_future_bars(source):
    day = source.trading_days()[100]
    bars = source.history("SYM00", day, 500)
    assert len(bars) == 101 and bars[-1].day == day


def test_csv_loader(tmp_path):
    (tmp_path / "abc.csv").write_text("date,open,high,low,close,volume\n2025-01-02,1,2,0.5,1.5,100\n2025-01-03,1.5,2,1,1.8,200\n")
    store = load_csv_dir(tmp_path, {"ABC": "Tech"})
    assert store.symbols() == [SymbolInfo("ABC", "Tech")]
    assert store.history("ABC", date(2025, 1, 2), 5)[-1].close == 1.5


def test_indicators():
    assert ind.sma([1, 2, 3, 4], 2) == 3.5 and ind.sma([1], 2) is None
    assert ind.rsi(list(range(20))) == 100.0
    bars = [Bar("X", date(2025, 1, i + 1), 10, 11, 9, 10, 1) for i in range(15)]
    assert ind.atr(bars, 14) == pytest.approx(2.0)


# ---------------------------------------------------------------- portfolio

def test_book_accounting_round_trip():
    book = PortfolioBook(capital_base=100_000)
    book.apply_fill(day=date(2025, 1, 2), symbol="X", side=Side.LONG, action=Action.OPEN,
                    quantity=100, price=50, commission=1)
    assert book.cash == 100_000 - 5_000 - 1
    assert book.equity({"X": 55}) == pytest.approx(100_000 - 1 + 500)
    trade = book.apply_fill(day=date(2025, 1, 9), symbol="X", side=Side.LONG, action=Action.CLOSE,
                            quantity=100, price=55, commission=1)
    assert trade.pnl == pytest.approx(498) and not book.positions
    assert book.cash == pytest.approx(100_498)


def test_book_rejects_closing_what_is_not_held():
    with pytest.raises(ValueError):
        PortfolioBook(capital_base=1000).apply_fill(day=date(2025, 1, 2), symbol="X", side=Side.LONG,
                                                    action=Action.CLOSE, quantity=1, price=1, commission=0)


# ---------------------------------------------------------------- risk engine

def evaluate(risk, ctx, proposals):
    return asyncio.run(risk.evaluate(ctx, proposals))


def test_risk_caps_size_by_risk_per_trade_and_position_size(source):
    day = source.trading_days()[250]
    ctx = ctx_at(day)
    price = last_prices(source, ["SYM01"], ctx.as_of)["SYM01"]
    risk = RiskEngine(PortfolioBook(capital_base=500_000), source,
                      RiskLimits(max_adv_fraction=1.0))  # liquidity out of the way
    wide = proposal(ctx, "SYM01", 100_000, price, price - 10)
    (d,) = evaluate(risk, ctx, [wide])
    assert d.verdict == "resize"
    assert d.approved_quantity <= 2_500 / 10 and d.approved_quantity * price <= 50_000 + price


def test_risk_vetoes_missing_stop_shorts_and_halts(source):
    day = source.trading_days()[250]
    ctx = ctx_at(day)
    price = last_prices(source, ["SYM02"], ctx.as_of)["SYM02"]
    book = PortfolioBook(capital_base=500_000)
    risk = RiskEngine(book, source)
    no_stop = proposal(ctx, "SYM02", 10, price, None)
    short = proposal(ctx, "SYM02", 10, price, price * 1.05, side=Side.SHORT)
    reasons = [d.reasons[0] for d in evaluate(risk, ctx, [no_stop, short])]
    assert reasons == ["no stop price", "short selling not enabled"]

    book.day_start_equity = book.peak_equity = 520_000  # today we are down 3.8%
    (d,) = evaluate(risk, ctx, [proposal(ctx, "SYM02", 10, price, price * 0.95)])
    assert d.verdict == "veto" and "daily loss" in d.reasons[0]


def test_risk_always_allows_exits_but_caps_them_at_held(source):
    day = source.trading_days()[250]
    ctx = ctx_at(day)
    book = PortfolioBook(capital_base=500_000)
    book.apply_fill(day=day, symbol="SYM03", side=Side.LONG, action=Action.OPEN, quantity=50, price=100, commission=0)
    book.day_start_equity = book.peak_equity = 600_000  # deep in loss: entries halted
    risk = RiskEngine(book, source)
    (d,) = evaluate(risk, ctx, [proposal(ctx, "SYM03", 80, 100, None, action=Action.CLOSE)])
    assert d.verdict == "resize" and d.approved_quantity == 50


def test_risk_requests_flatten_at_max_drawdown(source):
    day = source.trading_days()[250]
    book = PortfolioBook(capital_base=500_000)
    book.apply_fill(day=day, symbol="SYM03", side=Side.LONG, action=Action.OPEN, quantity=10, price=100, commission=0)
    risk = RiskEngine(book, source)
    assert risk.wants_flatten(ctx_at(day)) is None
    book.peak_equity = 1_000_000
    assert "drawdown" in risk.wants_flatten(ctx_at(day))


def test_sector_limit_across_one_batch(source):
    day = source.trading_days()[250]
    ctx = ctx_at(day)
    same_sector = [i.symbol for i in source.symbols() if i.sector == "Technology"][:2]
    risk = RiskEngine(PortfolioBook(capital_base=500_000), source,
                      RiskLimits(max_sector=0.10, max_position=0.10, risk_per_trade=1.0, max_adv_fraction=1.0))
    prices = last_prices(source, same_sector, ctx.as_of)
    props = [proposal(ctx, s, int(49_000 / prices[s]), prices[s], prices[s] * 0.9) for s in same_sector]
    first, second = evaluate(risk, ctx, props)
    assert first.verdict == "approve" and second.verdict == "resize"
    used = sum(d.approved_quantity * prices[p.symbol] for d, p in zip((first, second), props))
    assert used <= 50_000


# ---------------------------------------------------------------- broker

def test_commission_model():
    assert ibkr_tiered_commission(10, 100) == 0.35
    assert ibkr_tiered_commission(1_000, 100) == 3.5
    assert ibkr_tiered_commission(1_000, 0.10) == 1.0  # 1% of value cap


def test_broker_sanity_check_rejects_far_price(source):
    day = source.trading_days()[250]
    ctx = ctx_at(day)
    book = PortfolioBook(capital_base=500_000)
    broker = SimulatedBroker(book, source)
    price = last_prices(source, ["SYM04"], ctx.as_of)["SYM04"]
    p = proposal(ctx, "SYM04", 10, price * 1.10, price)
    d = RiskDecision(**{**p.model_dump(include={"run_id", "cycle", "as_of"}), "agent": "risk", "agent_version": "0"},
                     proposal_id=p.message_id, verdict="approve", approved_quantity=10)
    broker_ctx = ctx_at(day)
    (r,) = asyncio.run(broker.execute(broker_ctx, [(p, d)]))
    assert r.status == "rejected" and not book.positions


# ---------------------------------------------------------------- end to end

def test_full_simulation_keeps_books_consistent(source):
    result = asyncio.run(run(source, 500_000))
    days = len(source.trading_days()) - WARMUP_DAYS
    assert len(result.equity_curve) == days + 1
    snaps = [m for m in result.orchestrator.store.all() if isinstance(m, PortfolioSnapshot)]
    for s in snaps:
        assert s.gross_exposure <= s.equity * 1.0 * 1.10  # 100% gross limit, plus price drift
        assert s.positions <= 20
    assert result.performance.trades == len(result.book.trades)
