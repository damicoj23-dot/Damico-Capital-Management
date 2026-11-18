"""Synthetic daily market data for building and testing without a data vendor.

A seeded market factor with bull, bear and choppy regimes, sector factors and
stock-specific noise. Prices are fake: use this to test that the machinery
works, never to judge whether a strategy makes money.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from damico.data.source import Bar, BarStore, SymbolInfo

SECTORS = ("Technology", "Health Care", "Financials", "Industrials", "Consumer", "Energy")

# (daily drift, daily volatility, average length in days)
REGIMES = {
    "bull": (0.0007, 0.009, 180),
    "bear": (-0.0010, 0.018, 60),
    "chop": (0.0000, 0.012, 90),
}


def business_days(start: date, count: int) -> list[date]:
    days, d = [], start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def generate(
    n_symbols: int = 40,
    n_days: int = 756,
    start: date = date(2023, 1, 2),
    seed: int = 7,
) -> BarStore:
    rng = random.Random(seed)
    days = business_days(start, n_days)

    regime, left = "bull", REGIMES["bull"][2]
    market: list[float] = []
    for _ in days:
        if left <= 0:
            regime = rng.choice([r for r in REGIMES if r != regime])
            left = max(10, int(rng.expovariate(1 / REGIMES[regime][2])))
        mu, sigma, _ = REGIMES[regime]
        market.append(rng.gauss(mu, sigma))
        left -= 1

    sector_moves = {s: [rng.gauss(0, 0.006) for _ in days] for s in SECTORS}

    info, bars = [], {}
    for i in range(n_symbols):
        symbol = f"SYM{i:02d}"
        sector = SECTORS[i % len(SECTORS)]
        beta = rng.uniform(0.7, 1.4)
        alpha = rng.gauss(0.0002, 0.0004)
        idio = rng.uniform(0.008, 0.02)
        price = rng.uniform(20, 400)
        shares_per_day = rng.uniform(2e6, 2e7) / math.sqrt(price)
        rows = []
        for t, day in enumerate(days):
            r = alpha + beta * market[t] + sector_moves[sector][t] + rng.gauss(0, idio)
            open_ = price * (1 + rng.gauss(0, idio / 3))
            close = max(1.0, price * math.exp(r))
            span = abs(r) + rng.uniform(0.002, idio)
            high = max(open_, close) * (1 + span / 2)
            low = min(open_, close) * (1 - span / 2)
            volume = shares_per_day * rng.uniform(0.6, 1.6) * (1 + 10 * abs(r))
            rows.append(Bar(symbol, day, open_, high, low, close, volume))
            price = close
        info.append(SymbolInfo(symbol, sector))
        bars[symbol] = rows
    return BarStore(bars, info)
