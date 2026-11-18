"""Vendor-neutral market data interface.

Agents only see `MarketDataSource`. Today it is backed by a synthetic generator
or CSV files; an Alpaca (or other vendor) adapter implements the same methods later.

The one rule every source must keep: never return data dated after `as_of`.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Bar:
    symbol: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def dollar_volume(self) -> float:
        return self.close * self.volume


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    sector: str


class MarketDataSource(Protocol):
    def symbols(self) -> list[SymbolInfo]: ...

    def history(self, symbol: str, as_of: datetime | date, lookback: int) -> list[Bar]:
        """Up to `lookback` daily bars ending on or before `as_of`, oldest first."""
        ...

    def trading_days(self) -> list[date]: ...


def _as_date(value: datetime | date) -> date:
    return value.date() if isinstance(value, datetime) else value


class BarStore:
    """Daily bars held in memory, indexed for fast point-in-time lookups."""

    def __init__(self, bars: dict[str, list[Bar]], info: list[SymbolInfo]) -> None:
        self._bars = {s: sorted(b, key=lambda x: x.day) for s, b in bars.items()}
        self._days = {s: [b.day for b in rows] for s, rows in self._bars.items()}
        self._info = info

    def symbols(self) -> list[SymbolInfo]:
        return list(self._info)

    def history(self, symbol: str, as_of: datetime | date, lookback: int) -> list[Bar]:
        rows = self._bars.get(symbol)
        if not rows:
            return []
        end = bisect_right(self._days[symbol], _as_date(as_of))
        return rows[max(0, end - lookback) : end]

    def trading_days(self) -> list[date]:
        days = set()
        for d in self._days.values():
            days.update(d)
        return sorted(days)


def load_csv_dir(path: str | Path, sectors: dict[str, str] | None = None) -> BarStore:
    """Load one CSV per symbol (SYMBOL.csv with date,open,high,low,close,volume).

    Lets you test on real history downloaded by hand, with no vendor connection.
    """
    bars: dict[str, list[Bar]] = {}
    for file in sorted(Path(path).glob("*.csv")):
        symbol = file.stem.upper()
        with file.open(newline="") as fh:
            bars[symbol] = [
                Bar(
                    symbol=symbol,
                    day=date.fromisoformat(row["date"][:10]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                for row in csv.DictReader(fh)
            ]
    sectors = sectors or {}
    info = [SymbolInfo(s, sectors.get(s, "Unknown")) for s in sorted(bars)]
    return BarStore(bars, info)
