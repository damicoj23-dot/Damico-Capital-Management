"""Technical indicators on daily bars. Pure Python, latest value only."""

from __future__ import annotations

from collections.abc import Sequence

from damico.data.source import Bar


def sma(values: Sequence[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def ema(values: Sequence[float], n: int) -> float | None:
    if len(values) < n:
        return None
    k = 2 / (n + 1)
    out = sum(values[:n]) / n
    for v in values[n:]:
        out = v * k + out * (1 - k)
    return out


def atr(bars: Sequence[Bar], n: int = 14) -> float | None:
    if len(bars) < n + 1:
        return None
    trs = [
        max(b.high - b.low, abs(b.high - prev.close), abs(b.low - prev.close))
        for prev, b in zip(bars[:-1], bars[1:])
    ]
    return sum(trs[-n:]) / n


def rsi(values: Sequence[float], n: int = 14) -> float | None:
    if len(values) < n + 1:
        return None
    changes = [b - a for a, b in zip(values[:-1], values[1:])][-n:]
    gains = sum(c for c in changes if c > 0) / n
    losses = -sum(c for c in changes if c < 0) / n
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


def highest(values: Sequence[float], n: int) -> float | None:
    return max(values[-n:]) if len(values) >= n else None


def avg_dollar_volume(bars: Sequence[Bar], n: int = 20) -> float | None:
    if len(bars) < n:
        return None
    return sum(b.dollar_volume for b in bars[-n:]) / n
