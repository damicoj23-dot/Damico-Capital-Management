"""US equity market hours (regular session only; no extended hours in the MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketClock:
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    holidays: frozenset[date] = field(default_factory=frozenset)
    """Full-day NYSE closures. Load the exchange calendar here before going live."""

    def is_open(self, now: datetime) -> bool:
        local = now.astimezone(NEW_YORK)
        if local.weekday() >= 5 or local.date() in self.holidays:
            return False
        return self.open_time <= local.time() < self.close_time
