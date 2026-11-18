"""Human override controls. Changes take effect at the next cycle and are logged."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OverrideState:
    paused_entries: bool = False
    """No new entries (Action.OPEN). Exits still go through."""

    flatten_requested: bool = False
    """Close everything at the next cycle, then pause entries."""

    blocked_symbols: set[str] = field(default_factory=set)
    """No new entries in these symbols."""

    approve_each_trade_above: float | None = None
    """Entries with notional above this (USD) wait for a human approval."""

    approved_proposals: set[str] = field(default_factory=set)
    """Held proposal ids a human has approved; they re-enter the next cycle."""
