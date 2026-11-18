"""Typed messages that agents exchange.

Every message carries the same envelope (run, cycle, timestamp, producing agent
and version, and the ids of the messages it was built from) so that any
decision can be traced and replayed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> str:
    return uuid4().hex


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Instrument(str, Enum):
    STOCK = "stock"
    OPTION = "option"


class Action(str, Enum):
    """OPEN adds risk (new entry); CLOSE reduces it (exit, stop, take profit)."""

    OPEN = "open"
    CLOSE = "close"


class Message(BaseModel):
    """Envelope shared by every agent message."""

    model_config = ConfigDict(frozen=True)

    kind: ClassVar[str] = "message"

    message_id: str = Field(default_factory=new_id)
    run_id: str
    cycle: int
    as_of: datetime
    agent: str
    agent_version: str
    input_ids: tuple[str, ...] = ()
    degraded: bool = False
    """True when the agent failed or timed out and this is a neutral fallback."""


class MarketState(Message):
    kind: ClassVar[str] = "market_state"

    regime: Literal["trending", "ranging", "volatile", "unknown"] = "unknown"
    tradable_symbols: tuple[str, ...] = ()


class Signal(Message):
    kind: ClassVar[str] = "signal"

    symbol: str
    score: float = Field(ge=-1.0, le=1.0)
    setup: str = ""
    stop: float | None = None


class NewsEvent(Message):
    kind: ClassVar[str] = "news_event"

    symbol: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    materiality: float = Field(ge=0.0, le=1.0)
    summary: str = ""


class FundamentalScore(Message):
    kind: ClassVar[str] = "fundamental_score"

    symbol: str
    quality: float = Field(ge=0, le=100)
    value: float = Field(ge=0, le=100)
    growth: float = Field(ge=0, le=100)


class MacroRegime(Message):
    kind: ClassVar[str] = "macro_regime"

    regime: Literal["risk_on", "neutral", "risk_off"] = "neutral"
    exposure_cap: float = Field(default=1.0, ge=0.0, le=1.5)
    brief: str = ""


class TradeProposal(Message):
    kind: ClassVar[str] = "trade_proposal"

    symbol: str
    side: Side
    action: Action
    instrument: Instrument = Instrument.STOCK
    quantity: int = Field(gt=0)
    entry: float = Field(gt=0)
    stop: float | None = None
    target: float | None = None
    rationale: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.entry


class RiskDecision(Message):
    kind: ClassVar[str] = "risk_decision"

    proposal_id: str
    verdict: Literal["approve", "resize", "veto"]
    approved_quantity: int = Field(default=0, ge=0)
    reasons: tuple[str, ...] = ()

    @property
    def allows_trade(self) -> bool:
        return self.verdict in ("approve", "resize") and self.approved_quantity > 0


class ExecutionReport(Message):
    kind: ClassVar[str] = "execution_report"

    proposal_id: str | None = None
    symbol: str = ""
    status: Literal["submitted", "filled", "rejected", "flatten_submitted"]
    broker_order_id: str | None = None
    detail: str = ""


class OverrideEvent(Message):
    """A human override, kept in the record with who did it and why."""

    kind: ClassVar[str] = "override_event"

    change: str
    who: str
    why: str


class HoldNotice(Message):
    """A risk-approved proposal the orchestrator did not send, and why."""

    kind: ClassVar[str] = "hold_notice"

    proposal_id: str
    symbol: str
    reason: Literal["paused", "blocked_symbol", "awaiting_approval", "approval_expired"]
