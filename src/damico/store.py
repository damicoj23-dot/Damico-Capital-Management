"""Message store: the record of truth for every agent message.

The MVP will back this with Postgres; the in-memory store keeps the
orchestrator testable without a database.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, TypeVar

from damico.messages import Message

M = TypeVar("M", bound=Message)


class MessageStore(Protocol):
    def append(self, message: Message) -> None: ...

    def get(self, message_id: str) -> Message | None: ...

    def by_cycle(self, run_id: str, cycle: int) -> list[Message]: ...

    def latest(self, kind: type[M], symbol: str | None = None) -> M | None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._by_id: dict[str, Message] = {}
        self._by_kind: dict[str, list[Message]] = defaultdict(list)
        self._ordered: list[Message] = []

    def append(self, message: Message) -> None:
        if message.message_id in self._by_id:
            raise ValueError(f"duplicate message id {message.message_id}")
        self._by_id[message.message_id] = message
        self._by_kind[message.kind].append(message)
        self._ordered.append(message)

    def get(self, message_id: str) -> Message | None:
        return self._by_id.get(message_id)

    def by_cycle(self, run_id: str, cycle: int) -> list[Message]:
        return [m for m in self._ordered if m.run_id == run_id and m.cycle == cycle]

    def latest(self, kind: type[M], symbol: str | None = None) -> M | None:
        for message in reversed(self._by_kind.get(kind.kind, [])):
            if symbol is None or getattr(message, "symbol", None) == symbol:
                return message  # type: ignore[return-value]
        return None

    def all(self) -> list[Message]:
        return list(self._ordered)
