from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class MemoryUnavailableError(RuntimeError):
    """Memory is disabled, misconfigured, or temporarily unreachable."""


@dataclass(frozen=True, slots=True)
class MemoryItem:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryClient(ABC):
    name: str

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def contract_status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def search_memory(self, query: str, limit: int) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    async def get_recent_context(self, limit: int) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    async def add_memory(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> MemoryItem:
        raise NotImplementedError
