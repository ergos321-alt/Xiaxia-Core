from __future__ import annotations

from typing import Any

from app.memory.base import MemoryClient, MemoryItem


class FakeMemoryClient(MemoryClient):
    name = "fake"

    def __init__(
        self,
        search_items: list[MemoryItem] | None = None,
        recent_items: list[MemoryItem] | None = None,
    ) -> None:
        self.search_items = search_items or []
        self.recent_items = recent_items or []
        self.added_items: list[MemoryItem] = []

    @property
    def configured(self) -> bool:
        return True

    @property
    def contract_status(self) -> str:
        return "fake"

    async def search_memory(self, query: str, limit: int) -> list[MemoryItem]:
        return self.search_items[:limit]

    async def get_recent_context(self, limit: int) -> list[MemoryItem]:
        return self.recent_items[:limit]

    async def add_memory(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> MemoryItem:
        item = MemoryItem(content=content, metadata=metadata or {})
        self.added_items.append(item)
        return item
