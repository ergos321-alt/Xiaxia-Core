from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from app.memory.base import MemoryClient, MemoryItem, MemoryUnavailableError


class DisabledMemoryClient(MemoryClient):
    name = "disabled"

    def __init__(self, *, enabled: bool = False) -> None:
        self._enabled = enabled

    @property
    def configured(self) -> bool:
        return False

    @property
    def contract_status(self) -> str:
        return "configuration_missing" if self._enabled else "disabled"

    async def search_memory(self, query: str, limit: int) -> list[MemoryItem]:
        return []

    async def get_recent_context(self, limit: int) -> list[MemoryItem]:
        return []

    async def add_memory(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> MemoryItem:
        raise MemoryUnavailableError("Memory writeback is not configured")


class XiaxiaMemoryHttpClient(MemoryClient):
    """Strict client for the public Xiaxia Memory V1 production contract."""

    name = "xiaxia-memory-v1"
    SEARCH_PATH = "/api/v1/memories/search"
    RECENT_PATH = "/api/v1/context/recent"
    SAVE_PATH = "/api/v1/memories/save"
    RECENT_SECTIONS = (
        "recent_relationship",
        "recent_shared_life",
        "xiaxia_development",
        "ongoing_threads",
        "important_recent_events",
    )

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: float,
        recent_days: int = 14,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds
        self._recent_days = max(1, min(int(recent_days), 90))
        self._transport = transport

        parsed_base = urlparse(self._base_url)
        if parsed_base.scheme not in {"http", "https"} or not parsed_base.netloc:
            raise ValueError("MEMORY_BASE_URL must be an HTTP(S) origin")
        if not token:
            raise ValueError("MEMORY_TOKEN is required by Xiaxia Memory V1")

    @property
    def configured(self) -> bool:
        return True

    @property
    def contract_status(self) -> str:
        return "configured"

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    json=json_body,
                    params=query,
                )
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise MemoryUnavailableError("Xiaxia Memory request failed") from exc

    @staticmethod
    def _parse_record(raw: Any, *, recent_section: str | None = None) -> MemoryItem:
        if not isinstance(raw, dict):
            raise MemoryUnavailableError("Xiaxia Memory response record is invalid")
        content = raw.get("memory_text")
        if not isinstance(content, str) or not content.strip():
            raise MemoryUnavailableError("Xiaxia Memory response has no memory_text")
        metadata = {key: value for key, value in raw.items() if key != "memory_text"}
        if recent_section:
            metadata["recent_section"] = recent_section
        return MemoryItem(content=content, metadata=metadata)

    def _parse_memory_list(self, body: Any) -> list[MemoryItem]:
        if not isinstance(body, dict) or not isinstance(body.get("memories"), list):
            raise MemoryUnavailableError("Xiaxia Memory list response is invalid")
        return [self._parse_record(raw) for raw in body["memories"]]

    async def search_memory(self, query: str, limit: int) -> list[MemoryItem]:
        body = await self._request(
            "POST",
            self.SEARCH_PATH,
            json_body={
                "query": str(query),
                "filters": {},
                "top_k": max(1, min(int(limit), 20)),
            },
        )
        return self._parse_memory_list(body)

    async def get_recent_context(self, limit: int) -> list[MemoryItem]:
        body = await self._request(
            "GET",
            self.RECENT_PATH,
            query={
                "days": self._recent_days,
                "max_items": max(5, min(int(limit), 40)),
            },
        )
        if not isinstance(body, dict) or not isinstance(body.get("principle"), str):
            raise MemoryUnavailableError("Xiaxia Memory recent response is invalid")
        output: list[MemoryItem] = []
        for section in self.RECENT_SECTIONS:
            raw_items = body.get(section)
            if not isinstance(raw_items, list):
                raise MemoryUnavailableError("Xiaxia Memory recent section is invalid")
            output.extend(
                self._parse_record(raw, recent_section=section) for raw in raw_items
            )
        return output

    async def add_memory(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> MemoryItem:
        memory_input = {"memory_text": content, **(metadata or {})}
        body = await self._request(
            "POST", self.SAVE_PATH, json_body={"memories": [memory_input]}
        )
        items = self._parse_memory_list(body)
        if body.get("saved") != 1 or len(items) != 1:
            raise MemoryUnavailableError("Xiaxia Memory writeback count is invalid")
        return items[0]
