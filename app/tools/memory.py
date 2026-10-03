from __future__ import annotations

from typing import Any

from app.memory import MemoryClient
from app.tools.base import Tool, ToolDefinition, ToolExecutionError


class MemoryTool(Tool):
    definition = ToolDefinition(
        name="memory",
        description="Explicitly search, read recent, or write Xiaxia durable memory.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "recent", "add"]},
                "query": {"type": "string"},
                "content": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "metadata": {"type": "object"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    )

    def __init__(self, client: MemoryClient) -> None:
        self.client = client

    async def execute(self, arguments: dict[str, Any]) -> Any:
        action = arguments.get("action")
        limit = int(arguments.get("limit", 5))
        if not 1 <= limit <= 50:
            raise ToolExecutionError("limit must be between 1 and 50")
        if action == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolExecutionError("query is required for search")
            return await self.client.search_memory(query, limit)
        if action == "recent":
            return await self.client.get_recent_context(limit)
        if action == "add":
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ToolExecutionError("content is required for add")
            metadata = arguments.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ToolExecutionError("metadata must be an object")
            return await self.client.add_memory(content, metadata)
        raise ToolExecutionError("unsupported memory action")
