from __future__ import annotations

from app.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def definitions(self) -> list[dict[str, object]]:
        return [
            {
                "name": tool.definition.name,
                "description": tool.definition.description,
                "input_schema": tool.definition.input_schema,
            }
            for tool in self._tools.values()
        ]
