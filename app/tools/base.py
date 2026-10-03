from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


class Tool(ABC):
    definition: ToolDefinition

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError
