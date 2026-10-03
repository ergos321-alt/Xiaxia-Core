from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class ModelProviderError(RuntimeError):
    """A safe, provider-independent model failure."""


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class GenerateResult:
    text: str
    model: str


class ModelProvider(ABC):
    name: str

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def generate(self, messages: list[ModelMessage]) -> GenerateResult:
        raise NotImplementedError
