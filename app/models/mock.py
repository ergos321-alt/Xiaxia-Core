from __future__ import annotations

from app.models.base import GenerateResult, ModelMessage, ModelProvider


class MockModelProvider(ModelProvider):
    name = "mock"

    def __init__(self, reply: str = "Mock Xiaxia response") -> None:
        self.reply = reply
        self.last_messages: list[ModelMessage] = []

    @property
    def configured(self) -> bool:
        return True

    async def generate(self, messages: list[ModelMessage]) -> GenerateResult:
        self.last_messages = list(messages)
        return GenerateResult(text=self.reply, model="mock-model")
