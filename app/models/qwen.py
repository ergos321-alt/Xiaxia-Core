from __future__ import annotations

from typing import Any

import httpx

from app.models.base import (
    GenerateResult,
    ModelMessage,
    ModelProvider,
    ModelProviderError,
)


class QwenProvider(ModelProvider):
    """Qwen through Alibaba Cloud Model Studio's OpenAI-compatible endpoint."""

    name = "qwen"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._api_key and self._base_url and self._model)

    async def generate(self, messages: list[ModelMessage]) -> GenerateResult:
        if not self.configured:
            raise ModelProviderError("Qwen provider is not configured")

        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            text = body["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty model response")
            response_model = body.get("model") or self._model
            return GenerateResult(text=text.strip(), model=str(response_model))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            # Never include response bodies or credentials in the client-facing error.
            raise ModelProviderError("Qwen generation failed") from exc

    async def complete_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str]:
        if not self.configured:
            raise ModelProviderError("Qwen provider is not configured")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": messages,
                        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("invalid assistant message")
            content = message.get("content")
            calls = message.get("tool_calls")
            if content is not None and not isinstance(content, str):
                raise ValueError("invalid assistant content")
            if calls is not None and not isinstance(calls, list):
                raise ValueError("invalid tool calls")
            if not calls and (not isinstance(content, str) or not content.strip()):
                raise ValueError("empty model response")
            return message, str(body.get("model") or self._model)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelProviderError("Qwen generation failed") from exc
