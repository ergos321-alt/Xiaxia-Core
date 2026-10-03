from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_.:-]{1,160}$", re.IGNORECASE)
_SENSITIVE_ERROR_CODE = re.compile(
    r"authorization|bearer|cookie|token|secret|password|passwd|otp|"
    r"verification.?code|captcha|credential|session.?id|验证码|密码|口令",
    re.IGNORECASE,
)


def _safe_browser_error_code(value: object) -> str:
    candidate = str(value or "").strip()
    if not _SAFE_ERROR_CODE.fullmatch(candidate) or _SENSITIVE_ERROR_CODE.search(
        candidate
    ):
        return "browser_request_failed"
    return candidate


class BrowserClientError(RuntimeError):
    """A structured, secret-free Browser Hand failure."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class BrowserClient(ABC):
    name: str

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, action: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError


class DisabledBrowserClient(BrowserClient):
    name = "disabled"

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled

    @property
    def configured(self) -> bool:
        return False

    async def execute(self, action: str, arguments: dict[str, Any]) -> Any:
        raise BrowserClientError("browser_unavailable", retryable=True)


class BrowserHandClient(BrowserClient):
    """HTTP client for the one shared Browser Hand runtime; never launches Chromium."""

    name = "xiaxia-browser-hand-v1"
    _LOCAL_HOSTS: ClassVar[set[str]] = {"127.0.0.1", "localhost", "::1"}

    def __init__(
        self,
        *,
        base_url: str,
        auth_token: str | None = None,
        auth_token_file: Path | None = None,
        timeout_seconds: float = 12.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in self._LOCAL_HOSTS:
            raise ValueError("BROWSER_BASE_URL must be a localhost-only HTTP origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("BROWSER_BASE_URL must not contain credentials or extras")
        self._base_url = base_url.rstrip("/")
        self._token = auth_token or self._read_token(auth_token_file)
        if not self._token:
            raise ValueError("Browser Hand authentication is not configured")
        self._timeout = max(1.0, float(timeout_seconds))
        self._transport = transport

    @staticmethod
    def _read_token(path: Path | None) -> str | None:
        if path is None:
            return None
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "BROWSER_AUTH_TOKEN":
                return value.strip().strip('"').strip("'") or None
        return None

    @property
    def configured(self) -> bool:
        return True

    def _request_spec(
        self, action: str, arguments: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        if action == "status":
            return "GET", "/browser/status", None, None
        if action == "tabs":
            return "GET", "/tabs", None, None
        if action == "new_tab":
            return "POST", "/tabs", arguments, None
        if action == "close_tab":
            tab_id = arguments.get("tabId")
            if not isinstance(tab_id, str) or not tab_id:
                raise BrowserClientError("browser_tab_id_required")
            return "DELETE", f"/tabs/{tab_id}", None, None
        if action in {"switch_tab", "suspend_tab", "restore_tab"}:
            return "POST", f"/tabs/{action.removesuffix('_tab')}", arguments, None
        if action == "navigate":
            return "POST", "/navigate", arguments, None
        if action == "takeover_create":
            return "POST", "/takeover/sessions", arguments, None
        if action in {"takeover_status", "takeover_revoke", "takeover_consume"}:
            session_id = arguments.get("sessionId")
            if not isinstance(session_id, str) or not session_id:
                raise BrowserClientError("takeover_session_id_required")
            if action == "takeover_status":
                return "GET", f"/takeover/sessions/{session_id}", None, None
            if action == "takeover_consume":
                return "POST", f"/takeover/sessions/{session_id}/consume", None, None
            return "DELETE", f"/takeover/sessions/{session_id}", None, None
        if action == "read_page":
            return "GET", "/page", None, arguments
        if action in {
            "click",
            "type",
            "submit",
            "scroll",
            "seek",
            "back",
            "forward",
            "refresh",
            "screenshot",
        }:
            return "POST", f"/{action}", arguments, None
        raise BrowserClientError("unsupported_browser_action")

    async def execute(self, action: str, arguments: dict[str, Any]) -> Any:
        method, path, body, query = self._request_spec(action, arguments)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=headers,
                    json=body,
                    params=query,
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise BrowserClientError("malformed_browser_response") from exc
            if response.is_error:
                error = payload.get("error") if isinstance(payload, dict) else None
                code = error.get("code") if isinstance(error, dict) else None
                retryable = (
                    bool(error.get("retryable")) if isinstance(error, dict) else False
                )
                raise BrowserClientError(
                    _safe_browser_error_code(code),
                    retryable=retryable or response.status_code >= 500,
                )
            if (
                not isinstance(payload, dict)
                or payload.get("ok") is not True
                or "data" not in payload
            ):
                raise BrowserClientError("malformed_browser_response")
            return payload["data"]
        except httpx.TimeoutException as exc:
            raise BrowserClientError("browser_timeout", retryable=True) from exc
        except httpx.RequestError as exc:
            raise BrowserClientError("browser_unavailable", retryable=True) from exc
