from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "xiaxia-core"
    version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    identity_dir: Path = Path("identity")
    database_path: Path = Path("data/xiaxia-core.db")

    model_provider: str = "qwen"
    qwen_api_key: str | None = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-flash"
    model_timeout_seconds: float = 60.0
    model_temperature: float = 0.7
    model_max_tokens: int = 2048

    memory_enabled: bool = False
    memory_base_url: str | None = None
    memory_token: str | None = None
    memory_timeout_seconds: float = 10.0
    memory_search_limit: int = 8
    memory_recent_days: int = 14
    memory_recent_limit: int = 5
    max_history_messages: int = 20
    core_api_token: str | None = None
    reading_assertion_secret: str | None = None

    browser_tools_enabled: bool = False
    browser_native_tool_calling_enabled: bool = False
    browser_base_url: str = "http://127.0.0.1:8766"
    browser_auth_token: str | None = None
    browser_auth_token_file: Path | None = None
    browser_http_timeout_seconds: float = 30.0
    browser_status_timeout_seconds: float = 8.0
    browser_navigate_timeout_seconds: float = 25.0
    browser_read_timeout_seconds: float = 22.0
    browser_interactive_timeout_seconds: float = 18.0
    browser_status_retry_count: int = 1
    browser_overall_timeout_seconds: float = 65.0
    browser_decision_timeout_seconds: float = 10.0
    browser_max_tool_calls: int = 16
    browser_max_navigation_count: int = 4
    browser_max_page_chars: int = 12_000
    browser_max_screenshot_bytes: int = 3_000_000
    digital_identity_gateway_enabled: bool = False
    takeover_public_base_url: str | None = None
    takeover_ttl_seconds: int = 600

    continuity_enabled: bool = False
    continuity_database_url: str | None = None
    continuity_database_url_file: Path | None = None
    continuity_connect_timeout_seconds: float = 5.0
    continuity_xiaxia_identity_id: str = "xiaxia_primary"
    continuity_user_identity_id: str = "primary_user"
    continuity_source: str = "app"
    continuity_capsule_max_events: int = 12
    continuity_capsule_max_chars: int = 4_000
    continuity_recent_days: int = 30

    custom_gpt_continuity_enabled: bool = False
    custom_gpt_continuity_token: str | None = None
    custom_gpt_continuity_token_file: Path | None = None
    custom_gpt_continuity_request_max_bytes: int = 16_384
    custom_gpt_continuity_summary_max_chars: int = 1_000
    custom_gpt_continuity_metadata_max_chars: int = 2_048
    custom_gpt_continuity_metadata_max_depth: int = 4
    custom_gpt_continuity_rate_limit_per_minute: int = 60

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            service_name=os.getenv("SERVICE_NAME", "xiaxia-core"),
            version=os.getenv("SERVICE_VERSION", "0.1.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            identity_dir=Path(os.getenv("IDENTITY_DIR", "identity")),
            database_path=Path(os.getenv("DATABASE_PATH", "data/xiaxia-core.db")),
            model_provider=os.getenv("MODEL_PROVIDER", "qwen"),
            qwen_api_key=os.getenv("QWEN_API_KEY") or None,
            qwen_base_url=os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            qwen_model=os.getenv("QWEN_MODEL", "qwen-flash"),
            model_timeout_seconds=_float_env("MODEL_TIMEOUT_SECONDS", 60.0),
            model_temperature=_float_env("MODEL_TEMPERATURE", 0.7),
            model_max_tokens=_int_env("MODEL_MAX_TOKENS", 2048),
            memory_enabled=_bool_env("MEMORY_ENABLED", False),
            memory_base_url=os.getenv("MEMORY_BASE_URL") or None,
            memory_token=os.getenv("MEMORY_TOKEN") or None,
            memory_timeout_seconds=_float_env("MEMORY_TIMEOUT_SECONDS", 10.0),
            memory_search_limit=_int_env("MEMORY_SEARCH_LIMIT", 8),
            memory_recent_days=_int_env("MEMORY_RECENT_DAYS", 14),
            memory_recent_limit=_int_env("MEMORY_RECENT_LIMIT", 5),
            max_history_messages=_int_env("MAX_HISTORY_MESSAGES", 20),
            core_api_token=os.getenv("CORE_API_TOKEN") or None,
            reading_assertion_secret=os.getenv("READING_ASSERTION_SECRET") or None,
            browser_tools_enabled=_bool_env("BROWSER_TOOLS_ENABLED", False),
            browser_native_tool_calling_enabled=_bool_env(
                "BROWSER_NATIVE_TOOL_CALLING_ENABLED", False
            ),
            browser_base_url=os.getenv(
                "BROWSER_BASE_URL", "http://127.0.0.1:8766"
            ).rstrip("/"),
            browser_auth_token=os.getenv("BROWSER_AUTH_TOKEN") or None,
            browser_auth_token_file=(
                Path(os.environ["BROWSER_AUTH_TOKEN_FILE"])
                if os.getenv("BROWSER_AUTH_TOKEN_FILE")
                else None
            ),
            browser_http_timeout_seconds=_float_env(
                "BROWSER_HTTP_TIMEOUT_SECONDS", 30.0
            ),
            browser_status_timeout_seconds=_float_env(
                "BROWSER_STATUS_TIMEOUT_SECONDS", 8.0
            ),
            browser_navigate_timeout_seconds=_float_env(
                "BROWSER_NAVIGATE_TIMEOUT_SECONDS", 25.0
            ),
            browser_read_timeout_seconds=_float_env(
                "BROWSER_READ_TIMEOUT_SECONDS", 22.0
            ),
            browser_interactive_timeout_seconds=_float_env(
                "BROWSER_INTERACTIVE_TIMEOUT_SECONDS", 18.0
            ),
            browser_status_retry_count=_int_env(
                "BROWSER_STATUS_RETRY_COUNT", 1
            ),
            browser_overall_timeout_seconds=_float_env(
                "BROWSER_OVERALL_TIMEOUT_SECONDS", 65.0
            ),
            browser_decision_timeout_seconds=_float_env(
                "BROWSER_DECISION_TIMEOUT_SECONDS", 10.0
            ),
            browser_max_tool_calls=_int_env("BROWSER_MAX_TOOL_CALLS", 16),
            browser_max_navigation_count=_int_env("BROWSER_MAX_NAVIGATION_COUNT", 4),
            browser_max_page_chars=_int_env("BROWSER_MAX_PAGE_CHARS", 12_000),
            browser_max_screenshot_bytes=_int_env(
                "BROWSER_MAX_SCREENSHOT_BYTES", 3_000_000
            ),
            digital_identity_gateway_enabled=_bool_env(
                "DIGITAL_IDENTITY_GATEWAY_ENABLED", False
            ),
            takeover_public_base_url=(
                os.getenv("TAKEOVER_PUBLIC_BASE_URL", "").rstrip("/") or None
            ),
            takeover_ttl_seconds=_int_env("TAKEOVER_TTL_SECONDS", 600),
            continuity_enabled=_bool_env("CONTINUITY_ENABLED", False),
            continuity_database_url=os.getenv("CONTINUITY_DATABASE_URL") or None,
            continuity_database_url_file=(
                Path(os.environ["CONTINUITY_DATABASE_URL_FILE"])
                if os.getenv("CONTINUITY_DATABASE_URL_FILE")
                else None
            ),
            continuity_connect_timeout_seconds=_float_env(
                "CONTINUITY_CONNECT_TIMEOUT_SECONDS", 5.0
            ),
            continuity_xiaxia_identity_id=os.getenv(
                "CONTINUITY_XIAXIA_IDENTITY_ID", "xiaxia_primary"
            ),
            continuity_user_identity_id=os.getenv(
                "CONTINUITY_USER_IDENTITY_ID", "primary_user"
            ),
            continuity_source=os.getenv("CONTINUITY_SOURCE", "app"),
            continuity_capsule_max_events=_int_env("CONTINUITY_CAPSULE_MAX_EVENTS", 12),
            continuity_capsule_max_chars=_int_env(
                "CONTINUITY_CAPSULE_MAX_CHARS", 4_000
            ),
            continuity_recent_days=_int_env("CONTINUITY_RECENT_DAYS", 30),
            custom_gpt_continuity_enabled=_bool_env(
                "CUSTOM_GPT_CONTINUITY_ENABLED", False
            ),
            custom_gpt_continuity_token=(
                os.getenv("CUSTOM_GPT_CONTINUITY_TOKEN") or None
            ),
            custom_gpt_continuity_token_file=(
                Path(os.environ["CUSTOM_GPT_CONTINUITY_TOKEN_FILE"])
                if os.getenv("CUSTOM_GPT_CONTINUITY_TOKEN_FILE")
                else None
            ),
            custom_gpt_continuity_request_max_bytes=_int_env(
                "CUSTOM_GPT_CONTINUITY_REQUEST_MAX_BYTES", 16_384
            ),
            custom_gpt_continuity_summary_max_chars=_int_env(
                "CUSTOM_GPT_CONTINUITY_SUMMARY_MAX_CHARS", 1_000
            ),
            custom_gpt_continuity_metadata_max_chars=_int_env(
                "CUSTOM_GPT_CONTINUITY_METADATA_MAX_CHARS", 2_048
            ),
            custom_gpt_continuity_metadata_max_depth=_int_env(
                "CUSTOM_GPT_CONTINUITY_METADATA_MAX_DEPTH", 4
            ),
            custom_gpt_continuity_rate_limit_per_minute=_int_env(
                "CUSTOM_GPT_CONTINUITY_RATE_LIMIT_PER_MINUTE", 60
            ),
        )

    @property
    def model_configured(self) -> bool:
        if self.model_provider == "mock":
            return True
        return self.model_provider == "qwen" and bool(
            self.qwen_api_key and self.qwen_base_url and self.qwen_model
        )

    @property
    def memory_contract_configured(self) -> bool:
        return self.memory_enabled and bool(self.memory_base_url and self.memory_token)

    @property
    def browser_contract_configured(self) -> bool:
        return self.browser_tools_enabled and bool(
            self.browser_base_url
            and (self.browser_auth_token or self.browser_auth_token_file)
        )

    @property
    def continuity_contract_configured(self) -> bool:
        return self.continuity_enabled and bool(
            self.continuity_database_url or self.continuity_database_url_file
        )
