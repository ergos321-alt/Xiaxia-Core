from __future__ import annotations

import logging
from dataclasses import dataclass

from app.browser import BrowserClient, BrowserHandClient, DisabledBrowserClient
from app.config import Settings
from app.continuity import (
    ContinuityStore,
    DisabledContinuityStore,
    ExternalContinuityRateLimiter,
    PostgresContinuityStore,
)
from app.core import (
    BrowserOrchestrator,
    ContextAssembler,
    ContextPolicy,
    ConversationRuntime,
)
from app.core.digital_identity_gateway import DigitalIdentityGateway
from app.execution import ClaimGate, ExecutionLedger
from app.identity import IdentityLoader
from app.memory import DisabledMemoryClient, MemoryClient, XiaxiaMemoryHttpClient
from app.models import MockModelProvider, ModelProvider, QwenProvider
from app.storage import RuntimeStore
from app.tools import ToolRegistry
from app.tools.browser import BrowserTool, BrowserToolPolicy
from app.tools.memory import MemoryTool

logger = logging.getLogger("xiaxia.services")


@dataclass(slots=True)
class RuntimeServices:
    settings: Settings
    runtime: ConversationRuntime
    provider: ModelProvider
    memory: MemoryClient
    store: RuntimeStore
    tools: ToolRegistry
    browser: BrowserClient
    continuity: ContinuityStore
    external_continuity_limiter: ExternalContinuityRateLimiter
    digital_identity_gateway: DigitalIdentityGateway
    execution_ledger: ExecutionLedger


def build_provider(settings: Settings) -> ModelProvider:
    if settings.model_provider == "mock":
        return MockModelProvider()
    if settings.model_provider == "qwen":
        return QwenProvider(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            timeout_seconds=settings.model_timeout_seconds,
            temperature=settings.model_temperature,
            max_tokens=settings.model_max_tokens,
        )
    raise ValueError(f"Unsupported MODEL_PROVIDER: {settings.model_provider}")


def build_memory_client(settings: Settings) -> MemoryClient:
    if not settings.memory_contract_configured:
        return DisabledMemoryClient(enabled=settings.memory_enabled)
    assert settings.memory_base_url
    assert settings.memory_token
    return XiaxiaMemoryHttpClient(
        base_url=settings.memory_base_url,
        token=settings.memory_token,
        timeout_seconds=settings.memory_timeout_seconds,
        recent_days=settings.memory_recent_days,
    )


def build_browser_client(settings: Settings) -> BrowserClient:
    if not settings.browser_tools_enabled:
        return DisabledBrowserClient()
    try:
        return BrowserHandClient(
            base_url=settings.browser_base_url,
            auth_token=settings.browser_auth_token,
            auth_token_file=settings.browser_auth_token_file,
            timeout_seconds=settings.browser_http_timeout_seconds,
        )
    except (OSError, ValueError):
        # Browser configuration failure must not prevent ordinary chat startup.
        logger.warning("browser_client_configuration_unavailable")
        return DisabledBrowserClient(enabled=True)


def build_continuity_store(settings: Settings) -> ContinuityStore:
    if not settings.continuity_contract_configured:
        return DisabledContinuityStore(enabled=settings.continuity_enabled)
    try:
        database_url = settings.continuity_database_url
        if settings.continuity_database_url_file:
            database_url = settings.continuity_database_url_file.read_text(
                encoding="utf-8"
            ).strip()
        if not database_url:
            raise ValueError("continuity_database_url_required")
        return PostgresContinuityStore(
            database_url,
            connect_timeout_seconds=settings.continuity_connect_timeout_seconds,
        )
    except (OSError, ValueError, TypeError):
        logger.warning("continuity_store_configuration_unavailable")
        return DisabledContinuityStore(enabled=True)


def build_services(
    settings: Settings,
    *,
    provider_override: ModelProvider | None = None,
    memory_override: MemoryClient | None = None,
    browser_override: BrowserClient | None = None,
    continuity_override: ContinuityStore | None = None,
) -> RuntimeServices:
    identity = IdentityLoader(settings.identity_dir).load()
    store = RuntimeStore(settings.database_path)
    store.initialize()
    execution_ledger = ExecutionLedger(settings.database_path)
    execution_ledger.initialize()
    provider = provider_override or build_provider(settings)
    memory = memory_override or build_memory_client(settings)
    browser = browser_override or build_browser_client(settings)
    continuity = continuity_override or build_continuity_store(settings)
    policy = ContextPolicy(
        max_history_messages=settings.max_history_messages,
        memory_search_limit=settings.memory_search_limit,
        memory_recent_limit=settings.memory_recent_limit,
    )
    assembler = ContextAssembler(policy)
    browser_policy = BrowserToolPolicy(
        max_tool_calls=max(1, settings.browser_max_tool_calls),
        max_navigation_count=max(0, settings.browser_max_navigation_count),
        status_timeout_seconds=max(0.01, settings.browser_status_timeout_seconds),
        navigate_timeout_seconds=max(0.01, settings.browser_navigate_timeout_seconds),
        read_timeout_seconds=max(0.01, settings.browser_read_timeout_seconds),
        interactive_timeout_seconds=max(0.01, settings.browser_interactive_timeout_seconds),
        status_retry_count=max(0, settings.browser_status_retry_count),
        max_page_chars=max(1_000, settings.browser_max_page_chars),
        max_screenshot_bytes=max(1, settings.browser_max_screenshot_bytes),
    )
    browser_tool = BrowserTool(browser, browser_policy, execution_ledger)
    browser_orchestrator = BrowserOrchestrator(
        tool=browser_tool,
        planner=provider,
        enabled=settings.browser_tools_enabled,
        overall_timeout_seconds=settings.browser_overall_timeout_seconds,
        decision_timeout_seconds=settings.browser_decision_timeout_seconds,
    )
    runtime = ConversationRuntime(
        identity=identity,
        assembler=assembler,
        provider=provider,
        memory=memory,
        store=store,
        browser=browser_orchestrator,
        browser_native_tool_calling_enabled=settings.browser_native_tool_calling_enabled,
        continuity=continuity,
        continuity_enabled=settings.continuity_enabled,
        continuity_xiaxia_identity_id=settings.continuity_xiaxia_identity_id,
        continuity_user_identity_id=settings.continuity_user_identity_id,
        continuity_source=settings.continuity_source,
        continuity_capsule_max_events=max(1, settings.continuity_capsule_max_events),
        continuity_capsule_max_chars=max(500, settings.continuity_capsule_max_chars),
        continuity_recent_days=max(1, settings.continuity_recent_days),
        execution_ledger=execution_ledger,
        claim_gate=ClaimGate(),
    )
    tools = ToolRegistry()
    tools.register(MemoryTool(memory))
    if settings.browser_tools_enabled:
        tools.register(browser_tool)
    digital_identity_gateway = DigitalIdentityGateway(
        runtime=runtime,
        browser=browser,
        execution_ledger=execution_ledger,
        enabled=settings.digital_identity_gateway_enabled,
        public_base_url=settings.takeover_public_base_url,
        takeover_ttl_seconds=max(60, min(settings.takeover_ttl_seconds, 900)),
    )
    return RuntimeServices(
        settings=settings,
        runtime=runtime,
        provider=provider,
        memory=memory,
        store=store,
        tools=tools,
        browser=browser,
        continuity=continuity,
        external_continuity_limiter=ExternalContinuityRateLimiter(
            settings.custom_gpt_continuity_rate_limit_per_minute
        ),
        digital_identity_gateway=digital_identity_gateway,
        execution_ledger=execution_ledger,
    )
