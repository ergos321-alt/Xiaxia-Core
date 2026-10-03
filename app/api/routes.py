from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status

from app.api.schemas import (
    BrowserStatus,
    BrowserTaskRequest,
    BrowserTaskResumeRequest,
    ChatRequest,
    ChatResponse,
    ComponentStatus,
    ContinuityStatus,
    DatabaseStatus,
    ExternalContinuityCapsule,
    ExternalContinuityCapsuleEvent,
    ExternalContinuityContextResponse,
    ExternalContinuityEventWriteRequest,
    ExternalContinuityEventWriteResponse,
    HealthResponse,
    IdentityStatus,
    MemoryStatus,
    MemoryWriteRequest,
    MemoryWriteResponse,
    LifeRuntimeStatusResponse,
    ProactiveMessagesResponse,
    SharedConversationRequest,
    SharedConversationResponse,
    StatusResponse,
)
from app.browser import BrowserClientError
from app.continuity import (
    ContinuityError,
    ContinuityEvent,
    ContinuityEventInput,
    ContinuityValidationError,
    contains_sensitive_text,
    normalize_metadata,
)
from app.memory import MemoryUnavailableError
from app.models import ModelProviderError
from app.hermes_chat import (
    chat as hermes_chat,
    list_proactive_messages,
    register_shared_conversation,
)
from app.services import RuntimeServices

router = APIRouter()
logger = logging.getLogger("xiaxia.api")


def _services(request: Request) -> RuntimeServices:
    return request.app.state.services


def _authorize(services: RuntimeServices, authorization: str | None) -> None:
    expected = services.settings.core_api_token
    if not expected:
        return
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _authorize_writeback(services: RuntimeServices, authorization: str | None) -> None:
    if not services.settings.core_api_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explicit Memory writeback requires CORE_API_TOKEN",
        )
    _authorize(services, authorization)


def _external_token(services: RuntimeServices) -> str | None:
    settings = services.settings
    token = settings.custom_gpt_continuity_token
    if not token and settings.custom_gpt_continuity_token_file:
        try:
            token = settings.custom_gpt_continuity_token_file.read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            return None
    return token if token and len(token) >= 32 else None


def _authorize_external_continuity(
    services: RuntimeServices, authorization: str | None
) -> None:
    if not services.settings.custom_gpt_continuity_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    expected = _external_token(services)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "external_continuity_not_configured",
                "message": "External continuity is temporarily unavailable",
            },
        )
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not services.external_continuity_limiter.allow():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )


def _authorize_external_gateway(
    services: RuntimeServices, authorization: str | None
) -> None:
    if not services.settings.digital_identity_gateway_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    expected = _external_token(services)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "custom_browser_not_configured",
                "message": "Custom Browser capability is temporarily unavailable",
            },
        )
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not services.external_continuity_limiter.allow():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )


def _gateway_error(exc: BrowserClientError) -> HTTPException:
    code_status = {
        "sensitive_browser_task_rejected": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "takeover_source_mismatch": status.HTTP_403_FORBIDDEN,
        "takeover_session_not_found": status.HTTP_404_NOT_FOUND,
        "takeover_session_id_required": status.HTTP_422_UNPROCESSABLE_ENTITY,
    }
    return HTTPException(
        status_code=code_status.get(
            exc.code,
            status.HTTP_503_SERVICE_UNAVAILABLE
            if exc.retryable
            else status.HTTP_409_CONFLICT,
        ),
        detail={"code": exc.code, "retryable": exc.retryable},
    )
def _require_continuity_store(services: RuntimeServices) -> None:
    if not services.settings.continuity_enabled or not services.continuity.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "continuity_unavailable",
                "message": "Continuity is temporarily unavailable",
            },
        )


def _external_capsule_event(event: ContinuityEvent) -> ExternalContinuityCapsuleEvent:
    return ExternalContinuityCapsuleEvent(
        source=event.source,
        event_type=event.event_type,
        summary=event.summary,
        occurred_at=event.occurred_at,
        importance=event.importance,
        status=event.status,
        topic_key=event.metadata.get("topic_key"),
        current_task=event.metadata.get("current_task"),
        next_step=event.metadata.get("next_step"),
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services = _services(request)
    return HealthResponse(
        status="ok",
        service=services.settings.service_name,
        version=services.settings.version,
    )


@router.get("/v1/status", response_model=StatusResponse)
async def runtime_status(
    request: Request, authorization: str | None = Header(default=None)
) -> StatusResponse:
    services = _services(request)
    _authorize(services, authorization)
    database_ok = services.store.healthy()
    browser_reachable = False
    browser_ready = False
    if services.browser.configured:
        try:
            browser_runtime = await services.browser.execute("status", {})
            browser_reachable = isinstance(browser_runtime, dict)
            browser_ready = (
                browser_reachable and browser_runtime.get("state") == "ready"
            )
        except BrowserClientError:
            pass
    return StatusResponse(
        service=services.settings.service_name,
        version=services.settings.version,
        environment=services.settings.environment,
        model=ComponentStatus(
            name=services.provider.name, configured=services.provider.configured
        ),
        memory=MemoryStatus(
            name=services.memory.name,
            configured=services.memory.configured,
            contract_status=services.memory.contract_status,
        ),
        browser=BrowserStatus(
            name=services.browser.name,
            configured=services.browser.configured,
            enabled=services.settings.browser_tools_enabled,
            reachable=browser_reachable,
            browser_ready=browser_ready,
        ),
        continuity=ContinuityStatus(
            name=services.continuity.name,
            configured=services.continuity.configured,
            enabled=services.settings.continuity_enabled,
            status="ok" if services.continuity.healthy() else "error",
        ),
        database=DatabaseStatus(status="ok" if database_ok else "error"),
        identity=IdentityStatus(
            status="ok", files_loaded=services.runtime.identity.file_count
        ),
    )


@router.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    services = _services(request)
    _authorize(services, authorization)
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    try:
        result = await hermes_chat(
            message=payload.message.strip(),
            conversation_id=payload.conversation_id,
            request_id=request_id,
            settings=services.settings,
            memory=services.memory,
            store=services.store,
            browser_tool=(
                services.tools.get("browser")
                if services.settings.browser_tools_enabled
                and services.browser.configured
                else None
            ),
            execution_ledger=services.execution_ledger,
        )
    except ModelProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "model_unavailable",
                "message": "Model generation is temporarily unavailable",
                "request_id": request_id,
            },
        ) from exc
    response.headers["X-Model"] = result.model
    response.headers["X-Memory-Retrieved-Count"] = str(result.retrieved_memory_count)
    response.headers["X-Memory-Degraded"] = str(result.memory_degraded).lower()
    response.headers["X-Continuity-Degraded"] = "false"
    response.headers["X-Task-Status"] = result.task_status
    response.headers["X-Chat-Completed"] = "true"
    response.headers["X-Continuity-Capsule-Events"] = "0"
    return ChatResponse(
        reply=result.reply,
        conversation_id=result.conversation_id,
        request_id=result.request_id,
    )


@router.post(
    "/v1/life/conversation",
    response_model=SharedConversationResponse,
)
async def set_life_shared_conversation(
    payload: SharedConversationRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> SharedConversationResponse:
    services = _services(request)
    _authorize(services, authorization)
    try:
        await register_shared_conversation(
            payload.conversation_id,
            settings=services.settings,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return SharedConversationResponse(conversation_id=payload.conversation_id)


@router.get(
    "/v1/chat/proactive",
    response_model=ProactiveMessagesResponse,
)
async def get_proactive_messages(
    request: Request,
    conversation_id: str = Query(min_length=1, max_length=128),
    after_id: int = Query(default=0, ge=0),
    authorization: str | None = Header(default=None),
) -> ProactiveMessagesResponse:
    services = _services(request)
    _authorize(services, authorization)
    life_state = request.app.state.life_runtime.store.get_state()
    if life_state.shared_conversation_id != conversation_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        messages = await list_proactive_messages(conversation_id, after_id=after_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return ProactiveMessagesResponse(
        conversation_id=conversation_id,
        messages=messages,
    )


@router.get("/v1/life/status", response_model=LifeRuntimeStatusResponse)
async def life_runtime_status(
    request: Request,
    authorization: str | None = Header(default=None),
) -> LifeRuntimeStatusResponse:
    services = _services(request)
    _authorize(services, authorization)
    snapshot = await asyncio.to_thread(
        request.app.state.life_runtime.store.runtime_status,
        now=datetime.now(UTC),
    )
    return LifeRuntimeStatusResponse.model_validate(snapshot)


@router.post("/v1/memory", response_model=MemoryWriteResponse, status_code=201)
async def write_memory(
    payload: MemoryWriteRequest,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
) -> MemoryWriteResponse:
    """Explicit durable writeback; normal chat never invokes this endpoint."""

    services = _services(request)
    _authorize_writeback(services, authorization)
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    if not services.memory.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "memory_not_configured",
                "message": "Xiaxia Memory is not configured",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )
    contract_fields = payload.model_dump(
        mode="json", exclude={"memory_text"}, exclude_none=True
    )
    try:
        saved = await services.memory.add_memory(payload.memory_text, contract_fields)
    except MemoryUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "memory_unavailable",
                "message": "Xiaxia Memory writeback is temporarily unavailable",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        ) from exc
    return MemoryWriteResponse(
        memory={"memory_text": saved.content, **saved.metadata},
        request_id=request_id,
    )


@router.post(
    "/v1/continuity/events",
    response_model=ExternalContinuityEventWriteResponse,
    status_code=status.HTTP_201_CREATED,
    operation_id="writeContinuityEvent",
)
async def write_external_continuity_event(
    payload: ExternalContinuityEventWriteRequest,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
) -> ExternalContinuityEventWriteResponse:
    """Accept one meaningful Custom GPT event into the canonical timeline."""

    services = _services(request)
    _authorize_external_continuity(services, authorization)
    _require_continuity_store(services)
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    settings = services.settings
    if len(payload.summary) > settings.custom_gpt_continuity_summary_max_chars:
        raise HTTPException(status_code=422, detail="summary exceeds configured limit")
    stable_id = payload.idempotency_key or payload.source_event_id
    assert stable_id
    if contains_sensitive_text(payload.summary) or contains_sensitive_text(stable_id):
        raise HTTPException(
            status_code=422,
            detail="Continuity event contains credential-like content",
        )
    try:
        handoff_metadata = dict(payload.metadata)
        for key in ("topic_key", "current_task", "next_step"):
            value = getattr(payload, key)
            if value is not None:
                handoff_metadata[key] = value
        if payload.status != "active" and not (
            isinstance(handoff_metadata.get("topic_key"), str)
            and handoff_metadata["topic_key"].strip()
        ):
            raise HTTPException(
                status_code=422, detail="topic_key is required to close a topic"
            )
        metadata = normalize_metadata(
            handoff_metadata,
            max_chars=settings.custom_gpt_continuity_metadata_max_chars,
            max_depth=settings.custom_gpt_continuity_metadata_max_depth,
        )
    except ContinuityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    now = datetime.now(UTC)
    occurred_at = payload.occurred_at or now
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    occurred_at = occurred_at.astimezone(UTC)
    if occurred_at > now + timedelta(minutes=5):
        raise HTTPException(status_code=422, detail="occurred_at is too far in the future")
    if occurred_at < now - timedelta(days=365):
        raise HTTPException(status_code=422, detail="occurred_at is outside the accepted window")

    event = ContinuityEventInput(
        xiaxia_identity_id=settings.continuity_xiaxia_identity_id,
        user_identity_id=settings.continuity_user_identity_id,
        source="custom_gpt",
        event_type=payload.event_type,
        summary=payload.summary,
        importance=payload.importance,
        status=payload.status,
        occurred_at=occurred_at,
        metadata=metadata,
        request_id=request_id,
        source_event_id=f"external:custom_gpt:{stable_id}",
    )
    try:
        stored = await asyncio.to_thread(services.continuity.create_event, event)
    except ContinuityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ContinuityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "continuity_write_unavailable",
                "message": "Continuity write is temporarily unavailable",
                "request_id": request_id,
            },
        ) from exc
    logger.info(
        "external_continuity_event_written",
        extra={
            "fields": {
                "request_id": request_id,
                "event_id": stored.event_id,
                "source": "custom_gpt",
                "event_type": stored.event_type,
            }
        },
    )
    return ExternalContinuityEventWriteResponse(
        event_id=stored.event_id,
        event_type=payload.event_type,
        occurred_at=stored.occurred_at,
        request_id=request_id,
    )


@router.get(
    "/v1/continuity/context",
    response_model=ExternalContinuityContextResponse,
    operation_id="readContinuityContext",
)
async def read_external_continuity_context(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
) -> ExternalContinuityContextResponse:
    """Return the existing bounded canonical capsule, never a raw timeline."""

    services = _services(request)
    _authorize_external_continuity(services, authorization)
    _require_continuity_store(services)
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    settings = services.settings
    try:
        capsule = await asyncio.to_thread(
            services.continuity.load_capsule,
            xiaxia_identity_id=settings.continuity_xiaxia_identity_id,
            user_identity_id=settings.continuity_user_identity_id,
            max_events=max(1, settings.continuity_capsule_max_events),
            max_chars=max(500, settings.continuity_capsule_max_chars),
            recent_days=max(1, settings.continuity_recent_days),
        )
    except ContinuityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "continuity_read_unavailable",
                "message": "Continuity read is temporarily unavailable",
                "request_id": request_id,
            },
        ) from exc
    event_views = [_external_capsule_event(event) for event in capsule.events]
    cross_entry = [
        _external_capsule_event(event)
        for event in reversed(capsule.events)
        if event.source != "custom_gpt"
    ][:5]
    logger.info(
        "external_continuity_capsule_read",
        extra={
            "fields": {
                "request_id": request_id,
                "capsule_event_count": len(event_views),
                "capsule_char_count": capsule.char_count,
                "capsule_truncated": capsule.truncated,
            }
        },
    )
    return ExternalContinuityContextResponse(
        continuity_capsule=ExternalContinuityCapsule(
            recent_important_events=event_views,
            active_topic=capsule.active_topic,
            unfinished_topics=list(capsule.unfinished_topics),
            recent_browser_experiences=list(capsule.recent_browser_events),
            recent_cross_entry_events=cross_entry,
            truncated=capsule.truncated,
            char_count=capsule.char_count,
        ),
        request_id=request_id,
    )


async def _execute_browser_task(
    services: RuntimeServices, payload: BrowserTaskRequest, source: str
) -> dict[str, object]:
    try:
        return await services.digital_identity_gateway.execute(
            task=payload.task,
            source=source,  # type: ignore[arg-type]
            conversation_id=payload.conversation_id,
            allow_human_takeover=payload.allow_human_takeover,
        )
    except BrowserClientError as exc:
        raise _gateway_error(exc) from exc


@router.post("/v1/browser/tasks", operation_id="submitAppBrowserTask")
async def submit_app_browser_task(
    payload: BrowserTaskRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize(services, authorization)
    return await _execute_browser_task(services, payload, "app")


@router.post("/v1/custom/browser/tasks", operation_id="submitCustomBrowserTask")
async def submit_custom_browser_task(
    payload: BrowserTaskRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize_external_gateway(services, authorization)
    return await _execute_browser_task(services, payload, "custom_gpt")


async def _resume_browser_task(
    services: RuntimeServices,
    session_id: str,
    payload: BrowserTaskResumeRequest,
    source: str,
) -> dict[str, object]:
    try:
        return await services.digital_identity_gateway.resume(
            session_id=session_id,
            source=source,  # type: ignore[arg-type]
            conversation_id=payload.conversation_id,
        )
    except BrowserClientError as exc:
        raise _gateway_error(exc) from exc


@router.post(
    "/v1/browser/tasks/{session_id}/resume", operation_id="resumeAppBrowserTask"
)
async def resume_app_browser_task(
    session_id: str,
    payload: BrowserTaskResumeRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize(services, authorization)
    return await _resume_browser_task(services, session_id, payload, "app")


@router.post(
    "/v1/custom/browser/tasks/{session_id}/resume",
    operation_id="resumeCustomBrowserTask",
)
async def resume_custom_browser_task(
    session_id: str,
    payload: BrowserTaskResumeRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize_external_gateway(services, authorization)
    return await _resume_browser_task(
        services, session_id, payload, "custom_gpt"
    )


async def _browser_takeover_status(
    services: RuntimeServices, session_id: str, source: str
) -> dict[str, object]:
    try:
        return await services.digital_identity_gateway.status(
            session_id, source  # type: ignore[arg-type]
        )
    except BrowserClientError as exc:
        raise _gateway_error(exc) from exc


@router.get("/v1/browser/takeovers/{session_id}", operation_id="getAppTakeoverStatus")
async def app_takeover_status(
    session_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize(services, authorization)
    return await _browser_takeover_status(services, session_id, "app")


@router.get(
    "/v1/custom/browser/takeovers/{session_id}",
    operation_id="getCustomTakeoverStatus",
)
async def custom_takeover_status(
    session_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize_external_gateway(services, authorization)
    return await _browser_takeover_status(services, session_id, "custom_gpt")


async def _revoke_browser_takeover(
    services: RuntimeServices, session_id: str, source: str
) -> dict[str, object]:
    try:
        return await services.digital_identity_gateway.revoke(
            session_id, source  # type: ignore[arg-type]
        )
    except BrowserClientError as exc:
        raise _gateway_error(exc) from exc


@router.delete(
    "/v1/browser/takeovers/{session_id}", operation_id="revokeAppTakeover"
)
async def revoke_app_takeover(
    session_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize(services, authorization)
    return await _revoke_browser_takeover(services, session_id, "app")


@router.delete(
    "/v1/custom/browser/takeovers/{session_id}",
    operation_id="revokeCustomTakeover",
)
async def revoke_custom_takeover(
    session_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    services = _services(request)
    _authorize_external_gateway(services, authorization)
    return await _revoke_browser_takeover(
        services, session_id, "custom_gpt"
    )
