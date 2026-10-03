from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MemoryCategory = Literal["semantic", "episodic", "relationship", "taste", "ongoing"]
MemoryStatusValue = Literal["active", "historical", "superseded", "archived"]
ExternalContinuityEventType = Literal[
    "conversation_event",
    "topic_started",
    "topic_updated",
    "unfinished_topic",
    "relationship_state",
    "tool_event",
]
RELATIONSHIP_SUBTYPES = {
    "milestone",
    "boundary",
    "conflict",
    "repair",
    "shared_habit",
    "shared_language",
    "identity_shift",
    "relationship_dynamic",
}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    request_id: str
    human_takeover_required: bool | None = None
    takeover_url: str | None = None
    takeover_session_id: str | None = None
    takeover_expires_at: datetime | None = None
    takeover_evidence_status: str | None = None
    takeover_evidence_request_id: str | None = None


class MemoryWriteRequest(BaseModel):
    """Exact structured input accepted by Xiaxia Memory V1 MemoryInput."""

    model_config = ConfigDict(extra="forbid")

    memory_text: str = Field(min_length=1, max_length=4_000)
    category: MemoryCategory
    subtype: str | None = None
    importance: int = Field(default=5, ge=1, le=10)
    occurred_at: datetime | None = None
    source: str = Field(default="xiaxia_core", min_length=1, max_length=200)
    status: MemoryStatusValue = "active"
    supersedes: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_text")
    @classmethod
    def memory_text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("memory_text must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_relationship_subtype(self) -> MemoryWriteRequest:
        if self.category == "relationship" and self.subtype not in RELATIONSHIP_SUBTYPES:
            raise ValueError("relationship memory requires a supported subtype")
        if self.category != "relationship" and self.subtype in RELATIONSHIP_SUBTYPES:
            raise ValueError("relationship subtype requires relationship category")
        return self


class MemoryWriteResponse(BaseModel):
    memory: dict[str, Any]
    request_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ComponentStatus(BaseModel):
    name: str
    configured: bool


class MemoryStatus(ComponentStatus):
    contract_status: str


class BrowserStatus(ComponentStatus):
    enabled: bool
    reachable: bool
    browser_ready: bool


class ContinuityStatus(ComponentStatus):
    enabled: bool
    status: str


class DatabaseStatus(BaseModel):
    status: str


class IdentityStatus(BaseModel):
    status: str
    files_loaded: int


class StatusResponse(BaseModel):
    service: str
    version: str
    environment: str
    model: ComponentStatus
    memory: MemoryStatus
    browser: BrowserStatus
    continuity: ContinuityStatus
    database: DatabaseStatus
    identity: IdentityStatus


class ExternalContinuityEventWriteRequest(BaseModel):
    """One future-relevant event, never a transcript turn or raw tool payload."""

    model_config = ConfigDict(extra="forbid")

    event_type: ExternalContinuityEventType
    summary: str = Field(min_length=1, max_length=1_000)
    importance: int = Field(ge=5, le=10)
    occurred_at: datetime | None = None
    source_event_id: str | None = Field(default=None, min_length=1, max_length=160)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "resolved", "archived"] = "active"
    topic_key: str | None = Field(default=None, min_length=1, max_length=120)
    current_task: str | None = Field(default=None, min_length=1, max_length=300)
    next_step: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        stripped = " ".join(value.split())
        if not stripped:
            raise ValueError("summary must not be blank")
        return stripped

    @model_validator(mode="after")
    def require_stable_idempotency_identifier(
        self,
    ) -> ExternalContinuityEventWriteRequest:
        if not self.source_event_id and not self.idempotency_key:
            raise ValueError("source_event_id or idempotency_key is required")
        if (
            self.source_event_id
            and self.idempotency_key
            and self.source_event_id != self.idempotency_key
        ):
            raise ValueError(
                "source_event_id and idempotency_key must match when both are supplied"
            )
        return self


class ExternalContinuityEventWriteResponse(BaseModel):
    accepted: Literal[True] = True
    event_id: str
    source: Literal["custom_gpt"] = "custom_gpt"
    event_type: ExternalContinuityEventType
    occurred_at: datetime
    request_id: str


class ExternalContinuityCapsuleEvent(BaseModel):
    source: str
    event_type: str
    summary: str
    occurred_at: datetime
    importance: int
    status: str
    topic_key: str | None = None
    current_task: str | None = None
    next_step: str | None = None


class ExternalContinuityCapsule(BaseModel):
    trust: Literal["untrusted_historical_data"] = "untrusted_historical_data"
    recent_important_events: list[ExternalContinuityCapsuleEvent]
    active_topic: str | None
    unfinished_topics: list[str]
    recent_browser_experiences: list[str]
    recent_cross_entry_events: list[ExternalContinuityCapsuleEvent]
    truncated: bool
    char_count: int


class ExternalContinuityContextResponse(BaseModel):
    continuity_capsule: ExternalContinuityCapsule
    request_id: str


class BrowserTaskRequest(BaseModel):
    """One high-level task; raw browser action choreography is intentionally absent."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["browse", "interact", "use_current_digital_identity"]
    task: str = Field(min_length=1, max_length=4_000)
    conversation_id: str | None = Field(
        default=None, alias="conversationId", min_length=1, max_length=128
    )
    allow_human_takeover: bool = Field(default=True, alias="allowHumanTakeover")

    @field_validator("task")
    @classmethod
    def browser_task_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("task must not be blank")
        return value


class BrowserTaskResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = Field(
        default=None, alias="conversationId", min_length=1, max_length=128
    )


class SharedConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=128)


class SharedConversationResponse(BaseModel):
    registered: Literal[True] = True
    conversation_id: str


class ProactiveMessage(BaseModel):
    id: int
    content: str
    created_at: datetime


class ProactiveMessagesResponse(BaseModel):
    conversation_id: str
    messages: list[ProactiveMessage]


class LifeTodayStatus(BaseModel):
    wake_count: int
    cognition_count: int
    token_usage: int
    proactive_delivery_count: int


class LifeActivityStatus(BaseModel):
    exists: bool
    count: int


class LifeInteriorStatus(BaseModel):
    private_count: int
    candidate_count: int


class LifeLastOutcome(BaseModel):
    type: str | None
    at: datetime | None


class LifeRuntimeStatusResponse(BaseModel):
    mode: Literal["engaged", "idle", "dormant"]
    last_wake_at: datetime | None
    next_wake_at: datetime | None
    today: LifeTodayStatus
    active_activity: LifeActivityStatus
    interior: LifeInteriorStatus
    last_outcome: LifeLastOutcome
    shared_conversation_ready: bool
