from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

ContinuitySource = Literal[
    "app",
    "custom_gpt",
    "browser",
    "core",
    "reading_house",
    "diary_house",
    "watch_house",
    "proactive_runtime",
]
ContinuityEventType = Literal[
    "conversation_event",
    "browser_event",
    "tool_event",
    "topic_started",
    "topic_updated",
    "unfinished_topic",
    "relationship_state",
    "memory_candidate",
    "house_event",
    "system_event",
]

ALLOWED_SOURCES = {
    "app",
    "custom_gpt",
    "browser",
    "core",
    "reading_house",
    "diary_house",
    "watch_house",
    "proactive_runtime",
}
ALLOWED_EVENT_TYPES = {
    "conversation_event",
    "browser_event",
    "tool_event",
    "topic_started",
    "topic_updated",
    "unfinished_topic",
    "relationship_state",
    "memory_candidate",
    "house_event",
    "system_event",
}
_SECRET_KEY = re.compile(
    r"(?:authorization|bearer|cookie|local.?storage|password|passwd|passcode|"
    r"credential|api.?key|access.?token|refresh.?token|id.?token|secret|otp)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/=-]+|"
    r"(?:authorization(?:\s+header)?|cookie|password|passwd|passcode|token|"
    r"secret|otp|api[_ -]?key|supabase(?:\s+credential)?)"
    r"\s*(?::|=|\bis\b)\s*\S+)"
)


class ContinuityError(RuntimeError):
    """A sanitized continuity failure safe for Core degradation handling."""


class ContinuityValidationError(ContinuityError):
    pass


def contains_sensitive_text(value: str) -> bool:
    return bool(_SECRET_VALUE.search(value))


def safe_bounded_summary(value: str, *, limit: int = 500) -> str:
    normalized = " ".join(value.split())
    if contains_sensitive_text(normalized):
        return "本次经历涉及敏感输入，具体内容未写入连续性时间线。"
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 1)].rstrip() + "…"


def _normalize_json(value: Any, *, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        raise ContinuityValidationError("continuity_metadata_too_deep")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, UUID, Path)):
        return str(value)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _SECRET_KEY.search(key):
                continue
            normalized[key] = _normalize_json(
                raw_value, depth=depth + 1, max_depth=max_depth
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, depth=depth + 1, max_depth=max_depth)
            for item in value
        ]
    raise ContinuityValidationError("continuity_metadata_not_json_serializable")


def normalize_metadata(
    metadata: dict[str, Any], *, max_chars: int = 8_192, max_depth: int = 6
) -> dict[str, Any]:
    normalized = _normalize_json(metadata, max_depth=max_depth)
    assert isinstance(normalized, dict)
    serialized = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > max_chars:
        raise ContinuityValidationError("continuity_metadata_too_large")
    if contains_sensitive_text(serialized):
        raise ContinuityValidationError("continuity_metadata_contains_sensitive_value")
    return normalized


@dataclass(frozen=True, slots=True)
class ContinuityEventInput:
    xiaxia_identity_id: str
    user_identity_id: str
    source: ContinuitySource
    event_type: ContinuityEventType
    summary: str
    importance: int = 5
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    request_id: str | None = None
    conversation_id: str | None = None
    source_event_id: str | None = None

    def normalized(self) -> ContinuityEventInput:
        if self.source not in ALLOWED_SOURCES:
            raise ContinuityValidationError("invalid_continuity_source")
        if self.event_type not in ALLOWED_EVENT_TYPES:
            raise ContinuityValidationError("invalid_continuity_event_type")
        if not 1 <= self.importance <= 10:
            raise ContinuityValidationError("invalid_continuity_importance")
        if self.status not in {"active", "resolved", "archived"}:
            raise ContinuityValidationError("invalid_continuity_status")
        summary = safe_bounded_summary(self.summary, limit=1_000)
        if not summary:
            raise ContinuityValidationError("empty_continuity_summary")
        occurred_at = self.occurred_at
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        return ContinuityEventInput(
            xiaxia_identity_id=safe_bounded_summary(self.xiaxia_identity_id, limit=128),
            user_identity_id=safe_bounded_summary(self.user_identity_id, limit=128),
            source=self.source,
            event_type=self.event_type,
            summary=summary,
            importance=self.importance,
            status=self.status,
            metadata=normalize_metadata(self.metadata),
            occurred_at=occurred_at.astimezone(UTC),
            request_id=safe_bounded_summary(self.request_id, limit=128)
            if self.request_id
            else None,
            conversation_id=(
                safe_bounded_summary(self.conversation_id, limit=128)
                if self.conversation_id
                else None
            ),
            source_event_id=(
                safe_bounded_summary(self.source_event_id, limit=200)
                if self.source_event_id
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuityEvent:
    event_id: str
    xiaxia_identity_id: str
    user_identity_id: str
    source: str
    event_type: str
    summary: str
    occurred_at: datetime
    created_at: datetime
    importance: int
    status: str
    metadata: dict[str, Any]
    request_id: str | None = None
    conversation_id: str | None = None
    source_event_id: str | None = None

    def capsule_item(self) -> dict[str, Any]:
        item = {
            "event_id": self.event_id,
            "source": self.source,
            "event_type": self.event_type,
            "summary": self.summary,
            "occurred_at": self.occurred_at.isoformat(),
            "importance": self.importance,
            "status": self.status,
        }
        for key in ("topic_key", "current_task", "next_step"):
            if isinstance(self.metadata.get(key), str):
                item[key] = self.metadata[key]
        return item


@dataclass(frozen=True, slots=True)
class ContinuityCapsule:
    events: tuple[ContinuityEvent, ...] = ()
    active_topic: str | None = None
    unfinished_topics: tuple[str, ...] = ()
    recent_browser_events: tuple[str, ...] = ()
    char_count: int = 0
    truncated: bool = False

    def as_context(self) -> dict[str, Any]:
        return {
            "trust": "untrusted_historical_data",
            "events": [event.capsule_item() for event in self.events],
            "active_topic": self.active_topic,
            "unfinished_topics": list(self.unfinished_topics),
            "recent_browser_events": list(self.recent_browser_events),
            "truncated": self.truncated,
        }


def build_bounded_capsule(
    available_newest_first: list[ContinuityEvent],
    *,
    max_events: int,
    max_chars: int,
) -> ContinuityCapsule:
    latest_topics: dict[str, ContinuityEvent] = {}
    for event in available_newest_first:
        topic_key = event.metadata.get("topic_key")
        if isinstance(topic_key, str) and topic_key and topic_key not in latest_topics:
            latest_topics[topic_key] = event
    topic_events = sorted(
        (event for event in latest_topics.values() if event.status != "archived"),
        key=lambda event: (event.occurred_at, event.created_at, event.event_id),
        reverse=True,
    )
    selected_candidates = topic_events + [
        event
        for event in available_newest_first
        if not isinstance(event.metadata.get("topic_key"), str)
        or not event.metadata["topic_key"]
    ]

    def assemble(
        selected_newest_first: list[ContinuityEvent], is_truncated: bool
    ) -> ContinuityCapsule:
        chronological = tuple(reversed(selected_newest_first))
        active_topic = next(
            (
                event.summary
                for event in selected_newest_first
                if event.status == "active"
                and event.event_type
                in {"topic_started", "topic_updated", "unfinished_topic"}
            ),
            None,
        )
        if active_topic is None and not latest_topics:
            active_topic = next(
                (
                    event.summary
                    for event in selected_newest_first
                    if event.event_type == "conversation_event"
                ),
                None,
            )
        unfinished = tuple(
            event.summary
            for event in selected_newest_first
            if event.status == "active"
            and (
                event.event_type == "unfinished_topic"
                or (
                    isinstance(event.metadata.get("topic_key"), str)
                    and event.event_type in {"topic_started", "topic_updated"}
                )
            )
        )[:3]
        browser = tuple(
            event.summary
            for event in selected_newest_first
            if event.event_type == "browser_event"
        )[:3]
        capsule = ContinuityCapsule(
            events=chronological,
            active_topic=active_topic,
            unfinished_topics=unfinished,
            recent_browser_events=browser,
            truncated=is_truncated,
        )
        size = len(
            json.dumps(capsule.as_context(), ensure_ascii=False, separators=(",", ":"))
        )
        return ContinuityCapsule(
            events=capsule.events,
            active_topic=capsule.active_topic,
            unfinished_topics=capsule.unfinished_topics,
            recent_browser_events=capsule.recent_browser_events,
            char_count=size,
            truncated=is_truncated,
        )

    selected: list[ContinuityEvent] = []
    truncated = len(selected_candidates) > max_events
    for event in selected_candidates:
        if len(selected) >= max_events:
            truncated = True
            break
        candidate = assemble(selected + [event], truncated)
        if candidate.char_count > max_chars:
            truncated = True
            continue
        selected.append(event)
    return assemble(selected, truncated)


class ContinuityStore(ABC):
    name: str

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_event(self, event: ContinuityEventInput) -> ContinuityEvent:
        raise NotImplementedError

    @abstractmethod
    def list_events(
        self,
        *,
        xiaxia_identity_id: str,
        user_identity_id: str,
        limit: int,
        source: str | None = None,
        event_type: str | None = None,
        newest_first: bool = True,
    ) -> list[ContinuityEvent]:
        raise NotImplementedError

    @abstractmethod
    def load_capsule(
        self,
        *,
        xiaxia_identity_id: str,
        user_identity_id: str,
        max_events: int,
        max_chars: int,
        recent_days: int,
    ) -> ContinuityCapsule:
        raise NotImplementedError

    @abstractmethod
    def healthy(self) -> bool:
        raise NotImplementedError


class DisabledContinuityStore(ContinuityStore):
    name = "disabled"

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled

    @property
    def configured(self) -> bool:
        return False

    def create_event(self, event: ContinuityEventInput) -> ContinuityEvent:
        raise ContinuityError("continuity_not_configured")

    def list_events(self, **kwargs: Any) -> list[ContinuityEvent]:
        return []

    def load_capsule(self, **kwargs: Any) -> ContinuityCapsule:
        return ContinuityCapsule()

    def healthy(self) -> bool:
        return not self.enabled
