from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.continuity.base import (
    ContinuityCapsule,
    ContinuityError,
    ContinuityEvent,
    ContinuityEventInput,
    ContinuityStore,
    build_bounded_capsule,
)


class FakeContinuityStore(ContinuityStore):
    name = "fake-continuity"

    def __init__(
        self,
        events: list[ContinuityEvent] | None = None,
        *,
        fail_reads: bool = False,
        fail_writes: bool = False,
    ) -> None:
        self.events = events if events is not None else []
        self.fail_reads = fail_reads
        self.fail_writes = fail_writes

    @property
    def configured(self) -> bool:
        return True

    def create_event(self, event: ContinuityEventInput) -> ContinuityEvent:
        if self.fail_writes:
            raise ContinuityError("fake_continuity_write_failed")
        normalized = event.normalized()
        if normalized.source_event_id:
            for existing in self.events:
                if (
                    existing.xiaxia_identity_id == normalized.xiaxia_identity_id
                    and existing.user_identity_id == normalized.user_identity_id
                    and existing.source == normalized.source
                    and existing.source_event_id == normalized.source_event_id
                ):
                    return existing
        now = datetime.now(UTC)
        created = ContinuityEvent(
            event_id=str(uuid.uuid4()),
            xiaxia_identity_id=normalized.xiaxia_identity_id,
            user_identity_id=normalized.user_identity_id,
            source=normalized.source,
            event_type=normalized.event_type,
            summary=normalized.summary,
            occurred_at=normalized.occurred_at,
            created_at=now,
            importance=normalized.importance,
            status=normalized.status,
            metadata=normalized.metadata,
            request_id=normalized.request_id,
            conversation_id=normalized.conversation_id,
            source_event_id=normalized.source_event_id,
        )
        self.events.append(created)
        return created

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
        if self.fail_reads:
            raise ContinuityError("fake_continuity_read_failed")
        matching = [
            event
            for event in self.events
            if event.xiaxia_identity_id == xiaxia_identity_id
            and event.user_identity_id == user_identity_id
            and (source is None or event.source == source)
            and (event_type is None or event.event_type == event_type)
        ]
        matching.sort(
            key=lambda event: (event.occurred_at, event.created_at, event.event_id),
            reverse=newest_first,
        )
        return matching[: max(0, limit)]

    def load_capsule(
        self,
        *,
        xiaxia_identity_id: str,
        user_identity_id: str,
        max_events: int,
        max_chars: int,
        recent_days: int,
    ) -> ContinuityCapsule:
        if self.fail_reads:
            raise ContinuityError("fake_continuity_read_failed")
        cutoff = datetime.now(UTC) - timedelta(days=max(1, recent_days))
        newest = [
            event
            for event in self.list_events(
                xiaxia_identity_id=xiaxia_identity_id,
                user_identity_id=user_identity_id,
                limit=max(1, max_events * 4),
            )
            if event.occurred_at >= cutoff
            and event.status in {"active", "resolved"}
            and (
                event.importance >= 5
                or event.event_type in {"unfinished_topic", "browser_event"}
            )
        ]
        return build_bounded_capsule(
            newest,
            max_events=max_events,
            max_chars=max_chars,
        )

    def healthy(self) -> bool:
        return not self.fail_reads and not self.fail_writes
