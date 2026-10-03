from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.continuity.base import (
    ContinuityCapsule,
    ContinuityError,
    ContinuityEvent,
    ContinuityEventInput,
    ContinuityStore,
    build_bounded_capsule,
)


class PostgresContinuityStore(ContinuityStore):
    """Canonical continuity persistence; it owns no Memory or model behavior."""

    name = "postgresql"

    def __init__(
        self, database_url: str, *, connect_timeout_seconds: float = 5.0
    ) -> None:
        if not database_url:
            raise ValueError("continuity_database_url_required")
        self._database_url = database_url
        self._connect_timeout_seconds = max(1, int(connect_timeout_seconds))

    @property
    def configured(self) -> bool:
        return True

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self._database_url,
            connect_timeout=self._connect_timeout_seconds,
            row_factory=dict_row,
            prepare_threshold=None,
        )

    @staticmethod
    def _event(row: dict[str, Any]) -> ContinuityEvent:
        return ContinuityEvent(
            event_id=str(row["event_id"]),
            xiaxia_identity_id=row["xiaxia_identity_id"],
            user_identity_id=row["user_identity_id"],
            source=row["source"],
            event_type=row["event_type"],
            summary=row["summary"],
            occurred_at=row["occurred_at"],
            created_at=row["created_at"],
            importance=row["importance"],
            status=row["status"],
            metadata=row["metadata"],
            request_id=row["request_id"],
            conversation_id=row["conversation_id"],
            source_event_id=row["source_event_id"],
        )

    def create_event(self, event: ContinuityEventInput) -> ContinuityEvent:
        normalized = event.normalized()
        event_id = uuid.uuid4()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO continuity_events (
                        event_id, xiaxia_identity_id, user_identity_id, source,
                        event_type, summary, occurred_at, importance, status,
                        metadata, request_id, conversation_id, source_event_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (
                        xiaxia_identity_id, user_identity_id, source, source_event_id
                    ) WHERE source_event_id IS NOT NULL
                    DO NOTHING
                    RETURNING *
                    """,
                    (
                        event_id,
                        normalized.xiaxia_identity_id,
                        normalized.user_identity_id,
                        normalized.source,
                        normalized.event_type,
                        normalized.summary,
                        normalized.occurred_at,
                        normalized.importance,
                        normalized.status,
                        Jsonb(normalized.metadata),
                        normalized.request_id,
                        normalized.conversation_id,
                        normalized.source_event_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None and normalized.source_event_id:
                    cursor.execute(
                        """
                        SELECT * FROM continuity_events
                        WHERE xiaxia_identity_id = %s
                          AND user_identity_id = %s
                          AND source = %s
                          AND source_event_id = %s
                        """,
                        (
                            normalized.xiaxia_identity_id,
                            normalized.user_identity_id,
                            normalized.source,
                            normalized.source_event_id,
                        ),
                    )
                    row = cursor.fetchone()
                if row is None:
                    raise ContinuityError("continuity_event_write_failed")
                return self._event(row)
        except (psycopg.Error, ValueError, TypeError) as exc:
            raise ContinuityError("continuity_event_write_failed") from exc

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
        if limit <= 0:
            return []
        clauses = ["xiaxia_identity_id = %s", "user_identity_id = %s"]
        params: list[Any] = [xiaxia_identity_id, user_identity_id]
        if source:
            clauses.append("source = %s")
            params.append(source)
        if event_type:
            clauses.append("event_type = %s")
            params.append(event_type)
        params.append(min(limit, 100))
        direction = "DESC" if newest_first else "ASC"
        query = f"""
            SELECT * FROM continuity_events
            WHERE {" AND ".join(clauses)}
            ORDER BY occurred_at {direction}, created_at {direction}, event_id {direction}
            LIMIT %s
        """
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(query, params)
                return [self._event(row) for row in cursor.fetchall()]
        except psycopg.Error as exc:
            raise ContinuityError("continuity_timeline_read_failed") from exc

    def load_capsule(
        self,
        *,
        xiaxia_identity_id: str,
        user_identity_id: str,
        max_events: int,
        max_chars: int,
        recent_days: int,
    ) -> ContinuityCapsule:
        if max_events <= 0 or max_chars <= 0:
            return ContinuityCapsule()
        cutoff = datetime.now(UTC) - timedelta(days=max(1, recent_days))
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM continuity_events
                    WHERE xiaxia_identity_id = %s
                      AND user_identity_id = %s
                      AND occurred_at >= %s
                      AND status IN ('active', 'resolved')
                      AND (
                        importance >= 5
                        OR event_type IN ('unfinished_topic', 'browser_event')
                      )
                    ORDER BY occurred_at DESC, created_at DESC, event_id DESC
                    LIMIT %s
                    """,
                    (
                        xiaxia_identity_id,
                        user_identity_id,
                        cutoff,
                        min(100, max_events * 4),
                    ),
                )
                available = [self._event(row) for row in cursor.fetchall()]
                cursor.execute(
                    """
                    SELECT DISTINCT ON (metadata->>'topic_key') *
                    FROM continuity_events
                    WHERE xiaxia_identity_id = %s
                      AND user_identity_id = %s
                      AND occurred_at >= %s
                      AND metadata ? 'topic_key'
                      AND metadata->>'topic_key' <> ''
                    ORDER BY metadata->>'topic_key', occurred_at DESC,
                             created_at DESC, event_id DESC
                    """,
                    (xiaxia_identity_id, user_identity_id, cutoff),
                )
                available.extend(self._event(row) for row in cursor.fetchall())
        except psycopg.Error as exc:
            raise ContinuityError("continuity_capsule_read_failed") from exc

        newest = sorted(
            {event.event_id: event for event in available}.values(),
            key=lambda event: (event.occurred_at, event.created_at, event.event_id),
            reverse=True,
        )
        return build_bounded_capsule(
            newest,
            max_events=max_events,
            max_chars=max_chars,
        )

    def healthy(self) -> bool:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT to_regclass('public.continuity_events') IS NOT NULL"
                )
                row = cursor.fetchone()
                return bool(row and next(iter(row.values())))
        except psycopg.Error:
            return False
