"""Narrow Hermes tools that persist explicit conversation commitments in Life Runtime."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.life_runtime.store import LifeStore
from tools.registry import registry


def _parse_due_at(value: object, now: datetime) -> datetime:
    if not isinstance(value, str):
        raise ValueError("due_at must be an ISO 8601 string with timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("due_at must include an explicit timezone")
    due = parsed.astimezone(UTC)
    if due <= now:
        raise ValueError("due_at must be in the future")
    if due > now + timedelta(days=366):
        raise ValueError("due_at is outside the supported one-year range")
    return due


def _resolve_schedule_due(args: dict, now: datetime) -> datetime:
    has_due_at = "due_at" in args and args["due_at"] is not None
    has_delay = "delay_seconds" in args and args["delay_seconds"] is not None
    if has_due_at == has_delay:
        raise ValueError("provide exactly one of due_at or delay_seconds")
    if has_due_at:
        return _parse_due_at(args["due_at"], now)

    delay = args["delay_seconds"]
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise ValueError("delay_seconds must be an integer")
    if delay <= 0:
        raise ValueError("delay_seconds must be greater than zero")
    if delay > int(timedelta(days=366).total_seconds()):
        raise ValueError("delay_seconds is outside the supported one-year range")
    return now + timedelta(seconds=delay)


def register_life_tools(database_path: Path) -> None:
    store = LifeStore(database_path)

    def _failure(operation: str, exc: Exception) -> str:
        return json.dumps({
            "success": False,
            "operation": operation,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "receipt": "not_persisted",
        }, ensure_ascii=False)

    def schedule(args: dict, **kwargs) -> str:
        try:
            now = datetime.now(UTC)
            summary = str(args.get("summary") or "").strip()
            if not summary or len(summary) > 500:
                raise ValueError("summary must contain 1-500 characters")
            wake_reason = str(args.get("wake_reason") or "").strip()
            if len(wake_reason) > 1000:
                raise ValueError("wake_reason must contain at most 1000 characters")
            kind = str(args.get("kind") or "reminder").strip()
            if kind not in {"reminder", "commitment"}:
                raise ValueError("kind must be reminder or commitment")
            due = _resolve_schedule_due(args, now)
            state_payload = {"source": "conversation"}
            if wake_reason:
                state_payload["wake_reason"] = wake_reason
            activity = store.add_activity(
                kind=kind,
                summary=summary,
                now=now,
                status="waiting",
                state_payload=state_payload,
                next_wake_at=due,
            )
            store.ledger(
                "ACTIVITY_TRANSITION", "conversation_commitment_created", now=now,
                activity_id=activity.activity_id,
                details={"source": "conversation", "due_at": due.isoformat()},
            )
            receipt = {
                "success": True,
                "receipt": "persisted",
                "activity_id": activity.activity_id,
                "summary": activity.summary,
                "kind": activity.kind,
                "due_at": due.isoformat(),
            }
            if wake_reason:
                receipt["wake_reason"] = wake_reason
            return json.dumps(receipt, ensure_ascii=False)
        except (KeyError, ValueError, OSError) as exc:
            return _failure("schedule_life_activity", exc)

    def cancel(args: dict, **kwargs) -> str:
        try:
            now = datetime.now(UTC)
            activity_id = str(args.get("activity_id") or "").strip()
            activity = store.transition_activity(activity_id, "abandoned", now=now)
            store.ledger(
                "ACTIVITY_TRANSITION", "conversation_commitment_cancelled", now=now,
                activity_id=activity.activity_id,
            )
            return json.dumps({
                "success": True, "receipt": "persisted",
                "activity_id": activity.activity_id, "status": activity.status,
            })
        except (KeyError, ValueError, OSError) as exc:
            return _failure("cancel_life_activity", exc)

    def reschedule(args: dict, **kwargs) -> str:
        try:
            now = datetime.now(UTC)
            activity_id = str(args.get("activity_id") or "").strip()
            due = _parse_due_at(args.get("due_at"), now)
            current = store.get_activity(activity_id)
            if current.status not in {"active", "waiting"}:
                raise ValueError("only active or waiting activities can be rescheduled")
            activity = store.transition_activity(
                activity_id, current.status, now=now, next_wake_at=due
            )
            store.ledger(
                "ACTIVITY_TRANSITION", "conversation_commitment_rescheduled", now=now,
                activity_id=activity.activity_id, details={"due_at": due.isoformat()},
            )
            return json.dumps({
                "success": True, "receipt": "persisted",
                "activity_id": activity.activity_id,
                "summary": activity.summary, "due_at": due.isoformat(),
            }, ensure_ascii=False)
        except (KeyError, ValueError, OSError) as exc:
            return _failure("reschedule_life_activity", exc)

    registry.register(
        name="schedule_life_activity", toolset="xiaxia_life",
        description="Persist an explicit future conversation commitment for later Hermes cognition.",
        schema={
            "description": (
                "Schedule an explicit future reminder or commitment. summary is a concise status "
                "label. wake_reason is optional historical context for the due-time cognition. "
                "For relative durations such as "
                "'five minutes from now', provide delay_seconds and do not calculate an absolute "
                "timestamp. Use due_at only for an explicit absolute time. Provide exactly one of "
                "due_at or delay_seconds. Acknowledge scheduling only when success is true and "
                "receipt is persisted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise activity label for status displays; never the future message body.",
                    },
                    "wake_reason": {
                        "type": "string",
                        "description": "Optional historical context explaining why the activity should wake.",
                    },
                    "due_at": {
                        "type": "string",
                        "description": "Explicit absolute ISO 8601 timestamp with timezone.",
                    },
                    "delay_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 31622400,
                        "description": "Relative delay in whole seconds, calculated from server time.",
                    },
                    "kind": {"type": "string", "enum": ["reminder", "commitment"]},
                },
                "required": ["summary"],
            },
        },
        handler=schedule,
    )
    registry.register(
        name="cancel_life_activity", toolset="xiaxia_life",
        description="Cancel a previously scheduled Life activity by its receipt ID.",
        schema={
            "description": "Cancel a scheduled activity when the user explicitly withdraws it.",
            "parameters": {
                "type": "object",
                "properties": {"activity_id": {"type": "string"}},
                "required": ["activity_id"],
            },
        },
        handler=cancel,
    )
    registry.register(
        name="reschedule_life_activity", toolset="xiaxia_life",
        description="Move a scheduled Life activity to a new absolute time.",
        schema={
            "description": "Reschedule an existing activity after an explicit user request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {"type": "string"},
                    "due_at": {"type": "string"},
                },
                "required": ["activity_id", "due_at"],
            },
        },
        handler=reschedule,
    )
