from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class VerificationState(StrEnum):
    NOT_EXECUTED = "NOT_EXECUTED"
    UNVERIFIED = "UNVERIFIED"
    APPLIED_VERIFIED = "APPLIED_VERIFIED"
    APPLIED_UNVERIFIED = "APPLIED_UNVERIFIED"
    TARGET_MISSING = "TARGET_MISSING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CAPTURED_NOT_INTERPRETED = "CAPTURED_NOT_INTERPRETED"


class TaskStatus(StrEnum):
    NOT_EXECUTED = "not_executed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class ChatStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionRequestClosedError(RuntimeError):
    """Raised when code tries to attach a Browser call to a finalized request."""


@dataclass(frozen=True, slots=True)
class RequiredSubgoalInput:
    normalized_action: str
    original_required_subgoal: dict[str, object]
    required: bool = True
    target: dict[str, object] | None = None
    destination: str | None = None
    polarity: str = "positive"
    condition: dict[str, object] | None = None
    dependency_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRequestSnapshot:
    request_id: str
    conversation_id: str
    task_kind: str
    task_status: str
    chat_status: str
    started_at: str
    completed_at: str | None
    subgoals: tuple[dict[str, object], ...]
    attempts: tuple[dict[str, object], ...]

    @property
    def final_attempts(self) -> tuple[dict[str, object], ...]:
        latest: dict[str, dict[str, object]] = {}
        for item in self.attempts:
            action_id = str(item.get("action_id"))
            previous = latest.get(action_id)
            if previous is None or (
                int(item.get("attempt_number") or 0),
                int(item.get("call_sequence") or 0),
            ) > (
                int(previous.get("attempt_number") or 0),
                int(previous.get("call_sequence") or 0),
            ):
                latest[action_id] = item
        return tuple(
            sorted(
                latest.values(), key=lambda item: int(item.get("call_sequence") or 0)
            )
        )

    @property
    def verified_actions(self) -> frozenset[str]:
        return frozenset(
            str(item["normalized_action"])
            for item in self.final_attempts
            if item.get("normalized_action") != "status"
            and item.get("verification_state")
            in {
                VerificationState.APPLIED_VERIFIED.value,
                VerificationState.CAPTURED_NOT_INTERPRETED.value,
            }
        )


_SENSITIVE_KEY = re.compile(
    r"authorization|cookie|token|secret|password|passwd|otp|one.?time|"
    r"verification.?code|captcha|credential|takeover.?url|session.?id|exchange|"
    r"验证码|密码|口令|短信码",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = re.compile(
    r"(?:password|passwd|otp|one.?time.?code|verification.?code|captcha|"
    r"密码|验证码|口令|短信码)\s*(?:is|是|为|[:：])\s*\S+",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?:authorization\s*:\s*)?bearer\s+\S+|"
    r"(?:set-cookie\s*:|cookie\s*[:=])\s*\S+|"
    r"(?:access[_-]?token|token|secret|password|passwd|otp|one.?time.?code|"
    r"verification.?code|captcha|session[_-]?id)\s*(?:is|=|:|：)\s*\S+|"
    r"(?:密码|验证码|口令|短信码)\s*(?:是|为|=|:|：)\s*\S+",
    re.IGNORECASE,
)
_ALWAYS_REDACT_VALUE_KEYS = {"text", "value", "typedvalue", "inputvalue"}
_CONTENT_KEYS = {
    "content",
    "bodytext",
    "html",
    "base64",
    "visibletext",
    "nearbytext",
    "innertext",
    "outertext",
    "textcontent",
    "pagetext",
    "dom",
}
_URL_KEY_PARTS = ("url", "href", "uri", "destination", "location")
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_.:-]{1,160}$", re.IGNORECASE)


def _key_name(key: str | None) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (key or "").casefold())


def _looks_like_url(value: str, key: str | None) -> bool:
    normalized_key = _key_name(key)
    return value.startswith(("http://", "https://")) or any(
        part in normalized_key for part in _URL_KEY_PARTS
    )


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[REDACTED_URL]"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "[REDACTED_URL]"
    try:
        port = parsed.port
    except ValueError:
        return "[REDACTED_URL]"
    hostname = parsed.hostname.casefold()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None and not (
        (parsed.scheme == "http" and port == 80)
        or (parsed.scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"
    query = "[REDACTED_QUERY]" if parsed.query else ""
    return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path or "/", query, ""))


def _safe_error(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not _SAFE_ERROR_CODE.fullmatch(text) or _SENSITIVE_KEY.search(text):
        return "redacted_error"
    return text


def _safe_value(value: object, *, key: str | None = None, depth: int = 0) -> object:
    if depth > 8:
        return "[TRUNCATED_DEPTH]"
    normalized_key = _key_name(key)
    if key and (
        _SENSITIVE_KEY.search(key) or normalized_key in _ALWAYS_REDACT_VALUE_KEYS
    ):
        return "[REDACTED]"
    if normalized_key in _CONTENT_KEYS:
        length = len(value) if isinstance(value, (str, bytes, list, dict)) else None
        return f"[OMITTED_BROWSER_CONTENT length={length}]"
    if isinstance(value, str):
        if _looks_like_url(value, key):
            return _safe_url(value)
        if _SENSITIVE_TEXT.search(value) or _SECRET_ASSIGNMENT.search(value):
            return "[REDACTED_SENSITIVE_TEXT]"
        return value[:4_000]
    if isinstance(value, dict):
        return {
            str(item_key)[:200]: _safe_value(
                item_value, key=str(item_key), depth=depth + 1
            )
            for item_key, item_value in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth=depth + 1) for item in value[:200]]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)[:1_000]


def _json(value: object) -> str:
    return json.dumps(
        _safe_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _safe_string(value: object, *, key: str) -> str | None:
    safe = _safe_value(value, key=key)
    return safe if isinstance(safe, str) else None


def _parsed(value: str | None) -> object:
    if not value:
        return None
    return json.loads(value)


def _has_strings(data: dict[str, object], *keys: str) -> bool:
    return all(
        isinstance(data.get(key), str) and bool(str(data.get(key)).strip())
        for key in keys
    )


def _valid_readiness(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if value.get("readinessUnavailable") is True:
        return False
    if value.get("readableStateReached") is False:
        return False
    if value.get("readableStateReached") is True:
        return True
    return value.get("state") in {"stable", "ready", "degraded_ready"}


def _canonical_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    safe = _safe_url(value.strip())
    return None if safe == "[REDACTED_URL]" else safe


def _destination_matches(
    expected: str | None,
    actual: object,
    navigation: object = None,
) -> bool:
    if not expected:
        return True

    def comparison_url(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = urlsplit(value.strip())
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
                return None
            port = parsed.port
        except ValueError:
            return None
        host = parsed.hostname.casefold()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if port is not None and not (
            (parsed.scheme.casefold() == "http" and port == 80)
            or (parsed.scheme.casefold() == "https" and port == 443)
        ):
            host = f"{host}:{port}"
        return urlunsplit(
            (parsed.scheme.casefold(), host, parsed.path or "/", parsed.query, "")
        )

    expected_url = comparison_url(expected)
    actual_url = comparison_url(actual)
    if expected_url == actual_url:
        return True
    if not isinstance(navigation, dict):
        return False
    redirect_count = navigation.get("redirectCount")
    return (
        navigation.get("committed") is True
        and isinstance(redirect_count, int)
        and not isinstance(redirect_count, bool)
        and redirect_count > 0
        and comparison_url(navigation.get("requestedUrl")) == expected_url
        and comparison_url(navigation.get("responseUrl")) == actual_url
        and comparison_url(navigation.get("finalUrl")) == actual_url
    )


def _target_evidence(result: dict[str, object]) -> dict[str, str]:
    evidence = {
        key: str(result[key]).strip()
        for key in ("targetId", "ref", "targetHref")
        if isinstance(result.get(key), str) and str(result[key]).strip()
    }
    resolution = result.get("targetResolution")
    if isinstance(resolution, dict):
        evidence.update(
            {
                str(key): str(value).strip()
                for key, value in resolution.items()
                if isinstance(value, str) and value.strip()
            }
        )
    return evidence


def _target_evidence_matches(
    result: dict[str, object], expected_target: dict[str, object] | None
) -> bool:
    evidence = _target_evidence(result)
    if not evidence:
        return False
    if not expected_target:
        return True
    aliases = {
        "href": "targetHref",
        "targetId": "targetId",
        "ref": "ref",
        "name": "name",
        "label": "label",
        "targetText": "targetText",
    }
    comparable = [
        (aliases[key], str(value).casefold().strip())
        for key, value in expected_target.items()
        if key in aliases and isinstance(value, str) and value.strip()
    ]
    if not comparable:
        return True
    return any(
        actual_key in evidence
        and evidence[actual_key].casefold().strip() == expected_value
        for actual_key, expected_value in comparable
    )


def _valid_takeover_creation(result: dict[str, object]) -> bool:
    if not _has_strings(
        result,
        "sessionId",
        "takeoverUrl",
        "expiresAt",
        "authenticationState",
        "state",
        "controlOwnership",
    ):
        return False
    takeover_url = _canonical_url(result.get("takeoverUrl"))
    if takeover_url is None or not takeover_url.startswith("https://"):
        return False
    try:
        expires_at = datetime.fromisoformat(str(result["expiresAt"]))
    except ValueError:
        return False
    return (
        expires_at.tzinfo is not None
        and result.get("authenticationState")
        in {"unknown", "authenticated", "unauthenticated"}
        and result.get("state") in {"awaiting_user", "active"}
        and result.get("controlOwnership") == "HUMAN_CONTROL"
    )


def verify_browser_result(
    action: str,
    result: object,
    *,
    expected_destination: str | None = None,
    expected_target: dict[str, object] | None = None,
) -> VerificationState:
    """Derive verification only from an actual Browser result, never a plan."""

    if action == "tabs":
        return (
            VerificationState.APPLIED_VERIFIED
            if isinstance(result, list)
            and bool(result)
            and all(
                isinstance(item, dict)
                and (_has_strings(item, "id") or _has_strings(item, "tabId"))
                for item in result
            )
            else VerificationState.UNVERIFIED
        )
    if not isinstance(result, dict):
        return VerificationState.UNVERIFIED
    declared = result.get("result")
    if declared == VerificationState.APPLIED_VERIFIED.value and action in {
        "click",
        "type",
        "submit",
        "scroll",
        "seek",
    }:
        if not _has_strings(result, "tabId"):
            return VerificationState.UNVERIFIED
        if action == "scroll":
            position = result.get("position")
            return (
                VerificationState.APPLIED_VERIFIED
                if isinstance(position, dict)
                and any(
                    isinstance(position.get(axis), (int, float)) for axis in ("x", "y")
                )
                else VerificationState.UNVERIFIED
            )
        if action == "seek" and result.get("found") is not True:
            return VerificationState.UNVERIFIED
        if action == "submit" and _has_strings(result, "url"):
            return VerificationState.APPLIED_VERIFIED
        if action == "type":
            return (
                VerificationState.APPLIED_VERIFIED
                if isinstance(result.get("characters"), int)
                and result.get("characters", -1) >= 0
                and isinstance(result.get("valueLength"), int)
                and result.get("valueLength", -1) >= 0
                and _target_evidence_matches(result, expected_target)
                else VerificationState.UNVERIFIED
            )
        return (
            VerificationState.APPLIED_VERIFIED
            if _target_evidence_matches(result, expected_target)
            else VerificationState.UNVERIFIED
        )
    if declared == VerificationState.APPLIED_UNVERIFIED.value:
        return VerificationState.APPLIED_UNVERIFIED
    if declared == VerificationState.CAPTURED_NOT_INTERPRETED.value:
        if action == "screenshot" and isinstance(result.get("bytes"), int):
            return VerificationState.CAPTURED_NOT_INTERPRETED
        return VerificationState.UNVERIFIED
    if action == "status":
        if result.get("state") == "ready" and isinstance(result.get("generation"), int):
            return VerificationState.APPLIED_VERIFIED
    elif action in {"navigate", "new_tab"}:
        if (
            _has_strings(result, "tabId", "url", "title")
            and _valid_readiness(result.get("readiness"))
            and _destination_matches(
                expected_destination,
                result.get("url"),
                result.get("navigation"),
            )
        ):
            return VerificationState.APPLIED_VERIFIED
    elif action == "read_page":
        if (
            _has_strings(result, "tabId", "url", "title", "content")
            and isinstance(result.get("truncated"), bool)
            and _valid_readiness(result.get("readiness"))
            and (
                bool(str(result.get("content") or "").strip())
                or (
                    isinstance(result.get("semantic"), dict)
                    and bool(result.get("semantic"))
                )
            )
        ):
            return VerificationState.APPLIED_VERIFIED
    elif action in {"back", "forward"}:
        if (
            result.get("navigated") is True
            and _has_strings(result, "tabId", "url")
            and _valid_readiness(result.get("readiness"))
        ):
            return VerificationState.APPLIED_VERIFIED
        if result.get("navigated") is False:
            return VerificationState.APPLIED_UNVERIFIED
    elif action == "refresh":
        if _has_strings(result, "tabId", "url", "title") and _valid_readiness(
            result.get("readiness")
        ):
            return VerificationState.APPLIED_VERIFIED
    elif action == "close_tab":
        if _has_strings(result, "closedTabId"):
            return VerificationState.APPLIED_VERIFIED
    elif any(
        (
            action == "switch_tab" and _has_strings(result, "tabId", "url", "title"),
            action == "takeover_create" and _valid_takeover_creation(result),
            action == "takeover_status"
            and _has_strings(result, "sessionId", "state", "controlOwnership")
            and result.get("state") in {"awaiting_user", "active", "ended", "expired"},
            action == "takeover_revoke"
            and _has_strings(result, "sessionId", "state", "controlOwnership")
            and result.get("state") == "ended"
            and result.get("controlOwnership") == "AUTOMATION",
            action == "takeover_consume"
            and _has_strings(
                result, "sessionId", "state", "controlOwnership", "resumeTask"
            )
            and result.get("state") == "ended"
            and result.get("controlOwnership") == "AUTOMATION"
            and result.get("resumeConsumed") is True,
        )
    ):
        return VerificationState.APPLIED_VERIFIED
    return VerificationState.UNVERIFIED


def _error_verification(code: str) -> VerificationState:
    if code in {
        "semantic_target_not_found",
        "target_not_found",
        "browser_tab_not_found",
        "tab_not_found",
    }:
        return VerificationState.TARGET_MISSING
    if code in {
        "prohibited_browser_action",
        "purchase_confirmation_required",
        "url_not_allowed",
        "sensitive_browser_task_rejected",
        "takeover_active",
    }:
        return VerificationState.BLOCKED
    return VerificationState.FAILED


class ExecutionLedger:
    """Authoritative request-scoped Browser execution record in Core SQLite."""

    _SCHEMA_PATH = (
        Path(__file__).resolve().parents[2] / "migrations" / "002_execution_ledger.sql"
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self._write_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema = self._SCHEMA_PATH.read_text(encoding="utf-8")
        with self._write_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(schema)

    def begin_request(
        self, request_id: str, conversation_id: str, *, task_kind: str = "none"
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO execution_requests(
                    request_id, conversation_id, task_kind, task_status,
                    chat_status, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO NOTHING
                """,
                (
                    request_id,
                    conversation_id,
                    task_kind,
                    TaskStatus.NOT_EXECUTED.value,
                    ChatStatus.IN_PROGRESS.value,
                    now,
                ),
            )

    def register_subgoals(
        self,
        request_id: str,
        subgoals: Iterable[RequiredSubgoalInput],
        *,
        task_kind: str,
    ) -> tuple[str, ...]:
        items = tuple(subgoals)
        if not items:
            with self._write_lock, self._connect() as connection:
                connection.execute(
                    "UPDATE execution_requests SET task_kind = ? WHERE request_id = ?",
                    (task_kind, request_id),
                )
            return ()
        now = datetime.now(UTC).isoformat()
        action_ids = tuple(str(uuid.uuid4()) for _ in items)
        with self._write_lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT action_id FROM execution_subgoals WHERE request_id = ? ORDER BY sequence",
                (request_id,),
            ).fetchall()
            if existing:
                return tuple(str(row["action_id"]) for row in existing)
            connection.execute(
                """
                UPDATE execution_requests
                SET task_kind = ?, task_status = ?
                WHERE request_id = ?
                """,
                (task_kind, TaskStatus.NOT_EXECUTED.value, request_id),
            )
            for sequence, (action_id, item) in enumerate(zip(action_ids, items)):
                dependency = (
                    action_ids[item.dependency_sequence]
                    if item.dependency_sequence is not None
                    and 0 <= item.dependency_sequence < len(action_ids)
                    else None
                )
                connection.execute(
                    """
                    INSERT INTO execution_subgoals(
                        action_id, request_id, sequence, original_subgoal_json,
                        normalized_action, target_json, destination, required,
                        polarity, condition_json, dependency_action_id, source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'required_task', ?)
                    """,
                    (
                        action_id,
                        request_id,
                        sequence,
                        _json(item.original_required_subgoal),
                        item.normalized_action,
                        _json(item.target) if item.target else None,
                        _safe_string(item.destination, key="destination"),
                        int(item.required),
                        item.polarity,
                        _json(item.condition) if item.condition else None,
                        dependency,
                        now,
                    ),
                )
        return action_ids

    def _implicit_action(
        self,
        connection: sqlite3.Connection,
        request_id: str,
        action: str,
        call: dict[str, object],
    ) -> tuple[str, int]:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 AS next_sequence FROM execution_subgoals WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        action_id = str(uuid.uuid4())
        destination = _safe_string(call.get("url"), key="destination")
        target = {
            key: call[key]
            for key in (
                "targetId",
                "ref",
                "role",
                "name",
                "label",
                "targetText",
                "href",
            )
            if key in call
        }
        connection.execute(
            """
            INSERT INTO execution_subgoals(
                action_id, request_id, sequence, original_subgoal_json,
                normalized_action, target_json, destination, required,
                polarity, condition_json, dependency_action_id, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'positive', NULL, NULL, 'internal', ?)
            """,
            (
                action_id,
                request_id,
                sequence,
                _json({"kind": action, "source": "actual_browser_call"}),
                action,
                _json(target) if target else None,
                destination,
                datetime.now(UTC).isoformat(),
            ),
        )
        return action_id, sequence

    def begin_attempt(
        self,
        request_id: str,
        action: str,
        call: dict[str, object],
        *,
        action_id: str | None = None,
        page_generation: int | None = None,
    ) -> str:
        now = datetime.now(UTC).isoformat()
        attempt_id = str(uuid.uuid4())
        with self._write_lock, self._connect() as connection:
            request = connection.execute(
                """
                SELECT chat_status, completed_at FROM execution_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            if request is None:
                raise KeyError(f"unknown execution request: {request_id}")
            if (
                request["chat_status"] != ChatStatus.IN_PROGRESS.value
                or request["completed_at"] is not None
            ):
                raise ExecutionRequestClosedError(request_id)
            selected = None
            if action_id:
                selected = connection.execute(
                    """
                    SELECT action_id, sequence FROM execution_subgoals
                    WHERE request_id = ? AND action_id = ?
                    """,
                    (request_id, action_id),
                ).fetchone()
            if selected is None:
                resolved_action_id, sequence = self._implicit_action(
                    connection, request_id, action, call
                )
            else:
                resolved_action_id = str(selected["action_id"])
                sequence = int(selected["sequence"])
            row = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
                FROM execution_attempts WHERE request_id = ? AND action_id = ?
                """,
                (request_id, resolved_action_id),
            ).fetchone()
            attempt_number = int(row["next_attempt"])
            call_row = connection.execute(
                """
                SELECT COALESCE(MAX(call_sequence), -1) + 1 AS next_call
                FROM execution_attempts WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            call_sequence = int(call_row["next_call"])
            tab_id = (
                str(call.get("tabId")) if isinstance(call.get("tabId"), str) else None
            )
            page_identity = (
                f"generation:{page_generation}:tab:{tab_id}"
                if page_generation is not None and tab_id
                else None
            )
            destination = _safe_string(call.get("url"), key="destination")
            target = {
                key: call[key]
                for key in (
                    "targetId",
                    "ref",
                    "role",
                    "name",
                    "label",
                    "targetText",
                    "href",
                )
                if key in call
            }
            connection.execute(
                """
                INSERT INTO execution_attempts(
                    attempt_id, request_id, action_id, action_sequence, call_sequence, attempt_number,
                    normalized_action, target_json, destination, tab_id, page_identity,
                    page_generation, started_at, actual_browser_call_json,
                    verification_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    request_id,
                    resolved_action_id,
                    sequence,
                    call_sequence,
                    attempt_number,
                    action,
                    _json(target) if target else None,
                    destination,
                    tab_id,
                    page_identity,
                    page_generation,
                    now,
                    _json({"action": action, **call}),
                    VerificationState.UNVERIFIED.value,
                ),
            )
            connection.execute(
                """
                UPDATE execution_requests
                SET task_status = ?
                WHERE request_id = ? AND chat_status = ?
                """,
                (
                    TaskStatus.IN_PROGRESS.value,
                    request_id,
                    ChatStatus.IN_PROGRESS.value,
                ),
            )
        return attempt_id

    def complete_attempt(
        self,
        attempt_id: str,
        *,
        action: str,
        raw_result: object | None = None,
        error_code: str | None = None,
        page_generation: int | None = None,
    ) -> VerificationState:
        with self._connect() as connection:
            attempt = connection.execute(
                """
                SELECT destination, target_json FROM execution_attempts
                WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        expected_destination = attempt["destination"] if attempt is not None else None
        expected_target = (
            _parsed(attempt["target_json"])
            if attempt is not None and attempt["target_json"]
            else None
        )
        safe_error_code = _safe_error(error_code)
        state = (
            _error_verification(safe_error_code)
            if safe_error_code
            else verify_browser_result(
                action,
                raw_result,
                expected_destination=expected_destination,
                expected_target=(
                    expected_target if isinstance(expected_target, dict) else None
                ),
            )
        )
        raw_safe = _safe_value(raw_result)
        normalized = {
            "action": action,
            "verification_state": state.value,
        }
        if isinstance(raw_result, dict):
            for key in (
                "tabId",
                "url",
                "title",
                "navigated",
                "result",
                "found",
                "closedTabId",
                "activeTabId",
            ):
                if key in raw_result:
                    normalized[key] = _safe_value(raw_result[key], key=key)
        if safe_error_code:
            normalized["error_code"] = safe_error_code
        evidence_reference = (
            "sha256:" + hashlib.sha256(_json(normalized).encode("utf-8")).hexdigest()
        )
        tab_id = (
            str(raw_result.get("tabId"))
            if isinstance(raw_result, dict) and isinstance(raw_result.get("tabId"), str)
            else None
        )
        actual_url = _safe_string(
            raw_result.get("url") if isinstance(raw_result, dict) else None,
            key="actual_final_url",
        )
        actual_title = _safe_string(
            raw_result.get("title") if isinstance(raw_result, dict) else None,
            key="title",
        )
        page_identity = (
            f"generation:{page_generation}:tab:{tab_id}"
            if page_generation is not None and tab_id
            else None
        )
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE execution_attempts
                SET completed_at = ?, raw_browser_result_json = ?,
                    normalized_result_json = ?, verification_state = ?,
                    error_code = ?, partial_blocked = ?,
                    tab_id = COALESCE(?, tab_id),
                    page_identity = COALESCE(?, page_identity),
                    page_generation = COALESCE(?, page_generation),
                    actual_final_url = ?, actual_title = ?, evidence_reference = ?
                WHERE attempt_id = ?
                """,
                (
                    datetime.now(UTC).isoformat(),
                    _json(raw_safe) if raw_result is not None else None,
                    _json(normalized),
                    state.value,
                    safe_error_code,
                    int(
                        state
                        in {VerificationState.BLOCKED, VerificationState.TARGET_MISSING}
                    ),
                    tab_id,
                    page_identity,
                    page_generation,
                    actual_url,
                    actual_title,
                    evidence_reference,
                    attempt_id,
                ),
            )
        return state

    @staticmethod
    def _task_status(snapshot: ExecutionRequestSnapshot) -> TaskStatus:
        required = [item for item in snapshot.subgoals if item.get("required") is True]
        final_attempts = {
            str(item["action_id"]): item for item in snapshot.final_attempts
        }
        satisfied = 0
        blocked = False
        attempted_required = False
        for item in required:
            attempt = final_attempts.get(str(item["action_id"]))
            attempted_required = attempted_required or attempt is not None
            state = str(attempt.get("verification_state")) if attempt else None
            if state == VerificationState.APPLIED_VERIFIED.value or (
                item.get("normalized_action") == "screenshot"
                and state == VerificationState.CAPTURED_NOT_INTERPRETED.value
            ):
                satisfied += 1
            if state in {
                VerificationState.BLOCKED.value,
                VerificationState.TARGET_MISSING.value,
            }:
                blocked = True
        if required and satisfied == len(required):
            return TaskStatus.COMPLETED
        if required and satisfied:
            return TaskStatus.PARTIAL
        if blocked:
            return TaskStatus.BLOCKED
        if attempted_required:
            return TaskStatus.FAILED
        if required:
            return TaskStatus.NOT_EXECUTED
        actual_states = {
            str(item.get("verification_state"))
            for item in snapshot.final_attempts
            if item.get("normalized_action") != "status"
        }
        if actual_states & {
            VerificationState.APPLIED_VERIFIED.value,
            VerificationState.CAPTURED_NOT_INTERPRETED.value,
        }:
            return (
                TaskStatus.PARTIAL
                if actual_states
                & {
                    VerificationState.FAILED.value,
                    VerificationState.BLOCKED.value,
                    VerificationState.TARGET_MISSING.value,
                    VerificationState.APPLIED_UNVERIFIED.value,
                    VerificationState.UNVERIFIED.value,
                }
                else TaskStatus.COMPLETED
            )
        if actual_states & {
            VerificationState.BLOCKED.value,
            VerificationState.TARGET_MISSING.value,
        }:
            return TaskStatus.BLOCKED
        if actual_states:
            return TaskStatus.FAILED
        return TaskStatus.NOT_EXECUTED

    def evaluate_request(self, request_id: str) -> TaskStatus:
        snapshot = self.snapshot(request_id)
        return self._task_status(snapshot) if snapshot else TaskStatus.NOT_EXECUTED

    def finalize_request(
        self, request_id: str, *, chat_status: ChatStatus | None = None
    ) -> TaskStatus:
        now = datetime.now(UTC).isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE execution_attempts
                SET completed_at = ?, verification_state = ?, error_code = ?,
                    normalized_result_json = ?
                WHERE request_id = ? AND completed_at IS NULL
                """,
                (
                    now,
                    VerificationState.FAILED.value,
                    "request_finalized_with_incomplete_attempt",
                    _json(
                        {
                            "verification_state": VerificationState.FAILED.value,
                            "error_code": "request_finalized_with_incomplete_attempt",
                        }
                    ),
                    request_id,
                ),
            )
        snapshot = self.snapshot(request_id)
        if snapshot is None:
            return TaskStatus.NOT_EXECUTED
        task_status = self._task_status(snapshot)
        completed_at = (
            datetime.now(UTC).isoformat()
            if task_status != TaskStatus.IN_PROGRESS
            else None
        )
        with self._write_lock, self._connect() as connection:
            if chat_status is None:
                connection.execute(
                    """
                    UPDATE execution_requests SET task_status = ?, completed_at = ?
                    WHERE request_id = ?
                    """,
                    (task_status.value, completed_at, request_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE execution_requests
                    SET task_status = ?, chat_status = ?, completed_at = ?
                    WHERE request_id = ?
                    """,
                    (task_status.value, chat_status.value, completed_at, request_id),
                )
        return task_status

    def snapshot(self, request_id: str) -> ExecutionRequestSnapshot | None:
        with self._connect() as connection:
            connection.execute("BEGIN")
            request = connection.execute(
                "SELECT * FROM execution_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if request is None:
                return None
            subgoal_rows = connection.execute(
                """
                SELECT * FROM execution_subgoals
                WHERE request_id = ? ORDER BY sequence
                """,
                (request_id,),
            ).fetchall()
            attempt_rows = connection.execute(
                """
                SELECT * FROM execution_attempts
                WHERE request_id = ? ORDER BY call_sequence
                """,
                (request_id,),
            ).fetchall()
        subgoals = tuple(
            {
                "action_id": str(row["action_id"]),
                "sequence": int(row["sequence"]),
                "original_required_subgoal": _parsed(row["original_subgoal_json"]),
                "normalized_action": str(row["normalized_action"]),
                "target": _parsed(row["target_json"]),
                "destination": row["destination"],
                "required": bool(row["required"]),
                "polarity": str(row["polarity"]),
                "condition": _parsed(row["condition_json"]),
                "dependency_action_id": row["dependency_action_id"],
                "source": str(row["source"]),
            }
            for row in subgoal_rows
        )
        attempts = tuple(
            {
                "attempt_id": str(row["attempt_id"]),
                "request_id": str(row["request_id"]),
                "action_id": str(row["action_id"]),
                "action_sequence": int(row["action_sequence"]),
                "call_sequence": int(row["call_sequence"]),
                "attempt_number": int(row["attempt_number"]),
                "normalized_action": str(row["normalized_action"]),
                "target": _parsed(row["target_json"]),
                "destination": row["destination"],
                "tab_id": row["tab_id"],
                "page_identity": row["page_identity"],
                "page_generation": row["page_generation"],
                "started_at": str(row["started_at"]),
                "completed_at": row["completed_at"],
                "actual_browser_call": _parsed(row["actual_browser_call_json"]),
                "raw_browser_result": _parsed(row["raw_browser_result_json"]),
                "normalized_result": _parsed(row["normalized_result_json"]),
                "verification_state": str(row["verification_state"]),
                "error_code": row["error_code"],
                "partial_blocked": bool(row["partial_blocked"]),
                "actual_final_url": row["actual_final_url"],
                "actual_title": row["actual_title"],
                "evidence_reference": row["evidence_reference"],
            }
            for row in attempt_rows
        )
        return ExecutionRequestSnapshot(
            request_id=str(request["request_id"]),
            conversation_id=str(request["conversation_id"]),
            task_kind=str(request["task_kind"]),
            task_status=str(request["task_status"]),
            chat_status=str(request["chat_status"]),
            started_at=str(request["started_at"]),
            completed_at=request["completed_at"],
            subgoals=subgoals,
            attempts=attempts,
        )

    def previous_request(
        self, conversation_id: str, *, before_request_id: str
    ) -> ExecutionRequestSnapshot | None:
        with self._connect() as connection:
            current = connection.execute(
                """
                SELECT started_at, rowid FROM execution_requests
                WHERE request_id = ? AND conversation_id = ?
                """,
                (before_request_id, conversation_id),
            ).fetchone()
            if current is None:
                return None
            row = connection.execute(
                """
                SELECT request_id FROM execution_requests
                WHERE conversation_id = ?
                  AND (
                    started_at < ?
                    OR (started_at = ? AND rowid < ?)
                  )
                ORDER BY started_at DESC, rowid DESC LIMIT 1
                """,
                (
                    conversation_id,
                    current["started_at"],
                    current["started_at"],
                    current["rowid"],
                ),
            ).fetchone()
        return self.snapshot(str(row["request_id"])) if row else None
