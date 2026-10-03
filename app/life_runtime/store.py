from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

Mode = Literal["engaged", "idle", "dormant"]
ActivityStatus = Literal["active", "waiting", "completed", "abandoned"]
InteriorStatus = Literal["private", "candidate", "shared", "abandoned"]

ACTIVITY_TRANSITIONS: dict[str, frozenset[str]] = {
    "active": frozenset({"waiting", "completed", "abandoned"}),
    "waiting": frozenset({"active", "completed", "abandoned"}),
    "completed": frozenset(),
    "abandoned": frozenset(),
}
INTERIOR_TRANSITIONS: dict[str, frozenset[str]] = {
    "private": frozenset({"candidate", "abandoned"}),
    "candidate": frozenset({"private", "shared", "abandoned"}),
    "shared": frozenset(),
    "abandoned": frozenset(),
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return value.astimezone(UTC).isoformat()


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True, slots=True)
class LifeState:
    mode: Mode
    last_wake_at: datetime | None
    next_wake_at: datetime
    next_meaningful_wake_at: datetime | None
    daily_date: date
    daily_cognition_calls: int
    daily_tokens: int
    budget_pressure: bool
    life_session_id: str | None
    shared_conversation_id: str | None
    claim_token: str | None
    claim_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class LifeActivity:
    activity_id: str
    kind: str
    summary: str
    status: ActivityStatus
    state_payload: dict[str, Any]
    started_at: datetime
    updated_at: datetime
    next_wake_at: datetime | None


@dataclass(frozen=True, slots=True)
class LifeInterior:
    thought_id: str
    content: str
    status: InteriorStatus
    created_at: datetime
    updated_at: datetime
    revisit_at: datetime | None
    source_activity_id: str | None


@dataclass(frozen=True, slots=True)
class WakeClaim:
    token: str
    reason: str
    state: LifeState
    trigger_type: str
    activity_id: str | None
    thought_id: str | None
    due_at: datetime


class LifeStore:
    """SQLite authority for Life Runtime machine state, separate from Memory."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self, *, now: datetime | None = None, first_wake_after: timedelta = timedelta(hours=6)) -> None:
        current = now or utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS life_state (
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
                    mode TEXT NOT NULL CHECK(mode IN ('engaged','idle','dormant')),
                    last_wake_at TEXT,
                    next_wake_at TEXT NOT NULL,
                    daily_date TEXT NOT NULL,
                    daily_cognition_calls INTEGER NOT NULL DEFAULT 0,
                    daily_tokens INTEGER NOT NULL DEFAULT 0,
                    budget_pressure INTEGER NOT NULL DEFAULT 0,
                    life_session_id TEXT,
                    shared_conversation_id TEXT,
                    claim_token TEXT,
                    claim_expires_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS life_activities (
                    activity_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','waiting','completed','abandoned')),
                    state_payload TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    next_wake_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_life_activities_due
                    ON life_activities(status, next_wake_at);
                CREATE TABLE IF NOT EXISTS life_interior (
                    thought_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('private','candidate','shared','abandoned')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revisit_at TEXT,
                    source_activity_id TEXT,
                    FOREIGN KEY(source_activity_id) REFERENCES life_activities(activity_id)
                );
                CREATE INDEX IF NOT EXISTS idx_life_interior_due
                    ON life_interior(status, revisit_at);
                CREATE TABLE IF NOT EXISTS life_ledger (
                    ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    activity_id TEXT,
                    thought_id TEXT,
                    hermes_session_id TEXT,
                    api_calls INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    usage_unknown INTEGER NOT NULL DEFAULT 0,
                    receipt_status TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_life_ledger_time
                    ON life_ledger(occurred_at, ledger_id);
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO life_state(
                    singleton_id, mode, next_wake_at, daily_date, updated_at
                ) VALUES (1, 'idle', ?, ?, ?)
                """,
                (_iso(current + first_wake_after), current.date().isoformat(), _iso(current)),
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(life_state)")
            }
            if "shared_conversation_id" not in columns:
                connection.execute(
                    "ALTER TABLE life_state ADD COLUMN shared_conversation_id TEXT"
                )
            if "next_meaningful_wake_at" not in columns:
                connection.execute(
                    "ALTER TABLE life_state ADD COLUMN next_meaningful_wake_at TEXT"
                )
            self._refresh_meaningful_deadline(connection, current)

    def _state_from_row(self, row: sqlite3.Row) -> LifeState:
        next_wake = _dt(row["next_wake_at"])
        assert next_wake is not None
        return LifeState(
            mode=row["mode"],
            last_wake_at=_dt(row["last_wake_at"]),
            next_wake_at=next_wake,
            next_meaningful_wake_at=_dt(row["next_meaningful_wake_at"]),
            daily_date=date.fromisoformat(row["daily_date"]),
            daily_cognition_calls=row["daily_cognition_calls"],
            daily_tokens=row["daily_tokens"],
            budget_pressure=bool(row["budget_pressure"]),
            life_session_id=row["life_session_id"],
            shared_conversation_id=row["shared_conversation_id"],
            claim_token=row["claim_token"],
            claim_expires_at=_dt(row["claim_expires_at"]),
        )

    def get_state(self) -> LifeState:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM life_state WHERE singleton_id = 1").fetchone()
        if row is None:
            raise RuntimeError("LifeStore is not initialized")
        return self._state_from_row(row)

    def claim_due_wake(
        self,
        now: datetime,
        *,
        reason: str,
        lease: timedelta = timedelta(minutes=10),
    ) -> WakeClaim | None:
        token = str(uuid.uuid4())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM life_state WHERE singleton_id = 1").fetchone()
            if row is None:
                raise RuntimeError("LifeStore is not initialized")
            if date.fromisoformat(row["daily_date"]) != now.date():
                connection.execute(
                    """UPDATE life_state SET daily_date = ?, daily_cognition_calls = 0,
                       daily_tokens = 0, budget_pressure = 0, mode = CASE
                       WHEN mode = 'dormant' THEN 'idle' ELSE mode END, updated_at = ?
                       WHERE singleton_id = 1""",
                    (now.date().isoformat(), _iso(now)),
                )
                row = connection.execute("SELECT * FROM life_state WHERE singleton_id = 1").fetchone()
            due = connection.execute(
                """SELECT trigger_type, related_id, due_at FROM (
                       SELECT 'activity_due' AS trigger_type, activity_id AS related_id,
                              next_wake_at AS due_at
                       FROM life_activities
                       WHERE status IN ('active','waiting') AND next_wake_at IS NOT NULL
                       UNION ALL
                       SELECT 'interior_revisit_due', thought_id, revisit_at
                       FROM life_interior
                       WHERE status IN ('private','candidate') AND revisit_at IS NOT NULL
                   ) WHERE due_at <= ? ORDER BY due_at, trigger_type, related_id LIMIT 1""",
                (_iso(now),),
            ).fetchone()
            claim_expires = _dt(row["claim_expires_at"])
            if due is None:
                connection.rollback()
                return None
            if row["claim_token"] and claim_expires and claim_expires > now:
                connection.rollback()
                return None
            connection.execute(
                """UPDATE life_state SET claim_token = ?, claim_expires_at = ?, updated_at = ?
                   WHERE singleton_id = 1""",
                (token, _iso(now + lease), _iso(now)),
            )
            claimed = connection.execute("SELECT * FROM life_state WHERE singleton_id = 1").fetchone()
            connection.commit()
            return WakeClaim(
                token=token, reason=reason, state=self._state_from_row(claimed),
                trigger_type=due["trigger_type"],
                activity_id=due["related_id"] if due["trigger_type"] == "activity_due" else None,
                thought_id=due["related_id"] if due["trigger_type"] == "interior_revisit_due" else None,
                due_at=_dt(due["due_at"]),
            )
        finally:
            connection.close()

    def complete_wake(
        self,
        token: str,
        *,
        now: datetime,
        next_wake_at: datetime | None = None,
        mode: Mode | None = None,
    ) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE life_state SET last_wake_at = ?,
                   mode = COALESCE(?, mode), claim_token = NULL, claim_expires_at = NULL,
                   updated_at = ? WHERE singleton_id = 1 AND claim_token = ?""",
                (_iso(now), mode, _iso(now), token),
            )
            if result.rowcount == 1:
                self._refresh_meaningful_deadline(connection, now)
        return result.rowcount == 1

    def _refresh_meaningful_deadline(
        self, connection: sqlite3.Connection, now: datetime
    ) -> datetime | None:
        row = connection.execute(
            """SELECT min(due_at) AS due_at FROM (
                   SELECT next_wake_at AS due_at FROM life_activities
                   WHERE status IN ('active','waiting') AND next_wake_at IS NOT NULL
                   UNION ALL
                   SELECT revisit_at FROM life_interior
                   WHERE status IN ('private','candidate') AND revisit_at IS NOT NULL
               )"""
        ).fetchone()
        deadline = _dt(row["due_at"]) if row else None
        connection.execute(
            """UPDATE life_state SET next_meaningful_wake_at = ?, updated_at = ?
               WHERE singleton_id = 1""",
            (_iso(deadline), _iso(now)),
        )
        return deadline

    def refresh_meaningful_deadline(self, *, now: datetime) -> datetime | None:
        with self._connect() as connection:
            return self._refresh_meaningful_deadline(connection, now)

    def set_life_session(self, session_id: str, *, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE life_state SET life_session_id = ?, updated_at = ? WHERE singleton_id = 1",
                (session_id, _iso(now)),
            )

    def set_shared_conversation(self, session_id: str, *, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE life_state SET shared_conversation_id = ?, updated_at = ?
                   WHERE singleton_id = 1""",
                (session_id, _iso(now)),
            )

    def delivery_facts(
        self,
        thought_id: str,
        *,
        now: datetime,
        local_utc_offset_hours: int = 8,
    ) -> dict[str, Any]:
        offset = timedelta(hours=local_utc_offset_hours)
        local_date = (now + offset).date()
        day_start = datetime.combine(local_date, datetime.min.time(), tzinfo=UTC) - offset
        with self._connect() as connection:
            succeeded = connection.execute(
                """SELECT ledger_id FROM life_ledger WHERE thought_id = ?
                   AND event_type = 'DELIVERY_SUCCEEDED' LIMIT 1""",
                (thought_id,),
            ).fetchone()
            incomplete = connection.execute(
                """SELECT ledger_id FROM life_ledger WHERE thought_id = ?
                   AND event_type = 'DELIVERY_ATTEMPTED' AND receipt_status = 'incomplete'
                   LIMIT 1""",
                (thought_id,),
            ).fetchone()
            daily_count = connection.execute(
                """SELECT count(*) FROM life_ledger WHERE event_type = 'DELIVERY_SUCCEEDED'
                   AND occurred_at >= ?""",
                (_iso(day_start),),
            ).fetchone()[0]
            last_row = connection.execute(
                """SELECT occurred_at FROM life_ledger
                   WHERE event_type = 'DELIVERY_SUCCEEDED'
                   ORDER BY occurred_at DESC, ledger_id DESC LIMIT 1"""
            ).fetchone()
        return {
            "succeeded": succeeded is not None,
            "incomplete": incomplete is not None,
            "daily_count": int(daily_count),
            "last_succeeded_at": _dt(last_row["occurred_at"]) if last_row else None,
        }

    def claim_delivery_attempt(
        self,
        thought_id: str,
        *,
        now: datetime,
        reason: str,
    ) -> int | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            thought = connection.execute(
                "SELECT status FROM life_interior WHERE thought_id = ?", (thought_id,)
            ).fetchone()
            if thought is None or thought["status"] != "candidate":
                connection.rollback()
                return None
            existing = connection.execute(
                """SELECT ledger_id FROM life_ledger WHERE thought_id = ? AND (
                   event_type = 'DELIVERY_SUCCEEDED' OR
                   (event_type = 'DELIVERY_ATTEMPTED' AND receipt_status = 'incomplete'))
                   LIMIT 1""",
                (thought_id,),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                return None
            cursor = connection.execute(
                """INSERT INTO life_ledger(occurred_at, event_type, reason, thought_id,
                   usage_unknown, receipt_status, details)
                   VALUES (?, 'DELIVERY_ATTEMPTED', ?, ?, 0, 'incomplete', '{}')""",
                (_iso(now), reason, thought_id),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def runtime_status(
        self,
        *,
        now: datetime,
        local_utc_offset_hours: int = 8,
    ) -> dict[str, Any]:
        state = self.get_state()
        offset = timedelta(hours=local_utc_offset_hours)
        local_date = (now + offset).date()
        day_start = datetime.combine(local_date, datetime.min.time(), tzinfo=UTC) - offset
        with self._connect() as connection:
            wake_count = connection.execute(
                "SELECT count(*) FROM life_ledger WHERE event_type = 'WAKE' AND occurred_at >= ?",
                (_iso(day_start),),
            ).fetchone()[0]
            active_count = connection.execute(
                "SELECT count(*) FROM life_activities WHERE status IN ('active','waiting')"
            ).fetchone()[0]
            private_count = connection.execute(
                "SELECT count(*) FROM life_interior WHERE status = 'private'"
            ).fetchone()[0]
            candidate_count = connection.execute(
                "SELECT count(*) FROM life_interior WHERE status = 'candidate'"
            ).fetchone()[0]
            delivery_count = connection.execute(
                """SELECT count(*) FROM life_ledger WHERE event_type = 'DELIVERY_SUCCEEDED'
                   AND occurred_at >= ?""",
                (_iso(day_start),),
            ).fetchone()[0]
            last = connection.execute(
                """SELECT event_type, occurred_at FROM life_ledger
                   WHERE event_type IN ('REST','COGNITION_COMPLETED','COGNITION_ERROR',
                   'DELIVERY_SUCCEEDED','DELIVERY_FAILED','DELIVERY_SUPPRESSED')
                   ORDER BY ledger_id DESC LIMIT 1"""
            ).fetchone()
        outcome_type = None
        outcome_at = None
        if last is not None:
            outcome_type = {
                "COGNITION_COMPLETED": "COGNITION",
                "COGNITION_ERROR": "COGNITION_ERROR",
                "DELIVERY_SUCCEEDED": "DELIVERY",
                "DELIVERY_FAILED": "DELIVERY_FAILED",
                "DELIVERY_SUPPRESSED": "DELIVERY_SUPPRESSED",
            }.get(last["event_type"], last["event_type"])
            outcome_at = last["occurred_at"]
        return {
            "mode": state.mode,
            "last_wake_at": _iso(state.last_wake_at),
            "next_wake_at": _iso(state.next_meaningful_wake_at),
            "today": {
                "wake_count": int(wake_count),
                "cognition_count": state.daily_cognition_calls,
                "token_usage": state.daily_tokens,
                "proactive_delivery_count": int(delivery_count),
            },
            "active_activity": {"exists": bool(active_count), "count": int(active_count)},
            "interior": {
                "private_count": int(private_count),
                "candidate_count": int(candidate_count),
            },
            "last_outcome": {"type": outcome_type, "at": outcome_at},
            "shared_conversation_ready": state.shared_conversation_id is not None,
        }

    def record_usage(
        self,
        *,
        now: datetime,
        api_calls: int,
        total_tokens: int | None,
        call_limit: int,
        token_limit: int,
    ) -> LifeState:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM life_state WHERE singleton_id = 1").fetchone()
            same_day = date.fromisoformat(row["daily_date"]) == now.date()
            calls = (row["daily_cognition_calls"] if same_day else 0) + max(1, api_calls)
            tokens = (row["daily_tokens"] if same_day else 0) + max(0, total_tokens or 0)
            pressure = calls >= call_limit or tokens >= token_limit or total_tokens is None
            prior_mode = row["mode"] if same_day else ("idle" if row["mode"] == "dormant" else row["mode"])
            mode = "dormant" if pressure else prior_mode
            connection.execute(
                """UPDATE life_state SET daily_date = ?, daily_cognition_calls = ?, daily_tokens = ?,
                   budget_pressure = ?, mode = ?, updated_at = ? WHERE singleton_id = 1""",
                (now.date().isoformat(), calls, tokens, int(pressure), mode, _iso(now)),
            )
            updated = connection.execute("SELECT * FROM life_state WHERE singleton_id = 1").fetchone()
            connection.commit()
            return self._state_from_row(updated)
        finally:
            connection.close()

    def add_activity(
        self,
        *,
        kind: str,
        summary: str,
        now: datetime,
        status: ActivityStatus = "active",
        state_payload: dict[str, Any] | None = None,
        next_wake_at: datetime | None = None,
        activity_id: str | None = None,
    ) -> LifeActivity:
        identifier = activity_id or str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO life_activities(activity_id, kind, summary, status,
                   state_payload, started_at, updated_at, next_wake_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (identifier, kind, summary, status, json.dumps(state_payload or {}, ensure_ascii=False),
                 _iso(now), _iso(now), _iso(next_wake_at)),
            )
            self._refresh_meaningful_deadline(connection, now)
        return self.get_activity(identifier)

    def get_activity(self, activity_id: str) -> LifeActivity:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM life_activities WHERE activity_id = ?", (activity_id,)
            ).fetchone()
        if row is None:
            raise KeyError(activity_id)
        return LifeActivity(
            activity_id=row["activity_id"], kind=row["kind"], summary=row["summary"],
            status=row["status"], state_payload=json.loads(row["state_payload"]),
            started_at=_dt(row["started_at"]), updated_at=_dt(row["updated_at"]),
            next_wake_at=_dt(row["next_wake_at"]),
        )

    def due_activities(self, now: datetime) -> list[LifeActivity]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT activity_id FROM life_activities
                   WHERE status IN ('active','waiting') AND next_wake_at IS NOT NULL
                   AND next_wake_at <= ? ORDER BY next_wake_at, activity_id""",
                (_iso(now),),
            ).fetchall()
        return [self.get_activity(row["activity_id"]) for row in rows]

    def transition_activity(
        self,
        activity_id: str,
        status: ActivityStatus,
        *,
        now: datetime,
        state_payload: dict[str, Any] | None = None,
        next_wake_at: datetime | None = None,
    ) -> LifeActivity:
        current = self.get_activity(activity_id)
        if status != current.status and status not in ACTIVITY_TRANSITIONS[current.status]:
            raise ValueError(f"illegal activity transition: {current.status} -> {status}")
        payload = current.state_payload if state_payload is None else state_payload
        with self._connect() as connection:
            connection.execute(
                """UPDATE life_activities SET status = ?, state_payload = ?,
                   updated_at = ?, next_wake_at = ? WHERE activity_id = ?""",
                (status, json.dumps(payload, ensure_ascii=False), _iso(now),
                 _iso(next_wake_at), activity_id),
            )
            self._refresh_meaningful_deadline(connection, now)
        return self.get_activity(activity_id)

    def add_thought(
        self,
        *,
        content: str,
        now: datetime,
        status: InteriorStatus = "private",
        revisit_at: datetime | None = None,
        source_activity_id: str | None = None,
        thought_id: str | None = None,
    ) -> LifeInterior:
        identifier = thought_id or str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO life_interior(thought_id, content, status, created_at,
                   updated_at, revisit_at, source_activity_id) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (identifier, content, status, _iso(now), _iso(now), _iso(revisit_at), source_activity_id),
            )
            self._refresh_meaningful_deadline(connection, now)
        return self.get_thought(identifier)

    def get_thought(self, thought_id: str) -> LifeInterior:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM life_interior WHERE thought_id = ?", (thought_id,)
            ).fetchone()
        if row is None:
            raise KeyError(thought_id)
        return LifeInterior(
            thought_id=row["thought_id"], content=row["content"], status=row["status"],
            created_at=_dt(row["created_at"]), updated_at=_dt(row["updated_at"]),
            revisit_at=_dt(row["revisit_at"]), source_activity_id=row["source_activity_id"],
        )

    def due_thoughts(self, now: datetime) -> list[LifeInterior]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT thought_id FROM life_interior
                   WHERE status IN ('private','candidate') AND revisit_at IS NOT NULL
                   AND revisit_at <= ? ORDER BY revisit_at, thought_id""",
                (_iso(now),),
            ).fetchall()
        return [self.get_thought(row["thought_id"]) for row in rows]

    def transition_thought(
        self,
        thought_id: str,
        status: InteriorStatus,
        *,
        now: datetime,
        content: str | None = None,
        revisit_at: datetime | None = None,
    ) -> LifeInterior:
        current = self.get_thought(thought_id)
        if status != current.status and status not in INTERIOR_TRANSITIONS[current.status]:
            raise ValueError(f"illegal interior transition: {current.status} -> {status}")
        with self._connect() as connection:
            connection.execute(
                """UPDATE life_interior SET status = ?, content = ?, updated_at = ?,
                   revisit_at = ? WHERE thought_id = ?""",
                (status, content if content is not None else current.content,
                 _iso(now), _iso(revisit_at), thought_id),
            )
            self._refresh_meaningful_deadline(connection, now)
        return self.get_thought(thought_id)

    def ledger(
        self,
        event_type: str,
        reason: str,
        *,
        now: datetime,
        activity_id: str | None = None,
        thought_id: str | None = None,
        hermes_session_id: str | None = None,
        usage: dict[str, int] | None = None,
        usage_unknown: bool = False,
        receipt_status: str = "completed",
        details: dict[str, Any] | None = None,
    ) -> int:
        usage = usage or {}
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO life_ledger(occurred_at, event_type, reason, activity_id,
                   thought_id, hermes_session_id, api_calls, input_tokens, output_tokens,
                   total_tokens, usage_unknown, receipt_status, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (_iso(now), event_type, reason, activity_id, thought_id, hermes_session_id,
                 usage.get("api_calls"), usage.get("input_tokens"), usage.get("output_tokens"),
                 usage.get("total_tokens"), int(usage_unknown), receipt_status,
                 json.dumps(details or {}, ensure_ascii=False, default=str)),
            )
            return int(cursor.lastrowid)

    def update_receipt(
        self,
        ledger_id: int,
        *,
        receipt_status: str,
        now: datetime,
        hermes_session_id: str | None = None,
        usage: dict[str, int] | None = None,
        usage_unknown: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        usage = usage or {}
        with self._connect() as connection:
            connection.execute(
                """UPDATE life_ledger SET receipt_status = ?, hermes_session_id = ?,
                   api_calls = ?, input_tokens = ?, output_tokens = ?, total_tokens = ?,
                   usage_unknown = ?, details = ? WHERE ledger_id = ?""",
                (receipt_status, hermes_session_id, usage.get("api_calls"),
                 usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"),
                 int(usage_unknown), json.dumps({"updated_at": _iso(now), **(details or {})},
                                                ensure_ascii=False, default=str), ledger_id),
            )

    def ledger_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM life_ledger ORDER BY ledger_id").fetchall()
        return [dict(row) for row in rows]
