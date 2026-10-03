from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class StoredMessage:
    role: Literal["user", "assistant"]
    content: str


class RuntimeStore:
    """SQLite storage for Core conversation state only."""

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
        with self._write_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                """
            )

    def ensure_conversation(self, conversation_id: str | None = None) -> str:
        identifier = conversation_id or str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (identifier, now, now),
            )
        return identifier

    def get_messages(self, conversation_id: str, limit: int) -> list[StoredMessage]:
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()
        return [StoredMessage(role=row["role"], content=row["content"]) for row in rows]

    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (conversation_id, user, now),
            )
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                (conversation_id, assistant, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def healthy(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False
