-- Additive Phase 2A schema. This migration does not modify Continuity data.
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS execution_requests (
    request_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    task_kind TEXT NOT NULL DEFAULT 'none',
    task_status TEXT NOT NULL DEFAULT 'not_executed',
    chat_status TEXT NOT NULL DEFAULT 'in_progress',
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_execution_requests_conversation
    ON execution_requests(conversation_id, started_at, request_id);

CREATE TABLE IF NOT EXISTS execution_subgoals (
    action_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    original_subgoal_json TEXT NOT NULL,
    normalized_action TEXT NOT NULL,
    target_json TEXT,
    destination TEXT,
    required INTEGER NOT NULL,
    polarity TEXT NOT NULL,
    condition_json TEXT,
    dependency_action_id TEXT,
    source TEXT NOT NULL DEFAULT 'required_task',
    created_at TEXT NOT NULL,
    FOREIGN KEY(request_id) REFERENCES execution_requests(request_id),
    FOREIGN KEY(dependency_action_id) REFERENCES execution_subgoals(action_id),
    UNIQUE(request_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_execution_subgoals_request
    ON execution_subgoals(request_id, sequence);

CREATE TABLE IF NOT EXISTS execution_attempts (
    attempt_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    action_sequence INTEGER NOT NULL,
    call_sequence INTEGER NOT NULL,
    attempt_number INTEGER NOT NULL,
    normalized_action TEXT NOT NULL,
    target_json TEXT,
    destination TEXT,
    tab_id TEXT,
    page_identity TEXT,
    page_generation INTEGER,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    actual_browser_call_json TEXT NOT NULL,
    raw_browser_result_json TEXT,
    normalized_result_json TEXT,
    verification_state TEXT NOT NULL,
    error_code TEXT,
    partial_blocked INTEGER NOT NULL DEFAULT 0,
    actual_final_url TEXT,
    actual_title TEXT,
    evidence_reference TEXT,
    FOREIGN KEY(request_id) REFERENCES execution_requests(request_id),
    FOREIGN KEY(action_id) REFERENCES execution_subgoals(action_id),
    UNIQUE(request_id, call_sequence)
);

CREATE INDEX IF NOT EXISTS idx_execution_attempts_request
    ON execution_attempts(request_id, call_sequence);

COMMIT;
