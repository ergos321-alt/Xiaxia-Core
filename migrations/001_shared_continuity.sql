BEGIN;

CREATE TABLE IF NOT EXISTS continuity_events (
    event_id uuid PRIMARY KEY,
    xiaxia_identity_id text NOT NULL CHECK (char_length(xiaxia_identity_id) BETWEEN 1 AND 128),
    user_identity_id text NOT NULL CHECK (char_length(user_identity_id) BETWEEN 1 AND 128),
    source text NOT NULL CHECK (source IN (
        'app', 'custom_gpt', 'browser', 'core', 'reading_house',
        'diary_house', 'watch_house', 'proactive_runtime'
    )),
    event_type text NOT NULL CHECK (event_type IN (
        'conversation_event', 'browser_event', 'tool_event', 'topic_started',
        'topic_updated', 'unfinished_topic', 'relationship_state',
        'memory_candidate', 'house_event', 'system_event'
    )),
    summary text NOT NULL CHECK (char_length(summary) BETWEEN 1 AND 1000),
    occurred_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    importance smallint NOT NULL DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'archived')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    request_id text CHECK (request_id IS NULL OR char_length(request_id) <= 128),
    conversation_id text CHECK (conversation_id IS NULL OR char_length(conversation_id) <= 128),
    source_event_id text CHECK (source_event_id IS NULL OR char_length(source_event_id) <= 200)
);

CREATE UNIQUE INDEX IF NOT EXISTS continuity_events_idempotency_idx
ON continuity_events (xiaxia_identity_id, user_identity_id, source, source_event_id)
WHERE source_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS continuity_events_timeline_idx
ON continuity_events (xiaxia_identity_id, user_identity_id, occurred_at DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS continuity_events_source_idx
ON continuity_events (xiaxia_identity_id, user_identity_id, source, occurred_at DESC);

CREATE INDEX IF NOT EXISTS continuity_events_type_idx
ON continuity_events (xiaxia_identity_id, user_identity_id, event_type, occurred_at DESC);

CREATE INDEX IF NOT EXISTS continuity_events_active_idx
ON continuity_events (xiaxia_identity_id, user_identity_id, status, importance DESC, occurred_at DESC);

COMMIT;
