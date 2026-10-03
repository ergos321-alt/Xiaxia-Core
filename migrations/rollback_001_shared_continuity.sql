BEGIN;

-- This rollback owns only Shared Continuity V1 objects. It never touches
-- Identity, Memory/Mem0, conversation history, or Browser state.
DROP TABLE IF EXISTS continuity_events;

COMMIT;
