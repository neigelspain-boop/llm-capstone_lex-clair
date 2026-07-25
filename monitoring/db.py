"""Postgres persistence for lex-clair conversations, turns, and feedback.

Replaces the JSON+CSV persistence from Day 6 (ADRs #28, #31). Called
exclusively from app/streamlit_app.py; the RAG hot path (Plane II) has
no knowledge of this module, preserving ADR #10.

Schema locked in ADR #32. Kill-switch pattern locked in ADR #34
(Option B: module-level global _DB_HEALTHY, not Streamlit session state).

Public API (must stay stable — files 3, 4, 5, 8 depend on it):
    get_conn()               — lazy singleton, None if kill switch tripped
    is_healthy()             — kill switch check for the UI banner
    init_schema()            — idempotent CREATE TABLE IF NOT EXISTS
    save_conversation(conv)  — UPSERT conversation + turns
    load_all_conversations() — reconstruct Day 6 in-memory shape
    append_feedback(row)     — INSERT one feedback row

Silent-fallback contract (Day 5 doctrine):
    - Writes swallow-and-log on error; UI stays usable
    - Reads return empty dict/list on error
    - OperationalError flips kill switch; subsequent calls short-circuit
"""

from __future__ import annotations

# ========== imports + config = ==========

import json
import logging
import os
from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor

logger = logging.getLogger(__name__)

# Env-driven config. Caller (app/streamlit_app.py) loads .env on boot;
# db.py does not call load_dotenv itself to avoid double-loading.
_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
_DB = os.getenv("POSTGRES_DB", "lexclair")
_USER = os.getenv("POSTGRES_USER", "lexclair")
_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lexclair")


# ========== connection factory (lazy singleton + Option B kill switch) ==========

_CONN: psycopg2.extensions.connection | None = None
_DB_HEALTHY: bool = True


def is_healthy() -> bool:
    """Kill switch check. Streamlit calls this to render the warning banner."""
    return _DB_HEALTHY


def get_conn() -> psycopg2.extensions.connection | None:
    """Return a live connection, opening it lazily on first call.

    Returns None if the kill switch has tripped. Automatically reconnects
    if the cached connection was closed externally (e.g. Postgres restart
    during development).
    """
    global _CONN, _DB_HEALTHY

    if not _DB_HEALTHY:
        return None

    if _CONN is not None and _CONN.closed == 0:
        return _CONN

    try:
        _CONN = psycopg2.connect(
            host=_HOST,
            port=_PORT,
            dbname=_DB,
            user=_USER,
            password=_PASSWORD,
        )
        return _CONN
    except psycopg2.OperationalError as e:
        _DB_HEALTHY = False
        logger.warning(
            "Postgres unreachable at %s:%s — kill switch tripped. "
            "UI will degrade to in-memory only. Error: %s",
            _HOST, _PORT, e,
        )
        return None


def _reset_conn() -> None:
    """Test-only helper — force reconnect on next get_conn() call.

    Not part of the runtime API. Used by tests/test_ingestion_smoke.py
    and manual recovery after a Postgres restart during development.
    """
    global _CONN, _DB_HEALTHY
    if _CONN is not None:
        try:
            _CONN.close()
        except Exception:
            pass
    _CONN = None
    _DB_HEALTHY = True


# ========== schema init (idempotent) ==========

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT,
    answer_en TEXT,
    rewritten_query TEXT,
    chunks_retrieved INTEGER,
    chunks_reranked INTEGER,
    model_used TEXT,
    cost_usd DOUBLE PRECISION,
    elapsed_seconds DOUBLE PRECISION,
    citations TEXT,
    error BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation_id ON turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_turns_created_at ON turns(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_turn_id ON feedback(turn_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
"""


def init_schema() -> None:
    """Create tables + indexes if they don't exist. Safe to call every session start.

    First call is the true init; subsequent calls no-op via IF NOT EXISTS.
    Silently returns if the kill switch tripped — UI banner surfaces it.
    """
    conn = get_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_DDL)
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        logger.warning("init_schema failed: %s", e)


# ========== conversation CRUD ==========

def _parse_iso(value) -> datetime:
    """Accept a datetime or ISO-8601 string; return a tz-aware datetime.

    Day 6 stores timestamps as ISO strings in the JSON files and session
    state. Postgres wants datetimes. This bridge accepts both.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Python 3.11+ handles most ISO variants; normalise 'Z' for older versions.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"Cannot parse timestamp: {value!r}")


def save_conversation(conv: dict) -> None:
    """UPSERT one conversation and all its turns. Called at every mutation.

    Replaces _save_conversation() from Day 6's app/streamlit_app.py. Called
    on: new turn appended, flow.run completion, translation cache populated,
    feedback captured.

    Turn-level UPSERT (not delete-and-reinsert) is REQUIRED — feedback rows
    reference turn_id with ON DELETE CASCADE. Wiping turns to reinsert them
    would silently nuke all feedback. UPSERT preserves the FK link.
    """
    global _DB_HEALTHY
    conn = get_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            # Upsert the conversation row itself
            cur.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    conv["id"],
                    conv["title"],
                    _parse_iso(conv["created_at"]),
                    _parse_iso(conv["updated_at"]),
                ),
            )

            # Upsert each turn — see docstring for why UPSERT vs delete+reinsert
            for turn in conv.get("turns", []):
                result = turn.get("result") or {}
                cur.execute(
                    """
                    INSERT INTO turns (
                        id, conversation_id, question, answer, answer_en,
                        rewritten_query, chunks_retrieved, chunks_reranked,
                        model_used, cost_usd, elapsed_seconds,
                        citations, error
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        answer = EXCLUDED.answer,
                        answer_en = EXCLUDED.answer_en,
                        rewritten_query = EXCLUDED.rewritten_query,
                        chunks_retrieved = EXCLUDED.chunks_retrieved,
                        chunks_reranked = EXCLUDED.chunks_reranked,
                        model_used = EXCLUDED.model_used,
                        cost_usd = EXCLUDED.cost_usd,
                        elapsed_seconds = EXCLUDED.elapsed_seconds,
                        citations = EXCLUDED.citations,
                        error = EXCLUDED.error
                    """,
                    (
                        turn["turn_id"],
                        conv["id"],
                        turn["question"],
                        result.get("answer"),
                        turn.get("answer_en"),
                        result.get("rewritten_query"),
                        result.get("chunks_retrieved"),
                        result.get("chunks_reranked"),
                        result.get("model_used"),
                        result.get("cost_usd"),
                        result.get("elapsed_seconds"),
                        json.dumps(result.get("citations") or []),
                        bool(result.get("_error", False)),
                    ),
                )
        conn.commit()
    except psycopg2.OperationalError as e:
        _DB_HEALTHY = False
        try:
            conn.rollback()
        except Exception:
            pass
        logger.warning("save_conversation lost DB connection: %s", e)
    except psycopg2.Error as e:
        conn.rollback()
        logger.warning("save_conversation error: %s", e)


def load_all_conversations() -> dict[str, dict]:
    """Return all conversations keyed by id, in the Day 6 in-memory shape.

    Replaces _load_all_conversations() from Day 6. Reconstructs the exact
    dict shape ADR #31 locked, including the nested `result` dict rebuilt
    from columns — so rendering code in app/streamlit_app.py stays unchanged.

    Feedback is loaded in ONE query (DISTINCT ON), not N+1 per-turn queries.
    """
    conn = get_conn()
    if conn is None:
        return {}
    try:
        result: dict[str, dict] = {}
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Conversations
            cur.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
            """)
            for row in cur.fetchall():
                result[row["id"]] = {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "turns": [],
                }

            # Latest feedback per turn — one query, indexed on turn_id
            cur.execute("""
                SELECT DISTINCT ON (turn_id) turn_id, rating
                FROM feedback
                ORDER BY turn_id, created_at DESC
            """)
            feedback_by_turn = {r["turn_id"]: r["rating"] for r in cur.fetchall()}

            # Turns, in creation order per conversation
            cur.execute("""
                SELECT id, conversation_id, question, answer, answer_en,
                       rewritten_query, chunks_retrieved, chunks_reranked,
                       model_used, cost_usd, elapsed_seconds,
                       citations, error, created_at
                FROM turns
                ORDER BY created_at ASC
            """)
            for row in cur.fetchall():
                conv_id = row["conversation_id"]
                if conv_id not in result:
                    continue  # orphan turn (shouldn't happen with FK, defensive)

                # Reconstruct the flow.run() result dict shape.
                # A turn with no answer AND no error means flow hasn't run yet
                # → result=None matches Day 6's pre-execution state.
                if row["answer"] is not None or row["error"]:
                    result_dict = {
                        "answer": row["answer"],
                        "rewritten_query": row["rewritten_query"],
                        "chunks_retrieved": row["chunks_retrieved"],
                        "chunks_reranked": row["chunks_reranked"],
                        "model_used": row["model_used"],
                        "cost_usd": row["cost_usd"],
                        "elapsed_seconds": row["elapsed_seconds"],
                        "citations": json.loads(row["citations"]) if row["citations"] else [],
                    }
                    if row["error"]:
                        result_dict["_error"] = True
                else:
                    result_dict = None

                result[conv_id]["turns"].append({
                    "turn_id": row["id"],
                    "question": row["question"],
                    "result": result_dict,
                    "answer_en": row["answer_en"],
                    "feedback": feedback_by_turn.get(row["id"]),
                })

        return result
    except psycopg2.Error as e:
        logger.warning("load_all_conversations error: %s", e)
        return {}


# ========== feedback CRUD ==========

def append_feedback(row: dict, created_at: datetime | None = None) -> None:
    """Insert one feedback row. Replaces _append_feedback() from Day 6.

    `row` is the 10-column dict Day 6 sent to the CSV; only 4 columns
    (conversation_id, turn_id, rating, comment) map to DB storage —
    the rest are duplicative of the turns table.

    `created_at=None` uses DB DEFAULT NOW() — normal runtime path.
    `created_at=<datetime>` sets historical timestamp — used by migrate.py
    to preserve original CSV timestamps during import.
    """
    global _DB_HEALTHY
    conn = get_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            if created_at is None:
                cur.execute(
                    """
                    INSERT INTO feedback (conversation_id, turn_id, rating, comment)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        row["conversation_id"],
                        row["turn_id"],
                        int(row["rating"]),
                        row.get("comment") or None,
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO feedback
                        (conversation_id, turn_id, rating, comment, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        row["conversation_id"],
                        row["turn_id"],
                        int(row["rating"]),
                        row.get("comment") or None,
                        created_at,
                    ),
                )
        conn.commit()
    except psycopg2.OperationalError as e:
        _DB_HEALTHY = False
        try:
            conn.rollback()
        except Exception:
            pass
        logger.warning("append_feedback lost DB connection: %s", e)
    except psycopg2.Error as e:
        conn.rollback()
        logger.warning("append_feedback error: %s", e)