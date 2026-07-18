"""Write user queries and feedback into Postgres for dashboard monitoring."""
import uuid

import psycopg

from src.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_id UUID NOT NULL REFERENCES queries(id),
    is_positive BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _connect():
    """Open a new database connection using configured Postgres DSN."""
    return psycopg.connect(settings.postgres_dsn)


def init_schema() -> None:
    """Create monitoring tables if they do not already exist."""
    with _connect() as conn:
        conn.execute(SCHEMA)


def record_query(question: str, answer: str) -> str:
    """Store a user query and its answer, returning the generated query id."""
    query_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO queries (id, question, answer) VALUES (%s, %s, %s)",
            (query_id, question, answer),
        )
    return query_id


def record_feedback(query_id: str, is_positive: bool) -> None:
    """Store positive/negative feedback linked to a previously recorded query."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (query_id, is_positive) VALUES (%s, %s)",
            (query_id, is_positive),
        )
