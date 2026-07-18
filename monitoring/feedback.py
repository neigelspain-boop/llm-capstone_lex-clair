"""Writes queries and user feedback to Postgres. Grafana reads from the
same tables to build the monitoring dashboard (see monitoring/grafana/).
"""
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
    return psycopg.connect(settings.postgres_dsn)


def init_schema() -> None:
    with _connect() as conn:
        conn.execute(SCHEMA)


def record_query(question: str, answer: str) -> str:
    query_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO queries (id, question, answer) VALUES (%s, %s, %s)",
            (query_id, question, answer),
        )
    return query_id


def record_feedback(query_id: str, is_positive: bool) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (query_id, is_positive) VALUES (%s, %s)",
            (query_id, is_positive),
        )
