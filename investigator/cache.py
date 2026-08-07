"""Per-case SQLite verification cache, keyed (pass, prompt_version, subject, hash).

Forked from `scripts/local_audit/cache.py` for the path reason in ADR #70; the
key formula is byte-identical and a test asserts it.

Two rules the callers must honour, both learned in the original harness:

- **The `subject` slot is an arbitrary key**, not a filesystem path. Passes
  overload it (`{obligation_id}@{rule_version}|{case_id}|{fact_id}`,
  `piste|{legiarti_id}`) so cache identity can be finer or coarser than a file.
  It deliberately excludes line numbers, so an edit elsewhere in a document
  does not burn a verdict.
- **Never cache a `None` verdict.** A transport failure or an unparseable
  response is "no verdict", not "no". Caching one turns a blip into a permanent
  wrong answer; leaving it uncached means the next cycle simply retries.

Invalidation is entirely via `config.PROMPT_VERSIONS` plus each obligation's
own `rule_version` riding in the subject slot, so editing one obligation's rule
does not reprocess the others.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from investigator.config import CasePaths

SCHEMA = """
CREATE TABLE IF NOT EXISTS verifications (
    cache_key TEXT PRIMARY KEY,
    pass_name TEXT NOT NULL,
    subject TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    verified_at_sha TEXT,
    ts TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ========== connection ==========


def _connect(paths: CasePaths) -> sqlite3.Connection:
    paths.investigation_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(paths.cache_db)
    conn.execute(SCHEMA)
    return conn


# ========== key derivation ==========


def content_hash_for_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def content_hash_for_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return content_hash_for_text(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return "missing"


def make_key(pass_name: str, prompt_version: str, subject: str, content_hash: str) -> str:
    raw = f"{pass_name}|{prompt_version}|{subject}|{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ========== read / write ==========


def get(
    paths: CasePaths, pass_name: str, prompt_version: str, subject: str, content_hash: str
) -> dict | None:
    key = make_key(pass_name, prompt_version, subject, content_hash)
    conn = _connect(paths)
    try:
        row = conn.execute(
            "SELECT result_json FROM verifications WHERE cache_key = ?", (key,)
        ).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def set(
    paths: CasePaths,
    pass_name: str,
    prompt_version: str,
    subject: str,
    content_hash: str,
    result: dict,
    verified_at_sha: str | None = None,
) -> None:
    """Commit one verdict.

    Committed and closed per call rather than batched: that is what bounds a
    kill to at most one in-flight unit of work.
    """
    key = make_key(pass_name, prompt_version, subject, content_hash)
    conn = _connect(paths)
    try:
        conn.execute(
            """INSERT INTO verifications
                   (cache_key, pass_name, subject, prompt_version, content_hash,
                    result_json, verified_at_sha)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                   result_json = excluded.result_json,
                   verified_at_sha = excluded.verified_at_sha,
                   ts = datetime('now')""",
            (
                key,
                pass_name,
                subject,
                prompt_version,
                content_hash,
                json.dumps(result, ensure_ascii=False),
                verified_at_sha,
            ),
        )
        conn.commit()
    finally:
        conn.close()
