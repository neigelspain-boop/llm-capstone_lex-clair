"""SQLite verification cache, keyed on (pass, prompt_version, file, code_hash).

Replaces deletion_audit.py's append-only .processed_sections.txt, which had
no way to invalidate itself: editing the prompt silently left stale
"already processed" entries in place forever, requiring a manual delete.
Here, prompt_version and code_hash are part of the cache key itself, so
bumping config.PROMPT_VERSIONS or editing the source file both naturally
miss the cache and get reprocessed — no manual step required.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from scripts.local_audit import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS verifications (
    cache_key TEXT PRIMARY KEY,
    pass_name TEXT NOT NULL,
    file TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    verified_at_sha TEXT,
    ts TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ========== connection ==========


def _connect() -> sqlite3.Connection:
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.CACHE_DB)
    conn.execute(SCHEMA)
    return conn


# ========== key derivation ==========


def code_hash_for_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def code_hash_for_file(path: Path) -> str:
    if not path.is_file():
        # covers both "doesn't exist" and "exists but is a directory" — a
        # doc claim's candidate_files can legitimately name a package dir
        # (e.g. "eval") rather than a specific module.
        return "missing"
    try:
        return code_hash_for_text(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return "missing"


def make_key(pass_name: str, prompt_version: str, file_path: str, code_hash: str) -> str:
    raw = f"{pass_name}|{prompt_version}|{file_path}|{code_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ========== read / write ==========


def get(pass_name: str, prompt_version: str, file_path: str, code_hash: str) -> dict | None:
    key = make_key(pass_name, prompt_version, file_path, code_hash)
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT result_json FROM verifications WHERE cache_key = ?", (key,)
        ).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def set(pass_name: str, prompt_version: str, file_path: str, code_hash: str,
        result: dict, verified_at_sha: str | None = None) -> None:
    key = make_key(pass_name, prompt_version, file_path, code_hash)
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO verifications
                   (cache_key, pass_name, file, prompt_version, code_hash,
                    result_json, verified_at_sha)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                   result_json = excluded.result_json,
                   verified_at_sha = excluded.verified_at_sha,
                   ts = datetime('now')""",
            (key, pass_name, file_path, prompt_version, code_hash,
             json.dumps(result, ensure_ascii=False), verified_at_sha),
        )
        conn.commit()
    finally:
        conn.close()
