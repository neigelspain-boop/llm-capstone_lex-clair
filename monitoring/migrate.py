"""One-shot migration: Day 6 JSON + CSV artifacts → Day 7 Postgres.

Reads `data/conversations/*.json` (Day 6 conversation persistence, ADR #31)
and `data/feedback.csv` (Day 6 feedback log, ADR #28) and inserts them into
the Postgres tables owned by monitoring.db.

Idempotent by design (safe to re-run):
    - Conversations + turns use db.save_conversation() UPSERT semantics
    - Feedback rows checked against (turn_id, created_at) natural key
      before insert to prevent duplicates. Runtime feedback added via
      the UI between migrations is preserved untouched.

Silent-fallback per-row error handling (Day 3 doctrine):
    - Per-file try/except: log, skip, continue
    - Per-row try/except: log, skip, continue
    - End-of-run summary: files_read, conversations, turns, feedback
      imported / skipped-existing / errors
    - Drift check: if JSON turn.feedback count != CSV feedback rows,
      print a warning (no auto-repair — Day 9 buffer if it happens)

Usage:
    uv run python -m monitoring.migrate           # dry-run (default)
    uv run python -m monitoring.migrate --apply   # commit inserts

Reads DB config from env via monitoring.db; no schema of its own.
"""

from __future__ import annotations

# ========== imports + config ==========

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import DictCursor

from monitoring import db

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERSATIONS_DIR = ROOT / "data" / "conversations"
DEFAULT_FEEDBACK_CSV = ROOT / "data" / "feedback.csv"


# ========== conversation import ==========

def _load_conversation_files(conv_dir: Path) -> list[dict]:
    """Read every {uuid}.json in the conversations directory.

    Malformed files are logged and skipped, not fatal. Returns list in
    sorted-by-filename order for deterministic behaviour across runs.
    """
    if not conv_dir.exists():
        print(f"[warn] {conv_dir} does not exist — no conversations to migrate")
        return []

    conversations = []
    for path in sorted(conv_dir.glob("*.json")):
        try:
            with path.open() as f:
                conv = json.load(f)
            if not isinstance(conv, dict) or "id" not in conv:
                print(f"[warn] {path.name}: malformed (missing 'id'), skipping")
                continue
            conversations.append(conv)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] {path.name}: read failed ({e}), skipping")

    return conversations


def _import_conversations(conversations: list[dict], apply: bool) -> dict:
    """Import conversations + turns via db.save_conversation UPSERT.

    UPSERT means re-running is idempotent — same input yields same rows.
    Errors on individual conversations are logged and skipped (do not
    halt the migration).
    """
    stats = {"conversations": 0, "turns": 0, "errors": 0}

    for conv in conversations:
        try:
            n_turns = len(conv.get("turns", []))
            if apply:
                db.save_conversation(conv)
            stats["conversations"] += 1
            stats["turns"] += n_turns
        except Exception as e:
            stats["errors"] += 1
            print(f"[error] conv {conv.get('id', '?')}: {e}")
            traceback.print_exc(file=sys.stderr)

    return stats


# ========== feedback import ==========

def _load_feedback_csv(csv_path: Path) -> list[dict]:
    """Read Day 6's feedback.csv using stdlib csv (not pandas).

    Stdlib avoids the pandas NaN trap that would silently convert empty
    `comment` cells to NaN and break the DB insert. Project-wide
    convention on lex-clair CSVs.
    """
    if not csv_path.exists():
        print(f"[warn] {csv_path} does not exist — no feedback to migrate")
        return []

    rows = []
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except (OSError, csv.Error) as e:
        print(f"[error] failed to read {csv_path}: {e}")
        return []

    return rows


def _fetch_existing_feedback_keys() -> set[tuple[str, datetime]]:
    """Return (turn_id, created_at) tuples already present in feedback.

    Used to skip duplicate inserts on re-run without touching non-migration
    runtime data — feedback added via the UI between migrations is
    preserved unchanged.
    """
    conn = db.get_conn()
    if conn is None:
        return set()

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT turn_id, created_at FROM feedback")
            return {(r["turn_id"], r["created_at"]) for r in cur.fetchall()}
    except psycopg2.Error as e:
        print(f"[warn] could not fetch existing feedback keys: {e}")
        return set()


def _parse_csv_timestamp(s: str) -> datetime:
    """Parse Day 6's ISO timestamp string.

    Day 6 writes `datetime.now(timezone.utc).isoformat()` which produces
    '+00:00' offset (not 'Z'). The .replace() is defensive but harmless.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _import_feedback(rows: list[dict], apply: bool) -> dict:
    """Import feedback rows, deduplicating against existing (turn_id, ts) keys.

    In dry-run mode, the existing set is still updated in-memory so the
    reported "imported" count reflects true net-new inserts after dedup,
    matching what --apply would actually do.
    """
    stats = {"imported": 0, "skipped_existing": 0, "errors": 0}

    existing = _fetch_existing_feedback_keys()

    for row in rows:
        try:
            ts = _parse_csv_timestamp(row["timestamp"])
            key = (row["turn_id"], ts)

            if key in existing:
                stats["skipped_existing"] += 1
                continue

            if apply:
                db.append_feedback(row, created_at=ts)

            existing.add(key)  # dedup within the CSV itself
            stats["imported"] += 1
        except (KeyError, ValueError) as e:
            stats["errors"] += 1
            print(f"[error] feedback row {row.get('turn_id', '?')}: {e}")

    return stats


# ========== drift check (JSON turn.feedback vs CSV feedback rows) ==========

def _drift_check(conversations: list[dict], fb_stats: dict) -> None:
    """Warn if JSON turn.feedback count differs from CSV feedback row count.

    Day 6 dual-wrote feedback to both the turn dict and the CSV. If they
    drift, some UI-visible feedback won't appear in Grafana. This is a
    diagnostic-only check — no auto-repair. Deferred to Day 9 buffer if
    it ever fires.
    """
    json_feedback_count = sum(
        1
        for conv in conversations
        for turn in conv.get("turns", [])
        if turn.get("feedback") is not None
    )
    csv_feedback_count = fb_stats["imported"] + fb_stats["skipped_existing"]

    if json_feedback_count != csv_feedback_count:
        print()
        print(
            f"[warn] drift detected: JSON turn.feedback count = "
            f"{json_feedback_count}, CSV feedback rows = {csv_feedback_count}"
        )
        print(
            "       Some UI feedback may not appear in Grafana. "
            "Investigate manually if this matters."
        )


# ========== CLI entry point ==========

def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot migration: Day 6 JSON+CSV → Postgres."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually INSERT into Postgres. Default is dry-run (read + count only).",
    )
    parser.add_argument(
        "--conversations-dir", type=Path, default=DEFAULT_CONVERSATIONS_DIR,
        help=f"Directory of conversation *.json files "
             f"(default: {DEFAULT_CONVERSATIONS_DIR.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--feedback-csv", type=Path, default=DEFAULT_FEEDBACK_CSV,
        help=f"Path to feedback.csv "
             f"(default: {DEFAULT_FEEDBACK_CSV.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== monitoring.migrate — {mode} ===")
    print(f"conversations: {args.conversations_dir}")
    print(f"feedback:      {args.feedback_csv}")
    print()

    # Kill switch — refuse to APPLY without a live DB connection
    if args.apply:
        db.init_schema()
        if not db.is_healthy():
            print("[fatal] Postgres unreachable — kill switch tripped.")
            print("        Check `docker compose ps` and .env credentials.")
            return 1

    # Import conversations + turns
    conversations = _load_conversation_files(args.conversations_dir)
    print(f"[read] {len(conversations)} conversation file(s)")

    conv_stats = _import_conversations(conversations, apply=args.apply)

    # Import feedback rows
    feedback_rows = _load_feedback_csv(args.feedback_csv)
    print(f"[read] {len(feedback_rows)} feedback row(s)")

    fb_stats = _import_feedback(feedback_rows, apply=args.apply)

    # Cross-source drift diagnostic
    _drift_check(conversations, fb_stats)

    # End-of-run summary
    print()
    print(f"=== summary ({mode}) ===")
    print(f"conversations processed: {conv_stats['conversations']}")
    print(f"turns processed:         {conv_stats['turns']}")
    print(f"feedback imported:       {fb_stats['imported']}")
    print(f"feedback skipped (already present): {fb_stats['skipped_existing']}")
    print(f"errors:                  {conv_stats['errors'] + fb_stats['errors']}")

    if not args.apply:
        print()
        print("(dry-run — no rows were actually inserted)")
        print("Re-run with --apply to commit.")

    return 0 if (conv_stats["errors"] + fb_stats["errors"]) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())