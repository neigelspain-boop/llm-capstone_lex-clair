"""Post-process compliance_matrix.json to add persons_named per entry.

Zero LLM calls. Deterministic join: for each compliance entry's role_id,
find all case-scoped person_ids whose role_assignments include that role_id.
Emit persons_named list on the entry.

Preserves all existing fields byte-identically. Overwrites in place after
atomic .tmp + rename.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def enrich(case_id: str) -> dict:
    case_dir = Path(f"data/dossier/{case_id}")
    matrix_path = case_dir / "compliance_matrix.json"
    persons_path = case_dir / "persons.jsonl"

    if not matrix_path.exists():
        sys.exit(f"missing {matrix_path}")
    if not persons_path.exists():
        sys.exit(f"missing {persons_path} — run mentions + resolve first")

    matrix = json.loads(matrix_path.read_text())
    persons = [json.loads(l) for l in persons_path.read_text().splitlines() if l.strip()]

    # Build role_id -> [{person_id, canonical_name, confidence}] map
    role_to_persons: dict[str, list[dict]] = {}
    for p in persons:
        for ra in p.get("role_assignments", []):
            role_id = ra["role_id"]
            role_to_persons.setdefault(role_id, []).append({
                "person_id": p["person_id"],
                "canonical_name": p["canonical_name"],
                "confidence": p.get("confidence", "unknown"),
                "ambiguity_note": p.get("ambiguity_note"),
            })

    # Enrich each entry
    enriched_count = 0
    for entry in matrix.get("entries", []):
        role_id = entry.get("actor_role")
        persons_named = role_to_persons.get(role_id, [])
        # Exclude low-confidence persons from breached/met to preserve
        # fiability (matches ADR #53's rule even though we're not re-running).
        if entry.get("status") in ("breached", "met"):
            persons_named = [p for p in persons_named if p["confidence"] != "low"]
        entry["persons_named"] = persons_named
        if persons_named:
            enriched_count += 1

    # Atomic write
    tmp_path = matrix_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2))
    os.replace(tmp_path, matrix_path)

    return {
        "case_id": case_id,
        "total_entries": len(matrix.get("entries", [])),
        "entries_with_persons": enriched_count,
        "unique_roles_matched": len(set(e.get("actor_role") for e in matrix.get("entries", []) if e.get("persons_named"))),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    args = ap.parse_args()
    summary = enrich(args.case_id)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
