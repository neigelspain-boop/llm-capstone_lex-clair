"""Structured findings store: audit/findings.jsonl, id-keyed upsert.

Replaces deletion_audit.py's per-file Markdown pages + append-only INDEX.md,
neither of which could reflect a finding being fixed later — this store can,
via reconcile_pass(), which flips findings a pass no longer reproduces to
status="resolved" instead of leaving a stale entry forever.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from scripts.local_audit import config

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


# ========== id derivation ==========


def make_id(pass_name: str, file: str, claim: str) -> str:
    raw = f"{pass_name}|{file}|{claim}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ========== load / save ==========


def load_all() -> dict[str, dict]:
    if not config.FINDINGS_JSONL.exists():
        return {}
    store: dict[str, dict] = {}
    for line in config.FINDINGS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in obj:
            store[obj["id"]] = obj
    return store


def save_all(store: dict[str, dict]) -> None:
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(store[fid], ensure_ascii=False, sort_keys=True)
        for fid in sorted(store)
    ]
    config.FINDINGS_JSONL.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ========== upsert ==========


def normalize(raw: dict, verified_at_sha: str | None = None,
              prompt_version: str | None = None,
              self_consistency: dict | None = None) -> dict:
    """Fill in the full findings.jsonl schema from a pass's raw finding dict
    (as produced by conventions.py/passes/*.py: pass, file, line, claim,
    evidence, severity, confidence, related_files).
    """
    fid = make_id(raw["pass"], raw["file"], raw["claim"])
    now = _now()
    return {
        "id": fid,
        "pass": raw["pass"],
        "severity": raw.get("severity", "info"),
        "confidence": raw.get("confidence", "low"),
        "file": raw["file"],
        "related_files": raw.get("related_files", []),
        "line": raw.get("line", 0),
        "claim": raw["claim"],
        "evidence": raw.get("evidence", ""),
        "verified_at_sha": verified_at_sha,
        "prompt_version": prompt_version,
        "self_consistency": self_consistency,
        "status": "open",
        "first_seen": now,
        "last_seen": now,
    }


def upsert_many(raw_findings: list[dict], verified_at_sha: str | None = None,
                 prompt_version: str | None = None) -> dict[str, dict]:
    """Merge a batch of raw finding dicts into the store, preserving
    first_seen for findings that already existed and reopening any that had
    previously been marked resolved but reappeared.
    """
    store = load_all()
    now = _now()
    for raw in raw_findings:
        normalized = normalize(raw, verified_at_sha, prompt_version)
        fid = normalized["id"]
        if fid in store:
            existing = store[fid]
            existing.update({
                "severity": normalized["severity"],
                "confidence": normalized["confidence"],
                "related_files": normalized["related_files"],
                "line": normalized["line"],
                "evidence": normalized["evidence"],
                "verified_at_sha": normalized["verified_at_sha"],
                "prompt_version": normalized["prompt_version"],
                "last_seen": now,
                "status": "open",
            })
        else:
            store[fid] = normalized
    save_all(store)
    return store


def reconcile_pass(pass_name: str, current_ids: set[str]) -> None:
    """Findings previously emitted by `pass_name` but not in current_ids
    (i.e. this cycle's run no longer reproduces them) flip to
    status="resolved" rather than being deleted or left stale-but-open.
    """
    store = load_all()
    changed = False
    for fid, f in store.items():
        if f.get("pass") == pass_name and f.get("status") == "open" and fid not in current_ids:
            f["status"] = "resolved"
            changed = True
    if changed:
        save_all(store)
