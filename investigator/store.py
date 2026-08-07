"""Per-case findings store: id-keyed upsert, reconcile-on-fix, attach.

Forked from `scripts/local_audit/findings.py` rather than imported — the wheel
`packages` list does not ship `scripts/`, and that module reads its paths from
module globals. ADR #70 records both reasons.

Two deliberate differences from the original:

- **`save_all` is atomic.** The original's `write_text` truncates before
  writing; `local_audit` survives that because nobody kills it, and this
  harness is designed to be killed. Without `os.replace`, the idempotence the
  whole resumability argument rests on is false.
- **`attach()` exists**, and the fields it writes sit outside `upsert_many`'s
  update set, so a re-check can never erase adversarial work.

`make_id` stays formula-identical to the original; a test asserts the two agree
on a fixed corpus so a divergence becomes a decision rather than a discovery.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from investigator.config import CasePaths

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# What `upsert_many` refreshes on an existing finding. Everything absent from
# this set survives a re-run untouched: `first_seen` (discovery time),
# `confounders` and `calibration` (adversarial work), and the identity triple
# (`pass`, `subject`, `claim`).
_VOLATILE_FIELDS = (
    "severity",
    "confidence",
    "related_files",
    "line",
    "evidence",
    "verified_at_sha",
    "prompt_version",
    "tier",
    "tier_basis",
    "externalisable",
    "evidence_pointers",
    "obligation_id",
)


# ========== id derivation ==========


def make_id(pass_name: str, subject: str, claim: str) -> str:
    """Content hash of the identity triple.

    `claim` is identity: it must carry no counts, dates, amounts, fact_ids or
    day-deltas, or the same finding re-mints a new id every cycle and
    resolve-on-fix stops meaning anything. Volatile detail goes in `evidence`.
    """
    raw = f"{pass_name}|{subject}|{claim}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ========== construction ==========


def finding(
    pass_name: str,
    subject: str,
    claim: str,
    *,
    case_id: str,
    evidence: str = "",
    severity: str = "info",
    confidence: str = "low",
    tier: str = "T5",
    tier_basis: str = "",
    externalisable: bool = False,
    obligation_id: str | None = None,
    evidence_pointers: dict | None = None,
    related_files: tuple[str, ...] | list[str] = (),
    line: int = 0,
) -> dict:
    """The one constructor for a raw finding dict.

    `pass_name` is a required positional so a pass cannot emit a finding
    attributed to another pass — a mismatch between the emitted `pass` and the
    name the orchestrator reconciles under would silently resolve everything
    that pass ever produced.
    """
    return {
        "pass": pass_name,
        "subject": subject,
        "claim": claim,
        "case_id": case_id,
        "evidence": evidence,
        "severity": severity,
        "confidence": confidence,
        "tier": tier,
        "tier_basis": tier_basis,
        "externalisable": externalisable,
        "obligation_id": obligation_id,
        "evidence_pointers": evidence_pointers
        or {"fact_ids": [], "doc_ids": [], "chunk_ids": [], "statute_refs": [], "person_ids": []},
        "related_files": list(related_files),
        "line": line,
    }


# ========== load / save ==========


def load_all(paths: CasePaths) -> dict[str, dict]:
    if not paths.findings_jsonl.exists():
        return {}
    store: dict[str, dict] = {}
    for line in paths.findings_jsonl.read_text(encoding="utf-8").splitlines():
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


def save_all(paths: CasePaths, store: dict[str, dict]) -> None:
    """Write the whole store atomically.

    Sorted by id with `sort_keys=True`, so the file is byte-stable across runs
    and a diff shows only what actually changed. The tmp + `os.replace` is what
    makes a mid-write kill survivable: POSIX rename is atomic, so a reader ever
    sees the old file or the new one, never a truncated one.
    """
    paths.investigation_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(store[fid], ensure_ascii=False, sort_keys=True) for fid in sorted(store)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    tmp = paths.findings_jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, paths.findings_jsonl)


# ========== upsert ==========


def normalize(
    raw: dict,
    verified_at_sha: str | None = None,
    prompt_version: str | None = None,
    self_consistency: dict | None = None,
) -> dict:
    """Fill the full persisted schema from a pass's raw finding dict."""
    fid = make_id(raw["pass"], raw["subject"], raw["claim"])
    now = _now()
    return {
        "id": fid,
        "pass": raw["pass"],
        "subject": raw["subject"],
        "claim": raw["claim"],
        "case_id": raw["case_id"],
        "obligation_id": raw.get("obligation_id"),
        "severity": raw.get("severity", "info"),
        "confidence": raw.get("confidence", "low"),
        "tier": raw.get("tier", "T5"),
        "tier_basis": raw.get("tier_basis", ""),
        "externalisable": raw.get("externalisable", False),
        "evidence": raw.get("evidence", ""),
        "evidence_pointers": raw.get("evidence_pointers", {}),
        "related_files": raw.get("related_files", []),
        "line": raw.get("line", 0),
        "confounders": [],
        "calibration": [],
        "verified_at_sha": verified_at_sha,
        "prompt_version": prompt_version,
        "self_consistency": self_consistency,
        "status": "open",
        "first_seen": now,
        "last_seen": now,
    }


def upsert_many(
    paths: CasePaths,
    raw_findings: list[dict],
    verified_at_sha: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, dict]:
    """Merge raw findings into the store.

    An existing finding keeps its `first_seen`, its `confounders` and its
    `calibration`; only `_VOLATILE_FIELDS` are refreshed. A previously resolved
    finding that reappears is reopened with its original discovery timestamp
    intact, which is what makes "first seen" mean something across months.

    A tier change appends one calibration entry. That is an append, never a
    truncation — the adversarial record only ever grows.
    """
    store = load_all(paths)
    now = _now()
    for raw in raw_findings:
        normalized = normalize(raw, verified_at_sha, prompt_version)
        fid = normalized["id"]
        existing = store.get(fid)
        if existing is None:
            store[fid] = normalized
            continue
        if existing.get("tier") != normalized["tier"]:
            existing.setdefault("calibration", []).append(
                {
                    "ts": now,
                    "tier": normalized["tier"],
                    "confidence": normalized["confidence"],
                    "reason": f"réévaluation: {existing.get('tier')} → {normalized['tier']}"
                    f" ({normalized['tier_basis']})",
                    "actor": normalized["pass"],
                }
            )
        existing.update({k: normalized[k] for k in _VOLATILE_FIELDS})
        existing["last_seen"] = now
        existing["status"] = "open"
    save_all(paths, store)
    return store


def reconcile_pass(paths: CasePaths, pass_name: str, current_ids: set[str]) -> None:
    """Mark findings this pass no longer reproduces as resolved.

    Only ever called for a pass that reported `complete=True`. A truncated
    sweep is indistinguishable from a fixed finding at this layer, so the
    distinction has to be made before we get here.
    """
    store = load_all(paths)
    changed = False
    for fid, f in store.items():
        if f.get("pass") == pass_name and f.get("status") == "open" and fid not in current_ids:
            f["status"] = "resolved"
            changed = True
    if changed:
        save_all(paths, store)


# ========== adversarial attachment ==========


def attach(
    paths: CasePaths,
    finding_id: str,
    confounders: list[dict] | None = None,
    calibration_entry: dict | None = None,
) -> bool:
    """Attach confounders and/or a calibration entry to an existing finding.

    Writes only fields outside `_VOLATILE_FIELDS`, which is precisely why a
    later re-check cannot erase them — the same mechanism that preserves
    `first_seen`. Confounders are deduplicated on `text_fr` so re-running the
    attack pass is idempotent.

    Returns False when the finding is not in the store (it may have been
    reconciled away between passes), never raising: a missing target is a race,
    not a defect.
    """
    store = load_all(paths)
    target = store.get(finding_id)
    if target is None:
        return False

    if confounders:
        existing = target.setdefault("confounders", [])
        seen = {c.get("text_fr") for c in existing}
        added = [c for c in confounders if c.get("text_fr") not in seen]
        if added:
            existing.extend(added)
    if calibration_entry:
        target.setdefault("calibration", []).append(calibration_entry)

    save_all(paths, store)
    return True
