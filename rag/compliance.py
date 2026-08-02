"""Compliance matrix generator: facts x obligations -> compliance determinations
(Day B, Deliverable B3, ADR #43).

Plane Ib (facts.jsonl, actor_roles.jsonl, role_ambiguities.jsonl) is the
structured substrate for one case. This module is the first reasoning stage
over that substrate: for each actor role discovered in the case, it clusters
that role's facts, retrieves the statute articles most likely to define its
obligations (source_scope="statute", ADR #41), and asks an LLM to judge
whether each identifiable obligation was met, breached, is ambiguous, or
lacks sufficient evidence.

This is the one deliverable in the pipeline that justifies Opus-class
reasoning cost — every other LLM call in lex-clair uses a cheaper model
(Haiku for routing, Gemini Flash Lite for fact extraction, etc.). One call
per (actor_role, facts cluster) using anthropic/claude-opus-4.7 with
reasoning.effort="max" (confirmed live on OpenRouter 2026-07-28 — "max" is a
distinct reasoning-effort level, not a model suffix; the model slug itself
uses a dot, not a hyphen). The OpenAI SDK's typed chat.completions.create()
doesn't accept a bare `reasoning` kwarg, so it's passed via `extra_body`
(the SDK's standard mechanism for provider-specific fields) — OpenRouter
still receives "reasoning": {"effort": "max"} at the top level of the
request body.

Output is fully regenerated (no merge) on every run and persisted to
data/dossier/<case_id>/compliance_matrix.json. Deterministic entry_id
(SHA1 of statute_chunk_id + actor_role, truncated to 12 chars) keeps repeat
runs from creating duplicate rows if a future version adds merge logic.

CLI: python -m rag.compliance --case-id <id> [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from ingestion.clients import get_openrouter_client
from ingestion.dossier.facts import ActorRole, DOSSIER_DIR, Fact, RoleAmbiguity
from rag.compliance_prompts import COMPLIANCE_SYSTEM_PROMPT
from rag.retrieve import retrieve

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

COMPLIANCE_MODEL_ID = "anthropic/claude-opus-4.7"
RELEVANT_STATUTE_K = 8  # per fact cluster, retrieve top-k statute chunks for context
MAX_FACTS_PER_ROLE = 30  # see _cap_facts_chronologically

# Rough per-token USD rates for --dry-run token/cost estimates only, based
# on a live OpenRouter dry-call against anthropic/claude-opus-4.7 on
# 2026-07-28 ($0.008435 for 22 prompt / 333 completion tokens). Real
# (non-dry-run) calls use the exact `usage.cost` OpenRouter returns instead.
_EST_PROMPT_USD_PER_TOKEN = 5e-6
_EST_COMPLETION_USD_PER_TOKEN = 25e-6
# --dry-run only: no real max_tokens cap on the live call (ADR #49). This
# is a stand-in completion-length estimate for cost preview.
_DRY_RUN_EST_COMPLETION_TOKENS = 2048

# ADR #53: a dry-run cost estimate above this signals a prompt-size
# regression (e.g. an unbounded persons/cross-role block) — loud failure
# rather than a silently expensive real run.
DRY_RUN_COST_ALERT_USD = 25.0

_VALID_STATUSES = {"met", "breached", "ambiguous", "insufficient_evidence"}


# ========== schemas ==========

class ComplianceEntry(BaseModel):
    """One obligation determination for one (statute_chunk_id, actor_role) pair."""

    entry_id: str  # deterministic hash of statute_chunk_id + actor_role
    statute_chunk_id: str  # e.g., "cc-587" — the article defining the obligation
    statute_excerpt: str  # <=200 char verbatim quote of the obligation text
    obligation_summary: str  # 1-sentence what the article requires, in French
    actor_role: str  # snake_case role_id (e.g., "notaire_redacteur")
    status: Literal["met", "breached", "ambiguous", "insufficient_evidence"]
    evidence_fact_ids: list[str]  # facts supporting the status
    rationale: str  # 2-3 French sentences explaining the determination
    persons_named: list[dict] = Field(default_factory=list)  # [] on every case until D1-D4 ship (ADR #53)


class ComplianceMatrix(BaseModel):
    """Full compliance matrix for one case: all obligation determinations
    across all actor roles discovered in that case's dossier."""

    case_id: str
    generated_at: datetime
    model_id: str
    total_facts_considered: int
    total_entries: int
    entries: list[ComplianceEntry]
    unresolved_ambiguities: int = Field(
        description="Total count of role_ambiguities.jsonl entries for this "
        "case — a bare total for v1, not a per-entry link to specific "
        "ComplianceEntry determinations (see ADR #43 follow-ups)."
    )


# ========== clock (injectable for idempotency tests) ==========

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ========== artifact loading ==========

def _read_jsonl(path: Path, model: type[BaseModel]) -> list:
    """Read one jsonl file into a list of validated Pydantic objects.

    Mirrors ingestion/dossier/index.py's round-trip idiom
    (model_validate_json per line). Returns [] if the file doesn't exist —
    role_ambiguities.jsonl in particular may legitimately be empty/absent.
    """
    if not path.exists():
        return []
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_jsonl_dicts(path: Path) -> list[dict]:
    """Plain-dict JSONL reader for person records — no pydantic model exists
    for them yet (D1-D4 person-index pipeline unshipped, ADR #53). Returns
    [] if the file doesn't exist, matching _read_jsonl's convention.
    """
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_case_artifacts(case_id: str) -> tuple[list[Fact], list[ActorRole], list[RoleAmbiguity]]:
    """Load facts.jsonl, actor_roles.jsonl, role_ambiguities.jsonl for one case."""
    case_dir = DOSSIER_DIR / case_id
    facts = _read_jsonl(case_dir / "facts.jsonl", Fact)
    roles = _read_jsonl(case_dir / "actor_roles.jsonl", ActorRole)
    ambiguities = _read_jsonl(case_dir / "role_ambiguities.jsonl", RoleAmbiguity)
    return facts, roles, ambiguities


# ========== compliance-run cache (ADR #53) ==========

def _compliance_cache_key(role_id: str, fact_ids: list[str], user_message: str) -> str:
    """Deterministic cluster fingerprint: role_id + sorted fact_ids + a hash
    of the assembled prompt (so a retrieval or context change invalidates
    the cache entry, not just a fact-set change).
    """
    prompt_hash = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
    payload = f"{role_id}|{','.join(sorted(fact_ids))}|{prompt_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_compliance_cache(case_id: str) -> dict[str, dict]:
    """Load data/dossier/<case_id>/compliance_cache.jsonl into a {cache_key: entry} map."""
    cache_path = DOSSIER_DIR / case_id / "compliance_cache.jsonl"
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cache[entry["cache_key"]] = entry
    return cache


def _append_compliance_cache(case_id: str, entry: dict) -> None:
    """Append one cache entry — only called for a real (non-cached, non-dry-run) call."""
    cache_path = DOSSIER_DIR / case_id / "compliance_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ========== fact clustering ==========

def _group_facts_by_role(facts: list[Fact]) -> dict[str, list[Fact]]:
    """Group facts by exact actor_role string (no fuzzy dedup of near-duplicate
    role_ids — see ADR #43 follow-ups)."""
    groups: dict[str, list[Fact]] = {}
    for fact in facts:
        groups.setdefault(fact.actor_role, []).append(fact)
    return groups


def _role_label(role_id: str, roles: list[ActorRole]) -> str:
    """Look up a role's French label from the actor_roles catalogue, falling
    back to the raw role_id if the role wasn't catalogued."""
    for role in roles:
        if role.role_id == role_id:
            return role.label_fr
    return role_id


def _cap_facts_chronologically(role_id: str, facts: list[Fact]) -> list[Fact]:
    """Cap a role cluster at MAX_FACTS_PER_ROLE, keeping the chronologically
    earliest facts (they usually establish the timeline an obligation is
    measured against). Uncapped, heavy roles (20-40+ facts observed in the
    private case) push the user message past 5K tokens before statute
    chunks are even added, and Opus's coherence degrades reasoning across
    too many facts in one call.
    """
    if len(facts) <= MAX_FACTS_PER_ROLE:
        return facts

    sorted_facts = sorted(facts, key=lambda f: (f.date is None, f.date or ""))
    log.warning(
        "compliance: role_id=%s has %d facts, capping to %d (chronologically earliest kept)",
        role_id, len(facts), MAX_FACTS_PER_ROLE,
    )
    return sorted_facts[:MAX_FACTS_PER_ROLE]


# ========== retrieval query + prompt assembly ==========

def _build_retrieval_query(role_id: str, role_label: str, facts: list[Fact]) -> str:
    """Build the statute-retrieval query: role label + top unique action verbs."""
    verbs: list[str] = []
    for fact in facts:
        if fact.action not in verbs:
            verbs.append(fact.action)
        if len(verbs) >= 5:
            break
    return f"{role_label} ({role_id}) : obligations légales — {', '.join(verbs)}"


def _extract_cross_role_context(
    role_id: str,
    cluster_facts: list[Fact],
    all_facts: list[Fact],
    actor_roles: list[ActorRole],
) -> str:
    """Return French cross-role context block for the compliance LLM prompt.

    For each source_doc_id in cluster_facts, finds other role_ids that share
    that document, formats into a "Contexte inter-rôles" block. Uses label_fr
    from actor_roles for human-readable person identification.

    Returns "" if no cross-role linkages found. Deterministic (sorted output)
    to preserve compliance matrix idempotency.
    """
    cluster_doc_ids = {f.source_doc_id for f in cluster_facts}
    other_roles_in_shared_docs: dict[str, set[str]] = {}

    for fact in all_facts:
        if fact.actor_role == role_id:
            continue
        if fact.source_doc_id in cluster_doc_ids:
            other_roles_in_shared_docs.setdefault(fact.actor_role, set()).add(fact.source_doc_id)

    if not other_roles_in_shared_docs:
        return ""

    role_to_label = {r.role_id: r.label_fr for r in actor_roles}

    lines = [
        "Contexte inter-rôles :",
        "Dans les documents sources analysés pour ce rôle, les autres rôles suivants apparaissent également :",
    ]
    for other_role, shared_docs in sorted(other_roles_in_shared_docs.items()):
        label = role_to_label.get(other_role, other_role)
        shown_docs = sorted(shared_docs)[:3]
        doc_list = ", ".join(shown_docs)
        suffix = " (…)" if len(shared_docs) > 3 else ""
        lines.append(f"- Rôle : {other_role} — libellé : {label} — dans les documents [{doc_list}]{suffix}")
    lines.append("Considérez si les obligations de ces autres rôles interagissent avec celles du rôle actuel.")
    return "\n".join(lines)


def _extract_persons_context(case_persons: list[dict], entities: list[dict]) -> str:
    """Return French "Personnes impliquées" context block for the compliance
    LLM prompt, merging case_persons (case-specific: role_assignments,
    ambiguity_note) with entities (base identity: canonical_name, aliases),
    keyed by person_id.

    Returns "" if both lists are empty — the real-world state for every
    case today, since D1-D4 (mention extraction, entity resolution) haven't
    shipped (ADR #53). Deterministic (sorted output) to preserve compliance
    matrix idempotency, mirroring _extract_cross_role_context.
    """
    if not case_persons and not entities:
        return ""

    entities_by_id = {e["person_id"]: e for e in entities if "person_id" in e}
    case_by_id = {p["person_id"]: p for p in case_persons if "person_id" in p}
    person_ids = sorted(set(entities_by_id) | set(case_by_id))

    if not person_ids:
        return ""

    lines = ["Personnes impliquées :"]
    for pid in person_ids:
        entity = entities_by_id.get(pid, {})
        case_p = case_by_id.get(pid, {})
        canonical_name = entity.get("canonical_name") or case_p.get("canonical_name") or pid
        aliases = sorted(entity.get("aliases") or [])
        role_assignments = sorted(case_p.get("role_assignments") or [])
        ambiguity_note = case_p.get("ambiguity_note")

        line = f"- person_id={pid} nom={canonical_name}"
        if aliases:
            line += f" alias=[{', '.join(aliases)}]"
        if role_assignments:
            line += f" rôles=[{', '.join(role_assignments)}]"
        if ambiguity_note:
            line += f" ambiguïté={ambiguity_note}"
        lines.append(line)

    return "\n".join(lines)


def _merge_persons_named(case_persons: list[dict], entities: list[dict]) -> list[dict]:
    """Reduce a cluster's persons context to [{"person_id", "canonical_name"}, ...],
    sorted by person_id, for ComplianceEntry.persons_named."""
    entities_by_id = {e["person_id"]: e for e in entities if "person_id" in e}
    case_by_id = {p["person_id"]: p for p in case_persons if "person_id" in p}
    person_ids = sorted(set(entities_by_id) | set(case_by_id))

    merged: list[dict] = []
    for pid in person_ids:
        entity = entities_by_id.get(pid, {})
        case_p = case_by_id.get(pid, {})
        canonical_name = entity.get("canonical_name") or case_p.get("canonical_name") or pid
        merged.append({"person_id": pid, "canonical_name": canonical_name})
    return merged


def _build_user_message(
    role_id: str,
    role_label: str,
    facts: list[Fact],
    chunks: list[dict],
    all_facts: list[Fact],
    actor_roles: list[ActorRole],
    case_persons: list[dict] | None = None,
    entities: list[dict] | None = None,
) -> str:
    """Assemble the compact context blob: role, capped facts, retrieved statute
    chunks, and (when applicable) cross-role (C1, ADR #44) and persons
    (ADR #53) context blocks."""
    capped_facts = _cap_facts_chronologically(role_id, facts)

    facts_block = "\n".join(
        f'- fact_id={f.fact_id} date={f.date or "inconnue"} action="{f.action}" '
        f'target={f.target or "N/A"} distilled="{f.distilled_context or "N/A"}" '
        f'citation="{f.verbatim_quote}"'
        for f in capped_facts
    )
    chunks_block = "\n".join(
        f'- chunk_id={c["chunk_id"]} section={c["section_path"]} texte="{c["texte"]}"'
        for c in chunks
    )

    message = (
        f"Rôle d'acteur : {role_label} ({role_id})\n\n"
        f"Faits ({len(capped_facts)} sur {len(facts)}) :\n{facts_block}\n\n"
        f"Articles potentiellement applicables :\n{chunks_block}"
    )

    cross_role_block = _extract_cross_role_context(role_id, capped_facts, all_facts, actor_roles)
    if cross_role_block:
        message = f"{message}\n\n{cross_role_block}"

    persons_block = _extract_persons_context(case_persons or [], entities or [])
    if persons_block:
        message = f"{message}\n\n{persons_block}"

    return message


# ========== defensive JSON parsing ==========

def _recover_partial_entries(text: str, role_id: str) -> list[dict]:
    """Recover complete top-level {...} objects from text that failed a
    straight raw_decode() — used when max_tokens truncation cuts a response
    off mid-object. Scans left to right tracking brace depth and string
    state (honoring \\ escapes so a quote inside a string doesn't end it
    early); each span where depth returns to 0 is a candidate object,
    parsed independently. Logs a warning naming role_id with recovered vs.
    discarded counts when at least one entry is recovered; returns [] (with
    no warning here — the caller logs the outright-failure warning) if
    nothing could be recovered.
    """
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth == 0 and start is not None:
                objects.append(text[start:i + 1])
                start = None

    total = len(objects) + (1 if depth > 0 else 0)

    recovered: list[dict] = []
    for candidate in objects:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            recovered.append(parsed)

    if recovered:
        log.warning(
            "compliance: partial parse recovered %d complete entries, discarded %d "
            "incomplete/malformed for role_id=%s",
            len(recovered), total - len(recovered), role_id,
        )

    return recovered


def _parse_compliance_response(raw: str, role_id: str) -> list[dict]:
    """Parse the compliance LLM's raw JSON array text into a list of dicts.

    Mirrors ingestion/dossier/facts.py::_parse_llm_json (fence-strip +
    json.JSONDecoder().raw_decode(), tolerant of trailing prose commentary),
    adapted for a top-level list instead of dict. If that fast path fails —
    notably when max_tokens truncation (reasoning-effort tokens share the
    same budget as completion tokens, so this can happen before any visible
    JSON looks wrong) cuts the response off mid-object — falls back to
    _recover_partial_entries() to salvage any complete entries seen before
    the cut. Returns [] and logs a warning only if recovery also yields
    nothing, or the top-level type is wrong.
    """
    text = (raw or "").strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        recovered = _recover_partial_entries(text, role_id)
        if recovered:
            return recovered
        log.warning("compliance: JSON parse failed for role_id=%s: %s\nraw response: %s", role_id, e, raw)
        return []

    if not isinstance(obj, list):
        log.warning("compliance: top-level JSON is not a list for role_id=%s (got %s)", role_id, type(obj).__name__)
        return []

    return obj


def _entry_id(statute_chunk_id: str, actor_role: str) -> str:
    """Deterministic entry_id: SHA1(statute_chunk_id|actor_role)[:12]."""
    digest = hashlib.sha1(f"{statute_chunk_id}|{actor_role}".encode("utf-8")).hexdigest()
    return digest[:12]


def _build_entries(
    raw_entries: list[dict], actor_role: str, persons_named: list[dict] | None = None
) -> list[ComplianceEntry]:
    """Validate each raw entry dict into a ComplianceEntry; skip (with a
    warning) any that fail validation or are missing required fields.
    persons_named is the same cluster-level list on every entry (ADR #53)."""
    persons_named = persons_named or []
    entries: list[ComplianceEntry] = []
    for item in raw_entries:
        try:
            statute_chunk_id = item["statute_chunk_id"]
            entry = ComplianceEntry(
                entry_id=_entry_id(statute_chunk_id, actor_role),
                statute_chunk_id=statute_chunk_id,
                statute_excerpt=item["statute_excerpt"],
                obligation_summary=item["obligation_summary"],
                actor_role=actor_role,
                status=item["status"],
                evidence_fact_ids=item.get("evidence_fact_ids", []),
                rationale=item["rationale"],
                persons_named=persons_named,
            )
        except (KeyError, ValidationError) as e:
            log.warning("compliance: skipping invalid entry for actor_role=%s: %s", actor_role, e)
            continue
        entries.append(entry)
    return entries


# ========== LLM call ==========

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _call_compliance_llm(
    role_id: str,
    role_label: str,
    facts: list[Fact],
    chunks: list[dict],
    all_facts: list[Fact],
    actor_roles: list[ActorRole],
    case_persons: list[dict] | None = None,
    entities: list[dict] | None = None,
    cache: dict[str, dict] | None = None,
    dry_run: bool = False,
) -> tuple[list[dict], dict]:
    """Call Opus 4.7 (reasoning.effort="max") for one role cluster.

    Returns (parsed_entries, usage) where usage is
    {"prompt_tokens", "completion_tokens", "cost_usd", "estimated",
    "cache_hit", "cache_key"}. If `cache` is given and the cluster
    fingerprint (ADR #53, _compliance_cache_key) is already present, the
    cached entries are returned at zero cost with cache_hit=True — checked
    before both the dry-run estimate and the real call, so a cached rerun's
    dry-run report is also accurate. Otherwise, in --dry-run mode, no API
    call is made: tokens are estimated via a char/4 heuristic and cost via
    the approximate rates in _EST_*_USD_PER_TOKEN, with estimated=True.
    Otherwise, cost comes from OpenRouter's exact per-call usage.cost when
    present, falling back to the same estimate formula if it isn't.
    "cache_key" is returned uncached (None on a cache hit) so the caller can
    persist a fresh entry without recomputing the fingerprint.

    For real calls, reasoning-effort tokens and completion tokens share the
    same max_tokens budget for this model, so finish_reason can come back
    "length" (and get logged) before any visible JSON output completes —
    see _recover_partial_entries() for how the parser copes with that.
    """
    capped_facts = _cap_facts_chronologically(role_id, facts)
    user_message = _build_user_message(
        role_id, role_label, capped_facts, chunks, all_facts, actor_roles,
        case_persons=case_persons, entities=entities,
    )

    cache_key = None
    if cache is not None:
        fact_ids = [f.fact_id for f in capped_facts]
        cache_key = _compliance_cache_key(role_id, fact_ids, user_message)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached["raw_entries"], {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "estimated": False,
                "cache_hit": True,
                "cache_key": None,
            }

    if dry_run:
        prompt_tokens = _estimate_tokens(COMPLIANCE_SYSTEM_PROMPT) + _estimate_tokens(user_message)
        completion_tokens = _DRY_RUN_EST_COMPLETION_TOKENS
        cost = (
            prompt_tokens * _EST_PROMPT_USD_PER_TOKEN
            + completion_tokens * _EST_COMPLETION_USD_PER_TOKEN
        )
        return [], {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "estimated": True,
            "cache_hit": False,
            "cache_key": cache_key,
        }

    client = get_openrouter_client()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=COMPLIANCE_MODEL_ID,
        messages=[
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        extra_body={"reasoning": {"effort": "max"}},
    )
    elapsed = time.perf_counter() - t0
    raw = response.choices[0].message.content or ""
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    entries = _parse_compliance_response(raw, role_id)

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost = getattr(usage, "cost", None) if usage is not None else None
    if cost is None:
        cost = (
            prompt_tokens * _EST_PROMPT_USD_PER_TOKEN
            + completion_tokens * _EST_COMPLETION_USD_PER_TOKEN
        )

    log.info(
        "compliance call · role_id=%s finish_reason=%s prompt_tokens=%d "
        "completion_tokens=%d raw_chars=%d elapsed=%.1fs",
        role_id, finish_reason, prompt_tokens, completion_tokens, len(raw), elapsed,
    )
    if finish_reason == "length":
        log.warning(
            "compliance: response truncated by provider's max-output ceiling "
            "for role_id=%s (completion_tokens=%d) — see parser recovery",
            role_id, completion_tokens,
        )

    return entries, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "estimated": False,
        "cache_hit": False,
        "cache_key": cache_key,
    }


def _check_dry_run_cost_gate(total_cost: float) -> None:
    """Raise if a --dry-run cost estimate exceeds DRY_RUN_COST_ALERT_USD.

    Extracted as its own function (rather than an inline check) so the
    threshold is directly testable without constructing a fixture large
    enough to actually cross $25. Called only from the dry_run branch of
    generate_compliance_matrix, after the per-cluster and total estimates
    have already been printed — so the operator sees the numbers even
    though the run then aborts loudly (ADR #53).
    """
    if total_cost > DRY_RUN_COST_ALERT_USD:
        raise RuntimeError(
            f"compliance dry-run cost estimate ${total_cost:.2f} exceeds the "
            f"${DRY_RUN_COST_ALERT_USD:.2f} safety threshold — this signals a "
            "prompt-size regression. Review before running for real."
        )


# ========== orchestration ==========

def generate_compliance_matrix(
    case_id: str, limit: int | None = None, dry_run: bool = False
) -> ComplianceMatrix:
    """Generate the compliance matrix for one case; persist unless dry_run.

    Idempotent: fully regenerates data/dossier/<case_id>/compliance_matrix.json
    from scratch on every run (no merge logic). limit caps the number of
    role groups processed (dev cost cap). dry_run assembles the LLM calls,
    reports per-cluster and total token/cost estimates, and estimates
    tokens/cost without invoking the API or writing output — raising if the
    total exceeds DRY_RUN_COST_ALERT_USD (ADR #53).

    Cluster-level results are cached (ADR #53) at
    data/dossier/<case_id>/compliance_cache.jsonl, keyed by a fingerprint of
    (role_id, sorted fact_ids, prompt hash): an idempotent rerun with
    unchanged facts/retrieval/context is entirely cache hits, at zero
    marginal cost.

    case_persons/entities (ADR #53) are loaded once per case from
    data/dossier/<case_id>/persons.jsonl and data/dossier/entities/
    persons.jsonl — both [] in every case today, since the D1-D4 person-
    index pipeline (mention extraction, entity resolution) hasn't shipped.
    """
    t0 = time.time()

    # ADR #48: per-case FileHandler for the compliance run.
    # Overwritten each run (mode="w") so the log matches the current matrix.
    # Removed in finally so the handler doesn't leak to other module callers.
    log_dir = DOSSIER_DIR / case_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "compliance_run.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s"
    ))
    log.addHandler(file_handler)
    log.info(f"compliance run start · case_id={case_id} log_path={log_path}")

    try:
        facts, roles, ambiguities = _load_case_artifacts(case_id)
        groups = _group_facts_by_role(facts)

        case_persons_all = _read_jsonl_dicts(DOSSIER_DIR / case_id / "persons.jsonl")
        entities_all = _read_jsonl_dicts(DOSSIER_DIR / "entities" / "persons.jsonl")
        compliance_cache = _load_compliance_cache(case_id)

        role_ids = sorted(groups)
        if limit is not None:
            role_ids = role_ids[:limit]

        all_entries: list[ComplianceEntry] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        dry_run_cluster_summaries: list[dict] = []

        for role_id in role_ids:
            role_facts = groups[role_id]
            role_label = _role_label(role_id, roles)
            query = _build_retrieval_query(role_id, role_label, role_facts)
            chunks = retrieve(query, k=RELEVANT_STATUTE_K, source_scope="statute")

            # ADR #53: mentioned_person_ids doesn't exist on Fact yet (D1-D4
            # unshipped) — defensive getattr, real current-state handling.
            mentioned_ids: set[str] = set()
            for f in role_facts:
                mentioned_ids.update(getattr(f, "mentioned_person_ids", None) or [])
            cluster_case_persons = [p for p in case_persons_all if p.get("person_id") in mentioned_ids]
            cluster_entities = [e for e in entities_all if e.get("person_id") in mentioned_ids]

            raw_entries, usage = _call_compliance_llm(
                role_id, role_label, role_facts, chunks, facts, roles,
                case_persons=cluster_case_persons, entities=cluster_entities,
                cache=compliance_cache, dry_run=dry_run,
            )

            if not usage["cache_hit"] and not dry_run and usage["cache_key"] is not None:
                cache_entry = {
                    "cache_key": usage["cache_key"], "role_id": role_id, "raw_entries": raw_entries,
                }
                compliance_cache[usage["cache_key"]] = cache_entry
                _append_compliance_cache(case_id, cache_entry)

            persons_named = _merge_persons_named(cluster_case_persons, cluster_entities)
            log.info("compliance: role_id=%s persons_named_count=%d", role_id, len(persons_named))

            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_cost += usage["cost_usd"] or 0.0
            all_entries.extend(_build_entries(raw_entries, role_id, persons_named=persons_named))

            if dry_run:
                dry_run_cluster_summaries.append({
                    "role_id": role_id,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "cost_usd": usage["cost_usd"] or 0.0,
                    "cache_hit": usage["cache_hit"],
                })

        matrix = ComplianceMatrix(
            case_id=case_id,
            generated_at=_utcnow(),
            model_id=COMPLIANCE_MODEL_ID,
            total_facts_considered=len(facts),
            total_entries=len(all_entries),
            entries=all_entries,
            unresolved_ambiguities=len(ambiguities),
        )

        elapsed = time.time() - t0

        if not dry_run:
            out_dir = DOSSIER_DIR / case_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "compliance_matrix.json"
            out_path.write_text(
                json.dumps(matrix.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        if dry_run:
            for cluster in dry_run_cluster_summaries:
                cache_note = " (cache_hit)" if cluster["cache_hit"] else ""
                print(
                    f"compliance dry-run cluster · role_id={cluster['role_id']} "
                    f"prompt_tokens={cluster['prompt_tokens']} "
                    f"completion_tokens_est={cluster['completion_tokens']} "
                    f"cost_est=${cluster['cost_usd']:.4f}{cache_note}"
                )
            print(
                f"compliance dry-run · case_id={case_id} roles_processed={len(role_ids)} "
                f"est_prompt_tokens={total_prompt_tokens} est_completion_tokens={total_completion_tokens} "
                f"cost_est=${total_cost:.2f} elapsed={elapsed:.1f}s"
            )
            _check_dry_run_cost_gate(total_cost)
        else:
            status_counts = {"met": 0, "breached": 0, "ambiguous": 0, "insufficient_evidence": 0}
            for entry in all_entries:
                status_counts[entry.status] += 1

            print(
                f"compliance summary · case_id={case_id} roles_processed={len(role_ids)} "
                f"entries={len(all_entries)} met={status_counts['met']} "
                f"breached={status_counts['breached']} ambiguous={status_counts['ambiguous']} "
                f"insufficient={status_counts['insufficient_evidence']} elapsed={elapsed:.1f}s "
                f"cost_est=${total_cost:.2f}"
            )

        return matrix
    finally:
        log.removeHandler(file_handler)
        file_handler.close()


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m rag.compliance --case-id <id> [--limit N] [--dry-run]."""
    parser = argparse.ArgumentParser(
        description="Generate the compliance matrix for one case: facts x "
        "obligations -> met/breached/ambiguous/insufficient_evidence."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N role groups (dev cost cap)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="assemble the LLM calls but don't invoke them; print token estimates and skip API calls",
    )
    args = parser.parse_args()

    generate_compliance_matrix(args.case_id, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
