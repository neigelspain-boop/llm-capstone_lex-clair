"""CaseGraph — a frozen, read-only view of one case's Plane Ib artifacts.

Load-only, and that is a hard constraint rather than a simplification.
`Fact.fact_id` is index-positional (`f"{doc_id}-f{index:03d}"`,
`ingestion/dossier/facts.py`), so re-extracting a document renumbers its facts
and dangles every `evidence_fact_ids` edge in `persons.jsonl` and every
`evidence_fact_ids` list in `compliance_matrix.json`. Re-extraction is not a
refresh; it is a silent referential-integrity break across three committed
artifacts. Plane V therefore reads what Plane Ib produced and never regenerates
it.

Every artifact but `facts.jsonl` is optional, and their absence is information
rather than an error:

- no `persons.jsonl` (as in `demo`) means no person edges, not a crash;
- no `coverage.jsonl` (as in `vitrine`) means `coverage_known=False`, which
  forces every absence finding down to `unverifiable` strength. See ADR #70.

Contracts: `docs/investigator-spec.md` §2.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from ingestion.dossier.facts import ActorRole, Fact, RoleAmbiguity
from investigator import config
from investigator.lexicon import fold

log = logging.getLogger(__name__)

MATCH_FIELDS = ("action", "target", "verbatim_quote", "distilled_context")


# ========== the graph ==========


@dataclass(frozen=True)
class CaseGraph:
    """One case, loaded. Treat every field as immutable."""

    case_id: str
    facts: dict[str, Fact]
    roles: dict[str, ActorRole]
    ambiguities: tuple[RoleAmbiguity, ...]
    persons: dict[str, dict]

    # person_id -> fact_ids, and its inverse. The inverse is what
    # `Fact.mentioned_person_ids` would have carried; deriving it costs nothing
    # and leaves ADR #53 follow-up (d) untouched rather than half-done.
    person_facts: dict[str, frozenset[str]]
    fact_persons: dict[str, frozenset[str]]
    persons_by_role: dict[str, tuple[str, ...]]

    facts_by_role: dict[str, tuple[str, ...]]
    folded: dict[str, dict[str, str]]

    coverage: dict[str, str]
    coverage_known: bool
    doc_ids: frozenset[str]
    doc_text_folded: dict[str, str]

    statute: dict[str, dict]
    case_chunks: dict[str, dict]

    unparsed_fact_lines: int = 0
    ambiguous_fact_ids: frozenset[str] = field(default_factory=frozenset)

    # ---------- queries ----------

    def fact_text(self, fact_id: str, fields: list[str] | tuple[str, ...]) -> str:
        blob = self.folded.get(fact_id, {})
        return " ".join(blob.get(f, "") for f in fields)

    def doc_of(self, fact_id: str) -> str | None:
        fact = self.facts.get(fact_id)
        return fact.source_doc_id if fact else None

    def gate_ok(self, doc_id: str) -> bool:
        """True only when the faithfulness gate actually passed this document.

        Unknown coverage is not a pass. `vitrine` has no `coverage.jsonl`, so
        every call returns False there, and tier derivation degrades
        accordingly instead of quietly assuming the best case.
        """
        return self.coverage.get(doc_id) == "ok"

    def is_healthy(self) -> bool:
        return self.unparsed_fact_lines == 0


# ========== loading ==========


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("graph: unparseable JSONL line in %s", path.name)
    return rows


def _read_chunks_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    # keep_default_na=False is project-wide convention: pandas otherwise turns
    # an empty `titre` into NaN, and every downstream string op then sees a
    # float.
    df = pd.read_csv(path, keep_default_na=False)
    if "chunk_id" not in df.columns:
        return {}
    return {row["chunk_id"]: dict(row) for _, row in df.iterrows()}


def load_graph(case_id: str, dossier_dir: Path | None = None) -> CaseGraph:
    """Load one case. Raises only when the statute corpus is missing."""
    paths = config.CasePaths.for_case(case_id, dossier_dir=dossier_dir)
    base = paths.case_dir

    facts: dict[str, Fact] = {}
    unparsed = 0
    for row in _read_jsonl(base / "facts.jsonl"):
        try:
            fact = Fact.model_validate(row)
        except ValidationError as exc:
            # Counted, never dropped silently: an unreadable fact shrinks the
            # denominator every absence finding is measured against, so the
            # graph is marked unhealthy and `graph`'s integrity pass reports it.
            unparsed += 1
            log.warning("graph: fact failed validation in %s: %s", case_id, exc)
            continue
        facts[fact.fact_id] = fact

    roles = {}
    for row in _read_jsonl(base / "actor_roles.jsonl"):
        try:
            role = ActorRole.model_validate(row)
        except ValidationError as exc:
            log.warning("graph: actor_role failed validation in %s: %s", case_id, exc)
            continue
        roles[role.role_id] = role

    ambiguities: list[RoleAmbiguity] = []
    for row in _read_jsonl(base / "role_ambiguities.jsonl"):
        try:
            ambiguities.append(RoleAmbiguity.model_validate(row))
        except ValidationError as exc:
            log.warning("graph: role_ambiguity failed validation in %s: %s", case_id, exc)

    persons = {p["person_id"]: p for p in _read_jsonl(base / "persons.jsonl") if "person_id" in p}

    # ---------- person <-> fact edges ----------
    person_facts: dict[str, frozenset[str]] = {}
    fact_persons_acc: dict[str, set[str]] = {}
    persons_by_role_acc: dict[str, set[str]] = {}
    for pid, person in persons.items():
        owned: set[str] = set()
        for assignment in person.get("role_assignments", []):
            role_id = assignment.get("role_id")
            if role_id:
                persons_by_role_acc.setdefault(role_id, set()).add(pid)
            for fid in assignment.get("evidence_fact_ids", []):
                owned.add(fid)
                if fid in facts:
                    fact_persons_acc.setdefault(fid, set()).add(pid)
        person_facts[pid] = frozenset(owned)

    # ---------- derived indices ----------
    facts_by_role_acc: dict[str, list[str]] = {}
    folded: dict[str, dict[str, str]] = {}
    for fid, fact in facts.items():
        facts_by_role_acc.setdefault(fact.actor_role, []).append(fid)
        folded[fid] = {
            "action": fold(fact.action),
            "target": fold(fact.target),
            "verbatim_quote": fold(fact.verbatim_quote),
            "distilled_context": fold(fact.distilled_context),
        }

    coverage: dict[str, str] = {}
    coverage_path = base / "coverage.jsonl"
    for row in _read_jsonl(coverage_path):
        doc_id = row.get("doc_id")
        if doc_id:
            # coverage.jsonl is append-only, one row per verification run, so
            # the last row for a doc is its current status.
            coverage[doc_id] = row.get("status", "")
    coverage_known = coverage_path.exists() and bool(coverage)

    extracted = base / "extracted"
    doc_text_folded: dict[str, str] = {}
    if extracted.is_dir():
        for md in sorted(extracted.glob("*.md")):
            try:
                doc_text_folded[md.stem] = fold(md.read_text(encoding="utf-8"))
            except OSError as exc:
                log.warning("graph: cannot read %s: %s", md, exc)

    doc_ids = frozenset(doc_text_folded) | {f.source_doc_id for f in facts.values()}

    # Facts whose role assignment the extractor could not settle are excluded
    # from every predicate: reasoning about an obligation's bearer from a fact
    # whose bearer is disputed is exactly the inference the ambiguity records
    # exist to prevent.
    ambiguous = {fid for a in ambiguities for fid in a.fact_ids}

    statute = _read_chunks_csv(config.STATUTE_CHUNKS_CSV)
    if not statute:
        raise FileNotFoundError(
            f"statute corpus missing or unreadable: {config.STATUTE_CHUNKS_CSV}"
        )

    return CaseGraph(
        case_id=case_id,
        facts=facts,
        roles=roles,
        ambiguities=tuple(ambiguities),
        persons=persons,
        person_facts=person_facts,
        fact_persons={fid: frozenset(pids) for fid, pids in fact_persons_acc.items()},
        persons_by_role={r: tuple(sorted(p)) for r, p in sorted(persons_by_role_acc.items())},
        facts_by_role={r: tuple(sorted(f)) for r, f in sorted(facts_by_role_acc.items())},
        folded=folded,
        coverage=coverage,
        coverage_known=coverage_known,
        doc_ids=doc_ids,
        doc_text_folded=doc_text_folded,
        statute=statute,
        case_chunks=_read_chunks_csv(base / "chunks.csv"),
        unparsed_fact_lines=unparsed,
        ambiguous_fact_ids=frozenset(ambiguous),
    )
