"""Obligation models, pass/evaluation types, tier derivation, and the gate.

Three things live here that nothing else may duplicate:

- `Obligation` and its nested models — the parsed form of a catalog entry.
  Validation is strict and happens at load, so a malformed catalog fails before
  a cycle starts rather than mid-sweep.
- `assign_tier()` — the single place a finding's evidentiary tier is *derived*.
  Tiers are never asserted by a pass or written by hand.
- `externalisable_findings()` — the single path from the findings store to any
  shareable artifact.

Rationale: ADR #70.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal, NamedTuple

from pydantic import BaseModel, Field, model_validator

from investigator import config

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from investigator.budget import Budget
    from investigator.catalog import Catalog
    from investigator.config import CasePaths
    from investigator.graph import CaseGraph

# ========== vocabularies ==========

Tier = Literal["T1", "T2", "T3", "T4", "T5"]

# Lower index is stronger evidence. Ordering, not arithmetic: a "min" over
# tiers means the *weaker* of two, which is the larger index.
TIER_ORDER: dict[str, int] = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4}

EvaluationStatus = Literal[
    "satisfied", "gap", "unverifiable", "not_triggered", "window_breach"
]

# Statuses that rest on a fact being present, versus on one being absent. The
# distinction drives tier derivation: a quote either exists in the graph or it
# does not, so presence needs no coverage denominator — absence does.
PRESENCE_STATUSES = frozenset({"satisfied", "window_breach"})
ABSENCE_STATUSES = frozenset({"gap", "unverifiable"})

OBLIGATION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,60}$")

# `claim` text is a finding's identity, so it may only interpolate values that
# are stable for the lifetime of the obligation. Counts, dates, amounts,
# fact_ids and day-deltas belong in `evidence`, which is rewritten on every
# upsert.
CLAIM_PLACEHOLDERS = frozenset(
    {"obligation_id", "source_ref", "title_fr", "foreach_role", "foreach_key"}
)

DEFAULT_MATCH_FIELDS = ("action", "target", "verbatim_quote", "distilled_context")


# ========== catalog models ==========


class FactMatch(BaseModel):
    """One leaf predicate: does the graph hold a fact of this shape?

    Evaluated over a finite candidate set, so `matches == 0` is a decidable
    fact about the graph rather than a judgment. That is what lets absence be
    reported as a confident negative.
    """

    kind: Literal["fact_match"] = "fact_match"
    actor_roles: list[str] = Field(default_factory=list)
    any_terms_fr: list[str] = Field(default_factory=list)
    match_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_MATCH_FIELDS))
    after_trigger: bool = False
    before_or_at_trigger: bool = False
    bind_to_foreach: bool = False
    min_count: int = 1

    @model_validator(mode="after")
    def _check(self) -> "FactMatch":
        unknown = set(self.match_fields) - set(DEFAULT_MATCH_FIELDS)
        if unknown:
            raise ValueError(f"unknown match_fields: {sorted(unknown)}")
        if not self.match_fields:
            raise ValueError("match_fields must not be empty")
        if self.after_trigger and self.before_or_at_trigger:
            raise ValueError("after_trigger and before_or_at_trigger are exclusive")
        if self.min_count < 1:
            raise ValueError("min_count must be >= 1")
        return self


class EvidencePredicate(BaseModel):
    """Boolean combination of `FactMatch` leaves. At least one branch required."""

    all_of: list[FactMatch] = Field(default_factory=list)
    any_of: list[FactMatch] = Field(default_factory=list)
    none_of: list[FactMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _non_empty(self) -> "EvidencePredicate":
        if not (self.all_of or self.any_of or self.none_of):
            raise ValueError("expected_evidence must declare at least one branch")
        return self

    def leaves(self) -> list[FactMatch]:
        return [*self.all_of, *self.any_of, *self.none_of]


class Anchor(BaseModel):
    chunk_id: str
    note_fr: str = ""


class Source(BaseModel):
    kind: Literal["statute", "contract_clause", "deontology", "jurisprudence"]
    ref: str
    chunk_id: str | None = None
    legiarti_id: str | None = None
    doc_id_pattern: str | None = None
    excerpt_fr: str

    @model_validator(mode="after")
    def _check(self) -> "Source":
        if not self.excerpt_fr.strip():
            raise ValueError(f"{self.ref}: excerpt_fr must be non-empty")
        if self.kind == "statute" and not (self.chunk_id or self.legiarti_id):
            # An anchor absent from the 792-chunk corpus is legitimate — it
            # ships with legiarti_id and search.py reports it as unverifiable
            # rather than the catalog failing to load. But it must carry one or
            # the other, or nothing can ever check it.
            raise ValueError(f"{self.ref}: statute source needs chunk_id or legiarti_id")
        return self


class Bearer(BaseModel):
    actor_roles: list[str] = Field(min_length=1)
    bearer_note_fr: str = ""


class Window(BaseModel):
    from_event: FactMatch | None = None
    deadline_days: int | None = None
    deadline_note_fr: str = ""


class Foreach(BaseModel):
    """Universal quantification over graph-derived instances.

    Present so that satisfying one institution's notification does not resolve
    the findings for the others.
    """

    kind: Literal["role_instances"] = "role_instances"
    actor_roles: list[str] = Field(min_length=1)
    key: Literal["person_id"] = "person_id"


class EvidenceScope(BaseModel):
    """The denominator that makes an absence defensible.

    If the documents where the required event *would* appear are not in the
    graph, or did not pass the faithfulness gate, the outcome is
    `unverifiable`, not `gap`. See ADR #70.
    """

    doc_id_patterns: list[str] = Field(default_factory=list)
    require_gate_status: str = "ok"


class Obligation(BaseModel):
    obligation_id: str
    rule_version: str
    title_fr: str
    severity: Literal["critical", "high", "medium", "low", "info"] = "medium"
    externalisable: bool = True
    source: Source
    also_anchored: list[Anchor] = Field(default_factory=list)
    bearer: Bearer
    window: Window = Field(default_factory=Window)
    foreach: Foreach | None = None
    expected_evidence: EvidencePredicate
    evidence_scope: EvidenceScope = Field(default_factory=EvidenceScope)
    absence_tier_cap: Tier = "T3"
    presence_tier_cap: Tier = "T1"
    adjudicate: Literal["none", "llm_local", "llm_judgment", "cloud"] = "none"
    claim_template_fr: str
    confounders_seed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Obligation":
        if not OBLIGATION_ID_PATTERN.match(self.obligation_id):
            raise ValueError(f"obligation_id {self.obligation_id!r} is not a stable slug")
        if not self.rule_version.strip():
            raise ValueError(f"{self.obligation_id}: rule_version must be non-empty")

        placeholders = {
            name
            for _, name, _, _ in string.Formatter().parse(self.claim_template_fr)
            if name
        }
        unknown = placeholders - CLAIM_PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"{self.obligation_id}: claim_template_fr uses non-stable "
                f"placeholders {sorted(unknown)}; volatile values belong in evidence"
            )
        if self.foreach is None:
            bound = [leaf for leaf in self.expected_evidence.leaves() if leaf.bind_to_foreach]
            if bound:
                raise ValueError(
                    f"{self.obligation_id}: bind_to_foreach set but no foreach declared"
                )
            if placeholders & {"foreach_role", "foreach_key"}:
                raise ValueError(
                    f"{self.obligation_id}: claim_template_fr references a foreach "
                    "placeholder but no foreach is declared"
                )
        if any(
            leaf.after_trigger or leaf.before_or_at_trigger
            for leaf in self.expected_evidence.leaves()
        ) and self.window.from_event is None:
            raise ValueError(
                f"{self.obligation_id}: a leaf is trigger-relative but window.from_event "
                "is null"
            )
        return self


# ========== runtime types ==========


class PassResult(NamedTuple):
    """What every pass returns.

    `complete=False` means the pass did not reach the end of its subject list —
    a budget cutoff, a truncated sweep, an absent credential. The orchestrator
    then skips reconciliation, because an unreached finding is not a fixed one.
    Carried on the result rather than in a module global (which is what
    `scripts/local_audit/passes/slimming.py` does) so concurrent per-case runs
    stay independent.
    """

    findings: list[dict]
    complete: bool


@dataclass(frozen=True)
class GraphHealth:
    """Substrate quality, produced by the `graph` pass and consumed by tiering.

    A defect in the substrate invalidates every downstream negative, so a
    finding resting on a defective fact is capped at T5 regardless of what its
    predicate found.
    """

    coverage_known: bool
    defective_fact_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Evaluation:
    """One obligation evaluated against one graph (or one foreach instance)."""

    obligation_id: str
    status: EvaluationStatus
    matched_fact_ids: tuple[str, ...] = ()
    matched_doc_ids: tuple[str, ...] = ()
    matched_docs_all_gate_ok: bool = False
    # Deterministically ranked, <= CHECK_MAX_CANDIDATES. Unused in Phase 1;
    # this is the exact input the Phase 2 rescue tier consumes, so adding that
    # layer touches no other module.
    candidate_fact_ids: tuple[str, ...] = ()
    scope_doc_ids: tuple[str, ...] = ()
    scope_covered: bool = False
    trigger_fact_id: str | None = None
    window_breach_days: int | None = None
    foreach_key: str | None = None
    foreach_role: str | None = None


@dataclass(frozen=True)
class RunContext:
    case_id: str
    paths: "CasePaths"
    graph: "CaseGraph"
    catalog: "Catalog"
    budget: "Budget"
    health: GraphHealth
    sha: str | None = None


# ========== tier derivation ==========


def weaker(a: str, b: str) -> str:
    """The weaker of two tiers (the larger index in TIER_ORDER)."""
    return a if TIER_ORDER[a] >= TIER_ORDER[b] else b


def at_least(tier: str, minimum: str) -> bool:
    """True when `tier` is at least as strong as `minimum`."""
    return TIER_ORDER[tier] <= TIER_ORDER[minimum]


def assign_tier(
    ev: Evaluation,
    health: GraphHealth,
    obligation: Obligation,
    sc_meta: dict | None = None,
) -> tuple[str, str]:
    """Derive `(tier, tier_basis)` for one evaluation. Pure.

    Presence needs no coverage denominator — a verbatim quote is either in the
    graph or it is not. Absence does, which is why an uncovered scope can never
    produce better than T5. The obligation's own cap is applied last and can
    only weaken the result.
    """
    if ev.status in PRESENCE_STATUSES:
        if set(ev.matched_fact_ids) & health.defective_fact_ids:
            tier, basis = "T5", "presence_defaut_integrite"
        elif (
            len(set(ev.matched_doc_ids)) >= 2
            and ev.matched_docs_all_gate_ok
            and health.coverage_known
        ):
            tier, basis = "T1", "presence_multi_source_couvert"
        elif ev.matched_fact_ids:
            tier, basis = "T2", "presence_source_unique"
        else:
            tier, basis = "T5", "presence_sans_fait"
        cap = obligation.presence_tier_cap
    elif ev.status in ABSENCE_STATUSES:
        if set(ev.candidate_fact_ids) & health.defective_fact_ids:
            tier, basis = "T5", "absence_defaut_integrite"
        elif ev.scope_covered and health.coverage_known:
            tier, basis = "T3", "absence_perimetre_couvert"
        else:
            tier, basis = "T5", "absence_perimetre_non_couvert"
        cap = obligation.absence_tier_cap
    else:
        raise ValueError(f"{ev.obligation_id}: status {ev.status!r} produces no finding")

    if sc_meta is not None:
        # An adjudicated verdict without deterministic corroboration cannot
        # outrank T4, however confident the vote was.
        tier = weaker(tier, "T4")
        basis = f"{basis}+adjudication"

    return weaker(tier, cap), basis


# ========== the externalisation gate ==========


def _person_redaction_map(findings: Iterable[dict]) -> dict[str, str]:
    seen = sorted(
        {
            pid
            for f in findings
            for pid in f.get("evidence_pointers", {}).get("person_ids", [])
        }
    )
    return {pid: f"personne#{i}" for i, pid in enumerate(seen, start=1)}


def _redact(finding: dict, mapping: dict[str, str]) -> dict:
    out = dict(finding)
    pointers = dict(out.get("evidence_pointers", {}))
    pointers["person_ids"] = [
        mapping.get(pid, "personne#?") for pid in pointers.get("person_ids", [])
    ]
    out["evidence_pointers"] = pointers
    for field in ("claim", "evidence", "subject"):
        text = out.get(field)
        if isinstance(text, str):
            for pid, label in mapping.items():
                text = text.replace(pid, label)
            out[field] = text
    return out


def externalisable_findings(
    findings: Iterable[dict],
    case_id: str,
    min_tier: str = config.OUTBOUND_MIN_TIER,
    obligations: dict[str, Obligation] | None = None,
) -> list[dict]:
    """The ONLY path from the findings store to any outbound artifact.

    Drops anything weaker than `min_tier`, anything marked non-externalisable
    on the finding or on its obligation, anything already neutralised by a
    dispositive confounder, and anything still open to revision. Person ids are
    replaced by positional labels unless the case is one whose identifiers are
    already personas.

    Raises for a case that is not outbound-eligible at all — being unable to
    name a real case here is the point, not an inconvenience to route around.
    """
    if case_id not in config.ALLOW_OUTBOUND_CASES:
        raise ValueError(
            f"case {case_id!r} is not outbound-eligible; "
            f"ALLOW_OUTBOUND_CASES = {sorted(config.ALLOW_OUTBOUND_CASES)}"
        )
    if min_tier not in TIER_ORDER:
        raise ValueError(f"unknown tier: {min_tier!r}")

    kept: list[dict] = []
    for f in findings:
        tier = f.get("tier")
        if tier not in TIER_ORDER or not at_least(tier, min_tier):
            continue
        if not f.get("externalisable", False):
            continue
        if f.get("status") != "open":
            continue
        if any(c.get("dispositive") for c in f.get("confounders", [])):
            continue
        oid = f.get("obligation_id")
        if obligations is not None and oid is not None:
            ob = obligations.get(oid)
            if ob is not None and not ob.externalisable:
                continue
        kept.append(f)

    kept.sort(key=lambda f: (TIER_ORDER[f["tier"]], f["pass"], f["id"]))
    if case_id in config.CASE_IS_ANONYMISED:
        return kept
    mapping = _person_redaction_map(kept)
    return [_redact(f, mapping) for f in kept]
