"""Citation validation — does the catalog say what the sources actually say?

Validates the *catalog*, not the case. Every obligation asserts a source and
quotes it; this pass checks that the quoted text exists, is in force, and is
attributed correctly. It runs before `check` in the cycle, so a bad citation is
known before any obligation derived from it starts producing gaps.

**Zero LLM calls, by rule.** Every question here is a string comparison after
normalisation, and no LLM pass may re-derive what a deterministic check
answers.

This is not hypothetical rigour. The first draft of this project's catalog
anchored a unanimity obligation to "Code civil, art. 730-4 al. 2". The article
has one alinéa, and it *enables* release of estate funds in the proportion
stated in the acte de notoriété — close to the opposite. Layer 0 catches that
with no network access at all.

An anchor absent from the local corpus is not an error: the statute corpus is
792 chunks across 9 sources, and `cc-1204`, `cc-1344` and `cc-494-12` are
genuinely outside it. Those obligations ship with a `legiarti_id` and no
`chunk_id`, and are reported as unverifiable-here rather than wrong — a
distinction the live PISTE layer (deferred, ADR #70 follow-up (d)) closes.
"""
from __future__ import annotations

import logging
from fnmatch import fnmatch

from investigator import store
from investigator.graph import CaseGraph
from investigator.lexicon import fold
from investigator.schema import Obligation, PassResult, RunContext

log = logging.getLogger(__name__)

PASS_NAME = "search"

# Claims are identity, so each kind is a permanent string. `{obligation_id}`
# and `{source_ref}` are the only interpolations, and both are catalog-stable.
CLAIMS = {
    "chunk_absent": (
        "Citation {obligation_id} : le chunk_id cité pour {source_ref} est absent du "
        "corpus local."
    ),
    "non_ancree": (
        "Citation {obligation_id} : {source_ref} n'est ancrée à aucun chunk du corpus "
        "local et ne peut être vérifiée hors ligne."
    ),
    "abroge": (
        "Citation {obligation_id} : {source_ref} n'est pas en vigueur dans le corpus."
    ),
    "extrait_absent": (
        "Citation {obligation_id} : l'extrait cité n'apparaît pas dans le texte de "
        "{source_ref}."
    ),
    "legiarti_divergent": (
        "Citation {obligation_id} : l'identifiant LEGIARTI cité ne correspond pas à "
        "celui du corpus pour {source_ref}."
    ),
    "ancre_penale_absente": (
        "Citation {obligation_id} : une ancre pénale de {source_ref} est absente du "
        "corpus — la qualification pénale ne peut pas être étayée."
    ),
    "ancre_penale_abrogee": (
        "Citation {obligation_id} : une ancre pénale de {source_ref} n'est pas en "
        "vigueur dans le corpus."
    ),
    "ancre_secondaire_absente": (
        "Citation {obligation_id} : une ancre secondaire de {source_ref} est absente "
        "du corpus."
    ),
    "document_absent": (
        "Citation {obligation_id} : aucun document du dossier ne correspond au motif "
        "cité pour {source_ref}."
    ),
    "extrait_document_absent": (
        "Citation {obligation_id} : l'extrait cité pour {source_ref} n'apparaît dans "
        "aucun document correspondant du dossier."
    ),
}


def _finding(ctx: RunContext, obligation: Obligation, kind: str, evidence: str,
             severity: str) -> dict:
    return store.finding(
        PASS_NAME,
        subject=f"{obligation.obligation_id}@{obligation.rule_version}",
        claim=CLAIMS[kind].format(
            obligation_id=obligation.obligation_id, source_ref=obligation.source.ref
        ),
        case_id=ctx.case_id,
        evidence=evidence,
        severity=severity,
        confidence="high",
        tier="T1",
        tier_basis="verification_textuelle_deterministe",
        # A defect in our own catalog is an internal correction, never
        # something to put in front of a third party.
        externalisable=False,
        obligation_id=obligation.obligation_id,
        evidence_pointers={
            "fact_ids": [],
            "doc_ids": [],
            "chunk_ids": [obligation.source.chunk_id] if obligation.source.chunk_id else [],
            "statute_refs": [obligation.source.ref],
            "person_ids": [],
        },
        related_files=[
            ctx.catalog.origin.get(obligation.obligation_id, ""),
            "data/chunks.csv",
        ],
    )


# ========== per-source-kind validation ==========


def _validate_statute(ctx: RunContext, obligation: Obligation, graph: CaseGraph) -> list[dict]:
    src = obligation.source
    out: list[dict] = []

    if not src.chunk_id:
        return [
            _finding(
                ctx,
                obligation,
                "non_ancree",
                f"legiarti_id={src.legiarti_id} ; hors du corpus local "
                f"({len(graph.statute)} chunks). Vérification en ligne requise.",
                "low",
            )
        ]

    row = graph.statute.get(src.chunk_id)
    if row is None:
        return [
            _finding(
                ctx, obligation, "chunk_absent", f"chunk_id={src.chunk_id}", "high"
            )
        ]

    if str(row.get("etat", "")).upper() != "VIGUEUR":
        out.append(
            _finding(
                ctx, obligation, "abroge",
                f"chunk_id={src.chunk_id} etat={row.get('etat')}", "critical",
            )
        )

    if src.legiarti_id and row.get("legiarti_id") and src.legiarti_id != row["legiarti_id"]:
        out.append(
            _finding(
                ctx, obligation, "legiarti_divergent",
                f"catalogue={src.legiarti_id} corpus={row['legiarti_id']}", "high",
            )
        )

    if fold(src.excerpt_fr) not in fold(str(row.get("texte", ""))):
        out.append(
            _finding(
                ctx, obligation, "extrait_absent",
                f"chunk_id={src.chunk_id} · extrait cité de {len(src.excerpt_fr)} "
                f"caractères · texte du corpus de {len(str(row.get('texte', '')))}",
                "critical",
            )
        )

    for anchor in obligation.also_anchored:
        if anchor.chunk_id not in graph.statute:
            out.append(
                _finding(
                    ctx, obligation, "ancre_secondaire_absente",
                    f"chunk_id={anchor.chunk_id}", "medium",
                )
            )
    return out


def _validate_document(ctx: RunContext, obligation: Obligation, graph: CaseGraph) -> list[dict]:
    """Contract clauses and deontology anchored to a case document.

    Verified against the extracted markdown rather than the chunk table: a
    clause routinely spans a chunk boundary, and a boundary is an artefact of
    the chunker, not of the document.
    """
    src = obligation.source
    if not src.doc_id_pattern:
        return []

    matching = sorted(d for d in graph.doc_text_folded if fnmatch(d, src.doc_id_pattern))
    if not matching:
        return [
            _finding(
                ctx, obligation, "document_absent",
                f"motif={src.doc_id_pattern} · {len(graph.doc_text_folded)} document(s) "
                "extraits dans le dossier",
                "medium",
            )
        ]

    needle = fold(src.excerpt_fr)
    if any(needle in graph.doc_text_folded[d] for d in matching):
        return []
    return [
        _finding(
            ctx, obligation, "extrait_document_absent",
            f"motif={src.doc_id_pattern} · {len(matching)} document(s) correspondant(s) : "
            + ", ".join(matching[:4]),
            "high",
        )
    ]


# ========== the pass ==========


def _validate_penal_anchors(
    ctx: RunContext, obligation: Obligation, graph: CaseGraph
) -> list[dict]:
    """Penal anchors are checked for every obligation, whatever its source kind.

    A contract-clause obligation may carry a penal characterisation — the
    prelevement clause does — so this cannot live inside the statute branch.
    The point is that naming an offence requires a citable, in-force article:
    without this check, `gravite_interne: susceptible_qualification_penale`
    would be an assertion rather than a citation.
    """
    out: list[dict] = []
    for anchor in obligation.penal_anchors:
        row = graph.statute.get(anchor.chunk_id)
        if row is None:
            out.append(
                _finding(ctx, obligation, "ancre_penale_absente",
                         f"chunk_id={anchor.chunk_id}", "high")
            )
        elif str(row.get("etat", "")).upper() != "VIGUEUR":
            out.append(
                _finding(ctx, obligation, "ancre_penale_abrogee",
                         f"chunk_id={anchor.chunk_id} etat={row.get('etat')}", "critical")
            )
    return out


def run(ctx: RunContext) -> PassResult:
    """Validate every citation in the merged catalog.

    Layer 0 only. `complete=True` because the offline sweep is total; when the
    PISTE layer lands it must return False on a missing credential, since an
    absent API key resolving real findings would be the worst possible failure
    mode for a pass whose whole job is telling you when a citation is wrong.
    """
    findings: list[dict] = []
    for obligation in ctx.catalog.ordered():
        if obligation.source.kind == "statute":
            findings.extend(_validate_statute(ctx, obligation, ctx.graph))
        else:
            findings.extend(_validate_document(ctx, obligation, ctx.graph))
        findings.extend(_validate_penal_anchors(ctx, obligation, ctx.graph))
    return PassResult(findings=findings, complete=True)
