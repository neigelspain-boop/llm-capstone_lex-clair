"""Graph integrity — the substrate check every later negative depends on.

This pass does not extract anything. Extraction is Plane Ib's job and re-running
it would renumber `fact_id`s and dangle every person edge (see `graph.py`), so
Plane V loads what Plane Ib produced and checks that it hangs together.

It earns its slot because absence is this plane's primary signal, and **you may
not claim "absent" over a corrupt graph**. A dangling person edge, a fact whose
role was never catalogued, a document that produced no facts at all — each is a
reason a later "nothing attests this obligation" might be an artefact of the
substrate rather than a fact about the case. Reporting them is what lets the
tier function cap those findings honestly instead of the operator discovering
the problem after acting on one.

Zero LLM calls, zero network, no cache: the whole pass is dict work over an
already-loaded graph.
"""
from __future__ import annotations

from investigator import store
from investigator.graph import CaseGraph
from investigator.schema import GraphHealth, PassResult, RunContext

PASS_NAME = "graph"

# Closed enum. A finding's claim is its identity, so these strings are
# permanent — rewording one re-mints every finding of that kind.
CLAIMS = {
    "fact_ids_orphelins": "Intégrité du graphe : des identifiants de faits cités dans persons.jsonl n'existent pas.",
    "role_non_catalogue": "Intégrité du graphe : des faits portent un rôle absent de actor_roles.jsonl.",
    "document_sans_fait": "Intégrité du graphe : des documents extraits n'ont produit aucun fait.",
    "couverture_absente": "Intégrité du graphe : aucune donnée de couverture (coverage.jsonl) pour ce dossier.",
    "couverture_incomplete": "Intégrité du graphe : des documents n'ont pas passé le contrôle de fidélité.",
    "fait_illisible": "Intégrité du graphe : des lignes de facts.jsonl n'ont pas pu être validées.",
    "role_ambigu": "Intégrité du graphe : des faits portent une attribution de rôle non résolue.",
}


# ========== health ==========


def graph_health(graph: CaseGraph) -> GraphHealth:
    """Derive the substrate quality that tier assignment consumes.

    `coverage_known` folds in parse health deliberately: if some facts could
    not be read, the denominator an absence is measured against is unknown even
    where `coverage.jsonl` exists, and a gap must not be reported as though the
    record were complete.

    Facts with an unresolved role assignment are marked defective rather than
    dropped. Their quotes are real; what is disputed is *whose* conduct they
    evidence, which is precisely the inference an obligation's bearer clause
    makes.
    """
    return GraphHealth(
        coverage_known=graph.coverage_known and graph.is_healthy(),
        defective_fact_ids=graph.ambiguous_fact_ids,
    )


# ========== the pass ==========


def _finding(graph: CaseGraph, kind: str, evidence: str, severity: str) -> dict:
    return store.finding(
        PASS_NAME,
        subject=kind,
        claim=CLAIMS[kind],
        case_id=graph.case_id,
        evidence=evidence,
        severity=severity,
        confidence="high",
        tier="T1",
        tier_basis="constat_deterministe_sur_artefacts",
        externalisable=False,
        related_files=[f"data/dossier/{graph.case_id}/"],
    )


def run(ctx: RunContext) -> PassResult:
    """Report every substrate defect. Total by construction, so always complete."""
    graph = ctx.graph
    findings: list[dict] = []

    dangling = {
        pid: sorted(fids - set(graph.facts))
        for pid, fids in graph.person_facts.items()
        if fids - set(graph.facts)
    }
    if dangling:
        total = sum(len(v) for v in dangling.values())
        sample = ", ".join(sorted(dangling)[:5])
        findings.append(
            _finding(
                graph,
                "fact_ids_orphelins",
                f"{total} référence(s) orpheline(s) sur {len(dangling)} personne(s) : {sample}",
                "high",
            )
        )

    uncatalogued = sorted({f.actor_role for f in graph.facts.values()} - set(graph.roles))
    if uncatalogued:
        findings.append(
            _finding(
                graph,
                "role_non_catalogue",
                f"{len(uncatalogued)} rôle(s) : {', '.join(uncatalogued[:8])}",
                "medium",
            )
        )

    with_facts = {f.source_doc_id for f in graph.facts.values()}
    barren = sorted(set(graph.doc_text_folded) - with_facts)
    if barren:
        findings.append(
            _finding(
                graph,
                "document_sans_fait",
                f"{len(barren)} document(s) sur {len(graph.doc_text_folded)} : "
                f"{', '.join(barren[:5])}",
                "medium",
            )
        )

    if not graph.coverage_known:
        findings.append(
            _finding(
                graph,
                "couverture_absente",
                "Sans coverage.jsonl, une absence ne peut pas être distinguée d'un "
                "défaut de collecte : toute constatation d'absence est plafonnée à T5.",
                "high",
            )
        )
    else:
        failed = sorted(d for d, status in graph.coverage.items() if status != "ok")
        if failed:
            findings.append(
                _finding(
                    graph,
                    "couverture_incomplete",
                    f"{len(failed)} document(s) hors statut ok : {', '.join(failed[:5])}",
                    "medium",
                )
            )

    if graph.unparsed_fact_lines:
        findings.append(
            _finding(
                graph,
                "fait_illisible",
                f"{graph.unparsed_fact_lines} ligne(s) de facts.jsonl invalide(s) — "
                "le dénominateur des constatations d'absence est incomplet.",
                "critical",
            )
        )

    if graph.ambiguous_fact_ids:
        findings.append(
            _finding(
                graph,
                "role_ambigu",
                f"{len(graph.ambiguous_fact_ids)} fait(s) sur {len(graph.facts)} ; "
                "exclus des prédicats portant sur un rôle, et plafonnés à T5 lorsqu'ils "
                "étayent une constatation.",
                "info",
            )
        )

    return PassResult(findings=findings, complete=True)
