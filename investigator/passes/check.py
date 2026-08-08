"""Obligation × graph → gap, unverifiable, window breach, or satisfied.

The pass that generates the questions. For every obligation in the merged
catalog it asks: did the trigger fire, is the record that would show compliance
actually in the dossier, and does a fact attest the required conduct?

`evaluate()` is pure and calls nothing. Every leaf of an obligation's predicate
ranges over a finite candidate set drawn from the graph, so `matches == 0` is a
decidable fact rather than a judgment — which is the only reason an absence can
be reported at all.

Two distinctions do the real work:

- **`gap` is not `unverifiable`.** The first says the obligation was not
  performed; the second says the record needed to test it is not in the
  dossier. Conflating them is how an absence-driven system starts inventing
  breaches, so the scope check runs before the verdict and can only ever
  weaken it.
- **Undated facts never manufacture a gap.** Where an ordering cannot be
  established, a fact still counts towards satisfaction but cannot establish a
  deadline breach. 61 of vitrine's 235 facts carry no date; excluding them from
  matching would turn missing metadata into fabricated breaches.

Phase 1 makes no model call. `Evaluation.candidate_fact_ids` is computed and
carried anyway — it is exactly the input the Phase 2 rescue tier consumes, so
that layer arrives without touching this module's verdict logic.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from fnmatch import fnmatch

from dataclasses import replace

from investigator import cache, config, ollama, store
from investigator.graph import MATCH_FIELDS, CaseGraph
from investigator.lexicon import contains_term
from investigator.schema import (
    Evaluation,
    FactMatch,
    Obligation,
    PassResult,
    RunContext,
    assign_tier,
)

log = logging.getLogger(__name__)

PASS_NAME = "check"

# `gap` uses the catalog's own wording; the other outcomes use fixed frames so
# an author cannot accidentally give two statuses the same claim, hence the
# same finding id. All are identity — never interpolate a count or a date.
CLAIM_FRAMES = {
    "unverifiable": (
        "Obligation {obligation_id} ({source_ref}) : le périmètre documentaire "
        "nécessaire au contrôle n'est pas couvert par le dossier."
    ),
    "window_breach": (
        "Obligation {obligation_id} ({source_ref}) : l'exécution n'est attestée "
        "qu'au-delà de l'échéance."
    ),
    "satisfied": (
        "Obligation {obligation_id} ({source_ref}) : l'exécution est attestée par "
        "le dossier."
    ),
}
FOREACH_SUFFIX = " — {foreach_role}/{foreach_key}"


# ========== date helpers ==========


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# ========== predicate evaluation ==========


def _structural_candidates(
    leaf: FactMatch,
    graph: CaseGraph,
    trigger_date: date | None,
    bound_fact_ids: frozenset[str] | None,
    default_roles: tuple[str, ...] = (),
) -> list[str]:
    """Facts a leaf could match, before any term comparison.

    `default_roles` is what an evidence leaf falls back to when it names none:
    the obligation's bearer. Performance is conduct *by the party who owes it*,
    and without that default a single well-worded sentence anywhere in the
    dossier reports the obligation satisfied — a false negative on a gap, which
    is the one direction no later adjudication layer can repair. Trigger leaves
    pass no defaults, because an event may be caused by anyone.

    A fact whose role assignment is unresolved is excluded only when a role
    filter actually applies: the quote is real, what is disputed is whose
    conduct it evidences.
    """
    roles = leaf.actor_roles or ([] if leaf.any_actor else list(default_roles))
    if roles:
        pool: list[str] = []
        for role in roles:
            pool.extend(graph.facts_by_role.get(role, ()))
    else:
        pool = list(graph.facts)
    role_filtered = bool(roles)

    out: list[str] = []
    for fid in pool:
        if role_filtered and fid in graph.ambiguous_fact_ids:
            continue
        if bound_fact_ids is not None and fid not in bound_fact_ids:
            continue
        if leaf.after_trigger or leaf.before_or_at_trigger:
            fact_date = _as_date(graph.facts[fid].date)
            if fact_date is not None and trigger_date is not None:
                if leaf.after_trigger and fact_date < trigger_date:
                    continue
                if leaf.before_or_at_trigger and fact_date > trigger_date:
                    continue
            # Unknown ordering is not a mismatch — see the module docstring.
        out.append(fid)
    return sorted(set(out))


def _leaf_matches(
    leaf: FactMatch,
    graph: CaseGraph,
    trigger_date: date | None,
    bound_fact_ids: frozenset[str] | None,
    default_roles: tuple[str, ...] = (),
) -> list[str]:
    candidates = _structural_candidates(
        leaf, graph, trigger_date, bound_fact_ids, default_roles
    )
    if not leaf.any_terms_fr:
        return candidates
    matched = []
    for fid in candidates:
        blob = graph.fact_text(fid, leaf.match_fields)
        if any(contains_term(blob, term) for term in leaf.any_terms_fr):
            matched.append(fid)
    return matched


def _find_trigger(
    obligation: Obligation, graph: CaseGraph
) -> tuple[bool, str | None, date | None, bool]:
    """`(triggered, fact_id, trigger_date)`.

    A standing duty (`window.from_event is null`) is always triggered.
    Otherwise the trigger is the earliest or latest dated matching fact per
    `window.trigger_select`, falling back to the first by id so the choice
    stays deterministic on an entirely undated corpus.

    The fourth element is `ambiguous`: whether the matched facts disagree about
    *when* the triggering event happened. This corpus makes the problem
    concrete — the extinction trigger matches 45 dated facts spanning 1981 to
    2026 and recites three different deaths, because a dossier discusses
    earlier successions as background. Taking the earliest measured a deadline
    from a 1981 recital and produced a six-thousand-day "breach" at critical
    severity; taking the latest merely picks a different wrong one.

    Term matching cannot resolve which event is operative, so the honest move
    is to say so rather than choose. A trigger is still returned — the
    obligation is triggered, and its evidence can be evaluated — but callers
    must not compute a deadline from an ambiguous one.
    """
    event = obligation.window.from_event
    if event is None:
        return True, None, None, False
    matches = _leaf_matches(event, graph, trigger_date=None, bound_fact_ids=None)
    if not matches:
        return False, None, None, False
    dated = sorted(
        (d, fid) for fid in matches if (d := _as_date(graph.facts[fid].date)) is not None
    )
    if not dated:
        return True, matches[0], None, False
    ambiguous = len({d for d, _ in dated}) > 1
    chosen = dated[-1] if obligation.window.trigger_select == "latest" else dated[0]
    return True, chosen[1], chosen[0], ambiguous


def _scope(
    obligation: Obligation, graph: CaseGraph, trigger_fact_id: str | None = None
) -> tuple[tuple[str, ...], int]:
    """Documents where the required event would appear, and whether they're covered.

    With explicit `doc_id_patterns`, the author has said where to look. Without
    them — the normal case for a generic statute obligation, which cannot know
    a particular dossier's filing — the scope falls back to **the documents
    where the bearer's conduct is actually recorded**, plus the trigger's own
    document.

    The rejected alternative was "every document in the case". It is not more
    conservative, it is less useful in a way that hides real findings: this
    corpus has 8 documents of 55 whose gate verdict could not be parsed, so an
    all-documents scope is never fully covered and *every* obligation degrades
    to `unverifiable`. Narrowing to where the bearer appears keeps the coverage
    question answerable without ever asserting coverage the gate did not give.

    Returns `(docs, verified_count)`. Coverage is **graded, not binary**: on the
    real corpus 8 of 55 documents have an unparsed gate verdict, and requiring
    every document in scope to be verified threw away 28 confirmed documents
    because of 3 unknown ones — every obligation collapsed to `unverifiable`
    and no breach could ever be asserted. A wholly unverified scope still says
    nothing; a mostly-verified one says something weaker, and the tier carries
    that rather than the status discarding it.

    An empty scope stays uncovered by definition: nothing here could have shown
    compliance either way.
    """
    patterns = obligation.evidence_scope.doc_id_patterns
    if patterns:
        docs = tuple(
            sorted(d for d in graph.doc_ids if any(fnmatch(d, p) for p in patterns))
        )
    else:
        relevant = {
            graph.facts[fid].source_doc_id
            for role in obligation.bearer.actor_roles
            for fid in graph.facts_by_role.get(role, ())
        }
        if trigger_fact_id and trigger_fact_id in graph.facts:
            relevant.add(graph.facts[trigger_fact_id].source_doc_id)
        docs = tuple(sorted(relevant))
    if not docs or not graph.coverage_known:
        return docs, 0
    required = obligation.evidence_scope.require_gate_status
    return docs, sum(1 for d in docs if graph.coverage.get(d) == required)


def _rank_candidates(
    obligation: Obligation, graph: CaseGraph, trigger_date: date | None,
    bound_fact_ids: frozenset[str] | None,
) -> tuple[str, ...]:
    """Facts most worth a second look at this obligation. Deterministic.

    Unused in Phase 1. Ranked by how many of the obligation's terms a fact
    already mentions, then by id so ties never reorder between runs.
    """
    terms = [t for leaf in obligation.expected_evidence.leaves() for t in leaf.any_terms_fr]
    pool: set[str] = set()
    for role in obligation.bearer.actor_roles:
        pool.update(graph.facts_by_role.get(role, ()))
    if bound_fact_ids is not None:
        pool &= set(bound_fact_ids)

    scored = []
    for fid in pool:
        blob = graph.fact_text(fid, MATCH_FIELDS)
        hits = sum(1 for t in terms if contains_term(blob, t))
        scored.append((-hits, fid))
    return tuple(fid for _, fid in sorted(scored)[: config.CHECK_MAX_CANDIDATES])


def evaluate(
    obligation: Obligation,
    graph: CaseGraph,
    foreach_key: str | None = None,
    foreach_role: str | None = None,
) -> Evaluation:
    """Evaluate one obligation against one graph. Pure: no I/O, no network."""
    bound: frozenset[str] | None = None
    if foreach_key is not None:
        bound = graph.person_facts.get(foreach_key, frozenset())

    triggered, trigger_fact_id, trigger_date, trigger_ambiguous = _find_trigger(
        obligation, graph
    )
    scope_docs, scope_ok = _scope(obligation, graph, trigger_fact_id)
    scope_covered = bool(scope_docs) and scope_ok == len(scope_docs)

    if not triggered:
        return Evaluation(
            obligation_id=obligation.obligation_id,
            status="not_triggered",
            scope_doc_ids=scope_docs,
            scope_covered=scope_covered,
            scope_ok_docs=scope_ok,
            foreach_key=foreach_key,
            foreach_role=foreach_role,
        )

    predicate = obligation.expected_evidence
    bearer_roles = tuple(obligation.bearer.actor_roles)
    matched: set[str] = set()
    satisfied = True

    def _hits(leaf) -> list[str]:
        return _leaf_matches(
            leaf,
            graph,
            trigger_date,
            bound if leaf.bind_to_foreach else None,
            default_roles=bearer_roles,
        )

    for leaf in predicate.all_of:
        hits = _hits(leaf)
        if len(hits) < leaf.min_count:
            satisfied = False
        matched.update(hits)

    if predicate.any_of:
        any_ok = False
        for leaf in predicate.any_of:
            hits = _hits(leaf)
            if len(hits) >= leaf.min_count:
                any_ok = True
            matched.update(hits)
        satisfied = satisfied and any_ok

    for leaf in predicate.none_of:
        if len(_hits(leaf)) >= leaf.min_count:
            satisfied = False

    matched_ids = tuple(sorted(matched))
    matched_docs = tuple(sorted({graph.facts[f].source_doc_id for f in matched_ids}))
    breach_days: int | None = None

    if satisfied:
        status = "satisfied"
        deadline = obligation.window.deadline_days
        # A deadline is only testable against an unambiguous trigger. Where the
        # corpus disagrees about when the triggering event happened, the
        # obligation stays `satisfied` and the untested deadline is reported in
        # `evidence` — a claim of lateness resting on the wrong start date is
        # worse than no claim at all.
        if deadline is not None and trigger_date is not None and not trigger_ambiguous:
            dates = [d for f in matched_ids if (d := _as_date(graph.facts[f].date))]
            if dates:
                due = trigger_date + timedelta(days=deadline)
                if min(dates) > due:
                    status = "window_breach"
                    breach_days = (min(dates) - due).days
    else:
        # Any verified document in scope is enough to say the record was
        # searched. None at all means it was not — that stays `unverifiable`.
        status = "gap" if scope_ok else "unverifiable"

    return Evaluation(
        obligation_id=obligation.obligation_id,
        status=status,
        matched_fact_ids=matched_ids,
        matched_doc_ids=matched_docs,
        matched_docs_all_gate_ok=bool(matched_docs) and all(graph.gate_ok(d) for d in matched_docs),
        candidate_fact_ids=_rank_candidates(obligation, graph, trigger_date, bound),
        scope_doc_ids=scope_docs,
        scope_covered=scope_covered,
        scope_ok_docs=scope_ok,
        trigger_fact_id=trigger_fact_id,
        trigger_ambiguous=trigger_ambiguous,
        window_breach_days=breach_days,
        foreach_key=foreach_key,
        foreach_role=foreach_role,
    )


# ========== instances ==========


def instances(obligation: Obligation, graph: CaseGraph) -> list[tuple[str | None, str | None]]:
    """`(role, person_id)` pairs to evaluate, or a single `(None, None)`.

    A `foreach` obligation with no instances in the graph yields nothing at
    all, which is correct: there is no third party to have notified.
    """
    if obligation.foreach is None:
        return [(None, None)]
    out = [
        (role, pid)
        for role in obligation.foreach.actor_roles
        for pid in graph.persons_by_role.get(role, ())
    ]
    return sorted(set(out))


# ========== the pass ==========


def _claim(obligation: Obligation, ev: Evaluation) -> str:
    fields = {
        "obligation_id": obligation.obligation_id,
        "source_ref": obligation.source.ref,
        "title_fr": obligation.title_fr,
        "foreach_role": ev.foreach_role,
        "foreach_key": ev.foreach_key,
    }
    if ev.status == "gap":
        return " ".join(obligation.claim_template_fr.format(**fields).split())
    frame = CLAIM_FRAMES[ev.status]
    if ev.foreach_key is not None:
        frame += FOREACH_SUFFIX
    return frame.format(**fields)


def _evidence(
    ev: Evaluation,
    graph: CaseGraph,
    obligation: Obligation | None = None,
    adjudication_note: str = "",
) -> str:
    """Everything volatile. Never let any of this reach the claim."""
    bits = [
        f"statut={ev.status}",
        f"faits_retenus={len(ev.matched_fact_ids)}",
        f"périmètre={ev.scope_ok_docs}/{len(ev.scope_doc_ids)} document(s) vérifié(s)",
    ]
    if ev.trigger_fact_id:
        bits.append(f"déclencheur={ev.trigger_fact_id} ({graph.facts[ev.trigger_fact_id].date})")
    if ev.window_breach_days is not None:
        bits.append(f"retard={ev.window_breach_days} jour(s)")
    elif ev.trigger_ambiguous:
        bits.append("échéance non testée : date du fait déclencheur ambiguë dans le dossier")
    if ev.matched_fact_ids:
        bits.append("faits=" + ", ".join(ev.matched_fact_ids[:6]))
    elif ev.candidate_fact_ids:
        bits.append("candidats=" + ", ".join(ev.candidate_fact_ids[:6]))
    if adjudication_note:
        bits.append(adjudication_note)
    if (
        obligation is not None
        and obligation.absence_significance_fr
        and ev.status in ("gap", "unverifiable")
    ):
        bits.append("portée : " + obligation.absence_significance_fr.strip())
    return " · ".join(bits)


# ========== the rescue tier (Phase 2, local model) ==========

CLASSIFY_SYSTEM_PROMPT = """Tu es un vérificateur juridique. On te donne UNE clause \
d'obligation et UNE citation extraite d'un dossier.

Classe le rôle de cette citation par rapport à cette obligation précise :

- "execution"   : la citation atteste que l'obligation A ÉTÉ EXÉCUTÉE (acte accompli,
                  pièce remise, compte ouvert, somme versée, information délivrée).
- "stipulation" : la citation ÉNONCE l'obligation (texte de la convention, rappel de la
                  règle). Énoncer n'est pas exécuter.
- "demande"     : la citation RÉCLAME l'exécution, la relance ou s'en enquiert.
- "manquement"  : la citation indique que l'obligation N'A PAS été exécutée.
- "autre"       : sans rapport avec l'exécution de cette obligation.

ATTENTION : "execution" exige que l'acte décrit SOIT L'ACTE EXIGÉ PAR LA CLAUSE,
au profit du bénéficiaire désigné. Un acte quelconque accompli sur les mêmes biens
(vente, virement vers un autre compte, encaissement) N'EST PAS l'exécution de la
clause : c'est souvent l'inverse. Vérifie : qui reçoit ? est-ce ce qu'exige la clause ?

Seul "execution" vaut exécution. Dans le doute, ne réponds jamais "execution".

Réponds en JSON strict : {"role": "<une des cinq valeurs>", "motif": "<une phrase>"}"""


def _classify(ctx: RunContext, obligation: Obligation, fact_id: str) -> dict | None:
    """What role does one quote play with respect to one obligation?

    The whole reason this exists: a document *stipulating* a duty, one
    *demanding* it, one *reporting its breach* and one *evidencing performance*
    all share the same vocabulary. A term-matching predicate cannot tell them
    apart, and on the real corpus it read the sentence "les parts ont été cédées
    ... sans remploi documenté" — evidence of the breach — as evidence that
    restitution had been performed.

    One clause, one quote, five labels, cached per pair. `None` means no
    verdict, and leaves the deterministic result standing.

    The prompt's insistence that the act be *the act the clause requires, for
    the designated beneficiary* is load-bearing and was measured, not guessed:
    without it both qwen3:14b and qwen3:30b read "les parts ont été cédées et
    leur produit versé sur un compte ordinaire, sans remploi documenté" as
    `execution`, because an act had been performed on the right assets. With
    it, both return `manquement`. Model size did not decide that case; the
    question's precision did.
    """
    fact = ctx.graph.facts.get(fact_id)
    if fact is None:
        return None
    clause = obligation.source.excerpt_fr.strip()[: config.MAX_CLAUSE_CHARS]
    quote = fact.verbatim_quote.strip()[: config.MAX_QUOTE_CHARS]
    subject = f"{obligation.obligation_id}@{obligation.rule_version}|{ctx.case_id}|{fact_id}"
    content_hash = cache.content_hash_for_text(f"{quote}||{clause}")
    version = config.PROMPT_VERSIONS["check_performance"]

    cached = cache.get(ctx.paths, PASS_NAME, version, subject, content_hash)
    if cached is not None:
        return cached
    if not ctx.budget.take_local():
        return None
    verdict, meta = ollama.call_self_consistency(
        CLASSIFY_SYSTEM_PROMPT,
        f"Clause d'obligation :\n« {clause} »\n\nCitation du dossier :\n« {quote} »",
        verdict_key="role",
        model=ollama.resolve_model(ctx.local_model, config.OLLAMA_MODEL_RESCUE),
    )
    if verdict is None:
        return None  # never cached: no verdict is not a negative verdict
    result = dict(verdict)
    result["_self_consistency"] = meta
    cache.set(ctx.paths, PASS_NAME, version, subject, content_hash, result)
    return result


def _adjudicate(
    ctx: RunContext, obligation: Obligation, ev: Evaluation
) -> tuple[Evaluation, dict | None, str]:
    """Confirm or overturn a deterministic verdict. Both directions.

    The asymmetry that matters for this plane's purpose: a **false `satisfied`
    hides a breach**, which defeats the tool entirely, while a false `gap` costs
    the reader one check. So satisfaction is what gets verified hardest — an
    obligation counts as performed only if some quote is classified `execution`.

    A model-created gap is capped at T4 by `assign_tier` and carries the
    classifier's reason, so "never confidently invent a breach" survives while
    the dominant error is the one actually being corrected.
    """
    if obligation.adjudicate == "none":
        return ev, None, ""

    if ev.status in ("satisfied", "window_breach"):
        reasons = []
        for fact_id in ev.matched_fact_ids[: config.CHECK_MAX_CANDIDATES]:
            verdict = _classify(ctx, obligation, fact_id)
            if verdict is None:
                continue
            if verdict.get("role") == "execution":
                return ev, verdict.get("_self_consistency"), ""
            reasons.append(f"{fact_id}={verdict.get('role')}")
        if reasons:
            # Nothing in the matched set actually evidences performance.
            downgraded = replace(ev, status="gap", matched_fact_ids=(), matched_doc_ids=())
            return downgraded, None, "aucune pièce n'atteste l'exécution — " + ", ".join(reasons[:6])
        return ev, None, ""

    if ev.status == "gap":
        for fact_id in ev.candidate_fact_ids:
            verdict = _classify(ctx, obligation, fact_id)
            if verdict is None:
                continue
            if verdict.get("role") == "execution":
                fact = ctx.graph.facts[fact_id]
                rescued = replace(
                    ev,
                    status="satisfied",
                    matched_fact_ids=(fact_id,),
                    matched_doc_ids=(fact.source_doc_id,),
                    matched_docs_all_gate_ok=ctx.graph.gate_ok(fact.source_doc_id),
                )
                return rescued, verdict.get("_self_consistency"), ""
    return ev, None, ""


def run(ctx: RunContext) -> PassResult:
    """Evaluate every obligation, every instance.

    Deterministic by default. With `--local-llm`, every verdict whose obligation
    opts in is additionally adjudicated — satisfaction hardest, since a false
    `satisfied` hides a breach. `complete` goes False if the call budget ran out
    mid-sweep; a cutoff that reconciled would resolve real gaps and report the
    deletion as progress.
    """
    findings: list[dict] = []
    graph = ctx.graph
    complete = True
    adjudicating = ctx.local_model is not None and ollama.is_available()
    if ctx.local_model is not None and not adjudicating:
        log.warning("check: Ollama unreachable — deterministic verdicts only, pass marked partial")
        complete = False

    for obligation in ctx.catalog.ordered():
        for role, person_id in instances(obligation, graph):
            ev = evaluate(obligation, graph, foreach_key=person_id, foreach_role=role)
            if ev.status == "not_triggered":
                continue

            sc_meta, adjudication_note = None, ""
            if adjudicating:
                ev, sc_meta, adjudication_note = _adjudicate(ctx, obligation, ev)
                if ctx.budget.local_exhausted():
                    complete = False

            tier, basis = assign_tier(ev, ctx.health, obligation, sc_meta=sc_meta)
            subject = obligation.obligation_id
            if person_id is not None:
                subject = f"{obligation.obligation_id}|{role}|{person_id}"

            findings.append(
                store.finding(
                    PASS_NAME,
                    subject=subject,
                    claim=_claim(obligation, ev),
                    case_id=ctx.case_id,
                    evidence=_evidence(ev, graph, obligation, adjudication_note),
                    severity="info" if ev.status == "satisfied" else obligation.severity,
                    confidence="high" if tier in ("T1", "T2", "T3") else "low",
                    tier=tier,
                    tier_basis=basis,
                    externalisable=obligation.externalisable and ev.status != "satisfied",
                    obligation_id=obligation.obligation_id,
                    evidence_pointers={
                        "fact_ids": list(ev.matched_fact_ids),
                        "doc_ids": list(ev.matched_doc_ids),
                        "chunk_ids": [obligation.source.chunk_id] if obligation.source.chunk_id else [],
                        "statute_refs": [obligation.source.ref],
                        "person_ids": [person_id] if person_id else [],
                    },
                    related_files=[ctx.catalog.origin.get(obligation.obligation_id, "")],
                )
            )

    return PassResult(findings=findings, complete=complete)
