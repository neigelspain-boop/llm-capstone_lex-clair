"""Facts that cannot all be true — deterministic candidate generation.

The candidate generator *is* the pass. No model sees the corpus: pairs are
produced by four authored buckets over normalised text, and each collision is
reported on its own at low confidence and T5. Phase 2 adds an adjudication
layer on top of the same pair list, and if that layer proves unreliable it can
be removed without this pass losing anything.

Buckets, all over `lexicon.py`'s person-free vocabulary:

- `montant` — the same whole-euro amount asserted in facts that differ in date
  or in actor. Two institutions describing one payment differently is the
  signal; two-cent rounding is not, so cents are dropped from the bucket key.
- `date_instrument` — the same instrument assigned different dates.
- `sens_action` — the same actor and target with action verbs on opposite
  sides of an antonym axis.
- `denombrement` — incompatible party counts.

**Ordering is part of the contract.** Pairs are sorted `(bucket, a, b)` with
the lexicographically smaller `fact_id` first, and the cap truncates that
stable order. An unstable cap would re-mint finding ids every cycle and destroy
resolve-on-fix, which is the single easiest way to get this pass wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

from investigator import config, store
from investigator.graph import MATCH_FIELDS, CaseGraph
from investigator.lexicon import action_polarity, parse_amounts_eur, parse_party_counts
from investigator.schema import PassResult, RunContext

log = logging.getLogger(__name__)

PASS_NAME = "contradict"

BUCKET_LABELS = {
    "montant": "un même montant",
    "date_instrument": "une même date d'instrument",
    "sens_action": "un sens d'action opposé",
    "denombrement": "un dénombrement de parties",
}


@dataclass(frozen=True)
class FactPair:
    """Pair identity is order-independent: `fact_id_a` is always the smaller."""

    fact_id_a: str
    fact_id_b: str
    bucket: str
    bucket_key: str


# ========== candidate generation ==========


def _blob(graph: CaseGraph, fact_id: str) -> str:
    return graph.fact_text(fact_id, MATCH_FIELDS)


def _bucket_by_amount(graph: CaseGraph) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for fid in graph.facts:
        for amount in parse_amounts_eur(_blob(graph, fid)):
            out.setdefault(("montant", str(amount)), set()).add(fid)
    return out


def _bucket_by_count(graph: CaseGraph) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for fid in graph.facts:
        for count, noun in parse_party_counts(_blob(graph, fid)):
            out.setdefault(("denombrement", noun), set()).add(fid)
    return out


def _bucket_by_instrument_date(graph: CaseGraph) -> dict[tuple[str, str], set[str]]:
    """Facts about one document that assert different dates for it."""
    out: dict[tuple[str, str], set[str]] = {}
    for fid, fact in graph.facts.items():
        if fact.date:
            out.setdefault(("date_instrument", fact.source_doc_id), set()).add(fid)
    return out


def _polarity_pairs(graph: CaseGraph) -> list[FactPair]:
    """Same actor and target, opposite side of an antonym axis."""
    by_key: dict[tuple[str, str, str], dict[int, list[str]]] = {}
    for fid, fact in graph.facts.items():
        polarity = action_polarity(graph.folded[fid]["action"])
        if polarity is None:
            continue
        axis, sign = polarity
        key = (fact.actor_role, graph.folded[fid]["target"], axis)
        by_key.setdefault(key, {}).setdefault(sign, []).append(fid)

    pairs: list[FactPair] = []
    for (_, _, axis), sides in by_key.items():
        for a in sorted(sides.get(1, [])):
            for b in sorted(sides.get(-1, [])):
                lo, hi = sorted((a, b))
                pairs.append(FactPair(lo, hi, "sens_action", axis))
    return pairs


def candidate_pairs(graph: CaseGraph, max_pairs: int = config.CONTRADICT_MAX_PAIRS) -> list[FactPair]:
    """Every candidate incompatibility, deterministically ordered and capped."""
    pairs: list[FactPair] = []

    buckets: dict[tuple[str, str], set[str]] = {}
    buckets.update(_bucket_by_amount(graph))
    buckets.update(_bucket_by_count(graph))
    buckets.update(_bucket_by_instrument_date(graph))

    for (bucket, key), fact_ids in buckets.items():
        if len(fact_ids) < 2:
            continue
        for a, b in combinations(sorted(fact_ids), 2):
            fa, fb = graph.facts[a], graph.facts[b]
            if bucket == "date_instrument":
                # Same document, different asserted dates.
                if fa.date == fb.date:
                    continue
            elif fa.date == fb.date and fa.actor_role == fb.actor_role:
                # Same actor on the same day restating one thing is not a
                # contradiction candidate, it is corroboration.
                continue
            pairs.append(FactPair(a, b, bucket, key))

    pairs.extend(_polarity_pairs(graph))
    pairs.sort(key=lambda p: (p.bucket, p.bucket_key, p.fact_id_a, p.fact_id_b))
    return pairs[:max_pairs]


# ========== the pass ==========


def run(ctx: RunContext) -> PassResult:
    """Report every candidate collision.

    `complete` is False when the cap truncated the list: the pairs beyond it
    were never examined, and reconciling would mark them fixed.
    """
    graph = ctx.graph
    all_pairs = candidate_pairs(graph, max_pairs=10**9)
    kept = all_pairs[: config.CONTRADICT_MAX_PAIRS]

    findings = []
    for pair in kept:
        fa, fb = graph.facts[pair.fact_id_a], graph.facts[pair.fact_id_b]
        findings.append(
            store.finding(
                PASS_NAME,
                subject=f"{pair.bucket}|{pair.fact_id_a}|{pair.fact_id_b}",
                claim=(
                    f"Collision {BUCKET_LABELS[pair.bucket]} entre deux faits du dossier "
                    f"({pair.bucket})."
                ),
                case_id=ctx.case_id,
                evidence=(
                    f"clé={pair.bucket_key} · "
                    f"{pair.fact_id_a} [{fa.actor_role} {fa.date}] « {fa.verbatim_quote[:110]} » ‖ "
                    f"{pair.fact_id_b} [{fb.actor_role} {fb.date}] « {fb.verbatim_quote[:110]} »"
                ),
                severity="low",
                confidence="low",
                # Layer 0 alone cannot tell an incompatibility from a
                # coincidence of vocabulary; the tier says so.
                tier="T5",
                tier_basis="collision_deterministe_non_adjugee",
                externalisable=False,
                evidence_pointers={
                    "fact_ids": [pair.fact_id_a, pair.fact_id_b],
                    "doc_ids": sorted({fa.source_doc_id, fb.source_doc_id}),
                    "chunk_ids": [],
                    "statute_refs": [],
                    "person_ids": sorted(
                        graph.fact_persons.get(pair.fact_id_a, frozenset())
                        | graph.fact_persons.get(pair.fact_id_b, frozenset())
                    ),
                },
            )
        )

    complete = len(all_pairs) <= config.CONTRADICT_MAX_PAIRS
    if not complete:
        log.warning(
            "contradict: %d candidate pairs, capped at %d — not reconciling",
            len(all_pairs),
            config.CONTRADICT_MAX_PAIRS,
        )
    return PassResult(findings=findings, complete=complete)
