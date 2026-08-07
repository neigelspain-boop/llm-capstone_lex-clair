"""Plane V invariants.

Stance, mirroring tests/test_local_audit_smoke.py: assert *invariants*, never
counts. The number of findings, the tier distribution, and how many gaps the
catalog turns up on vitrine are all supposed to change as obligations are
authored and findings resolve. What must not change is that ids are stable,
that a partial run cannot resolve anything, that adversarial work survives a
re-check, and that a real case can never reach an outbound artifact.

Fast suite: no LLM call, no network, no Ollama. The only real data touched is
the committed `vitrine`/`demo` cases and `data/chunks.csv`.
"""
from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

import scripts.local_audit.cache as la_cache
import scripts.local_audit.findings as la_findings
from investigator import budget as budget_mod
from investigator import cache, catalog, config, graph, lexicon, schema, store
from investigator.schema import Evaluation, GraphHealth, Obligation

# ========== fixtures ==========


@pytest.fixture(scope="module")
def vitrine_graph():
    return graph.load_graph("vitrine")


@pytest.fixture
def paths(tmp_path):
    return config.CasePaths.for_case("vitrine", dossier_dir=tmp_path)


def _obligation(**overrides) -> Obligation:
    """A minimal valid obligation; overrides patch one field at a time."""
    base = {
        "obligation_id": "test-obligation",
        "rule_version": "v1",
        "title_fr": "Obligation de test",
        "severity": "high",
        "externalisable": True,
        "source": {
            "kind": "contract_clause",
            "ref": "Convention — clause de test",
            "doc_id_pattern": "*convention*",
            "excerpt_fr": "Le notaire remettra un extrait.",
        },
        "bearer": {"actor_roles": ["notaire_redacteur"]},
        "expected_evidence": {
            "all_of": [{"kind": "fact_match", "any_terms_fr": ["extrait"]}]
        },
        "claim_template_fr": "Obligation {obligation_id} ({source_ref}) : rien n'atteste l'exécution.",
    }
    base.update(overrides)
    return Obligation.model_validate(base)


def _finding(**overrides) -> dict:
    raw = store.finding(
        "check",
        "test-obligation",
        "Obligation test-obligation : rien n'atteste l'exécution.",
        case_id="vitrine",
        tier="T2",
        externalisable=True,
        obligation_id="test-obligation",
    )
    raw.update(overrides)
    return raw


# ========== group 2: fork equivalence with local_audit ==========


@pytest.mark.parametrize(
    "triple",
    [
        ("check", "qu-compte-dedie", "Obligation qu-compte-dedie : rien."),
        ("graph", "persons.jsonl", "Intégrité du graphe : fact_ids orphelins."),
        ("", "", ""),
        ("p|ipe", "sub|ject", "cl|aim"),
    ],
)
def test_make_id_matches_local_audit(triple):
    """The fork must not drift from its source formula.

    Two small honest modules beat one with a union schema (ADR #70), but a fork
    only stays defensible while a divergence is a decision rather than a
    discovery. This is that tripwire.
    """
    assert store.make_id(*triple) == la_findings.make_id(*triple)


@pytest.mark.parametrize(
    "quad",
    [
        ("check_rescue", "v1", "qu-compte-dedie@v1|vitrine|doc-f001", "abc123"),
        ("search", "v2", "piste|LEGIARTI000006430899", "deadbeef"),
    ],
)
def test_make_key_matches_local_audit(quad):
    assert cache.make_key(*quad) == la_cache.make_key(*quad)


# ========== group 3: catalog and obligation validation ==========


def test_claim_template_rejects_volatile_placeholders():
    """`claim` is identity, so a template may not interpolate a count."""
    with pytest.raises(ValidationError):
        _obligation(claim_template_fr="Obligation {obligation_id} : {fact_count} faits manquants.")


def test_claim_template_accepts_the_stable_allowlist():
    ob = _obligation(
        claim_template_fr="{title_fr} — {obligation_id} ({source_ref})"
    )
    assert "{obligation_id}" in ob.claim_template_fr


def test_foreach_placeholder_requires_a_foreach_block():
    with pytest.raises(ValidationError):
        _obligation(
            claim_template_fr="Obligation {obligation_id} : détenteur {foreach_key}."
        )


def test_bind_to_foreach_requires_a_foreach_block():
    with pytest.raises(ValidationError):
        _obligation(
            expected_evidence={
                "all_of": [{"kind": "fact_match", "bind_to_foreach": True}]
            }
        )


def test_trigger_relative_leaf_requires_a_trigger():
    """`after_trigger` with no `window.from_event` silently matches everything."""
    with pytest.raises(ValidationError):
        _obligation(
            expected_evidence={
                "all_of": [{"kind": "fact_match", "after_trigger": True}]
            }
        )


def test_statute_source_requires_an_anchor():
    """A statute obligation nothing can ever verify is worse than none."""
    with pytest.raises(ValidationError):
        _obligation(
            source={
                "kind": "statute",
                "ref": "Code civil, art. 999",
                "excerpt_fr": "Texte.",
            }
        )


def test_obligation_id_must_be_a_stable_slug():
    with pytest.raises(ValidationError):
        _obligation(obligation_id="Not A Slug")


def test_empty_predicate_is_rejected():
    with pytest.raises(ValidationError):
        _obligation(expected_evidence={})


def test_unknown_match_field_is_rejected():
    with pytest.raises(ValidationError):
        _obligation(
            expected_evidence={
                "all_of": [{"kind": "fact_match", "match_fields": ["not_a_field"]}]
            }
        )


def test_catalog_load_is_deterministically_ordered(tmp_path):
    cat_file = tmp_path / "c.yaml"
    cat_file.write_text(
        json.dumps(
            {
                "catalog_id": "t",
                "catalog_version": "v1",
                "person_free": True,
                "obligations": [
                    json.loads(_obligation(obligation_id="zzz-last").model_dump_json()),
                    json.loads(_obligation(obligation_id="aaa-first").model_dump_json()),
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = catalog.load("vitrine", dossier_dir=tmp_path, generic_path=cat_file)
    assert [o.obligation_id for o in loaded.ordered()] == ["aaa-first", "zzz-last"]


def test_catalog_overlay_wins_over_generic(tmp_path):
    generic = tmp_path / "generic.yaml"
    case_dir = tmp_path / "vitrine"
    case_dir.mkdir()
    overlay = case_dir / config.CASE_CATALOG_FILENAME

    def _write(path, severity):
        path.write_text(
            json.dumps(
                {
                    "catalog_id": "t",
                    "catalog_version": "v1",
                    "obligations": [
                        json.loads(_obligation(severity=severity).model_dump_json())
                    ],
                }
            ),
            encoding="utf-8",
        )

    _write(generic, "low")
    _write(overlay, "critical")
    loaded = catalog.load("vitrine", dossier_dir=tmp_path, generic_path=generic)
    assert loaded.get("test-obligation").severity == "critical"


def test_shipped_catalog_is_person_free_if_it_exists():
    """The committed catalog must never name a party.

    Skipped until the file is authored; it becomes load-bearing the moment it
    exists, which is why it is written before the catalog rather than after.
    """
    if not config.GENERIC_CATALOG.exists():
        pytest.skip("generic catalog not authored yet")
    parsed = catalog.load_file(config.GENERIC_CATALOG)
    assert parsed.person_free is True

    names = set()
    for person in graph.load_graph("vitrine").persons.values():
        names.add(person["canonical_name"])
        names.update(person.get("aliases", []))
    blob = lexicon.fold(config.GENERIC_CATALOG.read_text(encoding="utf-8"))
    leaked = sorted(n for n in names if len(n) > 4 and lexicon.fold(n) in blob)
    assert not leaked, f"vitrine identities present in the committed catalog: {leaked}"


# ========== group 5: tier derivation ==========


def test_presence_needs_two_covered_sources_for_t1():
    ev = Evaluation(
        obligation_id="o",
        status="satisfied",
        matched_fact_ids=("a", "b"),
        matched_doc_ids=("d1", "d2"),
        matched_docs_all_gate_ok=True,
    )
    tier, basis = schema.assign_tier(ev, GraphHealth(coverage_known=True), _obligation())
    assert (tier, basis) == ("T1", "presence_multi_source_couvert")


def test_presence_from_one_source_is_t2():
    ev = Evaluation(
        obligation_id="o",
        status="satisfied",
        matched_fact_ids=("a",),
        matched_doc_ids=("d1",),
        matched_docs_all_gate_ok=True,
    )
    tier, _ = schema.assign_tier(ev, GraphHealth(coverage_known=True), _obligation())
    assert tier == "T2"


def test_presence_on_a_defective_fact_collapses_to_t5():
    ev = Evaluation(
        obligation_id="o",
        status="satisfied",
        matched_fact_ids=("a", "b"),
        matched_doc_ids=("d1", "d2"),
        matched_docs_all_gate_ok=True,
    )
    health = GraphHealth(coverage_known=True, defective_fact_ids=frozenset({"b"}))
    tier, basis = schema.assign_tier(ev, health, _obligation())
    assert (tier, basis) == ("T5", "presence_defaut_integrite")


def test_a_gap_over_a_covered_scope_is_t3_and_never_better():
    ev = Evaluation(obligation_id="o", status="gap", scope_covered=True)
    tier, _ = schema.assign_tier(ev, GraphHealth(coverage_known=True), _obligation())
    assert tier == "T3"
    assert schema.TIER_ORDER[tier] > schema.TIER_ORDER["T2"]


def test_an_uncovered_scope_can_never_beat_t5():
    """This is vitrine's real situation: no coverage.jsonl, so no gap is strong."""
    ev = Evaluation(obligation_id="o", status="gap", scope_covered=False)
    tier, basis = schema.assign_tier(ev, GraphHealth(coverage_known=False), _obligation())
    assert (tier, basis) == ("T5", "absence_perimetre_non_couvert")


def test_absence_tier_cap_can_only_weaken():
    ev = Evaluation(obligation_id="o", status="gap", scope_covered=True)
    capped = _obligation(absence_tier_cap="T5")
    assert schema.assign_tier(ev, GraphHealth(coverage_known=True), capped)[0] == "T5"


def test_adjudicated_verdicts_cannot_outrank_t4():
    ev = Evaluation(
        obligation_id="o",
        status="satisfied",
        matched_fact_ids=("a", "b"),
        matched_doc_ids=("d1", "d2"),
        matched_docs_all_gate_ok=True,
    )
    tier, _ = schema.assign_tier(
        ev, GraphHealth(coverage_known=True), _obligation(), sc_meta={"runs": 3, "agree": 3}
    )
    assert tier == "T4"


def test_not_triggered_produces_no_finding():
    ev = Evaluation(obligation_id="o", status="not_triggered")
    with pytest.raises(ValueError):
        schema.assign_tier(ev, GraphHealth(coverage_known=True), _obligation())


# ========== group 6: the externalisation gate ==========


def _stored(**overrides) -> dict:
    f = store.normalize(_finding())
    f.update(overrides)
    return f


def test_a_real_case_is_never_outbound_eligible():
    """The whole point of the gate. `private` must be unreachable by name."""
    assert "private" not in config.ALLOW_OUTBOUND_CASES
    with pytest.raises(ValueError):
        schema.externalisable_findings([_stored()], case_id="private")


def test_gate_drops_findings_below_the_minimum_tier():
    kept = schema.externalisable_findings([_stored(tier="T5")], case_id="vitrine")
    assert kept == []


def test_gate_drops_non_externalisable_findings_at_any_tier():
    kept = schema.externalisable_findings(
        [_stored(tier="T1", externalisable=False)], case_id="vitrine"
    )
    assert kept == []


def test_gate_drops_findings_whose_obligation_is_not_externalisable():
    obligations = {"test-obligation": _obligation(externalisable=False)}
    kept = schema.externalisable_findings(
        [_stored(tier="T1")], case_id="vitrine", obligations=obligations
    )
    assert kept == []


def test_gate_drops_findings_neutralised_by_a_dispositive_confounder():
    neutralised = _stored(
        tier="T1", confounders=[{"text_fr": "Explication alternative", "dispositive": True}]
    )
    assert schema.externalisable_findings([neutralised], case_id="vitrine") == []


def test_gate_drops_resolved_findings():
    assert schema.externalisable_findings([_stored(tier="T1", status="resolved")], "vitrine") == []


def test_gate_keeps_a_strong_open_externalisable_finding():
    kept = schema.externalisable_findings([_stored(tier="T1")], case_id="vitrine")
    assert len(kept) == 1


def test_gate_redacts_person_ids_for_a_non_anonymised_case(monkeypatch):
    """A case that is outbound-eligible but not anonymised loses its person ids."""
    monkeypatch.setattr(config, "ALLOW_OUTBOUND_CASES", frozenset({"somecase"}))
    monkeypatch.setattr(config, "CASE_IS_ANONYMISED", frozenset())
    f = _stored(
        tier="T1",
        claim="Obligation X : détenteur vitrine-amundi.",
        evidence_pointers={"person_ids": ["vitrine-amundi"]},
    )
    kept = schema.externalisable_findings([f], case_id="somecase")
    assert kept[0]["evidence_pointers"]["person_ids"] == ["personne#1"]
    assert "vitrine-amundi" not in kept[0]["claim"]


def test_gate_output_is_deterministically_ordered():
    batch = [_stored(tier="T2", id="zzz"), _stored(tier="T1", id="aaa")]
    kept = schema.externalisable_findings(batch, case_id="vitrine")
    assert [f["tier"] for f in kept] == ["T1", "T2"]


# ========== group 7: resumability ==========


def test_save_all_is_atomic(paths, monkeypatch):
    """A kill during the write must leave the previous store intact.

    The forked original uses `write_text`, which truncates first; this harness
    is designed to be killed, so a crash between truncate and write would
    destroy every finding ever recorded.
    """
    store.upsert_many(paths, [_finding()])
    before = paths.findings_jsonl.read_bytes()

    def _boom(src, dst):
        raise OSError("killed mid-write")

    monkeypatch.setattr(store.os, "replace", _boom)
    with pytest.raises(OSError):
        store.upsert_many(paths, [_finding(claim="Une autre constatation.")])

    assert paths.findings_jsonl.read_bytes() == before


def test_upsert_preserves_first_seen(paths):
    store.upsert_many(paths, [_finding()])
    first = next(iter(store.load_all(paths).values()))["first_seen"]
    store.upsert_many(paths, [_finding(evidence="evidence has moved on")])
    again = next(iter(store.load_all(paths).values()))
    assert again["first_seen"] == first
    assert again["evidence"] == "evidence has moved on"


def test_a_resolved_finding_reopens_with_its_original_first_seen(paths):
    store.upsert_many(paths, [_finding()])
    fid = next(iter(store.load_all(paths)))
    first = store.load_all(paths)[fid]["first_seen"]

    store.reconcile_pass(paths, "check", current_ids=set())
    assert store.load_all(paths)[fid]["status"] == "resolved"

    store.upsert_many(paths, [_finding()])
    reopened = store.load_all(paths)[fid]
    assert reopened["status"] == "open"
    assert reopened["first_seen"] == first


def test_confounders_and_calibration_survive_a_recheck(paths):
    """The invariant that makes adversarial work durable (ADR #70).

    `attack` writes fields outside `upsert_many`'s update set, so re-running
    `check` cannot erase them — the same mechanism that preserves `first_seen`.
    """
    store.upsert_many(paths, [_finding()])
    fid = next(iter(store.load_all(paths)))
    store.attach(
        paths,
        fid,
        confounders=[{"text_fr": "La remise a pu être verbale.", "source": "catalog_seed"}],
        calibration_entry={"ts": "now", "tier": "T2", "reason": "seeded", "actor": "attack"},
    )

    store.upsert_many(paths, [_finding(evidence="recomputed")])

    after = store.load_all(paths)[fid]
    assert [c["text_fr"] for c in after["confounders"]] == ["La remise a pu être verbale."]
    assert len(after["calibration"]) >= 1


def test_attach_is_idempotent(paths):
    store.upsert_many(paths, [_finding()])
    fid = next(iter(store.load_all(paths)))
    seed = [{"text_fr": "Même explication.", "source": "catalog_seed"}]
    store.attach(paths, fid, confounders=seed)
    store.attach(paths, fid, confounders=seed)
    assert len(store.load_all(paths)[fid]["confounders"]) == 1


def test_attach_on_a_missing_finding_is_not_an_error(paths):
    assert store.attach(paths, "deadbeef0000", confounders=[{"text_fr": "x"}]) is False


def test_reconcile_only_touches_its_own_pass(paths):
    store.upsert_many(paths, [_finding(), _finding(**{"pass": "search", "subject": "cat.yaml"})])
    store.reconcile_pass(paths, "check", current_ids=set())
    statuses = {f["pass"]: f["status"] for f in store.load_all(paths).values()}
    assert statuses == {"check": "resolved", "search": "open"}


def test_upsert_is_byte_stable_across_identical_runs(paths):
    store.upsert_many(paths, [_finding()])
    first = paths.findings_jsonl.read_text(encoding="utf-8")
    store.upsert_many(paths, [_finding()])
    second = paths.findings_jsonl.read_text(encoding="utf-8")
    # `last_seen` is the only field allowed to move on an unchanged re-run.
    strip = lambda text: [  # noqa: E731
        {k: v for k, v in json.loads(line).items() if k != "last_seen"}
        for line in text.splitlines()
    ]
    assert strip(first) == strip(second)


# ========== group 7b: budget ==========


def test_local_budget_terminates_a_cycle():
    b = budget_mod.Budget(local_calls=2)
    assert b.take_local() and b.take_local()
    assert not b.take_local()
    assert b.local_exhausted()


def test_may_escalate_refuses_weak_and_non_externalisable_findings():
    b = budget_mod.Budget(cloud_enabled=True)
    assert b.may_escalate({"tier": "T1", "externalisable": True})
    assert not b.may_escalate({"tier": "T3", "externalisable": True})
    assert not b.may_escalate({"tier": "T1", "externalisable": False})


def test_may_escalate_is_false_when_cloud_is_disabled():
    """Phase 1 default. Nothing reaches a paid model without an explicit opt-in."""
    assert not budget_mod.Budget().may_escalate({"tier": "T1", "externalisable": True})


def test_cloud_ceiling_stops_escalation():
    b = budget_mod.Budget(cloud_enabled=True, usd_cycle_ceiling=0.001)
    b.charge_cloud("anthropic/claude-haiku-4.5", 10_000, 2_000)
    assert not b.may_escalate({"tier": "T1", "externalisable": True})


def test_unknown_model_raises_rather_than_costing_nothing():
    with pytest.raises(KeyError):
        budget_mod.Budget(cloud_enabled=True).charge_cloud("no/such-model", 1, 1)


# ========== group 8: lexicon determinism ==========


def test_amounts_parse_through_non_breaking_spaces():
    """The corpus writes `195 572 €` with U+00A0; a naive regex misses it."""
    folded = lexicon.fold("créance de 195 572 € et solde de 23 477,34 EUR")
    assert lexicon.parse_amounts_eur(folded) == {195572, 23477}


def test_amount_parsing_ignores_cents_so_the_same_payment_buckets_together():
    a = lexicon.parse_amounts_eur(lexicon.fold("23 477,34 €"))
    b = lexicon.parse_amounts_eur(lexicon.fold("23477 euros"))
    assert a == b


def test_folding_makes_accents_and_apostrophes_comparable():
    assert lexicon.fold("Créance d’un tiers") == lexicon.fold("CREANCE D'UN TIERS")


def test_action_polarity_is_stable_and_signed():
    axis_a, sign_a = lexicon.action_polarity(lexicon.fold("a refusé la demande"))
    axis_b, sign_b = lexicon.action_polarity(lexicon.fold("accepte la demande"))
    assert axis_a == axis_b
    assert sign_a == -sign_b


def test_action_polarity_returns_none_off_axis():
    assert lexicon.action_polarity(lexicon.fold("rédige un acte")) is None


# ========== group 9: the graph adapter on real data ==========


def test_every_person_edge_resolves_to_a_real_fact(vitrine_graph):
    """The inversion that stands in for Fact.mentioned_person_ids.

    A dangling edge here would mean a fact base and a person file that no
    longer agree — the exact breakage re-extraction would cause.
    """
    for fact_id in vitrine_graph.fact_persons:
        assert fact_id in vitrine_graph.facts


def test_the_inversion_actually_covers_facts(vitrine_graph):
    assert vitrine_graph.fact_persons, "no person edges — the inversion silently did nothing"


def test_vitrine_has_no_coverage_and_says_so(vitrine_graph):
    """Not a defect to route around: it is why vitrine's gaps cap at T5."""
    assert vitrine_graph.coverage_known is False
    assert vitrine_graph.gate_ok(next(iter(vitrine_graph.doc_ids))) is False


def test_demo_loads_without_a_persons_file():
    """`demo` has no persons.jsonl at all — absence, not emptiness."""
    g = graph.load_graph("demo")
    assert g.persons == {}
    assert g.fact_persons == {}
    assert g.coverage_known is False


def test_ambiguous_facts_are_identified_not_dropped(vitrine_graph):
    """A fact whose bearer is disputed stays loadable but is flagged.

    Reasoning about an obligation's bearer from a fact whose role assignment
    the extractor could not settle is exactly what role_ambiguities exist to
    prevent.
    """
    for fact_id in vitrine_graph.ambiguous_fact_ids:
        assert fact_id in vitrine_graph.facts


def test_folded_index_covers_every_fact(vitrine_graph):
    assert set(vitrine_graph.folded) == set(vitrine_graph.facts)


def test_statute_corpus_is_loaded_and_addressable(vitrine_graph):
    assert "cc-730-4" in vitrine_graph.statute
    assert vitrine_graph.statute["cc-730-4"]["etat"] == "VIGUEUR"


def test_case_paths_reject_traversal():
    for bad in ("../private", ".hidden", ""):
        with pytest.raises(ValueError):
            config.CasePaths.for_case(bad)


def test_investigation_state_stays_inside_the_case_directory():
    """Nothing Plane V writes may land outside the case dir.

    That containment is what makes ADR #66's ignore-by-default rule cover this
    plane without a new privacy rule of its own.
    """
    p = config.CasePaths.for_case("vitrine")
    for path in (p.findings_jsonl, p.cache_db, p.digest_md, p.watch_state, p.outbound_dir):
        assert p.case_dir in path.parents
