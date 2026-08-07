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
from pathlib import Path

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


def _synthetic_graph(facts_spec, *, coverage=None, ambiguous=(), statute=None):
    """Build a CaseGraph by hand. `facts_spec` is (fact_id, role, date, text)."""
    from ingestion.dossier.facts import Fact

    facts, folded, by_role = {}, {}, {}
    for fid, role, when, text in facts_spec:
        facts[fid] = Fact(
            fact_id=fid,
            date=when,
            actor_role=role,
            action=text,
            target=None,
            verbatim_quote=text,
            source_doc_id=fid.split("-f")[0],
        )
        folded[fid] = {
            "action": lexicon.fold(text),
            "target": "",
            "verbatim_quote": lexicon.fold(text),
            "distilled_context": "",
        }
        by_role.setdefault(role, []).append(fid)
    docs = {f.source_doc_id for f in facts.values()}
    return graph.CaseGraph(
        case_id="synthetic",
        facts=facts,
        roles={},
        ambiguities=(),
        persons={},
        person_facts={},
        fact_persons={},
        persons_by_role={},
        facts_by_role={r: tuple(sorted(v)) for r, v in by_role.items()},
        folded=folded,
        coverage=coverage if coverage is not None else {},
        coverage_known=coverage is not None,
        doc_ids=frozenset(docs),
        doc_text_folded={},
        statute=statute or {},
        case_chunks={},
        ambiguous_fact_ids=frozenset(ambiguous),
    )


# ========== group 4: predicate semantics ==========


def test_an_untriggered_obligation_produces_no_finding():
    from investigator.passes import check

    ob = _obligation(
        window={
            "from_event": {"kind": "fact_match", "any_terms_fr": ["décès"]},
        }
    )
    g = _synthetic_graph([("d-f001", "notaire_redacteur", "2026-01-01", "rédige un acte")])
    assert check.evaluate(ob, g).status == "not_triggered"


def test_evidence_defaults_to_the_bearer_so_a_stranger_cannot_satisfy_it():
    """The false-`satisfied` direction, which no later adjudication can repair.

    A leaf naming no role means "the party who owes the obligation", not
    "anyone" — otherwise one well-worded sentence from an unrelated third party
    reports the duty performed.
    """
    from investigator.passes import check

    ob = _obligation(
        evidence_scope={"doc_id_patterns": ["*"]},
        expected_evidence={"all_of": [{"kind": "fact_match", "any_terms_fr": ["extrait"]}]},
    )
    stranger = _synthetic_graph(
        [("d-f001", "fournisseur_energie", "2026-01-01", "transmet un extrait")],
        coverage={"d": "ok"},
    )
    assert check.evaluate(ob, stranger).status == "gap"

    bearer = _synthetic_graph(
        [("d-f001", "notaire_redacteur", "2026-01-01", "transmet un extrait")],
        coverage={"d": "ok"},
    )
    assert check.evaluate(ob, bearer).status == "satisfied"


def test_any_actor_widens_a_leaf_back_to_the_whole_graph():
    from investigator.passes import check

    ob = _obligation(
        evidence_scope={"doc_id_patterns": ["*"]},
        expected_evidence={
            "all_of": [{"kind": "fact_match", "any_actor": True, "any_terms_fr": ["extrait"]}]
        },
    )
    g = _synthetic_graph(
        [("d-f001", "fournisseur_energie", "2026-01-01", "transmet un extrait")],
        coverage={"d": "ok"},
    )
    assert check.evaluate(ob, g).status == "satisfied"


def test_a_gap_needs_a_covered_scope_or_it_is_only_unverifiable():
    """The distinction the whole design rests on."""
    from investigator.passes import check

    ob = _obligation(evidence_scope={"doc_id_patterns": ["*"]})
    covered = _synthetic_graph(
        [("d-f001", "notaire_redacteur", "2026-01-01", "rédige")], coverage={"d": "ok"}
    )
    unknown = _synthetic_graph([("d-f001", "notaire_redacteur", "2026-01-01", "rédige")])
    failed = _synthetic_graph(
        [("d-f001", "notaire_redacteur", "2026-01-01", "rédige")],
        coverage={"d": "parse_failed"},
    )
    assert check.evaluate(ob, covered).status == "gap"
    assert check.evaluate(ob, unknown).status == "unverifiable"
    assert check.evaluate(ob, failed).status == "unverifiable"


def test_a_deadline_breach_needs_both_dates_known():
    """Undated facts must never manufacture a breach."""
    from investigator.passes import check

    ob = _obligation(
        evidence_scope={"doc_id_patterns": ["*"]},
        window={
            "from_event": {"kind": "fact_match", "any_actor": True, "any_terms_fr": ["décès"]},
            "deadline_days": 30,
        },
        expected_evidence={
            "all_of": [{"kind": "fact_match", "any_terms_fr": ["extrait"], "after_trigger": True}]
        },
    )
    late = _synthetic_graph(
        [
            ("d-f001", "defunt", "2026-01-01", "constate le décès"),
            ("d-f002", "notaire_redacteur", "2026-06-01", "transmet un extrait"),
        ],
        coverage={"d": "ok"},
    )
    undated = _synthetic_graph(
        [
            ("d-f001", "defunt", "2026-01-01", "constate le décès"),
            ("d-f002", "notaire_redacteur", None, "transmet un extrait"),
        ],
        coverage={"d": "ok"},
    )
    assert check.evaluate(ob, late).status == "window_breach"
    assert check.evaluate(ob, undated).status == "satisfied"


def test_no_deadline_is_computed_from_an_ambiguous_trigger():
    """Regression: the corpus recites three different deaths.

    Taking the earliest match measured a six-month deadline from a 1981 recital
    and reported a 6593-day breach at `critical`. Where the dossier disagrees
    about when the triggering event happened, the deadline is not testable and
    the engine must say so instead of choosing.
    """
    from investigator.passes import check

    ob = _obligation(
        evidence_scope={"doc_id_patterns": ["*"]},
        window={
            "from_event": {"kind": "fact_match", "any_actor": True, "any_terms_fr": ["décès"]},
            "deadline_days": 183,
        },
        expected_evidence={
            "all_of": [{"kind": "fact_match", "any_terms_fr": ["extrait"], "after_trigger": True}]
        },
    )
    ambiguous = _synthetic_graph(
        [
            ("d-f001", "defunt", "1981-02-05", "constate le décès"),
            ("d-f002", "defunt", "2026-01-09", "constate le décès"),
            ("d-f003", "notaire_redacteur", "2026-03-01", "transmet un extrait"),
        ],
        coverage={"d": "ok"},
    )
    ev = check.evaluate(ob, ambiguous)
    assert ev.trigger_ambiguous is True
    assert ev.status == "satisfied"
    assert ev.window_breach_days is None

    unambiguous = _synthetic_graph(
        [
            ("d-f002", "defunt", "2026-01-09", "constate le décès"),
            ("d-f003", "notaire_redacteur", "2026-12-01", "transmet un extrait"),
        ],
        coverage={"d": "ok"},
    )
    later = check.evaluate(ob, unambiguous)
    assert later.trigger_ambiguous is False
    assert later.status == "window_breach"


def test_trigger_select_picks_the_operative_end_of_the_range():
    from investigator.passes import check

    facts = [
        ("d-f001", "defunt", "2020-01-01", "constate le décès"),
        ("d-f002", "defunt", "2026-01-09", "constate le décès"),
    ]
    g = _synthetic_graph(facts, coverage={"d": "ok"})
    event = {"kind": "fact_match", "any_actor": True, "any_terms_fr": ["décès"]}

    earliest = _obligation(window={"from_event": event})
    latest = _obligation(window={"from_event": event, "trigger_select": "latest"})
    assert check._find_trigger(earliest, g)[1] == "d-f001"
    assert check._find_trigger(latest, g)[1] == "d-f002"


def test_scope_falls_back_to_where_the_bearer_appears():
    """Not "every document": 8 of 55 real documents have an unparsed gate verdict,
    so an all-documents scope would make every obligation unverifiable."""
    from investigator.passes import check

    ob = _obligation()
    g = _synthetic_graph(
        [
            ("mine-f001", "notaire_redacteur", "2026-01-01", "rédige"),
            ("theirs-f001", "fournisseur_energie", "2026-01-01", "facture"),
        ],
        coverage={"mine": "ok", "theirs": "parse_failed"},
    )
    ev = check.evaluate(ob, g)
    assert ev.scope_doc_ids == ("mine",)
    assert ev.scope_covered is True


def test_foreach_instances_get_distinct_subjects_and_claims():
    from investigator.passes import check

    ob = _obligation(
        foreach={"kind": "role_instances", "actor_roles": ["gestionnaire_scpi"], "key": "person_id"},
        claim_template_fr="Obligation {obligation_id} : détenteur {foreach_role}/{foreach_key}.",
    )
    g = _synthetic_graph([("d-f001", "notaire_redacteur", "2026-01-01", "rédige")])
    g = graph.CaseGraph(
        **{**g.__dict__, "persons_by_role": {"gestionnaire_scpi": ("p-a", "p-b")}}
    )
    pairs = check.instances(ob, g)
    assert pairs == [("gestionnaire_scpi", "p-a"), ("gestionnaire_scpi", "p-b")]
    claims = {
        check._claim(ob, check.evaluate(ob, g, foreach_key=k, foreach_role=r))
        for r, k in pairs
    }
    assert len(claims) == 2


# ========== group 3b: search catches a wrong citation ==========


def _search_ctx(obligations, case_graph):
    from investigator.budget import Budget
    from investigator.catalog import Catalog
    from investigator.passes import extract

    cat = Catalog(
        obligations={o.obligation_id: o for o in obligations},
        origin={o.obligation_id: "test.yaml" for o in obligations},
        files=(),
    )
    return schema.RunContext(
        case_id="synthetic",
        paths=config.CasePaths.for_case("vitrine"),
        graph=case_graph,
        catalog=cat,
        budget=Budget(),
        health=extract.graph_health(case_graph),
    )


def test_search_catches_the_real_historical_citation_error(vitrine_graph):
    """The `art. 730-4 al. 2` error, caught with zero network calls.

    An earlier draft of this catalog anchored a unanimity obligation to
    "art. 730-4 al. 2". The article has one alinéa and it *enables* release in
    the proportion stated in the acte de notoriété. This is the pass earning
    its slot on a real input rather than a synthetic one.
    """
    from investigator.passes import search

    wrong = _obligation(
        source={
            "kind": "statute",
            "ref": "Code civil, art. 730-4 al. 2",
            "chunk_id": "cc-730-4",
            "excerpt_fr": "en cas de pluralité d'ayants droit, l'encaissement individuel "
            "des fonds réclamera un accord unanime",
        }
    )
    result = search.run(_search_ctx([wrong], vitrine_graph))
    kinds = {f["claim"] for f in result.findings}
    assert any("n'apparaît pas dans le texte" in k for k in kinds), kinds


def test_search_accepts_a_citation_that_is_actually_verbatim(vitrine_graph):
    from investigator.passes import search

    right = _obligation(
        source={
            "kind": "statute",
            "ref": "Code civil, art. 730-4",
            "chunk_id": "cc-730-4",
            "legiarti_id": "LEGIARTI000006430899",
            "excerpt_fr": "la libre disposition de ceux-ci dans la proportion indiquée à l'acte",
        }
    )
    assert search.run(_search_ctx([right], vitrine_graph)).findings == []


def test_search_reports_an_anchor_outside_the_local_corpus(vitrine_graph):
    """`cc-1204`, `cc-1344` and `cc-494-12` are genuinely outside the 792-chunk
    corpus. Unverifiable-here is not the same as wrong, and must not be
    reported as if it were."""
    from investigator.passes import search

    ob = _obligation(
        source={
            "kind": "statute",
            "ref": "Code civil, art. 1204",
            "legiarti_id": "LEGIARTI000000000000",
            "excerpt_fr": "On peut se porter fort pour un tiers.",
        }
    )
    findings = search.run(_search_ctx([ob], vitrine_graph)).findings
    assert len(findings) == 1
    assert "ne peut être vérifiée hors ligne" in findings[0]["claim"]


def test_search_reports_a_chunk_id_absent_from_the_corpus(vitrine_graph):
    from investigator.passes import search

    ob = _obligation(
        source={
            "kind": "statute",
            "ref": "Code civil, art. 9999",
            "chunk_id": "cc-9999",
            "excerpt_fr": "Texte inexistant.",
        }
    )
    findings = search.run(_search_ctx([ob], vitrine_graph)).findings
    assert "absent du corpus" in findings[0]["claim"]


def test_the_shipped_catalog_citations_all_verify(vitrine_graph):
    """Guards the committed catalog itself. A wrong citation cannot land."""
    if not config.GENERIC_CATALOG.exists():
        pytest.skip("generic catalog not authored yet")
    from investigator.passes import search

    obligations = catalog.load_file(config.GENERIC_CATALOG).obligations
    findings = search.run(_search_ctx(obligations, vitrine_graph)).findings
    assert findings == [], [f["claim"] for f in findings]


# ========== group 1: pass identity ==========


@pytest.mark.parametrize("module_name", ["extract", "check", "search"])
def test_every_finding_carries_its_own_pass_name(module_name, vitrine_graph):
    """A mismatch between the emitted pass and the reconciled name would
    silently resolve everything that pass ever produced."""
    import importlib

    module = importlib.import_module(f"investigator.passes.{module_name}")
    obligations = (
        catalog.load_file(config.GENERIC_CATALOG).obligations
        if config.GENERIC_CATALOG.exists()
        else [_obligation()]
    )
    result = module.run(_search_ctx(obligations, vitrine_graph))
    assert {f["pass"] for f in result.findings} <= {module.PASS_NAME}


@pytest.mark.parametrize("module_name", ["extract", "check", "search"])
def test_finding_ids_are_stable_and_unique_across_runs(module_name, vitrine_graph):
    import importlib

    module = importlib.import_module(f"investigator.passes.{module_name}")
    obligations = (
        catalog.load_file(config.GENERIC_CATALOG).obligations
        if config.GENERIC_CATALOG.exists()
        else [_obligation()]
    )
    ctx = _search_ctx(obligations, vitrine_graph)
    first = [store.make_id(f["pass"], f["subject"], f["claim"]) for f in module.run(ctx).findings]
    second = [store.make_id(f["pass"], f["subject"], f["claim"]) for f in module.run(ctx).findings]
    assert first == second
    assert len(first) == len(set(first))


def test_claims_carry_no_volatile_numbers(vitrine_graph):
    """Counts, dates and amounts belong in `evidence`, which is rewritten every
    run; a claim carrying one re-mints its finding id on every cycle."""
    import importlib

    obligations = (
        catalog.load_file(config.GENERIC_CATALOG).obligations
        if config.GENERIC_CATALOG.exists()
        else [_obligation()]
    )
    ctx = _search_ctx(obligations, vitrine_graph)
    # Volatile markers that must only ever appear in `evidence`. A claim
    # carrying one changes whenever the case does, so the finding it names can
    # never be tracked, resolved, or attacked across cycles.
    forbidden = ("faits_retenus", "périmètre=", "statut=", "candidats=", "déclencheur=", "retard=")
    for module_name in ("extract", "check", "search"):
        module = importlib.import_module(f"investigator.passes.{module_name}")
        for f in module.run(ctx).findings:
            assert not any(marker in f["claim"] for marker in forbidden), f["claim"]
            # Whatever numbers survive must come from the citation, never from
            # the case: identical claims must render identically on any dossier.
            assert f["claim"] == f["claim"].format(), f["claim"]


# ========== group 8b: contradict determinism ==========


def test_contradict_pairs_are_stable_across_runs(vitrine_graph):
    from investigator.passes import contradict

    first = contradict.candidate_pairs(vitrine_graph)
    second = contradict.candidate_pairs(vitrine_graph)
    assert first == second


def test_contradict_cap_truncates_a_stable_order(vitrine_graph):
    """An unstable cap re-mints finding ids every cycle and destroys
    resolve-on-fix — the easiest way to get this pass wrong."""
    from investigator.passes import contradict

    full = contradict.candidate_pairs(vitrine_graph, max_pairs=10**9)
    capped = contradict.candidate_pairs(vitrine_graph, max_pairs=5)
    assert capped == full[:5]


def test_contradict_reports_incomplete_when_capped(vitrine_graph):
    """A truncated sweep must not reconcile: the pairs beyond the cap were
    never examined, and resolving them would report deletion as progress."""
    from investigator.passes import contradict

    result = contradict.run(_search_ctx([_obligation()], vitrine_graph))
    if len(contradict.candidate_pairs(vitrine_graph, max_pairs=10**9)) > config.CONTRADICT_MAX_PAIRS:
        assert result.complete is False


def test_contradict_finds_a_planted_amount_collision():
    from investigator.passes import contradict

    g = _synthetic_graph(
        [
            ("a-f001", "notaire_redacteur", "2026-01-01", "prélève 23 477,34 € sur le compte"),
            ("b-f001", "etablissement_bancaire", "2026-03-01", "vire 23477 euros à l'étude"),
        ]
    )
    pairs = contradict.candidate_pairs(g)
    assert any(p.bucket == "montant" and p.bucket_key == "23477" for p in pairs)


def test_same_actor_same_day_is_corroboration_not_a_candidate():
    from investigator.passes import contradict

    g = _synthetic_graph(
        [
            ("a-f001", "notaire_redacteur", "2026-01-01", "prélève 1 000 €"),
            ("a-f002", "notaire_redacteur", "2026-01-01", "confirme 1 000 €"),
        ]
    )
    assert [p for p in contradict.candidate_pairs(g) if p.bucket == "montant"] == []


def test_contradict_finds_opposite_polarity_on_the_same_target():
    from investigator.passes import contradict

    g = _synthetic_graph(
        [
            ("a-f001", "heritier", "2026-01-01", "accepte la baisse de prix"),
            ("b-f001", "heritier", "2026-02-01", "refuse la baisse de prix"),
        ]
    )
    assert any(p.bucket == "sens_action" for p in contradict.candidate_pairs(g))


# ========== group 6b: attack seeds and preserves ==========


def test_attack_seeds_catalog_confounders_onto_strong_findings(paths, vitrine_graph):
    from investigator.passes import attack

    ob = _obligation(confounders_seed=["La remise a pu être verbale."])
    store.upsert_many(paths, [_finding(tier="T2")])
    ctx = _search_ctx([ob], vitrine_graph)
    ctx = schema.RunContext(**{**ctx.__dict__, "paths": paths})

    attack.run(ctx)
    stored = next(iter(store.load_all(paths).values()))
    assert [c["text_fr"] for c in stored["confounders"]] == ["La remise a pu être verbale."]
    assert stored["confounders"][0]["dispositive"] is False


def test_attack_leaves_t5_findings_alone(paths, vitrine_graph):
    """A T5 hypothesis is already flagged unreliable; a confounder would only
    dress it up."""
    from investigator.passes import attack

    ob = _obligation(confounders_seed=["Contre-argument."])
    store.upsert_many(paths, [_finding(tier="T5")])
    ctx = _search_ctx([ob], vitrine_graph)
    attack.run(schema.RunContext(**{**ctx.__dict__, "paths": paths}))
    assert next(iter(store.load_all(paths).values()))["confounders"] == []


def test_attack_is_idempotent_across_cycles(paths, vitrine_graph):
    from investigator.passes import attack

    ob = _obligation(confounders_seed=["Contre-argument."])
    store.upsert_many(paths, [_finding(tier="T2")])
    ctx = schema.RunContext(**{**_search_ctx([ob], vitrine_graph).__dict__, "paths": paths})
    attack.run(ctx)
    attack.run(ctx)
    stored = next(iter(store.load_all(paths).values()))
    assert len(stored["confounders"]) == 1
    assert len(stored["calibration"]) == 1


# ========== group 7c: the watch loop ==========


def test_artifact_hashes_record_absence_as_a_value(tmp_path):
    """A file appearing or disappearing must itself trigger a cycle."""
    from investigator import watch

    p = config.CasePaths.for_case("vitrine", dossier_dir=tmp_path)
    hashes = watch.artifact_hashes(p)
    assert hashes["facts.jsonl"] == "missing"
    assert hashes["corpus:chunks.csv"] != "missing"


def test_watch_records_incomplete_passes_for_the_next_cycle(tmp_path, monkeypatch):
    """Backlog mode: an incomplete pass is what keeps the loop from sleeping."""
    from investigator import watch

    p = config.CasePaths.for_case("demo", dossier_dir=tmp_path)
    p.investigation_dir.mkdir(parents=True)
    fake = orchestrator_report_stub()
    monkeypatch.setattr(watch.orchestrator, "run_cycle", lambda *a, **k: fake)
    watch.loop("demo", dossier_dir=tmp_path, once=True)
    state = json.loads(p.watch_state.read_text(encoding="utf-8"))
    assert state["incomplete_passes"] == ["contradict"]
    assert state["cycle_seq"] == 1


def orchestrator_report_stub():
    from investigator.orchestrator import CycleReport

    return CycleReport(
        case_id="demo",
        per_pass={"check": (3, True), "contradict": (40, False)},
        open_findings=5,
    )


# ========== end-to-end ==========


def test_a_full_cycle_is_idempotent(tmp_path):
    """Two cycles with no input change: no new ids, nothing reconciled."""
    import shutil

    from investigator import orchestrator

    src = config.DOSSIER_DIR / "demo"
    dst = tmp_path / "demo"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("investigation"))

    orchestrator.run_cycle("demo", dossier_dir=tmp_path)
    p = config.CasePaths.for_case("demo", dossier_dir=tmp_path)
    first = store.load_all(p)
    orchestrator.run_cycle("demo", dossier_dir=tmp_path)
    second = store.load_all(p)

    assert set(first) == set(second)
    assert {f["status"] for f in second.values()} == {"open"}
    assert all(second[k]["first_seen"] == first[k]["first_seen"] for k in first)


def test_render_outbound_refuses_a_real_case(tmp_path):
    from investigator import render

    p = config.CasePaths.for_case("private", dossier_dir=tmp_path)
    with pytest.raises(ValueError):
        render.render_outbound(p, "private", {})
    assert not p.outbound_dir.exists(), "a refused case must leave no partial output"


def test_render_outbound_sources_only_through_the_gate(tmp_path, monkeypatch):
    """Sentinel proof that there is no second path to an outbound artifact."""
    from investigator import render

    sentinel = [
        {
            "id": "sentinel0001",
            "pass": "check",
            "subject": "s",
            "claim": "SENTINEL",
            "tier": "T1",
            "evidence": "",
            "confounders": [],
        }
    ]
    monkeypatch.setattr(schema, "externalisable_findings", lambda *a, **k: sentinel)
    p = config.CasePaths.for_case("vitrine", dossier_dir=tmp_path)
    target = render.render_outbound(p, "vitrine", {"real": {"claim": "NOT THE SENTINEL"}})
    body = Path(target).read_text(encoding="utf-8")
    assert "SENTINEL" in body and "NOT THE SENTINEL" not in body


def test_only_render_writes_to_the_outbound_directory():
    """Source scan: one writer, so the property is inspectable, not aspirational."""
    root = config.PROJECT_ROOT / "investigator"
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if path.name not in {"render.py", "config.py"}
        and "outbound_dir" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"outbound_dir referenced outside render.py: {offenders}"


def test_investigation_state_stays_inside_the_case_directory():
    """Nothing Plane V writes may land outside the case dir.

    That containment is what makes ADR #66's ignore-by-default rule cover this
    plane without a new privacy rule of its own.
    """
    p = config.CasePaths.for_case("vitrine")
    for path in (p.findings_jsonl, p.cache_db, p.digest_md, p.watch_state, p.outbound_dir):
        assert p.case_dir in path.parents
