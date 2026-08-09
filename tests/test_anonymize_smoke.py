"""Anonymisation pipeline + verification gate (ADR #59).

Hermetic: DOSSIER_DIR is monkeypatched into tmp_path and the LLM sweep is
never invoked (use_llm=False), so these run in the fast suite.

**Every name, place, product and reference in this file is invented.** The
module under test exists to keep real identifiers out of published artifacts;
a test file that asserted against the real corpus's names would publish them
in git history, which is exactly the failure being defended against. The
fixtures below are built to have the same *shapes* as the real data — a
surname that is also a substring of an ordinary French word, an alias list
that omits the bare surname, a two-token trading name — because those shapes
are what the layers get wrong, not the specific letters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ingestion.dossier import anonymize, personas


# ========== fixtures (invented identifiers only) ==========

# Shapes deliberately mirrored from the real corpus:
#   DUCHEMIN  — alias list holds full forms, documents use the bare surname
#   VOLTAIRE  — accented/lowercased spellings appear in prose
#   GRANIMMO  — two-token trading name, documents use the first token alone
_PERSONS = [
    {
        "person_id": "p-duchemin-alphonse", "canonical_name": "DUCHEMIN Alphonse",
        "aliases": ["DUCHEMIN", "Alphonse DUCHEMIN"], "person_type": "natural_person",
    },
    {
        "person_id": "p-voltaire", "canonical_name": "Ariane VOLTAIRE",
        "aliases": ["VOLTAIRE"], "person_type": "natural_person",
    },
    {
        "person_id": "p-granimmo", "canonical_name": "GRANIMMO PATRIMOINE",
        "aliases": ["Granimmo Patrimoine"], "person_type": "legal_person",
    },
]

# "Vaillan" is the shape that matters here: a firm fragment that sits inside
# the ordinary French word "vaillants".
_EXTRAS = {
    "_readme": "invented — see module docstring",
    "SCPI PIERRAVENIR": "SCPI Kinto-Un",
    "PIERRAVENIR": "SCPI Kinto-Un",
    "Villeneuve-sur-Marne": "Namek",
    "Vaillan": "Cabinet Yardrat",
}


def _write_source_case(base: Path, case_id: str) -> None:
    case_dir = base / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "persons.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _PERSONS) + "\n",
        encoding="utf-8",
    )
    (case_dir / anonymize.EXTRA_IDENTIFIERS_FILENAME).write_text(
        json.dumps(_EXTRAS, ensure_ascii=False, indent=2), encoding="utf-8",
    )


@pytest.fixture
def dossier(tmp_path, monkeypatch) -> Path:
    # Both modules import DOSSIER_DIR by value, so each holds its own binding
    # and both have to be redirected or the roster silently loads from the
    # real data directory.
    monkeypatch.setattr(anonymize, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(personas, "DOSSIER_DIR", tmp_path)
    _write_source_case(tmp_path, "src")
    return tmp_path


@pytest.fixture
def extras(dossier) -> dict[str, str]:
    return anonymize.load_extra_identifiers("src")


# One real person as the resolver actually leaves them: three person_ids, a
# married name, an initialised form, and no single alias common to all three.
_SPLIT_IDENTITY = [
    {
        "person_id": "p-split-a", "canonical_name": "Roxane MARTIN",
        "aliases": ["Roxane MARTIN"], "person_type": "natural_person",
    },
    {
        "person_id": "p-split-b", "canonical_name": "MARTIN Roxane épouse BERNARD",
        "aliases": ["MARTIN Roxane épouse BERNARD"], "person_type": "natural_person",
    },
    {
        "person_id": "p-split-c", "canonical_name": "R. MARTIN",
        "aliases": ["R. MARTIN"], "person_type": "natural_person",
    },
]


def _write_persons_with_split_identity(base: Path, case_id: str) -> None:
    (base / case_id / "persons.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [*_PERSONS, *_SPLIT_IDENTITY]) + "\n",
        encoding="utf-8",
    )
    (base / case_id / "_anonymization_map.json").unlink(missing_ok=True)


# ========== identifier tables live outside tracked source ==========

def test_extra_identifiers_load_from_the_private_case_directory(dossier) -> None:
    """The table's KEYS are real identifiers, so it must never be a constant
    in tracked source — that would publish them permanently in git history,
    where the project's `grep private` pre-commit check cannot see them."""
    loaded = anonymize.load_extra_identifiers("src")
    assert loaded["PIERRAVENIR"] == "SCPI Kinto-Un"
    assert not any(k.startswith("_") for k in loaded), "doc keys must be skipped"


def test_refuses_to_run_without_the_extra_identifier_table(dossier) -> None:
    """An empty-and-fine result would silently disable a whole deterministic
    layer while still reporting success."""
    (dossier / "src" / anonymize.EXTRA_IDENTIFIERS_FILENAME).unlink()
    with pytest.raises(RuntimeError, match=anonymize.EXTRA_IDENTIFIERS_FILENAME):
        anonymize.load_extra_identifiers("src")


# ========== layer 1: known-entity replacement ==========

def test_longest_first_leaves_no_dangling_given_name(dossier, extras) -> None:
    """Replacing the bare surname before "SURNAME Given" would strand a bare,
    still-identifying given name in the output. Order is load-bearing."""
    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_text(
        "Le défunt DUCHEMIN Alphonse, père de famille.", mapping, "d1", extras, use_llm=False,
    )
    assert "Alphonse" not in out
    assert "DUCHEMIN" not in out


def test_aliases_of_one_person_collapse_to_one_pseudonym(dossier, extras) -> None:
    """A bare surname and "Given SURNAME" must not read as two people.

    Asserted on the mapping rather than on label text, so the test survives a
    change of persona vocabulary — the property is "one person, one
    pseudonym", not "the pseudonym looks like this".
    """
    mapping = anonymize.load_or_build_mapping("src")
    forms = ["DUCHEMIN", "DUCHEMIN Alphonse", "Alphonse DUCHEMIN"]
    assert len({mapping[anonymize._fold(f)] for f in forms}) == 1

    out = anonymize.anonymize_text(
        "DUCHEMIN a signé. Alphonse DUCHEMIN a confirmé. DUCHEMIN Alphonse est décédé.",
        mapping, "d1", extras, use_llm=False,
    )
    assert "duchemin" not in anonymize._fold(out)


def test_a_role_alias_is_not_treated_as_an_identifier(dossier, extras) -> None:
    """The resolver puts role designations in `aliases` — the real corpus grew
    "notaire instrumentaire" on a named notaire. A role is worn by whoever
    holds the office, so it identifies nobody.

    Two things break when it is taken for a name. "instrumentaire" becomes a
    distinctive token and rewrites the ordinary French word wherever it
    appears, and the verifier then hunts that word through the public case and
    reports ten leaks that are not leaks.
    """
    persons = [*_PERSONS, {
        "person_id": "p-mena", "canonical_name": "Ariane VOLTAIRE",
        "aliases": ["notaire instrumentaire", "notaire rédacteur"],
        "person_type": "natural_person",
    }]
    (dossier / "src" / "persons.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in persons) + "\n",
        encoding="utf-8",
    )
    (dossier / "src" / "_anonymization_map.json").unlink(missing_ok=True)

    names = [n for n, _type, _pid in anonymize._load_known_entities("src")]
    assert "notaire instrumentaire" not in names
    assert "notaire rédacteur" not in names
    assert "Ariane VOLTAIRE" in names          # the real name still is one

    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_text(
        "Le notaire instrumentaire a reçu l'acte.", mapping, "d1", extras, use_llm=False,
    )
    assert "instrumentaire" in out


def test_a_short_name_is_still_an_identifier(dossier) -> None:
    """The guard is "every token is structural", not "no distinctive token" —
    the latter also drops anything under four characters, which would discard
    EDF, a real entity in the corpus."""
    persons = [*_PERSONS, {
        "person_id": "p-edf", "canonical_name": "EDF",
        "aliases": [], "person_type": "legal_person",
    }]
    (dossier / "src" / "persons.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in persons) + "\n",
        encoding="utf-8",
    )
    assert "EDF" in [n for n, _t, _p in anonymize._load_known_entities("src")]


def test_replacement_is_accent_and_case_insensitive(dossier, extras) -> None:
    mapping = anonymize.load_or_build_mapping("src")
    for spelling in ("VOLTAIRE", "voltaire", "Voltaire", "VOLTAÏRE"):
        out = anonymize.anonymize_text(
            f"Maître {spelling} a reçu l'acte.", mapping, "d1", extras, use_llm=False,
        )
        assert anonymize._fold(spelling) not in anonymize._fold(out), spelling


def test_natural_and_legal_persons_get_distinct_label_families(dossier) -> None:
    """A company must never be given a person's name. The pools are separate
    so a reader can tell an heir from a bank at a glance."""
    mapping = anonymize.load_or_build_mapping("src")
    natural = mapping[anonymize._fold("Ariane VOLTAIRE")]
    legal = mapping[anonymize._fold("GRANIMMO PATRIMOINE")]

    assert natural in personas.FALLBACK_NATURAL_POOL or natural.startswith("Personne ")
    assert legal in personas.FALLBACK_LEGAL_POOL or legal.startswith("Organisme ")
    assert natural != legal


def test_mapping_is_stable_across_runs(dossier) -> None:
    """Reruns must not renumber people — the persisted map is what makes the
    build idempotent."""
    first = anonymize.load_or_build_mapping("src")
    second = anonymize.load_or_build_mapping("src")
    assert first == second


def test_mapping_is_written_under_the_private_case_never_the_target(dossier) -> None:
    """The map links pseudonyms back to real people. It is the one artifact
    that must never ship with the public case."""
    anonymize.load_or_build_mapping("src")
    assert (dossier / "src" / "_anonymization_map.json").exists()
    assert not (dossier / "vitrine" / "_anonymization_map.json").exists()


def test_refuses_to_run_without_the_known_entity_list(dossier) -> None:
    """Silently anonymising with an empty entity list would disable the
    strongest layer and still look successful."""
    with pytest.raises(RuntimeError, match="persons.jsonl"):
        anonymize.load_or_build_mapping("case_with_no_persons_file")


# ========== the persona roster (ADR #62) ==========

def _write_roster(base: Path, case_id: str, payload: dict) -> None:
    (base / case_id / personas.ROSTER_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def test_roster_pins_a_named_character_to_a_person(dossier, extras) -> None:
    """The readability point of the whole change: a succession argument
    written in letters cannot be followed."""
    monkeypatch_roster = {"bindings": {"p-voltaire": "Maître Beerus"}}
    _write_roster(dossier, "src", monkeypatch_roster)

    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_text(
        "Maître VOLTAIRE a reçu l'acte.", mapping, "d1", extras, use_llm=False,
    )
    assert "Maître Beerus" in out


def test_merge_group_collapses_split_person_ids_onto_one_character(dossier, extras) -> None:
    """The resolver splits one real person across several person_ids. Left
    alone the three grandchildren of the showcase case rendered as eight
    different people and the family tree read as nonsense."""
    _write_persons_with_split_identity(dossier, "src")
    _write_roster(dossier, "src", {
        "bindings": {"p-split-a": "Son Gohan"},
        "merge_groups": [["p-split-a", "p-split-b", "p-split-c"]],
    })

    mapping = anonymize.load_or_build_mapping("src")
    forms = ["Roxane MARTIN", "MARTIN Roxane épouse BERNARD", "R. MARTIN"]
    assert {mapping[anonymize._fold(f)] for f in forms} == {"Son Gohan"}


def test_roster_rejects_one_character_bound_to_two_unmerged_people(dossier) -> None:
    """Merging a notaire with an heir is the exact failure the generated
    scheme produced; the roster must not be able to reintroduce it."""
    _write_roster(dossier, "src", {
        "bindings": {"p-voltaire": "Maître Beerus", "p-duchemin-alphonse": "Maître Beerus"},
    })
    with pytest.raises(ValueError, match="bound to two unmerged people"):
        personas.load_roster("src")


def test_roster_rejects_a_replacement_the_gate_would_flag(dossier) -> None:
    """verify_anonymization re-runs the PII regexes over its own output, so a
    realistic fictional address fails the build it is meant to pass. Catch it
    at the roster, where it can be fixed, not 54 documents later."""
    _write_roster(dossier, "src", {"bindings": {"p-voltaire": "12 avenue Kamé, 77000 Namek"}})
    with pytest.raises(ValueError, match="not gate-safe"):
        personas.load_roster("src")


@pytest.mark.parametrize("unsafe", [
    "12 avenue Kamé",
    "secteur 77590",
    "beerus@etude-namek.fr",
    "FR76 Namek",
])
def test_gate_safety_contract_rejects_pii_shaped_replacements(unsafe) -> None:
    with pytest.raises(ValueError, match="not gate-safe"):
        personas.assert_gate_safe([unsafe])


@pytest.mark.parametrize("safe", [
    "Résidence Kamé, Namek",
    "secteur 4, Namek",
    "Maître Beerus",
    "Capsule Corporation",
])
def test_gate_safe_fictional_addresses_are_accepted(safe) -> None:
    personas.assert_gate_safe([safe])


def test_unrostered_entity_is_still_anonymised(dossier, extras) -> None:
    """A roster is a readability upgrade, never a privacy prerequisite.
    Leaving an unnamed person in place would be a leak."""
    _write_roster(dossier, "src", {"bindings": {"p-voltaire": "Maître Beerus"}})
    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_text(
        "DUCHEMIN Alphonse et GRANIMMO PATRIMOINE.", mapping, "d1", extras, use_llm=False,
    )
    assert "DUCHEMIN" not in out
    assert "GRANIMMO" not in out


def test_token_override_names_the_shared_family_surname(dossier, extras) -> None:
    """A surname owned by several people is ambiguous, so the generated
    scheme degrades it to "[nom]" — which is how the showcase came to read
    "Personne W ép. [nom]". The roster names it instead."""
    _write_persons_with_split_identity(dossier, "src")
    _write_roster(dossier, "src", {"token_overrides": {"martin": "Son"}})

    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_text(
        "La famille MARTIN est indivise.", mapping, "d1", extras, use_llm=False,
    )
    assert "Son" in out
    assert "[nom]" not in out


def test_stale_persisted_assignment_cannot_resurrect_a_merged_person(dossier) -> None:
    """The collision this fix exists for: replaying a flattened map with
    `mapping.update(existing)` let a stale entry beat the fresh assignment,
    putting two different people on one label and leaving another unused."""
    _write_persons_with_split_identity(dossier, "src")
    anonymize.load_or_build_mapping("src")  # persist a pre-merge map

    _write_roster(dossier, "src", {
        "bindings": {"p-split-a": "Son Gohan"},
        "merge_groups": [["p-split-a", "p-split-b", "p-split-c"]],
    })
    mapping = anonymize.load_or_build_mapping("src")

    forms = ["Roxane MARTIN", "MARTIN Roxane épouse BERNARD", "R. MARTIN"]
    assert {mapping[anonymize._fold(f)] for f in forms} == {"Son Gohan"}


def test_pseudonyms_do_not_move_between_runs(dossier) -> None:
    """Person-level pinning is what keeps reruns idempotent now that the
    flattened map is no longer replayed."""
    first = anonymize.load_or_build_mapping("src")
    second = anonymize.load_or_build_mapping("src")
    assert first == second


def test_placed_fiction_is_not_reported_as_a_residual_identifier(dossier) -> None:
    """"Maître Beerus" appears on nearly every page; _TITLED_RE would flag
    "Beerus" every time and drown the report the human read depends on."""
    _write_roster(dossier, "src", {"bindings": {"p-voltaire": "Maître Beerus"}})
    anonymize.load_or_build_mapping("src")
    _prepare_target(dossier, "## Page 1\n\nMaître Beerus a reçu l'acte à Namek.")

    report = anonymize.verify_anonymization("vitrine", source_case_id="src", raise_on_leak=False)
    assert report.ok
    assert "Beerus" not in report.residual_proper_nouns
    assert "Namek" not in report.residual_proper_nouns


def test_legal_persons_keep_their_real_names_when_the_roster_says_so(dossier, extras) -> None:
    """A bank or a tax office does not identify a private family, and naming
    them makes the published case concrete (ADR #62). The gate must agree, or
    the build fails on the very names it was told to publish."""
    _write_roster(dossier, "src", {"keep_real_legal_persons": True})
    mapping = anonymize.load_or_build_mapping("src")

    out = anonymize.anonymize_text(
        "GRANIMMO PATRIMOINE gère les parts ; DUCHEMIN Alphonse est héritier.",
        mapping, "d1", extras, use_llm=False,
    )
    assert "GRANIMMO PATRIMOINE" in out, "legal person must survive untouched"
    assert "DUCHEMIN" not in out, "natural persons are still anonymised"

    _prepare_target(dossier, "## Page 1\n\nGRANIMMO PATRIMOINE gère les parts.")
    report = anonymize.verify_anonymization("vitrine", source_case_id="src", raise_on_leak=False)
    assert report.ok, "a name the roster keeps real must not count as a leak"


def test_cascade_guard_catches_a_pseudonym_that_is_also_a_key(dossier) -> None:
    """Replacement is a sequence of passes over one growing string, so a
    pseudonym that is also another entry's key gets rewritten by a later pass
    and one person silently takes another's name. Nothing about that is
    visible in the output, so it cannot be left to a human read."""
    mapping = {"duchemin": "Ariane VOLTAIRE", "voltaire": "MERCIER"}
    with pytest.raises(RuntimeError, match="replacement cascade"):
        anonymize.assert_no_replacement_cascade(mapping, {})


def test_cascade_guard_allows_an_identity_entry(dossier) -> None:
    """A key mapped to itself marks a name the policy deliberately keeps.
    Rewriting it to itself is a no-op, not a cascade."""
    anonymize.assert_no_replacement_cascade(
        {"duchemin": "MERCIER"}, {"PIERRAVENIR": "PIERRAVENIR"},
    )


# ========== layer 2: structured PII ==========

@pytest.mark.parametrize("raw", [
    "Contact : jean.dupont@etude-notaire.fr",
    "Tél. 01 64 37 12 45",
    "12 rue des Lilas, 77000 Villeneuve",
    "IBAN FR76 3000 4008 2800 0123 4567 890",
])
def test_structured_pii_is_removed(raw) -> None:
    out = anonymize._apply_structured_pii(raw)
    assert "@" not in out or "[email]" in out
    assert out != raw, f"untouched: {raw}"


def test_passport_mrz_is_redacted_wholesale() -> None:
    """A scanned ID's machine-readable zone puts the surname inside an
    unbroken alphanumeric run, where no word-anchored name matching can reach
    it."""
    out = anonymize._apply_structured_pii("P<FRADUCHEMIN<<ALPHONSE<<<<<<<<<<<<<<")
    assert "DUCHEMIN" not in out
    assert "ALPHONSE" not in out


def test_creditor_ics_reference_is_redacted() -> None:
    """A creditor ICS names the organisation inside a reference number."""
    out = anonymize._apply_structured_pii("ICS  FR47ABC001007")
    assert "ABC001007" not in out


def test_gate_does_not_fire_on_ordinary_words_containing_a_name_fragment() -> None:
    """A firm fragment ("Vaillan") sits inside the French word "vaillants";
    a plain substring gate failed the build on real prose."""
    folded = anonymize._fold("Fraude : Soyons vaillants")
    assert not re.search(r"(?<![a-z0-9])vaillan(?![a-z0-9])", folded)


# ========== layer 3: named products and places ==========

def test_named_products_and_places_are_generalised(extras) -> None:
    out = anonymize._apply_extra_identifiers(
        "377 parts de SCPI PIERRAVENIR à Villeneuve-sur-Marne", extras,
    )
    assert "PIERRAVENIR" not in out
    assert "Villeneuve-sur-Marne" not in out


# ========== filenames ==========

def test_doc_id_is_anonymised_because_it_becomes_the_chunk_id(dossier, extras) -> None:
    """doc_id flows into chunk_id ("dossier-<case>-<doc_id>-cNNN"), which is
    committed in chunks.csv and rendered in citations — anonymising the text
    but not the filename would publish the name in every citation."""
    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_doc_id(
        "duchemin_a__01_creance__granimmo_facture_20260511", mapping, extras,
    )
    assert "duchemin" not in out
    assert "granimmo" not in out
    assert out


def test_doc_id_scrubs_a_five_digit_postal_code(dossier, extras) -> None:
    """Real stems end in the property's postal code
    ("..._immeuble_sis_a_<commune>_77590"). A 6-digit-and-up rule left those
    in place, publishing the commune in every citation of that document."""
    mapping = anonymize.load_or_build_mapping("src")
    out = anonymize.anonymize_doc_id("04_immeuble_sis_a_villeneuve_77590", mapping, extras)
    assert "77590" not in out


# ========== the verification gate ==========

def _prepare_target(base: Path, text: str) -> None:
    extracted = base / "vitrine" / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    (extracted / "doc.md").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("leaked_text", [
    pytest.param("Maître VOLTAIRE a reçu l'acte.", id="a_surviving_person_name"),
    pytest.param("Contact : a.b@etude.fr", id="a_surviving_email"),
    pytest.param("Étude sise 12 rue des Lilas.", id="a_surviving_address"),
    pytest.param("le notaire voltaïre a signé", id="accented_and_lowercased_variants"),
])
def test_gate_raises_on_leaked_identifiers(dossier, leaked_text) -> None:
    """Each leak class the gate must catch, one named case per class.

    The gate is the last thing standing between a real client dossier and a
    public commit, so every class it is supposed to catch stays individually
    named — a merged failure would say "a leak got through" without saying
    which kind.
    """
    _prepare_target(dossier, f"## Page 1\n\n{leaked_text}")
    with pytest.raises(RuntimeError, match="anonymisation gate FAILED"):
        anonymize.verify_anonymization("vitrine", source_case_id="src")


def test_gate_passes_on_clean_anonymised_text(dossier) -> None:
    _prepare_target(
        dossier,
        "## Page 1\n\nPersonne A a signé la convention de quasi-usufruit.\n"
        "Organisme B en est le gestionnaire.\n",
    )
    report = anonymize.verify_anonymization("vitrine", source_case_id="src")
    assert report.ok
    assert report.files_scanned == 1


def test_gate_scans_downstream_artifacts_not_just_markdown(dossier) -> None:
    """facts and compliance re-introduce text through an LLM after this
    module has run, so a leak can appear in an artifact anonymize.py never
    wrote."""
    _prepare_target(dossier, "## Page 1\n\nPersonne A a signé.")
    (dossier / "vitrine" / "facts.jsonl").write_text(
        json.dumps({"fact_id": "f1", "verbatim_quote": "Maître VOLTAIRE a reçu l'acte."}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="facts.jsonl"):
        anonymize.verify_anonymization("vitrine", source_case_id="src")


def test_gate_reports_unknown_proper_nouns_for_human_review(dossier) -> None:
    """The known-entity list covered only 10 of 18 sampled identifiers in the
    real corpus, so the gate surfaces unrecognised capitalised tokens rather
    than implying a clean scan proves absence."""
    _prepare_target(dossier, "## Page 1\n\nPersonne A demeure à MONTBRISON.")
    report = anonymize.verify_anonymization("vitrine", source_case_id="src", raise_on_leak=False)
    assert "MONTBRISON" in report.residual_proper_nouns


def test_raise_on_leak_false_returns_the_report_instead(dossier) -> None:
    _prepare_target(dossier, "## Page 1\n\nMaître VOLTAIRE.")
    report = anonymize.verify_anonymization("vitrine", source_case_id="src", raise_on_leak=False)
    assert not report.ok
    assert any("VOLTAIRE" in ident for _, ident in report.leaks)


# ========== artifact translation (ADR #62) ==========

def _write_source_artifacts(base: Path) -> None:
    src = base / "src"
    (src / "facts.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {
            "fact_id": "duchemin_a__00_note-f001",
            "date": "2024-03-08",
            "actor_role": "notaire_redacteur",
            "action": "recevoir une convention",
            "target": "parts de SCPI PIERRAVENIR",
            "verbatim_quote": "Ariane VOLTAIRE a reçu la convention le 8 mars 2024.",
            "source_doc_id": "duchemin_a__00_note",
            "source_chunk_id": "dossier-src-duchemin_a__00_note-c001",
            "distilled_context": "VOLTAIRE reçoit la convention.",
        },
    ]) + "\n", encoding="utf-8")

    (src / "actor_roles.jsonl").write_text(json.dumps({
        "role_id": "notaire_redacteur", "label_fr": "Notaire rédacteur",
        "grounding_note": "Il s'agit de Ariane VOLTAIRE.",
        "first_seen_doc_id": "duchemin_a__00_note",
        "fact_count": 1, "confidence": "high",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    (src / "compliance_matrix.json").write_text(json.dumps({
        "case_id": "src",
        "generated_at": "2026-07-30T16:25:43Z",
        "model_id": "anthropic/claude-opus-4.7",
        "total_facts_considered": 1,
        "total_entries": 1,
        "unresolved_ambiguities": 0,
        "entries": [{
            "entry_id": "abc123",
            "statute_chunk_id": "cc-730-1",
            "statute_excerpt": "La preuve de la qualité d'héritier peut résulter "
                               "d'un acte de notoriété, loi du 23 juin 2006.",
            "obligation_summary": "Le notaire doit viser l'acte de décès.",
            "actor_role": "notaire_redacteur",
            "status": "breached",
            "evidence_fact_ids": ["duchemin_a__00_note-f001"],
            "rationale": "Ariane VOLTAIRE n'a pas visé l'acte.",
        }],
    }, ensure_ascii=False), encoding="utf-8")


def test_artifacts_are_translated_not_regenerated(dossier, extras) -> None:
    """Re-running the LLM pipeline over anonymised markdown costs $25+ on the
    compliance stage alone and yields a DIFFERENT, unreviewed analysis.
    Translating publishes exactly the determinations that were paid for."""
    _write_source_artifacts(dossier)
    mapping = anonymize.load_or_build_mapping("src")
    counts = anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)

    assert counts["facts.jsonl"] == 1
    assert counts["compliance_matrix.json"] == 1

    fact = json.loads((dossier / "vitrine" / "facts.jsonl").read_text(encoding="utf-8"))
    assert "VOLTAIRE" not in fact["verbatim_quote"]
    assert "PIERRAVENIR" not in fact["target"]
    assert "duchemin" not in fact["fact_id"]
    assert fact["actor_role"] == "notaire_redacteur", "role ids are the substance, keep them"


def test_dates_and_amounts_pass_through_unchanged(dossier, extras) -> None:
    """ADR #62 accepts the residual risk of publishing real figures. A worked
    example whose numbers do not add up against the deed it quotes is worth
    less than the protection it buys, and the protection only ever bound
    against someone who already knew the case."""
    _write_source_artifacts(dossier)
    mapping = anonymize.load_or_build_mapping("src")
    anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)

    fact = json.loads((dossier / "vitrine" / "facts.jsonl").read_text(encoding="utf-8"))
    assert fact["date"] == "2024-03-08"
    assert "8 mars 2024" in fact["verbatim_quote"]

    out = anonymize.anonymize_text(
        "créance de 195 572 € née le 8 mars 2024", mapping, "d1", extras, use_llm=False,
    )
    assert "195 572 €" in out
    assert "8 mars 2024" in out


def test_evidence_fact_ids_still_resolve_after_translation(dossier, extras) -> None:
    """A matrix that cites fact ids no facts.jsonl contains renders an
    obligation with no evidence behind it."""
    _write_source_artifacts(dossier)
    mapping = anonymize.load_or_build_mapping("src")
    anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)

    facts = {
        json.loads(line)["fact_id"]
        for line in (dossier / "vitrine" / "facts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    matrix = json.loads((dossier / "vitrine" / "compliance_matrix.json").read_text(encoding="utf-8"))
    for entry in matrix["entries"]:
        for fact_id in entry["evidence_fact_ids"]:
            assert fact_id in facts, f"dangling evidence reference {fact_id}"


def test_statute_text_and_model_id_survive_untouched(dossier, extras) -> None:
    """Statute excerpts are public law. Running the date shift over them would
    silently misdate the articles the whole analysis rests on."""
    _write_source_artifacts(dossier)
    mapping = anonymize.load_or_build_mapping("src")
    anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)

    matrix = json.loads((dossier / "vitrine" / "compliance_matrix.json").read_text(encoding="utf-8"))
    entry = matrix["entries"][0]
    assert "loi du 23 juin 2006" in entry["statute_excerpt"]
    assert entry["statute_chunk_id"] == "cc-730-1"
    assert matrix["model_id"] == "anthropic/claude-opus-4.7"
    assert matrix["case_id"] == "vitrine"


def test_source_chunk_id_is_dropped_for_the_index_to_backfill(dossier, extras) -> None:
    """Translating it would be a guess that silently breaks every citation it
    does not happen to match; index_dossier recomputes it against the chunks
    actually written."""
    _write_source_artifacts(dossier)
    mapping = anonymize.load_or_build_mapping("src")
    anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)

    fact = json.loads((dossier / "vitrine" / "facts.jsonl").read_text(encoding="utf-8"))
    assert fact["source_chunk_id"] is None


def test_unknown_artifact_field_fails_the_build(dossier, extras) -> None:
    """A field with no policy is either an unreviewed leak or a silently
    corrupted value. Both are worse than a loud failure."""
    _write_source_artifacts(dossier)
    (dossier / "src" / "facts.jsonl").write_text(
        json.dumps({"fact_id": "d-f001", "surprise_field": "Ariane VOLTAIRE"}) + "\n",
        encoding="utf-8",
    )
    mapping = anonymize.load_or_build_mapping("src")
    with pytest.raises(RuntimeError, match="no anonymisation policy"):
        anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)


def test_persons_merge_into_one_record_per_real_person(dossier, extras) -> None:
    """Emitting the resolver's split ids unchanged would show the same
    character three times in the per-person panel."""
    _write_persons_with_split_identity(dossier, "src")
    _write_roster(dossier, "src", {
        "bindings": {"p-split-a": "Son Gohan"},
        "merge_groups": [["p-split-a", "p-split-b", "p-split-c"]],
    })
    (dossier / "src" / "persons.jsonl").write_text(
        (dossier / "src" / "persons.jsonl").read_text(encoding="utf-8"), encoding="utf-8",
    )
    mapping = anonymize.load_or_build_mapping("src")
    anonymize.anonymize_artifacts("vitrine", "src", mapping, extras)

    persons = [
        json.loads(line)
        for line in (dossier / "vitrine" / "persons.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gohan = [p for p in persons if p["canonical_name"] == "Son Gohan"]
    assert len(gohan) == 1, "split identities must collapse to one record"
    assert gohan[0]["person_id"] == "vitrine-son-gohan"
    assert gohan[0]["aliases"] == ["Son Gohan"], "real aliases must not survive"


# ========== end-to-end (no LLM) ==========

def test_anonymize_case_end_to_end(dossier) -> None:
    src_extracted = dossier / "src" / "extracted"
    src_extracted.mkdir(parents=True, exist_ok=True)
    (src_extracted / "duchemin_note.md").write_text(
        "## Page 1\n\n"
        "Affaire DUCHEMIN — créance de 195 572 € née d'une convention de quasi-usufruit "
        "du 8 mars 2024, reçue par Ariane VOLTAIRE. Parts de SCPI PIERRAVENIR "
        "(GRANIMMO PATRIMOINE).\n"
        "Contact : etude@example.fr\n",
        encoding="utf-8",
    )

    summary = anonymize.anonymize_case("vitrine", source_case_id="src", use_llm=False)

    assert summary["docs_written"] == 1
    out_files = list((dossier / "vitrine" / "extracted").glob("*.md"))
    assert len(out_files) == 1
    assert "duchemin" not in out_files[0].name

    text = out_files[0].read_text(encoding="utf-8")
    for leaked in ("DUCHEMIN", "VOLTAIRE", "PIERRAVENIR", "GRANIMMO", "etude@example.fr"):
        assert leaked not in text, f"leaked: {leaked}"
    # The legal substance must survive — that is what makes the showcase
    # useful — and so must the figures, which ADR #62 publishes verbatim.
    assert "quasi-usufruit" in text
    assert "convention" in text
    assert "195 572 €" in text
    assert "8 mars 2024" in text


# ========== repo-level guard (runs against real committed artifacts) ==========

def test_committed_vitrine_artifacts_contain_no_private_identifier() -> None:
    """Standing guard over whatever is actually committed.

    The generation-time gate only runs when someone regenerates the case.
    This one runs on every test invocation, so a hand-edited or partially
    regenerated vitrine artifact cannot quietly reintroduce a real name.
    Skips when neither case is present (fresh clone, private data absent).
    """
    from ingestion.index import CHUNKS_CSV

    root = CHUNKS_CSV.parent / "dossier"
    if not (root / "vitrine").exists() or not (root / "private" / "persons.jsonl").exists():
        pytest.skip("vitrine or private/persons.jsonl not present in this checkout")

    import ingestion.dossier.anonymize as real_anon

    report = real_anon.verify_anonymization(
        "vitrine", source_case_id="private", raise_on_leak=False,
    )
    assert report.ok, (
        "private identifiers found in committed vitrine artifacts: "
        + "; ".join(f"{f}: {i}" for f, i in report.leaks[:10])
    )


def test_committed_demo_investigation_contains_no_private_identifier() -> None:
    """The same standing guard over the de-identified transcript (ADR #78).

    demo/investigation/ is derived from a REAL case, unlike vitrine's, which
    was produced by running the investigator over already-anonymous input.
    Nothing else in the suite would notice if a hand-edit reintroduced a name,
    and the role register makes that easy to do by accident: correcting one
    binding is a two-character change with case-wide reach.
    """
    from ingestion.index import CHUNKS_CSV

    root = CHUNKS_CSV.parent / "dossier"
    if not (root / "demo" / "investigation" / "findings.jsonl").exists():
        pytest.skip("demo investigation transcript not present in this checkout")
    if not (root / "private" / "persons.jsonl").exists():
        pytest.skip("private/persons.jsonl not present in this checkout")

    import ingestion.dossier.anonymize as real_anon
    from ingestion.dossier import roles

    report = real_anon.verify_anonymization(
        "demo", source_case_id="private", raise_on_leak=False,
        roster=roles.load_roster("private"), extras=roles.load_identifiers("private"),
    )
    assert report.ok, (
        "private identifiers found in the committed demo transcript: "
        + "; ".join(f"{f}: {i}" for f, i in report.leaks[:10])
    )


def test_committed_demo_transcript_ids_verify_against_their_own_content() -> None:
    """A published finding id must be a hash of the text published beside it.

    The transcript is a translation: subject and claim are rewritten, and the
    id is sha256(pass|subject|claim). Carrying the source case's ids across
    would publish 95 identifiers that verify against nothing, and nobody
    reading the file could tell.
    """
    from investigator.store import make_id

    from ingestion.index import CHUNKS_CSV

    path = CHUNKS_CSV.parent / "dossier" / "demo" / "investigation" / "findings.jsonl"
    if not path.exists():
        pytest.skip("demo investigation transcript not present in this checkout")

    findings = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mismatched = [
        f["id"] for f in findings
        if f["id"] != make_id(f["pass"], f["subject"], f["claim"])
    ]
    assert not mismatched, f"finding ids do not match their content: {mismatched[:5]}"

    # ADR #71: the internal characterisation register never leaves, and a
    # committed public artifact is as far out as it gets.
    assert all(not f["qualification_interne"] for f in findings)
    assert all(not f["penal_refs"] for f in findings)
    assert {f["case_id"] for f in findings} == {"demo"}


# --- structured-PII coverage for financial identifiers (ADR #65) -------------


@pytest.mark.parametrize(
    "text",
    [
        "RCS Beaumont 315 429 837, siège social",       # marker before the number
        "compte n 254 440 017 ouvert en 2019",          # bare account reference
        "référence 019 349 002 du contrat",             # bare contract reference
        "agréée par l'AMF sous le n° GP 07000033",      # regulator approval, packed
        "agréée sous le n° GP 07 0000 33 délivré",      # regulator approval, spaced
        "Nicole GIRARD Design / Tel ; 514 813 9053",    # labelled phone, no leading 0
        "Tél. 40 54 44 44",                             # labelled phone, OCR-truncated
    ],
)
def test_financial_identifiers_are_redacted(text: str) -> None:
    """Account, contract and registration numbers must not survive.

    All five of these passed through the anonymiser byte-identical while
    verify_anonymization still reported leaks=0, because its leak check is
    persons.jsonl names + extra identifiers + these same regexes — none of
    which covered a 9-digit reference carrying no marker, or a two-letter
    agrément prefix. The redaction layer and the verifier share
    _PII_PATTERNS, so closing the gap here closes it in both.
    """
    from ingestion.dossier.anonymize import _apply_structured_pii

    assert _apply_structured_pii(text) != text, f"identifier survived: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "capital de 684 660 €",
        "montant de 123 456 789 € versé",
        "indemnité de 250 000 euros",
        "article 587 du code civil",
        "SCPI Notapierre détenue en usufruit",
        "ADR 63 est accepté",
    ],
)
def test_amounts_and_citations_are_not_mistaken_for_identifiers(text: str) -> None:
    """A monetary amount shares the 9-digit triplet shape with a SIREN.

    Over-redaction is not a safe default here: the compliance reasoning is
    built on amounts and article numbers, so eating them would quietly
    degrade every downstream answer rather than fail loudly.
    """
    from ingestion.dossier.anonymize import _apply_structured_pii

    assert _apply_structured_pii(text) == text, f"over-redacted: {text!r}"


# ========== role-designation register + investigation transcript (ADR #78) ==========


def test_public_law_citation_survives_but_a_dossier_reference_does_not() -> None:
    """The two have the same shape and opposite meanings.

    "Ordonnance n° 45-2590" is the text the whole analysis reasons from;
    redacting it deletes the authority a finding rests on. The instrument word
    is the only thing telling it apart from a dossier reference, and the
    transcript is full of both.
    """
    from ingestion.dossier.anonymize import _apply_structured_pii

    for citation in (
        "Ordonnance n° 45-2590 du 2 novembre 1945, art. 6-2",
        "Loi n° 2010-1615 du 23 décembre 2010",
        "décret n° 2016-661 relatif aux tarifs",
    ):
        assert _apply_structured_pii(citation) == citation, f"redacted public law: {citation!r}"

    assert "[réf.]" in _apply_structured_pii("dossier n° 433221100 déposé")


@pytest.mark.parametrize(
    "text",
    [
        "Décédé à Aubagne (77590) (FRANCE), le 24 mars 2022.",   # parenthesised postcode
        "PAYE FACT 1122334455667788 SUCCESSION",                 # unbroken account run
        "Validateur 1 : 9988776655 GRANIMMO",
    ],
)
def test_civil_status_postcode_and_long_digit_runs_are_redacted(text: str) -> None:
    """Both shapes survived into a published case before ADR #78.

    The postal rule needed a capitalised word AFTER the code and found a
    closing bracket, so every place-of-birth and place-of-death line kept its
    commune's code — the one line where a commune is guaranteed to appear.
    The long runs survived because the address rule's 40-character tail bit
    through the middle of them and left a still-identifying remainder.
    """
    from ingestion.dossier.anonymize import _apply_structured_pii

    assert _apply_structured_pii(text) != text, f"identifier survived: {text!r}"


@pytest.mark.parametrize(
    "text",
    ["le 20240308 la convention", "facture du 20260211", "acte 19991231"],
)
def test_datestamps_are_not_mistaken_for_reference_numbers(text: str) -> None:
    """Dates pass through unchanged (ADR #62), including in stem form."""
    from ingestion.dossier.anonymize import _apply_structured_pii

    assert _apply_structured_pii(text) == text, f"over-redacted a datestamp: {text!r}"


def test_doc_id_keeps_a_datestamp_and_destroys_a_registration_number() -> None:
    """The exemption is a calendar check, not a digit count.

    The corpus names its correspondence by date, so scrubbing the stamp erased
    the chronology from every citation in a transcript that reasons over
    sequence. Its registration numbers are the same length and open with the
    same two digits, so only a month/day test separates them.
    """
    from ingestion.dossier.anonymize import anonymize_doc_id

    kept = anonymize_doc_id("courrier_du_20260624_relance", {}, {})
    assert "20260624" in kept

    for registration in ("20447731", "20441277", "19883299", "433221100", "88112233445"):
        out = anonymize_doc_id(f"acte_n_{registration}", {}, {})
        assert registration not in out, f"registration number survived: {registration}"
        assert "ref" in out


def test_doc_id_replacement_is_word_anchored() -> None:
    """The bug that published "..._relsophie_des_parts_...".

    An unanchored replace matched a three-letter given name inside "releve"
    and a four-letter token inside "courante". Both are ordinary French words
    carrying the document's meaning, and both were corrupted in the committed
    showcase case.
    """
    from ingestion.dossier.anonymize import anonymize_doc_id

    mapping = {"eve": "Sophie", "cour": "CHAMBRE"}
    out = anonymize_doc_id("etude_releve_des_parts_correspondance_courante", mapping, {})
    assert "releve" in out
    assert "courante" in out
    assert "sophie" not in out

    # …while the same token standing alone between separators is still replaced.
    assert "sophie" in anonymize_doc_id("note_de_eve_2026", mapping, {})


def test_a_token_inside_a_replacement_is_never_scrubbed() -> None:
    """Blanking a word inside our own output deletes the designation.

    "gestionnaire_scpi" published as "gestionnaire_x", because "scpi" is a
    distinctive token of an unrelated identifier key and the filename scrubber
    ran after the substitution that placed it.
    """
    from ingestion.dossier.anonymize import _identifier_tokens

    tokens = _identifier_tokens(
        {"granimmo patrimoine": "gestionnaire_scpi"}, {"SCPI PIERRAVENIR": "scpi_1"},
    )
    # "scpi" occurs inside two replacements, so scrubbing it would blank part
    # of this pipeline's own output.
    assert "scpi" not in tokens
    # The identifiers themselves are still scrubbed — the exclusion is about
    # replacements, not about going easy on keys.
    assert "granimmo" in tokens
    assert "pierravenir" in tokens


def test_role_register_refuses_to_keep_real_legal_persons(tmp_path, monkeypatch) -> None:
    """The persona convention's exemption is a leak in role mode.

    Keeping a bank's real name is defensible when the parties carry invented
    names — a national bank does not identify a private family. It is not
    defensible when the transcript names nobody: the SCPI, the bank and the
    étude together re-identify the case, and nothing in the analysis needs them.
    """
    from ingestion.dossier import roles

    monkeypatch.setattr(roles, "DOSSIER_DIR", tmp_path)
    case = tmp_path / "src"
    case.mkdir()
    (case / roles.ROLE_ROSTER_FILENAME).write_text(
        json.dumps({"keep_real_legal_persons": True, "bindings": {}}), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keep_real_legal_persons"):
        roles.load_roster("src")


def test_role_mode_fallback_is_ordinal_not_lettered() -> None:
    """An unrostered entity still gets a designation, shaped like the others.

    "Personne A" among role slugs reads as a different kind of object than the
    entities around it — which is the readability failure ADR #59 recorded,
    reappearing from the other direction.
    """
    from ingestion.dossier import roles
    from ingestion.dossier.anonymize import assign_personas

    roster = personas.Roster(**roles.ROLE_VOCABULARY)
    assigned = assign_personas(
        [
            ("BEAUMONT Hilaire", "natural_person", "p-1"),
            ("GRANIMMO Patrimoine", "legal_person", "p-2"),
        ],
        roster,
    )
    assert assigned["p-1"] == "personne_1"
    assert assigned["p-2"] == "organisme_1"


def test_a_finding_field_with_no_policy_fails_the_build() -> None:
    """The fail-closed guard, on the schema most likely to grow.

    A finding is 26 fields of mixed structure and free prose, and the passes
    add to it. Defaulting to KEEP publishes whatever the next field holds;
    defaulting to TEXT corrupts the next structured one.
    """
    from ingestion.dossier.anonymize import FINDING_POLICY, _translate_record

    record = {"pass": "check", "subject": "x", "claim": "y", "nouveau_champ": "DUCHEMIN"}
    with pytest.raises(RuntimeError, match="nouveau_champ"):
        _translate_record(record, FINDING_POLICY, {}, {}, "findings.jsonl[0]", {})


def test_statute_refs_are_translated_because_they_are_not_always_statute() -> None:
    """An obligation drawn from a deed carries its SOURCE document here.

    Under a KEEP policy that published 87 document stems verbatim, inside a
    field whose name says it holds public law.
    """
    from ingestion.dossier.anonymize import _POINTER_POLICY, _translate_record

    ids = {"duchemin_s__02_successions__procuration": "dossier_02_successions_procuration"}
    out = _translate_record(
        {"statute_refs": ["Code civil, art. 730-4",
                          "duchemin_s__02_successions__procuration — Signer"]},
        {"statute_refs": _POINTER_POLICY["statute_refs"]},
        {}, {}, "pointers", ids,
    )
    assert out["statute_refs"][0] == "Code civil, art. 730-4"
    assert "duchemin" not in out["statute_refs"][1]
    assert "dossier_02_successions_procuration" in out["statute_refs"][1]


def test_scrub_is_idempotent_and_repairs_a_derived_case(tmp_path, monkeypatch) -> None:
    """The repair path for a pattern strengthened after a case was built.

    Re-deriving is the wrong remedy for a pattern fix: it rewrites every
    doc_id and therefore every chunk_id and citation, to correct text a
    targeted pass corrects exactly. Running it twice must be a no-op, or it is
    not safe to re-run after the next pattern change.
    """
    from ingestion.dossier import anonymize as anon

    monkeypatch.setattr(anon, "DOSSIER_DIR", tmp_path)
    case = tmp_path / "pub"
    (case / "extracted").mkdir(parents=True)
    (case / "extracted" / "note.md").write_text(
        "Décédé à Beaumont (77590) (FRANCE).\n", encoding="utf-8",
    )
    (case / "facts.jsonl").write_text(
        json.dumps({"fact_id": "note-f001", "verbatim_quote": "PAYE 1122334455667788"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    first = anon.scrub_case_pii("pub")
    assert set(first) == {"extracted/note.md", "facts.jsonl"}
    assert "77590" not in (case / "extracted" / "note.md").read_text(encoding="utf-8")
    assert "1122334455667788" not in (case / "facts.jsonl").read_text(encoding="utf-8")

    assert anon.scrub_case_pii("pub") == {}


def test_a_reference_quoted_back_inside_prose_is_redacted() -> None:
    """No id table can reach a number a model wrote into a sentence.

    The transcript's counter-arguments are generated text, and one of them
    quoted a document's registration number back as prose — "référencé sous
    <number>". There it is not a document stem, not a pointer and not a field:
    it is a word in a sentence, reachable only by a structural rule. The
    calendar test is what keeps that rule from eating the dates around it.
    """
    from ingestion.dossier.anonymize import _apply_structured_pii

    out = _apply_structured_pii(
        "Le document est intégré à l'acte notarié (référencé sous 20447731), "
        "signé le 20240308 pour un montant de 195 572 €."
    )
    assert "20447731" not in out
    assert "[réf.]" in out
    # …and the neighbours it must not touch.
    assert "20240308" in out
    assert "195 572 €" in out
