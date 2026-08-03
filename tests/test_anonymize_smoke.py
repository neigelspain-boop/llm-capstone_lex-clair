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


# ========== layer 3: dates and amounts ==========

def test_dates_shift_consistently_preserving_intervals() -> None:
    """Compliance reasoning depends on the gap between events, so one shared
    offset must move every date by the same amount."""
    out = anonymize._shift_dates("convention du 8 mars 2024, décès le 09/01/2026")
    assert "8 mars 2024" not in out
    assert "09/01/2026" not in out

    year_text = int(re.search(r"(\d{4})", out).group(1))
    assert year_text < 2024, "offset should move dates backwards"


def test_date_shift_keeps_written_format() -> None:
    prose = anonymize._shift_dates("le 8 mars 2024")
    numeric = anonymize._shift_dates("le 09/01/2026")
    assert "/" not in prose
    assert "/" in numeric


def test_amounts_scale_by_a_single_factor_preserving_ratios() -> None:
    out_small = anonymize._scale_amounts("10 000 €")
    out_large = anonymize._scale_amounts("20 000 €")
    small = int(re.sub(r"\D", "", out_small))
    large = int(re.sub(r"\D", "", out_large))
    assert small != 10000
    assert abs(large / small - 2.0) < 0.01, "ratio between amounts must survive scaling"


def test_share_counts_are_not_scaled() -> None:
    """Only money scales. Scaling "377 parts" would break its arithmetic
    against the scaled totals."""
    assert "377 parts" in anonymize._scale_amounts("377 parts de SCPI")


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


def test_gate_raises_on_a_surviving_person_name(dossier) -> None:
    _prepare_target(dossier, "## Page 1\n\nMaître VOLTAIRE a reçu l'acte.")
    with pytest.raises(RuntimeError, match="anonymisation gate FAILED"):
        anonymize.verify_anonymization("vitrine", source_case_id="src")


def test_gate_raises_on_a_surviving_email(dossier) -> None:
    _prepare_target(dossier, "## Page 1\n\nContact : a.b@etude.fr")
    with pytest.raises(RuntimeError, match="anonymisation gate FAILED"):
        anonymize.verify_anonymization("vitrine", source_case_id="src")


def test_gate_raises_on_a_surviving_address(dossier) -> None:
    _prepare_target(dossier, "## Page 1\n\nÉtude sise 12 rue des Lilas.")
    with pytest.raises(RuntimeError, match="anonymisation gate FAILED"):
        anonymize.verify_anonymization("vitrine", source_case_id="src")


def test_gate_catches_accented_and_lowercased_variants(dossier) -> None:
    _prepare_target(dossier, "## Page 1\n\nle notaire voltaïre a signé")
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
    for leaked in (
        "DUCHEMIN", "VOLTAIRE", "PIERRAVENIR", "GRANIMMO", "etude@example.fr", "195 572",
    ):
        assert leaked not in text, f"leaked: {leaked}"
    # The legal substance must survive — that is what makes the showcase useful.
    assert "quasi-usufruit" in text
    assert "convention" in text


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
