"""Smoke tests for the local audit harness's slimming pass.

Deliberately asserts *invariants* rather than the current finding count.
Recall against a fixed inventory ("the fence strip has 8 sites") is a
one-time calibration measurement — it is supposed to change as the slimming
batches land, so encoding it here would turn every successful fix into a red
test. What must hold at every point in that process:

- finding ids are stable across runs and unique within one (churn destroys
  the resolve-on-fix signal the whole workflow depends on);
- every finding carries the pass name the orchestrator reconciles on;
- every concept's seed still resolves (a dangling seed silently empties a
  concept and reads as "all fixed");
- the known false-positive classes stay filtered.

Layer 0 converters are fed synthetic tool output rather than the real
subprocesses, so this file stays in the fast suite.
"""
from __future__ import annotations

import pytest

from scripts.local_audit import concepts, findings, source_index as si
from scripts.local_audit.passes import slimming


# ========== fixtures ==========


@pytest.fixture(scope="module")
def index():
    """(files, funcs, by-qualname) for the whole scanned tree, built once."""
    return slimming._build_index()


@pytest.fixture(scope="module")
def deterministic_findings():
    """The AST tiers only — no static-tool subprocesses."""
    return slimming.run({})


# ========== finding identity ==========


def test_finding_ids_are_stable_across_runs():
    # Scoped to one plane: determinism is a property of the fingerprinting,
    # not of how many files it sees, and two full-tree passes cost ~6s.
    a = [findings.make_id(f["pass"], f["file"], f["claim"]) for f in slimming.run({}, dirs=["rag"])]
    b = [findings.make_id(f["pass"], f["file"], f["claim"]) for f in slimming.run({}, dirs=["rag"])]
    assert a == b


def test_finding_ids_are_unique_within_one_run(deterministic_findings):
    ids = [findings.make_id(f["pass"], f["file"], f["claim"]) for f in deterministic_findings]
    assert len(ids) == len(set(ids))


def test_every_finding_carries_the_reconciled_pass_name(deterministic_findings):
    # orchestrator._record derives its reconcile set from this field but
    # filters the store on the pass name it was called with; a mismatch
    # resolves every finding the pass has ever emitted.
    assert {f["pass"] for f in deterministic_findings} == {slimming.PASS_NAME}


def test_claims_carry_no_volatile_numbers(deterministic_findings):
    # claim text is finding identity. A count in the claim re-mints the
    # finding on every edit and resolves the old one.
    for f in deterministic_findings:
        if "length budget" in f["claim"] or "docstring mass" in f["claim"]:
            assert not any(ch.isdigit() for ch in f["claim"]), f["claim"]


# ========== registry health ==========


def test_every_concept_seed_resolves(index):
    _, funcs, by_qualname = index
    unresolved = []
    for concept in concepts.REGISTRY:
        fp, err = concepts.resolve_fingerprint(concept, by_qualname, funcs)
        if fp is None:
            unresolved.append(f"{concept.id}: {err}")
    assert not unresolved, "dangling seeds silently empty a concept: " + "; ".join(unresolved)


def test_concept_ids_are_unique():
    ids = [c.id for c in concepts.REGISTRY]
    assert len(ids) == len(set(ids))


def test_accepted_clones_all_carry_a_reason():
    assert all(reason.strip() for reason in concepts.ACCEPTED_CLONES.values())


def test_accepted_clone_sites_are_never_flagged(deterministic_findings):
    for f in deterministic_findings:
        key = f"{f['file']}::"
        for accepted in concepts.ACCEPTED_CLONES:
            if accepted.startswith(key) and "accepted" not in f["claim"]:
                qualname = accepted.split("::", 1)[1]
                assert f"`{qualname}`" not in f["claim"], f"{accepted} was flagged: {f['claim']}"


# ========== precision: the known false-positive classes ==========


def test_harness_never_audits_itself(deterministic_findings):
    for f in deterministic_findings:
        # concepts.py is the one legitimate target: a dangling-seed alarm
        # has to point at the registry that owns the seed.
        if "registry seed" in f["claim"]:
            continue
        assert not si.is_skipped_path(f["file"]), f["file"]


@pytest.mark.parametrize("rel,line,kind,name,reason", [
    ("ingestion/dossier/facts.py", 155, "method", "_actor_role_must_match_pattern",
     "@field_validator — vulture reports the decorator line, not the def line"),
    ("ingestion/dossier/facts.py", 162, "method", "_verbatim_quote_must_be_nonempty",
     "@field_validator"),
    ("tests/test_ingestion_smoke.py", 956, "variable", "include",
     "test-file binding; vulture cannot see through Mock"),
    ("scripts/local_audit/config.py", 50, "variable", "CROSS_PLANE_ALLOWED_SYMBOLS",
     "inside the harness itself"),
])
def test_vulture_false_positive_classes_are_filtered(rel, line, kind, name, reason):
    raw = f"{rel}:{line}: unused {kind} '{name}' (60% confidence)"
    out = slimming._vulture_findings({"vulture": [raw]})
    assert out == [], f"should be filtered ({reason}), got {out}"


def test_vulture_true_positive_survives_the_filters():
    # A plain undecorated module-level function outside tests/ is exactly the
    # case vulture is trustworthy on.
    raw = "eval/ground_truth.py:1: unused function 'definitely_not_real' (60% confidence)"
    assert len(slimming._vulture_findings({"vulture": [raw]})) == 1


def test_unused_import_that_tests_monkeypatch_is_not_reported():
    # ingestion/dossier/index.py imports CHUNKS_CSV without using it, and
    # tests/test_dossier_smoke.py monkeypatches it on that module. Acting on
    # ruff's F401 here breaks eight tests.
    ruff = [{
        "code": "F401",
        "filename": str(si.config.PROJECT_ROOT / "ingestion/dossier/index.py"),
        "message": "`ingestion.index.CHUNKS_CSV` imported but unused",
        "location": {"row": 57},
    }]
    assert slimming._ruff_findings({"ruff": ruff}) == []


def test_package_reexports_are_not_reported_as_unused():
    ruff = [{
        "code": "F401",
        "filename": str(si.config.PROJECT_ROOT / "ingestion/__init__.py"),
        "message": "`ingestion.load` imported but unused",
        "location": {"row": 2},
    }]
    assert slimming._ruff_findings({"ruff": ruff}) == []


# ========== layer 0 converters ==========


def test_length_budget_keeps_the_count_out_of_the_claim():
    radon = {"rag/compliance.py": [
        {"name": "big", "lineno": 10, "endline": 200, "rank": "D", "complexity": 30},
    ]}
    out = slimming._length_findings({"radon_cc": radon})
    assert len(out) == 1
    assert "200" not in out[0]["claim"]
    assert "191 lines" in out[0]["evidence"]


def test_length_budget_ignores_functions_under_the_budget():
    radon = {"rag/flow.py": [
        {"name": "small", "lineno": 1, "endline": 20, "rank": "A", "complexity": 2},
    ]}
    assert slimming._length_findings({"radon_cc": radon}) == []
