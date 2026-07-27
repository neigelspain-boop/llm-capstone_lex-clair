"""Smoke tests for the dossier ingestion pipeline (Plane I) · placeholders.

Mirrors tests/test_ingestion_smoke.py's role for the statute corpus: guards
the dossier pipeline's public entry points against import/signature drift.
All three tests are currently skipped — no implementation exists yet behind
extract.extract_case, gate.gate_case, facts.extract_case_facts, or
index.index_case (all raise NotImplementedError by design, see
ingestion/dossier/*.py). Unskip and fill in assertions as each deliverable
lands.
"""
from __future__ import annotations

import pytest


# ========== deliverable 1: extract.extract_case ==========

@pytest.mark.skip("deliverable 1 pending")
def test_extract_case_smoke() -> None:
    """extract.extract_case(case_id, raw_dir) should produce .md + .json sidecars per doc."""
    pass


# ========== deliverable 2: gate.gate_case ==========

@pytest.mark.skip("deliverable 2 pending")
def test_gate_case_smoke() -> None:
    """gate.gate_case(case_id) should append one coverage record per doc to coverage.jsonl."""
    pass


# ========== deliverable 3: facts.extract_case_facts + index.index_case ==========

@pytest.mark.skip("deliverable 3 pending")
def test_facts_and_index_case_smoke() -> None:
    """facts.extract_case_facts + index.index_case should produce facts.jsonl and
    indexed dossier chunks appended to the shared Chroma collection."""
    pass
