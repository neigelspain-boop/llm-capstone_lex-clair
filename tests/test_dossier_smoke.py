"""Smoke tests for the dossier ingestion pipeline (Plane I).

Mirrors tests/test_ingestion_smoke.py's role for the statute corpus: guards
the dossier pipeline's public entry points against import/signature drift.

Deliverable 1 (extract.py) is implemented — its tests run for real against
the committed fixture data/dossier/demo/raw/sample_text.pdf. Deliverables
2-3 (gate.py, facts.py + index.py) remain skipped: no implementation exists
yet behind gate.gate_case, facts.extract_case_facts, or index.index_case
(all raise NotImplementedError by design, see ingestion/dossier/*.py).
Unskip and fill in assertions as each deliverable lands.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.dossier import extract

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PDF = ROOT / "data" / "dossier" / "demo" / "raw" / "sample_text.pdf"

# The 3 known sentences baked into the fixture PDF's single text-layer page.
KNOWN_SENTENCES = [
    "This is sentence one for the lex-clair fixture.",
    "This is sentence two about succession law testing.",
    "This is sentence three verifying verbatim extraction.",
]


# ========== deliverable 1: extract.py ==========

@pytest.fixture
def dossier_root(tmp_path, monkeypatch):
    """Redirect extract.DOSSIER_DIR into an isolated tmp dir for hermetic tests."""
    monkeypatch.setattr(extract, "DOSSIER_DIR", tmp_path)
    return tmp_path


def test_extractor_produces_md_and_sidecar(dossier_root) -> None:
    """extract_document on the demo fixture writes a verbatim .md + a well-formed sidecar."""
    assert FIXTURE_PDF.exists(), f"missing fixture: {FIXTURE_PDF}"

    result = extract.extract_document(FIXTURE_PDF, case_id="demo")

    assert result.doc_id == "sample_text"
    assert result.mode == "text_pdf"
    assert result.page_count == 1
    assert result.md_path.exists()

    md_text = result.md_path.read_text(encoding="utf-8")
    for sentence in KNOWN_SENTENCES:
        assert sentence in md_text, f"missing sentence in extracted markdown: {sentence!r}"
    assert "## Page 1" in md_text

    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["doc_id"] == "sample_text"
    assert sidecar["source_filename"] == "sample_text.pdf"
    assert sidecar["kind"] == "pdf"
    assert sidecar["extraction_mode"] == "text_pdf"
    assert sidecar["extractor_model"] is None
    assert sidecar["page_count"] == 1
    assert sidecar["source_hash"] == hashlib.sha256(FIXTURE_PDF.read_bytes()).hexdigest()
    assert sidecar["extracted_at"], "extracted_at timestamp missing"


def test_extractor_is_idempotent(dossier_root, caplog) -> None:
    """Second extraction of the same file hits the cache — no re-extraction, no vision call.

    The fixture routes to text_pdf, so the Anthropic client is never invoked
    on either run (that route has no vision step at all). What actually
    proves caching is extract_text_pdf: called once on the real first run,
    not called again on the cached second run.
    """
    with patch(
        "ingestion.dossier.extract.extract_text_pdf", wraps=extract.extract_text_pdf
    ) as spy_extract_text_pdf, patch(
        "ingestion.dossier.extract.get_anthropic_client"
    ) as mock_get_client:
        result1 = extract.extract_document(FIXTURE_PDF, case_id="demo")
        assert spy_extract_text_pdf.call_count == 1
        mock_get_client.assert_not_called()

        with caplog.at_level("INFO"):
            result2 = extract.extract_document(FIXTURE_PDF, case_id="demo")

        assert spy_extract_text_pdf.call_count == 1, "extraction re-ran on a cache hit"
        mock_get_client.assert_not_called()

    assert "cached" in caplog.text
    assert result2.doc_id == result1.doc_id
    assert result2.mode == "text_pdf"


def test_extractor_router_selects_text_pdf_for_text() -> None:
    """detect_mimetype routes the typed-text fixture PDF to text_pdf, not image_pdf."""
    assert extract.detect_mimetype(FIXTURE_PDF) == "text_pdf"


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
