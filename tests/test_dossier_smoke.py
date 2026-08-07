"""Smoke tests for the dossier ingestion pipeline (Plane I).

Mirrors tests/test_ingestion_smoke.py's role for the statute corpus: guards
the dossier pipeline's public entry points against import/signature drift.

Deliverables 1-2 (extract.py, gate.py) are implemented — their tests run for
real against the committed fixture data/dossier/demo/raw/sample_text.pdf (and,
for gate.py, small synthetic PDFs built on the fly) with the Anthropic client
mocked. Deliverable 4 (facts.py) is now implemented and tested the same way
(mocked client, hermetic tmp_path DOSSIER_DIR). Deliverable 5 (index.py) is
now implemented: chunking + facts backfill tests run in the fast suite
(Chroma append stubbed out); the two tests that exercise a real Chroma
collection + real BGE-M3 embeddings are marked @pytest.mark.slow, matching
the precedent in tests/test_ingestion_smoke.py::test_rag_flow_end_to_end
(slow because it loads a real local model, not because it hits an LLM API).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import chromadb
import pandas as pd
import pytest
from pydantic import ValidationError

from ingestion.dossier import extract, facts, gate, index

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
    """Redirect extract.DOSSIER_DIR into an isolated tmp dir for hermetic tests.

    extract_document now computes source_relpath relative to
    DOSSIER_DIR/<case_id>/raw/, so the source file must actually live there —
    the committed fixture is copied in rather than read from its real repo
    path. Returns (tmp_path, path_to_copied_fixture).
    """
    monkeypatch.setattr(extract, "DOSSIER_DIR", tmp_path)
    raw_dir = tmp_path / "demo" / "raw"
    raw_dir.mkdir(parents=True)
    fixture_copy = raw_dir / FIXTURE_PDF.name
    fixture_copy.write_bytes(FIXTURE_PDF.read_bytes())
    return tmp_path, fixture_copy


def test_extractor_produces_md_and_sidecar(dossier_root) -> None:
    """extract_document on the demo fixture writes a verbatim .md + a well-formed sidecar."""
    _, source_pdf = dossier_root

    result = extract.extract_document(source_pdf, case_id="demo")

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
    assert sidecar["source_hash"] == hashlib.sha256(source_pdf.read_bytes()).hexdigest()
    assert sidecar["extracted_at"], "extracted_at timestamp missing"


def test_sidecar_contains_source_relpath(dossier_root) -> None:
    """The sidecar records source_relpath, relative to the case's raw/ directory."""
    _, source_pdf = dossier_root

    result = extract.extract_document(source_pdf, case_id="demo")

    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert "source_relpath" in sidecar
    assert sidecar["source_relpath"] == "sample_text.pdf"


def test_extract_handles_relative_raw_dir(tmp_path, monkeypatch):
    """Regression: extract must accept relative --raw-dir paths without crashing.

    Real CLI invocations often use relative paths (e.g. `--raw-dir data/dossier/...`),
    which broke path.relative_to() until extract_document resolved source paths
    to absolute internally.
    """
    # Set up a case with an absolute DOSSIER_DIR (as production does)
    abs_dossier = tmp_path / "data" / "dossier"
    case_root = abs_dossier / "reltest" / "raw" / "subdir"
    case_root.mkdir(parents=True)
    fixture_src = Path("data/dossier/demo/raw/sample_text.pdf").resolve()
    target = case_root / "sample_text.pdf"
    target.write_bytes(fixture_src.read_bytes())

    monkeypatch.setattr(extract, "DOSSIER_DIR", abs_dossier)

    # Change cwd so that raw_dir passed to extract_case becomes RELATIVE
    monkeypatch.chdir(tmp_path)
    rel_raw_dir = Path("data/dossier/reltest/raw")
    assert not rel_raw_dir.is_absolute()

    results = extract.extract_case("reltest", rel_raw_dir)

    assert len(results) == 1
    sidecar = json.loads(results[0].sidecar_path.read_text())
    assert "source_relpath" in sidecar
    assert sidecar["source_relpath"] == "subdir/sample_text.pdf"


def test_extract_backfills_source_relpath_on_cache_hit(dossier_root, caplog) -> None:
    """A pre-schema sidecar (no source_relpath) gets backfilled on a cache hit, no VLM call."""
    tmp_path, source_pdf = dossier_root

    extracted_dir = tmp_path / "demo" / "extracted"
    extracted_dir.mkdir(parents=True)

    doc_id = extract._doc_id_from_path(source_pdf)
    md_path = extracted_dir / f"{doc_id}.md"
    sidecar_path = extracted_dir / f"{doc_id}.json"

    md_path.write_text("## Page 1\n\nstale but present", encoding="utf-8")
    old_sidecar = {
        "doc_id": doc_id,
        "source_filename": source_pdf.name,
        # no source_relpath -- simulates a sidecar written before this deliverable
        "kind": "pdf",
        "page_count": 1,
        "extraction_mode": "text_pdf",
        "extractor_model": None,
        "extracted_at": "2026-01-01T00:00:00Z",
        "source_hash": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
    }
    sidecar_path.write_text(json.dumps(old_sidecar), encoding="utf-8")

    with patch("ingestion.dossier.extract.get_anthropic_client") as mock_get_client:
        with caplog.at_level("INFO"):
            result = extract.extract_document(source_pdf, case_id="demo")

    mock_get_client.assert_not_called()
    assert "cached" in caplog.text
    assert "backfilled" in caplog.text

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["source_relpath"] == "sample_text.pdf"
    assert result.doc_id == doc_id


def test_extractor_is_idempotent(dossier_root, caplog) -> None:
    """Second extraction of the same file hits the cache — no re-extraction, no vision call.

    The fixture routes to text_pdf, so the Anthropic client is never invoked
    on either run (that route has no vision step at all). What actually
    proves caching is extract_text_pdf: called once on the real first run,
    not called again on the cached second run.
    """
    _, source_pdf = dossier_root

    with patch(
        "ingestion.dossier.extract.extract_text_pdf", wraps=extract.extract_text_pdf
    ) as spy_extract_text_pdf, patch(
        "ingestion.dossier.extract.get_anthropic_client"
    ) as mock_get_client:
        result1 = extract.extract_document(source_pdf, case_id="demo")
        assert spy_extract_text_pdf.call_count == 1
        mock_get_client.assert_not_called()

        with caplog.at_level("INFO"):
            result2 = extract.extract_document(source_pdf, case_id="demo")

        assert spy_extract_text_pdf.call_count == 1, "extraction re-ran on a cache hit"
        mock_get_client.assert_not_called()

    assert "cached" in caplog.text
    assert result2.doc_id == result1.doc_id
    assert result2.mode == "text_pdf"


def test_extractor_router_selects_text_pdf_for_text() -> None:
    """detect_mimetype routes the typed-text fixture PDF to text_pdf, not image_pdf."""
    assert extract.detect_mimetype(FIXTURE_PDF) == "text_pdf"


# ========== deliverable 2: gate.py ==========

def _write_minimal_pdf(dest: Path, sentences: list[str]) -> None:
    """Write a minimal single-page, single-font PDF, one sentence per line.

    Same hand-rolled-PDF technique as the committed sample_text.pdf fixture,
    generalized to arbitrary sentences and generated on the fly (not
    committed — these are synthetic, throwaway fixtures for gate.py's tests,
    not a demonstrable case document). WinAnsiEncoding + latin-1 content
    bytes so accented French text round-trips correctly.
    """
    content_lines = ["BT", "/F1 12 Tf", "72 700 Td"]
    for i, sentence in enumerate(sentences):
        if i > 0:
            content_lines.append("0 -20 Td")
        escaped = sentence.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\n"
        b"stream\n" + content_stream + b"\nendstream",
    ]

    out = bytearray()
    out += b"%PDF-1.4\n"
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii")
        out += body
        out += b"\nendobj\n"

    xref_offset = len(out)
    n_objs = len(objects) + 1
    out += f"xref\n0 {n_objs}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")

    out += b"trailer\n"
    out += f"<< /Size {n_objs} /Root 1 0 R >>\n".encode("ascii")
    out += b"startxref\n"
    out += f"{xref_offset}\n".encode("ascii")
    out += b"%%EOF"

    dest.write_bytes(bytes(out))


def _fake_chat_response(content: str):
    """Minimal stand-in for an OpenAI-shaped chat.completions.create() response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def gate_case_dir(tmp_path, monkeypatch):
    """Redirect gate.DOSSIER_DIR into an isolated tmp dir with a fake case's extracted/ dir."""
    monkeypatch.setattr(gate, "DOSSIER_DIR", tmp_path)
    extracted_dir = tmp_path / "acme" / "extracted"
    extracted_dir.mkdir(parents=True)
    return tmp_path, extracted_dir


def test_gate_flags_missing_fact(gate_case_dir, tmp_path) -> None:
    """A degraded transcript that drops 'cc-587' should surface in missing_facts."""
    _, extracted_dir = gate_case_dir

    source_pdf = tmp_path / "source.pdf"
    _write_minimal_pdf(
        source_pdf,
        ["L'article cc-587 dispose que le quasi-usufruitier restitue au terme."],
    )

    md_path = extracted_dir / "creance.md"
    md_path.write_text(
        "## Page 1\n\nLe quasi-usufruitier restitue au terme.",  # cc-587 dropped
        encoding="utf-8",
    )

    fake_response = _fake_chat_response(json.dumps({
        "missing_facts": ["L'article cc-587 n'est pas mentionne dans l'extraction"],
        "mistranscriptions": [],
    }))
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response

    with patch("ingestion.dossier.gate.get_anthropic_client", return_value=mock_client):
        report = gate.verify_extraction(source_pdf, md_path)

    assert report.status == "ok"
    assert any("cc-587" in fact for fact in report.missing_facts), report.missing_facts
    assert report.mistranscriptions == []


def test_gate_passes_complete_extraction(gate_case_dir, tmp_path) -> None:
    """A faithful transcript should come back with no missing facts or mistranscriptions."""
    _, extracted_dir = gate_case_dir

    sentence = "L'article cc-587 dispose que le quasi-usufruitier restitue au terme."
    source_pdf = tmp_path / "source.pdf"
    _write_minimal_pdf(source_pdf, [sentence])

    md_path = extracted_dir / "creance.md"
    md_path.write_text(f"## Page 1\n\n{sentence}", encoding="utf-8")

    fake_response = _fake_chat_response(
        json.dumps({"missing_facts": [], "mistranscriptions": []})
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response

    with patch("ingestion.dossier.gate.get_anthropic_client", return_value=mock_client):
        report = gate.verify_extraction(source_pdf, md_path)

    assert report.status == "ok"
    assert report.missing_facts == []
    assert report.mistranscriptions == []


def test_gate_writes_jsonl(gate_case_dir, tmp_path) -> None:
    """verify_extraction appends exactly one well-formed line to coverage.jsonl."""
    dossier_root, extracted_dir = gate_case_dir

    source_pdf = tmp_path / "source.pdf"
    _write_minimal_pdf(source_pdf, ["Une phrase quelconque."])

    md_path = extracted_dir / "creance.md"
    md_path.write_text("## Page 1\n\nUne phrase quelconque.", encoding="utf-8")

    coverage_path = dossier_root / "acme" / "coverage.jsonl"
    assert not coverage_path.exists()

    fake_response = _fake_chat_response(
        json.dumps({"missing_facts": [], "mistranscriptions": []})
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response

    with patch("ingestion.dossier.gate.get_anthropic_client", return_value=mock_client):
        report = gate.verify_extraction(source_pdf, md_path)

    lines = coverage_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["doc_id"] == "creance" == report.doc_id
    assert record["missing_facts"] == []
    assert record["mistranscriptions"] == []
    assert record["status"] == "ok"
    assert record["checked_at"], "checked_at timestamp missing"


def test_gate_case_reads_source_from_sidecar(tmp_path, monkeypatch) -> None:
    """gate_case locates the source file via sidecar source_relpath, not filename matching.

    Puts the source under a subdirectory (raw/sub/facture.pdf) precisely
    because filename-only matching is unsafe once subdirectories exist — the
    private dossier has a duplicate filename in two different places.
    """
    monkeypatch.setattr(gate, "DOSSIER_DIR", tmp_path)

    case_id = "acme"
    raw_subdir = tmp_path / case_id / "raw" / "sub"
    raw_subdir.mkdir(parents=True)
    source_pdf = raw_subdir / "facture.pdf"
    _write_minimal_pdf(source_pdf, ["Une phrase quelconque."])

    extracted_dir = tmp_path / case_id / "extracted"
    extracted_dir.mkdir(parents=True)
    md_path = extracted_dir / "sub__facture.md"
    md_path.write_text("## Page 1\n\nUne phrase quelconque.", encoding="utf-8")

    sidecar_path = extracted_dir / "sub__facture.json"
    sidecar_path.write_text(
        json.dumps({
            "doc_id": "sub__facture",
            "source_filename": "facture.pdf",
            "source_relpath": "sub/facture.pdf",
            "kind": "pdf",
            "page_count": 1,
            "extraction_mode": "text_pdf",
            "extractor_model": None,
            "extracted_at": "2026-01-01T00:00:00Z",
            "source_hash": "irrelevant-for-this-test",
        }),
        encoding="utf-8",
    )

    with patch(
        "ingestion.dossier.gate.verify_extraction", wraps=gate.verify_extraction
    ) as spy_verify, patch(
        "ingestion.dossier.gate._call_verifier",
        return_value=json.dumps({"missing_facts": [], "mistranscriptions": []}),
    ):
        reports = gate.gate_case(case_id)

    spy_verify.assert_called_once_with(source_pdf, md_path)
    assert len(reports) == 1
    assert reports[0].status == "ok"


def test_gate_case_flags_source_missing(tmp_path, monkeypatch) -> None:
    """A sidecar whose source_relpath doesn't resolve to a real file is reported source_missing."""
    monkeypatch.setattr(gate, "DOSSIER_DIR", tmp_path)

    case_id = "acme"
    extracted_dir = tmp_path / case_id / "extracted"
    extracted_dir.mkdir(parents=True)

    md_path = extracted_dir / "ghost.md"
    md_path.write_text("## Page 1\n\ntext", encoding="utf-8")

    sidecar_path = extracted_dir / "ghost.json"
    sidecar_path.write_text(
        json.dumps({
            "doc_id": "ghost",
            "source_filename": "ghost.pdf",
            "source_relpath": "ghost.pdf",  # never actually created under raw/
            "kind": "pdf",
            "page_count": 1,
            "extraction_mode": "text_pdf",
            "extractor_model": None,
            "extracted_at": "2026-01-01T00:00:00Z",
            "source_hash": "irrelevant-for-this-test",
        }),
        encoding="utf-8",
    )

    with patch("ingestion.dossier.gate.verify_extraction") as mock_verify:
        reports = gate.gate_case(case_id)

    mock_verify.assert_not_called()
    assert len(reports) == 1
    assert reports[0].status == "source_missing"
    assert reports[0].doc_id == "ghost"


def test_build_gate_cli_registers_step(monkeypatch, capsys) -> None:
    """--step gate runs without --raw-dir, calling gate.gate_case(case_id) directly.

    gate.gate_case is what build.py's gate branch calls now — no more
    _iter_gate_reports (removed; gate.py locates sources via sidecar
    source_relpath, so build.py no longer needs its own source-lookup loop).
    """
    from ingestion.dossier import build

    stub_report = gate.CoverageReport(
        doc_id="facture", missing_facts=[], mistranscriptions=[], status="ok"
    )
    stub = MagicMock(return_value=[stub_report])
    monkeypatch.setattr(gate, "gate_case", stub)

    monkeypatch.setattr(
        "sys.argv",
        ["build.py", "--case-id", "demo", "--step", "gate"],
    )
    build.main()

    stub.assert_called_once_with("demo")

    out = capsys.readouterr().out
    assert (
        "gate summary · case_id=demo total=1 ok=1 warnings=0 "
        "source_missing=0 parse_failed=0"
        in out
    )


@pytest.mark.parametrize("step", [
    pytest.param("extract", id="extract"),
    pytest.param("all", id="all"),
])
def test_build_steps_reading_raw_files_require_raw_dir(monkeypatch, capsys, step) -> None:
    """Every STAGES_REQUIRING_RAW_DIR step exits via argparse.error.

    argparse cannot express "required unless --step is X" declaratively, so
    build.py enforces it after parsing. The failure must stay a clean exit
    naming the missing flag, not a downstream crash on a missing directory.
    """
    from ingestion.dossier import build

    monkeypatch.setattr(
        "sys.argv", ["build.py", "--case-id", "demo", "--step", step]
    )

    with pytest.raises(SystemExit):
        build.main()

    err = capsys.readouterr().err
    assert "raw-dir" in err
    assert step in err


# ========== deliverable 4: facts.extract_facts_and_roles + extract_case_facts ==========

_FACT_KWARGS = dict(
    fact_id="doc-f001",
    date=None,
    actor_role="notaire_redacteur",
    action="signe l'acte de notoriete",
    target="heritiers",
    verbatim_quote="Le notaire soussigne signe l'acte.",
    source_doc_id="doc",
    source_chunk_id=None,
)


def test_fact_schema_validates() -> None:
    """A Fact built with valid fields validates and round-trips its fields."""
    fact = facts.Fact(**_FACT_KWARGS)
    assert fact.actor_role == "notaire_redacteur"
    assert fact.source_chunk_id is None


def test_fact_actor_role_rejects_invalid_format() -> None:
    """actor_role must be snake_case: no spaces, no capitals, no accents, min 3 chars."""
    kwargs = {k: v for k, v in _FACT_KWARGS.items() if k != "actor_role"}
    for bad_role in ["Maître Jean", "notaire redacteur", "", "ab"]:
        with pytest.raises(ValidationError):
            facts.Fact(actor_role=bad_role, **kwargs)

    fact = facts.Fact(actor_role="notaire_redacteur", **kwargs)
    assert fact.actor_role == "notaire_redacteur"


def test_fact_verbatim_quote_rejects_empty() -> None:
    """verbatim_quote must be non-empty after stripping whitespace."""
    kwargs = {k: v for k, v in _FACT_KWARGS.items() if k != "verbatim_quote"}
    for bad_quote in ["", " "]:
        with pytest.raises(ValidationError):
            facts.Fact(verbatim_quote=bad_quote, **kwargs)


def test_fact_action_word_count_limit() -> None:
    """action allows up to 15 words; 16 words raises."""
    kwargs = {k: v for k, v in _FACT_KWARGS.items() if k != "action"}

    fifteen_words = " ".join(["mot"] * 15)
    fact = facts.Fact(action=fifteen_words, **kwargs)
    assert fact.action == fifteen_words

    sixteen_words = " ".join(["mot"] * 16)
    with pytest.raises(ValidationError):
        facts.Fact(action=sixteen_words, **kwargs)


@pytest.fixture
def facts_case_dir(tmp_path, monkeypatch):
    """Redirect facts.DOSSIER_DIR into an isolated tmp dir with a fake case's extracted/ dir."""
    monkeypatch.setattr(facts, "DOSSIER_DIR", tmp_path)
    extracted_dir = tmp_path / "acme" / "extracted"
    extracted_dir.mkdir(parents=True)
    return tmp_path, extracted_dir


def test_actor_role_catalogue_deduplicates(facts_case_dir, caplog) -> None:
    """Two docs discovering the same role_id with conflicting labels: first-seen wins, warning logged."""
    _, extracted_dir = facts_case_dir
    (extracted_dir / "doc_a.md").write_text("## Page 1\n\nLe notaire agit.", encoding="utf-8")
    (extracted_dir / "doc_b.md").write_text("## Page 1\n\nLe notaire agit encore.", encoding="utf-8")

    response_a = _fake_chat_response(json.dumps({
        "facts": [{"date": None, "actor_role": "notaire_redacteur", "action": "signe l'acte",
                   "target": None, "verbatim_quote": "Le notaire agit."}],
        "roles_discovered": [{"role_id": "notaire_redacteur", "label_fr": "Notaire rédacteur",
                               "grounding_note": "Notaire qui rédige l'acte.", "confidence": "high"}],
        "ambiguities": [],
    }))
    response_b = _fake_chat_response(json.dumps({
        "facts": [{"date": None, "actor_role": "notaire_redacteur", "action": "signe encore",
                   "target": None, "verbatim_quote": "Le notaire agit encore."}],
        "roles_discovered": [{"role_id": "notaire_redacteur", "label_fr": "Notaire instrumentaire",
                               "grounding_note": "Libelle different.", "confidence": "high"}],
        "ambiguities": [],
    }))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [response_a, response_b]

    with patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_client):
        with caplog.at_level("WARNING"):
            summary = facts.extract_case_facts("acme")

    assert summary["unique_roles"] == 1

    roles_path = extracted_dir.parent / "actor_roles.jsonl"
    lines = roles_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    role = json.loads(lines[0])
    assert role["role_id"] == "notaire_redacteur"
    assert role["label_fr"] == "Notaire rédacteur"  # first-seen (doc_a) wins
    assert "conflicting" in caplog.text


def test_extract_facts_deterministic_fact_id(facts_case_dir) -> None:
    """fact_ids are 1-indexed positions in the raw array, stable across repeated calls."""
    _, extracted_dir = facts_case_dir
    md_path = extracted_dir / "doc.md"
    md_path.write_text("## Page 1\n\nTrois faits.", encoding="utf-8")

    response = _fake_chat_response(json.dumps({
        "facts": [
            {"date": None, "actor_role": "notaire_redacteur", "action": "fait un",
             "target": None, "verbatim_quote": "Fait un."},
            {"date": None, "actor_role": "notaire_redacteur", "action": "fait deux",
             "target": None, "verbatim_quote": "Fait deux."},
            {"date": None, "actor_role": "notaire_redacteur", "action": "fait trois",
             "target": None, "verbatim_quote": "Fait trois."},
        ],
        "roles_discovered": [],
        "ambiguities": [],
    }))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response

    with patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_client):
        result1, _, _ = facts.extract_facts_and_roles(md_path, "doc")
        result2, _, _ = facts.extract_facts_and_roles(md_path, "doc")

    expected_ids = ["doc-f001", "doc-f002", "doc-f003"]
    assert [f.fact_id for f in result1] == expected_ids
    assert [f.fact_id for f in result2] == expected_ids


def test_extract_case_facts_idempotent(facts_case_dir) -> None:
    """Running extract_case_facts twice yields identical line counts in all 3 JSONL files."""
    _, extracted_dir = facts_case_dir
    (extracted_dir / "doc_a.md").write_text("## Page 1\n\nUn fait.", encoding="utf-8")
    (extracted_dir / "doc_b.md").write_text("## Page 1\n\nUn autre fait.", encoding="utf-8")

    def make_response(n):
        return _fake_chat_response(json.dumps({
            "facts": [{"date": None, "actor_role": "heritier_nu_proprietaire",
                       "action": f"fait numero {n}", "target": None,
                       "verbatim_quote": f"Fait {n}."}],
            "roles_discovered": [{"role_id": "heritier_nu_proprietaire",
                                   "label_fr": "Héritier nu-propriétaire",
                                   "grounding_note": "Titulaire de la nue-propriete.",
                                   "confidence": "high"}],
            "ambiguities": [],
        }))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        make_response(1), make_response(2), make_response(1), make_response(2),
    ]

    with patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_client):
        facts.extract_case_facts("acme")
        facts.extract_case_facts("acme")

    case_dir = extracted_dir.parent
    facts_lines = (case_dir / "facts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    roles_lines = (case_dir / "actor_roles.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ambig_lines = (case_dir / "role_ambiguities.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(facts_lines) == 2
    assert len(roles_lines) == 1
    assert len(ambig_lines) == 0


def test_extract_case_facts_handles_parse_failure(facts_case_dir, caplog) -> None:
    """One doc returns non-JSON, the other valid JSON: no exception, bad doc contributes 0."""
    _, extracted_dir = facts_case_dir
    (extracted_dir / "doc_bad.md").write_text("## Page 1\n\nTexte illisible.", encoding="utf-8")
    (extracted_dir / "doc_good.md").write_text("## Page 1\n\nUn fait clair.", encoding="utf-8")

    bad_response = _fake_chat_response("This is not JSON at all, sorry!")
    good_response = _fake_chat_response(json.dumps({
        "facts": [{"date": None, "actor_role": "notaire_redacteur", "action": "signe",
                   "target": None, "verbatim_quote": "Un fait clair."}],
        "roles_discovered": [],
        "ambiguities": [],
    }))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [bad_response, good_response]

    with patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_client):
        with caplog.at_level("WARNING"):
            summary = facts.extract_case_facts("acme")

    assert summary["facts_extracted"] == 1
    assert summary["parse_failed_docs"] == 1
    assert "doc_bad" in caplog.text


def test_ambiguity_records_written(facts_case_dir) -> None:
    """A provisional Fact plus its RoleAmbiguity: the ambiguity's fact_ids references the Fact."""
    _, extracted_dir = facts_case_dir
    md_path = extracted_dir / "doc.md"
    md_path.write_text("## Page 1\n\nMaitre VIGNERON envoie un mail.", encoding="utf-8")

    response = _fake_chat_response(json.dumps({
        "facts": [{"date": None, "actor_role": "notaire_associe", "action": "envoie un mail",
                   "target": None, "verbatim_quote": "Maitre VIGNERON envoie un mail."}],
        "roles_discovered": [{"role_id": "notaire_associe", "label_fr": "Notaire associe",
                               "grounding_note": "Notaire titulaire d'une part de societe.",
                               "confidence": "low"}],
        "ambiguities": [{
            "verbatim_quote": "Maitre VIGNERON envoie un mail.",
            "candidate_role_ids": ["notaire_associe", "notaire_stagiaire"],
            "note": "Statut de Maitre VIGNERON incertain faute de contexte.",
        }],
    }))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response

    with patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_client):
        summary = facts.extract_case_facts("acme")

    case_dir = extracted_dir.parent
    facts_lines = (case_dir / "facts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ambig_lines = (case_dir / "role_ambiguities.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(ambig_lines) == 1
    fact = json.loads(facts_lines[0])
    ambiguity = json.loads(ambig_lines[0])

    assert ambiguity["candidate_role_ids"] == ["notaire_associe", "notaire_stagiaire"]
    assert ambiguity["fact_ids"] == [fact["fact_id"]]
    assert summary["ambiguities"] == 1


def test_facts_and_index_case_smoke(tmp_path, monkeypatch) -> None:
    """facts.extract_case_facts, run through build.run_pipeline, produces facts.jsonl,
    actor_roles.jsonl, and role_ambiguities.jsonl for the demo case. Does not assert
    on indexing — that's Deliverable 5."""
    from ingestion.dossier import build

    monkeypatch.setattr(facts, "DOSSIER_DIR", tmp_path)
    extracted_dir = tmp_path / "demo" / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "sample_text.md").write_text(
        "## Page 1\n\nLe notaire signe l'acte de notoriete.", encoding="utf-8"
    )

    response = _fake_chat_response(json.dumps({
        "facts": [{"date": None, "actor_role": "notaire_redacteur",
                   "action": "signe l'acte de notoriete", "target": None,
                   "verbatim_quote": "Le notaire signe l'acte de notoriete."}],
        "roles_discovered": [{"role_id": "notaire_redacteur",
                               "label_fr": "Notaire rédacteur",
                               "grounding_note": "Notaire qui instrumente l'acte.",
                               "confidence": "high"}],
        "ambiguities": [],
    }))

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response

    with patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_client):
        summary = build.run_pipeline("demo", raw_dir=None, step="facts")

    for name in ("facts.jsonl", "actor_roles.jsonl", "role_ambiguities.jsonl"):
        assert (tmp_path / "demo" / name).exists(), f"missing {name}"

    facts_lines = (tmp_path / "demo" / "facts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(facts_lines) == 1
    fact = json.loads(facts_lines[0])
    assert fact["actor_role"] == "notaire_redacteur"
    assert fact["source_chunk_id"] is None

    assert summary["facts_extracted"] == 1
    assert summary["docs_processed"] == 1


# ========== deliverable 5: index.py ==========

def test_chunk_dossier_document_produces_namespace_prefixed_ids(tmp_path) -> None:
    """Every chunk_id matches dossier-<case_id>-<doc_id>-c<NNN>."""
    md_path = tmp_path / "doc.md"
    paragraph = "Phrase repetee pour allonger le texte du document de test. " * 6
    md_path.write_text(f"## Page 1\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}", encoding="utf-8")

    chunks = index.chunk_dossier_document(md_path, doc_id="doc", case_id="acme")

    assert len(chunks) >= 2, "fixture text too short to exercise multi-chunk splitting"
    for chunk_id in chunks["chunk_id"]:
        assert re.match(r"^dossier-acme-doc-c\d{3}$", chunk_id), chunk_id


def test_chunk_dossier_document_deterministic(tmp_path) -> None:
    """Chunking the same file twice yields identical chunk_id and texte lists."""
    md_path = tmp_path / "doc.md"
    paragraph = "Une autre phrase repetee pour forcer un decoupage multiple. " * 6
    md_path.write_text(f"## Page 1\n\n{paragraph}\n\n{paragraph}", encoding="utf-8")

    chunks1 = index.chunk_dossier_document(md_path, doc_id="doc", case_id="acme")
    chunks2 = index.chunk_dossier_document(md_path, doc_id="doc", case_id="acme")

    pd.testing.assert_frame_equal(chunks1, chunks2)


def test_chunk_dossier_document_respects_page_boundaries(tmp_path) -> None:
    """No chunk contains text from two different '## Page N' sections."""
    md_path = tmp_path / "doc.md"
    page1 = "MARKERONE. " + ("Phrase de la premiere page repetee. " * 40)
    page2 = "MARKERTWO. " + ("Phrase de la deuxieme page repetee encore plus. " * 40)
    md_path.write_text(f"## Page 1\n\n{page1}\n\n## Page 2\n\n{page2}", encoding="utf-8")

    chunks = index.chunk_dossier_document(md_path, doc_id="doc", case_id="acme")

    assert len(chunks) >= 2
    for _, row in chunks.iterrows():
        has_one = "MARKERONE" in row["texte"]
        has_two = "MARKERTWO" in row["texte"]
        assert not (has_one and has_two), row["texte"]
        assert row["section_path"] in ("Page 1", "Page 2")


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Redirect index.CHROMA_DIR into an isolated tmp dir with a real, empty
    Chroma collection, so Chroma-append tests never touch data/chroma/.

    Must patch the name bound *inside* ingestion.dossier.index, not
    ingestion.index.CHROMA_DIR — `from ingestion.index import CHROMA_DIR`
    binds a separate local name at import time that patching the origin
    module would not reach.
    """
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(index, "CHROMA_DIR", chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        index.COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    return collection


@pytest.fixture
def isolated_chunks_csv(tmp_path, monkeypatch):
    """Redirect index.CHUNKS_CSV into an isolated tmp CSV seeded with two
    dummy statute rows, so no test run can ever touch the
    real, tracked data/chunks.csv — and so tests can assert statute rows
    survive dossier appends untouched.

    Must patch the name bound *inside* ingestion.dossier.index, not
    ingestion.index.CHUNKS_CSV — same import-time local-binding reason as
    isolated_chroma's docstring.
    """
    statute_columns = [
        "chunk_id", "source", "source_label", "num", "section_path", "titre",
        "texte", "etat", "date_debut", "date_fin", "legiarti_id", "url",
    ]
    seed = pd.DataFrame([
        {"chunk_id": "cc-720", "source": "cc_successions", "source_label": "Code civil",
         "num": "720", "section_path": "", "titre": "", "texte": "dummy statute row",
         "etat": "VIGUEUR", "date_debut": "", "date_fin": "", "legiarti_id": "", "url": ""},
        {"chunk_id": "cc-587", "source": "cc_successions", "source_label": "Code civil",
         "num": "587", "section_path": "", "titre": "", "texte": "dummy statute row",
         "etat": "VIGUEUR", "date_debut": "", "date_fin": "", "legiarti_id": "", "url": ""},
    ], columns=statute_columns)
    csv_path = tmp_path / "statute_chunks.csv"
    seed.to_csv(csv_path, index=False)
    monkeypatch.setattr(index, "CHUNKS_CSV", csv_path)
    return csv_path


@pytest.mark.slow
def test_index_dossier_end_to_end_demo(
    tmp_path, monkeypatch, isolated_chroma, isolated_chunks_csv
) -> None:
    """index_dossier appends dossier chunks to Chroma without touching
    pre-existing (statute-like) rows.

    Slow: loads a real local BGE-M3 model (same justification as
    tests/test_ingestion_smoke.py::test_rag_flow_end_to_end), not because it
    hits any LLM API — this test makes zero API calls.
    """
    collection = isolated_chroma

    model = index.BGEM3FlagModel(index.EMBED_MODEL_ID, use_fp16=False, device="cpu")
    statute_texts = [
        "Les successions s'ouvrent par la mort.",
        "Le quasi-usufruit est un droit reel.",
    ]
    statute_ids = ["cc-720", "cc-587"]
    output = model.encode(
        statute_texts, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    dense_vecs = [[float(v) for v in vec] for vec in output["dense_vecs"]]
    collection.add(
        ids=statute_ids,
        embeddings=dense_vecs,
        documents=statute_texts,
        metadatas=[
            {"source": "cc_successions", "source_label": "Code civil", "num": n,
             "titre": "", "section_path": "", "url": "https://example.test"}
            for n in ("720", "587")
        ],
    )
    before_count = collection.count()
    assert before_count == 2

    monkeypatch.setattr(index, "DOSSIER_DIR", tmp_path)
    extracted_dir = tmp_path / "demo" / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "sample_text.md").write_text(
        "## Page 1\n\nLe notaire signe l'acte de notoriete.", encoding="utf-8"
    )

    result = index.index_dossier("demo")

    assert result.docs_indexed == 1
    assert result.chunks_created >= 1
    assert collection.count() == before_count + result.chunks_created

    all_ids = collection.get(include=[])["ids"]
    dossier_ids = [cid for cid in all_ids if cid.startswith("dossier-demo-")]
    assert len(dossier_ids) == result.chunks_created

    statute_rows = collection.get(ids=statute_ids, include=["documents"])
    assert dict(zip(statute_rows["ids"], statute_rows["documents"])) == dict(
        zip(statute_ids, statute_texts)
    )

    written = pd.read_csv(tmp_path / "demo" / "chunks.csv", keep_default_na=False)
    assert any(cid.startswith("dossier-demo-") for cid in written["chunk_id"])

    # ADR #58: the shared statute CSV is git-tracked, so dossier rows must
    # never reach it. load_index merges the per-case file above at load time
    # instead. Inverted from the pre-#58 assertion that required them here.
    written_shared = pd.read_csv(isolated_chunks_csv, keep_default_na=False)
    dossier_rows = written_shared[written_shared["chunk_id"].str.startswith("dossier-")]
    assert dossier_rows.empty, (
        f"dossier rows leaked into the tracked statute CSV: "
        f"{list(dossier_rows['chunk_id'])[:5]}"
    )
    assert list(written_shared.columns) == [
        "chunk_id", "source", "source_label", "num", "section_path", "titre",
        "texte", "etat", "date_debut", "date_fin", "legiarti_id", "url",
    ]


@pytest.mark.slow
def test_index_dossier_idempotent(
    tmp_path, monkeypatch, isolated_chroma, isolated_chunks_csv
) -> None:
    """Running index_dossier twice does not double-append or duplicate rows."""
    collection = isolated_chroma

    monkeypatch.setattr(index, "DOSSIER_DIR", tmp_path)
    extracted_dir = tmp_path / "demo" / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "sample_text.md").write_text(
        "## Page 1\n\nLe notaire signe l'acte de notoriete.", encoding="utf-8"
    )

    initial = pd.read_csv(isolated_chunks_csv, keep_default_na=False)
    initial_statute_count = len(initial[~initial["chunk_id"].str.startswith("dossier-")])

    result1 = index.index_dossier("demo")
    result2 = index.index_dossier("demo")

    assert result1.chunks_created == result2.chunks_created
    assert collection.count() == result2.chunks_created

    written = pd.read_csv(tmp_path / "demo" / "chunks.csv", keep_default_na=False)
    assert len(written) == result2.chunks_created
    assert written["chunk_id"].is_unique

    # The tracked statute CSV must be untouched by either run (ADR #58):
    # no dossier rows added, no statute rows lost.
    after_run2 = pd.read_csv(isolated_chunks_csv, keep_default_na=False)
    statute_count_2 = len(after_run2[~after_run2["chunk_id"].str.startswith("dossier-")])

    assert not after_run2["chunk_id"].str.startswith("dossier-").any()
    assert statute_count_2 == initial_statute_count


def test_facts_source_chunk_id_backfilled(tmp_path, monkeypatch, isolated_chunks_csv) -> None:
    """Both known facts get source_chunk_id backfilled after index_dossier."""
    monkeypatch.setattr(index, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(index, "append_to_chroma", MagicMock())

    case_dir = tmp_path / "acme"
    extracted_dir = case_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "doc.md").write_text(
        "## Page 1\n\nLe notaire signe l'acte de notoriete. "
        "Les heritiers acceptent la succession.",
        encoding="utf-8",
    )

    fact1 = facts.Fact(
        fact_id="doc-f001", date=None, actor_role="notaire_redacteur",
        action="signe l'acte", target=None,
        verbatim_quote="Le notaire signe l'acte de notoriete.",
        source_doc_id="doc", source_chunk_id=None,
    )
    fact2 = facts.Fact(
        fact_id="doc-f002", date=None, actor_role="heritier_nu_proprietaire",
        action="accepte la succession", target=None,
        verbatim_quote="Les heritiers acceptent la succession.",
        source_doc_id="doc", source_chunk_id=None,
    )
    facts_path = case_dir / "facts.jsonl"
    facts_path.write_text(
        fact1.model_dump_json() + "\n" + fact2.model_dump_json() + "\n", encoding="utf-8"
    )

    result = index.index_dossier("acme")

    assert result.facts_backfilled == 2
    assert result.facts_unmatched == 0

    written_chunk_ids = set(
        pd.read_csv(case_dir / "chunks.csv", keep_default_na=False)["chunk_id"]
    )
    reloaded = [
        facts.Fact.model_validate_json(line)
        for line in facts_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    for fact in reloaded:
        assert fact.source_chunk_id is not None
        assert fact.source_chunk_id.startswith("dossier-acme-")
        assert fact.source_chunk_id in written_chunk_ids


def test_facts_unmatched_logged_not_crashed(
    tmp_path, monkeypatch, caplog, isolated_chunks_csv
) -> None:
    """A fact whose verbatim_quote never appears in any chunk stays unmatched, not raised."""
    monkeypatch.setattr(index, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(index, "append_to_chroma", MagicMock())

    case_dir = tmp_path / "acme"
    extracted_dir = case_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "doc.md").write_text(
        "## Page 1\n\nLe notaire signe l'acte de notoriete.", encoding="utf-8"
    )

    fact = facts.Fact(
        fact_id="doc-f001", date=None, actor_role="notaire_redacteur",
        action="signe l'acte", target=None,
        verbatim_quote="Cette phrase n'existe nulle part dans le document.",
        source_doc_id="doc", source_chunk_id=None,
    )
    facts_path = case_dir / "facts.jsonl"
    facts_path.write_text(fact.model_dump_json() + "\n", encoding="utf-8")

    with caplog.at_level("INFO"):
        result = index.index_dossier("acme")

    assert result.facts_unmatched >= 1
    assert "doc-f001" in caplog.text

    reloaded = facts.Fact.model_validate_json(
        facts_path.read_text(encoding="utf-8").strip()
    )
    assert reloaded.source_chunk_id is None


def test_build_step_all_end_to_end_demo(
    tmp_path, monkeypatch, capsys, isolated_chunks_csv
) -> None:
    """--step all runs extract -> gate -> facts -> index in sequence end to end."""
    from ingestion.dossier import build

    monkeypatch.setattr(extract, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(gate, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(facts, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(index, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(index, "append_to_chroma", MagicMock())

    raw_dir = tmp_path / "demo" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / FIXTURE_PDF.name).write_bytes(FIXTURE_PDF.read_bytes())

    gate_response = _fake_chat_response(
        json.dumps({"missing_facts": [], "mistranscriptions": []})
    )
    facts_response = _fake_chat_response(json.dumps({
        "facts": [{"date": None, "actor_role": "notaire_redacteur",
                   "action": "verifie l'extraction", "target": None,
                   "verbatim_quote": KNOWN_SENTENCES[0]}],
        "roles_discovered": [{"role_id": "notaire_redacteur", "label_fr": "Notaire rédacteur",
                               "grounding_note": "Notaire qui redige l'acte.",
                               "confidence": "high"}],
        "ambiguities": [],
    }))

    mock_gate_client = MagicMock()
    mock_gate_client.chat.completions.create.return_value = gate_response
    mock_facts_client = MagicMock()
    mock_facts_client.chat.completions.create.return_value = facts_response

    with patch("ingestion.dossier.gate.get_anthropic_client", return_value=mock_gate_client), \
         patch("ingestion.dossier.facts.get_anthropic_client", return_value=mock_facts_client), \
         patch("ingestion.dossier.extract.get_anthropic_client") as mock_extract_client:
        summary = build.run_pipeline("demo", raw_dir=raw_dir, step="all")

    mock_extract_client.assert_not_called()  # text_pdf route has no vision step

    case_dir = tmp_path / "demo"
    assert (case_dir / "extracted" / "sample_text.md").exists()
    assert (case_dir / "coverage.jsonl").exists()
    assert (case_dir / "facts.jsonl").exists()
    assert (case_dir / "chunks.csv").exists()

    facts_lines = (case_dir / "facts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(facts_lines) == 1
    fact = json.loads(facts_lines[0])
    assert fact["source_chunk_id"] is not None
    assert fact["source_chunk_id"].startswith("dossier-demo-")

    assert summary["step"] == "all"
    assert summary["extract"]["doc_count"] == 1
    assert summary["index"]["chunks_created"] >= 1

    out = capsys.readouterr().out
    assert "gate summary" in out
    assert "facts summary" in out
    assert "index summary" in out
    assert "all summary" in out
