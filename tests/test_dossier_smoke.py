"""Smoke tests for the dossier ingestion pipeline (Plane I).

Mirrors tests/test_ingestion_smoke.py's role for the statute corpus: guards
the dossier pipeline's public entry points against import/signature drift.

Deliverables 1-2 (extract.py, gate.py) are implemented — their tests run for
real against the committed fixture data/dossier/demo/raw/sample_text.pdf (and,
for gate.py, small synthetic PDFs built on the fly) with the Anthropic client
mocked. Deliverable 3 (facts.py + index.py) remains skipped: no
implementation exists yet behind facts.extract_case_facts or index.index_case
(both raise NotImplementedError by design, see ingestion/dossier/*.py).
Unskip and fill in assertions once that deliverable lands.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingestion.dossier import extract, gate

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


def test_build_extract_requires_raw_dir(monkeypatch, capsys) -> None:
    """--step extract without --raw-dir exits cleanly via argparse.error, not a downstream crash."""
    from ingestion.dossier import build

    monkeypatch.setattr(
        "sys.argv", ["build.py", "--case-id", "demo", "--step", "extract"]
    )

    with pytest.raises(SystemExit):
        build.main()

    err = capsys.readouterr().err
    assert "raw-dir" in err
    assert "extract" in err


def test_build_all_requires_raw_dir(monkeypatch, capsys) -> None:
    """--step all without --raw-dir exits cleanly via argparse.error, not a downstream crash."""
    from ingestion.dossier import build

    monkeypatch.setattr(
        "sys.argv", ["build.py", "--case-id", "demo", "--step", "all"]
    )

    with pytest.raises(SystemExit):
        build.main()

    err = capsys.readouterr().err
    assert "raw-dir" in err
    assert "all" in err


# ========== deliverable 3: facts.extract_case_facts + index.index_case ==========

@pytest.mark.skip("deliverable 3 pending")
def test_facts_and_index_case_smoke() -> None:
    """facts.extract_case_facts + index.index_case should produce facts.jsonl and
    indexed dossier chunks appended to the shared Chroma collection."""
    pass
