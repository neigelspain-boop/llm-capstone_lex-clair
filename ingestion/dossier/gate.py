"""Faithfulness gate: verify extraction transcripts against source pages
(Plane I · offline).

extract.py's transcription (text-layer read or per-page VLM) can silently
drop or distort material facts (illegible stamps, skipped margins, truncated
tables, a misread digit in an amount). This module re-renders the original
source pages and asks a cheap evaluator model (Haiku 4.5) to compare them
against the extraction .md — a faithfulness check, not a re-extraction.

verify_extraction does NOT fix anything. It only reports. Fixing a flagged
extraction (re-running extract.py with a tuned prompt, or a manual pass) is
a later decision, out of scope for this module.

gate_case(case_id) verifies every extracted document in a case in one call.
It locates each source file via the sidecar's `source_relpath` field
(written by extract.py, relative to data/dossier/<case_id>/raw/) rather than
needing a separate raw_dir argument or matching on filename alone — filename
matching is unsafe, since the dossier can contain duplicate filenames across
different subdirectories. Sidecars written before source_relpath existed —
or whose recorded source file no longer resolves on disk — are reported
with status="source_missing" and are not verified; re-running extract.py
backfills the field at no VLM cost.

Inputs:  a case document's original source file (PDF/image) + its
         extraction .md written by extract.py (data/dossier/<case_id>/
         extracted/<doc_id>.md) — case_id and doc_id are both derived from
         md_path's own location, not passed in separately.
Outputs: data/dossier/<case_id>/coverage.jsonl — one JSON line per
         verification: {doc_id, checked_at, missing_facts, mistranscriptions,
         status}

Note on rasterization: extract.py's page-rasterization (pdf2image for PDFs,
Pillow for standalone images) is inlined inside extract_document's branches,
not factored into a standalone helper — and this module intentionally does
not modify extract.py. _rasterize_source_pages below re-implements the same
approach independently for the gate's own use, always rasterizing every PDF
page for vision comparison regardless of whether extract.py routed that
document through text_pdf or image_pdf (the point of the gate is a redundant
check that doesn't trust extract.py's own routing decision).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ingestion.clients import (get_anthropic_client, recover_json_objects,
                               strip_json_fences)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

GATE_MODEL_ID = "anthropic/claude-haiku-4.5"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

_VERIFIER_SYSTEM_PROMPT = """\
You are a faithfulness verifier. Compare the source document pages (images) \
with the extracted markdown transcript. Return a JSON object with keys:

- "missing_facts": a list of strings, each describing one material fact \
present in the source but not in the extraction.
- "mistranscriptions": a list of objects {"claimed": "...", "actual": "..."} \
where the extraction says something different from what the source shows.

Return an empty list for either key if none are found. Focus on: dates, \
monetary amounts, proper nouns, article numbers, article citations, and \
signatures. Ignore formatting differences (line breaks, whitespace, markdown \
styling).

Respond with JSON only — no markdown fences, no surrounding text.\
"""


# ========== result type ==========

@dataclass
class CoverageReport:
    """Outcome of verifying one extraction against its source pages."""

    doc_id: str
    missing_facts: list[str]
    mistranscriptions: list[dict]
    status: str  # "ok" | "parse_recovered" | "parse_failed" | "source_missing"


# ========== source rasterization ==========

def _rasterize_source_pages(source_path: Path) -> list[bytes]:
    """Render every page of a source document as a PNG image for vision comparison."""
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(source_path)
    elif suffix in _IMAGE_SUFFIXES:
        from PIL import Image

        images = [Image.open(source_path).convert("RGB")]
    else:
        raise ValueError(
            f"unsupported source file type for gate check: {suffix!r} ({source_path})"
        )

    pages_png: list[bytes] = []
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        pages_png.append(buf.getvalue())
    return pages_png


# ========== verifier call + response parsing ==========

class _PartialVerdict(Exception):
    """A verdict recovered from a truncated response — real, but incomplete.

    Carried as an exception rather than a return value so no caller can mistake
    it for a clean parse. A fidelity gate that under-reports is worse than one
    that admits it does not know.
    """

    def __init__(self, data: dict):
        super().__init__("verifier response truncated; recovered partial verdict")
        self.data = data


def _parse_verifier_response(raw: str) -> dict:
    """Parse the verifier's raw JSON text into {missing_facts, mistranscriptions}.

    Raises ValueError on any parse or shape failure — caller records
    status="parse_failed" and logs the raw response.
    """
    text = strip_json_fences(raw)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate trailing prose after valid JSON, as compliance does.
        try:
            data, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError as e:
            # Truncated. Recover the mistranscription objects that closed, but
            # NEVER present the result as a complete verdict: `missing_facts`
            # is a list of strings, so a cut-off list silently *shortens* the
            # list of problems and would make the document look more faithful
            # than it is. The caller records `parse_recovered`, which does not
            # satisfy `require_gate_status: ok`.
            recovered = recover_json_objects(text)
            if not recovered:
                raise ValueError(f"JSON parse failed: {e}") from e
            raise _PartialVerdict(
                {"missing_facts": [], "mistranscriptions": recovered}
            ) from e

    missing_facts = data.get("missing_facts", [])
    mistranscriptions = data.get("mistranscriptions", [])

    if not isinstance(missing_facts, list) or not all(
        isinstance(x, str) for x in missing_facts
    ):
        raise ValueError(f"missing_facts must be a list of strings, got {missing_facts!r}")

    if not isinstance(mistranscriptions, list) or not all(
        isinstance(x, dict) and "claimed" in x and "actual" in x
        for x in mistranscriptions
    ):
        raise ValueError(
            "mistranscriptions must be a list of {claimed, actual} dicts, "
            f"got {mistranscriptions!r}"
        )

    return {"missing_facts": missing_facts, "mistranscriptions": mistranscriptions}


def _call_verifier(pages_png: list[bytes], md_text: str, doc_id: str) -> str:
    """One Haiku 4.5 call comparing rasterized source pages to the extraction markdown."""
    client = get_anthropic_client()

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Extraction markdown to verify (doc_id={doc_id}):\n\n"
                f"{md_text}\n\n"
                "The following images are the original source pages, in order. "
                "Compare them against the markdown above and return your JSON verdict."
            ),
        }
    ]
    for page_png in pages_png:
        b64 = base64.b64encode(page_png).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )

    response = client.chat.completions.create(
        model=GATE_MODEL_ID,
        messages=[
            {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content or ""


# ========== coverage.jsonl persistence ==========

def _append_coverage_record(case_id: str, report: CoverageReport) -> Path:
    """Append one CoverageReport as a JSON line to data/dossier/<case_id>/coverage.jsonl."""
    coverage_path = DOSSIER_DIR / case_id / "coverage.jsonl"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "doc_id": report.doc_id,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "missing_facts": report.missing_facts,
        "mistranscriptions": report.mistranscriptions,
        "status": report.status,
    }
    with coverage_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return coverage_path


# ========== public entry points ==========

def verify_extraction(source_path: Path, md_path: Path) -> CoverageReport:
    """Verify one extraction transcript against its source pages; log + return the report.

    doc_id is derived from md_path's filename stem, case_id from md_path's
    grandparent directory name — both follow extract.py's fixed layout
    (data/dossier/<case_id>/extracted/<doc_id>.md), so neither needs to be
    passed in separately.
    """
    source_path = Path(source_path)
    md_path = Path(md_path)

    if md_path.parent.name != "extracted":
        raise ValueError(
            f"md_path must live under an 'extracted' directory, got {md_path}"
        )
    doc_id = md_path.stem
    case_id = md_path.parent.parent.name

    pages_png = _rasterize_source_pages(source_path)
    md_text = md_path.read_text(encoding="utf-8")

    raw = _call_verifier(pages_png, md_text, doc_id)

    try:
        parsed = _parse_verifier_response(raw)
        missing_facts = parsed["missing_facts"]
        mistranscriptions = parsed["mistranscriptions"]
        status = "ok"
    except _PartialVerdict as partial:
        log.warning(
            "gate: verifier response truncated for doc_id=%s — recovered %d "
            "mistranscription(s); fidelity recorded as unverified, not ok",
            doc_id, len(partial.data["mistranscriptions"]),
        )
        missing_facts = partial.data["missing_facts"]
        mistranscriptions = partial.data["mistranscriptions"]
        status = "parse_recovered"
    except ValueError as e:
        log.error(
            "gate parse failed for doc_id=%s: %s\nraw response: %s", doc_id, e, raw
        )
        missing_facts = []
        mistranscriptions = []
        status = "parse_failed"

    report = CoverageReport(
        doc_id=doc_id,
        missing_facts=missing_facts,
        mistranscriptions=mistranscriptions,
        status=status,
    )
    _append_coverage_record(case_id, report)

    return report


def gate_case(case_id: str) -> list[CoverageReport]:
    """Verify every extracted document in a case against its source pages.

    Locates each source file via the sidecar's `source_relpath` field
    (relative to data/dossier/<case_id>/raw/). Sidecars missing this
    field — i.e. extractions produced before the schema was extended —
    are returned with status='source_missing' and skipped from
    verification; re-running extract on the case will backfill the
    field with no VLM cost.
    """
    extracted_dir = DOSSIER_DIR / case_id / "extracted"
    sidecar_paths = sorted(extracted_dir.glob("*.json")) if extracted_dir.exists() else []

    reports: list[CoverageReport] = []
    for sidecar_path in sidecar_paths:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        doc_id = sidecar.get("doc_id", sidecar_path.stem)
        md_path = sidecar_path.with_suffix(".md")

        source_relpath = sidecar.get("source_relpath")
        if source_relpath is None:
            log.warning(
                "gate_case: sidecar for doc_id=%s has no source_relpath "
                "(re-run extract to backfill); reporting source_missing",
                doc_id,
            )
            reports.append(
                CoverageReport(
                    doc_id=doc_id, missing_facts=[], mistranscriptions=[],
                    status="source_missing",
                )
            )
            continue

        source_path = DOSSIER_DIR / case_id / "raw" / source_relpath
        if not source_path.exists():
            log.warning(
                "gate_case: source file for doc_id=%s not found at %s; "
                "reporting source_missing",
                doc_id, source_path,
            )
            reports.append(
                CoverageReport(
                    doc_id=doc_id, missing_facts=[], mistranscriptions=[],
                    status="source_missing",
                )
            )
            continue

        reports.append(verify_extraction(source_path, md_path))

    return reports
