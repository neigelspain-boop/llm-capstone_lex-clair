"""Raw dossier documents → verbatim markdown transcripts (Plane I · offline).

Routes each raw file in a case's dossier by mimetype and produces a verbatim
transcript plus a JSON metadata sidecar. Three routes:

- text_pdf   — PDF with an extractable text layer: pulled directly, no LLM call.
- image_pdf  — scanned/rasterized PDF with no text layer: one Anthropic Opus
               4.7 vision call per page.
- image      — standalone image file (jpg/png/tiff/...): one Opus 4.7 vision
               call for the single page.

Inputs:  data/dossier/<case_id>/raw/*  (mixed mimetypes, heir-supplied documents)
Outputs: data/dossier/<case_id>/extracted/<doc_id>.md    — verbatim transcript
         data/dossier/<case_id>/extracted/<doc_id>.json  — sidecar metadata:
             doc_id, source_path, mimetype, route, page_count, model_used,
             extracted_at, warnings

Downstream consumers: gate.py (coverage check against source pages),
facts.py (structured fact extraction from the .md), index.py (chunk + index
the .md into the hybrid retrieval store).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

VLM_MODEL_ID = "anthropic/claude-opus-4.7"


# ========== mimetype router ==========

def detect_mimetype(path: Path) -> Literal["text_pdf", "image_pdf", "image"]:
    """Classify a raw dossier file: text-layer PDF, image-only PDF, or a bare image."""
    raise NotImplementedError(
        "spec: inspect `path` — for .pdf, probe for an extractable text layer "
        "(text_pdf) vs. none (image_pdf); for image extensions (.jpg/.png/.tiff/...), "
        "return 'image'. Raise ValueError on an unsupported extension."
    )


# ========== extraction routes ==========

def extract_text_pdf(path: Path) -> str:
    """Pull verbatim text directly from a text-layer PDF, no VLM call."""
    raise NotImplementedError(
        "spec: extract the embedded text layer from every page of `path` in "
        "reading order and return it verbatim as a single markdown string. "
        "No LLM call — this route exists precisely to avoid one."
    )


def extract_via_vlm(pages: list[bytes], doc_id: str) -> str:
    """Send each page image to Opus 4.7 vision and concatenate a verbatim transcript."""
    raise NotImplementedError(
        "spec: for each page image in `pages` (in order), issue one "
        f"{VLM_MODEL_ID} vision call instructing verbatim transcription "
        "(no summarization, no paraphrase, preserve original language). "
        "Concatenate per-page markdown output in page order, separated by "
        "a page-break marker, and return the combined string."
    )


# ========== document + case orchestration ==========

def extract_document(path: Path, case_id: str, doc_id: str) -> dict:
    """Route one raw file through the correct extraction path and write its outputs.

    Writes data/dossier/<case_id>/extracted/<doc_id>.md and the matching
    .json sidecar; returns the sidecar dict.
    """
    raise NotImplementedError(
        "spec: call detect_mimetype(path); dispatch to extract_text_pdf for "
        "'text_pdf' or rasterize pages + extract_via_vlm for 'image_pdf'/'image'. "
        "Write the transcript to "
        "DOSSIER_DIR/<case_id>/extracted/<doc_id>.md and a sidecar JSON with "
        "{doc_id, source_path, mimetype, route, page_count, model_used, "
        "extracted_at, warnings} to the sibling .json path. Return the sidecar dict."
    )


def extract_case(case_id: str, raw_dir: Path) -> list[dict]:
    """Extract every raw document in a case's dossier; return all sidecar dicts."""
    raise NotImplementedError(
        "spec: iterate files under `raw_dir`, assign each a stable doc_id "
        "(derived from filename), call extract_document(path, case_id, doc_id) "
        "for each, and return the list of resulting sidecar dicts. Continue "
        "past per-document failures and surface them in the returned sidecars "
        "rather than aborting the whole case."
    )
