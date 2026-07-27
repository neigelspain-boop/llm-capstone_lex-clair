"""Raw dossier documents → verbatim markdown transcripts (Plane I · offline).

Routes each raw file in a case's dossier by mimetype and produces a verbatim
transcript plus a JSON metadata sidecar. Three routes:

- text_pdf   — PDF where every page yields >50 characters via pdfplumber:
               deterministic text-layer extraction, no LLM call.
- image_pdf  — PDF where any page yields <=50 characters via pdfplumber
               (scan-inside-a-PDF wrapper): pages rasterized via pdf2image,
               one Qwen3-VL vision call per page (via OpenRouter's
               OpenAI-compatible surface).
- image      — standalone image file (.png/.jpg/.jpeg): one Qwen3-VL vision
               call for the single page (via OpenRouter's OpenAI-compatible
               surface).

Vision calls go one page at a time — never batched — to preserve fidelity on
dense pages. The system prompt enforces verbatim transcription: preserve
line breaks and structure, no paraphrase/summary/interpretation, strikethrough
markdown for struck-through text, preserve French diacritics exactly, mark
unreadable spans `[illisible]`, and reject handwriting as
`[handwritten — out of scope]` per project constraint.

Per-page outputs (from either route) are concatenated into a single .md file
with `## Page N` H2 dividers.

Inputs:  data/dossier/<case_id>/raw/**  (mixed mimetypes, heir-supplied documents,
         arbitrarily nested in subdirectories, e.g. raw/Bossavit_s/01_Creance/*.pdf)
Outputs: data/dossier/<case_id>/extracted/<doc_id>.md    — verbatim transcript
         data/dossier/<case_id>/extracted/<doc_id>.json  — sidecar metadata:
             doc_id, source_filename, source_relpath, kind, page_count,
             extraction_mode, extractor_model, extracted_at (ISO 8601),
             source_hash (sha256)

<doc_id> is derived deterministically from the source path via slugify — no
random component, so re-running extraction on the same raw file always
targets the same output paths. extracted/ stays flat (no mirrored subtree):
every path segment between the nearest ancestor "raw" directory and the file
itself is slugified (strip accents, lowercase, spaces/punctuation →
underscores) and joined with "__", so raw/Bossavit_s/01_Creance/facture.pdf
becomes doc_id "bossavit_s__01_creance__facture" — collision-safe across
subdirectories without downstream stages needing to walk a tree. Idempotent:
if the .md exists and the sidecar's source_hash matches the current file's
hash, extraction is skipped (logged as "cached") and no vision call is made.

source_relpath (the source path relative to data/dossier/<case_id>/raw/) is
what lets gate.py relocate each document's original file later — filename
alone is not collision-safe across subdirectories (the private dossier has
at least one duplicate filename in two different places). A cache hit on a
sidecar written before this field existed backfills it in place, at no VLM
cost, so already-extracted cases upgrade just by re-running extract.

Downstream consumers: gate.py (coverage check against source pages),
facts.py (structured fact extraction from the .md), index.py (chunk + index
the .md into the hybrid retrieval store).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdfplumber

from ingestion.clients import get_anthropic_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

VLM_MODEL_ID = "qwen/qwen3-vl-235b-a22b-instruct"

# Minimum extracted characters per page for a PDF page to count as "text".
TEXT_PAGE_MIN_CHARS = 50

_SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

ExtractionMode = Literal["text_pdf", "image_pdf", "image"]

_VLM_SYSTEM_PROMPT = """\
Tu es un système de transcription verbatim de documents juridiques français.

Règles strictes :
- Transcris VERBATIM le texte visible sur la page, sans jamais paraphraser ni résumer.
- Préserve les sauts de ligne et la structure des sections (titres, listes, paragraphes, tableaux).
- N'interprète pas le sens du texte : transcris uniquement ce qui est écrit.
- Le texte barré (biffé) doit être transcrit en markdown strikethrough, ex : ~~texte barré~~.
- Préserve tous les diacritiques français exactement (é, è, à, ç, etc.).
- Marque tout contenu illisible par [illisible].
- Le contenu manuscrit est hors périmètre : remplace-le par [handwritten — out of scope], \
ne tente pas de le transcrire.

Ne produis aucun commentaire ni explication — uniquement la transcription verbatim de la page.\
"""


# ========== result type ==========

@dataclass
class ExtractResult:
    """Outcome of extracting one dossier document."""

    doc_id: str
    md_path: Path
    sidecar_path: Path
    mode: ExtractionMode
    page_count: int


# ========== path → doc_id ==========

def _slugify_segment(segment: str) -> str:
    """Strip accents, lowercase, and collapse anything non-alnum into underscores."""
    normalized = unicodedata.normalize("NFKD", segment)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    return ascii_only or "doc"


def _doc_id_from_path(path: Path) -> str:
    """Deterministic, collision-safe doc_id from a source path's subtree under raw/.

    Segments from the nearest ancestor directory named "raw" down to (and
    including) the filename stem are each slugified and joined with "__", so
    files in different subdirectories never collide even after flattening
    into extracted/. Files with no "raw" ancestor in their path (e.g. ad hoc
    test fixtures) fall back to just the slugified filename stem.
    """
    parts = path.parent.parts
    if "raw" in parts:
        last_raw_idx = len(parts) - 1 - parts[::-1].index("raw")
        rel_parts = parts[last_raw_idx + 1:]
    else:
        rel_parts = ()
    segments = [_slugify_segment(p) for p in rel_parts] + [_slugify_segment(path.stem)]
    return "__".join(segments)


# ========== mimetype router ==========

def detect_mimetype(path: Path) -> ExtractionMode:
    """Classify a raw dossier file: text-layer PDF, image-only PDF, or a bare image."""
    suffix = path.suffix.lower()

    if suffix in _IMAGE_SUFFIXES:
        return "image"

    if suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) <= TEXT_PAGE_MIN_CHARS:
                    return "image_pdf"
        return "text_pdf"

    raise ValueError(f"unsupported dossier file type: {suffix!r} ({path})")


# ========== extraction routes ==========

def extract_text_pdf(path: Path) -> str:
    """Pull verbatim text directly from a text-layer PDF, no VLM call."""
    pages_md = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages_md.append(f"## Page {i}\n\n{text}")
    return "\n\n".join(pages_md)


def extract_via_vlm(pages: list[bytes], doc_id: str) -> str:
    """Send each page image to Qwen3-VL vision (one call per page) and concatenate a verbatim transcript."""
    client = get_anthropic_client()
    pages_md = []
    for i, page_png in enumerate(pages, start=1):
        b64 = base64.b64encode(page_png).decode("ascii")
        response = client.chat.completions.create(
            model=VLM_MODEL_ID,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Transcris verbatim la page {i} de ce document (doc_id={doc_id}).",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        pages_md.append(f"## Page {i}\n\n{text}")
    return "\n\n".join(pages_md)


# ========== document + case orchestration ==========

def extract_document(source_path: Path, case_id: str) -> ExtractResult:
    """Route one raw file through the correct extraction path and write its outputs.

    doc_id is derived from source_path via _doc_id_from_path — not passed
    in — so the same file always resolves to the same output paths. Writes
    data/dossier/<case_id>/extracted/<doc_id>.md and the matching .json
    sidecar; returns an ExtractResult. Idempotent — skips re-extraction (and
    any vision call) if the source file's hash matches what's already
    recorded in the sidecar.
    """
    path = Path(source_path).resolve()
    doc_id = _doc_id_from_path(path)
    extracted_dir = DOSSIER_DIR / case_id / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    md_path = extracted_dir / f"{doc_id}.md"
    sidecar_path = extracted_dir / f"{doc_id}.json"

    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    if md_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("source_hash") == source_hash:
            log.info("cached extraction for %s (doc_id=%s)", path.name, doc_id)

            if "source_relpath" not in sidecar:
                sidecar["source_relpath"] = str(
                    path.relative_to(DOSSIER_DIR / case_id / "raw")
                )
                sidecar_path.write_text(
                    json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                log.info("backfilled source_relpath in sidecar for doc_id=%s", doc_id)

            return ExtractResult(
                doc_id=sidecar["doc_id"],
                md_path=md_path,
                sidecar_path=sidecar_path,
                mode=sidecar["extraction_mode"],
                page_count=sidecar["page_count"],
            )

    mode = detect_mimetype(path)

    if mode == "text_pdf":
        md_text = extract_text_pdf(path)
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
        extractor_model = None
        kind = "pdf"

    elif mode == "image_pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(path)
        pages_png: list[bytes] = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            pages_png.append(buf.getvalue())
        md_text = extract_via_vlm(pages_png, doc_id)
        page_count = len(pages_png)
        extractor_model = VLM_MODEL_ID
        kind = "pdf"

    else:  # image
        from PIL import Image

        image = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        md_text = extract_via_vlm([buf.getvalue()], doc_id)
        page_count = 1
        extractor_model = VLM_MODEL_ID
        kind = "image"

    md_path.write_text(md_text, encoding="utf-8")

    sidecar = {
        "doc_id": doc_id,
        "source_filename": path.name,
        "source_relpath": str(path.relative_to(DOSSIER_DIR / case_id / "raw")),
        "kind": kind,
        "page_count": page_count,
        "extraction_mode": mode,
        "extractor_model": extractor_model,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_hash": source_hash,
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return ExtractResult(
        doc_id=doc_id,
        md_path=md_path,
        sidecar_path=sidecar_path,
        mode=mode,
        page_count=page_count,
    )


def extract_case(
    case_id: str, raw_dir: Path, limit: int | None = None
) -> list[ExtractResult]:
    """Extract every raw document under a case's raw dir, recursing into subdirectories.

    `limit` caps the number of documents processed (in sorted path order) —
    useful for smoke-testing a large dossier without a full run.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    if limit is not None:
        paths = paths[:limit]

    results: list[ExtractResult] = []
    seen_doc_ids: set[str] = set()

    for path in paths:
        doc_id = _doc_id_from_path(path)
        if doc_id in seen_doc_ids:
            raise ValueError(
                f"duplicate doc_id {doc_id!r} derived from {path!r} "
                f"in case {case_id!r} — rename the source file or subdirectory"
            )
        seen_doc_ids.add(doc_id)
        results.append(extract_document(path, case_id))

    return results
