"""Coverage gate: catch VLM extraction dropout before facts.py trusts the transcript
(Plane I · offline).

extract.py's per-page VLM transcription can silently drop material facts
(illegible stamps, skipped margins, truncated tables). This module re-reads
the original source pages with a cheap evaluator model (Haiku 4.5) and asks
it to name any materially relevant fact present in the source but absent
from the extraction — a coverage check, not a re-extraction.

Inputs:  (source_pages: list[bytes], extraction_md: str) per document —
         source_pages are the same rasterized page images extract.py used
         (or would have used) for VLM extraction; extraction_md is the
         .md transcript extract.py already wrote.
Outputs: data/dossier/<case_id>/coverage.jsonl — one JSON line per document:
             {doc_id, checked_at, missing_facts: list[str], verdict}

Downstream: build.py treats a non-empty missing_facts list as a pipeline
warning (surfaced to the operator), not a hard failure — facts.py still
runs against whatever extraction_md contains.
"""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

GATE_MODEL_ID = "anthropic/claude-haiku-4.5"


# ========== evaluator ==========

def evaluate_coverage(source_pages: list[bytes], extraction_md: str, doc_id: str) -> list[str]:
    """One Haiku 4.5 call comparing source pages to the extraction; list missing facts."""
    raise NotImplementedError(
        f"spec: issue one {GATE_MODEL_ID} call with `source_pages` and "
        "`extraction_md`, instructing it to list any materially relevant fact "
        "(dates, names, amounts, signatures, stamps) visible in the source "
        "pages but absent or garbled in extraction_md. Return the list of "
        "missing-fact descriptions as plain strings; an empty list means "
        "full coverage."
    )


# ========== document + case orchestration ==========

def gate_document(case_id: str, doc_id: str, source_pages: list[bytes], extraction_md: str) -> dict:
    """Run evaluate_coverage for one document and append the result to coverage.jsonl."""
    raise NotImplementedError(
        "spec: call evaluate_coverage(source_pages, extraction_md, doc_id); "
        "build a record {doc_id, checked_at, missing_facts, verdict} where "
        "verdict is 'complete' if missing_facts is empty else 'incomplete'; "
        "append the record as one JSON line to "
        "DOSSIER_DIR/<case_id>/coverage.jsonl and return the record."
    )


def gate_case(case_id: str) -> list[dict]:
    """Run gate_document over every extracted document in a case; return all records."""
    raise NotImplementedError(
        "spec: for each <doc_id>.md under "
        "DOSSIER_DIR/<case_id>/extracted/, load its sidecar to locate the "
        "original source pages, call gate_document(case_id, doc_id, "
        "source_pages, extraction_md), and return the list of logged records."
    )
