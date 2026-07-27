"""Verbatim transcripts → structured, atomic legal facts (Plane I · offline).

Each dossier document's verbatim .md transcript (from extract.py) is reduced
to a list of validated, atomic `Fact` records — the fact base that
rag/flow.py's compliance-gap analysis (`flow.analyze(case_id)`) reasons over
alongside the statute corpus.

NOTE: `pydantic` is currently only a transitive dependency (pulled in by
other packages per uv.lock) — it is not yet in `pyproject.toml`
`[project.dependencies]`. Add it with `uv add pydantic` before implementing
this module.

Inputs:  data/dossier/<case_id>/extracted/<doc_id>.md  (from extract.py)
Outputs: data/dossier/<case_id>/facts.jsonl — one JSON line per Fact
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

FACTS_MODEL_ID = "anthropic/claude-opus-4.7"


# ========== Fact schema ==========

class Fact(BaseModel):
    """One atomic, sourced legal fact extracted from a dossier document."""

    fact_id: str            # stable id, e.g. f"{source_doc_id}-{n}"
    date: str                # ISO 8601 date, or "" if undated in the source
    actor_role: str          # e.g. "notaire", "héritier", "usufruitier"
    action: str              # the act performed, in plain French
    target: str               # what/whom the action was directed at
    verbatim_quote: str       # exact source text the fact was derived from
    source_doc_id: str        # doc_id of the originating dossier document
    source_chunk_id: str       # chunk_id once source_doc_id is indexed (index.py)


# ========== extraction ==========

def extract_facts_from_document(extraction_md: str, doc_id: str) -> list[Fact]:
    """One Opus 4.7 call per document: parse extraction_md into a list of Facts."""
    raise NotImplementedError(
        f"spec: issue one {FACTS_MODEL_ID} call with `extraction_md`, "
        "instructing it to enumerate every atomic material fact as structured "
        "output matching the Fact schema (fact_id, date, actor_role, action, "
        "target, verbatim_quote, source_doc_id=doc_id, source_chunk_id — left "
        "'' until index.py assigns real chunk_ids). Validate the structured "
        "output against Fact and return the list of parsed Fact objects."
    )


# ========== persistence + case orchestration ==========

def write_facts(case_id: str, facts: list[Fact]) -> Path:
    """Append Facts as JSON lines to data/dossier/<case_id>/facts.jsonl."""
    raise NotImplementedError(
        "spec: append one JSON line per Fact (via Fact.model_dump_json()) to "
        "DOSSIER_DIR/<case_id>/facts.jsonl, creating parent dirs as needed, "
        "and return the path written to."
    )


def extract_case_facts(case_id: str) -> list[Fact]:
    """Extract facts from every extracted document in a case; write and return them."""
    raise NotImplementedError(
        "spec: for each <doc_id>.md under "
        "DOSSIER_DIR/<case_id>/extracted/, call "
        "extract_facts_from_document(extraction_md, doc_id), accumulate all "
        "Facts, call write_facts(case_id, facts), and return the combined list."
    )
