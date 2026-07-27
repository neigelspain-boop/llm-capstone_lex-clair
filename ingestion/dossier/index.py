"""Dossier transcripts → incremental additions to the hybrid retrieval store
(Plane I · offline).

Chunks each case's extracted .md transcripts and appends them to the
existing BM25 + Chroma indexes built by ingestion/index.py — it does NOT
rebuild the corpus from scratch. This is the key contrast with
ingestion.index.build_chroma, which shutil.rmtree()s and recreates the
Chroma directory; that path would destroy the statute corpus embeddings
every time a new case dossier is indexed.

Inputs:  data/dossier/<case_id>/extracted/<doc_id>.md
Outputs: - data/dossier/<case_id>/chunks.csv — appended dossier-local chunk
            rows (never touches data/chunks.csv, the statute corpus CSV)
         - the existing Chroma collection (ingestion.index.COLLECTION) —
            appended to via collection.add(), not recreated
         chunk_id format: dossier-<case_id>-<doc_id>-<chunk_num>

Every `pd.read_csv` call added to this module's implementation MUST pass
`keep_default_na=False` (project-wide convention — see ingestion/chunk.py,
ingestion/index.py; guards against the pandas NaN silent-failure trap).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Reuse the statute corpus's Chroma/BGE-M3 config rather than duplicating it.
from ingestion.index import CHROMA_DIR, COLLECTION, EMBED_MODEL_ID, infer_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"


# ========== chunking ==========

def chunk_dossier_document(md_text: str, case_id: str, doc_id: str) -> pd.DataFrame:
    """Split one extraction transcript into dossier chunk rows."""
    raise NotImplementedError(
        "spec: split `md_text` into chunk-sized rows, reusing "
        "ingestion.chunk granularity logic where the shape matches (article- "
        "vs section-level splitting doesn't directly apply to free-form "
        "dossier prose — adapt, don't force-fit). Assign "
        "chunk_id = f'dossier-{case_id}-{doc_id}-{n}' for n in reading order. "
        "Return a DataFrame with at minimum the columns required by "
        "ingestion.index._validate_chunk_schema (chunk_id, source, "
        "source_label, num, titre, section_path, texte, url) so it can be "
        "embedded and indexed with the same code path as the statute corpus."
    )


# ========== index append (no rebuild) ==========

def append_to_chroma(chunks: pd.DataFrame, device: str | None = None) -> None:
    """Embed chunk texts with BGE-M3 and append them to the existing Chroma collection."""
    raise NotImplementedError(
        "spec: open the existing persistent Chroma collection at CHROMA_DIR "
        "(client.get_collection(COLLECTION) — must already exist, do not "
        "create/recreate it), embed `chunks['texte']` with EMBED_MODEL_ID via "
        "infer_device(device), and call collection.add(ids=..., "
        "embeddings=..., documents=..., metadatas=...) for the new chunk_ids "
        "only. Must not call ingestion.index.build_chroma or otherwise touch "
        "existing statute-corpus rows."
    )


def append_to_dossier_chunks_csv(chunks: pd.DataFrame, case_id: str) -> Path:
    """Append dossier chunk rows to data/dossier/<case_id>/chunks.csv."""
    raise NotImplementedError(
        "spec: append `chunks` to DOSSIER_DIR/<case_id>/chunks.csv, creating "
        "the file with a header if it doesn't exist yet, else appending "
        "without a duplicate header. Never write to data/chunks.csv (the "
        "statute corpus CSV owned by ingestion/chunk.py). Return the path "
        "written to."
    )


# ========== case orchestration ==========

def index_case(case_id: str) -> pd.DataFrame:
    """Chunk and index every extracted document in a case; return the combined chunks."""
    raise NotImplementedError(
        "spec: for each <doc_id>.md under "
        "DOSSIER_DIR/<case_id>/extracted/, call chunk_dossier_document(md_text, "
        "case_id, doc_id), concatenate the resulting DataFrames, then call "
        "append_to_dossier_chunks_csv(chunks, case_id) and "
        "append_to_chroma(chunks). Return the combined chunk DataFrame."
    )
