"""Offline index builder: BM25 + Chroma from chunks.csv."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from minsearch import Index
import chromadb
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# module constants for repo paths, Chroma collection config, and BM25 fields
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_CSV = ROOT / "data" / "chunks.csv"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "lex_clair_articles"
EMBED_MODEL_ID = "BAAI/bge-m3"
BATCH_SIZE = 32

TEXT_FIELDS = ["num", "titre", "section_path", "texte"]
KEYWORD_FIELDS = ["chunk_id", "source", "source_label"]
REQUIRED_COLUMNS = KEYWORD_FIELDS + TEXT_FIELDS + ["url"]


def _validate_chunk_schema(chunks: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS).difference(chunks.columns)
    if missing:
        raise ValueError(
            "chunks DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    null_columns = [col for col in REQUIRED_COLUMNS if chunks[col].isnull().any()]
    if null_columns:
        counts = ", ".join(
            f"{col}={int(chunks[col].isnull().sum())}" for col in null_columns
        )
        raise ValueError(
            "chunks DataFrame contains nulls in required columns: " + counts
        )

    duplicates = int(chunks["chunk_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"chunk_id contains {duplicates} duplicate values")


def _prepare_bm25_documents(chunks: pd.DataFrame) -> list[dict[str, str]]:
    return chunks.astype(str, copy=False).to_dict(orient="records")


# BM25 builder: preserve full row metadata while indexing the canonical fields

def build_bm25(chunks: pd.DataFrame) -> Index:
    """Build an in-memory BM25 index from chunk DataFrame rows.

    The function is intentionally pure: it validates the incoming DataFrame
    and returns a new Index instance without mutating the caller's data.
    """
    _validate_chunk_schema(chunks)
    documents = _prepare_bm25_documents(chunks)

    index = Index(
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS,
    )
    index.fit(documents)

    log.info("built BM25 index for %d chunks", len(chunks))
    return index


# build_chroma: create or replace a persistent Chroma collection from chunks
# and embed chunk texts with BGE-M3 for downstream vector retrieval.
def build_chroma(chunks: pd.DataFrame, device: str) -> chromadb.api.models.Collection.Collection:
    """Build a persistent Chroma collection from chunk CSV rows."""
    _validate_chunk_schema(chunks)

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    model = BGEM3FlagModel(
        EMBED_MODEL_ID,
        use_fp16=(device == "cuda"),
        device=device,
    )

    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    texts = chunks["texte"].astype(str).tolist()
    metadatas = (
        chunks[
            ["source", "source_label", "num", "titre", "section_path", "url"]
        ]
        .fillna("")
        .astype(str, copy=False)
        .to_dict(orient="records")
    )

    batch_count = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), total=batch_count, desc="indexing chunks"):
        end = start + BATCH_SIZE
        batch_ids = chunk_ids[start:end]
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]

        output = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vecs = [
            [float(value) for value in vector]
            for vector in output["dense_vecs"]
        ]

        collection.add(
            ids=batch_ids,
            embeddings=dense_vecs,
            documents=batch_texts,
            metadatas=batch_metadatas,
        )

    log.info("built Chroma collection %r with %d chunks", COLLECTION, len(chunks))
    return collection


# infer_device: shared device selection logic for build and load workflows

def infer_device(device: str | None = None) -> str:
    if device:
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        log.warning("torch is not installed; defaulting to cpu for embeddings")
        return "cpu"


def build_all(
    src: Path = CHUNKS_CSV,
    device: str | None = None,
) -> tuple[Index, chromadb.api.models.Collection.Collection]:
    """Build BM25 and Chroma from the canonical chunk CSV."""
    device = infer_device(device)
    log.info("reading chunks from %s", src)
    chunks = pd.read_csv(src, keep_default_na=False)

    bm25 = build_bm25(chunks)
    collection = build_chroma(chunks, device)
    log.info("finished index build with device=%s", device)
    return bm25, collection


# ========== Chroma/row-table reconciliation (ADR #63) ==========

def reconcile_chroma(
    chunk_ids: set[str],
    collection: chromadb.api.models.Collection.Collection,
    apply: bool = False,
) -> dict:
    """Diff the persisted Chroma collection against the row table.

    Two failure modes, only one of them fixable here:

    - ORPHANS — vectors whose chunk_id is absent from the row table. Left
      behind whenever a case's CSV is removed or its case_id renamed, because
      ingestion.dossier.index only purges the `dossier-{case_id}-` prefix it
      is currently writing (ADR #38) and cannot know about a prefix that no
      longer exists. These are deletable, and this is what `apply=True` does.
    - UNVECTORISED — row-table chunks with no vector. Reported only: fixing
      them means re-embedding, which is build_chroma's job, not a diff's.

    Chroma is real state, so deletion is opt-in: the default is a dry run
    that reports and changes nothing (same idiom as rag.compliance's
    --dry-run). Returns the counts either way.
    """
    persisted = collection.get(include=["metadatas"])
    persisted_ids = persisted["ids"]
    metadatas = persisted["metadatas"] or [{} for _ in persisted_ids]

    orphans = [cid for cid in persisted_ids if cid not in chunk_ids]
    unvectorised = sorted(chunk_ids.difference(persisted_ids))

    by_source: dict[str, int] = {}
    orphan_set = set(orphans)
    for cid, meta in zip(persisted_ids, metadatas):
        if cid in orphan_set:
            source = str((meta or {}).get("source", "<no source metadata>"))
            by_source[source] = by_source.get(source, 0) + 1

    log.info(
        "reconcile: %d vectors in Chroma, %d chunks in row table",
        len(persisted_ids), len(chunk_ids),
    )
    if orphans:
        log.warning("reconcile: %d orphan vector(s) with no row-table entry:", len(orphans))
        for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
            log.warning("  source=%-24s %d vector(s)", source, count)
    if unvectorised:
        log.warning(
            "reconcile: %d row-table chunk(s) with no vector (e.g. %s) — dense retrieval "
            "cannot reach them; rebuild the index to embed them",
            len(unvectorised), unvectorised[0],
        )
    if not orphans and not unvectorised:
        log.info("reconcile: Chroma and the row table agree; nothing to do")

    deleted = 0
    if orphans and apply:
        collection.delete(ids=orphans)
        deleted = len(orphans)
        log.info("reconcile: deleted %d orphan vector(s)", deleted)
    elif orphans:
        log.info("reconcile: dry run — re-run with --apply to delete these %d vector(s)", len(orphans))

    return {
        "persisted": len(persisted_ids),
        "row_table": len(chunk_ids),
        "orphans": len(orphans),
        "orphans_by_source": by_source,
        "unvectorised": len(unvectorised),
        "deleted": deleted,
    }


def _reconcile_cli(apply: bool) -> dict:
    """Assemble the merged row table and reconcile the persisted collection.

    Imports ingestion.load lazily: load.py imports build_bm25/infer_device
    from this module, so a module-level import here would be circular. The
    row table must come from load_index's merge (statute CSV + every per-case
    dossier CSV, ADR #58) — reconciling against the statute CSV alone would
    condemn every legitimate dossier vector as an orphan.
    """
    from ingestion.load import DOSSIER_DIR, _read_dossier_chunks

    statute_chunks = pd.read_csv(CHUNKS_CSV, keep_default_na=False)
    frames = [statute_chunks, *_read_dossier_chunks(
        list(statute_chunks.columns), DOSSIER_DIR
    )]
    chunks = pd.concat(frames, ignore_index=True) if len(frames) > 1 else statute_chunks
    chunk_ids = set(chunks["chunk_id"].astype(str))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION)
    return reconcile_chroma(chunk_ids, collection, apply=apply)


def main() -> None:
    """Command-line entrypoint for BM25 + Chroma index building."""
    parser = argparse.ArgumentParser(description="Build BM25 + Chroma from chunks CSV.")
    parser.add_argument("--src", type=Path, default=CHUNKS_CSV, help="source chunks CSV")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="embedding device; inferred from torch.cuda.is_available() if omitted",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="diff Chroma against the row table instead of building; reports only",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="with --reconcile, delete orphan vectors (destructive; default is a dry run)",
    )
    args = parser.parse_args()

    if args.apply and not args.reconcile:
        parser.error("--apply is only meaningful with --reconcile")
    if args.reconcile:
        _reconcile_cli(apply=args.apply)
        return

    build_all(src=args.src, device=args.device)


# Chroma builder: embed chunks with BGE-M3 and persist the collection to disk
# Orchestration: build both BM25 and Chroma from chunks.csv, defaulting device via torch

if __name__ == "__main__":
    main()