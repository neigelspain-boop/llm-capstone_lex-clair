"""Dossier transcripts → incremental additions to the hybrid retrieval store
(Plane I · offline).

Chunks each case's extracted .md transcripts and appends them to the
existing Chroma collection built by ingestion/index.py — it does NOT
rebuild the corpus from scratch. This is the key contrast with
ingestion.index.build_chroma, which shutil.rmtree()s and recreates the
Chroma directory; that path would destroy the statute corpus embeddings
every time a new case dossier is indexed.

BM25 has no append mechanism (ingestion.index.build_bm25 is a pure
in-memory rebuild), but ingestion.load.load_index() rebuilds it fresh on
every cold boot. HybridRetriever.chunks (ingestion/load.py) does a row
lookup for every hit regardless of retrieval mode, so a dossier chunk_id
with no row there raises KeyError (ADR #39, correcting ADR #38).

ADR #58 changed where that row comes from. This module previously wrote
dossier rows into the shared, git-tracked data/chunks.csv; load_index now
merges every per-case data/dossier/*/chunks.csv at load time instead. The
tracked statute CSV therefore stays statute-only by construction, and
private client names cannot reach a tracked file at all — the earlier
design left them one `git add` away from publication, past a privacy gate
that greps filenames rather than content.

Inputs:  data/dossier/<case_id>/extracted/<doc_id>.md
         data/dossier/<case_id>/facts.jsonl
Outputs: - data/dossier/<case_id>/chunks.csv — this case's dossier chunk
            rows; both the audit surface and what load_index merges
         - the existing Chroma collection (ingestion.index.COLLECTION) —
            appended to via collection.add(), not recreated
         - data/dossier/<case_id>/facts.jsonl — rewritten with each Fact's
            source_chunk_id backfilled
         chunk_id format: dossier-<case_id>-<doc_id>-c<chunk_num:03d>

CLI: python -m ingestion.dossier.index --case-id <id>

Every `pd.read_csv` call added to this module's implementation MUST pass
`keep_default_na=False` (project-wide convention — see ingestion/chunk.py,
ingestion/index.py; guards against the pandas NaN silent-failure trap).
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import chromadb
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel

from ingestion.dossier.facts import Fact
from ingestion.index import (
    CHROMA_DIR,
    CHUNKS_CSV,
    COLLECTION,
    EMBED_MODEL_ID,
    _validate_chunk_schema,
    infer_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

CHUNK_COLUMNS = [
    "chunk_id", "source", "source_label", "num", "titre", "section_path", "texte", "url",
]
CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
BATCH_SIZE = 32

_PAGE_HEADER_RE = re.compile(r"(?m)^## Page (\d+)\s*$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý])")


# ========== chunking ==========

def _split_into_pages(md_text: str) -> list[tuple[int, str]]:
    """Split on '## Page N' H2 headers into [(page_num, page_text), ...] in
    document order. Falls back to treating the whole document as Page 1
    (with a warning) if no header is found — defensive only, extract.py
    always emits these headers.
    """
    matches = list(_PAGE_HEADER_RE.finditer(md_text))
    if not matches:
        log.warning("no '## Page N' headers found — treating whole document as Page 1")
        stripped = md_text.strip()
        return [(1, stripped)] if stripped else []

    pages: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        page_text = md_text[start:end].strip()
        if page_text:
            pages.append((page_num, page_text))
    return pages


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_END_RE.split(text) if p.strip()]


def _char_split(text: str, size: int) -> list[str]:
    """Last-resort character split. Backs off to the nearest preceding
    whitespace so no cut lands mid-word; if a run has no whitespace within
    the window, extends forward to the next whitespace instead of forcing a
    mid-word cut — "never mid-word" is a hard invariant, occasionally
    letting a piece exceed `size` for pathological unbroken text.
    """
    pieces, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
            else:
                next_ws = text.find(" ", end)
                end = next_ws if next_ws != -1 else n
        pieces.append(text[start:end].strip())
        start = end
    return [p for p in pieces if p]


def _split_page_text(text: str, target_size: int, overlap: int) -> list[str]:
    """Recursive splitter: paragraph boundaries first, then sentences, then
    raw character split as a last resort. Consecutive chunks carry a
    word-boundary-snapped overlap from the tail of the previous chunk.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_size:
        return [text]

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= target_size:
            units.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= target_size:
                units.append(sent)
            else:
                units.extend(_char_split(sent, target_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if not current or len(candidate) <= target_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:]
        ws = tail.find(" ")
        tail = tail[ws + 1:] if ws != -1 else ""
        current = f"{tail}\n\n{unit}" if tail else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_dossier_document(md_path: Path, doc_id: str, case_id: str) -> pd.DataFrame:
    """Split one extraction transcript into dossier chunk rows.

    Splits hard on '## Page N' boundaries first (no chunk ever spans two
    pages), then recursively within each page (paragraph -> sentence ->
    character, ~800 chars with ~100-char overlap). chunk_id is
    deterministic and 1-indexed in document reading order.
    """
    md_text = md_path.read_text(encoding="utf-8")
    pages = _split_into_pages(md_text)

    try:
        url = str(md_path.resolve().relative_to(ROOT))
    except ValueError:
        url = str(md_path)

    rows = []
    chunk_num = 1
    for page_num, page_text in pages:
        for chunk_text in _split_page_text(page_text, CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS):
            rows.append({
                "chunk_id": f"dossier-{case_id}-{doc_id}-c{chunk_num:03d}",
                "source": f"dossier-{case_id}",
                "source_label": f"Dossier {case_id}",
                "num": doc_id,
                "titre": "",
                "section_path": f"Page {page_num}",
                "texte": chunk_text,
                "url": url,
            })
            chunk_num += 1

    return pd.DataFrame(rows, columns=CHUNK_COLUMNS)


# ========== index append (no rebuild) ==========

def append_to_chroma(chunks: pd.DataFrame, case_id: str, device: str | None = None) -> None:
    """Embed chunk texts with BGE-M3 and append them to the existing Chroma
    collection.

    Deletes any existing dossier-<case_id>- prefixed rows first (Chroma has
    no server-side prefix filter, so ids are listed and filtered
    client-side), then re-adds the fresh batch — this keeps repeated runs
    idempotent without corrupting the shared collection. Never calls
    ingestion.index.build_chroma/build_all, never touches statute rows.
    """
    device = infer_device(device)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION)
    except Exception as e:
        raise RuntimeError(
            f"Chroma collection {COLLECTION!r} not found in {CHROMA_DIR}. "
            "Run `uv run python -m ingestion.index` to build the statute corpus first."
        ) from e

    prefix = f"dossier-{case_id}-"
    existing_ids = collection.get(include=[])["ids"]
    stale_ids = [cid for cid in existing_ids if cid.startswith(prefix)]
    if stale_ids:
        log.info("removing %d stale Chroma rows for case_id=%s before re-add", len(stale_ids), case_id)
        collection.delete(ids=stale_ids)

    if chunks.empty:
        log.info("no dossier chunks to add to Chroma for case_id=%s", case_id)
        return

    model = BGEM3FlagModel(EMBED_MODEL_ID, use_fp16=(device == "cuda"), device=device)
    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    texts = chunks["texte"].astype(str).tolist()
    metadatas = (
        chunks[["source", "source_label", "num", "titre", "section_path", "url"]]
        .fillna("").astype(str, copy=False).to_dict(orient="records")
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        output = model.encode(
            texts[start:end], batch_size=BATCH_SIZE,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        dense_vecs = [[float(v) for v in vec] for vec in output["dense_vecs"]]
        collection.add(
            ids=chunk_ids[start:end], embeddings=dense_vecs,
            documents=texts[start:end], metadatas=metadatas[start:end],
        )

    log.info(
        "appended %d dossier chunks to Chroma collection %r for case_id=%s",
        len(chunks), COLLECTION, case_id,
    )


def append_to_dossier_chunks_csv(chunks: pd.DataFrame, case_id: str) -> Path:
    """Write this case's dossier chunk rows to DOSSIER_DIR/<case_id>/chunks.csv.

    Full overwrite, not a literal line-append — chunking is deterministic
    so the file's content is idempotent across re-runs, and it only ever
    holds this case's own rows.

    This is both the audit surface and the retriever's source of dossier
    rows: ingestion.load.load_index merges every per-case file at load time
    (ADR #58). Dossier text is deliberately never written into the shared,
    git-tracked data/chunks.csv — the earlier append_to_statute_chunks_csv
    did exactly that and put private client names one `git add` away from
    publication.
    """
    path = DOSSIER_DIR / case_id / "chunks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(path, index=False)
    log.info("wrote %d dossier chunk rows to %s", len(chunks), path)
    return path


# ========== facts backfill ==========

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _match_quote_to_chunk(quote: str, candidates: list[tuple[str, str]]) -> str | None:
    """Find the chunk_id whose text contains `quote`.

    Exact substring match first; falls back to a whitespace-collapsed,
    lowercased comparison. `candidates` must already be sorted by chunk_id
    so the first match wins deterministically when several chunks match.
    """
    for chunk_id, text in candidates:
        if quote in text:
            return chunk_id

    norm_quote = _normalize_ws(quote).lower()
    for chunk_id, text in candidates:
        if norm_quote in _normalize_ws(text).lower():
            return chunk_id
    return None


def _backfill_fact_chunk_ids(case_id: str, chunks: pd.DataFrame) -> tuple[int, int]:
    """Backfill source_chunk_id on every Fact in this case's facts.jsonl.

    Every fact's match is recomputed unconditionally on each run (never
    special-cased on "already set") so repeated index_dossier calls stay
    idempotent. Returns (facts_backfilled, facts_unmatched).
    """
    facts_path = DOSSIER_DIR / case_id / "facts.jsonl"
    if not facts_path.exists():
        log.info("no facts.jsonl for case_id=%s — skipping backfill", case_id)
        return 0, 0

    all_facts = [
        Fact.model_validate_json(line)
        for line in facts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    chunks_by_doc: dict[str, list[tuple[str, str]]] = {}
    if not chunks.empty:
        for doc_id, group in chunks.groupby("num", sort=False):
            rows = group.sort_values("chunk_id")
            chunks_by_doc[doc_id] = list(zip(rows["chunk_id"], rows["texte"]))

    matched = unmatched = 0
    updated: list[Fact] = []
    for fact in all_facts:
        chunk_id = _match_quote_to_chunk(
            fact.verbatim_quote, chunks_by_doc.get(fact.source_doc_id, [])
        )
        if chunk_id is not None:
            matched += 1
        else:
            unmatched += 1
            log.info(
                "facts backfill: no chunk match for fact_id=%s (doc_id=%s)",
                fact.fact_id, fact.source_doc_id,
            )
        updated.append(fact.model_copy(update={"source_chunk_id": chunk_id}))

    updated.sort(key=lambda f: (f.source_doc_id, f.fact_id))
    with facts_path.open("w", encoding="utf-8") as f:
        for fact in updated:
            f.write(fact.model_dump_json() + "\n")

    return matched, unmatched


# ========== result schema ==========

class DossierIndexResult(BaseModel):
    """Summary of one index_dossier(case_id) run."""

    case_id: str
    docs_indexed: int
    chunks_created: int
    facts_backfilled: int
    facts_unmatched: int
    elapsed: float


# ========== case orchestration ==========

def index_dossier(case_id: str) -> DossierIndexResult:
    """Chunk every extracted document in a case, index it, and backfill facts.

    Reads every .md under DOSSIER_DIR/<case_id>/extracted/, chunks each via
    chunk_dossier_document, aggregates, writes the per-case chunks.csv,
    syncs the shared data/chunks.csv, appends to the shared Chroma
    collection (idempotently), then backfills source_chunk_id on every
    Fact in the case's facts.jsonl.
    """
    t0 = time.time()

    extracted_dir = DOSSIER_DIR / case_id / "extracted"
    md_paths = sorted(extracted_dir.rglob("*.md"))

    frames = [
        chunk_dossier_document(p, doc_id=p.stem, case_id=case_id) for p in md_paths
    ]
    chunks = (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=CHUNK_COLUMNS)
    )
    if not chunks.empty:
        _validate_chunk_schema(chunks)

    append_to_dossier_chunks_csv(chunks, case_id)
    append_to_chroma(chunks, case_id)
    facts_backfilled, facts_unmatched = _backfill_fact_chunk_ids(case_id, chunks)

    return DossierIndexResult(
        case_id=case_id,
        docs_indexed=len(md_paths),
        chunks_created=len(chunks),
        facts_backfilled=facts_backfilled,
        facts_unmatched=facts_unmatched,
        elapsed=time.time() - t0,
    )


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.index --case-id <id>

    Re-running is safe: chunking is deterministic, the per-case CSV is a full
    overwrite, append_to_chroma deletes this case's stale rows before re-adding,
    and the facts backfill is idempotent.
    """
    parser = argparse.ArgumentParser(
        description="Chunk one case's extracted documents, append them to the "
        "shared Chroma collection, and backfill source_chunk_id on its facts. "
        "Never writes to the git-tracked data/chunks.csv (ADR #58)."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    args = parser.parse_args()

    result = index_dossier(args.case_id)
    print(
        f"dossier index · case_id={result.case_id} docs={result.docs_indexed} "
        f"chunks={result.chunks_created} facts_backfilled={result.facts_backfilled} "
        f"facts_unmatched={result.facts_unmatched} elapsed={result.elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
