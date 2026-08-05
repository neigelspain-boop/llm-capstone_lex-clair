"""Load offline ingestion artifacts into a hybrid BM25 + dense retriever.

The retriever's row table is assembled at load time from two kinds of
source (ADR #58): the tracked statute CSV (data/chunks.csv) plus every
per-case dossier CSV under data/dossier/*/chunks.csv, which are gitignored
for private cases. Dossier text is deliberately never written into the
tracked statute CSV — see _read_dossier_chunks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import chromadb
from minsearch import Index
from FlagEmbedding import BGEM3FlagModel

# Shared ingestion constants and helpers imported from index.py.
from ingestion.index import (
    CHUNKS_CSV,
    CHROMA_DIR,
    COLLECTION,
    EMBED_MODEL_ID,
    build_bm25,
    infer_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant used for score aggregation.
RRF_K = 60

# BM25 over-fetch when a scope filter is active — see search()'s fetch_n.
SCOPED_BM25_FETCH = 200
DEFAULT_BM25_BOOST = {
    "texte": 1.0,
    "titre": 1.0,
    "section_path": 1.0,
    "num": 1.0,
}


# Per-case dossier chunk CSVs, merged into the row table at load time.
DOSSIER_DIR = CHUNKS_CSV.parent / "dossier"


# ========== dossier chunk merge (ADR #58) ==========

def _read_dossier_chunks(statute_columns: list[str], dossier_dir: Path) -> list[pd.DataFrame]:
    """Read every per-case data/dossier/<case_id>/chunks.csv, aligned onto
    the statute CSV's column set.

    load_index() needs a row for every chunk_id Chroma can return, or
    search() raises KeyError when hydrating a dossier hit (ADR #39). That
    used to be satisfied by appending dossier rows into the shared,
    git-tracked data/chunks.csv — which put real client names one `git add`
    away from being published, and the project's filename-based privacy grep
    would not have caught it. Merging here instead keeps the tracked CSV
    statute-only by construction: dossier text lives only in paths that are
    already gitignored for private cases, and no code path can move it out.

    Missing, empty, and header-less files are skipped rather than raised on:
    one zero-byte chunks.csv would otherwise abort load_index and take the
    whole app down with it, and a case with no indexed documents is a normal
    state, not a corpus error. Statute-only columns (etat, date_debut,
    date_fin, legiarti_id) fill as "" for dossier rows, never NaN, per
    project convention.
    """
    if not dossier_dir.exists():
        return []

    frames: list[pd.DataFrame] = []
    for case_csv in sorted(dossier_dir.glob("*/chunks.csv")):
        try:
            case_chunks = pd.read_csv(case_csv, keep_default_na=False)
        except pd.errors.EmptyDataError:
            log.warning("load_index: skipping empty dossier chunks file %s", case_csv)
            continue
        if case_chunks.empty or "chunk_id" not in case_chunks.columns:
            continue
        frames.append(case_chunks.reindex(columns=statute_columns, fill_value=""))
        log.info("load_index: merged %d dossier chunks from %s", len(case_chunks), case_csv)
    return frames


# ========== source_scope filtering (ADR #41) ==========

def _scope_predicate(source_scope: str):
    """Return a chunk_id -> bool predicate for the given source_scope.

    Raises ValueError for anything other than "statute", "dossier",
    "blended", or "case:<id>". Called first thing in search() so bad
    input fails before any embedding/query work.
    """
    if source_scope == "statute":
        return lambda cid: not cid.startswith("dossier-")
    if source_scope == "dossier":
        return lambda cid: cid.startswith("dossier-")
    if source_scope == "blended":
        return lambda cid: True
    if source_scope.startswith("case:"):
        case_id = source_scope.removeprefix("case:")
        if not case_id:
            raise ValueError(
                f"invalid source_scope: {source_scope!r} — case id must not be empty"
            )
        prefix = f"dossier-{case_id}-"
        return lambda cid: cid.startswith(prefix)
    raise ValueError(
        f"invalid source_scope: {source_scope!r} — must be 'statute', 'dossier', "
        f"'blended', or 'case:<id>'"
    )


# HybridRetriever: stateful BM25 + dense-vector retriever for query time.
@dataclass
class HybridRetriever:
    """Hybrid BM25 + dense-vector retriever for query time."""

    bm25: Index                                          # in-memory, rebuilt from chunks.csv
    vectors: chromadb.api.models.Collection.Collection   # persistent Chroma collection
    embed_model: BGEM3FlagModel                          # query-time embedder
    chunks: pd.DataFrame                                 # chunk metadata indexed by chunk_id
    device: str = field(default="cpu")                   # GPU/CPU runtime device
    bm25_boost_dict: dict[str, float] = field(default_factory=lambda: DEFAULT_BM25_BOOST.copy())

    def _dossier_sources(self) -> list[str]:
        """The `source` metadata values belonging to dossier chunks
        ("dossier-<case_id>"), derived from the row table so no separate
        registry can drift out of sync with what is actually indexed."""
        dossier_ids = [
            cid for cid in self.chunks.index if str(cid).startswith("dossier-")
        ]
        if not dossier_ids:
            return []
        return sorted(set(self.chunks.loc[dossier_ids, "source"].astype(str)))

    def _statute_sources(self) -> list[str]:
        """The `source` metadata values belonging to statute chunks, derived
        from the row table — the positive counterpart of _dossier_sources().

        Exists so the "statute" scope can be expressed as an allowlist rather
        than a denylist. See _chroma_scope_filter for why that distinction is
        load-bearing (ADR #63)."""
        statute_ids = [
            cid for cid in self.chunks.index if not str(cid).startswith("dossier-")
        ]
        if not statute_ids:
            return []
        return sorted(set(self.chunks.loc[statute_ids, "source"].astype(str)))

    def _chroma_scope_filter(self, source_scope: str) -> dict | None:
        """Translate source_scope into a Chroma `where` clause, or None for
        no server-side filter.

        This is what makes scope filtering exact on the dense side rather
        than a post-hoc cut of a fixed candidate pool: Chroma returns k*3
        documents that already match the scope, so a narrow scope can never
        starve (ADR #58, tightening ADR #41).

        Every scope is expressed as a positive allowlist so an unrecognised
        `source` is excluded rather than admitted (ADR #63). "statute" used
        to be the denylist {"$nin": dossier_sources}, which fails open twice
        over: both lists are derived from the row table, so a vector whose
        case has been removed or renamed is absent from dossier_sources, and
        the denylist therefore *admits* it. Those orphans then displace real
        statute hits inside the k*3 budget before the chunk_id predicate cuts
        them, silently starving the reranker's candidate pool — and a dossier
        source the row table has never heard of is exactly the one that must
        not leak into the default scope.

        Returns None only for "blended", which is unfiltered by definition.
        """
        if source_scope == "blended":
            return None
        if source_scope == "statute":
            statute_sources = self._statute_sources()
            # An empty allowlist would match everything on some backends;
            # fall through to the chunk_id predicate instead of guessing.
            return {"source": {"$in": statute_sources}} if statute_sources else None
        if source_scope == "dossier":
            dossier_sources = self._dossier_sources()
            return {"source": {"$in": dossier_sources}} if dossier_sources else None
        if source_scope.startswith("case:"):
            return {"source": f"dossier-{source_scope.removeprefix('case:')}"}
        return None

    def search(
        self,
        query: str,
        k: int = 10,
        boost_dict: dict[str, float] | None = None,
        mode: str = "hybrid",
        source_scope: str = "statute",
    ) -> list[dict]:
        """Return top-k chunks fused from BM25 and dense retrieval.

        source_scope (ADR #41, default "statute") filters the fused
        candidate set before the top-k cut: "statute" (default),
        "dossier", "case:<id>", or "blended" (no filter).
        """
        predicate = _scope_predicate(source_scope)

        if mode not in ("bm25", "vector", "hybrid"):
            raise ValueError("mode must be one of 'bm25', 'vector', or 'hybrid'")

        # Candidate budget. Both backends return their top-N over the WHOLE
        # corpus and the scope predicate is applied to the fused pool
        # afterwards, so once statute and dossier share one index a scope can
        # starve: a dossier-flavoured query fills every slot with dossier
        # chunks and source_scope="statute" (the default!) filters them all
        # out, returning nothing. Chroma is filtered server-side below, which
        # fixes the dense half exactly. BM25 has no equivalent negation
        # filter, so it over-fetches instead — the index is in-memory and a
        # few thousand rows, so a wider fetch is free (ADR #58).
        fetch_n = k * 3 if source_scope == "blended" else max(k * 3, SCOPED_BM25_FETCH)

        # encode the query with the same embedding flags used for index building
        qvec = self.embed_model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"][0]

        bm25_hits = []
        vec_ids: list[str] = []

        if mode in ("bm25", "hybrid"):
            bm25_hits = self.bm25.search(
                query=query,
                boost_dict=boost_dict or self.bm25_boost_dict,
                num_results=fetch_n,
            )

        if mode in ("vector", "hybrid"):
            query_kwargs: dict = {
                "query_embeddings": [qvec.tolist()],
                "n_results": k * 3,
            }
            where = self._chroma_scope_filter(source_scope)
            if where is not None:
                query_kwargs["where"] = where
            vec_result = self.vectors.query(**query_kwargs)
            vec_ids = vec_result["ids"][0]

        # reciprocal rank fusion across BM25 and dense results
        scores: dict[str, float] = {}
        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, cid in enumerate(vec_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        # source_scope filter: applied to the fused candidate pool, before
        # the top-k cut, so a narrow scope draws from the full k*3 over-fetch
        # rather than being cut down further after truncation (ADR #41).
        scores = {cid: s for cid, s in scores.items() if predicate(cid)}

        # Orphan guard (ADR #63): a chunk_id can be present in Chroma but
        # absent from the row table when a case's CSV is removed or renamed
        # without its vectors being purged. Hydrating one raises KeyError and
        # takes down the whole query, so drop them before the top-k cut —
        # dropping after it would silently shorten the result list instead.
        orphans = [cid for cid in scores if cid not in self.chunks.index]
        if orphans:
            log.warning(
                "search: dropping %d candidate(s) present in Chroma but absent from the "
                "row table (e.g. %s) — the index is out of sync; run "
                "`python -m ingestion.index --reconcile` to inspect",
                len(orphans), orphans[0],
            )
            scores = {cid: s for cid, s in scores.items() if cid not in orphans}

        top_ids = sorted(scores, key=scores.get, reverse=True)[:k]

        # hydrate results from the indexed chunk table
        results = []
        for cid in top_ids:
            row = self.chunks.loc[cid]
            results.append({
                "chunk_id": cid,
                "num": row["num"],
                "titre": row["titre"],
                "section_path": row["section_path"],
                "texte": row["texte"],
                "source": row["source"],
                "source_label": row["source_label"],
                "url": row["url"],
                "rrf_score": scores[cid],
            })
        return results


# load_index: public cold-boot entry point for query-time retrieval.

def load_index(
    src: Path = CHUNKS_CSV,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION,
    device: str | None = None,
    dossier_dir: Path | None = None,
) -> HybridRetriever:
    """Load persisted ingestion artifacts and return a query retriever.

    The row table and BM25 index are built from the statute CSV plus every
    per-case dossier CSV (ADR #58, see _read_dossier_chunks). Retrieval
    stays statute-only by default regardless: source_scope defaults to
    "statute" (ADR #41), so merged dossier rows are only reachable through
    an explicit dossier/blended/case:<id> scope.
    """
    device = infer_device(device)
    log.info("load_index: device=%s", device)

    # read chunks and index by chunk_id
    log.info("reading chunks from %s", src)
    statute_chunks = pd.read_csv(src, keep_default_na=False)
    frames = [statute_chunks, *_read_dossier_chunks(
        list(statute_chunks.columns), dossier_dir or DOSSIER_DIR
    )]
    chunks = (
        pd.concat(frames, ignore_index=True) if len(frames) > 1 else statute_chunks
    )
    log.info(
        "load_index: %d chunks total (%d statute, %d dossier)",
        len(chunks), len(statute_chunks), len(chunks) - len(statute_chunks),
    )
    chunks_indexed = chunks.set_index("chunk_id", drop=False)

    # rebuild BM25 from chunk rows
    bm25 = build_bm25(chunks)

    # open persisted Chroma collection
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Chroma directory not found at {chroma_dir}. Run `uv run python -m ingestion.index` first."
        )
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Chroma collection {collection_name!r} not found in {chroma_dir}. Run `uv run python -m ingestion.index` to rebuild it."
        ) from e
    log.info("opened Chroma collection %r with %d chunks", collection_name, collection.count())

    # Detect-only desync check (ADR #63). load_index stays read-only: a
    # mutating reconcile at load time would delete real state as a side
    # effect of opening the app. Reporting is enough — search() drops
    # orphans defensively, and `--reconcile --apply` is the deliberate fix.
    if collection.count() != len(chunks):
        log.warning(
            "load_index: Chroma holds %d vectors but the row table has %d chunks — "
            "the index is out of sync; run `python -m ingestion.index --reconcile`",
            collection.count(), len(chunks),
        )

    # load the embed model with matching runtime flags
    embed_model = BGEM3FlagModel(
        EMBED_MODEL_ID,
        use_fp16=(device == "cuda"),
        device=device,
    )
    log.info("loaded embedding model %r on %s", EMBED_MODEL_ID, device)

    return HybridRetriever(
        bm25=bm25,
        vectors=collection,
        embed_model=embed_model,
        chunks=chunks_indexed,
        device=device,
    )


if __name__ == "__main__":
    # smoke run through load_index()
    retriever = load_index()
    print("=" * 70)
    print("HybridRetriever smoke · query: quasi-usufruit et notaire")
    print("=" * 70)
    for hit in retriever.search("quasi-usufruit et notaire", k=5):
        url_ok = "YES" if hit["url"].startswith("http") else "NO"
        print(
            f"  {hit['chunk_id']:20s} rrf={hit['rrf_score']:.4f} "
            f"num={hit['num']:8s} url={url_ok}  {hit['texte'][:60]}"
        )