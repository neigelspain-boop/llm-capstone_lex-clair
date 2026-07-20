"""Load offline ingestion artifacts into a hybrid BM25 + dense retriever."""
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
DEFAULT_BM25_BOOST = {
    "texte": 1.0,
    "titre": 1.0,
    "section_path": 1.0,
    "num": 1.0,
}


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

    def search(
        self,
        query: str,
        k: int = 10,
        boost_dict: dict[str, float] | None = None,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Return top-k chunks fused from BM25 and dense retrieval."""

        if mode not in ("bm25", "vector", "hybrid"):
            raise ValueError("mode must be one of 'bm25', 'vector', or 'hybrid'")

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
                num_results=k * 3,
            )

        if mode in ("vector", "hybrid"):
            vec_result = self.vectors.query(
                query_embeddings=[qvec.tolist()],
                n_results=k * 3,
            )
            vec_ids = vec_result["ids"][0]

        # reciprocal rank fusion across BM25 and dense results
        scores: dict[str, float] = {}
        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, cid in enumerate(vec_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

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
) -> HybridRetriever:
    """Load persisted ingestion artifacts and return a query retriever."""
    device = infer_device(device)
    log.info("load_index: device=%s", device)

    # read chunks and index by chunk_id
    log.info("reading chunks from %s", src)
    chunks = pd.read_csv(src, keep_default_na=False)
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