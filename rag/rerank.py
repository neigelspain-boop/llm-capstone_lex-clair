"""Cross-encoder reranking of retrieval candidates.

Public surface: `rerank(query, hits, k=5)` scores each (query, hit["texte"])
pair jointly with a cross-encoder and returns the top-k hits ordered by
rerank_score descending.

Why cross-encoder vs bi-encoder: BGE-M3 (used in retrieve) encodes query
and doc independently, then compares vectors. Fast, less accurate.
BGE-reranker-v2-m3 scores the (query, doc) pair jointly through a
transformer — slower per pair but more precise. Standard pattern:
bi-encoder for 20 candidates → cross-encoder for top 5.

Model: BAAI/bge-reranker-v2-m3, 2.27GB, same family as BGE-M3 embedder.
Cached at ~/.cache/huggingface/ from Day 4 pre-flight.

Loader: sentence-transformers.CrossEncoder rather than
FlagEmbedding.FlagReranker. Reason: FlagEmbedding 1.4.x has a known bug
where the reranker silently loads the slow XLMRobertaTokenizer, which
lacks `prepare_for_model` in modern transformers, causing AttributeError
at compute_score() time. Direct AutoTokenizer + AutoModel also hit the
same slow-tokenizer fallback in this environment. sentence-transformers
handles the loader path cleanly and produces identical scores (same
underlying model weights).

Cold-start cost: first `rerank()` call loads the model into VRAM (~5s
from local cache). Subsequent calls are forward passes only, ~50-200ms
for 20 pairs on RTX 3060.
"""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from ingestion.index import infer_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512


# Module-level singleton: lazy-loaded on first rerank() call.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Return the cached CrossEncoder, initializing on first call."""
    global _reranker
    if _reranker is None:
        device = infer_device(None)
        log.info("loading reranker %r on %s", RERANKER_MODEL_ID, device)
        _reranker = CrossEncoder(
            RERANKER_MODEL_ID,
            max_length=MAX_LENGTH,
            device=device,
        )
    return _reranker


def rerank(query: str, hits: list[dict], k: int = 5) -> list[dict]:
    """Rescore hits with a cross-encoder and return top-k.

    Each returned hit is the original dict plus a `rerank_score` field,
    sorted by score descending. Original ordering (rrf_score) is discarded.
    Raw logits are returned as score — no sigmoid activation, since we
    only sort, we don't threshold.
    """
    if not hits:
        return []

    pairs = [(query, h["texte"]) for h in hits]
    scores = get_reranker().predict(pairs, convert_to_numpy=True).tolist()

    scored = [{**h, "rerank_score": float(s)} for h, s in zip(hits, scores)]
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    # Standalone smoke: retrieve 20 → rerank to 5 → print top-5 with scores.
    from rag.retrieve import retrieve

    query = "quasi-usufruit et notaire"
    hits = retrieve(query, k=20)
    top = rerank(query, hits, k=5)

    print(f"\nreranked {len(hits)} → {len(top)} for query: {query!r}\n")
    for h in top:
        print(
            f"  {h['chunk_id']:20s} score={h['rerank_score']:+.4f}  "
            f"{h['texte'][:60]}"
        )