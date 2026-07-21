"""Thin query-time wrapper over the Plane I retriever.

Public surface: `retrieve(query, k=20)` returns the top-k chunks for a
query. Mode is hardcoded to "vector" per ADR #20 (2026-07-20):
vector_only beat bm25/hybrid/hybrid_tuned on synthetic GT
(Hit@10=0.7695 vs hybrid=0.7224). Post-launch retest committed once
Day 7 monitoring collects real query logs.

Cross-plane boundary (ADR #10, locked): this module only imports
`load_index` and `HybridRetriever` from `ingestion/`. Nothing else.

Cold-start cost: first `retrieve()` call triggers `load_index()` which
loads BGE-M3 (~10s) + opens Chroma (~1s) + rebuilds BM25 (~2s). Every
subsequent call is fast because the module-level singleton is set.
"""
from __future__ import annotations

from ingestion import load_index, HybridRetriever


# Module-level singleton: lazy-loaded on first retrieve() call.
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return the cached retriever, initializing on first call."""
    global _retriever
    if _retriever is None:
        _retriever = load_index()
    return _retriever


def retrieve(query: str, k: int = 20) -> list[dict]:
    """Retrieve top-k chunks for a query using vector-only search.

    Returns raw hits from HybridRetriever.search — full field set including
    chunk_id, num, titre, section_path, texte, source, source_label, url,
    rrf_score. Callers (rerank, flow) do their own field selection.
    """
    return get_retriever().search(query, k=k, mode="vector")


if __name__ == "__main__":
    # Standalone smoke: verifies cold-boot + one search returns k=20 hits.
    hits = retrieve("quasi-usufruit et notaire")
    print(f"retrieved {len(hits)} hits")
    for h in hits[:3]:
        print(f"  {h['chunk_id']:20s} rrf={h['rrf_score']:.4f}  {h['texte'][:60]}")