"""Thin query-time wrapper over the Plane I retriever.

Public surface: `retrieve(query, k=20)` returns the top-k chunks for a
query. Mode is "hybrid" with a damped lexical arm per ADR #64
(2026-08-05), superseding ADR #20's vector-only lock-in.

ADR #20 measured hit@10/mrr@10 on the raw retriever at k=10 and found
vector_only (0.7695) beat hybrid (0.7224). Both halves of that setup
were wrong for this pipeline: rag/flow.py retrieves k=20 and reranks
with a cross-encoder that discards rrf ordering entirely, so what the
retriever owes the pipeline is coverage of the top-20 pool, not
ordering — and the fusion it tested weighted a much weaker lexical arm
equally with a much stronger dense one. Re-measured at the pipeline's
own shape over the full 1584-query ground truth, hybrid at
weights=(0.3, 1.0) wins both stages:

    config          hit@20   hit@5 (reranked)
    vector_only     0.8333   0.7475
    hybrid (0.3,1)  0.8491   0.7595

Paired McNemar on the same queries: pool +43/-18 (p=0.0019), reranked
+33/-14 (p=0.0079). Significant at both stages, though the end-to-end
effect is modest (+1.2pp).

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


# RRF arm weights (bm25, dense) — ADR #64. Uniform (1.0, 1.0) is what
# ADR #20 measured and rejected: BM25 scores hit@20 well below dense here,
# so at equal weight its rank-0 hit (1/60) outranks a *correct* dense
# rank-5 hit (1/65) and drags the fusion below vector-only. Damping the
# lexical arm keeps its coverage contribution without letting it override.
DEFAULT_RRF_WEIGHTS = (0.3, 1.0)


def retrieve(query: str, k: int = 20, source_scope: str = "statute") -> list[dict]:
    """Retrieve top-k chunks for a query using weighted hybrid search.

    source_scope defaults to "statute" (ADR #41): prevents dossier case
    chunks from leaking into the general-purpose baseline flow. Passed
    straight through to HybridRetriever.search().

    Returns raw hits from HybridRetriever.search — full field set including
    chunk_id, num, titre, section_path, texte, source, source_label, url,
    rrf_score. Callers (rerank, flow) do their own field selection.
    """
    return get_retriever().search(
        query, k=k, mode="hybrid", source_scope=source_scope,
        weights=DEFAULT_RRF_WEIGHTS,
    )


if __name__ == "__main__":
    # Standalone smoke: verifies cold-boot + one search returns k=20 hits.
    hits = retrieve("quasi-usufruit et notaire")
    print(f"retrieved {len(hits)} hits")
    for h in hits[:3]:
        print(f"  {h['chunk_id']:20s} rrf={h['rrf_score']:.4f}  {h['texte'][:60]}")