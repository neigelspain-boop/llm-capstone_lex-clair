"""End-to-end RAG orchestrator: query → grounded French answer with citations.

Public surface: `run(query, verbose=False)` — the sole entry point Day 5
LLM eval, Day 6 Streamlit UI, and Day 7 monitoring will consume.

Pipeline stages:
  1. rewrite the query with legal-register vocabulary (retrieval-side only)
  2. retrieve k=20 candidates via vector search (ADR #20)
  3. rerank to top-k=5 with a cross-encoder (ADR #21)
  4. build a French answer prompt with grounding + citation discipline
  5. generate the answer with gpt-4o-mini

Architectural discipline (§2.5 of Day 4 plan):
  - The REWRITTEN query is used for retrieval + rerank (matches the corpus's
    legal register).
  - The ORIGINAL query is inserted into the answer prompt (the user hears
    an answer to what they actually asked).

Return shape locked here for downstream consumers:
  {
    "answer": str,               # French, ~3-part structure
    "citations": list[dict],     # [{chunk_id, num, url}, ...]
    "rewritten_query": str,      # for debugging / eval
    "chunks_retrieved": int,     # always 20 in V1
    "chunks_reranked": int,      # always 5 in V1
    "model_used": str,           # "gpt-4o-mini" in V1
    "cost_usd": float,           # includes rewrite + generate calls
    "elapsed_seconds": float,    # end-to-end wall time
  }
"""
from __future__ import annotations

import logging
from time import time

from rag import generate, prompt, rerank, retrieve, rewrite


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# gpt-4o-mini pricing (source: openai.com/pricing, Day 3 ADR #19).
# Kept at flow layer, not generate layer, so pricing updates touch one file.
GPT_4O_MINI_INPUT_PER_M = 0.15
GPT_4O_MINI_OUTPUT_PER_M = 0.60

RETRIEVE_K = 20
RERANK_K = 5


def _compute_cost(tokens: dict, model: str = "gpt-4o-mini") -> float:
    """Compute USD cost from token counts. V1: gpt-4o-mini only."""
    if model != "gpt-4o-mini":
        return 0.0  # unknown pricing → fail-quiet, not fail-loud
    input_cost = tokens["prompt_tokens"] * GPT_4O_MINI_INPUT_PER_M / 1_000_000
    output_cost = tokens["completion_tokens"] * GPT_4O_MINI_OUTPUT_PER_M / 1_000_000
    return input_cost + output_cost


def run(query: str, verbose: bool = False) -> dict:
    """Run the full RAG flow. Returns the locked dict spec (see module docstring)."""
    t0 = time()

    # 1. rewrite (retrieval-side only, silent-fallback on failure)
    rewritten = rewrite.rewrite(query)
    if verbose:
        log.info("rewritten: %s", rewritten)

    # 2. retrieve k=20 via vector search
    candidates = retrieve.retrieve(rewritten, k=RETRIEVE_K)
    if verbose:
        log.info("retrieved %d candidates", len(candidates))

    # 3. rerank to top-5 with cross-encoder
    top_chunks = rerank.rerank(rewritten, candidates, k=RERANK_K)
    if verbose:
        log.info("reranked to %d chunks (top: %s)",
                 len(top_chunks), top_chunks[0]["chunk_id"] if top_chunks else "none")

    # 4. build prompt using ORIGINAL query, not rewritten
    p = prompt.build(query, top_chunks)
    if verbose:
        log.info("prompt built: %d chars", len(p))

    # 5. generate
    answer, tokens = generate.generate(p)

    took = time() - t0

    return {
        "answer": answer,
        "citations": [
            {"chunk_id": c["chunk_id"], "num": c["num"], "url": c["url"]}
            for c in top_chunks
        ],
        "rewritten_query": rewritten,
        "chunks_retrieved": len(candidates),
        "chunks_reranked": len(top_chunks),
        "model_used": generate.MODEL,
        "cost_usd": _compute_cost(tokens, generate.MODEL),
        "elapsed_seconds": took,
    }


if __name__ == "__main__":
    # Standalone smoke: end-to-end on the Day 4 deliverable query.
    result = run(
        "Ma grand-mère a vendu la maison en usufruit, que puis-je faire ?",
        verbose=True,
    )

    print(f"\n{'=' * 70}")
    print(f"answer ({len(result['answer'])} chars):")
    print(f"{'=' * 70}")
    print(result["answer"])
    print(f"\n{'=' * 70}")
    print(f"citations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  {c['chunk_id']:20s} art. {c['num']:8s} {c['url']}")
    print(f"\n{'=' * 70}")
    print(f"cost: ${result['cost_usd']:.5f}")
    print(f"elapsed: {result['elapsed_seconds']:.2f}s")
    print(f"model: {result['model_used']}")
    print(f"rewritten: {result['rewritten_query']}")