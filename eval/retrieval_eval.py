"""Retrieval evaluation harness (rubric: needs >=2 approaches compared,
best one wins). Metrics: hit rate and MRR against a labelled eval set of
(question, expected_doc_id) pairs.

Run with:
    uv run python -m eval.retrieval_eval
"""
from typing import Callable

from src.rag.retriever import get_vector_store

# TODO: replace with a real labelled set: [{"question": ..., "expected_id": ...}, ...]
EVAL_SET: list[dict] = []


def hit_rate_and_mrr(retriever: Callable[[str], list[dict]], eval_set: list[dict]) -> dict:
    hits = 0
    reciprocal_ranks = []
    for item in eval_set:
        results = retriever(item["question"])
        ids = [r["id"] for r in results]
        if item["expected_id"] in ids:
            hits += 1
            rank = ids.index(item["expected_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    n = len(eval_set) or 1
    return {"hit_rate": hits / n, "mrr": sum(reciprocal_ranks) / n}


def approach_vector_only(question: str) -> list[dict]:
    return get_vector_store().search(question, k=5)


def approach_hybrid(question: str) -> list[dict]:
    # TODO: combine with keyword/BM25 search for a second approach to compare
    raise NotImplementedError


APPROACHES = {
    "vector_only": approach_vector_only,
    "hybrid": approach_hybrid,
}


def main():
    for name, retriever in APPROACHES.items():
        metrics = hit_rate_and_mrr(retriever, EVAL_SET)
        print(name, metrics)


if __name__ == "__main__":
    main()
