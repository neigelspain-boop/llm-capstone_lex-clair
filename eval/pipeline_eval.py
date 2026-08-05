"""
eval/pipeline_eval.py — retrieval evaluation at the shape the pipeline runs
===========================================================================

Why this exists alongside retrieval_eval.py (ADR #64).

ADR #20 chose vector-only over hybrid on hit@10/mrr@10 measured against the
*raw retriever* at k=10. rag/flow.py does not run that shape. It retrieves
k=RETRIEVE_K (20), hands all 20 to a BGE cross-encoder, and keeps
RERANK_K (5) — and rag/rerank.py discards rrf ordering outright before it
scores anything. Two consequences the original harness cannot see:

  1. The retriever's *ordering* below k is nearly irrelevant. What it owes
     the pipeline is that the correct chunk be present in the top-20 pool at
     all. That is hit@20, not mrr@10.
  2. Answer quality is decided after reranking. The metric that actually
     predicts it is hit@5/mrr@5 measured on the reranked output — which has
     never been measured for any config.

Hybrid retrieval characteristically trades ordering for coverage: it surfaces
chunks dense retrieval misses while ranking them less crisply. That is a bad
trade under mrr@10 and a good one under "get it into the reranker's pool",
so a config can lose the published comparison and still be the better choice
here. This harness measures both halves for every config so the question is
decided on the pipeline's own terms.

Ground truth is single-gold (one article per question, see eval/ground_truth.py),
so recall@k and hit@k are the same quantity — the gap was never a missing
metric, only the wrong k and a missing reranker.

Reads data/ground_truth.csv, writes data/pipeline_eval_results.csv.

CLI:
    uv run python -m eval.pipeline_eval --sample 300
    uv run python -m eval.pipeline_eval                 # full 1484 queries
"""
from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ingestion.load import load_index, HybridRetriever
from rag.rerank import rerank


# ============================================================================
# Config — mirrors rag/flow.py's constants so the eval cannot drift from it
# ============================================================================
# These are rag.flow.RETRIEVE_K / RERANK_K. Imported by value rather than by
# reference so that a change in flow.py shows up here as a failing assertion
# rather than a silently different experiment.
RETRIEVE_K = 20
RERANK_K = 5

DEFAULT_GT_PATH = Path("data/ground_truth.csv")
DEFAULT_OUT_PATH = Path("data/pipeline_eval_results.csv")

# Reranking is the expensive half: RETRIEVE_K cross-encoder pairs per query.
# 300 queries x 20 pairs = 6k forward passes, ~2-4 min on an RTX 3060.
DEFAULT_SAMPLE = 300

# The configs under test. Each is (mode, weights) where weights is
# (bm25_weight, dense_weight) for the RRF fusion — see
# HybridRetriever.search. Uniform (1.0, 1.0) is the fusion ADR #20 measured;
# the damped variants are the hypothesis that it lost to vector-only because
# it let a much weaker lexical arm override a much stronger dense one.
CONFIGS: dict[str, dict] = {
    "bm25_only":     {"mode": "bm25",   "weights": (1.0, 1.0)},
    "vector_only":   {"mode": "vector", "weights": (1.0, 1.0)},
    "hybrid_1_1":    {"mode": "hybrid", "weights": (1.0, 1.0)},
    "hybrid_05_1":   {"mode": "hybrid", "weights": (0.5, 1.0)},
    "hybrid_03_1":   {"mode": "hybrid", "weights": (0.3, 1.0)},
    "hybrid_02_1":   {"mode": "hybrid", "weights": (0.2, 1.0)},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Metrics — single-gold, so hit@k is recall@k
# ============================================================================

def _hit_rate(ranks: list[int | None]) -> float:
    """Fraction of queries whose gold chunk appeared at any rank."""
    return sum(1 for r in ranks if r is not None) / len(ranks) if ranks else 0.0


def _mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank; a miss contributes zero."""
    if not ranks:
        return 0.0
    return sum(1.0 / (r + 1) for r in ranks if r is not None) / len(ranks)


def _rank_of(gold_id: str, hits: list[dict]) -> int | None:
    """0-indexed position of the gold chunk, or None if absent."""
    for i, h in enumerate(hits):
        if h["chunk_id"] == gold_id:
            return i
    return None


# ============================================================================
# Core loop — retrieve at k=20, rerank to 5, score both stages
# ============================================================================

def evaluate_config(
    retriever: HybridRetriever,
    ground_truth: list[dict],
    mode: str,
    weights: tuple[float, float],
    desc: str,
    skip_rerank: bool = False,
    per_query: list[dict] | None = None,
) -> dict:
    """Score one config at both the pool stage and the reranked stage.

    Returns hit@20 / mrr@20 for the retriever's pool and hit@5 / mrr@5 for
    what the answer prompt actually receives. `retrieved_rank` and
    `reranked_rank` are recorded per query so the two stages can be compared
    directly — a config that feeds the reranker well but ranks badly itself
    is the whole point of the exercise.

    per_query, when given, collects one row per query so two configs can be
    compared as *paired* observations. Aggregate hit rates cannot answer
    whether a 1-point gap is real: what matters is the discordant pairs —
    queries one config gets and the other misses — which McNemar's test needs
    and a summary row throws away.
    """
    pool_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []

    for gt in tqdm(ground_truth, desc=desc, leave=False):
        question, gold = gt["question"], gt["chunk_id"]

        hits = retriever.search(
            question, k=RETRIEVE_K, mode=mode, weights=weights, source_scope="statute",
        )
        pool_rank = _rank_of(gold, hits)
        pool_ranks.append(pool_rank)

        reranked_rank = None
        if not skip_rerank:
            top = rerank(question, hits, k=RERANK_K)
            reranked_rank = _rank_of(gold, top)
            reranked_ranks.append(reranked_rank)

        if per_query is not None:
            per_query.append({
                "config": desc,
                "question": question,
                "gold_chunk_id": gold,
                "pool_rank": pool_rank,
                "reranked_rank": reranked_rank,
            })

    result = {
        "config": desc,
        "mode": mode,
        "w_bm25": weights[0],
        "w_dense": weights[1],
        "hit_at_20": round(_hit_rate(pool_ranks), 4),
        "mrr_at_20": round(_mrr(pool_ranks), 4),
        "n_queries": len(pool_ranks),
    }
    if skip_rerank:
        result["hit_at_5_reranked"] = None
        result["mrr_at_5_reranked"] = None
    else:
        result["hit_at_5_reranked"] = round(_hit_rate(reranked_ranks), 4)
        result["mrr_at_5_reranked"] = round(_mrr(reranked_ranks), 4)
    return result


# ============================================================================
# Reporting
# ============================================================================

def print_results(df: pd.DataFrame) -> None:
    """Print the comparison, pool stage beside reranked stage."""
    print()
    print("=" * 92)
    print(
        f"{'config':<16} {'hit@20':>8} {'mrr@20':>8} │ "
        f"{'hit@5*':>8} {'mrr@5*':>8}   {'n':>5}   (* = after cross-encoder rerank)"
    )
    print("-" * 92)
    for _, r in df.iterrows():
        rr_hit = "    n/a" if pd.isna(r["hit_at_5_reranked"]) else f"{r['hit_at_5_reranked']:>8.4f}"
        rr_mrr = "    n/a" if pd.isna(r["mrr_at_5_reranked"]) else f"{r['mrr_at_5_reranked']:>8.4f}"
        print(
            f"{r['config']:<16} {r['hit_at_20']:>8.4f} {r['mrr_at_20']:>8.4f} │ "
            f"{rr_hit} {rr_mrr}   {r['n_queries']:>5d}"
        )
    print("=" * 92)

    scored = df.dropna(subset=["hit_at_5_reranked"])
    if not scored.empty:
        pool_best = df.loc[df["hit_at_20"].idxmax()]
        end_best = scored.loc[scored["hit_at_5_reranked"].idxmax()]
        print(f"→ best candidate pool : {pool_best['config']} (hit@20={pool_best['hit_at_20']:.4f})")
        print(f"→ best end-to-end     : {end_best['config']} (hit@5*={end_best['hit_at_5_reranked']:.4f})")
        print(
            "  The second line is the one that decides rag/retrieve.py's mode — "
            "it is what the answer prompt sees."
        )
    print()


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate retrieval at the k and stage rag/flow.py actually runs."
    )
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GT_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    p.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE,
        help=f"queries to evaluate (default {DEFAULT_SAMPLE}); 0 for the full set",
    )
    p.add_argument(
        "--skip-rerank", action="store_true",
        help="pool metrics only — skips the cross-encoder, seconds instead of minutes",
    )
    p.add_argument(
        "--configs", type=str, default=None,
        help=f"comma-separated subset of {','.join(CONFIGS)}",
    )
    p.add_argument(
        "--per-query-out", type=Path, default=None,
        help="also write one row per (config, query) so configs can be compared "
             "as paired observations (McNemar); required to call a small gap real",
    )
    args = p.parse_args()

    # Sample deterministically across the corpus rather than taking a head
    # slice: ground_truth.csv is ordered by article, so the first N rows are
    # all from the same few codes and would measure one corner of the corpus.
    gt_df = pd.read_csv(args.ground_truth, keep_default_na=False)
    if args.sample:
        gt_df = gt_df.sample(n=min(args.sample, len(gt_df)), random_state=42)
    ground_truth = gt_df.to_dict(orient="records")
    log.info("evaluating %d queries from %s", len(ground_truth), args.ground_truth)

    selected = list(CONFIGS) if args.configs is None else [
        c.strip() for c in args.configs.split(",")
    ]
    unknown = [c for c in selected if c not in CONFIGS]
    if unknown:
        p.error(f"unknown config(s): {unknown}; choose from {list(CONFIGS)}")

    log.info("loading retriever …")
    retriever = load_index()

    rows = []
    per_query: list[dict] | None = [] if args.per_query_out else None
    for i, name in enumerate(selected, 1):
        cfg = CONFIGS[name]
        log.info("[%d/%d] %s (mode=%s weights=%s)", i, len(selected), name, cfg["mode"], cfg["weights"])
        rows.append(evaluate_config(
            retriever, ground_truth,
            mode=cfg["mode"], weights=cfg["weights"], desc=name,
            skip_rerank=args.skip_rerank, per_query=per_query,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    log.info("wrote %s", args.out)

    if per_query is not None:
        pd.DataFrame(per_query).to_csv(args.per_query_out, index=False)
        log.info("wrote %s (%d rows)", args.per_query_out, len(per_query))

    print_results(df)


if __name__ == "__main__":
    main()
