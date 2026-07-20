"""
eval/retrieval_eval.py — 4-config retrieval evaluation harness
==============================================================

[... your full docstring from the previous message goes here ...]
"""
from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================
import argparse
import logging
import random
import sys
from math import sqrt
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from ingestion.load import load_index, HybridRetriever, DEFAULT_BM25_BOOST


# ============================================================================
# Config — paths, thresholds, search-space bounds
# ============================================================================
# Defaults are all overridable via CLI. Values chosen to match Zoomcamp
# module 04 / project reference 02-evaluating-retrieval.md exactly.
DEFAULT_GT_PATH   = Path("data/ground_truth.csv")
DEFAULT_OUT_PATH  = Path("data/retrieval_eval_results.csv")
DEFAULT_K         = 10
DEFAULT_N_ITER    = 20
DEFAULT_VAL_SIZE  = 100

# BM25 boost-tuning search space. (0.0, 3.0) mirrors Zoomcamp reference;
# 0.0 lower bound intentionally allows the optimizer to zero-out a field.
BOOST_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "texte":        (0.0, 3.0),
    "titre":        (0.0, 3.0),
    "section_path": (0.0, 3.0),
    "num":          (0.0, 3.0),
}

# Fix the RNG for reproducibility of the random search across peer-review runs.
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Metrics — pure functions, no I/O, easy to unit-test if we ever need to
# ============================================================================
# relevance_matrix[i] is a list of booleans: for query i, was each of the
# top-k results the correct chunk? Only one True per query is possible
# (there's only one correct chunk per question in synthetic ground truth).
def hit_rate(relevance_matrix: list[list[bool]]) -> float:
    """Fraction of queries whose correct chunk appears anywhere in top-k."""
    if not relevance_matrix:
        return 0.0
    hits = sum(1 for row in relevance_matrix if any(row))
    return hits / len(relevance_matrix)


def mrr(relevance_matrix: list[list[bool]]) -> float:
    """Mean reciprocal rank. Zero contribution if correct chunk not in top-k."""
    if not relevance_matrix:
        return 0.0
    total = 0.0
    for row in relevance_matrix:
        for rank, is_correct in enumerate(row):
            if is_correct:
                total += 1.0 / (rank + 1)
                break
    return total / len(relevance_matrix)


# ============================================================================
# Core eval loop — apply a retriever function to a ground-truth set
# ============================================================================
# retriever_fn takes (question: str) → list[dict with 'chunk_id' key].
# We build the boolean relevance matrix and hand it to the metric functions.
def evaluate(
    retriever_fn: Callable[[str], list[dict]],
    ground_truth: list[dict],
    k: int,
    desc: str = "evaluating",
) -> dict:
    """Return {'hit_rate': ..., 'mrr': ..., 'n': ...} for this retriever."""
    relevance_matrix: list[list[bool]] = []
    for gt in tqdm(ground_truth, desc=desc, leave=False):
        results = retriever_fn(gt["question"])
        correct_id = gt["chunk_id"]
        row = [r["chunk_id"] == correct_id for r in results[:k]]
        relevance_matrix.append(row)

    return {
        "hit_rate": hit_rate(relevance_matrix),
        "mrr":      mrr(relevance_matrix),
        "n":        len(relevance_matrix),
    }


# ============================================================================
# Boost tuning — random search over BM25 field weights on validation set
# ============================================================================
# Zoomcamp's simple_optimize pattern. We tune to maximize Hit Rate@k because
# it's the primary rubric-visible metric; MRR moves along for the ride.
def simple_optimize(
    param_ranges: dict[str, tuple[float, float]],
    objective: Callable[[dict], float],
    n_iter: int,
) -> tuple[dict, float]:
    """Return (best_params, best_score) from n_iter random samples."""
    best_params: dict = {}
    best_score: float = float("-inf")

    for i in tqdm(range(n_iter), desc="tuning boosts"):
        current = {field: random.uniform(low, high) for field, (low, high) in param_ranges.items()}
        score = objective(current)
        if score > best_score:
            best_score = score
            best_params = current
            log.info("  iter %d: new best hit_rate=%.4f  boosts=%s",
                     i + 1, best_score, {k: round(v, 3) for k, v in current.items()})
    return best_params, best_score


def boost_l2_from_uniform(boosts: dict[str, float]) -> float:
    """L2 distance of boost vector from [1, 1, 1, 1] — proves tuning did something."""
    return sqrt(sum((v - 1.0) ** 2 for v in boosts.values()))


# ============================================================================
# 4-config comparison orchestrator
# ============================================================================
# Splits ground truth val/test, runs the 4 configs on the test set, tunes
# boosts on the val set for config 4. Returns a DataFrame ready to write.
def compare_configs(
    retriever: HybridRetriever,
    ground_truth: list[dict],
    k: int,
    n_iter: int,
    val_size: int,
) -> pd.DataFrame:
    """Run all 4 configs, return results DataFrame."""

    # --- Split validation / test
    val_set = ground_truth[:val_size]
    test_set = ground_truth[val_size:]
    log.info("split: val=%d test=%d", len(val_set), len(test_set))

    # --- Config 1: BM25 only, default (uniform) boosts
    log.info("[1/4] evaluating bm25_only")
    r1 = evaluate(
        lambda q: retriever.search(q, k=k, mode="bm25"),
        test_set, k=k, desc="bm25_only",
    )

    # --- Config 2: dense vector only (Chroma / BGE-M3)
    log.info("[2/4] evaluating vector_only")
    r2 = evaluate(
        lambda q: retriever.search(q, k=k, mode="vector"),
        test_set, k=k, desc="vector_only",
    )

    # --- Config 3: hybrid with default (uniform) BM25 boosts
    log.info("[3/4] evaluating hybrid (default boosts)")
    r3 = evaluate(
        lambda q: retriever.search(q, k=k, mode="hybrid"),
        test_set, k=k, desc="hybrid",
    )

    # --- Config 4: tune BM25 boosts on val set, then evaluate on test set
    log.info("[4/4] tuning BM25 boosts on val set (n_iter=%d)", n_iter)
    def objective(boosts: dict) -> float:
        result = evaluate(
            lambda q: retriever.search(q, k=k, mode="hybrid", boost_dict=boosts),
            val_set, k=k, desc="  val",
        )
        return result["hit_rate"]

    random.seed(RANDOM_SEED)
    best_boosts, best_val_score = simple_optimize(BOOST_PARAM_RANGES, objective, n_iter)
    log.info("best boosts: %s  (val hit_rate=%.4f)", best_boosts, best_val_score)
    log.info("      L2 distance from uniform: %.3f", boost_l2_from_uniform(best_boosts))

    log.info("[4/4] evaluating hybrid_tuned on test set")
    r4 = evaluate(
        lambda q: retriever.search(q, k=k, mode="hybrid", boost_dict=best_boosts),
        test_set, k=k, desc="hybrid_tuned",
    )

    # --- Assemble results
    rows = [
        {"config": "bm25_only",     **_flatten(r1), "boost_distance": 0.0,
         "notes": "baseline (lexical)"},
        {"config": "vector_only",   **_flatten(r2), "boost_distance": 0.0,
         "notes": "baseline (semantic)"},
        {"config": "hybrid",        **_flatten(r3), "boost_distance": 0.0,
         "notes": "RRF fusion, uniform boosts"},
        {"config": "hybrid_tuned",  **_flatten(r4),
         "boost_distance": round(boost_l2_from_uniform(best_boosts), 3),
         "notes": f"boosts={ {k: round(v, 3) for k, v in best_boosts.items()} }"},
    ]
    return pd.DataFrame(rows)


def _flatten(result: dict) -> dict:
    """Rename keys for CSV output."""
    return {
        "hit_rate_at_10": round(result["hit_rate"], 4),
        "mrr_at_10":      round(result["mrr"], 4),
        "n_queries":      result["n"],
    }


# ============================================================================
# Pretty-print — same content as the CSV, but readable in the terminal
# ============================================================================
def print_results(df: pd.DataFrame) -> None:
    """Print a nicely aligned results table to stdout."""
    print()
    print("=" * 80)
    print(f"{'config':<16} {'hit@10':>8} {'mrr@10':>8} {'n':>6} {'|Δ|':>6}  notes")
    print("-" * 80)
    for _, row in df.iterrows():
        print(
            f"{row['config']:<16} "
            f"{row['hit_rate_at_10']:>8.4f} "
            f"{row['mrr_at_10']:>8.4f} "
            f"{row['n_queries']:>6d} "
            f"{row['boost_distance']:>6.3f}  {row['notes']}"
        )
    print("=" * 80)
    # Identify the winner
    winner = df.loc[df["hit_rate_at_10"].idxmax()]
    print(f"→ best: {winner['config']}  (hit@10={winner['hit_rate_at_10']:.4f})")
    print()


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    """Parse CLI, load artifacts, run comparison, write CSV, print table."""
    p = argparse.ArgumentParser(description="Compare 4 retrieval configs on ground truth")
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GT_PATH,  help="input ground-truth CSV")
    p.add_argument("--out",          type=Path, default=DEFAULT_OUT_PATH, help="output results CSV")
    p.add_argument("--k",            type=int,  default=DEFAULT_K,        help="top-k for Hit Rate & MRR")
    p.add_argument("--n-iter",       type=int,  default=DEFAULT_N_ITER,   help="random-search iterations")
    p.add_argument("--val-size",     type=int,  default=DEFAULT_VAL_SIZE, help="validation set size")
    p.add_argument("--sample",       type=int,  default=None,             help="if set, take first N ground-truth rows")
    args = p.parse_args()

    # --- Load retriever (cold start ~10s for BGE-M3)
    log.info("loading retriever …")
    retriever = load_index()

    # --- Load and (optionally) sample ground truth
    gt_df = pd.read_csv(args.ground_truth, keep_default_na=False)
    if args.sample is not None:
        gt_df = gt_df.head(args.sample)
    ground_truth = gt_df.to_dict(orient="records")
    log.info("loaded %d ground-truth rows from %s", len(ground_truth), args.ground_truth)

    if len(ground_truth) <= args.val_size:
        log.error("ground truth (%d rows) must be larger than val-size (%d)",
                  len(ground_truth), args.val_size)
        sys.exit(1)

    # --- Run the 4-config comparison
    results_df = compare_configs(
        retriever=retriever,
        ground_truth=ground_truth,
        k=args.k,
        n_iter=args.n_iter,
        val_size=args.val_size,
    )

    # --- Write + print
    results_df.to_csv(args.out, index=False)
    log.info("wrote %s", args.out)
    print_results(results_df)


if __name__ == "__main__":
    main()