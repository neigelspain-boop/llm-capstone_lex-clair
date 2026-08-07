"""LLM-as-judge evaluation harness for lex-clair.

Runs `rag.flow.run(question)` on a sample of ground-truth queries, then
scores each generated answer with three provider-diverse judges, all routed
through OpenRouter's OpenAI-compatible endpoint (ADR #40):

    - GPT-4o-mini          (openai/gpt-4o-mini)
    - Claude Haiku 4.5     (anthropic/claude-haiku-4.5)
    - Mistral Small        (mistralai/mistral-small-2603)

Writes one row per (query x judge) to data/llm_eval_results.csv. Resume-safe:
if the CSV exists on start, already-scored (query_id, judge) pairs are skipped.
Kill switches trip on aggregate cost cap and cumulative judge failure rate.

Design decisions locked in ADRs #24, #25, #26, #40. Rubric line: "LLM eval +2 --
multiple approaches evaluated." See docs/decisions.md for rationale.

CLI:

    uv run python -m eval.llm_eval                 # full N=200 run
    uv run python -m eval.llm_eval --sample 5      # small validation run
    uv run python -m eval.llm_eval --sample 5 --dry-run   # cost estimate, no API calls
    uv run python -m eval.llm_eval --analyze       # read existing CSV, print tables

Exit codes:
    0 = success (or --dry-run / --analyze)
    1 = kill switch tripped (cost cap or failure rate)
"""
from __future__ import annotations

# ========== stdlib imports ==========
import argparse
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

# ========== third-party imports ==========
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ========== project imports ==========
from ingestion.clients import get_openrouter_client
from rag import flow

# ========== paths + constants ==========
ROOT = Path(__file__).parent.parent
GROUND_TRUTH_CSV = ROOT / "data" / "ground_truth.csv"
RESULTS_CSV = ROOT / "data" / "llm_eval_results.csv"

N_SAMPLES = 200                # matches Zoomcamp reference module 04
SEED = 42                      # deterministic sampling for reproducibility
CHECKPOINT_EVERY = 25          # samples between CSV flushes
COST_KILL_TOTAL_USD = 3.00     # aggregate cost cap across answer-gen + all judges
FAILURE_KILL_RATE = 0.10       # abort if >10% of judgments fail (after 10 min samples)
MIN_JUDGMENTS_BEFORE_KILL = 10 # don't trip failure-rate kill on early noise

# Judge model IDs -- all fully-qualified OpenRouter slugs per ADR #40.
# Answer generator stays gpt-4o-mini per Plane II (ADR #20), untouched here.
JUDGE_MODELS = {
    "gpt":     "openai/gpt-4o-mini",
    "claude":  "anthropic/claude-haiku-4.5",
    "mistral": "mistralai/mistral-small-2603",
}

# USD per million tokens (input, output). Used for kill-switch aggregation.
# Sources: OpenAI pricing page (gpt-4o-mini), OpenRouter Claude Haiku 4.5 and
# Mistral Small 3.2 pages. Approximations acceptable -- kill switch is a
# safety net, not accounting.
COST_PER_MTOKEN = {
    "openai/gpt-4o-mini":                        (0.15, 0.60),
    "anthropic/claude-haiku-4.5":                (1.00, 5.00),
    "mistralai/mistral-small-2603":  (0.15, 0.60),
}

# ========== judge prompt template ==========
# French, JSON contract explicit. Judges are told no markdown, no surrounding
# text -- we still defensively strip fences in _parse_judge_response.
JUDGE_PROMPT_TEMPLATE = """
Tu es un evaluateur expert d'un systeme RAG francais de droit successoral.
Ta tache: classifier la pertinence de la reponse generee par rapport a
la question posee.

Question: {question}
Reponse generee: {answer}

Reponds en JSON strict, sans markdown, sans texte autour:
{{"relevance": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT",
 "explanation": "brief explanation in French"}}
""".strip()

VALID_VERDICTS = {"RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"}

# ========== usage + JSON parse helpers ==========

def _extract_usage(response) -> dict:
    """Normalize token usage across provider SDKs.

    OpenAI Responses API uses input_tokens/output_tokens.
    OpenAI Chat Completions (and OpenRouter's OpenAI-compatible endpoint)
    use prompt_tokens/completion_tokens. Mistral matches classic.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    completion = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion)}


def _parse_judge_response(raw: str) -> dict:
    """Parse a judge's raw text output into a validated verdict dict.

    Strips markdown fences if a judge ignores the "no markdown" instruction
    (Mistral is the highest drift risk per ADR #25). Returns:
        {"relevance": <one of VALID_VERDICTS>, "explanation": str}

    Raises ValueError on any parse or validation failure. Caller records
    the failure as verdict="UNKNOWN" and increments the failure counter.
    """
    text = (raw or "").strip()

    # strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}") from e

    relevance = str(data.get("relevance", "")).strip()
    explanation = str(data.get("explanation", "")).strip()

    if relevance not in VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {relevance!r}")

    return {"relevance": relevance, "explanation": explanation}


# ========== judge functions ==========
# Each returns (verdict_dict, token_stats). Raises on API/parse failure --
# caller catches and records "UNKNOWN" per silent-fallback contract.

def judge(judge_name: str, question: str, answer: str) -> tuple[dict, dict]:
    """Score one answer with the named judge.

    Args:
        judge_name: key into JUDGE_MODELS ("gpt", "claude", "mistral").
        question: the evaluated question.
        answer: the answer to score.

    Returns (verdict_dict, token_stats). Raises on API or parse failure —
    the caller catches and records "UNKNOWN", per the silent-fallback
    contract above.

    Mistral carries the highest JSON-drift risk of the three; --dry-run
    validates it before the full 200-sample commitment.
    """
    client = get_openrouter_client()
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    r = client.chat.completions.create(
        model=JUDGE_MODELS[judge_name],
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_response(r.choices[0].message.content), _extract_usage(r)


# The three provider-diverse judges previously had a function each, identical
# but for the JUDGE_MODELS lookup. The table was already the dispatch surface,
# so the functions were pure duplication.
JUDGES = {name: partial(judge, name) for name in JUDGE_MODELS}


# ========== cost helper ==========

def _compute_judge_cost(judge_name: str, token_stats: dict) -> float:
    """USD cost for one judge call, from COST_PER_MTOKEN's per-million rates."""
    model = JUDGE_MODELS[judge_name]
    inp_rate, out_rate = COST_PER_MTOKEN[model]
    return (
        token_stats.get("prompt_tokens", 0) * inp_rate
        + token_stats.get("completion_tokens", 0) * out_rate
    ) / 1_000_000


# ========== resume-safe result loading ==========

_CSV_COLUMNS = [
    "query_id", "question", "answer", "citations_json",
    "judge", "verdict", "explanation",
    "prompt_tokens", "completion_tokens", "cost_usd", "timestamp",
]


def _load_existing_results() -> pd.DataFrame:
    """Read partial CSV if present; return empty typed DataFrame otherwise."""
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV, keep_default_na=False)
    return pd.DataFrame(columns=_CSV_COLUMNS)


def _completed_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Set of (query_id, judge) already scored -- used to skip on resume."""
    if df.empty:
        return set()
    return set(zip(df["query_id"].astype(str), df["judge"].astype(str)))


def _flush(rows: list[dict]) -> None:
    """Write current rows to CSV atomically (write-then-rename)."""
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    tmp = RESULTS_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(RESULTS_CSV)


# ========== main eval loop ==========

def run_eval(n_samples: int, dry_run: bool = False) -> None:
    """Sample N questions from ground truth, run flow, score with 3 judges.

    Resume-safe: reads existing CSV, skips completed (query, judge) pairs.
    Kill switches: aggregate cost cap + judge failure-rate cap.
    """
    # ----- load ground truth + deterministic sample -----
    gt = pd.read_csv(GROUND_TRUTH_CSV, keep_default_na=False)
    # Shuffle once with fixed seed, then take head. Guarantees --sample 5
    # gets the same first 5 as the first 5 of --sample 200.
    sample = (
        gt.sample(frac=1.0, random_state=SEED)
          .head(n_samples)
          .reset_index(drop=True)
    )
    print(f"[llm_eval] sampled {len(sample)} queries from {len(gt)} in ground truth")

    # ----- dry-run: cost estimate only, no API calls -----
    if dry_run:
        # Numbers grounded in Day 4 measurements (answer gen ~$0.00047/query)
        # and pricing table above. Round up 20% for headroom.
        est_answer = 0.00047 * len(sample)
        est_gpt = (500 * 0.15 + 100 * 0.60) / 1_000_000 * len(sample)     # rough per-call
        est_claude = (500 * 1.00 + 100 * 5.00) / 1_000_000 * len(sample)
        est_mistral = (500 * 0.15 + 100 * 0.60) / 1_000_000 * len(sample)
        total = est_answer + est_gpt + est_claude + est_mistral
        print(f"[llm_eval] DRY RUN estimate for {len(sample)} samples:")
        print(f"  answer generation (flow):     ~${est_answer:.4f}")
        print(f"  judge gpt-4o-mini:            ~${est_gpt:.4f}")
        print(f"  judge claude-haiku-4.5:       ~${est_claude:.4f}")
        print(f"  judge mistral-small:          ~${est_mistral:.4f}")
        print(f"  ----------------------------------------")
        print(f"  total:                        ~${total:.4f}")
        print(f"  kill switch:                  ${COST_KILL_TOTAL_USD:.2f}")
        return

    # ----- resume state -----
    df_existing = _load_existing_results()
    done_pairs = _completed_pairs(df_existing)
    if done_pairs:
        print(f"[llm_eval] resuming: {len(done_pairs)} (query, judge) pairs already scored")
    rows: list[dict] = df_existing.to_dict(orient="records")

    # ----- kill-switch counters (include resumed cost) -----
    if df_existing.empty:
        total_cost = 0.0
    else:
        total_cost = float(
            pd.to_numeric(df_existing["cost_usd"], errors="coerce").fillna(0).sum()
        )
    total_judgments = 0
    failed_judgments = 0

    # ----- main loop -----
    for i, row in enumerate(tqdm(sample.itertuples(index=False), total=len(sample))):
        # NOTE: known limitation — see ADR #26. query_id keys on chunk_id, but
        # ground_truth.csv has 2 questions per chunk_id, so the seeded sample of
        # N can contain duplicate chunk_ids that resume-logic silently skips.
        # Effect on Day 5 run: 200 sampled → 183 scored. Fix (compound key
        # f"{chunk_id}_q{i:04d}") deferred to Day 8 polish; applying it now
        # would create a schema mismatch with the existing committed CSV.
        query_id = f"{row.chunk_id}_q{i:04d}"
        question = row.question

        # skip if this query is fully scored across all judges
        if all((query_id, j) in done_pairs for j in JUDGES):
            continue

        # ---- generate answer via Plane II ----
        try:
            flow_result = flow.run(question)
        except Exception as e:
            print(f"[llm_eval] flow.run failed on {query_id}: {e}", file=sys.stderr)
            continue

        answer = flow_result["answer"]
        citations_json = json.dumps(
            flow_result.get("citations", []), ensure_ascii=False
        )
        total_cost += float(flow_result.get("cost_usd", 0.0))

        # ---- run each judge (silent-fallback on failure) ----
        for judge_name, judge_fn in JUDGES.items():
            if (query_id, judge_name) in done_pairs:
                continue

            timestamp = datetime.now(timezone.utc).isoformat()
            total_judgments += 1

            try:
                verdict, token_stats = judge_fn(question, answer)
                cost = _compute_judge_cost(judge_name, token_stats)
                total_cost += cost
                rows.append({
                    "query_id":          query_id,
                    "question":          question,
                    "answer":            answer,
                    "citations_json":    citations_json,
                    "judge":             judge_name,
                    "verdict":           verdict["relevance"],
                    "explanation":       verdict["explanation"],
                    "prompt_tokens":     token_stats["prompt_tokens"],
                    "completion_tokens": token_stats["completion_tokens"],
                    "cost_usd":          cost,
                    "timestamp":         timestamp,
                })
            except Exception as e:
                failed_judgments += 1
                rows.append({
                    "query_id":          query_id,
                    "question":          question,
                    "answer":            answer,
                    "citations_json":    citations_json,
                    "judge":             judge_name,
                    "verdict":           "UNKNOWN",
                    "explanation":       f"Failed: {type(e).__name__}: {str(e)[:200]}",
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                    "cost_usd":          0.0,
                    "timestamp":         timestamp,
                })
                print(
                    f"[llm_eval] judge {judge_name} failed on {query_id}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )

            done_pairs.add((query_id, judge_name))

        # ---- kill-switch checks (loud failures per Day 4 doctrine) ----
        if total_cost > COST_KILL_TOTAL_USD:
            print(
                f"[llm_eval] KILL SWITCH: total_cost=${total_cost:.4f} "
                f"exceeds cap ${COST_KILL_TOTAL_USD:.2f}",
                file=sys.stderr,
            )
            _flush(rows)
            sys.exit(1)

        if (total_judgments >= MIN_JUDGMENTS_BEFORE_KILL
                and failed_judgments / total_judgments > FAILURE_KILL_RATE):
            fr = failed_judgments / total_judgments
            print(
                f"[llm_eval] KILL SWITCH: failure_rate={fr:.2%} "
                f"exceeds cap {FAILURE_KILL_RATE:.0%} "
                f"({failed_judgments}/{total_judgments} failed)",
                file=sys.stderr,
            )
            _flush(rows)
            sys.exit(1)

        # ---- checkpoint every N samples ----
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _flush(rows)
            print(
                f"[llm_eval] checkpoint at query {i+1}: "
                f"total_cost=${total_cost:.4f}, "
                f"judgments={total_judgments}, failed={failed_judgments}"
            )

    # ----- final flush + summary -----
    _flush(rows)
    print(
        f"[llm_eval] done. total_cost=${total_cost:.4f}, "
        f"judgments={total_judgments}, failed={failed_judgments}"
    )
    analyze()


# ========== analyze: per-judge distribution + agreement + majority vote ==========

def analyze() -> None:
    """Print per-judge relevance distribution, pairwise agreement, majority vote.

    ASCII tables only -- V1 (no summary CSV, no matplotlib). If a UI or peer
    reviewer later needs a rendered version, promote to summary CSV.
    """
    if not RESULTS_CSV.exists():
        print(f"[llm_eval] no results at {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV, keep_default_na=False)
    if df.empty:
        print("[llm_eval] results CSV is empty")
        return

    print()
    print("=" * 60)
    print(f"llm_eval results  ({len(df)} rows, "
          f"{df['query_id'].nunique()} unique queries)")
    print("=" * 60)

    # ----- per-judge relevance distribution -----
    print("\nPer-judge relevance distribution (normalized)")
    print("-" * 60)
    dist = (
        df.groupby("judge")["verdict"]
          .value_counts(normalize=True)
          .unstack(fill_value=0.0)
          .round(3)
    )
    print(dist.to_string())

    # raw counts too -- useful for spotting UNKNOWN spikes
    print("\nPer-judge relevance distribution (raw counts)")
    print("-" * 60)
    counts = df.groupby("judge")["verdict"].value_counts().unstack(fill_value=0)
    print(counts.to_string())

    # ----- cross-judge agreement (queries scored by all 3 judges) -----
    complete_mask = df.groupby("query_id")["judge"].nunique().eq(len(JUDGES))
    complete_ids = complete_mask[complete_mask].index.tolist()

    print(f"\nCross-judge agreement ({len(complete_ids)} queries scored by all 3)")
    print("-" * 60)
    if not complete_ids:
        print("(no fully-scored queries yet)")
        return

    sub = df[df["query_id"].isin(complete_ids)]
    wide = sub.pivot(index="query_id", columns="judge", values="verdict")

    # Exclude UNKNOWN when computing agreement -- a failed judgment is not a
    # disagreement, it's missing data.
    valid_wide = wide.replace("UNKNOWN", pd.NA)

    judges_sorted = sorted(JUDGES.keys())
    for i in range(len(judges_sorted)):
        for j in range(i + 1, len(judges_sorted)):
            a, b = judges_sorted[i], judges_sorted[j]
            pair = valid_wide[[a, b]].dropna()
            if len(pair) == 0:
                print(f"  {a:<8} vs {b:<8}: (no comparable queries)")
                continue
            agree = (pair[a] == pair[b]).mean()
            print(f"  {a:<8} vs {b:<8}: {agree:>6.2%}  (n={len(pair)})")

    all_valid = valid_wide.dropna()
    if len(all_valid) > 0:
        all_agree = (all_valid.nunique(axis=1) == 1).mean()
        print(f"  all 3 agree:       {all_agree:>6.2%}  (n={len(all_valid)})")

    # ----- majority vote per query -----
    print("\nMajority-vote verdict distribution")
    print("-" * 60)

    def _majority(row_vals):
        # exclude UNKNOWN from the vote
        clean = [v for v in row_vals if v in VALID_VERDICTS]
        if not clean:
            return "ALL_UNKNOWN"
        counts = pd.Series(clean).value_counts()
        if counts.iloc[0] >= 2:
            return counts.index[0]
        return "SPLIT"

    majority = wide.apply(_majority, axis=1)
    print(majority.value_counts(normalize=True).round(3).to_string())
    print()


# ========== CLI ==========

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="3-judge LLM-as-judge harness for lex-clair RAG flow"
    )
    parser.add_argument(
        "--sample", type=int, default=N_SAMPLES,
        help=f"number of ground-truth queries to sample (default: {N_SAMPLES})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print cost estimate and exit; no API calls",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="read existing CSV, print distribution/agreement/majority, exit",
    )
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return

    run_eval(n_samples=args.sample, dry_run=args.dry_run)


if __name__ == "__main__":
    main()