"""
eval/ground_truth.py — synthetic Q/A generation for retrieval evaluation
========================================================================

[... your existing docstring stays as-is at top of file ...]
"""
from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

# ============================================================================
# Config — load env, define constants and thresholds in one place
# ============================================================================
# .env holds OPENAI_API_KEY; load before OpenAI() is instantiated.
load_dotenv()

# Default paths — all overridable via CLI flags.
DEFAULT_ARTICLES = Path("data/articles.csv")
DEFAULT_OUT      = Path("data/ground_truth.csv")
DEFAULT_FAILED   = Path("data/ground_truth_failed.txt")

# Model choice locked in the plan §6. Dated snapshot, not -latest alias.
DEFAULT_MODEL       = "gpt-4o-mini-2024-07-18"
DEFAULT_N           = 2
DEFAULT_TEMPERATURE = 0.7   # 0.7 for Q1/Q2 diversity; drop to 0.0 for full determinism.

# gpt-4o-mini pricing per 1M tokens (July 2024 snapshot).
PRICE_INPUT_PER_1M  = 0.150
PRICE_OUTPUT_PER_1M = 0.600

# Ops thresholds — Ctrl-C survival, cost brake, failure brake.
CHECKPOINT_EVERY   = 50      # write CSV every N articles
COST_KILL_AVG_USD  = 0.005   # abort if avg cost/call exceeds this
COST_KILL_AFTER_N  = 20      # only start checking cost after this many calls
FAILURE_KILL_RATE  = 0.10    # abort if >10% of calls fail (after 10 calls)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Pydantic schemas — constrain GPT-4o-mini output to a valid JSON shape
# ============================================================================
# structured output guarantees the model returns exactly this shape via
# `responses.parse(text_format=QuestionList)`.
class Question(BaseModel):
    """One generated question."""
    question: str = Field(description="Plain-French question, 10-25 words, no legal jargon.")


class QuestionList(BaseModel):
    """List of N questions returned per article."""
    questions: list[Question]


# ============================================================================
# Prompt template — the exact French text sent to the model
# ============================================================================
# Locked in the design spec above. Do not edit without re-running --sample 50
# and re-inspecting output before touching the full corpus.
PROMPT_TEMPLATE = """\
Tu génères des questions d'évaluation pour un système de recherche juridique
destiné à des héritiers non-juristes (petits-enfants confrontés à une succession).

À partir de l'article ci-dessous, génère exactement {n} questions différentes
qu'un héritier non-juriste pourrait poser et auxquelles cet article répond
directement.

RÈGLES:
- Langue: français courant, pas de jargon juridique.
- Registre: un petit-enfant qui découvre une succession, pas un avocat.
- Ne cite PAS le numéro de l'article dans la question.
- Ne recopie PAS les termes exacts de l'article — reformule.
- Chaque question doit être auto-portée (compréhensible sans contexte).
- Longueur: 10 à 25 mots par question.

ARTICLE:
Source: {source_label}, article {num}
Section: {section_path}
Texte: {texte}
"""


def build_prompt(row: pd.Series, n: int) -> str:
    """Fill the template for one article row. Empty section_path → '(racine)'."""
    return PROMPT_TEMPLATE.format(
        n=n,
        source_label=row["source_label"],
        num=row["num"],
        section_path=row["section_path"] or "(racine)",
        texte=row["texte"],
    )


# ============================================================================
# Single-article call — one prompt in, N questions + token usage out
# ============================================================================
# Raises on API/network/parse errors; the main loop's try/except handles them.
def generate_questions(
    client: OpenAI,
    row: pd.Series,
    n: int,
    model: str,
    temperature: float,
) -> tuple[list[str], int, int]:
    """Return (list of question strings, input_tokens, output_tokens)."""
    prompt = build_prompt(row, n)
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=QuestionList,
        temperature=temperature,
    )
    parsed: QuestionList = response.output_parsed
    questions = [q.question.strip() for q in parsed.questions if q.question.strip()]
    return (
        questions,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


# ============================================================================
# Resume safety — read what's already generated, skip on re-run
# ============================================================================
# Makes the script idempotent. If the CSV exists, we keep its rows and only
# work on articles whose chunk_id isn't in it yet.
def load_existing(out_path: Path) -> tuple[list[dict], set[str]]:
    """Return (existing_rows_as_dicts, set_of_chunk_ids_already_done)."""
    if not out_path.exists():
        return [], set()
    df = pd.read_csv(out_path, keep_default_na=False)
    log.info("resuming: %d rows already in %s", len(df), out_path)
    return df.to_dict(orient="records"), set(df["chunk_id"].astype(str))


# ============================================================================
# Checkpoint writer — full-file rewrite is fine at this scale
# ============================================================================
# 1584 rows × 2 cols is trivial. Simpler than append+dedup logic.
def write_checkpoint(rows: list[dict], out_path: Path) -> None:
    """Overwrite out_path with the current accumulated rows."""
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================================
# Main generation loop — the actual work happens here
# ============================================================================
# Per-article API call → append rows → periodically checkpoint → guard rails.
def build_ground_truth(
    articles_path: Path,
    out_path: Path,
    failed_path: Path,
    n_per: int,
    sample: int | None,
    model: str,
    temperature: float,
    dry_run: bool,
) -> None:
    """Generate N questions per article, write to out_path, resume-safe."""

    # --- Load source articles (optionally take head sample for validation runs)
    articles = pd.read_csv(articles_path, keep_default_na=False)
    if sample is not None:
        articles = articles.head(sample)
    log.info("loaded %d articles from %s", len(articles), articles_path)

    # --- Resume: figure out what's still to do
    rows, done = load_existing(out_path)
    to_do = articles[~articles["chunk_id"].isin(done)]
    log.info("%d done, %d to generate", len(done), len(to_do))

    # --- Dry run: just print projected cost and exit
    if dry_run:
        est_in  = len(to_do) * 600
        est_out = len(to_do) * 120
        est_cost = (est_in * PRICE_INPUT_PER_1M + est_out * PRICE_OUTPUT_PER_1M) / 1_000_000
        log.info("dry-run: would generate %d pairs at ~$%.4f", len(to_do) * n_per, est_cost)
        return

    # --- Init OpenAI client (raises here if key is missing — fail fast)
    client = OpenAI()

    # --- Counters for kill switches and final summary
    total_in_tokens  = 0
    total_out_tokens = 0
    failures         = 0
    processed        = 0

    # --- The loop
    for _, row in tqdm(to_do.iterrows(), total=len(to_do), desc="generating"):
        try:
            questions, in_tok, out_tok = generate_questions(
                client, row, n_per, model, temperature
            )
            total_in_tokens  += in_tok
            total_out_tokens += out_tok

            if not questions:
                log.warning("empty response for %s", row["chunk_id"])
            elif len(questions) != n_per:
                log.warning(
                    "%s: expected %d questions, got %d", row["chunk_id"], n_per, len(questions)
                )

            for q in questions:
                rows.append({"chunk_id": row["chunk_id"], "question": q})

        except Exception as e:
            # OpenAIError, pydantic.ValidationError, network errors, etc.
            failures += 1
            log.warning("failed on %s: %s: %s", row["chunk_id"], type(e).__name__, e)
            with failed_path.open("a") as f:
                f.write(f"{row['chunk_id']}\t{type(e).__name__}: {e}\n")

        processed += 1

        # --- Checkpoint every N articles to survive Ctrl-C
        if processed % CHECKPOINT_EVERY == 0:
            write_checkpoint(rows, out_path)
            log.info("checkpoint: %d rows written", len(rows))

        # --- Cost kill switch: bail if average cost/call is way off spec
        if processed >= COST_KILL_AFTER_N:
            avg_cost = (
                total_in_tokens * PRICE_INPUT_PER_1M
                + total_out_tokens * PRICE_OUTPUT_PER_1M
            ) / 1_000_000 / processed
            if avg_cost > COST_KILL_AVG_USD:
                log.error(
                    "COST KILL: avg $%.5f/call > threshold $%.5f — aborting",
                    avg_cost, COST_KILL_AVG_USD,
                )
                write_checkpoint(rows, out_path)
                sys.exit(1)

        # --- Failure kill switch: bail if >10% of calls are erroring out
        if processed >= 10 and failures / processed > FAILURE_KILL_RATE:
            log.error(
                "FAILURE KILL: %d/%d calls failed (>%.0f%%) — aborting",
                failures, processed, FAILURE_KILL_RATE * 100,
            )
            write_checkpoint(rows, out_path)
            sys.exit(1)

    # --- Final write
    write_checkpoint(rows, out_path)

    # --- Summary
    total_cost = (
        total_in_tokens * PRICE_INPUT_PER_1M
        + total_out_tokens * PRICE_OUTPUT_PER_1M
    ) / 1_000_000
    log.info("=" * 60)
    log.info("done: %d rows in %s", len(rows), out_path)
    log.info("tokens: in=%d out=%d", total_in_tokens, total_out_tokens)
    log.info("cost:   $%.4f  (%d calls, %d failures)", total_cost, processed, failures)
    if failures > 0:
        log.info("failed chunk_ids logged to %s", failed_path)


# ============================================================================
# CLI — argparse surface matching the design spec
# ============================================================================
def main() -> None:
    """Parse CLI flags and run the generator."""
    p = argparse.ArgumentParser(description="Generate synthetic ground-truth Q/A for retrieval eval")
    p.add_argument("--articles",    type=Path,  default=DEFAULT_ARTICLES,    help="input articles CSV")
    p.add_argument("--out",         type=Path,  default=DEFAULT_OUT,         help="output ground-truth CSV")
    p.add_argument("--failed",      type=Path,  default=DEFAULT_FAILED,      help="failed chunk_id log")
    p.add_argument("--n",           type=int,   default=DEFAULT_N,           help="questions per article")
    p.add_argument("--sample",      type=int,   default=None,                help="if set, only first N articles")
    p.add_argument("--model",       type=str,   default=DEFAULT_MODEL,       help="OpenAI model id")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="sampling temperature")
    p.add_argument("--dry-run",     action="store_true",                     help="skip API calls, print cost estimate")
    args = p.parse_args()

    build_ground_truth(
        articles_path=args.articles,
        out_path=args.out,
        failed_path=args.failed,
        n_per=args.n,
        sample=args.sample,
        model=args.model,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()