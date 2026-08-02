"""Fact-level distillation: ceremony-stripped substance summaries (Attempt 2,
Deliverable D5, ADR #52).

Facts today carry only `verbatim_quote` — the exact source sentence(s), kept
for forensic verification, but often ~80% ceremony (address blocks, standard
French legal formulas, restated prior correspondence) around ~20% legal
substance. This module adds a second field, `distilled_context`, alongside
it: a dense, ceremony-stripped restatement of the fact's substance, produced
by Haiku 4.5 from the fact's verbatim_quote plus surrounding source-document
context. `verbatim_quote` is never mutated — compliance reasons from
`distilled_context` and cross-verifies against `verbatim_quote` before
finalizing a verdict (fiability constraint, wired in D6).

Inputs:  data/dossier/<case_id>/facts.jsonl              (from facts.py)
         data/dossier/<case_id>/extracted/<doc_id>.md     (from extract.py)
Outputs: data/dossier/<case_id>/facts.jsonl               — distilled_context
                                                             backfilled in place
         data/dossier/<case_id>/distill_cache.jsonl       — content-hash cache

Facts are loaded and rewritten as raw dicts (json.loads/json.dumps), not
through the Fact pydantic model — this keeps every untouched field
byte-identical across a distillation run instead of risking pydantic
re-serialization drift.

CLI: python -m ingestion.dossier.distill --case-id <id> [--dry-run] [--fact-id <one>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from ingestion.clients import get_openrouter_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

DISTILL_MODEL = "anthropic/claude-haiku-4.5"
DISTILL_MAX_OUTPUT_TOKENS = 500  # distilled_context is dense but short
DISTILL_TEMPERATURE = 0.0  # determinism requirement
CONTEXT_WINDOW_CHARS = 2000  # surrounding source text fed to Haiku per fact

# Rough per-token USD rates for --dry-run token/cost estimates only, matching
# the anthropic/claude-haiku-4.5 rate in the model palette (~1 / ~5 per M).
# Real (non-dry-run) calls use the exact usage.cost OpenRouter returns.
_EST_PROMPT_USD_PER_TOKEN = 1e-6
_EST_COMPLETION_USD_PER_TOKEN = 5e-6
_DRY_RUN_EST_COMPLETION_TOKENS = 150  # 2-5 dense sentences, well under the 500 cap

SYSTEM_PROMPT = """\
Tu es un assistant juridique qui produit des notes de synthese denses a \
partir de correspondance et documents de succession francais.

On te fournit deux elements pour un fait juridique deja extrait :
1. La citation verbatim (verbatim_quote) qui soutient ce fait.
2. Le contexte du document source qui l'entoure.

Ta tache : produire une restitution dense, factuelle et depouillee de toute \
ceremonie, de la substance de ce fait — 2 a 5 phrases de prose dense, rien \
d'autre.

A CONSERVER ABSOLUMENT, sans exception :
- Toute date.
- Tout nom (personnes physiques et morales).
- Tout montant.
- Toute reference a un acte ou document anterieur (dates, types d'actes, \
numeros d'article).
- Toute citation directe d'un aveu, refus, contradiction, ou citation d'un \
texte de loi.
- L'identite du signataire.

A SUPPRIMER :
- Les blocs d'adresse et formules de politesse ("Je vous prie d'agreer...", \
"Madame, Monsieur...").
- Les codes de reference ou numeros de dossier, SAUF s'ils etablissent une \
chaine de traçabilite entre documents.
- La correspondance anterieure simplement rappelee, SAUF si la restitution \
elle-meme contredit un fait anterieur ou constitue la premiere mention d'un \
document.
- Les listes de pieces jointes, SAUF si la piece jointe EST la substance du \
fait.

Ne produis aucun preambule, aucune explication, aucun commentaire meta. \
N'ajoute et n'infere aucune information au-dela de ce que le texte source \
soutient. Reponds uniquement avec la restitution dense, en francais.
"""


# ========== source context loading ==========

def _load_source_context(case_id: str, source_doc_id: str) -> str:
    """Read the full verbatim .md transcript for one document."""
    md_path = DOSSIER_DIR / case_id / "extracted" / f"{source_doc_id}.md"
    return md_path.read_text(encoding="utf-8")


def _extract_fact_neighborhood(
    source_text: str, verbatim_quote: str, window: int = CONTEXT_WINDOW_CHARS
) -> str:
    """Return `window` chars of source_text centered on verbatim_quote.

    Falls back to source_text[:window] (with a logged warning) if the quote
    isn't found via substring match — e.g. whitespace normalization drift
    between extraction and fact extraction.
    """
    idx = source_text.find(verbatim_quote)
    if idx == -1:
        log.warning(
            "distill: verbatim_quote not found via substring match in source "
            "text (len=%d); falling back to source_text[:%d]",
            len(verbatim_quote), window,
        )
        return source_text[:window]

    half = window // 2
    start = max(0, idx - half)
    end = min(len(source_text), idx + len(verbatim_quote) + half)
    return source_text[start:end]


# ========== content-hash cache ==========

def _cache_key(verbatim_quote: str, source_context: str) -> str:
    """Deterministic SHA-256 cache key over (verbatim_quote + source_context[:200]).

    Pure function of its string inputs — same inputs always produce the same
    key, including across process restarts.
    """
    payload = (verbatim_quote + source_context[:200]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_distill_cache(case_id: str) -> dict[str, dict]:
    """Load data/dossier/<case_id>/distill_cache.jsonl into a {cache_key: entry} map."""
    cache_path = DOSSIER_DIR / case_id / "distill_cache.jsonl"
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cache[entry["cache_key"]] = entry
    return cache


def _append_distill_cache(case_id: str, entry: dict) -> None:
    """Append one cache entry — only called for a real (non-cached) API call."""
    cache_path = DOSSIER_DIR / case_id / "distill_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ========== per-fact distillation ==========

def distill_fact(
    fact: dict, source_context: str, cache: dict[str, dict], dry_run: bool = False
) -> tuple[str, dict]:
    """Distill one fact's verbatim_quote + source_context into dense prose.

    Returns (distilled_context, usage) where usage is
    {"prompt_tokens", "completion_tokens", "cost_usd", "cache_hit", "estimated"}.
    Cache-checked via _cache_key(fact["verbatim_quote"], source_context)
    against the in-memory `cache` map (loaded once per case by distill_case).
    On a miss, dry_run returns a token/cost estimate without calling the API
    or touching the cache; a real run calls Haiku 4.5 and returns fresh text.
    """
    key = _cache_key(fact["verbatim_quote"], source_context)
    cached = cache.get(key)
    if cached is not None:
        return cached["distilled_context"], {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "cache_hit": True,
            "estimated": False,
        }

    user_message = (
        f"Citation verbatim :\n{fact['verbatim_quote']}\n\n"
        f"Contexte du document source :\n{source_context}"
    )

    if dry_run:
        prompt_tokens = (len(SYSTEM_PROMPT) + len(user_message)) // 4
        completion_tokens = _DRY_RUN_EST_COMPLETION_TOKENS
        cost = (
            prompt_tokens * _EST_PROMPT_USD_PER_TOKEN
            + completion_tokens * _EST_COMPLETION_USD_PER_TOKEN
        )
        return "", {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "cache_hit": False,
            "estimated": True,
        }

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=DISTILL_MODEL,
        temperature=DISTILL_TEMPERATURE,
        max_tokens=DISTILL_MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    distilled_context = (response.choices[0].message.content or "").strip()

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost = getattr(usage, "cost", None) if usage is not None else None
    if cost is None:
        cost = (
            prompt_tokens * _EST_PROMPT_USD_PER_TOKEN
            + completion_tokens * _EST_COMPLETION_USD_PER_TOKEN
        )

    return distilled_context, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "cache_hit": False,
        "estimated": False,
    }


# ========== case-level orchestration ==========

def distill_case(case_id: str, dry_run: bool = False, fact_id: str | None = None) -> dict:
    """Backfill distilled_context on every fact (or one, via fact_id) in a case.

    Loads facts.jsonl as raw dicts (not through the Fact model) so untouched
    fields stay byte-identical on rewrite. Atomic rewrite via .tmp ->
    os.replace; the .tmp file is removed if os.replace raises, so a failure
    never leaves facts.jsonl half-written or a stray .tmp behind. dry_run
    never rewrites facts.jsonl or the cache.
    """
    facts_path = DOSSIER_DIR / case_id / "facts.jsonl"
    raw_facts = [
        json.loads(line)
        for line in facts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cache = _load_distill_cache(case_id)

    facts_distilled = 0
    cache_hits = 0
    total_cost = 0.0

    for fact in raw_facts:
        if fact_id is not None and fact["fact_id"] != fact_id:
            continue

        source_text = _load_source_context(case_id, fact["source_doc_id"])
        source_context = _extract_fact_neighborhood(source_text, fact["verbatim_quote"])

        distilled_context, usage = distill_fact(fact, source_context, cache, dry_run=dry_run)
        total_cost += usage["cost_usd"] or 0.0

        if usage["cache_hit"]:
            cache_hits += 1
            fact["distilled_context"] = distilled_context
            continue

        if dry_run:
            facts_distilled += 1
            continue

        fact["distilled_context"] = distilled_context
        facts_distilled += 1

        key = _cache_key(fact["verbatim_quote"], source_context)
        entry = {"cache_key": key, "distilled_context": distilled_context, "usage": usage}
        cache[key] = entry
        _append_distill_cache(case_id, entry)

    if not dry_run:
        tmp_path = facts_path.with_suffix(".jsonl.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for fact in raw_facts:
                    f.write(json.dumps(fact, ensure_ascii=False) + "\n")
            os.replace(tmp_path, facts_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    log.info(
        "distill: case_id=%s total_facts=%d facts_distilled=%d cache_hits=%d "
        "total_cost=$%.4f dry_run=%s",
        case_id, len(raw_facts), facts_distilled, cache_hits, total_cost, dry_run,
    )

    return {
        "case_id": case_id,
        "total_facts": len(raw_facts),
        "facts_distilled": facts_distilled,
        "cache_hits": cache_hits,
        "total_cost": total_cost,
    }


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.distill --case-id <id> [--dry-run] [--fact-id <one>]."""
    parser = argparse.ArgumentParser(
        description="Backfill distilled_context on every fact in a case's facts.jsonl."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="assemble the LLM calls but don't invoke them; print token/cost estimates only",
    )
    parser.add_argument(
        "--fact-id", type=str, default=None,
        help="distill only this one fact_id, instead of the whole case",
    )
    args = parser.parse_args()

    summary = distill_case(args.case_id, dry_run=args.dry_run, fact_id=args.fact_id)
    print(
        f"distill summary · case_id={summary['case_id']} "
        f"total_facts={summary['total_facts']} "
        f"facts_distilled={summary['facts_distilled']} "
        f"cache_hits={summary['cache_hits']} "
        f"cost_est=${summary['total_cost']:.2f}"
    )


if __name__ == "__main__":
    main()
