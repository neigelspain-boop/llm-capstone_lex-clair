"""Per-doc named-entity mention extraction. Stage A of the person pipeline.

Reads data/dossier/<case_id>/extracted/<doc_id>.md (Attempt 1 extract output).
Emits data/dossier/<case_id>/_mentions/<doc_id>.json (one file per doc).

Idempotent via content-hash of source markdown. Non-destructive: does not
touch facts.jsonl, actor_roles.jsonl, or any other pipeline artifact.

Dormant by default. Not wired into build.py. Invoked via its own CLI.

ADR #54 documents context, decision, consequences.

CLI: python -m ingestion.dossier.mentions --case-id <id> [--doc-id <one>]
     [--force] [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ingestion.clients import (estimate_cost_usd, extract_usage,
                              get_openrouter_client, parse_json_list)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

_MENTIONS_MODEL = "anthropic/claude-haiku-4.5"
_MENTIONS_MAX_OUTPUT_TOKENS = 4096
_MENTIONS_TEMPERATURE = 0.0
_MENTIONS_SCHEMA_VERSION = 1

SYSTEM_PROMPT = """\
Tu es un extracteur d'entites nommees pour documents juridiques francais.

Ta tache : identifier toutes les personnes physiques et entites juridiques
mentionnees nommement dans le document fourni.

Regles strictes :
- Extrait UNIQUEMENT ce qui apparait explicitement dans le texte. Aucune inference.
- Personnes physiques : nom + prenom(s) si visibles, avec titres (Maitre, M., Mme, Me).
- Entites juridiques : denomination complete (SCP, SARL, societe civile, caisse, administration).
- Pour chaque mention, capture le surface_form litteral tel qu'il apparait (pas de normalisation).
- Fournis un context_snippet de 1 a 3 phrases entourant la mention.
- Ne deduplique PAS : chaque occurrence est une mention distincte, meme si le nom se repete.
- Si tu identifies un role juridique probable (notaire, heritier, mandataire, etc.), signale-le comme role_hint, mais laisse null si ce n'est pas ancre textuellement.

Sortie : JSON strict, tableau d'objets avec les champs :
  - surface_form: string (verbatim)
  - kind: "person" | "entity"
  - context_snippet: string (1-3 phrases du document)
  - role_hint: string | null

Aucun texte en dehors du JSON.
"""


# ========== content-hash cache ==========

def _mentions_cache_path(case_id: str, doc_id: str) -> Path:
    return DOSSIER_DIR / case_id / "_mentions" / f"{doc_id}.json"


def _hash_source(markdown: str) -> str:
    """Deterministic SHA-256 (first 16 hex chars) of the source markdown."""
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:16]


def _load_cache(cache_path: Path) -> tuple[str, list[dict]] | None:
    """Returns (source_hash, mentions_list) or None if cache miss.

    Cache file format:
    {
      "schema_version": 1,
      "source_hash": "sha256:...",
      "model": "anthropic/claude-haiku-4.5",
      "generated_at": "ISO8601",
      "mentions": [
        {"surface_form": "...", "kind": "person", "context_snippet": "...", "role_hint": null},
        ...
      ]
    }
    """
    if not cache_path.exists():
        return None
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    return entry["source_hash"], entry["mentions"]


def _write_cache_atomic(cache_path: Path, source_hash: str, mentions: list[dict]) -> None:
    """Atomic write via .tmp -> os.replace; .tmp removed if os.replace raises."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": _MENTIONS_SCHEMA_VERSION,
        "source_hash": source_hash,
        "model": _MENTIONS_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mentions": mentions,
    }
    tmp_path = cache_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ========== defensive JSON parsing ==========

def _parse_mentions_json(raw: str, doc_id: str) -> list[dict] | None:
    """Parse the extractor's raw response into a list of mention dicts.

    Returns None and logs a warning naming doc_id on any failure — callers
    must NOT cache a None result, only a real (possibly empty) list.
    """
    return parse_json_list(raw, "mentions", f"doc_id={doc_id}", log)


# ========== per-doc LLM call ==========

def _call_haiku_for_mentions(markdown: str, doc_id: str) -> tuple[list[dict] | None, dict]:
    """Returns (mentions, usage_dict). usage_dict has prompt_tokens,
    completion_tokens, cost_usd. Handles JSON parse failure by logging +
    returning (None, usage_dict) — None signals "nothing to cache", distinct
    from a real, legitimately empty [] extraction.
    """
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=_MENTIONS_MODEL,
        temperature=_MENTIONS_TEMPERATURE,
        max_tokens=_MENTIONS_MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    usage_dict = extract_usage(response, _MENTIONS_MODEL)

    mentions = _parse_mentions_json(raw, doc_id)
    return mentions, usage_dict


# ========== per-doc orchestration ==========

def extract_mentions_for_doc(case_id: str, doc_id: str, *, force: bool = False) -> dict:
    """Cache-first. Returns summary dict {mentions_count, cost_usd, from_cache: bool}."""
    md_path = DOSSIER_DIR / case_id / "extracted" / f"{doc_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"no extracted document at {md_path}")

    markdown = md_path.read_text(encoding="utf-8")
    source_hash = _hash_source(markdown)
    cache_path = _mentions_cache_path(case_id, doc_id)

    if not force:
        cached = _load_cache(cache_path)
        if cached is not None and cached[0] == source_hash:
            return {"mentions_count": len(cached[1]), "cost_usd": 0.0, "from_cache": True}

    mentions, usage = _call_haiku_for_mentions(markdown, doc_id)

    if mentions is None:
        # Parse failure: nothing valid to cache.
        return {"mentions_count": 0, "cost_usd": usage["cost_usd"], "from_cache": False}

    _write_cache_atomic(cache_path, source_hash, mentions)
    return {"mentions_count": len(mentions), "cost_usd": usage["cost_usd"], "from_cache": False}


# ========== case-level orchestration ==========

def extract_mentions_for_case(case_id: str, *, force: bool = False, dry_run: bool = False) -> dict:
    """Iterates all docs in data/dossier/{case_id}/extracted/*.md.

    Dry-run: no API calls; per-doc cost estimate from markdown length
    (matching distill.py's token-estimate formula), summed across docs.
    Real run: calls extract_mentions_for_doc per doc, sums usage.
    Logs summary at INFO on completion.
    """
    extracted_dir = DOSSIER_DIR / case_id / "extracted"
    md_paths = sorted(extracted_dir.glob("*.md"))

    docs_processed = 0
    mentions_extracted = 0
    cache_hits = 0
    total_cost = 0.0

    if dry_run:
        for md_path in md_paths:
            markdown = md_path.read_text(encoding="utf-8")
            prompt_tokens = (len(SYSTEM_PROMPT) + len(markdown)) // 4
            completion_tokens = _MENTIONS_MAX_OUTPUT_TOKENS // 4  # rough upper-bound estimate
            total_cost += estimate_cost_usd(_MENTIONS_MODEL, prompt_tokens, completion_tokens)
            docs_processed += 1
    else:
        for md_path in md_paths:
            doc_id = md_path.stem
            result = extract_mentions_for_doc(case_id, doc_id, force=force)
            docs_processed += 1
            mentions_extracted += result["mentions_count"]
            total_cost += result["cost_usd"]
            if result["from_cache"]:
                cache_hits += 1

    log.info(
        "mentions: case_id=%s docs_processed=%d mentions_extracted=%d cache_hits=%d "
        "total_cost=$%.4f dry_run=%s",
        case_id, docs_processed, mentions_extracted, cache_hits, total_cost, dry_run,
    )

    return {
        "case_id": case_id,
        "docs_processed": docs_processed,
        "mentions_extracted": mentions_extracted,
        "cache_hits": cache_hits,
        "total_cost": total_cost,
    }


# ========== CLI entrypoint ==========

def _cli() -> None:
    """argparse: --case-id (required), --doc-id (optional single-doc), --force, --dry-run, --verbose."""
    parser = argparse.ArgumentParser(
        description="Extract named-entity mentions from every extracted document in a case."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--doc-id", type=str, default=None,
        help="extract mentions for only this one doc_id, instead of the whole case",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore the content-hash cache; re-call the LLM",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="assemble the LLM calls but don't invoke them; print token/cost estimates only",
    )
    parser.add_argument("--verbose", action="store_true", help="enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.doc_id is not None:
        try:
            if args.dry_run:
                md_path = DOSSIER_DIR / args.case_id / "extracted" / f"{args.doc_id}.md"
                markdown = md_path.read_text(encoding="utf-8")
                prompt_tokens = (len(SYSTEM_PROMPT) + len(markdown)) // 4
                completion_tokens = _MENTIONS_MAX_OUTPUT_TOKENS // 4
                cost = estimate_cost_usd(_MENTIONS_MODEL, prompt_tokens, completion_tokens)
                print(f"mentions dry-run · doc_id={args.doc_id} cost_est=${cost:.4f}")
            else:
                result = extract_mentions_for_doc(args.case_id, args.doc_id, force=args.force)
                print(
                    f"mentions summary · doc_id={args.doc_id} "
                    f"mentions_count={result['mentions_count']} "
                    f"from_cache={result['from_cache']} "
                    f"cost_est=${result['cost_usd']:.4f}"
                )
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    summary = extract_mentions_for_case(args.case_id, force=args.force, dry_run=args.dry_run)
    print(
        f"mentions summary · case_id={summary['case_id']} "
        f"docs_processed={summary['docs_processed']} "
        f"mentions_extracted={summary['mentions_extracted']} "
        f"cache_hits={summary['cache_hits']} "
        f"cost_est=${summary['total_cost']:.4f}"
    )


if __name__ == "__main__":
    _cli()
