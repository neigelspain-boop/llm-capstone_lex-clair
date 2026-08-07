"""Role-scoped person resolution: direct extraction from distilled_context,
consolidated into persons.jsonl (Attempt 2, Deliverable D3, ADR #55).

The original person-pipeline design (ADR #54) planned D3 to consume
`_mentions/*.json` (D2's per-doc raw entity extractions) plus facts.jsonl.
In practice, every fact in the private case already carries `actor_role`
(Attempt 1's extractor) and a dense, name-preserving `distilled_context`
(D5, ADR #52) — routing through mentions.py's raw per-doc lists would only
add an indirection a resolver would have to cross-reference back against
the same distilled_context anyway. D3 instead extracts canonical persons
directly, role by role: for each actor_role with facts, Haiku 4.5 reads
the role's label/grounding note plus its (ambiguity-filtered) facts'
distilled_context (falling back to verbatim_quote) and returns canonical
persons with aliases, type, confidence, and evidence. Results are then
merged across roles by accent/case-normalized canonical_name into single
person records with a role_assignments list.

mentions.py remains shipped and dormant (ADR #54) — this module does not
read `_mentions/*.json` and does not modify facts.jsonl, actor_roles.jsonl,
or role_ambiguities.jsonl.

Inputs:  data/dossier/<case_id>/facts.jsonl
         data/dossier/<case_id>/actor_roles.jsonl
         data/dossier/<case_id>/role_ambiguities.jsonl
Outputs: data/dossier/<case_id>/persons.jsonl               — final output
         data/dossier/<case_id>/_persons_cache/<role_id>.json — content-hash cache

Dormant by default. Not wired into build.py. Invoked via its own CLI.

CLI: python -m ingestion.dossier.resolve --case-id <id> [--role-id <one>]
     [--force] [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from ingestion.clients import (estimate_cost_usd, extract_usage,
                              get_openrouter_client, parse_json_list)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

RESOLVE_MODEL = "anthropic/claude-haiku-4.5"
RESOLVE_MAX_OUTPUT_TOKENS = 2048
RESOLVE_TEMPERATURE = 0.0
RESOLVE_SCHEMA_VERSION = 1
MAX_FACTS_PER_ROLE = 20  # caps prompt size; facts are date-sorted before truncation

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# Honorific/legal-form prefixes stripped by _person_merge_key so cross-role
# merge doesn't treat "Maître X" and "X" as different people.
HONORIFIC_PREFIXES = (
    "maître", "maitre", "me", "monsieur", "madame", "mme", "mlle",
    "mademoiselle", "m.", "mr", "mr.", "docteur", "dr", "dr.",
    "professeur", "pr.", "scp", "sarl", "sas", "sa",
)

SYSTEM_PROMPT = """Tu es un assistant juridique qui identifie les personnes \
physiques et morales jouant un rôle précis dans un dossier de succession \
française.

Tu reçois :
- un rôle : son identifiant (role_id), son libellé en français (label_fr) \
et une note de contextualisation (grounding_note)
- une liste de faits assignés à ce rôle, chacun sous la forme \
"fact_id: ... | date: ... | distilled: ..."

Pour chaque personne canonique (physique ou morale) qui joue précisément ce \
rôle dans le dossier, identifie :
- canonical_name : la forme la plus complète et la plus formelle observée \
(ex. "Maître Eve-Marie MENA", pas "MENA")
- aliases : toutes les variantes de surface observées dans les faits \
(ex. ["MENA", "Eve-Marie MENA", "E. MENA", "Me MENA"])
- person_type : "natural_person" ou "legal_person"
- confidence : "high", "medium" ou "low"
- ambiguity_note : une chaîne de caractères expliquant l'incertitude, ou \
null si l'identité de la personne ne fait aucun doute
- evidence_fact_ids : la liste des fact_id (parmi ceux fournis) qui \
mentionnent cette personne dans ce rôle

Règles strictes :
- N'extrais que les personnes explicitement nommées ou identifiables sans \
ambiguïté dans les faits fournis. Aucune inférence à partir de \
connaissances externes.
- Si aucun nom clair n'apparaît pour ce rôle, réponds avec un tableau vide. \
Ne fabrique jamais de nom.
- N'attribue "high" que si la personne est nommée dans au moins 50% des \
faits fournis pour ce rôle. Sinon "medium" ou "low".
- Renseigne ambiguity_note chaque fois que confidence vaut "low" OU que \
plusieurs personnes candidates semblent également valides pour ce rôle.
- Réponds uniquement avec un tableau JSON strict, sans aucun commentaire, \
texte d'accompagnement ni balise markdown.

Pour le champ canonical_name, utilisez systématiquement la forme la plus \
complète et formelle observée dans les faits fournis (avec titre \
professionnel, prénom complet, nom de famille). N'émettez jamais deux \
entrées distinctes qui ne différeraient que par le titre honorifique \
(Maître, Monsieur, Madame, etc.) ou par la présence/absence du prénom. \
Ne créez pas d'entrée pour une personne désignée uniquement par \
'Monsieur X' ou 'Madame X' sans prénom si le contexte des faits pour ce \
rôle permet raisonnablement d'identifier la personne complète."""


# ========== content-hash cache ==========

def _resolve_cache_path(case_id: str, role_id: str) -> Path:
    return DOSSIER_DIR / case_id / "_persons_cache" / f"{role_id}.json"


def _hash_role_context(role_id: str, facts_slice: list[dict]) -> str:
    """Deterministic SHA-256 (first 16 hex chars) over role_id + this role's facts.

    fact_ids are sorted before joining so the hash is stable regardless of
    any upstream reordering; distilled_context excerpts are concatenated in
    facts_slice's given order, which _group_facts_by_role already makes
    deterministic (date-sorted, then truncated).
    """
    fact_ids_sorted = sorted(f["fact_id"] for f in facts_slice)
    excerpts = "".join((f.get("distilled_context") or "")[:200] for f in facts_slice)
    payload = (role_id + "".join(fact_ids_sorted) + excerpts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_role_cache(cache_path: Path) -> tuple[str, list[dict]] | None:
    """Returns (source_hash, persons_list) or None if cache miss.

    Cache file format:
    {
      "schema_version": 1,
      "source_hash": "...",
      "model": "anthropic/claude-haiku-4.5",
      "generated_at": "ISO8601",
      "persons": [{"canonical_name": "...", "aliases": [...], ...}, ...]
    }
    """
    if not cache_path.exists():
        return None
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    return entry["source_hash"], entry["persons"]


def _write_role_cache_atomic(cache_path: Path, source_hash: str, persons: list[dict]) -> None:
    """Atomic write via .tmp -> os.replace; .tmp removed if os.replace raises."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": RESOLVE_SCHEMA_VERSION,
        "source_hash": source_hash,
        "model": RESOLVE_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "persons": persons,
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

def _parse_persons_json(raw: str, role_id: str) -> list[dict] | None:
    """Parse the resolver's raw response into a list of person dicts.

    Returns None and logs a warning naming role_id on any failure — callers
    must NOT cache a None result, only a real (possibly empty) list.
    """
    return parse_json_list(raw, "resolve", f"role_id={role_id}", log)


# ========== fact grouping by role ==========

def _group_facts_by_role(facts_path: Path, ambiguities_path: Path) -> dict[str, list[dict]]:
    """Groups facts by actor_role, excluding ambiguity-flagged facts.

    Facts whose fact_id appears in any role_ambiguities entry's fact_ids
    are dropped from every role's context — an ambiguous fact doesn't
    contribute reliable signal to resolving who plays a given role. Each
    role's remaining facts are sorted by date (nulls last, deterministic)
    and truncated to MAX_FACTS_PER_ROLE.
    """
    facts = [
        json.loads(line)
        for line in facts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    ambiguities: list[dict] = []
    if ambiguities_path.exists():
        ambiguities = [
            json.loads(line)
            for line in ambiguities_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    excluded_fact_ids: set[str] = set()
    for amb in ambiguities:
        excluded_fact_ids.update(amb.get("fact_ids", []))

    grouped: dict[str, list[dict]] = {}
    for fact in facts:
        if fact["fact_id"] in excluded_fact_ids:
            continue
        grouped.setdefault(fact["actor_role"], []).append(fact)

    for role_id, role_facts in grouped.items():
        role_facts.sort(key=lambda f: (f["date"] is None, f["date"] or ""))
        grouped[role_id] = role_facts[:MAX_FACTS_PER_ROLE]

    return grouped


def _build_role_prompt(role_id: str, role_meta: dict, facts_for_role: list[dict]) -> str:
    lines = [
        f"role_id : {role_id}",
        f"label_fr : {role_meta['label_fr']}",
        f"grounding_note : {role_meta['grounding_note']}",
        "",
        "Faits :",
    ]
    for f in facts_for_role:
        distilled = f.get("distilled_context") or f["verbatim_quote"]
        lines.append(f"fact_id: {f['fact_id']} | date: {f['date']} | distilled: {distilled}")
    return "\n".join(lines)


# ========== per-role LLM call ==========

def _call_haiku_for_role(
    role_id: str, role_meta: dict, facts_for_role: list[dict], dry_run: bool = False
) -> tuple[list[dict] | None, dict]:
    """Returns (persons, usage_dict). usage_dict has prompt_tokens,
    completion_tokens, cost_usd, estimated. dry_run computes a token/cost
    estimate without calling the API (persons is None, nothing to cache).
    A JSON parse failure also returns (None, usage_dict) — None signals
    "nothing to cache", distinct from a real, legitimately empty []
    extraction.
    """
    user_message = _build_role_prompt(role_id, role_meta, facts_for_role)

    if dry_run:
        prompt_tokens = (len(SYSTEM_PROMPT) + len(user_message)) // 4
        completion_tokens = RESOLVE_MAX_OUTPUT_TOKENS // 4
        cost = estimate_cost_usd(RESOLVE_MODEL, prompt_tokens, completion_tokens)
        return None, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "estimated": True,
        }

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=RESOLVE_MODEL,
        temperature=RESOLVE_TEMPERATURE,
        max_tokens=RESOLVE_MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    usage_dict = extract_usage(response, RESOLVE_MODEL)

    persons = _parse_persons_json(raw, role_id)
    return persons, usage_dict


# ========== per-role orchestration ==========

def _drop_bare_surname_placeholders(role_id: str, persons: list[dict]) -> list[dict]:
    """Drops bare-surname candidates (e.g. "Monsieur BOSSAVIT") from a
    single role's freshly-resolved persons list when another candidate in
    the SAME list shares that surname as a whitespace-separated token of
    its own merge key (e.g. "Eric BOSSAVIT" contains token "bossavit").
    Token-match, not prefix "starts with": names here are "firstname
    SURNAME" ordered, so the surname is the LAST token. Symmetric — if
    two bare-surname candidates reduce to the same single-token key
    (e.g. "Monsieur BOSSAVIT" / "Madame BOSSAVIT"), each supersedes the
    other and BOTH are dropped. Logs one warning per dropped record."""
    keys = [_person_merge_key(p["canonical_name"]) for p in persons]
    keep: list[dict] = []
    for i, p in enumerate(persons):
        key = keys[i]
        if key and " " not in key:
            superseded_by = next(
                (
                    persons[j]["canonical_name"]
                    for j, other_key in enumerate(keys)
                    if j != i and key in other_key.split(" ")
                ),
                None,
            )
            if superseded_by is not None:
                log.warning(
                    "resolve: dropping bare-surname placeholder for role_id=%s: "
                    "canonical_name=%r superseded by canonical_name=%r",
                    role_id, p["canonical_name"], superseded_by,
                )
                continue
        keep.append(p)
    return keep


def resolve_role(
    case_id: str,
    role_id: str,
    role_meta: dict,
    facts_for_role: list[dict],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Cache-first. Returns {persons, cost_usd, from_cache}.

    Cache is checked (and can be hit) even under dry_run — mirrors
    distill.py's distill_fact ordering, where the cache check happens
    before the dry-run branch. Only a real cache miss triggers a dry-run
    cost estimate or a real API call.
    """
    cache_path = _resolve_cache_path(case_id, role_id)
    source_hash = _hash_role_context(role_id, facts_for_role)

    if not force:
        cached = _load_role_cache(cache_path)
        if cached is not None and cached[0] == source_hash:
            return {"persons": cached[1], "cost_usd": 0.0, "from_cache": True}

    persons, usage = _call_haiku_for_role(role_id, role_meta, facts_for_role, dry_run=dry_run)

    if dry_run or persons is None:
        return {"persons": [], "cost_usd": usage["cost_usd"], "from_cache": False}

    persons = _drop_bare_surname_placeholders(role_id, persons)

    _write_role_cache_atomic(cache_path, source_hash, persons)
    return {"persons": persons, "cost_usd": usage["cost_usd"], "from_cache": False}


# ========== case-level orchestration: merge + write persons.jsonl ==========

def _person_merge_key(name: str) -> str:
    """Accent/case/whitespace-insensitive key used only to detect the same
    canonical person across roles — not used for person_id generation.
    Also strips a single leading French honorific/legal-form prefix
    (Maître, Monsieur, SCP, ...) so "Maître X" and "X" collapse to the
    same key; only the first matching prefix at the very start is
    stripped (one pass — stacked honorifics are not repeatedly stripped)."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    leading_trimmed = ascii_only.lstrip()
    for prefix in HONORIFIC_PREFIXES:
        if leading_trimmed == prefix:
            leading_trimmed = ""
            break
        if (
            leading_trimmed.startswith(prefix)
            and leading_trimmed[len(prefix):len(prefix) + 1] in (" ", "\t")
        ):
            leading_trimmed = leading_trimmed[len(prefix):]
            break
    return re.sub(r"\s+", " ", leading_trimmed).strip()


def _slugify(text: str) -> str:
    """Strip accents, lowercase, and collapse anything non-alnum into hyphens."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "-", ascii_only)
    ascii_only = re.sub(r"-+", "-", ascii_only).strip("-")
    return ascii_only or "person"


def _merge_persons_across_roles(case_id: str, role_persons: dict[str, list[dict]]) -> list[dict]:
    """Merges per-role person extractions into case-scoped persons, keyed
    by accent/case-normalized canonical_name. On a merge collision, the
    longer canonical_name string wins (approximates "fullest, most formal
    form"); the shorter form is kept as an alias. Person-level confidence
    is the max across role_assignments; ambiguity_note is the first
    non-null one, in role-processing order.
    """
    merged: dict[str, dict] = {}

    for role_id, persons in role_persons.items():
        for p in persons:
            key = _person_merge_key(p["canonical_name"])
            role_assignment = {
                "role_id": role_id,
                "confidence": p["confidence"],
                "ambiguity_note": p.get("ambiguity_note"),
                "evidence_fact_ids": p.get("evidence_fact_ids", []),
            }

            if key not in merged:
                merged[key] = {
                    "canonical_name": p["canonical_name"],
                    "aliases": list(p.get("aliases", [])),
                    "person_type": p.get("person_type"),
                    "role_assignments": [role_assignment],
                }
                continue

            rec = merged[key]
            if len(p["canonical_name"]) > len(rec["canonical_name"]):
                if rec["canonical_name"] not in rec["aliases"]:
                    rec["aliases"].append(rec["canonical_name"])
                rec["canonical_name"] = p["canonical_name"]
            elif p["canonical_name"] not in rec["aliases"] and p["canonical_name"] != rec["canonical_name"]:
                rec["aliases"].append(p["canonical_name"])
            for alias in p.get("aliases", []):
                if alias not in rec["aliases"]:
                    rec["aliases"].append(alias)
            rec["role_assignments"].append(role_assignment)

    persons_out = []
    for rec in merged.values():
        confidences = [ra["confidence"] for ra in rec["role_assignments"]]
        person_confidence = max(confidences, key=lambda c: _CONFIDENCE_RANK.get(c, 0))
        ambiguity_note = next(
            (ra["ambiguity_note"] for ra in rec["role_assignments"] if ra["ambiguity_note"]), None
        )
        person_id = f"{case_id}-{_slugify(rec['canonical_name'])}"
        persons_out.append({
            "person_id": person_id,
            "canonical_name": rec["canonical_name"],
            "aliases": rec["aliases"],
            "person_type": rec["person_type"],
            "confidence": person_confidence,
            "ambiguity_note": ambiguity_note,
            "role_assignments": rec["role_assignments"],
        })

    return persons_out


def resolve_case(case_id: str, *, force: bool = False, dry_run: bool = False) -> dict:
    """Resolves persons for every actor_role with fact_count > 0, merges
    across roles, and atomically writes persons.jsonl. dry_run never
    writes the cache or persons.jsonl.
    """
    case_dir = DOSSIER_DIR / case_id
    facts_path = case_dir / "facts.jsonl"
    actor_roles_path = case_dir / "actor_roles.jsonl"
    ambiguities_path = case_dir / "role_ambiguities.jsonl"

    actor_roles = [
        json.loads(line)
        for line in actor_roles_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    facts_by_role = _group_facts_by_role(facts_path, ambiguities_path)

    total_roles_processed = 0
    cache_hits = 0
    total_cost = 0.0
    role_persons: dict[str, list[dict]] = {}

    for role in actor_roles:
        if role["fact_count"] == 0:
            continue
        role_id = role["role_id"]
        facts_for_role = facts_by_role.get(role_id, [])
        if not facts_for_role:
            # every fact for this role was ambiguity-flagged: no context to resolve
            continue

        result = resolve_role(case_id, role_id, role, facts_for_role, force=force, dry_run=dry_run)
        total_roles_processed += 1
        total_cost += result["cost_usd"]
        if result["from_cache"]:
            cache_hits += 1
        role_persons[role_id] = result["persons"]

    if dry_run:
        log.info(
            "resolve: case_id=%s dry_run=True total_roles_processed=%d total_cost=$%.4f",
            case_id, total_roles_processed, total_cost,
        )
        return {
            "total_roles_processed": total_roles_processed,
            "total_persons_resolved": 0,
            "cache_hits": cache_hits,
            "total_cost": total_cost,
            "dry_run": True,
        }

    persons = _merge_persons_across_roles(case_id, role_persons)

    persons_path = case_dir / "persons.jsonl"
    tmp_path = persons_path.with_suffix(".jsonl.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for p in sorted(persons, key=lambda p: p["person_id"]):
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        os.replace(tmp_path, persons_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    log.info(
        "resolve: case_id=%s total_roles_processed=%d total_persons_resolved=%d "
        "cache_hits=%d total_cost=$%.4f dry_run=False",
        case_id, total_roles_processed, len(persons), cache_hits, total_cost,
    )

    return {
        "total_roles_processed": total_roles_processed,
        "total_persons_resolved": len(persons),
        "cache_hits": cache_hits,
        "total_cost": total_cost,
        "dry_run": False,
    }


# ========== CLI entrypoint ==========

def _cli() -> None:
    """argparse: --case-id (required), --role-id (optional single-role), --force, --dry-run, --verbose."""
    parser = argparse.ArgumentParser(
        description="Resolve canonical persons per actor_role and write persons.jsonl."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--role-id", type=str, default=None,
        help="resolve only this one role_id (does not touch persons.jsonl — a full "
        "merge needs every role)",
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

    if args.role_id is not None:
        try:
            case_dir = DOSSIER_DIR / args.case_id
            actor_roles = [
                json.loads(line)
                for line in (case_dir / "actor_roles.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            roles_by_id = {r["role_id"]: r for r in actor_roles}
            if args.role_id not in roles_by_id:
                raise ValueError(f"unknown role_id: {args.role_id}")

            facts_by_role = _group_facts_by_role(
                case_dir / "facts.jsonl", case_dir / "role_ambiguities.jsonl"
            )
            facts_for_role = facts_by_role.get(args.role_id, [])

            result = resolve_role(
                args.case_id, args.role_id, roles_by_id[args.role_id], facts_for_role,
                force=args.force, dry_run=args.dry_run,
            )
            print(
                f"resolve summary · role_id={args.role_id} "
                f"persons_count={len(result['persons'])} "
                f"from_cache={result['from_cache']} "
                f"cost_est=${result['cost_usd']:.4f}"
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    summary = resolve_case(args.case_id, force=args.force, dry_run=args.dry_run)
    print(
        f"resolve summary · case_id={args.case_id} "
        f"total_roles_processed={summary['total_roles_processed']} "
        f"total_persons_resolved={summary['total_persons_resolved']} "
        f"cache_hits={summary['cache_hits']} "
        f"cost_est=${summary['total_cost']:.4f} "
        f"dry_run={summary['dry_run']}"
    )


if __name__ == "__main__":
    _cli()
