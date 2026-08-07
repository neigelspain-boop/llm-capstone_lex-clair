"""Verbatim transcripts → structured, atomic legal facts + discovered actor
roles (Plane I · offline).

Each dossier document's verbatim .md transcript (from extract.py) is reduced
to a list of validated, atomic `Fact` records — the fact base that
rag/flow.py's compliance-gap analysis (`flow.analyze(case_id)`) reasons over
alongside the statute corpus.

Actor roles are DISCOVERED per case, not drawn from a hardcoded enum: a
closed set of roles would make the pipeline case-specific and unreplicable
across different French succession dossiers (see ADR #36 in
docs/decisions.md). Each fact's `actor_role` is a validated snake_case
string; roles are catalogued per case in `actor_roles.jsonl` with a French
label and legal grounding note, and role assignments the extractor couldn't
disambiguate are logged in `role_ambiguities.jsonl` rather than silently
resolved.

Inputs:  data/dossier/<case_id>/extracted/<doc_id>.md  (from extract.py)
Outputs: data/dossier/<case_id>/facts.jsonl            — one JSON line per Fact
         data/dossier/<case_id>/actor_roles.jsonl       — one JSON line per
                                                           unique discovered role
         data/dossier/<case_id>/role_ambiguities.jsonl  — one JSON line per
                                                           unresolved ambiguity

Downstream consumers: index.py (Deliverable 5) backfills source_chunk_id on
each Fact once the source document is chunked/indexed. Day B analysis reads
facts.jsonl + actor_roles.jsonl + role_ambiguities.jsonl as its complete
Plane Ib output.

Model family chosen for provider diversity — extract uses Qwen, gate uses
Anthropic, facts uses Google, preserving cross-provider bias-independence
across the pipeline.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from ingestion.clients import get_anthropic_client, strip_json_fences

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== module constants ==========

ROOT = Path(__file__).resolve().parents[2]
DOSSIER_DIR = ROOT / "data" / "dossier"

# Gemini 3.1 Flash Lite: GA, text+multimodal input, priced for high-volume
# extraction workloads. Verified live on OpenRouter's catalogue 2026-07-27
# after google/gemini-2.0-flash-001 (retired 2026-03-31) started 404ing —
# see ADR #37 in docs/decisions.md.
FACT_EXTRACTOR_MODEL_ID = "google/gemini-3.1-flash-lite"

# snake_case, starts with a lowercase letter, 3-61 chars total.
ROLE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,60}$")

_FACT_EXTRACTOR_SYSTEM_PROMPT = """\
Tu es un extracteur de faits juridiques d'un document français de succession, écrit pour un système d'analyse comparative fait/norme.

On te fournit la transcription verbatim d'un document. Tu dois produire trois choses en un seul objet JSON :

1. Les faits juridiquement significatifs — actions prises par un acteur, états juridiques constatés, obligations ou délais qui s'imposent à un acteur.
2. Le catalogue des rôles d'acteurs que tu identifies dans ce document, avec pour chaque rôle un label français et une justification juridique.
3. Les ambiguïtés de rôle — cas où tu ne peux pas trancher entre deux ou plusieurs rôles pour un même acteur, faute d'information dans le texte.

Contraintes générales sur les faits :
- Chaque fait doit avoir une citation verbatim (verbatim_quote) : la phrase exacte du document qui soutient le fait, copiée sans reformulation. Pas de citation, pas de fait.
- Le champ action est une phrase verbale courte (≤ 15 mots), en français.
- Le champ date au format ISO 8601 "YYYY-MM-DD" seulement si la date est sans ambiguïté. Sinon, null.
- Ne pas inférer d'intentions ni attribuer de mauvaise foi. Extraire ce qui est écrit.
- Ne pas dédoubler un même fait s'il apparaît plusieurs fois dans le document.

Contraintes sur les rôles :
- Chaque acteur mentionné dans un fait doit être caractérisé par un role_id, PAS par son nom personnel. Ex: pas "Maître MENA" mais "notaire_redacteur". Pas "Roxane BOSSAVIT" mais "heritier_nu_proprietaire".
- Les role_id sont en snake_case, sans accents, sans espaces, 3 à 61 caractères, commençant par une lettre minuscule. Ex: "notaire_redacteur", "gestionnaire_scpi", "heritier_nu_proprietaire", "mediateur_notariat", "assureur_rcp", "chambre_notaires", "administration_fiscale", "tribunal_judiciaire", "juge_des_contentieux".
- Les role_id doivent être juridiquement fondés dans le droit français de la succession, de la responsabilité notariale, ou de la procédure civile. Le principe : un tiers avocat ou magistrat doit reconnaître immédiatement le rôle à partir du role_id, indépendamment du nom personnel de l'acteur.
- Un même role_id peut couvrir plusieurs acteurs individuels (ex: trois héritiers nus-propriétaires partagent le rôle "heritier_nu_proprietaire").
- Si un acteur n'a manifestement pas de rôle juridique reconnaissable, utilise "autre" et documente pourquoi dans le grounding_note.

Contraintes sur les ambiguïtés :
- Si un acteur pourrait raisonnablement être caractérisé par deux role_id différents et que le texte ne permet pas de trancher, émet une ambiguïté ET choisis provisoirement le role_id qui te semble le plus probable pour construire le fait. Liste dans candidate_role_ids les rôles alternatifs.
- Exemple : "Maître VIGNERON envoie un mail" — sans contexte supplémentaire sur son statut, tu ne peux pas savoir si c'est un "notaire_associe" ou un "notaire_stagiaire". Émets un fait provisoire avec "notaire_associe" ET une ambiguïté listant ["notaire_associe", "notaire_stagiaire"] avec une note.
- Ne pas émettre d'ambiguïté si le texte lève lui-même la doute.

Champ confidence sur les rôles :
- "high" : rôle explicitement nommé dans le texte, sans ambiguïté (ex: "Le notaire soussigné Maître MENA...").
- "medium" : rôle inféré du contexte mais raisonnablement clair.
- "low" : rôle inféré avec incertitude — probablement accompagné d'une entrée dans ambiguities.

Retourne un objet JSON strict, sans texte autour, sans fences markdown, avec la structure suivante :

{
  "facts": [
    {
      "date": "YYYY-MM-DD" ou null,
      "actor_role": "<role_id snake_case>",
      "action": "<verbe court>",
      "target": "<complément>" ou null,
      "verbatim_quote": "<phrase exacte du document>"
    },
    ...
  ],
  "roles_discovered": [
    {
      "role_id": "<snake_case>",
      "label_fr": "<label lisible en français>",
      "grounding_note": "<une phrase : pourquoi ce rôle existe en droit français>",
      "confidence": "high" | "medium" | "low"
    },
    ...
  ],
  "ambiguities": [
    {
      "verbatim_quote": "<phrase du document qui déclenche l'ambiguïté>",
      "candidate_role_ids": ["<role_id_1>", "<role_id_2>", ...],
      "note": "<explication courte>"
    },
    ...
  ]
}

Si le document ne contient aucun fait extractable (page vide, en-tête seul, illisible), retourne {"facts": [], "roles_discovered": [], "ambiguities": []}.
"""


# ========== schemas ==========

class Fact(BaseModel):
    """One atomic, sourced legal fact extracted from a dossier document."""

    fact_id: str                  # deterministic: f"{doc_id}-f{index:03d}"
    date: str | None              # ISO 8601 "YYYY-MM-DD" if unambiguous, else None
    actor_role: str               # snake_case, validated against ROLE_ID_PATTERN
    action: str                   # short verb phrase, <=15 words
    target: str | None            # object of action, <=15 words
    verbatim_quote: str           # exact source sentence(s), no paraphrase
    source_doc_id: str            # matches sidecar doc_id
    source_chunk_id: str | None = None  # None here — Deliverable 5 backfills
    distilled_context: str | None = Field(
        default=None,
        description=(
            "Ceremony-stripped substance summary. Populated by distill.py post "
            "facts extraction. Compliance reasons from this AND verifies "
            "against verbatim_quote."
        ),
    )

    @field_validator("actor_role")
    @classmethod
    def _actor_role_must_match_pattern(cls, v: str) -> str:
        if not ROLE_ID_PATTERN.match(v):
            raise ValueError(f"actor_role {v!r} is not a valid snake_case role id")
        return v

    @field_validator("verbatim_quote")
    @classmethod
    def _verbatim_quote_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("verbatim_quote must be non-empty")
        return v

    @field_validator("action")
    @classmethod
    def _action_word_count_limit(cls, v: str) -> str:
        if len(v.split()) > 15:
            raise ValueError(f"action exceeds 15 words: {v!r}")
        return v


class ActorRole(BaseModel):
    """Catalogue entry for one actor role discovered in a case."""

    role_id: str              # matches ROLE_ID_PATTERN
    label_fr: str             # human-readable French label, e.g. "Notaire rédacteur"
    grounding_note: str       # one sentence: legal or contextual justification
    first_seen_doc_id: str    # first doc where this role was observed
    fact_count: int           # count of facts using this role in the case
    confidence: Literal["high", "medium", "low"]

    @field_validator("role_id")
    @classmethod
    def _role_id_must_match_pattern(cls, v: str) -> str:
        if not ROLE_ID_PATTERN.match(v):
            raise ValueError(f"role_id {v!r} is not a valid snake_case role id")
        return v


class RoleAmbiguity(BaseModel):
    """One unresolved role assignment surfaced for later (interactive) resolution."""

    ambiguity_id: str                # deterministic: f"{doc_id}-a{index:03d}"
    source_doc_id: str
    verbatim_quote: str              # source text triggering the ambiguity
    candidate_role_ids: list[str]    # 2+ candidates the LLM couldn't pick between
    note: str                        # LLM's short explanation of the ambiguity
    fact_ids: list[str]              # facts that used one of the candidates provisionally


# ========== defensive JSON parsing ==========

def _parse_llm_json(raw: str, doc_id: str) -> dict | None:
    """Parse the extractor's raw response into a dict.

    Strips ```json fences and uses json.JSONDecoder().raw_decode() so
    trailing prose commentary (which some OpenRouter models emit despite
    instructions) doesn't break the parse. Returns None and logs a warning
    naming doc_id on any failure — callers treat that as "no facts/roles/
    ambiguities extracted from this document".
    """
    text = strip_json_fences(raw)


    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        log.warning(
            "facts: JSON parse failed for doc_id=%s: %s\nraw response: %s", doc_id, e, raw
        )
        return None

    if not isinstance(obj, dict):
        log.warning("facts: top-level JSON is not an object for doc_id=%s", doc_id)
        return None

    return obj


# ========== per-document extraction ==========

def _build_facts(raw_facts: list, doc_id: str) -> list[Fact]:
    """Validate each raw fact dict; skip (with a warning) any that fail."""
    facts: list[Fact] = []
    for i, item in enumerate(raw_facts, start=1):
        fact_id = f"{doc_id}-f{i:03d}"
        try:
            fact = Fact(
                fact_id=fact_id,
                date=item.get("date"),
                actor_role=item["actor_role"],
                action=item["action"],
                target=item.get("target"),
                verbatim_quote=item["verbatim_quote"],
                source_doc_id=doc_id,
                source_chunk_id=None,
            )
        except (ValidationError, KeyError) as e:
            log.warning("facts: skipping invalid fact %s for doc_id=%s: %s", fact_id, doc_id, e)
            continue
        facts.append(fact)
    return facts


def _build_roles(raw_roles: list, doc_id: str, facts: list[Fact]) -> list[ActorRole]:
    """Validate each raw role dict; skip (with a warning) any that fail."""
    roles: list[ActorRole] = []
    for item in raw_roles:
        role_id = item.get("role_id", "")
        fact_count = sum(1 for f in facts if f.actor_role == role_id)
        try:
            role = ActorRole(
                role_id=role_id,
                label_fr=item["label_fr"],
                grounding_note=item["grounding_note"],
                first_seen_doc_id=doc_id,
                fact_count=fact_count,
                confidence=item["confidence"],
            )
        except (ValidationError, KeyError) as e:
            log.warning("facts: skipping invalid role %r for doc_id=%s: %s", role_id, doc_id, e)
            continue
        roles.append(role)
    return roles


def _build_ambiguities(raw_ambiguities: list, doc_id: str, facts: list[Fact]) -> list[RoleAmbiguity]:
    """Validate each raw ambiguity dict; skip (with a warning) any that fail.

    fact_ids links to every validated Fact in this document whose actor_role
    is one of the ambiguity's candidate_role_ids — the provisional fact(s)
    the extractor built using one of the disputed candidates.
    """
    ambiguities: list[RoleAmbiguity] = []
    for i, item in enumerate(raw_ambiguities, start=1):
        ambiguity_id = f"{doc_id}-a{i:03d}"
        try:
            candidate_role_ids = item["candidate_role_ids"]
            fact_ids = [f.fact_id for f in facts if f.actor_role in candidate_role_ids]
            ambiguity = RoleAmbiguity(
                ambiguity_id=ambiguity_id,
                source_doc_id=doc_id,
                verbatim_quote=item["verbatim_quote"],
                candidate_role_ids=candidate_role_ids,
                note=item["note"],
                fact_ids=fact_ids,
            )
        except (ValidationError, KeyError) as e:
            log.warning(
                "facts: skipping invalid ambiguity %s for doc_id=%s: %s", ambiguity_id, doc_id, e
            )
            continue
        ambiguities.append(ambiguity)
    return ambiguities


def extract_facts_and_roles(
    md_path: Path, doc_id: str, _parse_failed: list[str] | None = None,
) -> tuple[list[Fact], list[ActorRole], list[RoleAmbiguity]]:
    """One FACT_EXTRACTOR_MODEL_ID call per document: parse its verbatim .md
    transcript into validated Facts, discovered ActorRoles, and any
    RoleAmbiguities.

    `_parse_failed` is a private bookkeeping list: if the LLM response fails
    JSON parsing, doc_id is appended to it (when provided) so a case-level
    caller can distinguish "parse failed" from "document legitimately has no
    extractable facts" — both otherwise return the same ([], [], []).
    External callers can omit it.
    """
    md_text = md_path.read_text(encoding="utf-8")

    client = get_anthropic_client()
    response = client.chat.completions.create(
        model=FACT_EXTRACTOR_MODEL_ID,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": _FACT_EXTRACTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Voici la transcription verbatim du document (doc_id={doc_id}) :\n\n"
                    f"{md_text}\n\n"
                    "Extrais les faits, les rôles, et les ambiguïtés au format JSON strict."
                ),
            },
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    parsed = _parse_llm_json(raw, doc_id)
    if parsed is None:
        if _parse_failed is not None:
            _parse_failed.append(doc_id)
        return [], [], []

    facts = _build_facts(parsed.get("facts", []), doc_id)
    roles = _build_roles(parsed.get("roles_discovered", []), doc_id, facts)
    ambiguities = _build_ambiguities(parsed.get("ambiguities", []), doc_id, facts)
    return facts, roles, ambiguities


# ========== idempotent persistence ==========

def _write_case_jsonl(case_id: str, docs_processed: set[str], new_items: list,
                      filename: str, model, doc_field: str, sort_key) -> Path:
    """Rewrite one per-case JSONL artifact, replacing only reprocessed docs.

    Args:
        case_id: case directory under DOSSIER_DIR.
        docs_processed: source_doc_ids whose existing rows are superseded.
        new_items: pydantic instances to write for those docs.
        filename: artifact name within the case directory.
        model: pydantic class used to parse existing rows.
        doc_field: attribute naming an item's source document.
        sort_key: total ordering applied before writing, so the file is
            byte-stable across runs and the matrix stays idempotent.

    Returns the path written.
    """
    path = DOSSIER_DIR / case_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = model.model_validate_json(line)
            if getattr(item, doc_field) not in docs_processed:
                kept.append(item)

    combined = sorted(kept + new_items, key=sort_key)

    with path.open("w", encoding="utf-8") as f:
        for item in combined:
            f.write(item.model_dump_json() + "\n")
    return path


# The three artifacts differ only in these four values. Note actor_roles
# keys its doc on first_seen_doc_id rather than source_doc_id, and sorts on
# a single field where the other two sort on a tuple — which is why the
# structural clone detector matches only two of the three.
_CASE_ARTIFACTS = {
    "facts": ("facts.jsonl", Fact, "source_doc_id",
              lambda f: (f.source_doc_id, f.fact_id)),
    "actor_roles": ("actor_roles.jsonl", ActorRole, "first_seen_doc_id",
                    lambda r: r.role_id),
    "role_ambiguities": ("role_ambiguities.jsonl", RoleAmbiguity, "source_doc_id",
                         lambda a: (a.source_doc_id, a.ambiguity_id)),
}


def _write_facts_jsonl(case_id: str, docs_processed: set[str], new_facts: list[Fact]) -> Path:
    return _write_case_jsonl(case_id, docs_processed, new_facts, *_CASE_ARTIFACTS["facts"])


def _write_actor_roles_jsonl(case_id: str, docs_processed: set[str], new_roles: list[ActorRole]) -> Path:
    return _write_case_jsonl(case_id, docs_processed, new_roles, *_CASE_ARTIFACTS["actor_roles"])


def _write_role_ambiguities_jsonl(
    case_id: str, docs_processed: set[str], new_ambiguities: list[RoleAmbiguity]
) -> Path:
    return _write_case_jsonl(
        case_id, docs_processed, new_ambiguities, *_CASE_ARTIFACTS["role_ambiguities"]
    )


# ========== case aggregation ==========

def extract_case_facts(case_id: str) -> dict:
    """Extract facts, discovered roles, and ambiguities from every extracted
    document in a case; write the three JSONL sidecars and return a summary.

    Walks every .md under DOSSIER_DIR/<case_id>/extracted/, calling
    extract_facts_and_roles(doc_path, doc_id) once per document. Aggregates
    across all documents:
      - facts.jsonl: one Fact per line, sorted by (source_doc_id, fact_id).
      - actor_roles.jsonl: one ActorRole per unique role_id in the case —
        first-seen label_fr/grounding_note win on conflict (logged), fact_count
        is recomputed from the final aggregated facts list.
      - role_ambiguities.jsonl: one RoleAmbiguity per line.
    All three files are written idempotently: existing entries for docs
    processed this run are dropped and replaced before rewriting.
    """
    t0 = time.time()

    extracted_dir = DOSSIER_DIR / case_id / "extracted"
    md_paths = sorted(extracted_dir.rglob("*.md"))
    docs_processed = {p.stem for p in md_paths}

    all_facts: list[Fact] = []
    all_ambiguities: list[RoleAmbiguity] = []
    role_catalogue: dict[str, ActorRole] = {}
    parse_failed_docs: list[str] = []

    for md_path in md_paths:
        doc_id = md_path.stem
        doc_facts, doc_roles, doc_ambiguities = extract_facts_and_roles(
            md_path, doc_id, _parse_failed=parse_failed_docs
        )
        all_facts.extend(doc_facts)
        all_ambiguities.extend(doc_ambiguities)

        for role in doc_roles:
            existing = role_catalogue.get(role.role_id)
            if existing is None:
                role_catalogue[role.role_id] = role
            elif existing.label_fr != role.label_fr or existing.grounding_note != role.grounding_note:
                log.warning(
                    "facts: conflicting label/grounding for role_id=%s "
                    "(first seen in %s, conflict in %s) — keeping first-seen version",
                    role.role_id, existing.first_seen_doc_id, doc_id,
                )

    final_roles = [
        role.model_copy(
            update={"fact_count": sum(1 for f in all_facts if f.actor_role == role_id)}
        )
        for role_id, role in role_catalogue.items()
    ]

    _write_facts_jsonl(case_id, docs_processed, all_facts)
    _write_actor_roles_jsonl(case_id, docs_processed, final_roles)
    _write_role_ambiguities_jsonl(case_id, docs_processed, all_ambiguities)

    elapsed = time.time() - t0

    if parse_failed_docs:
        log.info("facts: parse failed for docs: %s", parse_failed_docs)

    return {
        "case_id": case_id,
        "docs_processed": len(docs_processed),
        "facts_extracted": len(all_facts),
        "unique_roles": len(final_roles),
        "ambiguities": len(all_ambiguities),
        "parse_failed_docs": len(parse_failed_docs),
        "elapsed": elapsed,
    }
