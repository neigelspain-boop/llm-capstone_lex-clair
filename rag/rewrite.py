"""Query rewriting: enrich a plain-French query with legal-register vocab.

Public surface: `rewrite(query)` returns an expanded query string containing
legal terms (usufruit, nue-propriété, réserve héréditaire, quasi-usufruit,
etc.) that dense retrieval alone might miss.

Rationale: the target user is a non-lawyer (grandchild facing succession)
who types "ma grand-mère a vendu la maison". Dense retrieval on that raw
query struggles because the corpus is written in legal register. Rewriting
inserts the vocabulary the user *would have used if they knew the law*.

Called retrieval-side only (see flow.py): the answer prompt uses the
ORIGINAL query so the LLM answers what the user actually asked, not what
we translated for the retriever.

Falls back to the original query on any failure: rewriting is an
enhancement, not a correctness path. A dead OpenAI shouldn't take down
the whole flow.

Cost: ~150 input + 40 output tokens per query on gpt-4o-mini ≈ $0.00005
per call. Negligible at Day 4 volumes.

Routed through OpenRouter (ADR #40) — get_openrouter_client() returns an
OpenAI-compatible client pointed at OpenRouter, so the model ID must be the
fully-qualified OpenRouter slug ("openai/gpt-4o-mini").
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ingestion.clients import get_openrouter_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REWRITE_MODEL = "openai/gpt-4o-mini"
TEMPERATURE = 0.3

REWRITE_PROMPT = """
Tu enrichis une question posée en français courant avec le vocabulaire juridique français correspondant,
pour améliorer la recherche dans un corpus de textes de loi (Code civil, Code des assurances, CGI,
ordonnance sur les notaires).

Question originale: {query}

Renvoie une version enrichie qui:
- Garde le sens exact de la question originale
- Ajoute les termes juridiques pertinents (usufruit, nue-propriété, quasi-usufruit,
  succession, réserve héréditaire, quotité disponible, responsabilité du notaire,
  action directe, abus de confiance, etc.) uniquement s'ils sont pertinents
- Ne dépasse pas 40 mots
- Reste une seule phrase ou question, pas une liste
"""


class RewrittenQuery(BaseModel):
    """Structured output for the query rewrite call."""
    enriched_query: str = Field(description="Query enriched with legal vocabulary, ≤40 words.")


def rewrite(query: str) -> str:
    """Rewrite a plain-French query with legal-register vocabulary.

    Returns the enriched query on success. Returns the original query
    unchanged on any failure (empty API response, parse error, network).
    """
    try:
        response = get_openrouter_client().beta.chat.completions.parse(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            response_format=RewrittenQuery,
            temperature=TEMPERATURE,
        )
        enriched = response.choices[0].message.parsed.enriched_query.strip()
        if not enriched:
            log.warning("rewrite returned empty; falling back to original query")
            return query
        return enriched
    except Exception as e:
        log.warning("rewrite failed (%s); falling back to original query", e)
        return query


if __name__ == "__main__":
    # Standalone smoke: 3 representative queries showing before/after.
    samples = [
        "Ma grand-mère a vendu la maison en usufruit",
        "Le notaire a-t-il fait une erreur ?",
        "Qu'est-ce que le quasi-usufruit ?",
    ]
    for q in samples:
        r = rewrite(q)
        print(f"\n  IN : {q}")
        print(f"  OUT: {r}")
        