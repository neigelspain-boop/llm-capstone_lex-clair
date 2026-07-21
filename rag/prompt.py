"""Prompt templates for the answer synthesis step.

Static file, no logic beyond string formatting. Kept separate from
generate.py so peer reviewers can inspect and iterate the prompt
without touching the API wrapper, and so Day 6 UI code can show the
prompt for debugging without importing the OpenAI client.

Two templates:
  - ANSWER_TEMPLATE: the system-level instructions + question + context
  - ENTRY_TEMPLATE: format for a single retrieved chunk inside the context

Grounding discipline (baked into ANSWER_TEMPLATE):
  - MUST cite article URLs, not just article numbers
  - MUST refuse if context doesn't cover the question
  - MUST NOT invent article numbers or URLs
"""
from __future__ import annotations


ANSWER_TEMPLATE = """
Tu es un assistant juridique qui répond à des questions de succession posées
par des non-juristes en France (petits-enfants confrontés à un litige successoral,
notamment sur des conventions de quasi-usufruit).

QUESTION:
{query}

ARTICLES DE LOI PERTINENTS:
{context}

RÈGLES:
- Réponds en français courant, sans jargon inutile. Explique les termes juridiques
  quand tu les emploies.
- Cite systématiquement les articles utilisés avec leur URL complète. Format:
  "art. {{num}} du {{source_label}} ({{url}})".
- Si les articles fournis ne couvrent pas la question, dis-le explicitement.
  N'invente jamais un article, un numéro ou une URL.
- Structure ta réponse ainsi:
    1. Réponse directe et brève (2-3 phrases)
    2. Explication détaillée avec citations
    3. Limites de la réponse si des zones grises subsistent

RÉPONSE:
""".strip()


ENTRY_TEMPLATE = """
[chunk_id: {chunk_id}]
Source: {source_label}
Article: {num}
Titre: {titre}
Section: {section_path}
URL: {url}

{texte}
""".strip()


def build(query: str, chunks: list[dict]) -> str:
    """Build the full answer prompt from a query and top-k chunks.

    Chunks are formatted with ENTRY_TEMPLATE and joined with double
    newlines, then inserted into ANSWER_TEMPLATE. Order is preserved
    (rerank order = highest relevance first).
    """
    context = "\n\n".join(ENTRY_TEMPLATE.format(**c) for c in chunks)
    return ANSWER_TEMPLATE.format(query=query, context=context)


if __name__ == "__main__":
    # Standalone smoke: build a prompt from real retrieve+rerank output.
    from rag.retrieve import retrieve
    from rag.rerank import rerank

    query = "Qu'est-ce que le quasi-usufruit ?"
    hits = retrieve(query, k=20)
    top = rerank(query, hits, k=5)
    prompt = build(query, top)

    print(f"prompt length: {len(prompt)} chars, {len(prompt.split())} words\n")
    print(prompt[:1200])
    print("\n... [truncated] ...")
    print(prompt[-400:])
    