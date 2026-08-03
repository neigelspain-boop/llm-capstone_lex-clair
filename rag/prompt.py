"""Prompt templates for the answer synthesis step.

Static file, no logic beyond string formatting. Kept separate from
generate.py so peer reviewers can inspect and iterate the prompt
without touching the API wrapper, and so Day 6 UI code can show the
prompt for debugging without importing the OpenAI client.

Templates come in matched pairs, one per kind of context (ADR #62):
  - ANSWER_TEMPLATE / ENTRY_TEMPLATE          — statute articles
  - DOSSIER_ANSWER_TEMPLATE / DOSSIER_ENTRY_TEMPLATE — pieces of a case file
  - BLENDED_ANSWER_TEMPLATE                   — both at once

`build` picks by inspecting the chunks, so no caller has to pass the scope
and a blended retrieval is handled correctly by construction.

The split exists because the statute template is actively wrong for case
documents. It labels the context "ARTICLES DE LOI PERTINENTS" and demands
citations formatted "art. {num} du {source_label} ({url})" — fields a dossier
chunk does not have. Asked "quel est le litige dans ce dossier ?" over case
documents it produced a hedged summary citing "art. non spécifié" at
"https://data/dossier/...", which is both useless and a fabricated URL. A
question about a case file wants names, dates, amounts and the document each
came from; the dossier template asks for exactly that.

Grounding discipline (baked into every template):
  - MUST cite its sources — article URL for statute, document name for a piece
  - MUST refuse if context doesn't cover the question
  - MUST NOT invent article numbers, URLs, names, dates or amounts
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


DOSSIER_ANSWER_TEMPLATE = """
Tu es un assistant juridique qui aide des non-juristes à comprendre LEUR PROPRE
dossier de succession. On te donne des pièces de ce dossier (courriers, actes,
conventions, factures) et une question portant sur les faits de l'affaire.

QUESTION:
{query}

PIÈCES DU DOSSIER:
{context}

RÈGLES:
- Réponds à partir des PIÈCES, pas de généralités juridiques. Une réponse qui
  décrirait n'importe quelle succession est une mauvaise réponse.
- Sois CONCRET : nomme les personnes telles qu'elles apparaissent dans les
  pièces, donne les dates, les montants et les références exacts. Écris par
  exemple « Mme X a mis en demeure Maître Y le 30 juin 2026 », jamais « une
  héritière a relancé le notaire ».
- Pour chaque affirmation, indique la pièce d'où elle vient, au format
  "(pièce : {{num}}, p. {{page}})". N'utilise JAMAIS le format "art. ... du ..."
  ni d'URL pour une pièce de dossier : ce sont des documents privés, pas des
  articles de loi.
- N'invente rien : ni nom, ni date, ni montant, ni document. Si les pièces ne
  permettent pas de répondre, dis-le explicitement et indique ce qui manque.
- Structure ta réponse ainsi:
    1. Réponse directe et brève (2-3 phrases), qui nomme les parties et l'objet
       du litige
    2. Les faits établis, dans l'ordre chronologique, chacun rattaché à sa pièce
    3. Ce que les pièces ne permettent pas d'établir

RÉPONSE:
""".strip()


BLENDED_ANSWER_TEMPLATE = """
Tu es un assistant juridique qui aide des non-juristes à comprendre leur dossier
de succession. On te donne À LA FOIS des articles de loi et des pièces du
dossier de l'utilisateur.

QUESTION:
{query}

CONTEXTE (articles de loi et pièces du dossier):
{context}

RÈGLES:
- Distingue toujours ce qui vient de la LOI de ce qui vient du DOSSIER.
- Pour un article de loi, cite "art. {{num}} du {{source_label}} ({{url}})".
  Pour une pièce du dossier, cite "(pièce : {{num}}, p. {{page}})" — jamais
  d'URL, ce sont des documents privés.
- Sois concret sur les faits : nomme les personnes, les dates et les montants
  tels qu'ils figurent dans les pièces.
- N'invente rien : ni article, ni URL, ni nom, ni date, ni montant.
- Structure ta réponse ainsi:
    1. Réponse directe et brève (2-3 phrases)
    2. Ce que disent les pièces du dossier
    3. Ce que dit la loi applicable
    4. Limites de la réponse

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


DOSSIER_ENTRY_TEMPLATE = """
[chunk_id: {chunk_id}]
Pièce: {num}
Page: {page}

{texte}
""".strip()


# chunk_id prefix written by ingestion.dossier.index for every case document.
DOSSIER_CHUNK_PREFIX = "dossier-"


def _is_dossier(chunk: dict) -> bool:
    return str(chunk.get("chunk_id", "")).startswith(DOSSIER_CHUNK_PREFIX)


def _page_of(chunk: dict) -> str:
    """Page label for a dossier chunk.

    Dossier chunks carry the page in `titre` ("Page 3") because chunking is
    page-aware; fall back to the section path, then to a dash, rather than
    printing "None" into a citation the model is told to reproduce verbatim.
    """
    return str(chunk.get("titre") or chunk.get("section_path") or "—")


def build(query: str, chunks: list[dict]) -> str:
    """Build the full answer prompt from a query and top-k chunks.

    The template is chosen from the chunks themselves rather than from a
    caller-supplied scope: retrieval is what actually decides whether the
    model is looking at law, at a case file, or at both, and reading it here
    keeps the two from disagreeing. Order is preserved (rerank order =
    highest relevance first).
    """
    has_dossier = any(_is_dossier(c) for c in chunks)
    has_statute = any(not _is_dossier(c) for c in chunks)

    entries = [
        DOSSIER_ENTRY_TEMPLATE.format(**{**c, "page": _page_of(c)})
        if _is_dossier(c)
        else ENTRY_TEMPLATE.format(**c)
        for c in chunks
    ]
    context = "\n\n".join(entries)

    if has_dossier and has_statute:
        template = BLENDED_ANSWER_TEMPLATE
    elif has_dossier:
        template = DOSSIER_ANSWER_TEMPLATE
    else:
        template = ANSWER_TEMPLATE
    return template.format(query=query, context=context)


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
    