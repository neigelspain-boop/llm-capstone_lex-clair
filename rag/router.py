"""Query router: classifies user query intent and picks a retrieval source_scope.

Public surface: `route_query(query, active_case_id=None)` returns a
`RouteDecision` — never raises. B1 (ADR #41) shipped `source_scope`
filtering on `HybridRetriever.search()` but every caller had to hardcode
the scope string. This module automates that choice: a single Haiku 4.5
classification call decides the query's intent, then `route_query` maps
intent + `active_case_id` to a `source_scope` deterministically in Python
(the model never sees or chooses `source_scope` directly).

Call pattern mirrors `ingestion/dossier/gate.py::_call_verifier` /
`_parse_verifier_response` (plain `chat.completions.create`, hand-parsed
JSON text with code-fence stripping) rather than `rag/rewrite.py`'s
`.beta.chat.completions.parse()` structured-output path — a router must
degrade to a safe default on a malformed response instead of raising, and
that failure mode is exactly what manual JSON parsing exists to catch.

Cost: ~150 input + <50 output tokens per query on Haiku 4.5, well under
$0.001/query. Routed through OpenRouter (ADR #40).
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from ingestion.clients import get_openrouter_client, strip_json_fences

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROUTER_MODEL_ID = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 256

_CLASSIFIER_INTENTS = {"statute_lookup", "case_factual", "gap_analysis", "other"}
_CONFIDENCE_LEVELS = {"high", "medium", "low"}

ROUTER_SYSTEM_PROMPT = """Tu es un routeur de requêtes pour un système RAG juridique français.

On te donne une requête utilisateur et éventuellement un identifiant de dossier actif. Tu dois classer l'intention :

- "statute_lookup" : question sur le droit en général, sans référence à un cas spécifique.
- "case_factual" : question sur les faits d'un dossier spécifique (qui a fait quoi, quand, etc.).
- "gap_analysis" : question sur la conformité d'un dossier au droit (obligation respectée ? manquement ?).
- "other" : autre (small talk, ambigu, hors-sujet).

Renvoie un objet JSON strict, sans texte autour :
{
  "intent": "<un des 4>",
  "confidence": "high" | "medium" | "low",
  "rationale": "<une phrase courte>"
}
"""


# ========== structured result ==========

class RouteDecision(BaseModel):
    """Router output: classified intent, resolved source_scope, confidence, rationale.

    intent="override" means the caller passed an explicit source_scope and
    the router was skipped (see rag/flow.py::run()); it never comes from
    the Haiku classifier itself, whose output is restricted to the other
    4 values (enforced by `_parse_router_response`).
    """
    intent: Literal["statute_lookup", "case_factual", "gap_analysis", "other", "override"]
    source_scope: str
    confidence: Literal["high", "medium", "low"]
    rationale: str


# ========== response parsing ==========

def _parse_router_response(raw: str) -> dict:
    """Parse the classifier's raw JSON text into {intent, confidence, rationale}.

    Mirrors ingestion/dossier/gate.py::_parse_verifier_response: strips
    ```json ... ``` fences if the model ignores the "JSON only" instruction,
    then json.loads, then validates shape. Raises ValueError on any parse
    or shape failure; the caller catches it and returns a safe default.
    """
    text = strip_json_fences(raw)


    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}") from e

    intent = data.get("intent")
    if intent not in _CLASSIFIER_INTENTS:
        raise ValueError(f"unknown or missing intent: {intent!r}")

    confidence = data.get("confidence")
    if confidence not in _CONFIDENCE_LEVELS:
        confidence = "low"

    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)

    return {"intent": intent, "confidence": confidence, "rationale": rationale}


# ========== public entry point ==========

def route_query(query: str, active_case_id: str | None = None) -> RouteDecision:
    """Classify query intent and return source_scope for retrieval.

    Deterministic scope mapping from (intent, active_case_id):

    With an active case (ADR #66) — the dossier is never out of scope, so no
    intent may resolve to bare "statute". The router decides emphasis, not
    whether the dossier is visible:
      - statute_lookup  -> "case+statute:{case_id}"
      - case_factual    -> "case:{case_id}"   (pure fact lookup, no statute)
      - gap_analysis    -> "case+statute:{case_id}"
      - other           -> "case+statute:{case_id}"

    Without an active case — unchanged from ADR #42:
      - statute_lookup            -> "statute"
      - case_factual, no case_id  -> downgrade to "statute", confidence="low"
      - gap_analysis, no case_id  -> downgrade to "statute", confidence="low"
      - other                     -> "statute" (safest fallback)

    Two behaviours changed here. `statute_lookup` and `other` used to resolve
    to "statute" even with a case selected, so a general legal question — or
    anything the classifier found ambiguous, which includes "is a dossier
    attached?" — answered from the statute corpus while the user was looking
    at a dossier. And `gap_analysis` used to resolve to "blended", which is
    not a filter at all: it reaches every case in the index, including other
    clients'. "case+statute:{id}" fixes both, and is the reason that scope
    exists.

    Never raises: any call or parse failure logs a warning and returns a
    safe default (intent="other", confidence="low") — scoped to the active
    case when there is one, so a router outage degrades to the right corpus
    rather than silently dropping the dossier.
    """
    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=ROUTER_MODEL_ID,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Requête : {query}\nDossier actif : {active_case_id or 'aucun'}",
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        parsed = _parse_router_response(response.choices[0].message.content or "")
    except Exception as e:
        log.warning("router: call/parse failed (%s); falling back to safe default", e)
        # "Safe" now means the active case stays in scope. Falling back to
        # bare "statute" would make a router outage look like a dossier with
        # nothing in it — the failure mode this ADR #66 mapping exists to
        # remove. With no active case the old fallback is still correct.
        return RouteDecision(
            intent="other",
            source_scope=(
                f"case+statute:{active_case_id}" if active_case_id else "statute"
            ),
            confidence="low",
            rationale=f"router failure: {e}",
        )

    intent = parsed["intent"]
    confidence = parsed["confidence"]
    rationale = parsed["rationale"]

    if intent == "statute_lookup":
        source_scope = f"case+statute:{active_case_id}" if active_case_id else "statute"
    elif intent == "case_factual":
        if active_case_id:
            source_scope = f"case:{active_case_id}"
        else:
            source_scope, confidence = "statute", "low"
            rationale = f"case_factual sans dossier actif — repli statute. {rationale}"
    elif intent == "gap_analysis":
        if active_case_id:
            source_scope = f"case+statute:{active_case_id}"
        else:
            source_scope, confidence = "statute", "low"
            rationale = f"gap_analysis sans dossier actif — repli statute. {rationale}"
    else:  # "other"
        source_scope = f"case+statute:{active_case_id}" if active_case_id else "statute"

    return RouteDecision(
        intent=intent, source_scope=source_scope, confidence=confidence, rationale=rationale,
    )


if __name__ == "__main__":
    # Standalone smoke: representative queries across all 4 intents.
    samples = [
        ("qu'est-ce que le quasi-usufruit ?", None),
        ("quand le notaire a-t-il envoye la mise en demeure ?", "demo"),
        ("le notaire a-t-il respecte l'obligation de l'article 815-9 ?", "demo"),
        ("le notaire a-t-il respecte l'obligation de l'article 815-9 ?", None),
        ("bonjour, comment ca va ?", None),
    ]
    for q, case_id in samples:
        d = route_query(q, active_case_id=case_id)
        print(f"\n  IN : {q!r} (case={case_id})")
        print(f"  OUT: intent={d.intent} scope={d.source_scope} conf={d.confidence}")
        print(f"       {d.rationale}")
