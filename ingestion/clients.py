"""Centralized LLM client factory and response plumbing for lex-clair.

Single place for API-key handling so credential wiring doesn't get
duplicated across ingestion modules, plus the small response-shaping
helpers every LLM caller needs regardless of plane (ADR #67).

get_openrouter_client() is the canonical factory (ADR #40): all LLM calls
route through OpenRouter's OpenAI-compatible endpoint for a single credit
pool and usage dashboard. Callers must use fully-qualified OpenRouter model
slugs (e.g. "openai/gpt-4o-mini", "anthropic/claude-haiku-4.5",
"mistralai/mistral-small-2603").

get_anthropic_client() is the last surviving deprecated alias, kept because
Plane Ib's extract/gate/facts modules still call it and tests patch it by
name in twelve places. get_openai_client() and get_mistral_client() were
removed once the transition they existed for was complete and neither had a
caller left anywhere in the tree.
"""
from __future__ import annotations

import json
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ========== lazy client singleton ==========

_openrouter_client = None


def get_openrouter_client():
    """Lazy-singleton OpenRouter client (OPENROUTER_API_KEY from environment)."""
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        _openrouter_client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


# ========== deprecated provider-named alias ==========
# Kept so existing callers keep working during the transition (ADR #40).

def get_anthropic_client():
    """Deprecated: use get_openrouter_client()."""
    warnings.warn(
        "get_anthropic_client() is deprecated; use get_openrouter_client()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_openrouter_client()


# ========== response plumbing (ADR #67) ==========


def strip_json_fences(raw: str | None) -> str:
    """Strip a markdown code fence off an LLM response and return the payload.

    Args:
        raw: the model's message content, possibly None.

    Returns the fence-free, whitespace-trimmed text. Never raises, never
    logs, never decodes — decoding and the policy for a failed decode belong
    to the caller, and deliberately differ across this codebase: gate.py
    raises ValueError so its caller can record status="parse_failed",
    facts.py returns None meaning "nothing extracted", compliance.py returns
    [] after attempting partial recovery. Those are caller contracts, not
    accidents, and unifying them would be a silent behaviour change no test
    covers.

    So this handles only the part that was genuinely identical at eight call
    sites across three planes: the four lines that undo ```json fencing.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def parse_json_list(raw: str, context: str, source: str,
                    log_target=None) -> list[dict] | None:
    """Parse an LLM response expected to hold a JSON array.

    Args:
        raw: the model's raw message content.
        context: label for the log line, e.g. "mentions" or "resolve".
        source: identifier of what was being parsed, e.g. a doc_id or role_id.
        log_target: logger to warn on; defaults to this module's.

    Returns the decoded list, or None on a fence/decode/shape failure, having
    logged a warning naming `source`. Uses raw_decode so trailing prose
    commentary — which some OpenRouter models emit despite instructions —
    doesn't break an otherwise valid parse.

    Callers must not cache a None result: it means "no verdict", not "empty".
    """
    logger = log_target or log
    text = strip_json_fences(raw)
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as e:
        logger.warning("%s: JSON parse failed for %s: %s\nraw response: %s",
                       context, source, e, raw)
        return None
    if not isinstance(obj, list):
        logger.warning("%s: top-level JSON is not an array for %s", context, source)
        return None
    return obj
