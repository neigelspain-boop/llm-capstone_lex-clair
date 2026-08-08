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


# ========== cost catalog (ADR #68) ==========

# USD per MILLION tokens (input, output), keyed by OpenRouter model slug.
# The single source of these numbers, which previously existed in three
# incompatible representations across five files: per-million pairs
# (eval/llm_eval.py COST_PER_MTOKEN), per-token pairs (_EST_*_USD_PER_TOKEN
# in distill/mentions/resolve/compliance), and a catalog dict
# (rag/generate.py ANSWER_MODELS). All three agreed on every shared model —
# they encode CLAUDE.md's model palette — so unifying them changed no number.
# Agreement was luck, not design: nothing kept them in step.
#
# Per-million, not per-token, because that is how every provider publishes
# and how the palette table reads, so a rate can be checked against a
# pricing page without arithmetic.
MODEL_RATES_USD_PER_M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini":           (0.15, 0.60),
    "anthropic/claude-haiku-4.5":   (1.00, 5.00),
    "anthropic/claude-opus-4.7":    (15.0, 75.0),
    "moonshotai/kimi-k3":           (3.00, 15.0),
    "mistralai/mistral-small-2603": (0.15, 0.60),
}


def estimate_cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD for a call, from MODEL_RATES_USD_PER_M.

    Raises KeyError for an unlisted model. Deliberately not a silent 0.0:
    ADR #45 records that flow.py::_compute_cost was deleted precisely
    because it zeroed cost for every non-default model, and a dry-run
    preview that under-reports is worse than one that fails.

    Estimates only. Real calls should prefer the provider's own
    `usage.cost` — see extract_usage.
    """
    rate_in, rate_out = MODEL_RATES_USD_PER_M[model_id]
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1_000_000


def extract_usage(response, model_id: str | None = None) -> dict:
    """Normalize a response's token usage, preferring the provider's cost.

    Args:
        response: an OpenAI-compatible response object, possibly lacking usage.
        model_id: OpenRouter slug. When given and the provider reported no
            cost, an estimate from MODEL_RATES_USD_PER_M fills in and
            "estimated" is True.

    Returns {prompt_tokens, completion_tokens, cost_usd, estimated}. cost_usd
    is None only when the provider reported none and no model_id was passed.

    Reads defensively: OpenRouter's OpenAI-compatible endpoint uses
    prompt_tokens/completion_tokens, the Responses API uses
    input_tokens/output_tokens, and a mocked client may carry neither.
    """
    usage = getattr(response, "usage", None)
    prompt = (getattr(usage, "prompt_tokens", None)
              or getattr(usage, "input_tokens", None) or 0)
    completion = (getattr(usage, "completion_tokens", None)
                  or getattr(usage, "output_tokens", None) or 0)
    cost = getattr(usage, "cost", None) if usage is not None else None

    estimated = False
    if cost is None and model_id is not None:
        cost = estimate_cost_usd(model_id, prompt, completion)
        estimated = True

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cost_usd": cost,
        "estimated": estimated,
    }


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


def recover_json_objects(text: str) -> list[dict]:
    """Complete JSON objects from a response that failed to decode.

    A model whose output is cut off — by a token ceiling or by exhausting the
    context window — leaves valid objects followed by one truncated fragment.
    A straight `json.loads` discards all of them, including the ones that
    closed. This walks the text tracking brace depth and string state
    (honouring `\\` escapes, so a quote inside a string does not end it early)
    and collects every object that opened and closed.

    Objects are grouped by nesting depth and the **shallowest depth that yielded
    any** is returned. That single rule covers both shapes this codebase
    produces without the caller having to know which it is asking about:
    a top-level list `[{...}, {...}]` closes its objects at depth 0, while a
    wrapper `{"obligations": [{...}, {...}` — truncated, so the wrapper never
    closes — has its objects at depth 1. Scanning only depth 0 finds nothing in
    the second case, which is precisely the bug this replaced.

    Lives here rather than in a pass because two planes need it: Plane II's
    compliance call shares one budget between reasoning and output, and Plane
    V's discovery reads a whole page and emits structured output for all of it.
    `ingestion/clients.py` is the one module every plane may import (ADR #67).

    Returns `[]` when nothing closed. The caller decides what that means: for a
    model verdict it is "no verdict", never "no".
    """
    by_depth: dict[int, list[str]] = {}
    depth = 0
    starts: dict[int, int] = {}
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            starts[depth] = i
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            if depth in starts:
                by_depth.setdefault(depth, []).append(text[starts.pop(depth):i + 1])

    for level in sorted(by_depth):
        recovered = []
        for candidate in by_depth[level]:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                recovered.append(parsed)
        if recovered:
            return recovered
    return []


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
