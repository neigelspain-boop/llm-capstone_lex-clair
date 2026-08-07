"""Thin client for the local Ollama instance (qwen3:14b), mirroring
ingestion.clients.get_openrouter_client's role as the single place that
knows how to reach an LLM provider — here, a local one instead of
OpenRouter, so no API key or cost tracking is involved.

Every call in this module returns a *judgment*, never asks for one framed as
"suggest a fix" or "write the code" — see the report-only guardrail in the
project plan. Callers pass instructions that end in a citation+verdict ask;
nothing here has a code-writing code path.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import requests

from scripts.local_audit import config

log = logging.getLogger(__name__)


# ========== single call ==========


def call_json(system_prompt: str, user_prompt: str, temperature: float = 0.0,
              think: bool | None = None, model: str | None = None) -> dict | list | None:
    """POST to Ollama's /api/chat in JSON mode, return the parsed body.

    think overrides config.OLLAMA_THINK for this call — most passes ask a
    simple lookup/classification question where the reasoning trace just
    burns time (see config.OLLAMA_THINK's own docstring), but a pass doing
    real comparative judgment (duplication.py, weighing whether two
    functions' divergence looks intentional) benefits enough from it to be
    worth the extra latency.

    model overrides config.OLLAMA_MODEL — see config.OLLAMA_MODEL_JUDGMENT's
    docstring for which passes use the larger model and why.

    Returns None (with a logged warning) on transport failure or
    unparseable JSON — callers must treat None as "no verdict", never as a
    false/negative answer.
    """
    try:
        resp = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL if model is None else model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "think": config.OLLAMA_THINK if think is None else think,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": temperature,
                    "num_ctx": config.OLLAMA_NUM_CTX,
                },
            },
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.warning("ollama_client: request failed: %s", e)
        return None

    content = resp.json().get("message", {}).get("content", "")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text)
            return obj
        except json.JSONDecodeError:
            log.warning("ollama_client: unparseable JSON response: %s", text[:300])
            return None


# ========== self-consistency (majority vote) ==========


def call_self_consistency(system_prompt: str, user_prompt: str, verdict_key: str,
                           n: int = config.SELF_CONSISTENCY_RUNS,
                           temperature: float = config.SELF_CONSISTENCY_TEMPERATURE,
                           agree_threshold: int = config.SELF_CONSISTENCY_AGREE_THRESHOLD,
                           think: bool | None = None,
                           model: str | None = None,
                           ) -> tuple[dict | None, dict]:
    """Run the same judgment call `n` times and only return a verdict if at
    least `agree_threshold` runs agree on `verdict_key`'s value. This is the
    "precision over recall" tactic that only makes sense at zero marginal
    cost (self-consistency.md concern in the harness plan) — a single cloud
    call would never be repeated 3x just to reduce false positives.

    Returns (majority_result_or_None, meta) where meta = {"runs": n, "agree": k,
    "votes": {...}} for the self_consistency field on the finding.
    """
    results = []
    for _ in range(n):
        r = call_json(system_prompt, user_prompt, temperature=temperature, think=think, model=model)
        if r is not None and verdict_key in r:
            results.append(r)

    votes = Counter(str(r[verdict_key]) for r in results)
    meta = {"runs": n, "agree": 0, "votes": dict(votes)}
    if not votes:
        return None, meta

    top_value, top_count = votes.most_common(1)[0]
    meta["agree"] = top_count
    if top_count < agree_threshold:
        return None, meta

    # return the first result whose verdict matches the majority value
    for r in results:
        if str(r[verdict_key]) == top_value:
            return r, meta
    return None, meta
