"""The only place Plane V talks to a local model.

Forked from `scripts/local_audit/ollama_client.py` for the reason in ADR #70,
with the request budget threaded through so a pass can report `complete=False`
when it runs out rather than silently truncating.

Two contracts callers must honour, both learned in the original harness:

- **`None` means "no verdict", never "no".** A timeout, a transport error, or
  unparseable output must never be read as a negative answer, and must never be
  cached — the next cycle simply retries. Reading a failure as "no" is how a
  transient blip becomes a permanent wrong finding.
- **Nothing here decides anything on its own.** Every call answers one narrow
  question about one short span. The model never sees the obligation set, never
  sees the corpus, and never chooses which finding to emit; the deterministic
  pass has already done that.

There are no transport retries. Self-consistency exists for *precision*, not
reliability: three samples at a low temperature, and a verdict only when they
agree. Reliability comes from not caching failures.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import requests

from ingestion.clients import recover_json_objects
from investigator import config

log = logging.getLogger(__name__)


# ========== single call ==========


def call_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    think: bool | None = None,
    model: str | None = None,
    timeout: int | None = None,
    num_ctx: int | None = None,
) -> dict | list | None:
    """One structured call. Returns None on any failure — never raises.

    `num_ctx` is per call because one value cannot serve both kinds of work.
    Judgment sends one clause against one quote and wants the default, sized
    against GPU residency; extraction reads a whole page and emits structured
    output for all of it, and a window too small there truncates the response.
    """
    body = {
        "model": model or config.OLLAMA_MODEL_RESCUE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "think": config.OLLAMA_THINK if think is None else think,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_ctx": num_ctx or config.OLLAMA_NUM_CTX},
    }
    try:
        response = requests.post(
            config.OLLAMA_URL, json=body, timeout=timeout or config.OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        text = response.json()["message"]["content"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        log.warning("ollama: call failed (%s) — no verdict", exc)
        return None

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except json.JSONDecodeError:
        pass

    # Truncated: the context ran out mid-response. Salvage the objects that
    # closed rather than discarding the whole answer — that was throwing away
    # real extractions because the last one was cut off.
    recovered = recover_json_objects(text)
    if recovered:
        log.warning(
            "ollama: response truncated — recovered %d complete object(s), "
            "discarded the incomplete tail",
            len(recovered),
        )
        return {"obligations": recovered} if "obligations" in text[:200] else recovered

    log.warning("ollama: unparseable response — no verdict: %s", text[:200])
    return None


# ========== self-consistency ==========


def call_self_consistency(
    system_prompt: str,
    user_prompt: str,
    verdict_key: str,
    n: int = config.SELF_CONSISTENCY_RUNS,
    temperature: float = config.SELF_CONSISTENCY_TEMPERATURE,
    agree_threshold: int = config.SELF_CONSISTENCY_AGREE_THRESHOLD,
    think: bool | None = None,
    model: str | None = None,
) -> tuple[dict | None, dict]:
    """Sample `n` times and return a verdict only if enough samples agree.

    Returns `(verdict, meta)`. A `None` verdict means no consensus or no usable
    response — the caller must treat it as "not adjudicated" and leave the
    deterministic result standing.
    """
    results = []
    for _ in range(n):
        out = call_json(system_prompt, user_prompt, temperature, think, model)
        if isinstance(out, dict) and verdict_key in out:
            results.append(out)

    if not results:
        return None, {"runs": n, "agree": 0, "votes": {}}

    votes = Counter(str(r[verdict_key]) for r in results)
    top, count = votes.most_common(1)[0]
    meta = {"runs": n, "agree": count, "votes": dict(votes), "model": model or config.OLLAMA_MODEL_RESCUE}
    if count < agree_threshold:
        return None, meta
    return next(r for r in results if str(r[verdict_key]) == top), meta


# ========== cached adjudication ==========


def cached_verdict(
    paths,
    pass_name: str,
    prompt_version: str,
    subject: str,
    content_hash: str,
    system_prompt: str,
    user_prompt: str,
    verdict_key: str,
    model: str,
    budget,
    think: bool | None = None,
    agree_threshold: int = config.SELF_CONSISTENCY_AGREE_THRESHOLD,
) -> dict | None:
    """Look up, else spend a call, else give up — the one adjudication path.

    Every pass that asks a model to judge something needs the identical dance:
    check the cache, claim a call from the budget, sample with self-consistency,
    refuse to cache a non-verdict, store, return. Written out per pass it was
    three near-identical copies, which the slimming audit duly flagged.

    Returns None for a cache miss that could not be filled — budget exhausted,
    transport failure, or no consensus. **The caller must read None as "not
    adjudicated" and leave its deterministic result standing**, never as a
    negative verdict.
    """
    from investigator import cache  # local: cache imports config, config does not import this

    cached = cache.get(paths, pass_name, prompt_version, subject, content_hash)
    if cached is not None:
        return cached
    if not budget.take_local():
        return None

    verdict, meta = call_self_consistency(
        system_prompt,
        user_prompt,
        verdict_key=verdict_key,
        agree_threshold=agree_threshold,
        think=think,
        model=model,
    )
    if verdict is None:
        # Never cached: a failed or split sample is not a decided "no", and
        # caching it would turn a transient into a permanent wrong answer.
        return None

    result = dict(verdict)
    result["_self_consistency"] = meta
    result["_model"] = model
    cache.set(paths, pass_name, prompt_version, subject, content_hash, result)
    return result


# ========== model selection ==========

# Sentinel meaning "each pass picks its own default". `--local-llm` sets this,
# so the volume tier stays on the GPU-resident 14B while comparative judgment
# goes to the 30B; `--model X` overrides both.
AUTO = "auto"


def resolve_model(requested: str | None, default: str) -> str:
    """The model a pass should actually call."""
    if requested in (None, AUTO):
        return default
    return requested


# ========== availability ==========


def is_available(timeout: int = 5) -> bool:
    """Whether Ollama is reachable at all.

    Checked once before a pass spends its budget, so an unreachable daemon
    produces one clear log line and `complete=False` rather than a slow parade
    of identical timeouts.
    """
    try:
        requests.get(config.OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False
