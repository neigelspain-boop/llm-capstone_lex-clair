"""Answer synthesis via OpenRouter, with a runtime-selectable model catalog.

Public surface: `generate(prompt, model_key=DEFAULT_ANSWER_MODEL_KEY)` returns
(answer_text, usage). usage is a dict with prompt_tokens, completion_tokens,
cost_usd, model_id, model_key — same names Day 5 LLM eval and
monitoring/db.py logging already expect for prompt_tokens/completion_tokens.

ADR #45 (Day C, Deliverable C2): answer generation is no longer pinned to a
single hardcoded model. ANSWER_MODELS catalogs the models a caller may pick
via model_key: a fast default (gpt-4o-mini) and two frontier reasoning-effort
models (Opus 4.7, Kimi K3) for hard queries. rag/flow.py::run() plumbs its
answer_model param through to model_key here; app/streamlit_app.py (C3) will
be the first caller to actually vary it.

Reasoning-effort models pass extra_body={"reasoning": {"effort": ...}} — the
same OpenRouter mechanism rag/compliance.py uses for its Opus call (ADR #43).

Routed through OpenRouter (ADR #40) — get_openrouter_client() returns an
OpenAI-compatible client pointed at OpenRouter, so each catalog model_id must
be the fully-qualified OpenRouter slug (e.g. "openai/gpt-4o-mini").
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv

from ingestion.clients import get_openrouter_client


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ========== answer model catalog (ADR #45) ==========

ANSWER_MODELS: dict[str, dict] = {
    "gpt-4o-mini": {
        "model_id": "openai/gpt-4o-mini",
        "cost_input_per_m": 0.15,
        "cost_output_per_m": 0.60,
        "reasoning_effort": None,
        "max_tokens": 500,
        "label_fr": "⚡ Rapide (GPT-4o-mini)",
    },
    "opus-4.7": {
        "model_id": "anthropic/claude-opus-4.7",
        "cost_input_per_m": 15.0,
        "cost_output_per_m": 75.0,
        "reasoning_effort": "max",
        "max_tokens": 4096,
        "label_fr": "🧠 Réflexion approfondie (Claude Opus 4.7)",
    },
    "kimi-k3": {
        "model_id": "moonshotai/kimi-k3",
        "cost_input_per_m": 3.0,
        "cost_output_per_m": 15.0,
        "reasoning_effort": "max",
        "max_tokens": 4096,
        "label_fr": "🔬 Réflexion approfondie (Kimi K3)",
    },
}
DEFAULT_ANSWER_MODEL_KEY = "gpt-4o-mini"


def generate(prompt: str, model_key: str = DEFAULT_ANSWER_MODEL_KEY) -> tuple[str, dict]:
    """Call the LLM with a prompt and return the answer + usage.

    model_key must be a key in ANSWER_MODELS; raises ValueError otherwise.
    Reasoning-effort models (opus-4.7, kimi-k3) pass
    extra_body={"reasoning": {"effort": cfg["reasoning_effort"]}}.

    Returns:
        (answer_text, usage) where usage is:
        {"prompt_tokens": int, "completion_tokens": int, "cost_usd": float,
         "model_id": str, "model_key": str}
    """
    if model_key not in ANSWER_MODELS:
        raise ValueError(
            f"unknown model_key: {model_key!r}; must be one of {list(ANSWER_MODELS)}"
        )
    cfg = ANSWER_MODELS[model_key]

    kwargs = {
        "model": cfg["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg["max_tokens"],
    }
    if cfg["reasoning_effort"]:
        kwargs["extra_body"] = {"reasoning": {"effort": cfg["reasoning_effort"]}}

    response = get_openrouter_client().chat.completions.create(**kwargs)
    answer = response.choices[0].message.content

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "cost_usd": (
            response.usage.prompt_tokens * cfg["cost_input_per_m"] / 1_000_000
            + response.usage.completion_tokens * cfg["cost_output_per_m"] / 1_000_000
        ),
        "model_id": cfg["model_id"],
        "model_key": model_key,
    }
    return answer, usage


if __name__ == "__main__":
    # Standalone smoke: retrieve → rerank → build prompt → generate.
    from rag.retrieve import retrieve
    from rag.rerank import rerank
    from rag.prompt import build

    query = "Qu'est-ce que le quasi-usufruit ?"
    hits = retrieve(query, k=20)
    top = rerank(query, hits, k=5)
    prompt = build(query, top)
    answer, usage = generate(prompt)

    print(f"\nquery: {query}")
    print(f"model: {usage['model_id']}")
    print(f"tokens: prompt={usage['prompt_tokens']}, "
          f"completion={usage['completion_tokens']}")
    print(f"cost: ${usage['cost_usd']:.5f}")
    print(f"\nanswer:\n{answer}")
