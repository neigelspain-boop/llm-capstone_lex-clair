"""Answer synthesis via OpenAI.

Public surface: `generate(prompt)` returns (answer_text, token_stats).
token_stats is a dict with prompt_tokens, completion_tokens, total_tokens
— same shape as OpenAI's usage field, same names Day 5 LLM eval will
consume, same names monitoring/db.py will log (Day 7).

Model: gpt-4o-mini. Hardcoded per plan §6.4 (Day 4 is single-provider).
Day 5 refactor introduces provider abstraction (Anthropic, Mistral) if
needed for the judge pipeline; generate.py may or may not be refactored
at that time.

Cost: gpt-4o-mini is $0.15/M input, $0.60/M output. A typical answer
call with 5 chunks × ~300 tokens context + 400-token answer costs
~$0.00048. Same order of magnitude as rewrite.py (~$0.00005).
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 500


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return the cached OpenAI client, initializing on first call."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set — check .env file")
        _client = OpenAI()
    return _client


def generate(prompt: str) -> tuple[str, dict]:
    """Call the LLM with a prompt and return the answer + token usage.

    Returns:
        (answer_text, token_stats) where token_stats is:
        {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    response = get_client().responses.create(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    answer = response.output_text
    token_stats = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return answer, token_stats


if __name__ == "__main__":
    # Standalone smoke: retrieve → rerank → build prompt → generate.
    from rag.retrieve import retrieve
    from rag.rerank import rerank
    from rag.prompt import build

    query = "Qu'est-ce que le quasi-usufruit ?"
    hits = retrieve(query, k=20)
    top = rerank(query, hits, k=5)
    prompt = build(query, top)
    answer, tokens = generate(prompt)

    print(f"\nquery: {query}")
    print(f"tokens: prompt={tokens['prompt_tokens']}, "
          f"completion={tokens['completion_tokens']}, total={tokens['total_tokens']}")
    print(f"cost: ${(tokens['prompt_tokens'] * 0.15 + tokens['completion_tokens'] * 0.60) / 1_000_000:.5f}")
    print(f"\nanswer:\n{answer}")