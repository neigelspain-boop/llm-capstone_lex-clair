"""Centralized LLM provider clients for lex-clair (Plane I · offline).

Single place for API-key handling so credential wiring doesn't get
duplicated across ingestion modules.

get_openrouter_client() is the canonical factory (ADR #40): all LLM calls
route through OpenRouter's OpenAI-compatible endpoint for a single credit
pool and usage dashboard. get_openai_client(), get_anthropic_client(), and
get_mistral_client() are deprecated aliases kept for callers mid-transition
— each now just returns get_openrouter_client(). Callers must use
fully-qualified OpenRouter model slugs (e.g. "openai/gpt-4o-mini",
"anthropic/claude-haiku-4.5", "mistralai/mistral-small-3.2-24b-instruct").
"""
from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

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


# ========== deprecated provider-named aliases ==========
# Kept so existing callers keep working during the transition (ADR #40).

def get_openai_client():
    """Deprecated: use get_openrouter_client()."""
    warnings.warn(
        "get_openai_client() is deprecated; use get_openrouter_client()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_openrouter_client()


def get_anthropic_client():
    """Deprecated: use get_openrouter_client()."""
    warnings.warn(
        "get_anthropic_client() is deprecated; use get_openrouter_client()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_openrouter_client()


def get_mistral_client():
    """Deprecated: use get_openrouter_client()."""
    warnings.warn(
        "get_mistral_client() is deprecated; use get_openrouter_client()",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_openrouter_client()
