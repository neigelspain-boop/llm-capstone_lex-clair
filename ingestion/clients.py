"""Centralized LLM provider clients for lex-clair (Plane I · offline).

Single place for API-key handling so credential wiring doesn't get
duplicated across ingestion modules. Mirrors the lazy-singleton pattern in
eval/llm_eval.py.

get_anthropic_client() does NOT return a native `anthropic.Anthropic()`
client. Per docs/decisions.md ("Claude judge via OpenRouter — not direct
Anthropic API"), direct Anthropic billing is blocked in the EU for this
account, so Claude-family calls route through OpenRouter's OpenAI-compatible
endpoint instead. Callers must use `anthropic/`-prefixed OpenRouter model
slugs (e.g. "anthropic/claude-opus-4.7", "anthropic/claude-haiku-4.5").
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ========== lazy client singletons ==========
# One client per provider, instantiated on first use.

_openai_client = None
_anthropic_client = None
_mistral_client = None


def get_openai_client():
    """Lazy-singleton OpenAI client (OPENAI_API_KEY from environment)."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def get_anthropic_client():
    """Lazy-singleton OpenRouter client for Claude-family (Anthropic) models.

    Returns an OpenAI-compatible client pointed at OpenRouter, not the
    native anthropic SDK — see module docstring.
    """
    global _anthropic_client
    if _anthropic_client is None:
        from openai import OpenAI
        _anthropic_client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    return _anthropic_client


def get_mistral_client():
    """Lazy-singleton Mistral client (MISTRAL_API_KEY from environment)."""
    global _mistral_client
    if _mistral_client is None:
        from mistralai.client import Mistral
        _mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return _mistral_client
