"""Pass 3: LLM prompt-engineering audit.

Small, stable site list — this project's core product is prompts, so these
files get read whole and reviewed for internal-consistency problems that
only careful reading (not a linter) can catch: prompt-vs-parser schema
drift, language-toggle inconsistency. The max_tokens-too-small-for-
reasoning-effort bug class (CLAUDE.md's own documented ADR #49 gotcha) is a
deterministic regex check, not an LLM question — flagged mechanically here,
same "don't spend LLM judgment on what a rule already answers" principle as
the other passes.
"""
from __future__ import annotations

import re

from scripts.local_audit import config, ollama_client, static_tools

PROMPT_SITES = [
    "rag/generate.py",
    "rag/compliance.py",
    "rag/router.py",
    "rag/rewrite.py",
    "ingestion/dossier/extract.py",
    "ingestion/dossier/distill.py",
    "ingestion/dossier/mentions.py",
    "ingestion/dossier/resolve.py",
    "ingestion/dossier/gate.py",
    "ingestion/dossier/facts.py",
    "ingestion/dossier/anonymize.py",
]

SYSTEM_PROMPT = """You review a single file from a French legal RAG system's prompt-construction code for internal-consistency problems only — not general code quality.

Look specifically for:
1. Schema drift: does a prompt tell the LLM to return JSON/fields that don't match the Pydantic model (or manual parser) that consumes the response elsewhere in the same file?
2. Language-toggle inconsistency: if the code supports multiple languages (French/English), does a prompt's stated language stay consistent with what the surrounding code expects?
3. Any other internal contradiction between what a prompt promises the LLM and what the code that consumes the LLM's response assumes.

Do not flag: general style issues, missing docstrings, or anything a linter would already catch — you are here for problems only careful reading of prompt intent versus code behavior can find. If genuinely nothing stands out, say so with an empty list — do not invent an issue to have something to report.

Respond ONLY with valid JSON: {"issues": [{"description": "...", "evidence": "specific snippet or line reference", "severity": "critical"|"high"|"medium"|"low"}]}
"""

REASONING_BUDGET_RE = re.compile(
    # matches both the literal OpenRouter call shape {"effort": "max"} and
    # this project's own catalog convention {"reasoning_effort": "max"}
    r'["\']?\w*effort["\']?\s*:\s*["\'](max|high)["\']', re.IGNORECASE
)
MAX_TOKENS_RE = re.compile(r"max_tokens[\"']?\s*[:=]\s*(\d+)")
REASONING_SAFE_BUDGET = 8192  # CLAUDE.md's own documented threshold discussion (ADR #49)


def _enclosing_block(text: str, pos: int, fallback_radius: int = 400) -> str:
    """Smallest {...} span enclosing `pos`, found by brace-depth scanning
    outward in both directions. A blind +-N char window bridges adjacent
    dict entries in compact catalogs (e.g. rag/generate.py's ANSWER_MODELS,
    where each model's config is only ~7 lines) and wrongly pairs one
    entry's reasoning_effort with a neighboring entry's max_tokens. Falls
    back to a plain window if no balanced brace is found (e.g. the literal
    extra_body={...} call-site shape isn't inside an outer dict literal).
    """
    depth = 0
    start = None
    for i in range(pos, -1, -1):
        c = text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                start = i
                break
            depth -= 1
    if start is None:
        return text[max(0, pos - fallback_radius): pos + fallback_radius]

    depth = 0
    end = None
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        end = min(len(text), start + fallback_radius * 2)
    return text[start:end + 1]


def _check_reasoning_budget(rel_path: str, text: str) -> list[dict]:
    """Deterministic: flag max_tokens values that look too small to share
    between reasoning tokens and output JSON on a reasoning-effort model.
    Scoped to the smallest enclosing {...} block, not a blind char window —
    a heuristic, not full call-graph analysis, hence confidence="medium".
    """
    findings = []
    for m in REASONING_BUDGET_RE.finditer(text):
        window = _enclosing_block(text, m.start())
        for tm in MAX_TOKENS_RE.finditer(window):
            value = int(tm.group(1))
            if value < REASONING_SAFE_BUDGET:
                line = text.count("\n", 0, m.start()) + 1
                findings.append({
                    "pass": "prompt_audit",
                    "file": rel_path,
                    "line": line,
                    "related_files": [],
                    "claim": (
                        f"reasoning effort='max'/'high' near a max_tokens={value} budget "
                        f"(< {REASONING_SAFE_BUDGET}) — reasoning tokens and output JSON share "
                        f"this budget (CLAUDE.md's own documented ADR #49 gotcha)."
                    ),
                    "evidence": window.strip()[:200],
                    "severity": "high",
                    "confidence": "medium",
                })
                break  # one flag per reasoning-effort occurrence is enough
    return findings


def run(static_results: dict | None = None, sites: list[str] | None = None) -> list[dict]:
    static_results = static_results or {}
    sites = sites or PROMPT_SITES

    out_findings: list[dict] = []
    for rel_path in sites:
        path = config.PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")

        out_findings += _check_reasoning_budget(rel_path, text)

        grounding = static_tools.grounding_for_file(rel_path, static_results) if static_results else ""
        excerpt = text[:10000] + ("\n... [truncated]" if len(text) > 10000 else "")
        user_prompt = f"File: {rel_path}\n\n{grounding}\n\n--- CODE ---\n{excerpt}\n--- END CODE ---"

        # Open-ended review (schema drift, language-toggle consistency) is
        # the same class of genuine-judgment task as duplication.py's
        # pairwise call — same reasoning for think=True + the larger model
        # (see config.OLLAMA_MODEL_JUDGMENT).
        result = ollama_client.call_json(
            SYSTEM_PROMPT, user_prompt, temperature=0.0,
            think=True, model=config.OLLAMA_MODEL_JUDGMENT,
        )
        issues = result.get("issues", []) if isinstance(result, dict) else []
        for issue in issues:
            if not isinstance(issue, dict) or not issue.get("description"):
                continue
            out_findings.append({
                "pass": "prompt_audit",
                "file": rel_path,
                "line": 0,
                "related_files": [],
                "claim": issue["description"],
                "evidence": issue.get("evidence", ""),
                "severity": issue.get("severity", "medium"),
                "confidence": "low",  # single-pass, no self-consistency for open-ended review
            })

    return out_findings
