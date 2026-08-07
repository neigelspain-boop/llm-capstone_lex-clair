"""Shared claim extractor + verifier, used by passes/doc_drift.py (and
reusable by any future pass that needs "does this prose statement match the
code" judgment).

Two-step pipeline per the harness plan: extract falsifiable claims from a
prose chunk (one LLM call), then verify each claim (deterministic path/AST
check first — zero LLM cost — LLM only for what that can't decide).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from scripts.local_audit import config, ollama_client

# ========== prompts ==========

EXTRACT_SYSTEM_PROMPT = """You audit a software project's documentation for claims that can be checked against the actual source code.

Given a chunk of prose from a project doc, extract every FALSIFIABLE claim it makes about the code's current state — not opinions, not plans, not history that isn't checkable today.

For each claim, classify claim_type:
- "exists": the doc claims a specific file/module/symbol exists
- "not_exists": the doc claims a specific file/module/symbol does NOT exist ("X does not exist anywhere", "no Y on any branch", etc.)
- "other": any other checkable claim about current behavior, a value (model name, branch name, line count, config value), or a cross-reference (ADR number, function name) — anything that isn't a pure existence claim

For "exists"/"not_exists" claims, candidate_files MUST be concrete relative file paths if the doc names them (e.g. "ingestion/dossier/mentions.py"), or your best guess at the path from context. For "other" claims, candidate_files should list any file(s) the claim is about, if identifiable.

Skip: aspirational statements ("we should X"), pure opinion, claims with no way to check them against code.

Respond ONLY with valid JSON: {"claims": [{"claim_text": "...", "claim_type": "exists"|"not_exists"|"other", "candidate_files": ["..."]}]}
If there are no checkable claims in this chunk, respond {"claims": []}. No markdown, no prose outside the JSON.
"""

VERIFY_SYSTEM_PROMPT = """You verify a single claim from project documentation against the actual current source code shown to you.

Decide: does the code shown SUPPORT the claim, CONTRADICT it, or is there not enough shown to tell?

Be conservative: only say "fail" (contradicted) if the evidence clearly contradicts the claim. If the file/excerpt shown doesn't cover what the claim is about, say "uncertain" rather than guessing.

Respond ONLY with valid JSON: {"verdict": "pass"|"fail"|"uncertain", "evidence": "one sentence citing what you saw", "severity": "critical"|"high"|"medium"|"low"|"info"}
severity applies only when verdict is "fail": how misleading is this to a reader relying on the doc (e.g. a stale branch name is "low"; "this pipeline doesn't exist" when it does is "high"). Omit or use "info" when verdict is not "fail".
"""


# ========== chunking ==========


def chunk_markdown_by_h2(path) -> list[tuple[str, str]]:
    """Split a markdown file into (heading, chunk_text) pairs on '## ' headers."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    current_heading = "(preamble)"
    current_lines: list[str] = []

    def flush():
        if current_lines and any(l.strip() for l in current_lines):
            chunks.append((current_heading, "\n".join(current_lines).strip()))

    for line in lines:
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return chunks


# ========== extraction ==========


def extract_claims(chunk_heading: str, chunk_text: str, source_label: str) -> list[dict]:
    user_prompt = f"Source: {source_label} — \"{chunk_heading}\"\n\n---\n{chunk_text}\n---"
    result = ollama_client.call_json(EXTRACT_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    if isinstance(result, dict):
        claims = result.get("claims", [])
    elif isinstance(result, list):
        claims = result
    else:
        claims = []
    return [c for c in claims if isinstance(c, dict) and c.get("claim_text")]


# ========== deterministic exists/not_exists resolution ==========


def _symbol_exists_in_file(path, symbol: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, FileNotFoundError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return True
    return False


def _looks_like_path(candidate: str) -> bool:
    """The extraction step sometimes names a non-file token as a
    "candidate_files" entry — a git branch ("v2-agentic"), a technology
    ("Postgres"), a git ref ("main") — because the doc prose mentions it in
    a sentence that also happens to be a checkable claim. Treating those as
    files to check for produces a confident-looking but wrong "missing"
    verdict (there's no *file* called "main", even though the branch is
    real). Only strings that look like an actual repo path — contain a "/"
    or end in a file extension — are trusted for the deterministic
    existence check or read as a file excerpt for the LLM verifier.
    """
    if "/" in candidate:
        return True
    return bool(re.search(r"\.[A-Za-z0-9]{1,6}$", candidate))


def _resolve_path(candidate: str) -> Path | None:
    """A doc's own wording rarely spells out a full repo-relative path (e.g.
    "no mentions.py/resolve.py on any branch" names bare filenames). Try the
    literal path from PROJECT_ROOT first; if that misses, fall back to a
    basename search across the repo (skipping the usual noise dirs) so a
    real file isn't reported "missing" just because the claim text under-
    specified its location.
    """
    direct = config.PROJECT_ROOT / candidate
    if direct.exists():
        return direct
    basename = Path(candidate).name
    for hit in config.PROJECT_ROOT.rglob(basename):
        rel = hit.relative_to(config.PROJECT_ROOT)
        if any(part in config.PY_SKIP_DIRS for part in rel.parts):
            continue
        return hit
    return None


def resolve_exists_claim(claim: dict) -> dict | None:
    """Deterministically resolve an exists/not_exists claim via Path.exists()
    (with a basename-search fallback, see _resolve_path). Returns None if the
    claim isn't that type or names no candidate files (falls through to the
    LLM verifier in that case).
    """
    claim_type = claim.get("claim_type")
    if claim_type not in ("exists", "not_exists"):
        return None
    candidate_files = [c for c in (claim.get("candidate_files") or []) if c and _looks_like_path(c)]
    if not candidate_files:
        return None

    presence = []
    for cf in candidate_files:
        found = _resolve_path(cf)
        presence.append((cf if found is None else str(found.relative_to(config.PROJECT_ROOT)), found is not None))

    any_present = any(present for _, present in presence)
    contradicted = any_present if claim_type == "not_exists" else not any_present

    evidence = "; ".join(f"{cf}: {'exists' if present else 'missing'}" for cf, present in presence)
    return {
        "verdict": "fail" if contradicted else "pass",
        "evidence": evidence,
        "severity": "high" if contradicted else "info",
        "method": "deterministic_path_check",
    }


# ========== LLM verification ==========


def _read_excerpt(rel_path: str, max_chars: int = 3000) -> str:
    if not _looks_like_path(rel_path):
        # Not a plausible file reference (a branch name, a technology name,
        # etc.) — don't fabricate "[X does not exist]" evidence for
        # something that was never a file claim in the first place.
        return f"[{rel_path!r} is not a file path — treat as a non-file reference, not evidence of absence]"
    p = _resolve_path(rel_path)  # same basename-fallback as the deterministic check
    if p is None:
        return f"[{rel_path} does not exist]"
    if p.is_dir():
        names = sorted(x.name for x in p.iterdir())[:30]
        return f"[{rel_path} is a directory, contents: {', '.join(names)}]"
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n... [truncated]" if len(text) > max_chars else "")


def verify_claim_llm(claim: dict, extra_grounding: str = "") -> dict | None:
    candidate_files = [c for c in (claim.get("candidate_files") or []) if c]
    excerpts = "\n\n".join(
        f"--- {cf} ---\n{_read_excerpt(cf)}" for cf in candidate_files
    ) or "(no specific file identified — use the grounding context only)"

    user_prompt = (
        f"Claim: {claim['claim_text']}\n\n"
        f"{extra_grounding}\n\n"
        f"Relevant source excerpts:\n{excerpts}"
    )
    result, meta = ollama_client.call_self_consistency(
        VERIFY_SYSTEM_PROMPT, user_prompt, verdict_key="verdict"
    )
    if result is not None:
        result["_self_consistency"] = meta
    return result
