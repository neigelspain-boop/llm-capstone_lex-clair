"""Pass 1: doc/ADR/CLAUDE.md drift.

For each prose chunk in CLAUDE.md and docs/decisions.md: extract falsifiable
claims (one LLM call), then verify each — exists/not_exists claims resolve
deterministically (Path.exists()/AST scan, zero LLM cost); everything else
goes through the self-consistency LLM verifier in claims.py. Only "fail"
(contradicted) verdicts become findings — "pass" and "uncertain" are
discarded, matching the harness's precision-over-recall stance.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from scripts.local_audit import cache, claims, config

log = logging.getLogger(__name__)

DEFAULT_SOURCES = [
    config.PROJECT_ROOT / "CLAUDE.md",
    config.PROJECT_ROOT / "docs" / "decisions.md",
]


def _current_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "(unknown)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "(unknown)"


def _claim_key(claim_text: str) -> str:
    return hashlib.sha256(claim_text.encode("utf-8")).hexdigest()[:16]


def _verify_code_hash(claim: dict) -> str:
    candidate_files = [c for c in (claim.get("candidate_files") or []) if c]
    file_state = "|".join(
        cache.code_hash_for_file(config.PROJECT_ROOT / cf) for cf in candidate_files
    ) or "no-files"
    return cache.code_hash_for_text(claim["claim_text"] + "|" + file_state)


def process_chunk(label: str, heading: str, chunk_text: str, extra_grounding: str) -> list[dict]:
    """Extract + verify claims for a single (source, heading, chunk_text)
    unit, returning any resulting findings. Factored out of run() so a
    single known chunk can be validated in isolation — a full sweep across
    CLAUDE.md + docs/decisions.md is dozens of LLM calls deep (by design:
    this is the harness meant to run unattended for hours, not something to
    wait on interactively) and isolating one chunk is the practical way to
    check a specific claim end to end.
    """
    extract_pv = config.PROMPT_VERSIONS["doc_drift_extract"]
    verify_pv = config.PROMPT_VERSIONS["doc_drift_verify"]
    chunk_key = f"{label}#{heading}"

    chunk_hash = cache.code_hash_for_text(chunk_text)
    cached_extract = cache.get("doc_drift_extract", extract_pv, chunk_key, chunk_hash)
    if cached_extract is not None:
        extracted = cached_extract.get("claims", [])
        log.info("doc_drift: %s -> %d claim(s) (cached)", heading, len(extracted))
    else:
        extracted = claims.extract_claims(heading, chunk_text, label)
        cache.set("doc_drift_extract", extract_pv, chunk_key, chunk_hash, {"claims": extracted})
        log.info("doc_drift: %s -> %d claim(s)", heading, len(extracted))

    chunk_findings: list[dict] = []
    for claim in extracted:
        if "claim_text" not in claim:
            continue

        result = claims.resolve_exists_claim(claim)
        confidence = "high"

        if result is None:
            verify_key = f"{chunk_key}#claim={_claim_key(claim['claim_text'])}"
            v_hash = _verify_code_hash(claim)
            cached_verify = cache.get("doc_drift_verify", verify_pv, verify_key, v_hash)
            if cached_verify is not None:
                result = cached_verify
            else:
                result = claims.verify_claim_llm(claim, extra_grounding)
                if result is not None:
                    cache.set("doc_drift_verify", verify_pv, verify_key, v_hash, result)
            confidence = "medium"

        if result is None or result.get("verdict") != "fail":
            continue

        log.info("doc_drift: FLAGGED (%s) %s", confidence, claim["claim_text"][:100])
        chunk_findings.append({
            "pass": "doc_drift",
            "file": label,
            "line": 0,
            "related_files": claim.get("candidate_files") or [],
            "claim": claim["claim_text"],
            "evidence": result.get("evidence", ""),
            "severity": result.get("severity", "medium"),
            "confidence": confidence,
        })
    return chunk_findings


def run(sources: list[Path] | None = None) -> list[dict]:
    sources = sources or DEFAULT_SOURCES
    branch = _current_branch()
    extra_grounding = f"Ground truth context (verified independently, trust this): current git branch is '{branch}'."

    out_findings: list[dict] = []

    for source in sources:
        if not source.exists():
            log.warning("doc_drift: source not found: %s", source)
            continue
        label = str(source.relative_to(config.PROJECT_ROOT))
        chunks = claims.chunk_markdown_by_h2(source)
        log.info("doc_drift: %s -> %d chunk(s)", label, len(chunks))

        for i, (heading, chunk_text) in enumerate(chunks, 1):
            log.info("doc_drift: [%d/%d] processing %s", i, len(chunks), heading)
            out_findings += process_chunk(label, heading, chunk_text, extra_grounding)

    return out_findings
