"""Orchestrates a full sweep or an incremental cycle across every pass,
upserting results into the shared findings store and regenerating the
digest. Entry point for both a one-shot manual run and watch.py's loop.
"""
from __future__ import annotations

import logging
import subprocess

from scripts.local_audit import config, conventions, digest, findings, static_tools
from scripts.local_audit.passes import doc_drift, duplication, prompt_audit, slimming

log = logging.getLogger(__name__)


def _current_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _record(pass_name: str, raw_findings: list[dict], sha: str | None,
            prompt_version: str | None = None, reconcile: bool = True) -> None:
    """Upsert a pass's findings and resolve the ones it no longer reproduces.

    `reconcile=False` for a pass that ran only partially this cycle (budget
    cutoff, transport failure): its unreached findings are still valid, and
    resolving them would be indistinguishable from a real fix.

    Note current_ids is derived from each finding's own "pass" value while
    reconcile_pass filters on `pass_name` — a mismatch between the two
    silently resolves everything the pass has ever emitted.
    """
    store = findings.upsert_many(raw_findings, verified_at_sha=sha, prompt_version=prompt_version)
    current_ids = {
        findings.make_id(f["pass"], f["file"], f["claim"]) for f in raw_findings
    }
    if reconcile:
        findings.reconcile_pass(pass_name, current_ids)
    open_count = sum(1 for f in store.values() if f["pass"] == pass_name and f["status"] == "open")
    log.info("orchestrator: pass=%s produced %d finding(s) this cycle, %d open total",
              pass_name, len(raw_findings), open_count)


def run_full_sweep() -> str:
    """Everything: Layer 0 + all three LLM passes over their full scope."""
    sha = _current_sha()
    log.info("orchestrator: full sweep starting (sha=%s)", sha)

    static_results = static_tools.run_all()

    _record("convention", conventions.check_all(), sha)
    _record("slimming", slimming.run(static_results), sha)
    _record("doc_drift", doc_drift.run(), sha)
    _record("duplication", duplication.run(static_results), sha)
    _record("prompt_audit", prompt_audit.run(static_results), sha)
    _record("slimming_divergence", slimming.run_divergence(static_results), sha,
            prompt_version=config.PROMPT_VERSIONS["slimming_divergence"],
            reconcile=slimming.LAST_DIVERGENCE_COMPLETE)

    path = digest.regenerate()
    log.info("orchestrator: full sweep done, digest at %s", path)
    return sha or "(unknown)"


def run_incremental(changed_files: list[str] | None = None) -> str:
    """Layer 0 + doc_drift always (both are cache-cheap when nothing
    relevant changed); duplication/prompt_audit only if a changed file
    touches config.HEAVY_PASS_TRIGGERS — otherwise they wait for the next
    full sweep, since neither is cheaply cache-invalidated per file.
    """
    changed_files = changed_files or []
    sha = _current_sha()
    log.info("orchestrator: incremental cycle (sha=%s, %d changed file(s))", sha, len(changed_files))

    static_results = static_tools.run_all()

    _record("convention", conventions.check_all(), sha)
    # Deterministic and cheap — fingerprinting ~600 functions costs about two
    # seconds, less than conventions.check_all(), so it runs every cycle
    # rather than waiting behind HEAVY_PASS_TRIGGERS (which is tuned for
    # duplication's re-embedding cost and names three RAG-specific paths that
    # have nothing to do with a repo-wide slimming scan).
    _record("slimming", slimming.run(static_results), sha)
    _record("doc_drift", doc_drift.run(), sha)

    trigger_heavy = any(
        cf.startswith(trigger) for cf in changed_files for trigger in config.HEAVY_PASS_TRIGGERS
    )
    if trigger_heavy:
        log.info("orchestrator: changed files touch a heavy-pass trigger — running duplication + prompt_audit")
        _record("duplication", duplication.run(static_results), sha)
        _record("prompt_audit", prompt_audit.run(static_results), sha)
        _record("slimming_divergence", slimming.run_divergence(static_results), sha,
                prompt_version=config.PROMPT_VERSIONS["slimming_divergence"],
                reconcile=slimming.LAST_DIVERGENCE_COMPLETE)
    else:
        log.info("orchestrator: no heavy-pass trigger touched — deferring duplication/prompt_audit to next full sweep")

    path = digest.regenerate()
    log.info("orchestrator: incremental cycle done, digest at %s", path)
    return sha or "(unknown)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_full_sweep()
