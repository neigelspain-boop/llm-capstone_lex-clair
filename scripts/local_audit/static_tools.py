"""Layer 0: deterministic static analysis, re-run fresh every cycle.

Zero LLM calls. Wraps ruff/radon/vulture/bandit/pip-audit — each already
answers a class of question perfectly and near-instantly, so no LLM pass may
re-derive what these report. `grounding_for_file()` is the seam: it turns
this layer's output into the short context block every LLM prompt in
passes/*.py is grounded with, so the model correlates with known lint/
complexity signal instead of re-discovering it.

Tools run via `uvx <tool>` (isolated tool venv — none of them need the
project's own installed packages to analyze its source) except pip-audit,
which audits the project's *resolved dependencies* and therefore runs inside
the project's own uv environment via `uv run --with pip-audit`, matching the
`uv run --with <pkg>` pattern the repo's own deletion_audit.py already used.
"""
from __future__ import annotations

import json
import logging
import subprocess

from scripts.local_audit import config

log = logging.getLogger(__name__)

# ========== subprocess helpers ==========


def _run(cmd: list[str], timeout: int = 120) -> tuple[str, str, int]:
    """Run a subprocess, returning (stdout, stderr, returncode).

    Never raises on nonzero exit — every tool here uses nonzero to mean
    "findings exist", not "the tool failed".
    """
    try:
        proc = subprocess.run(
            cmd, cwd=config.PROJECT_ROOT, capture_output=True,
            text=True, timeout=timeout, check=False,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except FileNotFoundError as e:
        log.warning("static_tools: command not found: %s (%s)", cmd[0], e)
        return "", str(e), -1
    except subprocess.TimeoutExpired as e:
        log.warning("static_tools: %s timed out after %ss", cmd[0], timeout)
        return "", str(e), -1


def _write_static(name: str, text: str) -> None:
    config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
    (config.STATIC_DIR / name).write_text(text, encoding="utf-8")


# ========== individual tools ==========


def run_ruff() -> list[dict]:
    stdout, stderr, _ = _run(
        ["uvx", "ruff", "check", "--output-format=json", *config.SCAN_DIRS]
    )
    _write_static("ruff.json", stdout or "[]")
    try:
        return json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        log.warning("static_tools: ruff output was not valid JSON: %s", stderr[:200])
        return []


def run_radon_cc() -> dict:
    stdout, stderr, _ = _run(["uvx", "radon", "cc", "-j", *config.SCAN_DIRS])
    _write_static("radon_cc.json", stdout or "{}")
    try:
        return json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        log.warning("static_tools: radon cc output was not valid JSON: %s", stderr[:200])
        return {}


def run_radon_mi() -> dict:
    stdout, stderr, _ = _run(["uvx", "radon", "mi", "-j", *config.SCAN_DIRS])
    _write_static("radon_mi.json", stdout or "{}")
    try:
        return json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        log.warning("static_tools: radon mi output was not valid JSON: %s", stderr[:200])
        return {}


def run_radon_raw() -> dict:
    """Per-file SLOC/LLOC/comment/blank counts.

    Feeds the slimming pass's docstring-mass metric — the one Layer 0 signal
    that speaks directly to "this file is too long", as opposed to "this
    function is too complex" (radon cc) or "this file is hard to maintain"
    (radon mi).
    """
    stdout, stderr, _ = _run(["uvx", "radon", "raw", "-j", *config.SCAN_DIRS])
    _write_static("radon_raw.json", stdout or "{}")
    try:
        return json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        log.warning("static_tools: radon raw output was not valid JSON: %s", stderr[:200])
        return {}


def run_vulture() -> list[str]:
    stdout, stderr, _ = _run(["uvx", "vulture", *config.SCAN_DIRS])
    _write_static("vulture.txt", stdout)
    return [line for line in stdout.splitlines() if line.strip()]


def run_bandit() -> dict:
    # -r on SCAN_DIRS only (never project root) is what keeps .venv out —
    # the prior ad hoc run scanned "." and drowned in 143k .venv findings.
    stdout, stderr, _ = _run(
        ["uvx", "bandit", "-r", *config.SCAN_DIRS, "-f", "json", "-q"]
    )
    _write_static("bandit.json", stdout or "{}")
    try:
        return json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        log.warning("static_tools: bandit output was not valid JSON: %s", stderr[:200])
        return {}


def run_pip_audit() -> dict:
    stdout, stderr, _ = _run(
        ["uv", "run", "--with", "pip-audit", "pip-audit", "-f", "json"], timeout=180
    )
    _write_static("pip_audit.json", stdout or "{}")
    try:
        return json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        log.warning("static_tools: pip-audit output was not valid JSON: %s", stderr[:200])
        return {}


# ========== orchestration ==========


def run_all() -> dict:
    """Run every deterministic tool fresh and return the combined result.

    Always call this at the start of a sweep/incremental cycle before any
    LLM pass — the results are Layer 0's grounding context.
    """
    return {
        "ruff": run_ruff(),
        "radon_cc": run_radon_cc(),
        "radon_mi": run_radon_mi(),
        "radon_raw": run_radon_raw(),
        "vulture": run_vulture(),
        "bandit": run_bandit(),
        "pip_audit": run_pip_audit(),
    }


# ========== grounding context for LLM prompts ==========


def grounding_for_file(rel_path: str, static_results: dict) -> str:
    """Short text block summarizing this file's Layer 0 signal.

    Fed into every LLM prompt that reasons about `rel_path`, so the model
    never re-derives lint-level facts and can correlate its own judgment
    with what's already known for free.
    """
    lines: list[str] = []

    ruff_hits = [f for f in static_results.get("ruff", []) if f.get("filename", "").endswith(rel_path)]
    if ruff_hits:
        codes = ", ".join(sorted({h.get("code", "?") for h in ruff_hits}))
        lines.append(f"ruff: {len(ruff_hits)} finding(s), codes: {codes}")

    cc = static_results.get("radon_cc", {})
    for key, blocks in cc.items():
        if key.endswith(rel_path):
            worst = max(blocks, key=lambda b: b.get("complexity", 0), default=None)
            if worst:
                lines.append(
                    f"radon complexity: worst block {worst.get('name')} "
                    f"rank={worst.get('rank')} complexity={worst.get('complexity')}"
                )
            break

    mi = static_results.get("radon_mi", {})
    for key, val in mi.items():
        if key.endswith(rel_path):
            lines.append(f"radon maintainability index: {val.get('mi', '?')} (rank {val.get('rank', '?')})")
            break

    vulture_hits = [ln for ln in static_results.get("vulture", []) if rel_path in ln]
    if vulture_hits:
        lines.append(f"vulture: {len(vulture_hits)} unreferenced-code hit(s) (informational only)")

    if not lines:
        return "No Layer 0 (ruff/radon/vulture) findings for this file."
    return "Layer 0 static-analysis signal for this file:\n" + "\n".join(f"- {l}" for l in lines)
