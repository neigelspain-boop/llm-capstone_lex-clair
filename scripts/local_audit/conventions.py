"""Layer 0: CLAUDE.md's explicit "non-negotiable" conventions, checked
mechanically wherever a regex/AST test can actually decide the question.

Zero LLM calls (same rule as static_tools.py: this stays Layer 0). Checks
that need real judgment — is this specific import a legitimate shared-infra
exception, is this specific ADR gap intentional — still run here, since the
volume is small (single digits to a couple dozen hits across the repo), but
are emitted at confidence="low"/severity="info" rather than asserted as
violations, so a human (or a later LLM pass) makes the actual call.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from scripts.local_audit import config, source_index

log = logging.getLogger(__name__)

ADR_HEADER_RE = re.compile(r"^##\s.*?ADR #(\d+)", re.MULTILINE)


# ========== helpers ==========

# Both moved to source_index.py so passes/slimming.py can share them without
# importing a Layer-0 checker's privates. Aliased back under the original
# names: every call site below is unchanged.
_collect_py_files = source_index.collect_py_files
_finding = source_index.finding


# ========== cross-plane import boundary ==========


def check_cross_plane_imports() -> list[dict]:
    """CLAUDE.md: "load_index() is the single Plane I -> Plane II interface.
    Do not add hidden cross-plane paths." Flags any ingestion.* import from
    rag/eval/app/monitoring that isn't `from ingestion.load import load_index`.
    Low confidence by design — ingestion.clients (shared credential wiring)
    is a plausible legitimate exception, not obviously a boundary violation;
    this surfaces candidates for review, it doesn't assert a verdict.
    """
    findings = []
    for plane in sorted(config.CROSS_PLANE_ALLOWED_IMPORTERS):
        for py_file in _collect_py_files([plane]):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            rel = str(py_file.relative_to(config.PROJECT_ROOT))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ingestion"):
                    names = {a.name for a in node.names}
                    if node.module == "ingestion.load" and names == {"load_index"}:
                        continue  # the sanctioned contract
                    findings.append(_finding(
                        "convention", rel, node.lineno,
                        f"Cross-plane import from {plane}/ reaches into Plane I "
                        f"outside the load_index() contract.",
                        f"from {node.module} import {', '.join(sorted(names))}",
                        severity="medium", confidence="low",
                    ))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("ingestion") and alias.name != "ingestion.load":
                            findings.append(_finding(
                                "convention", rel, node.lineno,
                                f"Cross-plane import from {plane}/ reaches into Plane I "
                                f"outside the load_index() contract.",
                                f"import {alias.name}",
                                severity="medium", confidence="low",
                            ))
    return findings


# ========== keep_default_na=False ==========


def check_csv_na_handling() -> list[dict]:
    """CLAUDE.md: "keep_default_na=False on every pd.read_csv call that reads
    lex-clair-produced CSVs." AST-finds every pd.read_csv(...) call and checks
    whether keep_default_na=False is among its keyword arguments. High
    confidence — this is a literal, checkable rule, not a judgment call.
    """
    findings = []
    for py_file in _collect_py_files(config.SCAN_DIRS):
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = str(py_file.relative_to(config.PROJECT_ROOT))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_read_csv = (
                (isinstance(func, ast.Attribute) and func.attr == "read_csv")
                or (isinstance(func, ast.Name) and func.id == "read_csv")
            )
            if not is_read_csv:
                continue

            has_na_false = any(
                kw.arg == "keep_default_na"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if not has_na_false:
                findings.append(_finding(
                    "convention", rel, node.lineno,
                    "pd.read_csv call missing keep_default_na=False.",
                    ast.unparse(node) if hasattr(ast, "unparse") else "(call site)",
                    severity="high", confidence="medium",
                ))
    return findings


# ========== section-comment dividers ==========


def check_section_dividers() -> list[dict]:
    """CLAUDE.md: "Section-commented code: # ========== dividers above every
    major block, consistently." Only checked for the four RAG planes (not
    tests/scripts, where small ad hoc utilities like
    scripts/enrich_compliance_with_persons.py legitimately skip this), and
    only for files with enough top-level defs/classes that the convention
    would plausibly apply. Info-severity, low-confidence: a style observation
    for review, not an asserted violation.
    """
    findings = []
    plane_dirs = list(config.PLANE_DIRS.values())
    for py_file in _collect_py_files(plane_dirs):
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = str(py_file.relative_to(config.PROJECT_ROOT))

        top_level_defs = sum(
            1 for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        divider_count = source.count("# ==========")
        if top_level_defs >= 3 and divider_count == 0:
            findings.append(_finding(
                "convention", rel, 1,
                "File has multiple top-level defs/classes but no "
                "'# ==========' section dividers.",
                f"{top_level_defs} top-level def/class blocks, 0 dividers",
                severity="info", confidence="low",
            ))
    return findings


# ========== docstring presence ==========


def check_public_docstrings() -> list[dict]:
    """CLAUDE.md's "spec first" convention, as amended by ADR #69.

    ADR #69 moves design *rationale* out of docstrings and into ADRs, leaving
    the contract behind. This check is its counterweight: it flags any public
    (non-underscore) top-level function or class in a plane that has no
    docstring at all, so trimming rationale cannot quietly become deleting
    the contract too.

    Private helpers are exempt — the convention has never demanded a
    docstring on every `_`-prefixed function, and requiring one would produce
    noise rather than a signal.
    """
    findings = []
    for py_file in _collect_py_files(list(config.PLANE_DIRS.values())):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = str(py_file.relative_to(config.PROJECT_ROOT))

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node):
                continue
            findings.append(_finding(
                "convention", rel, node.lineno,
                f"Public `{node.name}` has no docstring.",
                "ADR #69 moves rationale to ADRs but keeps the contract "
                "(arguments, returns, invariants, failure modes) in the "
                "docstring. An empty one is not the intended end state.",
                severity="medium", confidence="high",
            ))
    return findings


# ========== ADR numbering ==========


def check_adr_numbering(decisions_md: Path | None = None) -> list[dict]:
    """Flags gaps in docs/decisions.md's ADR numbering. High confidence that
    a gap exists (pure arithmetic on parsed headers); info severity, since
    CLAUDE.md itself documents some gaps as intentional (e.g. "candidate ADR
    #47", D1/D4 "open" per ADR #50/#51) — this is a bookkeeping surface, not
    an assertion that every gap is a bug.
    """
    path = decisions_md or (config.PROJECT_ROOT / "docs" / "decisions.md")
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    numbers = sorted({int(m) for m in ADR_HEADER_RE.findall(text)})
    if not numbers:
        return []

    findings = []
    for prev, nxt in zip(numbers, numbers[1:]):
        if nxt - prev > 1:
            missing = list(range(prev + 1, nxt))
            findings.append(_finding(
                "convention", "docs/decisions.md", 0,
                f"ADR numbering gap between #{prev} and #{nxt}: missing {missing}.",
                f"parsed ADR headers: ...#{prev}, #{nxt}...",
                severity="info", confidence="high",
            ))
    return findings


# ========== orchestration ==========


def check_all() -> list[dict]:
    findings: list[dict] = []
    findings += check_cross_plane_imports()
    findings += check_csv_na_handling()
    findings += check_section_dividers()
    findings += check_public_docstrings()
    findings += check_adr_numbering()
    return findings
