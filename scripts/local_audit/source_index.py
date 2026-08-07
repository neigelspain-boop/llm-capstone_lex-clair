"""Shared source-walking primitives: file collection, function extraction,
and the normalized-AST fingerprints the slimming pass clusters on.

Zero LLM calls, zero subprocesses — pure `ast`. This module exists because
`conventions.py` and `passes/slimming.py` both need to walk the tree, and a
pass reaching into a Layer-0 checker's private helpers would invert the
dependency direction. `collect_py_files` and `finding` moved here verbatim
from conventions.py; conventions.py now imports them back under its old
private names, so nothing else in that file changed.

The fingerprint design is the load-bearing part. Two tiers, deliberately
normalized to *different* strengths:

- **Function tier** (`keep_attr=False`) erases attribute names along with
  identifiers, so `facts.py::_parse_llm_json`, `mentions.py::_parse_mentions_json`
  and `resolve.py::_parse_persons_json` collapse to one fingerprint despite
  logging different strings and returning different container types.
- **Window tier** (`keep_attr=True`) keeps attribute names, so a 4-statement
  window reading `usage.prompt_tokens`/`usage.completion_tokens` does not
  collide with an unrelated attribute chain of the same shape. At window
  scale the structure alone is too weak a signal to identify a concept.

Constants are reduced to their *type* (`int`, `str`), never their value —
that is what makes two functions match when they differ only in a filename
literal or a log message. Docstrings are stripped before fingerprinting, so
editing prose never re-mints a cluster (which would churn finding ids).
"""
from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from scripts.local_audit import config

log = logging.getLogger(__name__)


# ========== file collection ==========


def collect_py_files(dirs: list[str]) -> list[Path]:
    """Every .py file under `dirs`, skipping config.PY_SKIP_DIRS.

    Moved verbatim from conventions._collect_py_files. Sorted so every
    downstream ordering (and therefore every findings.jsonl write) is
    deterministic across runs.
    """
    files = []
    for d in dirs:
        base = config.PROJECT_ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            rel = p.relative_to(config.PROJECT_ROOT)
            if any(part in config.PY_SKIP_DIRS for part in rel.parts):
                continue
            files.append(p)
    return sorted(files)


def is_skipped_path(rel_path: str) -> bool:
    """True when `rel_path` sits under a PY_SKIP_DIRS directory.

    Layer 0's tools are invoked on config.SCAN_DIRS via `uvx` (static_tools.py),
    which bypasses collect_py_files entirely — and SCAN_DIRS includes
    "scripts", so vulture and ruff currently report on the audit harness
    itself. Every converter that turns their raw output into findings must
    filter through here, or the harness audits its own auditor.
    """
    return any(part in config.PY_SKIP_DIRS for part in Path(rel_path).parts)


# ========== finding construction ==========


def finding(pass_name: str, file: str, line: int, claim: str, evidence: str,
            severity: str, confidence: str,
            related_files: tuple[str, ...] | list[str] = ()) -> dict:
    """Build one raw finding dict in the shape findings.normalize() expects.

    Moved from conventions._finding, with related_files promoted from a
    hardcoded [] to a parameter (the slimming pass needs to name sibling
    sites; conventions.py passes nothing and still gets []).

    `pass_name` is the first argument on purpose: orchestrator._record
    derives its reconcile set from the "pass" value inside each dict but
    filters the store on the pass_name string it was called with, so a
    mismatch silently resolves every finding the pass ever emitted. Taking
    it as a required positional makes that drift hard to introduce.

    `claim` is identity — findings.make_id hashes (pass, file, claim) only.
    Never interpolate a count, a hash, or a line number into it; those
    belong in `evidence`, which is updated in place on every upsert.
    """
    return {
        "pass": pass_name, "file": file, "line": line,
        "related_files": list(related_files), "claim": claim,
        "evidence": evidence, "severity": severity, "confidence": confidence,
    }


# ========== function extraction ==========


@dataclass(frozen=True)
class FuncNode:
    """One top-level function or one method, with its docstring separated.

    `body` excludes the docstring statement, so `lineno`/`end_lineno` bound
    the whole def while `body` is what gets fingerprinted.
    """
    qualname: str
    file: str
    lineno: int
    end_lineno: int
    decorators: tuple[str, ...]
    node: ast.FunctionDef | ast.AsyncFunctionDef
    body: tuple[ast.stmt, ...]
    doc_lines: int

    @property
    def length(self) -> int:
        return self.end_lineno - self.lineno + 1


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    names = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return tuple(names)


def _strip_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[tuple[ast.stmt, ...], int]:
    body = list(node.body)
    doc_lines = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        doc_node = body.pop(0)
        doc_lines = (doc_node.end_lineno or doc_node.lineno) - doc_node.lineno + 1
    return tuple(body), doc_lines


def parse_file(path: Path) -> ast.Module | None:
    """Parse a file, returning None (not raising) on syntax or read errors."""
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError) as e:
        log.debug("source_index: cannot parse %s (%s)", path, e)
        return None


def iter_functions(path: Path, tree: ast.Module | None = None):
    """Yield a FuncNode per top-level function and per method.

    One level of class nesting only, matching duplication._collect_functions:
    inner functions and methods of nested classes are not collected. They are
    rarely the shape of duplication this pass is looking for, and including
    them would double the fingerprint population for little signal.
    """
    tree = tree if tree is not None else parse_file(path)
    if tree is None:
        return
    rel = str(path.relative_to(config.PROJECT_ROOT))

    def _make(node, qualname: str) -> FuncNode:
        body, doc_lines = _strip_docstring(node)
        return FuncNode(
            qualname=qualname, file=rel, lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            decorators=_decorator_names(node), node=node, body=body,
            doc_lines=doc_lines,
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield _make(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield _make(sub, f"{node.name}.{sub.name}")


# ========== normalized-AST fingerprints ==========

# Fields erased during normalization: every identifier-like name. What
# survives is node type, tree shape, and constant *types*.
_ERASED_FIELDS = frozenset({
    "name", "id", "arg", "module", "asname", "vararg", "kwarg", "attr",
})


def normalized_dump(node: ast.AST, keep_attr: bool) -> str:
    """Structure-only serialization of an AST node.

    `keep_attr=True` preserves `.attr` access names (window tier);
    `keep_attr=False` erases them along with every other identifier
    (function tier). See the module docstring for why the two tiers are
    normalized to different strengths.
    """
    out: list[str] = []

    def walk(n) -> None:
        if isinstance(n, ast.AST):
            out.append(type(n).__name__)
            if isinstance(n, ast.Constant):
                # Type, never value — this is what lets two functions match
                # when they differ only in a filename or a log string.
                out.append(f"<{type(n.value).__name__}>")
                return
            if keep_attr and isinstance(n, ast.Attribute):
                out.append(f".{n.attr}")
            for field, value in ast.iter_fields(n):
                if field in _ERASED_FIELDS:
                    continue
                out.append(f"|{field}")
                walk(value)
        elif isinstance(n, list):
            out.append("[")
            for item in n:
                walk(item)
            out.append("]")
        elif n is not None:
            out.append(f"<{type(n).__name__}>")

    walk(node)
    return "".join(out)


def fingerprint(nodes, keep_attr: bool) -> str:
    """Stable short hash over a sequence of statements."""
    payload = "\x00".join(normalized_dump(n, keep_attr) for n in nodes)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def node_mass(nodes) -> int:
    """Total AST node count across a statement sequence.

    The size gate is mass, not line count: a run of four `print(f"...")`
    calls spans four lines but carries almost no structure, and filtering on
    lines lets exactly that boilerplate through as a false cluster.
    """
    return sum(len(list(ast.walk(n))) for n in nodes)


def is_expression_only(nodes) -> bool:
    """True when every statement is a bare expression (calls, prints, logs).

    Such a window is boilerplate rather than a concept — it has no binding,
    no control flow, and no return, so two matching ones share a calling
    style, not an implementation.
    """
    return all(isinstance(n, ast.Expr) for n in nodes)


def statement_windows(func: FuncNode, k: int):
    """Yield (start_index, tuple_of_k_consecutive_statements) from a body.

    Windows are taken over the top level of the function body only, not over
    nested blocks — the concepts this pass tracks (fence strip, usage/cost
    extraction, argparse quartets) are all flat statement runs.
    """
    body = func.body
    if len(body) < k:
        return
    for i in range(len(body) - k + 1):
        yield i, body[i:i + k]


def span_text(path: Path, start_line: int, end_line: int) -> str:
    """Source text for a line span, with a line-number gutter.

    Used verbatim as the LLM prompt's code block, and hashed for the cache
    key — the two must be byte-identical or the cache stops matching what
    was actually judged.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    chunk = lines[start_line - 1:end_line]
    return "\n".join(f"{start_line + i:>5} | {ln}" for i, ln in enumerate(chunk))
