"""Pass 4 — code slimming: redundancy, dead code, and size budgets.

Report-only, like every pass here. Nothing in this module asks the local
model to produce a diff, a patch, or a replacement snippet; the LLM tier
answers "does this span violate clause N of this written rule" and nothing
else. Applying a fix is a separate, human-initiated step.

Two entry points, deliberately under two different pass names:

- `run()` -> pass "slimming". Fully deterministic: concept-registry matches,
  unregistered clone discovery, and the Layer 0 converters (vulture, ruff
  F401, per-function length, docstring mass). Fast enough — ~600 functions
  fingerprinted in about two seconds — to run on every incremental cycle.
- `run_divergence()` -> pass "slimming_divergence". The LLM tier, cache-gated
  per span and budgeted per cycle.

**The pass names must stay separate.** orchestrator._record derives its
reconcile set from the findings it was handed but filters the store on the
pass name it was called with, so one shared name would mean any cycle that
ran the deterministic tier alone would mark every LLM finding "resolved" —
silently, and indistinguishably from a real fix.

Design note on why the LLM tier is shaped the way it is: passes/duplication.py
asks two whole functions whether they share a responsibility, and both
qwen3:14b and qwen3:30b answer wrongly 3/3 on its own calibration case, with
near-identical reasoning. That is whole-function anchoring — a 126-line
function containing an 8-line cost block compresses to "compliance analysis"
before any comparison happens, and the block disappears from the model's
representation. So here the model never sees a function, never sees two
implementations at once, and never decides membership: fingerprints decide
membership first, and the model checks one short span against one written
rule. Same shape as claims.py's verify path, which already works on the 14b.
"""
from __future__ import annotations

import ast
import collections
import json
import logging
import re
from pathlib import Path

from scripts.local_audit import cache, concepts, config, ollama_client
from scripts.local_audit import source_index as si

log = logging.getLogger(__name__)

PASS_NAME = "slimming"
DIVERGENCE_PASS_NAME = "slimming_divergence"

# Set by run_divergence(): False when the per-cycle budget cut the run short,
# so the orchestrator can skip reconciliation rather than resolve findings
# that were simply never re-checked this cycle.
LAST_DIVERGENCE_COMPLETE = True

_SCAN = list(config.PLANE_DIRS.values()) + ["tests"]

# Names that denote a per-token or per-million model price. Deliberately
# narrow: matching on "COST" alone would catch budget ceilings and kill
# switches, which are policy, not prices.
_RATE_NAME_RE = re.compile(r"USD_PER_TOKEN|PER_MTOKEN|USD_PER_M\b|_DRY_RUN_RATES")


# ========== shared source index ==========


def _build_index(dirs: list[str] | None = None):
    """Collect every function once; both tiers reason over the same list."""
    files = si.collect_py_files(dirs or _SCAN)
    funcs = [f for p in files for f in si.iter_functions(p)]
    return files, funcs, {f"{f.file}::{f.qualname}": f for f in funcs}


def _severity_for(site_file: str, base: str) -> str:
    """Test-file findings drop to info.

    Collapsing tests is a real win but a different kind of decision from
    collapsing production code — a merged test can tell you less on failure
    than the four it replaced. Ranking them below production keeps the digest
    ordered by what should be done first.
    """
    return "info" if site_file.startswith("tests/") else base


# ========== tier A: concept registry ==========


def _concept_findings(funcs, index) -> list[dict]:
    out: list[dict] = []

    for concept in concepts.REGISTRY:
        fp, err = concepts.resolve_fingerprint(concept, index, funcs)
        if fp is None:
            # Loud, not silent. A concept whose seed vanished matches nothing,
            # and "matches nothing" is indistinguishable from "fully fixed"
            # unless we say so — every finding under it would otherwise flip
            # to resolved on the next cycle.
            out.append(si.finding(
                PASS_NAME, "scripts/local_audit/concepts.py", 0,
                f"Concept registry seed for `{concept.id}` no longer resolves.",
                f"{err} — the concept matched nothing this cycle and its "
                f"findings will read as resolved until the seed is repointed.",
                severity="high", confidence="high",
            ))
            continue

        sites = concepts.members(concept, fp, funcs)
        if len(sites) < 2:
            continue

        canonical_key = concept.canonical
        for site in sites:
            if site.key == canonical_key:
                continue
            reason = concepts.ACCEPTED_CLONES.get(site.key)
            others = [s.key for s in sites if s.key != site.key]

            if concept.canonical_state == "accepted_duplication" or reason:
                claim = (
                    f"`{site.qualname}` implements concept `{concept.id}` "
                    f"({concept.title}); the duplication is accepted."
                )
                evidence = reason or concept.note
                severity, confidence = "info", "high"
            elif concept.canonical_state == "to_create":
                claim = (
                    f"Concept `{concept.id}` ({concept.title}) has no canonical "
                    f"implementation; `{site.qualname}` is one of several "
                    f"parallel implementations."
                )
                evidence = f"{len(sites)} matching sites. {concept.note}".strip()
                severity = _severity_for(site.file, concept.severity)
                confidence = "high"
            else:
                claim = (
                    f"`{site.qualname}` re-implements concept `{concept.id}` "
                    f"({concept.title}) instead of the canonical implementation "
                    f"at {concept.canonical}."
                )
                evidence = f"{len(sites)} matching sites. {concept.note}".strip()
                severity = _severity_for(site.file, concept.severity)
                confidence = "high"

            out.append(si.finding(
                PASS_NAME, site.file, site.lineno, claim, evidence,
                severity=severity, confidence=confidence, related_files=others,
            ))

    return out


# ========== tier A: unregistered clone discovery ==========


def _stray_rate_constant_findings(files) -> list[dict]:
    """Module-level per-token/per-million price constants outside the catalog.

    Deterministic by design, and this is the check that replaced an LLM one.
    Asked whether a span used a module-local rate, qwen3 quoted exactly the
    right line and then judged it compliant 3/3 — correctly, because nothing
    in a span reveals whether `_DIVERGENCE_EST_PROMPT_USD_PER_TOKEN` IS the
    shared catalog. The deciding fact is where the name is *defined*, which
    is a lookup, and static_tools.py's standing rule is that no LLM pass
    re-derives what a deterministic tool answers.

    Scoped to module-level assignments rather than to concept members: after
    ADR #68 the cost sites all call the shared helper, so a span-scoped check
    would have nothing left to look at, while the axiom it enforces — one
    price table — still needs guarding against the next re-introduction.
    """
    out: list[dict] = []
    for path in files:
        rel = str(path.relative_to(config.PROJECT_ROOT))
        if rel == config.RATE_CATALOG_FILE:
            continue  # the catalog itself
        tree = si.parse_file(path)
        if tree is None:
            continue
        for node in tree.body:
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            for t in targets:
                if not isinstance(t, ast.Name) or not _RATE_NAME_RE.search(t.id):
                    continue
                out.append(si.finding(
                    PASS_NAME, rel, node.lineno,
                    f"`{t.id}` is a module-local price constant outside the "
                    f"shared rate catalog.",
                    f"Model prices belong in {config.RATE_CATALOG_FILE}'s "
                    f"MODEL_RATES_USD_PER_M (ADR #68), keyed by model slug. "
                    f"Three copies of the same palette previously drifted apart "
                    f"here.",
                    severity=_severity_for(rel, "high"), confidence="high",
                ))
    return out


def _registered_site_keys(funcs, index) -> set[str]:
    keys: set[str] = set()
    for concept in concepts.REGISTRY:
        fp, err = concepts.resolve_fingerprint(concept, index, funcs)
        if fp is None:
            continue
        keys.update(s.key for s in concepts.members(concept, fp, funcs))
    return keys


def _discovery_findings(funcs, index) -> list[dict]:
    """Structural clone clusters with no registry entry yet.

    Emitted at info so they never outrank a registered concept, and gated
    hard on size — this tier's job is to surface candidates for promotion
    into concepts.py, not to be a second opinion on everything.
    """
    covered = _registered_site_keys(funcs, index) | set(concepts.ACCEPTED_CLONES)

    buckets: dict[str, list] = collections.defaultdict(list)
    for f in funcs:
        if not f.body or si.node_mass(f.body) < config.SLIMMING_MIN_FUNCTION_MASS:
            continue
        if f.length < config.SLIMMING_MIN_CLONE_LINES:
            continue
        buckets[si.fingerprint(f.body, keep_attr=False)].append(f)

    out: list[dict] = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        keys = [f"{g.file}::{g.qualname}" for g in group]
        if any(k in covered for k in keys):
            continue
        for g in group:
            others = [k for k in keys if k != f"{g.file}::{g.qualname}"]
            out.append(si.finding(
                PASS_NAME, g.file, g.lineno,
                f"`{g.qualname}` is structurally identical to other functions "
                f"and has no entry in the concept registry.",
                f"{len(group)} sites share this normalized-AST fingerprint, "
                f"~{g.length} lines each. Promote to a concept in "
                f"scripts/local_audit/concepts.py or record it as an accepted "
                f"clone with a reason.",
                severity="info", confidence="medium", related_files=others,
            ))
    return out


# ========== layer 0: vulture -> findings ==========

_VULTURE_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+): unused (?P<kind>[a-z ]+?) "
    r"'(?P<name>[^']+)' \((?P<conf>\d+)% confidence\)$"
)


def _pydantic_or_decorated(path: Path, lineno: int) -> bool:
    """True when the flagged line is a decorated def or a model field.

    vulture cannot see that @field_validator methods are called by pydantic,
    nor that annotated class attributes on a BaseModel are populated by it.
    Both are structural facts an AST check settles for free, and together
    they account for 9 of the 20 live hits.

    Note vulture reports the *first decorator's* line for a decorated def,
    not the `def` line — ast puts `node.lineno` on the `def`. Matching only
    the def line lets every @field_validator through, so the whole decorator
    block through to the def counts as the reported region.
    """
    tree = si.parse_file(path)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
            if start <= lineno <= node.lineno:
                return True
        if isinstance(node, ast.ClassDef):
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            bases |= {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
            if not any(b == "BaseModel" or b.endswith("Model") for b in bases):
                continue
            for sub in node.body:
                if isinstance(sub, (ast.AnnAssign, ast.Assign)) and sub.lineno == lineno:
                    return True
    return False


def _dynamic_session_state(path: Path, name: str) -> bool:
    """True when a Streamlit session_state key of this name is read dynamically."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "session_state" in text and (f'"{name}"' in text or f"'{name}'" in text)


def _vulture_findings(static_results: dict) -> list[dict]:
    out: list[dict] = []
    for line in static_results.get("vulture", []):
        m = _VULTURE_RE.match(line.strip())
        if not m:
            log.debug("slimming: unparsed vulture line: %s", line)
            continue
        rel, lineno = m["path"], int(m["line"])
        kind, name, conf = m["kind"], m["name"], int(m["conf"])

        # Layer 0's tools run via uvx on config.SCAN_DIRS, which includes
        # "scripts" — so they report on this harness. PY_SKIP_DIRS is only
        # enforced by the Python-side collectors, never by the subprocesses.
        if si.is_skipped_path(rel) or conf < config.VULTURE_MIN_CONFIDENCE:
            continue
        path = config.PROJECT_ROOT / rel
        if not path.exists():
            continue
        # Under tests/, vulture cannot see through Mock: every .side_effect
        # and .return_value it flags is live. Dead *functions* in a test file
        # are still worth reporting, so only the binding kinds are dropped.
        if kind in ("attribute", "variable") and rel.startswith("tests/"):
            continue
        if kind == "attribute" and _dynamic_session_state(path, name):
            continue
        if _pydantic_or_decorated(path, lineno):
            continue

        out.append(si.finding(
            PASS_NAME, rel, lineno,
            f"Unused {kind} '{name}' has no reference in the scanned tree.",
            f"vulture, {conf}% confidence. Verify by grep before deleting — "
            f"vulture's analysis is name-based, so a same-named symbol "
            f"elsewhere can mask a genuinely dead one.",
            severity=_severity_for(rel, "low"),
            confidence="high" if conf >= 90 else "medium",
        ))
    return out


# ========== layer 0: ruff F401 -> findings ==========


_PATCHED_NAMES: set[str] | None = None


def _names_patched_by_tests() -> set[str]:
    """Symbol names that tests reach for as module attributes.

    An import can be unused in its own module body and still be load-bearing:
    ingestion/dossier/index.py imports CHUNKS_CSV without referencing it, and
    tests/test_dossier_smoke.py monkeypatches it *on that module*. Deleting it
    on ruff's word breaks eight tests. Any name a test patches by string is
    therefore off-limits to the unused-import converter.
    """
    global _PATCHED_NAMES
    if _PATCHED_NAMES is not None:
        return _PATCHED_NAMES
    names: set[str] = set()
    pattern = re.compile(r"""["']([A-Za-z_][\w.]*)["']""")
    for path in si.collect_py_files(["tests"]):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "setattr" not in line and "patch(" not in line and "patch." not in line:
                continue
            for match in pattern.findall(line):
                names.add(match.rsplit(".", 1)[-1])
    _PATCHED_NAMES = names
    return names


def _ruff_findings(static_results: dict) -> list[dict]:
    out: list[dict] = []
    patched = _names_patched_by_tests()
    for hit in static_results.get("ruff", []):
        if hit.get("code") != "F401":
            continue
        symbol = (hit.get("message", "").split("`")[1:2] or [""])[0]
        if symbol and symbol.rsplit(".", 1)[-1] in patched:
            continue
        filename = hit.get("filename", "")
        try:
            rel = str(Path(filename).relative_to(config.PROJECT_ROOT))
        except ValueError:
            rel = filename
        if si.is_skipped_path(rel):
            continue
        # Package re-exports are deliberate: ruff itself suggests __all__
        # rather than deletion, and removing them changes the public surface.
        if Path(rel).name == "__init__.py":
            continue
        out.append(si.finding(
            PASS_NAME, rel, (hit.get("location") or {}).get("row", 0),
            f"Unused import: {hit.get('message', 'F401')}.",
            "ruff F401. Safe to delete unless it is an intentional re-export.",
            severity=_severity_for(rel, "low"), confidence="high",
        ))
    return out


# ========== layer 0: size budgets ==========


def _length_findings(static_results: dict) -> list[dict]:
    """Functions over the per-function line budget, from radon cc's spans.

    radon already reports lineno/endline per block, so this costs nothing
    beyond a subtraction. The count stays out of the claim: it changes on
    every edit, and claim text is finding identity.
    """
    out: list[dict] = []
    for key, blocks in (static_results.get("radon_cc") or {}).items():
        rel = key.lstrip("./")
        if si.is_skipped_path(rel):
            continue
        for b in blocks:
            length = b.get("endline", 0) - b.get("lineno", 0) + 1
            if length <= config.SLIMMING_FUNCTION_LINE_BUDGET:
                continue
            name = b.get("name", "?")
            qual = f"{b['classname']}.{name}" if b.get("classname") else name
            out.append(si.finding(
                PASS_NAME, rel, b.get("lineno", 0),
                f"`{qual}` exceeds the per-function length budget.",
                f"{length} lines (budget {config.SLIMMING_FUNCTION_LINE_BUDGET}), "
                f"radon rank {b.get('rank', '?')}, complexity {b.get('complexity', '?')}.",
                severity=_severity_for(rel, "info"), confidence="high",
            ))
    return out


def _docstring_mass_findings(files) -> list[dict]:
    """Files whose docstrings dominate the line count.

    This is the metric that speaks to the actual complaint — files being too
    long — as opposed to complexity or maintainability. The convention this
    repo follows is spec-first docstrings, so the finding is not "write less
    documentation": it is that rationale prose belongs in an ADR and only the
    contract belongs in the docstring.
    """
    out: list[dict] = []
    for path in files:
        rel = str(path.relative_to(config.PROJECT_ROOT))
        tree = si.parse_file(path)
        if tree is None:
            continue
        try:
            total = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if total < 100:
            continue

        doc_lines = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                d = body[0]
                doc_lines += (d.end_lineno or d.lineno) - d.lineno + 1

        ratio = doc_lines / total
        if ratio <= config.SLIMMING_DOCSTRING_RATIO or doc_lines < config.SLIMMING_DOCSTRING_MIN_LINES:
            continue
        out.append(si.finding(
            PASS_NAME, rel, 1,
            "File's docstring mass exceeds the contract/rationale split budget.",
            f"{doc_lines} docstring lines of {total} ({ratio:.0%}). Move design "
            f"rationale to its ADR and keep the contract (arguments, returns, "
            f"invariants, failure modes) in the docstring.",
            severity=_severity_for(rel, "info"), confidence="high",
        ))
    return out


# ========== tier C: concept divergence (LLM) ==========

DIVERGENCE_SYSTEM_PROMPT = """\
You check ONE short code span against ONE written rule from this codebase's
concept registry.

The span is ALREADY KNOWN to implement this concept — a deterministic
fingerprint match established that before you were called. Do not re-decide
it, do not comment on the span's overall purpose, and do not consider what
the surrounding function is for. Only the span and the rule are in scope.

For each numbered clause of the rule, answer exactly one of:
  "satisfied"      — a line in the span does what the clause requires
  "violated"       — a line in the span does something the clause forbids,
                     or does the required thing differently
  "not_applicable" — no line in the span decides this clause

Quote the exact line from the span that decides each clause. If you cannot
quote a line, the answer is "not_applicable". Never guess.

Respond ONLY with valid JSON:
{"any_violation": true|false,
 "clauses": [{"n": 1, "verdict": "satisfied"|"violated"|"not_applicable",
              "evidence": "<exact line copied from the span>"}]}
"""


def _divergence_user_prompt(concept, span: str, grounding: str) -> str:
    clauses = "\n".join(f"{i}. {c}" for i, c in enumerate(concept.rule, 1))
    return (
        f"Concept: {concept.id} — {concept.title}\n\n"
        f"Rule clauses:\n{clauses}\n\n"
        f"Code span:\n{span}\n\n"
        f"{grounding}\n"
    )


def run_divergence(static_results: dict | None = None) -> list[dict]:
    """Check each non-canonical member of an llm_divergence concept against
    its rule clauses.

    Cache identity deliberately excludes the line number: an edit anywhere
    above the span would otherwise miss the cache and burn a call for an
    unchanged span. The concept's rule_version rides in the key so editing
    one rule invalidates only that concept's verdicts.
    """
    global LAST_DIVERGENCE_COMPLETE
    LAST_DIVERGENCE_COMPLETE = True
    static_results = static_results or {}

    from scripts.local_audit import static_tools

    _, funcs, index = _build_index()
    prompt_version = config.PROMPT_VERSIONS["slimming_divergence"]
    budget = config.SLIMMING_LLM_MAX_CALLS_PER_CYCLE
    out: list[dict] = []

    for concept in concepts.REGISTRY:
        if not concept.llm_divergence or not concept.rule:
            continue
        fp, err = concepts.resolve_fingerprint(concept, index, funcs)
        if fp is None:
            log.warning("slimming: divergence skipped, %s", err)
            continue

        for site in concepts.members(concept, fp, funcs):
            if site.key in concepts.ACCEPTED_CLONES:
                continue
            path = config.PROJECT_ROOT / site.file
            span = si.span_text(path, site.span_start, site.span_end)
            if not span:
                continue

            cache_id = f"{concept.id}@{concept.rule_version}|{site.key}"
            code_hash = cache.code_hash_for_text(span)
            result = cache.get(DIVERGENCE_PASS_NAME, prompt_version, cache_id, code_hash)

            if result is None:
                if budget <= 0:
                    # Budget-blocked, so this cycle did not re-check every
                    # site. Reconciling now would resolve still-valid findings.
                    LAST_DIVERGENCE_COMPLETE = False
                    continue
                budget -= 1
                grounding = static_tools.grounding_for_file(site.file, static_results)
                verdict, meta = ollama_client.call_self_consistency(
                    DIVERGENCE_SYSTEM_PROMPT,
                    _divergence_user_prompt(concept, span, grounding),
                    verdict_key="any_violation",
                    think=False,
                )
                if verdict is None:
                    # No verdict is not a negative verdict — never cache it,
                    # so a transport failure retries next cycle.
                    LAST_DIVERGENCE_COMPLETE = False
                    continue
                result = dict(verdict)
                result["_self_consistency"] = meta
                cache.set(DIVERGENCE_PASS_NAME, prompt_version, cache_id, code_hash, result)

            if not result.get("any_violation"):
                continue
            for clause in result.get("clauses", []):
                if clause.get("verdict") != "violated":
                    continue
                n = clause.get("n")
                if not isinstance(n, int) or not (1 <= n <= len(concept.rule)):
                    continue
                meta = result.get("_self_consistency") or {}
                out.append(si.finding(
                    DIVERGENCE_PASS_NAME, site.file, site.lineno,
                    f"`{site.qualname}` implements concept `{concept.id}` but "
                    f"violates clause {n}: {concept.rule[n - 1]}",
                    f"model quoted: {str(clause.get('evidence', ''))[:200]} "
                    f"(self-consistency {meta.get('agree', '?')}/{meta.get('runs', '?')})",
                    severity=_severity_for(site.file, concept.severity),
                    confidence="medium",
                ))

    return out


# ========== orchestration ==========


def run(static_results: dict | None = None, dirs: list[str] | None = None) -> list[dict]:
    """Deterministic tier: registry matches, discovery, and Layer 0 converters."""
    static_results = static_results or {}
    files, funcs, index = _build_index(dirs)

    out: list[dict] = []
    out += _concept_findings(funcs, index)
    out += _stray_rate_constant_findings(files)
    out += _discovery_findings(funcs, index)
    out += _vulture_findings(static_results)
    out += _ruff_findings(static_results)
    out += _length_findings(static_results)
    out += _docstring_mass_findings(files)

    log.info("slimming: %d deterministic finding(s) across %d functions",
             len(out), len(funcs))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from scripts.local_audit import static_tools
    results = run(static_tools.run_all())
    print(json.dumps(results, indent=2, ensure_ascii=False))
