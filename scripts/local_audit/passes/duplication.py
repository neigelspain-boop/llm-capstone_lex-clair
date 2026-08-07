"""Pass 2: cross-file semantic duplication.

Candidate generation happens BEFORE any LLM call, to avoid O(n^2) pairwise
comparison across the repo's ~17K LOC: BGE-M3 embedding similarity (the same
model/flags ingestion/load.py already uses for retrieval, so no new
dependency) plus a small keyword-bucket boost seeded from the one duplication
finding already confirmed by hand this session (rag/generate.py vs
rag/compliance.py disagreeing on how to compute LLM cost). Only the capped
candidate list goes to the LLM, one judgment call per pair.

Calibration note: candidate generation is validated against that known
pair — it lands at embedding-similarity rank 17 of 421 keyword-eligible
pairs, comfortably inside the default budget, and the captured text (after
the keyword-windowed body extraction below) does show both functions' real
cost-computation code, not just their docstrings. The LLM's *judgment* on
that specific pair is wrong at both qwen3:14b AND qwen3:30b (MoE, think=True,
hybrid GPU+CPU) — 3/3 self-consistency, same conclusion, near-identical
reasoning from both models: "different overall responsibility (compliance-
specific vs. general-purpose), therefore not the same concept." Since a
6x-larger model reached the identical wrong answer, this isn't primarily a
model-capability ceiling — it's DUPLICATION_SYSTEM_PROMPT asking the wrong-
shaped question. "Do these functions have the same overall responsibility"
correctly gets "no" for two functions with different top-level purposes,
even when they share one narrow computation (cost calculation) that
actually diverges. A sharper prompt would ask about shared *sub-behaviors*
within otherwise-different functions, not overall sameness — untried here;
flagging as the concrete next thing to try rather than re-tuning blind.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

from scripts.local_audit import cache, config, ollama_client, static_tools

log = logging.getLogger(__name__)

DUPLICATION_SYSTEM_PROMPT = """You review two functions from the same codebase that a similarity search flagged as possibly handling the same responsibility.

Decide:
1. same_concept: do these two functions genuinely do the same kind of job (not just similar names/shape)?
2. diverges_unintentionally: if same_concept is true, do they handle it in meaningfully different ways that look like unplanned drift (inconsistent logic, one handling an edge case the other doesn't, different correctness posture) rather than a deliberate, well-motivated difference?

Be conservative — if the difference looks like an intentional design choice (documented in a comment, or clearly justified by different contexts), same_concept or diverges_unintentionally should be false.

Respond ONLY with valid JSON:
{"same_concept": true|false, "diverges_unintentionally": true|false, "explanation": "one or two sentences", "severity": "critical"|"high"|"medium"|"low"}
severity only matters when diverges_unintentionally is true.
"""


@dataclass
class FuncEntry:
    qualname: str
    file: str
    lineno: int
    text: str  # name + docstring + relevant body snippet(s), used both for embedding and keyword matching


# ========== collection ==========


def _relevant_body_snippet(body_lines: list[str], radius: int = 6, max_windows: int = 2,
                            head_fallback: int = 20) -> list[str]:
    """Prefer body lines near a DUPLICATION_KEYWORDS hit — the concept a
    pair was likely flagged for — over blindly the first N lines. Functions
    in this codebase run long (one candidate was 128 lines), so a flat head
    slice reliably captures the docstring and reliably misses the actual
    implementation further down; a keyword-centered window finds it instead.
    Falls back to the head when no keyword appears in the body at all (true
    for pure-embedding candidates, which weren't chosen via keyword match).
    """
    hit_indices = [
        i for i, line in enumerate(body_lines)
        if any(kw in line.lower() for kw in config.DUPLICATION_KEYWORDS)
    ]
    if not hit_indices:
        return body_lines[:head_fallback]

    spans: list[tuple[int, int]] = []
    for idx in hit_indices:
        if len(spans) >= max_windows:
            break
        lo, hi = max(0, idx - radius), min(len(body_lines), idx + radius + 1)
        if any(not (hi <= s_lo or lo >= s_hi) for s_lo, s_hi in spans):
            continue  # overlaps a window already taken
        spans.append((lo, hi))

    spans.sort()
    out: list[str] = []
    for lo, hi in spans:
        if out:
            out.append("...")
        out.extend(body_lines[lo:hi])
    return out or body_lines[:head_fallback]


def _collect_functions(dirs: list[str]) -> list[FuncEntry]:
    entries: list[FuncEntry] = []
    for d in dirs:
        base = config.PROJECT_ROOT / d
        if not base.exists():
            continue
        for py_file in sorted(base.rglob("*.py")):
            rel = py_file.relative_to(config.PROJECT_ROOT)
            if any(part in config.PY_SKIP_DIRS for part in rel.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except SyntaxError:
                continue
            lines = source.splitlines()

            def add(node, qualname: str):
                # This codebase's own convention (CLAUDE.md: "spec first,
                # docstring-level specification written and reviewed before
                # implementation") means most functions carry a substantial
                # docstring — capturing a flat N lines from the `def` line
                # mostly captures the docstring and never reaches real code.
                # Get the docstring separately (unlimited length) and start
                # the body-line window *after* it, so the LLM actually sees
                # implementation, not just the spec.
                doc = ast.get_docstring(node) or ""
                stmts = node.body
                code_stmts = stmts
                if (stmts and isinstance(stmts[0], ast.Expr)
                        and isinstance(getattr(stmts[0], "value", None), ast.Constant)
                        and isinstance(stmts[0].value.value, str)):
                    code_stmts = stmts[1:]
                body_start = (code_stmts[0].lineno - 1) if code_stmts else (node.lineno - 1)
                body_end = node.end_lineno if getattr(node, "end_lineno", None) else body_start + 20
                full_body = lines[body_start:body_end]
                snippet = _relevant_body_snippet(full_body)
                text = f"{qualname}\n{doc}\n--- body ---\n" + "\n".join(snippet)
                entries.append(FuncEntry(qualname, str(rel), node.lineno, text))

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(node, node.name)
                elif isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            add(sub, f"{node.name}.{sub.name}")
    return entries


# ========== candidate generation ==========


def _pair_key(a: FuncEntry, b: FuncEntry) -> tuple[str, str]:
    return tuple(sorted([f"{a.file}:{a.qualname}", f"{b.file}:{b.qualname}"]))


def _keyword_eligible_pairs(entries: list[FuncEntry]) -> set[tuple[str, str]]:
    """Cross-file pairs sharing >=1 keyword — a *prefilter*, not a ranking.
    Keyword-sharing alone is common enough in this repo (421 cross-file
    pairs share at least one of the 7 keywords) that shared-keyword COUNT
    turned out to be a poor ranking signal: a keyword-dense docstring on one
    unrelated function (mentioning several of retry/cost/truncat/etc. while
    documenting general robustness behavior) out-scores the real target
    pair, which shares only the single keyword "cost" but is the actual
    match. Embedding similarity (see generate_candidates) does the ranking;
    this only decides the eligible pool.
    """
    kw_sets = []
    for e in entries:
        low = e.text.lower()
        kw_sets.append(frozenset(kw for kw in config.DUPLICATION_KEYWORDS if kw in low))

    candidate_idx = [i for i, kws in enumerate(kw_sets) if kws]
    eligible: set[tuple[str, str]] = set()
    for ii in range(len(candidate_idx)):
        i = candidate_idx[ii]
        for jj in range(ii + 1, len(candidate_idx)):
            j = candidate_idx[jj]
            a, b = entries[i], entries[j]
            if a.file == b.file:
                continue
            if kw_sets[i] & kw_sets[j]:
                eligible.add(_pair_key(a, b))
    return eligible


def _similarity_matrix(entries: list[FuncEntry]):
    """Encode every entry once (BGE-M3, same model/flags as ingestion/load.py
    uses for retrieval) and return the full cosine-similarity matrix. One
    embedding pass serves both the keyword-filtered ranking and the pure-
    embedding discovery slice below — embedding compute is cheap and
    batchable up into the low thousands of entries; the O(n^2) cost this
    harness actually needs to avoid is LLM *judgment* calls, not embedding.
    Returns None if the embedding deps aren't available.
    """
    try:
        import numpy as np
        from FlagEmbedding import BGEM3FlagModel

        from ingestion.index import EMBED_MODEL_ID, infer_device
    except ImportError as e:
        log.warning("duplication: embedding deps unavailable (%s); skipping embedding-based ranking", e)
        return None

    device = infer_device(None)
    model = BGEM3FlagModel(EMBED_MODEL_ID, use_fp16=(device == "cuda"), device=device)
    vecs = model.encode(
        [e.text for e in entries], return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )["dense_vecs"]
    vecs = np.asarray(vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    return unit @ unit.T


def generate_candidates(dirs: list[str] | None = None) -> list[tuple[FuncEntry, FuncEntry]]:
    dirs = dirs or list(config.PLANE_DIRS.values())
    entries = _collect_functions(dirs)
    log.info("duplication: %d function/method entries collected", len(entries))
    if len(entries) < 2:
        return []

    kw_eligible = _keyword_eligible_pairs(entries)
    log.info("duplication: %d cross-file pair(s) share >=1 keyword", len(kw_eligible))

    sims = _similarity_matrix(entries)
    if sims is None:
        # no embedding available — fall back to keyword pool alone, capped
        # arbitrarily rather than not running the pass at all.
        pairs = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if _pair_key(entries[i], entries[j]) in kw_eligible:
                    pairs.append((entries[i], entries[j]))
        return pairs[: config.DUPLICATION_TOP_K]

    # reserve a slice of the budget for pure-embedding discovery (pairs that
    # share no keyword at all but are highly similar) so the keyword list
    # isn't the only way a candidate can surface.
    embedding_only_budget = max(4, config.DUPLICATION_TOP_K // 5)
    keyword_budget = config.DUPLICATION_TOP_K - embedding_only_budget

    keyword_scored = []
    embedding_only_scored = []
    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = entries[i], entries[j]
            if a.file == b.file:
                continue
            score = float(sims[i, j])
            if _pair_key(a, b) in kw_eligible:
                keyword_scored.append((score, a, b))
            else:
                embedding_only_scored.append((score, a, b))

    keyword_scored.sort(key=lambda t: t[0], reverse=True)
    embedding_only_scored.sort(key=lambda t: t[0], reverse=True)

    return (
        [(a, b) for _, a, b in keyword_scored[:keyword_budget]]
        + [(a, b) for _, a, b in embedding_only_scored[:embedding_only_budget]]
    )


# ========== LLM judgment ==========


def _pair_cache_key(a: FuncEntry, b: FuncEntry) -> tuple[str, str]:
    """A pair's identity for caching. code_hash covers both functions'
    captured text (not the whole file) so an unrelated edit elsewhere in
    either file doesn't force a re-judgment of a pair that hasn't actually
    changed.
    """
    file_path = f"{a.file}:{a.qualname}|{b.file}:{b.qualname}"
    code_hash = cache.code_hash_for_text(a.text + "\x00" + b.text)
    return file_path, code_hash


def _judge_pair(a: FuncEntry, b: FuncEntry, static_results: dict) -> dict | None:
    prompt_version = config.PROMPT_VERSIONS["duplication_verify"]
    file_path, code_hash = _pair_cache_key(a, b)

    cached = cache.get("duplication_verify", prompt_version, file_path, code_hash)
    if cached is not None:
        return cached

    grounding = (
        static_tools.grounding_for_file(a.file, static_results)
        + "\n\n"
        + static_tools.grounding_for_file(b.file, static_results)
    )
    user_prompt = (
        f"Function A: {a.qualname} ({a.file}:{a.lineno})\n{a.text}\n\n"
        f"Function B: {b.qualname} ({b.file}:{b.lineno})\n{b.text}\n\n"
        f"{grounding}"
    )
    # Weighing whether a divergence is unintentional needs real comparative
    # reasoning, not a lookup — worth both the extra latency of thinking mode
    # and the larger judgment model (see config.OLLAMA_MODEL_JUDGMENT).
    # Validated against the known rag/generate.py vs rag/compliance.py
    # cost-calc pair: qwen3:14b consistently missed the divergence 3/3 even
    # with think=True and both functions' actual cost logic in view.
    result, meta = ollama_client.call_self_consistency(
        DUPLICATION_SYSTEM_PROMPT, user_prompt, verdict_key="diverges_unintentionally",
        think=True, model=config.OLLAMA_MODEL_JUDGMENT,
    )
    if result is not None:
        result["_self_consistency"] = meta
        cache.set("duplication_verify", prompt_version, file_path, code_hash, result)
    return result


# ========== orchestration ==========


def run(static_results: dict | None = None, dirs: list[str] | None = None) -> list[dict]:
    static_results = static_results if static_results is not None else static_tools.run_all()
    candidates = generate_candidates(dirs)
    log.info("duplication: %d candidate pair(s) after generation", len(candidates))

    out_findings: list[dict] = []
    for a, b in candidates:
        result = _judge_pair(a, b, static_results)
        if result is None:
            continue
        if not (result.get("same_concept") and result.get("diverges_unintentionally")):
            continue

        out_findings.append({
            "pass": "duplication",
            "file": a.file,
            "line": a.lineno,
            "related_files": [b.file],
            "claim": (
                f"{a.qualname} ({a.file}) and {b.qualname} ({b.file}) handle the same "
                f"concept but appear to diverge unintentionally."
            ),
            "evidence": result.get("explanation", ""),
            "severity": result.get("severity", "medium"),
            "confidence": "medium",
        })

    return out_findings
