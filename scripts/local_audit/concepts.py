"""The axiom registry: concepts this codebase should implement exactly once.

A *concept* is one narrow behaviour — "strip ```json fences off an LLM
response", "turn a response's usage object into a USD figure" — together
with the written rule that behaviour must satisfy and the location of its
canonical implementation. Everything the slimming pass reports is a
deviation from an entry here.

Two design rules make this work where passes/duplication.py did not:

1. **Membership is decided deterministically, never by the LLM.** A concept
   names a *seed site*; the pass fingerprints that seed's normalized AST at
   run time and every structural match is a member. The local model is only
   ever asked whether an already-identified member violates a written rule.
   duplication.py's calibration failure was whole-function anchoring — asked
   whether two functions share a responsibility, both qwen3:14b and qwen3:30b
   compressed each 100+ line function to its dominant purpose and lost the
   8-line block that actually mattered. Shrinking the unit of analysis is the
   fix; classification framing alone is not.

2. **Fingerprints are derived from seeds at run time, never hardcoded.** A
   literal hex fingerprint in this file would go stale the moment its
   canonical implementation was touched, silently emptying the concept — and
   because findings resolve when a pass stops reproducing them, every finding
   under that concept would flip to "resolved" as if it had been fixed. When
   a seed no longer resolves the pass says so out loud instead.

Concept ids are permanent: they appear in claim strings, and findings.make_id
hashes the claim. Renaming one re-mints every finding under it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from scripts.local_audit import source_index as si

log = logging.getLogger(__name__)


# ========== schema ==========


@dataclass(frozen=True)
class Concept:
    """One canonical behaviour, its rule, and how to find its implementations.

    `detector` selects the matching strategy:
      - "function_clone": members are whole functions whose body fingerprint
        equals the seed function's (attribute names erased).
      - "stmt_window": members are `window_k`-statement runs whose fingerprint
        equals the seed window's (attribute names kept — at window scale
        structure alone is too weak to identify a concept).

    `canonical_state`:
      - "exists": `canonical` is the implementation everything should defer to.
      - "to_create": no canonical yet; every member is a parallel
        implementation and the fix is to create one.
      - "accepted_duplication": deliberately kept. Emitted at info severity
        with the reason attached, so the ledger stays visible rather than
        being silently dropped from the scan.

    `rule` clauses are prose, one behaviour each, and are the *only* text the
    LLM tier is asked to check against. They are registry-authored, never
    model output — that is what keeps divergence claims stable across
    self-consistency runs.
    """
    id: str
    title: str
    canonical: str
    canonical_state: str
    detector: str
    seed_site: str
    severity: str
    rule: tuple[str, ...] = ()
    rule_version: str = "v1"
    seed_marker: str = ""
    window_k: int = 0
    # Statements past the fingerprint window to include in the span shown to
    # the LLM. The window is *identity* — it must stay narrow or it stops
    # matching across sites — but a rule can reference behaviour just outside
    # it. llm_cost_from_usage is the case in point: its k=4 window captures
    # the token extraction, while the rate constants that clause 3 asks about
    # live in the very next statement. Without this the model correctly
    # answers "not_applicable" and a real divergence goes unreported.
    span_extra_stmts: int = 0
    llm_divergence: bool = False
    # Regex over identifier names appearing in a member's span. A match means
    # the span reaches for a module-local constant where the concept requires
    # a shared one. Deterministic on purpose — see the calibration note on
    # llm_cost_from_usage for why this specific question must not go to the
    # model.
    local_constant_pattern: str = ""
    adr: str | None = None
    note: str = ""
    exempt: frozenset[str] = frozenset()
    exempt_reason: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Site:
    """One concrete location matching a concept."""
    concept_id: str
    file: str
    qualname: str
    lineno: int
    span_start: int
    span_end: int

    @property
    def key(self) -> str:
        return f"{self.file}::{self.qualname}"


# ========== accepted clones (the "we decided to keep this" ledger) ==========

# Sites the detector correctly identifies as structural clones but which must
# NOT be merged. Suppressed from both registry findings and unregistered
# cluster discovery. Every entry needs a reason, and the reason is rendered
# into the digest — a suppression nobody can read is indistinguishable from
# a bug in the detector.
ACCEPTED_CLONES: dict[str, str] = {
    "app/streamlit_app.py::get_flow":
        "@st.cache_resource keys on the decorated function's identity; merging "
        "the two accessors would collapse two independent caches into one.",
    "app/streamlit_app.py::get_compliance":
        "Same as get_flow — separate cache identity is the point of the "
        "duplication, not an accident.",
    "ingestion/clients.py::get_anthropic_client":
        "Deprecation shim with live callers (3 modules plus 12 patch targets "
        "in tests/test_dossier_smoke.py). Its two siblings have none.",
    "ingestion/dossier/index.py::_split_page_text":
        "Dossier chunking is a different algorithm from ingestion/chunk.py's "
        "statute chunker by design (ADR #39); shape overlap is coincidental.",
}


# ========== the registry ==========

REGISTRY: tuple[Concept, ...] = (

    # ---------- LLM response handling ----------

    Concept(
        id="llm_json_fence_strip",
        title="strip markdown code fences off an LLM JSON response",
        canonical="",
        canonical_state="to_create",
        detector="stmt_window",
        seed_site="ingestion/dossier/facts.py::_parse_llm_json",
        seed_marker='text.startswith("```")',
        window_k=2,
        severity="medium",
        adr="candidate #67",
        rule=(
            "The fence-stripping logic is called from a single shared helper, "
            "not re-implemented at the call site.",
        ),
        note=(
            "Eight sites carry the identical four lines. Only the strip itself "
            "is common: the decoder (json.loads vs raw_decode) and the failure "
            "policy (raise / return None / return []) differ by caller "
            "contract and must NOT be unified — gate.py raises so its caller "
            "can record status='parse_failed', compliance.py returns [] after "
            "_recover_partial_entries salvage. Extract the pure string->string "
            "part only."
        ),
    ),

    Concept(
        id="llm_json_parse_full",
        title="parse an LLM JSON response: fence strip, raw_decode, warn-and-None",
        canonical="ingestion/dossier/facts.py::_parse_llm_json",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/dossier/facts.py::_parse_llm_json",
        severity="high",
        adr="candidate #67",
        rule=(
            "There is one implementation of the fence-strip + raw_decode + "
            "warn-and-return-None sequence, parameterised by the expected "
            "top-level container type.",
        ),
        note=(
            "mentions.py and resolve.py are byte-identical to facts.py after "
            "renaming, and both docstrings admit the mirroring. All three "
            "share the same failure contract, so unlike the fence-strip "
            "concept these can be collapsed whole."
        ),
    ),

    # ---------- cost accounting ----------

    Concept(
        id="llm_cost_from_usage",
        title="derive prompt/completion tokens and a USD cost from a response's usage",
        canonical="",
        canonical_state="to_create",
        detector="stmt_window",
        seed_site="ingestion/dossier/distill.py::distill_fact",
        seed_marker="completion_tokens = getattr",
        window_k=4,
        span_extra_stmts=1,
        severity="high",
        adr="candidate #68",
        llm_divergence=False,
        local_constant_pattern=r"_EST_|USD_PER_TOKEN|_DRY_RUN_RATES",
        rule=(
            "Token counts are read from the response's usage object "
            "defensively, tolerating a missing usage attribute.",
            "The provider-reported cost is preferred when present, and the "
            "estimate is only a fallback.",
            "The per-token rate constants come from the shared cost catalog, "
            "not from module-local constants or inline numeric literals.",
        ),
        note=(
            "Three incompatible rate representations exist across five files: "
            "per-million float pairs (eval/llm_eval.py COST_PER_MTOKEN), "
            "per-token float pairs (_EST_*_USD_PER_TOKEN in distill/mentions/"
            "resolve), and a catalog dict (rag/generate.py ANSWER_MODELS). "
            "eval/llm_eval.py's dry-run additionally hardcodes rates that "
            "already exist in its own COST_PER_MTOKEN.\n\n"
            "CALIBRATION — the LLM tier was tried here and switched off. This "
            "is passes/duplication.py's motivating case, which it fails 3/3 on "
            "both qwen3:14b and qwen3:30b. The rewritten single-span, "
            "clause-checklist prompt did fix the framing problem: the model "
            "stopped reasoning about the enclosing function and quoted exactly "
            "the right line for clause 3 — `prompt_tokens * "
            "_DIVERGENCE_EST_PROMPT_USD_PER_TOKEN`. It then marked it "
            "'satisfied', 3/3. Correctly so, on the evidence it had: nothing "
            "in the span says whether that identifier IS the shared catalog. "
            "The missing fact is where the constant is *defined*, which is a "
            "symbol lookup, not a judgment — and static_tools.py's own rule is "
            "that no LLM pass may re-derive what a deterministic tool already "
            "answers. So clause 3 became local_constant_pattern below. Clauses "
            "1 and 2 stay as documentation but need no model either: all five "
            "sites are byte-identical, so they are satisfied everywhere by "
            "construction. The lesson generalises — before sending a clause to "
            "the model, check that the span actually contains what decides it."
        ),
    ),

    # ---------- persistence idioms ----------

    Concept(
        id="content_hash_cache_load",
        title="load a content-hash-keyed JSON cache, invalidating on hash mismatch",
        canonical="ingestion/dossier/mentions.py::_load_cache",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/dossier/mentions.py::_load_cache",
        severity="medium",
        rule=(),
        note="mentions.py and resolve.py only; both are Plane Ib, so a shared "
             "helper stays inside one plane.",
    ),

    Concept(
        id="content_hash_cache_write",
        title="atomically write a content-hash-keyed JSON cache",
        canonical="ingestion/dossier/mentions.py::_write_cache_atomic",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/dossier/mentions.py::_write_cache_atomic",
        severity="medium",
        rule=(),
        note="Nine identical lines shared with resolve.py. Same plane.",
    ),

    Concept(
        id="jsonl_append_cache_load",
        title="load a JSONL append-cache into a dict keyed by cache key",
        canonical="",
        canonical_state="accepted_duplication",
        detector="function_clone",
        seed_site="ingestion/dossier/distill.py::_load_distill_cache",
        severity="info",
        rule=(),
        note=(
            "distill.py (Plane Ib) and compliance.py (Plane II). Nine identical "
            "lines, but merging them needs a shared module reachable from both "
            "planes, and CLAUDE.md sanctions exactly one such exception "
            "(ingestion/clients.py) which is about credential wiring, not "
            "persistence. Kept deliberately: the indirection would cost more "
            "than the 9 lines it saves."
        ),
    ),

    Concept(
        id="jsonl_append_cache_append",
        title="append one entry to a JSONL append-cache",
        canonical="",
        canonical_state="accepted_duplication",
        detector="function_clone",
        seed_site="ingestion/dossier/distill.py::_append_distill_cache",
        severity="info",
        rule=(),
        note="Same Plane Ib/II split as jsonl_append_cache_load.",
    ),

    Concept(
        id="jsonl_read",
        title="read a JSONL file into a list, skipping blank lines",
        canonical="rag/compliance.py::_read_jsonl",
        canonical_state="exists",
        detector="function_clone",
        seed_site="rag/compliance.py::_read_jsonl",
        severity="low",
        rule=(),
        note="_read_jsonl and _read_jsonl_dicts sit 16 lines apart in the same "
             "file and differ only in whether a pydantic model is applied.",
    ),

    # ---------- table-drivable families ----------

    Concept(
        id="dossier_write_jsonl",
        title="write a deduped, sorted JSONL artifact for one dossier case",
        canonical="ingestion/dossier/facts.py::_write_facts_jsonl",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/dossier/facts.py::_write_facts_jsonl",
        severity="medium",
        rule=(),
        note=(
            "Three near-identical writers differing only in filename, model "
            "class, dedup field and sort key. _write_actor_roles_jsonl is "
            "expected NOT to match at the exact-fingerprint tier (it sorts on "
            "a tuple, the others on a single key) — that near-miss is correct "
            "behaviour, not a detector bug, and it still belongs in the same "
            "table-driven collapse."
        ),
    ),

    Concept(
        id="eval_judge_fn",
        title="dispatch one eval judge call for a named model",
        canonical="eval/llm_eval.py::judge_gpt",
        canonical_state="exists",
        detector="function_clone",
        seed_site="eval/llm_eval.py::judge_gpt",
        severity="medium",
        rule=(),
        note="Three functions identical modulo JUDGE_MODELS[key]. The JUDGES "
             "name->function table already exists in the same file, so the "
             "table is not the missing piece — the three functions are the "
             "redundancy.",
    ),

    Concept(
        id="pipeline_stage_runner",
        title="run one named build stage, logging start/finish and timing it",
        canonical="ingestion/build.py::_run_stage",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/build.py::_run_stage",
        severity="medium",
        rule=(),
        note="ingestion/dossier/build.py's copy is byte-identical including "
             "the docstring. Plane I and Plane Ib share an import namespace, "
             "so importing across is not a boundary violation.",
    ),

    Concept(
        id="deprecated_client_shim",
        title="provider-named client alias superseded by get_openrouter_client",
        canonical="ingestion/clients.py::get_openrouter_client",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/clients.py::get_anthropic_client",
        severity="low",
        adr="#40",
        rule=(),
        note=(
            "ADR #40 made get_openrouter_client the single factory. Three "
            "provider-named shims were kept for callers mid-transition; "
            "get_openai_client and get_mistral_client were deleted once that "
            "transition finished with no callers left. get_anthropic_client "
            "survives (12 patch targets in tests/test_dossier_smoke.py plus 3 "
            "live call sites) and is now the seed, so the concept keeps its "
            "shape and fires again if a fourth shim ever appears. Vulture "
            "found only one of the two dead shims — it missed "
            "get_openai_client because app/streamlit_app.py defines a "
            "same-named function and vulture's analysis is name-based, not "
            "import-aware. The clone cluster is what caught it."
        ),
    ),

    Concept(
        id="fold_replace_longest_first",
        title="apply a replacement map to text, longest key first, case-folded",
        canonical="ingestion/dossier/anonymize.py::_apply_entity_map",
        canonical_state="exists",
        detector="function_clone",
        seed_site="ingestion/dossier/anonymize.py::_apply_entity_map",
        severity="low",
        rule=(),
        note="Both sites are in anonymize.py, 70 lines apart.",
    ),
)


REGISTRY_BY_ID = {c.id: c for c in REGISTRY}


# ========== seed resolution ==========


def _find_seed_function(seed_site: str, index: dict[str, si.FuncNode]) -> si.FuncNode | None:
    return index.get(seed_site)


def resolve_fingerprint(concept: Concept, index: dict[str, si.FuncNode],
                        funcs: list[si.FuncNode]) -> tuple[str | None, str]:
    """Compute this concept's fingerprint from its seed site, at run time.

    Returns (fingerprint, error_message). A None fingerprint means the seed
    could not be resolved — the caller must emit a loud finding and skip the
    concept rather than treating "no members" as "nothing to report".

    For window concepts the marker usually appears in several *overlapping*
    windows (a 4-statement window slid across a 6-statement block contains
    the marker three times), and those windows have different fingerprints
    with different reach. Picking the first match would silently under-count:
    seeding llm_cost_from_usage on the earliest marker-containing window
    finds 3 sites, one statement later finds 5, because the earlier window
    drags in a preceding statement that only three of the five callers share.
    So: among marker-containing windows, take the one with the widest reach,
    ties broken by earliest position for determinism.
    """
    seed = _find_seed_function(concept.seed_site, index)
    if seed is None:
        return None, f"seed site {concept.seed_site} not found"

    if concept.detector == "function_clone":
        if not seed.body:
            return None, f"seed function {concept.seed_site} has an empty body"
        return si.fingerprint(seed.body, keep_attr=False), ""

    if concept.detector == "stmt_window":
        from pathlib import Path

        from scripts.local_audit import config
        path = Path(config.PROJECT_ROOT) / seed.file

        candidates: list[tuple[int, int, str]] = []
        for _, window in si.statement_windows(seed, concept.window_k):
            start = window[0].lineno
            end = window[-1].end_lineno or window[-1].lineno
            if concept.seed_marker not in si.span_text(path, start, end):
                continue
            fp = si.fingerprint(window, keep_attr=True)
            reach = len(members(concept, fp, funcs))
            candidates.append((reach, -start, fp))

        if not candidates:
            return None, (
                f"seed marker {concept.seed_marker!r} not found in any "
                f"{concept.window_k}-statement window of {concept.seed_site}"
            )
        return max(candidates)[2], ""

    return None, f"unknown detector {concept.detector!r}"


def members(concept: Concept, fp: str, funcs: list[si.FuncNode]) -> list[Site]:
    """Every site in the tree whose fingerprint matches the concept's.

    Deduped to at most one site per (file, qualname): the sliding-window
    detector otherwise reports four "sites" for four overlapping windows
    inside a single function, which are one occurrence, not four.
    """
    found: list[Site] = []
    seen: set[str] = set()

    for f in funcs:
        if concept.detector == "function_clone":
            if not f.body or si.fingerprint(f.body, keep_attr=False) != fp:
                continue
            site = Site(concept.id, f.file, f.qualname, f.lineno, f.lineno, f.end_lineno)
        else:
            site = None
            for idx, window in si.statement_windows(f, concept.window_k):
                if si.fingerprint(window, keep_attr=True) != fp:
                    continue
                start = window[0].lineno
                # Span may reach past the identity window — see span_extra_stmts.
                last = f.body[min(idx + concept.window_k - 1 + concept.span_extra_stmts,
                                  len(f.body) - 1)]
                site = Site(
                    concept.id, f.file, f.qualname, start, start,
                    last.end_lineno or last.lineno,
                )
                break
            if site is None:
                continue

        if site.key in seen:
            continue
        seen.add(site.key)
        found.append(site)

    return sorted(found, key=lambda s: (s.file, s.lineno))
