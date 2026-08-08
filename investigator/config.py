"""Paths, tunables, and prompt-version tags for Plane V.

Every path a cycle touches is derived from `CasePaths.for_case(case_id)`, so a
run is confined to one case directory and a test can redirect the whole plane
by passing `dossier_dir`. This is the one structural difference from
`scripts/local_audit/config.py`, whose paths are module globals pointing at a
single `audit/` directory.

Rationale: ADR #70.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ========== paths ==========

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOSSIER_DIR = PROJECT_ROOT / "data" / "dossier"
STATUTE_CHUNKS_CSV = PROJECT_ROOT / "data" / "chunks.csv"

CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"
GENERIC_CATALOG = CATALOG_DIR / "generic_fr_succession.yaml"

# Per-case overlay, read from inside the case directory. Ignored by default for
# every case but `demo`/`vitrine` (ADR #66), which is what keeps a real case's
# clause excerpts and doc_id globs out of git without a new rule.
CASE_CATALOG_FILENAME = "obligations.yaml"

# Discovery writes here and never to CASE_CATALOG_FILENAME. The rename is the
# trust boundary: a machine-written duty must pass a human before it can produce
# a finding that names a professional to an insurer.
CASE_CATALOG_PROPOSAL_FILENAME = "obligations.proposed.yaml"

INVESTIGATION_DIRNAME = "investigation"


@dataclass(frozen=True)
class CasePaths:
    """Every file one cycle may read or write, resolved for a single case.

    `outbound_dir` sits inside the case directory rather than at the repo root
    on purpose: it inherits ADR #66's ignore-by-default rule instead of needing
    its own, so a real case's gated output cannot be committed by forgetting a
    `.gitignore` line. The invariant it carries is unchanged — `render.py` is
    the only module that may write there, and it sources findings solely from
    `schema.externalisable_findings()`.
    """

    case_id: str
    case_dir: Path
    investigation_dir: Path
    findings_jsonl: Path
    cache_db: Path
    digest_md: Path
    watch_state: Path
    outbound_dir: Path
    case_catalog: Path
    case_catalog_proposal: Path

    @classmethod
    def for_case(cls, case_id: str, dossier_dir: Path | None = None) -> "CasePaths":
        if not case_id or "/" in case_id or case_id.startswith("."):
            raise ValueError(f"invalid case_id: {case_id!r}")
        base = (dossier_dir or DOSSIER_DIR) / case_id
        inv = base / INVESTIGATION_DIRNAME
        return cls(
            case_id=case_id,
            case_dir=base,
            investigation_dir=inv,
            findings_jsonl=inv / "findings.jsonl",
            cache_db=inv / "cache.sqlite3",
            digest_md=inv / "DIGEST.md",
            watch_state=inv / ".watch_state.json",
            outbound_dir=inv / "outbound",
            case_catalog=base / CASE_CATALOG_FILENAME,
            case_catalog_proposal=base / CASE_CATALOG_PROPOSAL_FILENAME,
        )


# ========== externalisation ==========

# Cases whose findings may be rendered to an outbound artifact at all. A real
# case is never on this list; enabling one is a deliberate, reviewed edit, not
# a CLI flag. See schema.externalisable_findings().
ALLOW_OUTBOUND_CASES = frozenset({"vitrine", "demo"})

# Cases already carrying anonymised identifiers, where a person_id in rendered
# output is a persona rather than a real party. Everything else is redacted to
# {actor_role}#{ordinal} on the outbound path.
CASE_IS_ANONYMISED = frozenset({"vitrine", "demo"})

OUTBOUND_MIN_TIER = "T2"


# ========== prompt versions ==========
# Part of the cache key (cache.py). Bump a tag whenever the prompt text or the
# evidence a verdict was computed from changes, so entries under the old
# wording miss naturally instead of needing a manual cache delete.
#
# Phase 1 calls no model, so only `search_offline` is live. The Phase 2 tags
# are declared here so the cache key shape is fixed before the layers land.

PROMPT_VERSIONS = {
    "search_offline": "v1",
    "search_piste": "v1",
    # Phase 2 — classifies one quote against one clause into
    # execution/stipulation/demande/manquement/autre. Used in BOTH
    # directions: it can withdraw a gap, and it can overturn a
    # satisfaction that rested on a mere mention.
    # v2: the prompt now requires the act to be the act the clause demands,
    # for the designated beneficiary. Under v1 both model sizes read a sale of
    # the assets into an ordinary account as performance of a restitution duty.
    # v3: OLLAMA_THINK is on and the call goes to both JUDGE_MODELS. A verdict
    # reached without reasoning is not the same computation, so v2 entries must
    # miss rather than be served — the cache key's model prefix already orphans
    # them, and the bump makes the reason explicit rather than incidental.
    "check_performance": "v3",
    "contradict": "v1",         # Phase 2 — qwen3:30b, one pair of quotes
    "attack": "v1",             # Phase 2 — qwen3:30b, confounder prose
    # Reads a clause-bearing document and proposes obligations. Cached per
    # (document, page window) so re-running a case is nearly free.
    # v2: think=False. Windows cached under v1 were produced with a reasoning
    # trace and must miss rather than be mixed with the rest.
    "discover": "v2",
}


# ========== pass tunables ==========

# Candidate pairs per cycle, capped before any adjudication. The cap truncates
# a stable sort; an unstable one would re-mint finding ids every cycle and
# destroy the resolve-on-fix signal entirely.
CONTRADICT_MAX_PAIRS = 40

# Facts handed to a future rescue call per gap, ranked deterministically.
CHECK_MAX_CANDIDATES = 8

# Quote/clause lengths for Phase 2 prompts. Declared now because they bound
# what `Evaluation.candidate_fact_ids` is worth carrying.
MAX_QUOTE_CHARS = 400
MAX_CLAUSE_CHARS = 200


# ========== local models (Ollama) ==========
# Phase 2. Nothing here is touched unless a run passes --local-llm; the
# deterministic verdict is computed first either way, and a model can only
# adjust it in the directions each pass documents.

OLLAMA_URL = "http://localhost:11434/api/chat"

# The classification tier's default single model, kept for callers that ask for
# one. The classifier itself now consults BOTH models — see JUDGE_MODELS.
OLLAMA_MODEL_RESCUE = "qwen3:14b"

# Every model that judges a classification.
#
# One entry, deliberately. The dual-judge form was borrowed from
# rag/compliance.py (Opus + Kimi) and eval/llm_eval.py (GPT + Claude + Mistral),
# but that doctrine rests on *different failure modes*, not on different
# parameter counts — and every model available locally is Qwen. Two sizes of one
# family share a tokenizer and a training corpus, so they agree for the same
# reasons and they are wrong for the same reasons: under the v1 prompt both the
# 14B and the 30B labelled a sale into an ordinary account `execution`, and it
# was prompt precision, not the second opinion, that fixed it. In the live run
# they agreed on 3 of 3 facts. A same-family second judge is confirmation bias
# with a compute bill — and it cost roughly 196 model loads, because only one
# model fits in 12 GB.
#
# The combiner still handles two or more judges and still reports divergence, so
# adding a genuinely independent one (a cloud model through the OpenRouter path
# budget.may_escalate already gates) is an entry in this tuple, not a rewrite.
JUDGE_MODELS = ("qwen3:30b",)

# Genuine comparative judgment. qwen3:30b is Qwen's MoE variant: ~19 GB at Q4
# does not fit a 12 GB card, but Ollama splits it (as many layers on GPU as
# fit, the rest on CPU RAM) and MoE sparsity keeps the CPU-side cost per token
# low. Slower per call than the all-GPU 14B path, spent only on the handful of
# pairs and findings these passes actually judge.
OLLAMA_MODEL_JUDGMENT = "qwen3:30b"

# Local inference costs nothing per call and has no deadline, so the timeout is
# sized for the slowest honest answer rather than to bound wall-clock.
OLLAMA_TIMEOUT = 900
# Sized for a reasoning trace, not for residency. Measured on this card:
# qwen3:14b totals 20.5 GB at 32768 and 10.3 GB at 4096, so the KV cache runs
# ~0.3 MB/token. 8192 lands near 11.8 GB against 12,288 MB — tight, and the
# right trade now that thinking is on: a trace truncated by a small window
# produces malformed JSON, which the client correctly reads as "no verdict",
# which silently costs a judgment. Do not lower this for throughput.
OLLAMA_NUM_CTX = 8192
# On, deliberately. local_audit sets this False because a hidden trace can burn
# a request budget before any JSON appears — but that hazard is a *timeout*,
# and the timeout above is generous. These are legal judgments on which a real
# claim rests; reasoning before answering is worth the tokens, and latency is
# not a cost that matters on local hardware.
OLLAMA_THINK = True
# The 12 GB card is shared with this project's own BGE-M3 embedder and
# reranker. A short keep_alive means the model unloads between bursts instead
# of permanently occupying the card; it costs a reload on the next call.
OLLAMA_KEEP_ALIVE = "2m"

SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_TEMPERATURE = 0.4
# Confirming a deterministic verdict and overturning one are not symmetric
# evidentiary acts, so they do not share a threshold. The deterministic result
# is the prior: a bare majority may uphold it, but reversing it requires every
# sample of every model to agree. Passed explicitly at the call site so the
# asymmetry is visible where the decision is made.
SELF_CONSISTENCY_AGREE_THRESHOLD = 2
SELF_CONSISTENCY_OVERTURN_THRESHOLD = SELF_CONSISTENCY_RUNS


# ========== budget ==========

# Local calls are free in USD but seconds each, so their cap is what makes a
# cycle terminate. The USD ceilings mirror rag/compliance.py's dry-run/real-run
# pair; rates themselves live only in ingestion/clients.py (ADR #68).
# The cap exists so a cycle terminates, not so it finishes quickly. Local calls
# are free; a tight cap silently drops judgments, and a truncated sweep is
# already safe (complete=False never reconciles) and resumable (watch.py's
# backlog mode). Bound a run deliberately with --local-calls when you mean to.
LOCAL_CALLS_PER_CYCLE = 5000
CLOUD_CALLS_PER_CYCLE = 20
PISTE_CALLS_PER_CYCLE = 20
USD_CYCLE_CEILING = 1.00
USD_SESSION_CEILING = 10.00


# ========== continuous operation (watch.py) ==========

WATCH_POLL_SECONDS = 300
FULL_SWEEP_INTERVAL_SECONDS = 24 * 3600

# Artifacts whose content hash triggers an incremental cycle. Plane V's subject
# is data, not code, so this replaces local_audit's `git rev-parse HEAD` poll.
WATCHED_CASE_ARTIFACTS = (
    "facts.jsonl",
    "actor_roles.jsonl",
    "role_ambiguities.jsonl",
    "persons.jsonl",
    "coverage.jsonl",
    CASE_CATALOG_FILENAME,
)
