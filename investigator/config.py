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
    "check_rescue": "v1",       # Phase 2 — qwen3:14b, one quote vs one clause
    "contradict": "v1",         # Phase 2 — qwen3:30b, one pair of quotes
    "attack": "v1",             # Phase 2 — qwen3:30b, confounder prose
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

# Rescue is high-volume and narrow — one short quote against one short clause —
# so it defaults to the 14B, which is fully GPU-resident on a 12 GB card.
OLLAMA_MODEL_RESCUE = "qwen3:14b"

# Genuine comparative judgment. qwen3:30b is Qwen's MoE variant: ~19 GB at Q4
# does not fit a 12 GB card, but Ollama splits it (as many layers on GPU as
# fit, the rest on CPU RAM) and MoE sparsity keeps the CPU-side cost per token
# low. Slower per call than the all-GPU 14B path, spent only on the handful of
# pairs and findings these passes actually judge.
OLLAMA_MODEL_JUDGMENT = "qwen3:30b"

OLLAMA_TIMEOUT = 600  # split inference on a 30B is minutes, not seconds
OLLAMA_NUM_CTX = 32768
# Thinking plus format="json" can burn the whole budget on a hidden trace
# before emitting any JSON. Self-consistency compensates for what a
# single-shot non-thinking answer gives up.
OLLAMA_THINK = False
# The 12 GB card is shared with this project's own BGE-M3 embedder and
# reranker. A short keep_alive means the model unloads between bursts instead
# of permanently occupying the card; it costs a reload on the next call.
OLLAMA_KEEP_ALIVE = "2m"

SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_TEMPERATURE = 0.4
SELF_CONSISTENCY_AGREE_THRESHOLD = 2


# ========== budget ==========

# Local calls are free in USD but seconds each, so their cap is what makes a
# cycle terminate. The USD ceilings mirror rag/compliance.py's dry-run/real-run
# pair; rates themselves live only in ingestion/clients.py (ADR #68).
LOCAL_CALLS_PER_CYCLE = 400
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
