"""Shared paths, constants, and prompt-version tags for the local audit harness.

This harness is local dev tooling, not part of the shipped four-plane RAG
system (see CLAUDE.md) — it audits the repo, it does not read from or write
into rag/eval/app/monitoring's runtime paths. Its own outputs live entirely
under audit/ (gitignored, unchanged from the prior deletion_audit.py layout).
"""
from __future__ import annotations

from pathlib import Path

# ========== paths ==========

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "audit"
STATIC_DIR = AUDIT_DIR / "static"

CACHE_DB = AUDIT_DIR / "cache.sqlite3"
FINDINGS_JSONL = AUDIT_DIR / "findings.jsonl"
DIGEST_MD = AUDIT_DIR / "DIGEST.md"
WATCH_STATE = AUDIT_DIR / ".watch_state.json"

# ========== scan scope ==========

# Directories walked for .py file collection. Deliberately does NOT include
# scripts/local_audit itself (self-referential noise) or notebooks/data/docs
# (not source code the passes reason about).
SCAN_DIRS = ["ingestion", "rag", "eval", "app", "monitoring", "investigator", "tests", "scripts"]

PY_SKIP_DIRS = {
    "__pycache__", ".pytest_cache", ".git", ".venv", "node_modules",
    ".agents", ".claude", ".streamlit", ".vscode", "notebooks", "data",
    "audit", "local_audit",
}

# Plane membership for the cross-plane import check (conventions.py). Mirrors
# CLAUDE.md's four-plane architecture; ingestion/dossier is Plane Ib but
# shares the ingestion.* import namespace with Plane I.
PLANE_DIRS = {
    "ingestion": "ingestion",
    "rag": "rag",
    "eval": "eval",
    "app": "app",
    "monitoring": "monitoring",
    "investigator": "investigator",
}
# Only these planes may import ingestion.* directly, and only through the
# load_index() contract (CLAUDE.md: "load_index() is the single Plane I ->
# Plane II interface. Do not add hidden cross-plane paths.").
#
# investigator (Plane V, ADR #70) reads Plane Ib artifacts through
# ingestion.dossier.facts' models and prices calls through ingestion.clients.
# Neither goes through load_index(), so both surface here as findings — that is
# the check working as its docstring describes ("surfaces candidates for
# review, it doesn't assert a verdict"), and ADR #70 records them as reviewed.
# CROSS_PLANE_ALLOWED_SYMBOLS is deliberately NOT widened to silence them:
# that would weaken the check for every plane in order to legitimise one.
CROSS_PLANE_ALLOWED_IMPORTERS = {"rag", "eval", "app", "monitoring", "investigator"}
CROSS_PLANE_ALLOWED_SYMBOLS = {"load_index"}

# ========== local LLM (Ollama) ==========

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_TIMEOUT = 180
OLLAMA_NUM_CTX = 32768  # model supports 40960; leave headroom for output
# qwen3:14b has a "thinking" capability that (with format="json" forcing a
# structured final answer) can burn the entire request budget on a hidden
# reasoning trace before ever emitting JSON, timing out on tasks that aren't
# actually hard. Self-consistency (3x voting) already compensates for
# whatever judgment quality a single-shot non-thinking answer gives up.
OLLAMA_THINK = False
# This machine's GPU is 12GB total; qwen3:14b at Q4_K_M holds ~9.8GB of it
# while loaded, which starves this same project's own BGE-M3 embedder/
# reranker (and therefore `uv run pytest` and the Streamlit app) of VRAM if
# left loaded indefinitely. A short keep_alive means the model unloads
# between bursts of audit activity — watch.py's poll gaps, or simply not
# running the harness while doing other GPU-bound work on this box — instead
# of permanently occupying the card. Costs a reload (~5s) on the next call.
OLLAMA_KEEP_ALIVE = "2m"

# Second model for passes doing genuine comparative judgment rather than a
# lookup/extraction (duplication.py's pairwise divergence call, prompt_audit's
# open-ended review) — qwen3:14b's judgment ceiling was directly confirmed on
# duplication.py's known cost-calc-divergence case: wrong 3/3 self-consistency
# even with think=True and the relevant code in view. qwen3:30b is Qwen's MoE
# variant (~3B active params/token despite 30B total, per Ollama's own "small
# MoE model" description) — the right shape for this hardware (Ryzen 5500 +
# 64GB RAM + RTX 3060 12GB): too large (~19GB Q4) to fit fully in 12GB VRAM,
# but Ollama automatically splits it (as many layers as fit on GPU, rest on
# CPU RAM), and MoE sparsity keeps the CPU-side compute per token cheap
# despite the nominal size — unlike forcing a dense model that size onto CPU,
# which would multiply the full parameter count against every token. Slower
# per call than the all-GPU 14B path, spent only on the ~40 pairs / ~10 files
# these two passes actually judge, not on doc_drift's high-volume extraction.
OLLAMA_MODEL_JUDGMENT = "qwen3:30b"

# ========== prompt versions ==========
# Part of the cache key (cache.py). Bump the relevant tag whenever a pass's
# prompt text changes, so cached verdicts under the old prompt are treated
# as stale and reprocessed automatically — fixes deletion_audit.py's
# .processed_sections.txt, which had no such invalidation.

PROMPT_VERSIONS = {
    "doc_drift_extract": "v1",
    # v2: claims.py's verify path now filters non-path-like candidate_files
    # (branch names, technology names) out of both the deterministic check
    # and the LLM excerpt grounding, instead of reading them as "[X does not
    # exist]" — a real change to what evidence a cached verdict was computed
    # from, not just wording, so bumping the version is what forces the fix
    # to actually take effect instead of silently serving pre-fix verdicts.
    "doc_drift_verify": "v2",
    # v2: _judge_pair now calls with think=True — a real behavioral change,
    # not just wording, that a v1-keyed cache entry (written while
    # think=False was hardcoded) would otherwise silently mask.
    # v3: switched to OLLAMA_MODEL_JUDGMENT (qwen3:30b) — a different model
    # producing verdicts is unambiguously not the same cache entry as v2's.
    "duplication_verify": "v3",
    "prompt_audit": "v1",
    # Governs the divergence prompt *template* only. Each concept in
    # concepts.py carries its own rule_version, which rides in the cache
    # key alongside this — so editing one concept's rule invalidates that
    # concept's verdicts without reprocessing every other concept.
    "slimming_divergence": "v1",
}

# ========== slimming pass ==========

# Window-tier size gate, keyed by window length. Mass = AST node count, not
# lines: a run of four print(f"...") calls spans four lines but carries
# almost no structure, and a line-based gate lets exactly that through as a
# false cluster.
SLIMMING_MIN_WINDOW_MASS = {2: 25, 3: 32, 4: 40}
# Function-tier gate for *discovery* of unregistered clusters only. Registry
# concepts match by seed fingerprint regardless of size — ingestion/clients.py's
# deprecation shims are mass-15 and would be invisible under this floor.
SLIMMING_MIN_FUNCTION_MASS = 25
SLIMMING_MIN_CLONE_LINES = 10  # discovery only: shorter clones aren't worth a finding
SLIMMING_FUNCTION_LINE_BUDGET = 80
SLIMMING_DOCSTRING_RATIO = 0.22
SLIMMING_DOCSTRING_MIN_LINES = 80
# Cold-start spread for the LLM tier. Steady state is 100% cache hits and
# zero calls, so this only bites on the first sweep after a rule change.
SLIMMING_LLM_MAX_CALLS_PER_CYCLE = 24
VULTURE_MIN_CONFIDENCE = 60
# The one file allowed to define model price constants (ADR #68).
RATE_CATALOG_FILE = "ingestion/clients.py"

# ========== self-consistency ==========

SELF_CONSISTENCY_RUNS = 3
SELF_CONSISTENCY_TEMPERATURE = 0.4
SELF_CONSISTENCY_AGREE_THRESHOLD = 2  # of 3 must agree on the core verdict

# ========== duplication candidate generation ==========

DUPLICATION_TOP_K = 40  # candidate pairs per cycle, capped before any LLM call
DUPLICATION_KEYWORDS = [
    "cost", "retry", "recover", "repair", "truncat", "parse_json", "dry_run",
]

# ========== continuous operation (watch.py) ==========

WATCH_POLL_SECONDS = 300  # ~5 min
FULL_SWEEP_INTERVAL_SECONDS = 24 * 3600  # re-run duplication/prompt_audit at least daily
# Incremental cycles skip the two LLM passes that aren't cache-invalidated
# per-file (duplication re-embeds everything; prompt_audit has no cache at
# all) unless a changed file touches one of these — otherwise they wait for
# the next full sweep.
HEAVY_PASS_TRIGGERS = ("rag/generate.py", "rag/compliance.py", "ingestion/dossier/")
