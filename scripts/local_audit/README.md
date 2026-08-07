# Local audit harness

Local-only code-audit tooling driven by a local Ollama model (`qwen3:14b`),
designed to run for free, indefinitely, on hardware the project doesn't pay
per-token for. **Report-only**: every pass writes structured findings to
`audit/findings.jsonl` and a regenerated `audit/DIGEST.md`. Nothing in this
package writes to any source file outside `audit/` — the local model never
sees a prompt asking it to produce a diff, patch, or code snippet. Applying
a fix is always a separate, human-initiated step afterward.

Supersedes the old root-level `deletion_audit.py`, which asked the model
"can this be deleted?" using a non-import-aware regex reference finder — it
mostly just rediscovered pytest test functions as false positives, a problem
[vulture](https://github.com/jendrikseipp/vulture) already solves for free
and instantly. This harness is aimed instead at judgment calls a static
analyzer structurally can't make: does a doc's claim about the code still
hold, do two functions that look unrelated actually handle the same concept
and disagree, does a prompt's stated contract match the schema parsing it.

## Layers

- **Layer 0** (`static_tools.py`, `conventions.py`) — zero LLM calls. Runs
  ruff/radon/vulture/bandit/pip-audit fresh, plus CLAUDE.md's checkable
  conventions (cross-plane imports, `keep_default_na=False`, section
  dividers, ADR-numbering gaps). Its output grounds every LLM prompt below.
- **Pass 1 — doc/ADR/CLAUDE.md drift** (`passes/doc_drift.py`): extracts
  falsifiable claims from `CLAUDE.md`/`docs/decisions.md`, verifies each
  (existence claims resolve deterministically; everything else goes to the
  LLM with 3x self-consistency voting).
- **Pass 2 — cross-file semantic duplication** (`passes/duplication.py`):
  BGE-M3 embedding + keyword-bucket candidate generation (avoids O(n²)
  pairwise LLM comparison), then an LLM judgment call per candidate pair
  (`think=True`, `OLLAMA_MODEL_JUDGMENT` — this one needs real comparative
  reasoning, not a lookup). Candidate generation and grounding are validated
  against a known real case; the LLM judgment on subtle behavioral
  divergence (as opposed to structurally obvious duplication) is wrong on
  both qwen3:14b and qwen3:30b for that case, with near-identical reasoning
  from both — pointing at the system prompt's framing ("same overall
  responsibility") rather than model size as the actual weak link. See the
  calibration note in the module docstring for the concrete next thing to
  try (ask about shared sub-behaviors, not overall sameness).
- **Pass 4 — code slimming** (`passes/slimming.py`, `concepts.py`,
  `source_index.py`): finds redundancy, dead code, and size-budget breaches.
  Deterministic — the LLM tier exists but is currently switched off for every
  concept (see below). Membership is decided by normalized-AST fingerprints
  at two granularities: whole functions (identifiers *and* attribute names
  erased, so three parse helpers that differ only in log strings collapse to
  one cluster) and sliding k-statement windows (attribute names kept, since
  at window scale structure alone is too weak to identify a concept).
  Findings come from four sources: matches against the hand-authored
  **concept registry** in `concepts.py`, unregistered clone clusters at info
  severity (candidates for promotion into the registry), the Layer 0
  converters below, and per-file docstring mass.

  Registry concepts name a *seed site* rather than a hardcoded fingerprint;
  the fingerprint is recomputed from the seed every run. A hardcoded one
  would go stale the moment its canonical implementation was touched,
  silently emptying the concept — and since findings resolve when a pass
  stops reproducing them, every finding under it would read as fixed. When a
  seed stops resolving the pass says so at high severity instead. This
  already fired once for real: deleting `ingestion/clients.py`'s dead shims
  removed the `deprecated_client_shim` seed, and the alarm caught it.

  `ACCEPTED_CLONES` is the visible "we decided to keep this" ledger, and the
  reason is rendered into the digest. Without it the digest never converges
  and gets ignored — `app/streamlit_app.py`'s `get_flow`/`get_compliance` are
  genuine structural clones that must not be merged, because
  `@st.cache_resource` keys on the decorated function's identity.

  **Why the LLM tier is off.** It was built, wired, cached, and calibrated
  against `duplication.py`'s own motivating case — the cost-computation
  divergence — then switched off, because the honest result was that it adds
  nothing there. The single-span clause-checklist prompt *did* fix
  duplication.py's framing problem: the model stopped reasoning about the
  enclosing function and quoted precisely the right line. It then judged that
  line compliant, 3/3, and it was right to on the evidence it had — nothing
  in a code span reveals whether `_DIVERGENCE_EST_PROMPT_USD_PER_TOKEN` is
  the shared catalog or a module-local. That missing fact is a symbol
  lookup, and this harness's own Layer 0 rule says no LLM pass may re-derive
  what a deterministic tool answers. The clause became a regex over
  identifier names (`Concept.local_constant_pattern`) and finds all five
  sites, including the one the model missed. The generalisable lesson, and
  the check to apply before enabling `llm_divergence` on any concept: verify
  the span actually contains what decides the clause.

- **Pass 3 — LLM prompt-engineering audit** (`passes/prompt_audit.py`):
  reviews the ~10 files that build prompts for this project's LLM calls for
  schema drift and language-toggle inconsistency.

## Running it

```bash
# one-shot: everything, full scope
uv run python -m scripts.local_audit.orchestrator

# leave it running: polls git HEAD every 5 min, incremental cycles,
# full sweep at least once a day
uv run python -m scripts.local_audit.watch
```

Output: `audit/findings.jsonl` (id-keyed, upserted — status flips to
`resolved` when a pass no longer reproduces a finding rather than
accumulating forever) and `audit/DIGEST.md` (regenerated each cycle, open
findings grouped by severity then pass).

## Hardware note

Two models, chosen for this machine (Ryzen 5500, 64GB RAM, RTX 3060 12GB):

- `qwen3:14b` (Q4_K_M, ~9.8GB) — default model, fully GPU-resident. For a
  model that fits entirely in VRAM, GPU is unambiguously faster than CPU
  here: the 3060's ~360GB/s memory bandwidth vs. ~40-50GB/s for dual-channel
  DDR4 on this CPU is roughly a 7-10x gap, and LLM token generation is
  bandwidth-bound. Used for doc_drift's high-volume claim extraction, where
  call count matters more than judgment depth.
- `qwen3:30b` (`OLLAMA_MODEL_JUDGMENT`, MoE — ~3B active params/token
  despite 30B total, ~18GB) — used for duplication.py and prompt_audit.py's
  genuine-judgment calls. Too large for 12GB VRAM alone; Ollama
  automatically splits it (`ollama ps` shows ~50/50 CPU/GPU on this
  machine), and MoE sparsity keeps the CPU-side compute per token cheap
  despite the nominal size, unlike forcing an equivalently large dense model
  onto CPU. Meaningfully slower per call (~150-200s vs. single-digit
  seconds) — spent only on the ~40 pairs / ~10 files these two passes
  actually judge, not on every call in the harness.

Either model loaded means less VRAM for this project's own BGE-M3
embedder/reranker (`uv run pytest` and the Streamlit app both need the
GPU) — `qwen3:14b` alone already gets close to the 12GB ceiling.
`config.OLLAMA_KEEP_ALIVE` is set short (`"2m"`) so a model unloads between
bursts of audit activity instead of pinning the card indefinitely, but a
full sweep or an active `watch.py` poll cycle will hold whichever model is
in use for the duration of that run — avoid kicking off a full sweep right
before you need the GPU for something else.

## Cache invalidation

`cache.py`'s SQLite store keys every verification on
`(pass, prompt_version, file, code_hash)`. Editing a pass's prompt text?
Bump the matching entry in `config.PROMPT_VERSIONS` — that alone makes every
cached verdict under the old wording miss and reprocess, no manual cache
clear required.
