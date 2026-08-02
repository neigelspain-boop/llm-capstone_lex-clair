# lex-clair — Claude Code context

French legal RAG helping non-lawyer heirs understand succession rights in quasi-usufruit conventions and notaire professional liability. DataTalks.Club LLM Zoomcamp 2026 capstone.

## Current state

- **Attempt 1 baseline shipped on `main`** (tag `attempt-1-baseline`). Day 7 complete: Postgres persistence + 6-panel Grafana dashboard. 23 tests passing. 17 rubric points locked. Do not modify `main` — it's the fallback submission.
- **Active work on `v2-agentic`**: dual-corpus (statute + user dossier) adversarial gap-analysis pivot. Days A, B, C shipped. 94 tests passing. See `docs/decisions.md` for ADRs #38-#46 and `docs/response_doctrine.md` (v1 doctrine, superseded — v2 invalidates §6.3-6.5 corpus-only assumptions and §6.4 single-model constraint).

## Four-plane architecture — plane membership equals tree position

- **Plane I — Statute ingestion** (offline): `ingestion/`, `data/chunks.csv`. Legal statute corpus via PISTE API (Légifrance). 792 chunks across 9 sources.
- **Plane Ib — Dossier ingestion** (offline): `ingestion/dossier/`, `data/dossier/`. PDF → verbatim extraction (Opus 4.7 vision) → Haiku 4.5 faithfulness gate → facts + actor_roles + role_ambiguities JSONL. Per-case audit CSV + shared corpus sync via `append_to_statute_chunks_csv` (ADR #39).
- **Plane II — RAG flow** (online): `rag/`. `flow.run(query, source_scope=None, active_case_id=None, answer_model="gpt-4o-mini")` for Q&A. `rag.compliance.generate_compliance_matrix(case_id)` for compliance-matrix mode.
- **Plane III — Measurement**: `eval/`. Retrieval eval, LLM-as-judge harness with 3 provider-diverse judges (GPT-4o-mini + Haiku 4.5 + Mistral Small — all via OpenRouter as of D0).
- **Plane IV — UI/Operations**: `app/`, `monitoring/`. Streamlit UI (multi-conversation + language toggle + answer-model toggle), Postgres persistence, Grafana dashboards.

**Any file that doesn't fit cleanly into one plane is a smell.** Cross-plane imports only through defined contracts. `load_index()` is the single Plane I → Plane II interface. Do not add hidden cross-plane paths.

## Person pipeline + distillation — Plane Ib, partially shipped

- Plane Ib's person-index pipeline was planned as three stages: **mentions** (extract person mentions per fact) → **resolve** (cluster mentions into canonical entities, case-local and global) → **distill** (ceremony-stripped substance summary per fact). Only **distill** has shipped (`ingestion/dossier/distill.py`, ADR #52). Mentions and resolve do not exist anywhere in this repo — no `mentions.py`/`resolve.py` on any branch — and remain open per ADR #50 (D1) and ADR #51 (D4).
- `Fact.distilled_context: str | None` is populated by `distill.py`; `verbatim_quote` is never mutated. The anticipated `Fact.mentioned_person_ids` field (mentions-stage output) does **not** exist yet — `rag/compliance.py` reads it defensively via `getattr(f, "mentioned_person_ids", None) or []`, so the person-context plumbing (ADR #53) is real code exercised as a no-op, not dead code behind a flag.
- Model choices: shipped stages use Haiku 4.5 (distill, temperature 0.0) and Opus 4.7 max (compliance reasoning over `distilled_context` + `verbatim_quote`, ADR #53). Planned (unbuilt) stages: Gemini flash-lite for mentions (high-volume, low-reasoning extraction), Haiku 4.5 for resolve (clustering/dedup judgment).
- See ADR #50-#53 in `docs/decisions.md` for the full history of what's built vs. open.

## Retrieval — source-scoped hybrid + router

- `HybridRetriever.search(query, source_scope="statute")` — RRF fusion of BM25 + BGE-M3 vector search, then BGE-reranker cross-encoder rescoring. `source_scope` values: `"statute"` (default), `"dossier"`, `"case:{id}"`, `"blended"`. Enforced at the Chroma filter level, not post-hoc. ADR #41 (B1). Default `"statute"` matches v1 behavior and closes the ADR #39 privacy blocker.
- `rag/retrieve.retrieve()` threads `source_scope` through to the underlying retriever.
- `rag/flow.run()` calls `rag.router.route_query()` when `source_scope=None` (default). Router uses Haiku 4.5 via OpenRouter to classify intent → scope. Route decision surfaced in return dict as `route_decision: RouteDecision`. ADR #42 (B2).

## Compliance gap analysis — Plane II analytical surface

- `rag/compliance.py::generate_compliance_matrix(case_id)` produces `data/dossier/{case_id}/compliance_matrix.json` — per-role obligation assessment against retrieved statute articles.
- Facts grouped by exact `actor_role` string; each cluster gets one LLM call to Opus 4.7 max via OpenRouter. ADR #43 (B3).
- **Cross-role context block** (ADR #44, C1): when facts across roles share a source document, the user message gains a "Contexte inter-rôles" section listing other roles and their labels. System prompt updated to weigh cross-role obligation interactions. No schema change, no re-extraction — pure prompt enrichment. Deterministic (sorted output), preserves matrix idempotency.
- Compliance matrix output is gitignored for private cases (`data/dossier/private/`). Demo fixture (`data/dossier/demo/`) is committed for reproducibility.
- **Known truncation risk** (candidate ADR #47, Attempt 2): `max_tokens=4096` is shared between reasoning tokens and output JSON on Opus 4.7 max. Fact-heavy roles (10+ facts, prompt_tokens >5000) risk `finish_reason=length` with `raw_chars=0`. Loud warning logs fire; parser has bracket-tracking incremental recovery. Observed loss rate on private case Day C run: ~52% of calls truncated. Fix path: raise budget to 8192 or split fact-heavy clusters. Do not tune inside Attempt 1 — cost curves shift.
- **Verify-conclude + person integration** (ADR #53, D6): the prompt requires reasoning from `distilled_context` but verifying against `verbatim_quote` before finalizing `breached`/`met`, downgrading to `insufficient_evidence` on any mismatch. A "Personnes impliquées" section and `ComplianceEntry.persons_named` are wired but inert — `persons_named` is `[]` on every entry until the D1-D4 person pipeline ships (see Person pipeline + distillation, above). Cluster-level results are cached at `data/dossier/{case_id}/compliance_cache.jsonl`; `--dry-run` reports per-cluster token/cost estimates and raises if the total exceeds `DRY_RUN_COST_ALERT_USD` ($25).

## Multi-model answer generation — runtime catalog (ADR #45, C2)

- `rag/generate.py::ANSWER_MODELS` catalog:
  - `gpt-4o-mini` (default): `openai/gpt-4o-mini`, no reasoning, max_tokens=500, $0.15/$0.60 per M
  - `opus-4.7`: `anthropic/claude-opus-4.7`, reasoning=max, max_tokens=4096, $15/$75 per M
  - `kimi-k3`: `moonshotai/kimi-k3`, reasoning=max, max_tokens=4096, $3/$15 per M
- `flow.run(query, answer_model="gpt-4o-mini")` accepts any catalog key; return dict includes `answer_model_key` alongside `model_used`. `ValueError` on unknown key.
- **Cost calculation lives in `generate.py` via the catalog.** `flow.py::_compute_cost` was deleted in C2 — it would have silently zeroed cost for every non-default model.
- Streamlit sidebar `Modèle` segmented control (⚡ Rapide / 🧠 Opus 4.7 / 🔬 Kimi K3) with per-model cost hint. Session state key: `answer_model`. ADR #46 (C3).

## OpenRouter SDK transport patterns

- **All LLM calls route through OpenRouter** (unified credit pool, single client factory `ingestion.clients.get_openrouter_client`). Model provider diversity preserved by routing GPT + Anthropic + Mistral + Moonshot through the same OpenAI-compatible surface. ADR #40 (D0).
- **Reasoning effort**: `client.chat.completions.create(..., extra_body={"reasoning": {"effort": "max"}})`. Top-level `reasoning=` kwarg is rejected by the OpenAI SDK — use `extra_body`.
- **Kimi K3 slug**: `moonshotai/kimi-k3` — NOT `moonshot/kimi-k3`. Missing `ai` returns 404 "model not found" with unhelpful error text.
- **Reasoning budget vs output budget**: both count against `max_tokens`. Reasoning-heavy prompts with long expected output need generous budgets or must cap output structure.
- **Mistral Small on OpenRouter**: use `mistralai/mistral-small-2603`. The alias `mistral-small-latest` has no OpenRouter mapping — 402/404 fail.

## Model palette (locked as of Day C)

| Purpose | Model ID | Reasoning | Cost (USD/M in/out) |
|---------|----------|-----------|---------------------|
| Cheap generation (default) | `openai/gpt-4o-mini` | no | 0.15 / 0.60 |
| Query router | `anthropic/claude-haiku-4.5` | no | ~1 / ~5 |
| Compliance reasoning | `anthropic/claude-opus-4.7` | max | ~15 / ~75 |
| Deep-think answer (opt-in) | `anthropic/claude-opus-4.7` | max | ~15 / ~75 |
| Alternative deep-think (opt-in) | `moonshotai/kimi-k3` | max | 3 / 15 |
| Dossier extraction | Opus 4.7 vision + Haiku 4.5 gate | — | mixed |
| Eval judges | GPT-4o-mini + Haiku 4.5 + Mistral Small (`mistralai/mistral-small-2603`) | no | mixed |

## Locked technical stack

- Python 3.12, `uv` for package management, pytest for testing
- Streamlit (UI), ChromaDB (vector), BM25 + vector hybrid retrieval with RRF, BGE-reranker via `sentence-transformers.CrossEncoder`
- Postgres 16 (persistence), Grafana 11 (dashboards), docker-compose for services
- Environment: Ubuntu, conventional commit messages

## Project conventions — non-negotiable

- **Spec first, code second**: docstring-level specification written and reviewed before implementation. No exceptions.
- **Section-commented code**: `# ==========` dividers above every major block, consistently. Existing files follow this — match it.
- **ADRs at the moment of decision**: any architectural choice → `docs/decisions.md` gets an ADR entry the same session, not retroactively. When an ADR supersedes an earlier one, the Context section MUST cite the superseded ADR by number.
- **`keep_default_na=False` on every `pd.read_csv`** call that reads lex-clair-produced CSVs. Pandas NaN trap is documented project-wide convention.
- **Kill switches even when dormant**: `monitoring/db.py` uses module-level `_DB_HEALTHY` global as a one-way kill switch (ADR #34).
- **Small commits, conventional messages**: `git diff --cached --stat` before every commit; split unrelated changes into separate commits.
- **Privacy grep before every commit**: `git diff --cached --name-only | grep -E "private"` must print nothing before pushing. Private case data lives locally in `data/dossier/private/` (gitignored). After push, `git ls-tree -r origin/v2-agentic --name-only | grep -c "dossier/private"` must return 0.
- **Never rebase or force-push shared branches**. `main` and `v2-agentic` are both shared with `origin`.

## Testing

- Fast suite: `uv run pytest tests/ -v` (runs in <5s, must be green before any commit). 94 tests passing as of Day C close.
- Slow suite: `uv run pytest tests/ -m slow -v` (integration tests, run before session close)
- New tests go in `tests/test_*.py` matching existing conventions in `tests/test_ingestion_smoke.py`.

## Key docs

- `docs/decisions.md` — ADRs #1-#46 as of Day C close. Read before any architectural change.
- `docs/response_doctrine.md` — v1 response doctrine. §6.3-6.5 corpus-only assumptions and §6.4 single-model constraint superseded by Day C ADRs.
- `README.md` — project entry point for reviewers.
- `docs/plan_day_a.md`, `docs/plan_day_b.md`, `docs/plan_day_c.md` — per-day execution plans (kept for provenance).

## Communication style Claude Code should use

- Lead with the plan or the answer. No preamble.
- Evidence-tiered reasoning. State confidence explicitly (high / best-effort).
- Flag confounders before conclusions.
- One logical change per turn. If the request implies multiple changes, propose them as separate sequential steps in Plan mode.
- If uncertain about a file's plane membership or an existing convention, read the neighbouring files or ask before writing.

## What NOT to do

- Do not modify files on `main`. All work goes to `v2-agentic`.
- Do not add cross-plane imports outside the `load_index()` contract.
- Do not touch `docs/response_doctrine.md` — supersession happens via a new ADR + a v2 doctrine file, not in-place edits.
- Do not commit files under `data/raw/`, `data/chroma/`, `data/conversations/`, `data/dossier/private/`, or `data/feedback.csv` — all gitignored.
- Do not run destructive DB commands without explicit confirmation. Postgres data is real state.
- Do not index dossier chunks into the shared corpus without source-scope filtering active (ADR #39 privacy blocker).
- Do not tune `compliance.py` `max_tokens` inside Attempt 1 window — deferred to Attempt 2 (candidate ADR #47).