# lex-clair — Claude Code context

French legal RAG helping non-lawyer heirs understand succession rights in quasi-usufruit conventions and notaire professional liability. DataTalks.Club LLM Zoomcamp 2026 capstone.

## Current state

- **Attempt 1 baseline shipped on `main`** (tag `attempt-1-baseline`). Day 7 complete: Postgres persistence + 6-panel Grafana dashboard. 23 tests passing. 17 rubric points locked. Do not modify `main` — it's the fallback submission.
- **Active work on `v2-persons`**: dual-corpus (statute + user dossier) adversarial gap-analysis pivot, now dossier-first (ADR #66). Days A-D shipped, plus the person pipeline and the anonymisation subsystem. 251 tests collected. See `docs/decisions.md` for ADRs #38-#70.
- **In flight — Plane V investigator** (ADR #70, spec in `docs/investigator-spec.md`). Phase 1 to 11 Aug: all five passes ship, **none calls an LLM** — a full cycle is deterministic and costs $0.00. Phase 2, 12-14 Aug: local qwen3 adjudication layers, each additive over a pass that already works. Working checklist at `investigator/PLAN.md` (temporary; deleted when Phase 2 lands).

## Five-plane architecture — plane membership equals tree position

- **Plane I — Statute ingestion** (offline): `ingestion/`, `data/chunks.csv`. Legal statute corpus via PISTE API (Légifrance). 792 chunks across 9 sources.
- **Plane Ib — Dossier ingestion** (offline): `ingestion/dossier/`, `data/dossier/`. PDF → verbatim extraction (Opus 4.7 vision) → Haiku 4.5 faithfulness gate → facts + actor_roles + role_ambiguities JSONL → distill → mentions → resolve. Per-case chunks CSV; dossier chunks are merged into the corpus **at load time** (ADR #58) — `append_to_statute_chunks_csv` was retired and no longer exists. Also holds the anonymisation subsystem: `anonymize.py` (four-layer PII redaction + verification gate), `personas.py` (pinned court-style persona roster → the public `vitrine` case) and `roles.py` (role-designation register → the public `demo` investigation transcript, ADR #78).
- **Plane II — RAG flow** (online): `rag/`. `flow.run(query, source_scope=None, active_case_id=None, answer_model="gpt-4o-mini")` for Q&A. `rag.compliance.generate_compliance_matrix(case_id)` for compliance-matrix mode.
- **Plane III — Measurement**: `eval/`. Retrieval eval, LLM-as-judge harness with 3 provider-diverse judges (GPT-4o-mini + Haiku 4.5 + Mistral Small — all via OpenRouter as of D0).
- **Plane IV — UI/Operations**: `app/`, `monitoring/`. Streamlit UI (multi-conversation + language toggle + answer-model toggle), Postgres persistence, Grafana dashboards.
- **Plane V — Investigation** (offline, long-running): `investigator/`, `data/dossier/{case_id}/investigation/`. Obligation-catalog-driven gap analysis over Plane Ib artifacts. Where Plane II answers a query, Plane V *generates* the questions — comparing what should be in the documents (per statute, contract, deontology) against what is, with **absence** as the primary signal. Five passes hard-wired by name: `graph → search → check → contradict → attack`. Report-only: writes nothing outside the case dir except through the single gate in `schema.externalisable_findings()`. ADR #70.

**Any file that doesn't fit cleanly into one plane is a smell.** Cross-plane imports only through defined contracts. `load_index()` is the single Plane I → Plane II interface. Do not add hidden cross-plane paths.

## Person pipeline + distillation — Plane Ib, shipped with one gap

- All three stages exist and are committed: **distill** (`distill.py`, ADR #52) writes a ceremony-stripped substance summary per fact; **mentions** (`mentions.py`, ADR #54) extracts person mentions per document to `_mentions/<doc_id>.json`; **resolve** (`resolve.py`, ADR #55) clusters into canonical entities and writes `persons.jsonl`. ADR #55 supersedes #54's assumption that resolve would consume `_mentions/*.json` — it reads `distilled_context` per role instead.
- ADR #50 and #51 were **reserved but never written**. D1's scope was absorbed into D2 (ADR #54); D4 (global cross-case entity store) is still deferred.
- **The one real gap**: `Fact.mentioned_person_ids` still does not exist on the `Fact` model (`ingestion/dossier/facts.py`). `rag/compliance.py` reads it defensively via `getattr(f, "mentioned_person_ids", None) or []`, so `_persons_context_for_role` filters on a field that is always absent and `ComplianceEntry.persons_named` is `[]` on every entry — even though `persons.jsonl` is now fully populated. Neither `mentions.py` nor `resolve.py` writes back onto facts. Backfilling that field is the open follow-up (`docs/decisions.md`, ADR #53 follow-up (d)).
- Model choices: distill and mentions both use Haiku 4.5 (temperature 0.0); resolve uses Haiku 4.5 for clustering; compliance reasoning uses Opus 4.7 max over `distilled_context` + `verbatim_quote` (ADR #53). No Gemini model is used anywhere.
- See ADR #52-#55 in `docs/decisions.md` for the full history.

## Retrieval — source-scoped hybrid + router

- `HybridRetriever.search(query, source_scope="statute")` — **weighted** RRF fusion of BM25 + BGE-M3 vector search (`DEFAULT_BM25_BOOST`), then BGE-reranker cross-encoder rescoring. ADR #64 supersedes ADR #20: the pipeline ran `mode="vector"` for most of its life, and uniform RRF let the weak BM25 arm override the strong vector arm, so the weighting is load-bearing, not cosmetic.
- `source_scope` values: `"statute"`, `"dossier"`, `"case:{id}"`, `"case+statute:{id}"` (the gap-analysis default since ADR #66), `"blended"`. Enforced at the Chroma filter level, not post-hoc, and **fail-closed as an allowlist** since ADR #63 — a denylist left 824 orphan vectors reachable.
- `rag/retrieve.retrieve()` threads `source_scope` through to the underlying retriever.
- `rag/flow.run()` calls `rag.router.route_query()` when `source_scope=None` (default). Router uses Haiku 4.5 via OpenRouter to classify intent → scope. Route decision surfaced in return dict as `route_decision: RouteDecision`. ADR #42 (B2).

## Compliance gap analysis — Plane II analytical surface

- `rag/compliance.py::generate_compliance_matrix(case_id)` produces `data/dossier/{case_id}/compliance_matrix.json` — per-role obligation assessment against retrieved statute articles.
- Facts grouped by exact `actor_role` string; each cluster gets one LLM call to Opus 4.7 max via OpenRouter. ADR #43 (B3).
- **Cross-role context block** (ADR #44, C1): when facts across roles share a source document, the user message gains a "Contexte inter-rôles" section listing other roles and their labels. System prompt updated to weigh cross-role obligation interactions. No schema change, no re-extraction — pure prompt enrichment. Deterministic (sorted output), preserves matrix idempotency.
- Every case directory is gitignored by default; only `demo/` and `vitrine/` are whitelisted by name (ADR #66). `vitrine` is the anonymised public case the README's commands use; `demo` is the smoke fixture and its JSONL files are intentionally empty.
- **Truncation — fixed, not open.** ADR #49 uncapped `MAX_OUTPUT_TOKENS` on the compliance call. The old `max_tokens=4096` was shared between reasoning and output on Opus 4.7 max, so fact-heavy `notaire_*` clusters hit `finish_reason=length` (~52% loss on the Day C private run) and ADR #48 added a per-case `compliance_run.log` to make that silence visible. Candidate ADR #47 was never written — #49 superseded it. The bracket-tracking incremental recovery in the parser remains as a second line of defence.
- **Verify-conclude + person integration** (ADR #53, D6): the prompt requires reasoning from `distilled_context` but verifying against `verbatim_quote` before finalizing `breached`/`met`, downgrading to `insufficient_evidence` on any mismatch. A "Personnes impliquées" section and `ComplianceEntry.persons_named` are wired but still inert — not because the person pipeline is unshipped (it is), but because `Fact.mentioned_person_ids` was never added (see Person pipeline + distillation, above). Cluster-level results are cached at `data/dossier/{case_id}/compliance_cache.jsonl`; `--dry-run` reports per-cluster token/cost estimates and raises if the total exceeds `DRY_RUN_COST_ALERT_USD` ($25).

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

- Python >=3.11 (`pyproject.toml`), `uv` for package management, pytest for testing
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
- **Privacy grep before every commit**: every case directory under `data/dossier/` is gitignored by default and only `demo/`/`vitrine/` are whitelisted (ADR #66), so the check is that no unexpected case appears: `git diff --cached --name-only | grep "data/dossier/" | grep -vE "data/dossier/(demo|vitrine)/"` must print nothing. After push, verify the same against `origin/v2-persons`.
- **Never rebase or force-push shared branches**. `main` and `v2-persons` are both shared with `origin`.

## Testing

- Fast suite: `uv run pytest tests/ -v` (must be green before any commit). 251 tests collected, 6 marked `slow`. There is no `conftest.py` yet — fixtures are file-local.
- Slow suite: `uv run pytest tests/ -m slow -v` (integration tests, run before session close)
- New tests go in `tests/test_*.py` matching existing conventions in `tests/test_ingestion_smoke.py`.

## Key docs

- `docs/decisions.md` — ADRs #1-#78. Read before any architectural change. **Known bookkeeping gaps**: #47, #50, #51, #59, #60, #61, #62 have no entry. #59-#62 are the load-bearing ones — the anonymisation subsystem shipped under them and later ADRs cite them as if they exist. #65 is also filed out of order, before #64.
- `docs/investigator-spec.md` — Plane V contracts: obligation catalog schema, the five pass contracts, finding/tier schema, the externalisation gate, resumability invariants. Rationale is ADR #70 (per ADR #69: spec carries the contract, ADR carries the why).
- `docs/lex-clair-system-map.html` — rendered architecture map.
- `README.md` — project entry point for reviewers.
- `scripts/local_audit/README.md` — local-only, report-only code-audit harness driven by a local Ollama model. Findings in `audit/DIGEST.md` (gitignored, regenerated per run).

## Communication style Claude Code should use

- Lead with the plan or the answer. No preamble.
- Evidence-tiered reasoning. State confidence explicitly (high / best-effort).
- Flag confounders before conclusions.
- One logical change per turn. If the request implies multiple changes, propose them as separate sequential steps in Plan mode.
- If uncertain about a file's plane membership or an existing convention, read the neighbouring files or ask before writing.

## What NOT to do

- Do not modify files on `main`. All work goes to `v2-persons`.
- Do not add cross-plane imports outside the `load_index()` contract. `ingestion/clients.py` is the one sanctioned shared-infra exception.
- Do not commit files under `data/raw/`, `data/chroma/`, `data/conversations/`, or `data/feedback.csv`, or any case directory under `data/dossier/` other than `demo/` and `vitrine/` — all gitignored.
- Do not run destructive DB commands without explicit confirmation. Postgres data is real state.
- Do not weaken the source-scope allowlist in `ingestion/load.py` — it is fail-closed by design (ADR #63) and is what keeps dossier chunks out of statute-scoped answers.
- Do not re-cap `compliance.py` `max_tokens`. ADR #49 uncapped it deliberately; reasoning and output share the budget on Opus 4.7 max.