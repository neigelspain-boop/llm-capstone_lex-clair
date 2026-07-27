# lex-clair — Claude Code context

French legal RAG helping non-lawyer heirs understand succession rights in quasi-usufruit conventions and notaire professional liability. DataTalks.Club LLM Zoomcamp 2026 capstone.

## Current state

- **Attempt 1 baseline shipped on `main`** (tag `attempt-1-baseline`). Day 7 complete: Postgres persistence + 6-panel Grafana dashboard. 23 tests passing. 17 rubric points locked. Do not modify `main` — it's the fallback submission.
- **Active work on `v2-agentic`**: pivoting from corpus-only Q&A RAG to dual-corpus (statute + user dossier) compliance-gap analysis. See `docs/decisions.md` for ADRs and `docs/response_doctrine.md` (v1 doctrine, being superseded — the v2 pivot invalidates §6.3–6.5).

## Four-plane architecture — plane membership equals tree position

- **Plane I — Ingestion** (offline): `ingestion/`, `data/`. Legal statute corpus via PISTE API (Légifrance), plus dossier extraction pipeline for v2.
- **Plane II — RAG flow** (online): `rag/`. `flow.run(query)` for Q&A mode. `flow.analyze(case_id)` for v2 compliance-matrix mode.
- **Plane III — Measurement**: `eval/`. Retrieval eval, LLM-as-judge harness with 3 provider-diverse judges.
- **Plane IV — UI/Operations**: `app/`, `monitoring/`. Streamlit UI, Postgres persistence, Grafana dashboards.

**Any file that doesn't fit cleanly into one plane is a smell.** Cross-plane imports only through defined contracts. `load_index()` is the single Plane I → Plane II interface. Do not add hidden cross-plane paths.

## Locked technical stack

- Python 3.12, `uv` for package management, pytest for testing
- Streamlit (UI), ChromaDB (vector), BM25 + vector hybrid retrieval with RRF, BGE-reranker via `sentence-transformers.CrossEncoder`
- OpenAI + Anthropic (via OpenRouter `openai/`-compatible client) + Mistral (`mistralai` v2 SDK) LLM providers
- Postgres 16 (persistence), Grafana 11 (dashboards), docker-compose for services
- Environment: Ubuntu, conventional commit messages

## Project conventions — non-negotiable

- **Spec first, code second**: docstring-level specification written and reviewed before implementation. No exceptions.
- **Section-commented code**: `# ==========` dividers above every major block, consistently. Existing files follow this — match it.
- **ADRs at the moment of decision**: any architectural choice → `docs/decisions.md` gets an ADR entry the same session, not retroactively.
- **`keep_default_na=False` on every `pd.read_csv`** call that reads lex-clair-produced CSVs. Pandas NaN trap is documented project-wide convention (ADR references). Silent failures > loud failures is the wrong direction — we want loud.
- **Kill switches even when dormant**: monitoring/db.py uses module-level `_DB_HEALTHY` global as a one-way kill switch (ADR #34).
- **Small commits, conventional messages**: `git diff --cached --stat` before every commit; split unrelated changes into separate commits. No monolithic "day X" commits.
- **Never rebase or force-push shared branches**. `main` and `v2-agentic` are both shared with `origin`.

## Testing

- Fast suite: `uv run pytest tests/ -v` (runs in <5s, must be green before any commit)
- Slow suite: `uv run pytest tests/ -m slow -v` (integration tests, run before session close)
- New tests go in `tests/test_*.py` matching existing conventions in `tests/test_ingestion_smoke.py`

## Key docs (read when relevant, not on every turn)

- `docs/decisions.md` — all ADRs. Read before any architectural change.
- `docs/response_doctrine.md` — v1 response doctrine. v2 will supersede parts of this.
- `README.md` — project entry point for reviewers.

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
- Do not commit files under `data/raw/`, `data/chroma/`, `data/conversations/`, or `data/feedback.csv` — all gitignored.
- Do not run destructive DB commands without explicit confirmation. Postgres data is real state.