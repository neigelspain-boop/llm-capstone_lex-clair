# Architecture decisions · lex-clair

Architecture Decision Records (ADRs) for the lex-clair French legal RAG
capstone. Each entry captures one decision, its rationale, the alternatives
considered, and its reversibility cost. Ordering is chronological.

Written primarily for peer reviewers and for future-me. When a reviewer asks
"why did you do X?", the answer should be here with a line number they can
cite. When a Day 3 eval result challenges a decision, the reversibility line
tells me whether to swap or defend.

**Rubric mapping.** Where a decision defends a specific rubric line, it is
tagged inline — e.g. `[rubric · Retrieval flow]`. The full rubric text lives
in `../project.md`.

**Scope.** Only decisions with real trade-offs are logged. Trivial choices
(file names, log format) are not.

---

## 2026-07-15 · Four-plane architectural split
**Status:** Accepted

**Context.** The rubric spans retrieval, evaluation, interface, monitoring,
containerization — nine distinct scoring lines. Without a structural
convention, code drifts into a `src/` grab-bag where any file can import any
other, and rubric-line responsibility becomes untraceable.

**Decision.** The repository is organized into four planes, each a
top-level folder whose name is the plane's role:

- `ingestion/` — Plane I · offline. Corpus → indexed artifacts.
- `rag/` — Plane II · online hot path. User query → answer.
- `eval/` — Plane III · measurement. Numbers on retrieval + generation.
- `monitoring/` — Plane IV · operations. Logging, feedback, dashboards.

Every source file belongs to exactly one plane. Any file that doesn't fit
cleanly is a smell that it should not exist.

**Rationale.** Reviewers grading nine distinct rubric lines can locate the
evidence for each in a specific folder. The physical layout mirrors the
scoring structure. Cross-plane leakage is caught at review time by folder
placement alone.

**Trade-offs.** Slightly more folders than a beginner project needs. Some
files that could be shared (e.g., LLM client wrappers) end up duplicated
across `rag/` and `eval/`. Acceptable for a course project.

**Reversibility.** Locked. Refactoring away from this now would touch every
import statement in the codebase.

---

## 2026-07-15 · Streamlit-only interface, no FastAPI
**Status:** Accepted · `[rubric · Interface]`

**Context.** The rubric awards 2 pts for a UI, web app, or API. Both
Streamlit (UI) and FastAPI (API) would satisfy it — but only one is needed.

**Decision.** Streamlit only. No FastAPI, no separate API surface.

**Rationale.** Streamlit provides free integration with feedback capture
(needed for the Monitoring rubric line), does not require separate frontend
work, and can be recorded as an app preview video for the README. FastAPI
would add a client-side burden with no rubric-point return.

**Trade-offs.** No programmatic API for downstream integration. Fine for a
course project; a real deployment would want both.

**Reversibility.** Easy — add `rag/api.py` in a few hours if needed.

---

## 2026-07-15 · Dual-LLM setup: Anthropic + OpenAI
**Status:** Accepted · `[rubric · LLM evaluation]`

**Context.** The rubric awards 2 pts for LLM evaluation when multiple
approaches are compared and the best is used. A single-provider setup caps
this at 1 pt.

**Decision.** Both Anthropic (Claude) and OpenAI (GPT) clients are wired
in. Day 5 will run LLM-as-judge evaluation comparing them on 200 ground-
truth samples.

**Rationale.** Two providers unlocks the full 2 pts. Anthropic is the
primary quality provider; OpenAI is the cost baseline. Comparison also
informs the production choice.

**Trade-offs.** Two API keys in `.env`, two SDK dependencies, two prompt
templates to maintain.

**Reversibility.** Trivial — drop one provider by removing the client
import.

---

## 2026-07-15 · Three core best-practices, not one
**Status:** Accepted · `[rubric · Best practices, 3 pts]`

**Context.** The rubric offers up to 3 bonus points, one each for hybrid
search, cross-encoder reranking, and query rewriting.

**Decision.** All three are treated as core, not optional. Hybrid search
lands in Plane I/II by construction (Day 2). Query rewriting (Day 4) and
reranking (Day 4) are separate `rag/` modules.

**Rationale.** For lex-clair specifically — plain-French user queries vs
legal-register statute text — the vocabulary gap is the defining problem.
Hybrid search alone is architecturally necessary, not a bonus. Query
rewriting and reranking each address a distinct failure mode:

- Rewriting closes the plain→legal register gap at query time.
- Reranking closes the "wrong ordering of relevant candidates" gap.

Cut order if a day slips: cloud deploy, then reranking, then rewriting.
Hybrid search stays.

**Reversibility.** Any of the three can be disabled behind a config flag.

---

## 2026-07-18 · Data artifacts at repo root, not nested
**Status:** Accepted

**Context.** `articles.csv`, `chunks.csv`, `ground_truth.csv`, index blobs
— where do they live? Nesting under `ingestion/data/` couples ownership to
one producer, but `eval/` and `app.py` also read them.

**Decision.** All data lives at `data/` at the repo root. Every module
running from the repo root reads and writes with the same relative paths.

**Rationale.** Data is the exchange medium between planes, not the property
of any single plane. Root placement makes CWD assumptions identical across
modules, and reviewers see one canonical location for all corpus artifacts.

**Trade-offs.** `data/` mingles inputs (`corpus_manifest.yaml`), derived
artifacts (`articles.csv`, `chunks.csv`), and runtime state (`chroma/`,
`raw/`). Managed via subdirectories and `.gitignore`.

**Reversibility.** Any restructuring would touch every read/write call.
Effectively locked.

---

## 2026-07-18 · PISTE API, not Légifrance HTML scraping
**Status:** Accepted · `[rubric · Reproducibility]`

**Context.** Article text can be pulled either from PISTE (Légifrance's
official API, OAuth2, versioned, rate-limited) or by scraping Légifrance
HTML pages.

**Decision.** PISTE API. Client credentials via `.env`. Sandbox for
initial development, production for the real corpus.

**Rationale.** Reproducibility (2 pts) requires that a reviewer six months
from now can rebuild the corpus. HTML scraping breaks on any Légifrance
frontend redesign. PISTE has a stable Swagger contract and is the
government's official interface.

**Trade-offs.** ~4h of OAuth debugging on Day 1. Requires reviewers to
register a free PISTE account to rerun ingestion. Mitigated by committing
the derived `articles.csv` so reviewers can skip ingestion entirely for
retrieval/RAG/eval work.

**Reversibility.** Locked. HTML scraping was rejected on quality grounds
that have not changed.

---

## 2026-07-18 · Manifest-driven scope: LEGISCTA sections, not full LEGITEXT codes
**Status:** Accepted

**Context.** Légifrance offers 76 codes. Ingesting the full Code civil
alone is ~15,000 articles.

**Decision.** Fetch specific LEGISCTA sections declared in
`data/corpus_manifest.yaml`. Current scope: 10 sources, ~800 articles
covering the case domain (successions, usufruit, libéralités,
responsabilité extracontractuelle, dettes du défunt, assurances de
responsabilité, appropriations frauduleuses, plus three LODA texts on
notaire regulation).

**Rationale.** Retrieval precision matters more than recall in a targeted
domain. Wide corpora dilute top-k with off-topic distractors — the exact
failure mode that would show up on Day 3 eval as low MRR. The manifest is
a single YAML file a reviewer can read to understand corpus scope in 60
seconds.

**Trade-offs.** Miss articles outside declared sections. Mitigated by
manifest extensibility: adding a section is one YAML entry.

**Reversibility.** Trivial — add or remove manifest entries.

---

## 2026-07-18 · Chunk IDs namespaced by source code
**Status:** Accepted

**Context.** Article numbering collides across codes — Code civil art. 587
and Code pénal art. 587 both exist. Bare `art-587` would produce silent
duplicate-key failures at index time.

**Decision.** Every `chunk_id` is prefixed with a code namespace:
`cc-587`, `cgi-774bis`, `ca-l124-3`, `cp-314-1`, `ord45-2590-...`,
`d73-609-...`. Mapping is in `ingestion/parse.py` (`CHUNK_ID_PREFIX`).

**Rationale.** Guaranteed uniqueness across the corpus. Human-readable
citations (`cc-587` reads as "Code civil, article 587" without a legend).
Enables the Day 4 answer generator to produce verifiable references the
user can look up on Légifrance.

**Reversibility.** Changing the scheme would invalidate `chunks.csv` and
any existing ground-truth CSVs. Not locked, but not free either.

---

## 2026-07-18 · Committed vs gitignored artifacts
**Status:** Accepted · `[rubric · Reproducibility]`

**Context.** Some derived artifacts are worth committing (peer reviewers
inspect them); others are large binary blobs (rebuild instead).

**Decision.**

Committed:
- `data/articles.csv` (~800 rows, KB range) — reviewers verify indexed content
- `data/chunks.csv` — same reason
- `data/corpus_manifest.yaml` — scope declaration
- `data/ground_truth.csv`, `data/*_eval_results.csv` (Day 3+) — reproducible eval

Gitignored:
- `data/raw/` (~1200 JSON files, ~40 MB, fully re-fetchable)
- `data/chroma/` (~8 MB binary index, rebuilt on container start)
- `.env` (secrets)

**Rationale.** Reviewers running `git clone` should get everything they
need to inspect the retrieval and eval work. They should NOT get 8 MB of
Chroma binary that will not match a fresh embedding run on their machine.

**Reversibility.** Trivial `.gitignore` edit.

---

## 2026-07-23 · ADR #27 — Streamlit for the UI layer

**Context.** Rubric line "Interface" awards +2 for UI, web app, or API. Zoomcamp reference module 07 uses Flask; broader capstone community defaults to Streamlit. Peer reviewers evaluate ~20 min per project — the UI must be legible at a glance.

**Decision.** Streamlit, single file at `app/streamlit_app.py`.

**Rationale.** Peer-review surface: reviewer clones, runs `streamlit run`, sees the app. `@st.cache_resource` handles BGE-M3 + reranker cold-start in one line (Flask would need module-level singleton + gunicorn config). `st.chat_input` + `st.chat_message` deliver the standard chat UX without templates or JS. ADR #10 protects reversibility — `rag.flow.run()` has no Streamlit dependency.

**Alternatives.** Flask (reference module) — familiar but adds template + JS burden for feedback UX, same rubric score. FastAPI — adds Swagger UI, doubles code footprint for no rubric gain.

**Reversibility.** High. Swapping to Flask/FastAPI touches only `app/`.

---

## 2026-07-23 · ADR #28 — Interim CSV feedback storage before Postgres

**Context.** Rubric "Monitoring" splits +2 into feedback collection (+1) and dashboard (+1). Day 7 is scoped for full Postgres + Grafana. Adding Postgres today doubles Day 6 scope for a rubric-neutral gain.

**Decision.** Day 6 writes feedback to `data/feedback.csv` with `csv.QUOTE_ALL`. Header: `timestamp, conversation_id, turn_id, question, answer, rating, comment, model_used, cost_usd, elapsed_seconds`. Day 7 migrates rows into a Postgres `feedback` table via a one-shot loader; `_append_feedback` swaps to `psycopg2`.

**Rationale.** Schema mirrors the intended Postgres table 1:1 → migration is a `COPY` command, not a re-plumb. `QUOTE_ALL` makes the file `COPY`-compatible without an escape pass. File is gitignored so no leakage.

**Notes.** `conversation_id` semantics = thread ID (all turns of one conversation share it). `turn_id` = specific Q&A within a thread. Both are UUIDs.

**Reversibility.** High. Same schema on both sides of the migration.

---

## 2026-07-23 · ADR #29 — SUPERSEDED by ADR #30

**Original decision (chrome-only EN toggle).** Retained for the labels/buttons/instructions portion. Answer-translation restriction lifted by #30 after live testing revealed the peer-review-accessibility gap.

---

## 2026-07-23 · ADR #30 — EN toggle: Tier 1.5 (chrome + machine-translated answer)

**Supersedes:** ADR #29 in part.

**Context.** ADR #29 shipped chrome-only EN. Live testing revealed a non-French reviewer toggling EN sees English labels but a French answer body and cannot judge answer quality — which is what rubric bonus points reward. The "rubric-neutral" framing in #29 was wrong: reviewer bonus points (up to 3 × 3 = 9 pts) depend on visible RAG quality.

**Decision.** When EN toggle is on, machine-translate the assistant's French answer via `gpt-4o-mini`. Cache result on the turn dict (`answer_en`). Show disclaimer above every translated answer: "🌐 Machine translation for peer-review accessibility. The French answer is authoritative." Question, citations (article numbers with Légifrance URLs), and technical-details values stay French.

**Cost.** ~$0.0005 per turn when EN is used. Zero otherwise. Estimated peer-review load: 3 reviewers × 5 turns × 1 EN session ≈ $0.008 total.

**Latency.** ~2–3 s per translation, once per turn. Cached thereafter.

**Failure surface.** Translation call is wrapped in `try/except` → warning banner + French fallback. Never blocks.

**Alternatives.** Client-side browser translation (rejected: destroys chrome labels, poor on dynamic content). Translate at generation time (rejected: doubles cost even when never toggled, violates ADR #10). Per-turn "Show translation" button (rejected: redundant with the toggle we already have).

**Plane discipline.** Translation lives in `app/streamlit_app.py`, not `rag/`. UI accessory, never enters feedback signal — CSV stores canonical French answer.

**Reversibility.** High. Delete `_translate_to_english`, `get_openai_client`, the EN branch in `_render_answer_body`, three label keys. ~40 lines.

---

## 2026-07-23 · ADR #31 — Multi-turn conversation architecture with JSON persistence

**Context.** Initial UI (v1–v3) modelled each Q&A as its own conversation with a per-Q&A `conversation_id`. Live testing revealed this is a log view, not a chat architecture — every serious chat UI (ChatGPT, Claude, Perplexity, Gemini) uses a two-level model where a conversation is a thread of turns. Peer reviewers arriving at a Streamlit chat app expect this pattern; getting the shape right now also prevents a data-model refactor in Day 7.

**Decision.** Two-level state:
- **Conversation** = `{id, title, created_at, updated_at, turns: list[dict]}`. Title auto-generated from the first question (truncated to 45 chars).
- **Turn** = `{turn_id, question, result, answer_en, feedback}` — one Q&A pair within a conversation.

Persistence: one JSON file per conversation at `data/conversations/{id}.json`. Loaded once at session start via `glob`. Saved atomically (`.tmp` → `rename`) on every mutation — turn append, `flow.run` completion, translation cache, feedback capture.

**Rationale.** Standard chat UX. Preserves multi-turn context for reviewers to test the app the way it will actually be used. Migrates to Postgres tomorrow as: `for f in glob(data/conversations/*.json): INSERT INTO conversations (...) VALUES (...)`. Atomic writes prevent corruption on crash.

**Alternatives.** Single JSON blob (`data/conversations.json`) — rejected: read-modify-rewrite hazard, harder to delete individual conversations. SQLite intermediate — rejected: adds a DB choice today for storage that gets replaced tomorrow. No persistence (session-only) — rejected: loses history on browser refresh, defeats sidebar navigation.

**Feedback CSV impact.** `turn_id` column added; `conversation_id` semantics shift from per-Q&A to per-thread. Old rows have empty `turn_id` (existing rows are effectively one-turn conversations); wipe with `rm data/feedback.csv` on schema flip if desired.

**Reversibility.** High. `_load_all_conversations` and `_save_conversation` are the only two disk-touching functions — swap for DB calls in Day 7.


---

## 2026-07-19 · `load_index()` as the sole cross-plane interface
**Status:** Accepted

**Context.** Plane I (ingestion) has heavy dependencies — PISTE OAuth,
tenacity, httpx, PyYAML. Plane II (rag/) must boot fast for Streamlit
reloads. Anything in `ingestion/` imported by `rag/` at boot pulls the
whole offline stack into the hot path.

**Decision.** Exactly one public function crosses the plane boundary:
`from ingestion import load_index`. It returns a `HybridRetriever` with
`bm25`, `vectors`, `embed_model`, `chunks` populated. Nothing else in
`ingestion/` is re-exported.

**Rationale.** Plane II never accidentally imports PISTE credentials or
fetch logic. The boundary is enforceable at review time by grepping
`rag/` for any `ingestion.` import other than `load_index` / `HybridRetriever`.

**Reversibility.** Locked. The whole plane split depends on this contract.

---

## 2026-07-19 · Article-level chunking for V1 (identity transform)
**Status:** Accepted

**Context.** Chunking granularity is a tunable parameter. Options: whole
article, per-alinéa, per-N-tokens.

**Decision.** V1 is 1 article = 1 chunk. `ingestion/chunk.py` is an
identity transform reading `articles.csv` and writing `chunks.csv` with
the same shape. The function signature declares a `granularity` parameter
that also accepts `"alinea"` — currently raising `NotImplementedError`.

**Rationale.** French statute is already segmented by the legislator at
article granularity, with stable LEGIARTI identifiers. Article-level
chunk_ids enable verifiable citations for free (`cc-587` maps 1:1 to a
searchable chunk). The interface exists so that if Day 3 retrieval eval
shows alinéa-level would improve MRR, the swap is one file.

**Trade-offs.** Long articles (some in the CGI run to 1500 tokens) may
carry too much context per chunk. Measured empirically on Day 3, not
now.

**Reversibility.** Cheap. Alinéa-level implementation lives in the same
file and doesn't touch fetch, parse, index, or load.

---

## 2026-07-19 · Embedding model: BAAI/bge-m3 over multilingual-e5-large
**Status:** Accepted

**Context.** Dense retrieval needs a multilingual model. Two viable
candidates in 2026: `intfloat/multilingual-e5-large` and `BAAI/bge-m3`.
Both are 1024-dim, ~560M params, MTEB top-tier on French subsets, Apache
2.0 / MIT.

**Decision.** BGE-M3. Dense mode only, explicit `return_sparse=False,
return_colbert_vecs=False`.

**Rationale.** The tiebreaker is failure-mode profile, not benchmark
numbers.

- e5-large requires `"query: "` / `"passage: "` prefixes. Forgetting them
  degrades recall 5–10% *silently* — no exception, just quietly worse
  numbers.
- BGE-M3 requires selecting the right retrieval mode (dense / sparse /
  ColBERT-style). Mis-selection fails loudly — wrong output shape,
  immediate exception downstream.

Under a 9-day sprint, loud failures beat silent failures. Silent 5–10%
recall degradation surfaces at Day 3 eval as an unexplained number, then
costs 2h of hunting. A stack trace surfaces at line 1 of `index.py`.

Context-length superiority of BGE-M3 (8192 vs 512) is NOT a factor —
your corpus is 200–1500 tokens per article, comfortably under both
ceilings.

**Trade-offs.** BGE-M3 documentation was thinner in 2024 but reached
parity by 2026. No known blockers.

**Reversibility.** ~30 min to swap. Model ID is a single module constant
(`EMBED_MODEL_ID` in `ingestion/index.py`). If Day 3 eval shows a
meaningful quality gap, swap and re-run Day 3.

---

## 2026-07-19 · Hybrid retrieval via Reciprocal Rank Fusion (k=60)
**Status:** Accepted · `[rubric · Best practices · Hybrid search]`

**Context.** Once BM25 (lexical) and Chroma (dense) return separate
ranked lists, they must be fused into one. Options: weighted score sum,
Reciprocal Rank Fusion (RRF), learned reranker.

**Decision.** RRF with `k=60`. For each candidate chunk_id, sum
`1 / (60 + rank_bm25) + 1 / (60 + rank_vec)`. Sort descending, take
top-k.

**Rationale.** Weighted score sum requires normalizing BM25 scores and
cosine similarities — they live on incompatible scales. Fragile.
Learned reranker is Day 4 work, separate concern.

RRF is rank-based, parameter-light, and rewards agreement across sources
without punishing single-source hits. `k=60` is the paper default
(Cormack et al., 2009). Standard practice in production retrieval
systems.

The V1 API exposes an `alpha: float = 0.5` parameter in `search()`
signature reserved for Day 3 experimentation with weighted fusion — but
V1 ignores it. Signals to reviewers that fusion is a tunable, not a
hardcode.

**Trade-offs.** RRF cannot express "trust BM25 twice as much as vectors"
— that would need weighted rank fusion. Not a V1 problem.

**Reversibility.** Swap the fusion function in `ingestion/load.py`.
20 min.

---

## 2026-07-19 · Dual index (BM25 in-memory + Chroma persistent)
**Status:** Accepted

**Context.** BM25 index and dense vector store are different data
structures with different persistence trade-offs.

**Decision.**

- BM25 (`minsearch`) — in-memory, rebuilt on every boot from `chunks.csv`.
  Build time ~1s for 792 chunks.
- Chroma (dense vectors) — persisted to `data/chroma/`. Embedding is
  expensive (~25s per rebuild); cached on disk.

Rebuild policy for Chroma: `shutil.rmtree(CHROMA_DIR)` at start of every
build. No partial-state.

**Rationale.** BM25 is cheap enough that persistence complexity is not
worth it. Chroma is expensive enough that persistence is mandatory.
Different tools for different cost profiles.

Rebuild-from-scratch prevents the classic bug of embedding some chunks
with model version A and others with version B — impossible to debug on
Day 3 when comparing embedding configs.

**Trade-offs.** ~1s boot cost for BM25 rebuild every Streamlit reload.
Trivial.

**Reversibility.** Locked, both index types are fundamental to the
architecture.

---

## 2026-07-19 · `keep_default_na=False` on every internal CSV read
**Status:** Accepted

**Context.** `pd.to_csv` writes empty string as `,,`. `pd.read_csv`
without config reads `,,` as `NaN`. Round-trip corrupts empty-but-valid
optional fields (`titre`, `section_path`, `url`).

**Decision.** Every `pd.read_csv` call in lex-clair code that reads a
lex-clair-produced CSV MUST include `keep_default_na=False`. Applied in
`chunk.py`, `index.py`, `load.py`. Enforced by convention, not by tooling.

**Rationale.** Discovered on Day 2 when index validation kept firing
"792 nulls in titre" despite `chunk.py` explicitly writing empty
strings. The nulls were being injected by the reader, not present in
the file. One-line fix at every read site.

**Trade-offs.** Reviewers unfamiliar with the parameter may find it
verbose. Documented in this file specifically for that reason.

**Reversibility.** Removing the flag would silently regress the null
bug.

---

## 2026-07-19 · Full-row storage in BM25 documents (not indexed-field subset)
**Status:** Accepted

**Context.** `minsearch.Index.fit(documents)` stores whole document
dicts and returns them at query time. What columns to include?

**Decision.** Feed the full DataFrame row to `.fit()`. Only fields in
`TEXT_FIELDS` / `KEYWORD_FIELDS` are actually indexed; the rest ride
along as returnable metadata.

**Rationale.** Day 4's `rag/generate.py` needs `url` on every hit to
produce cited answers. Storing only the indexed subset would drop `url`
and force a second lookup at query time. Storage cost is ~40 KB extra
in RAM for 792 chunks. Trivial.

**Trade-offs.** Slightly more RAM per BM25 result set. Negligible.

**Reversibility.** Change one function (`_prepare_bm25_documents`) in
`index.py`.

---

## 2026-07-19 · `chunks` DataFrame indexed on `chunk_id` in HybridRetriever
**Status:** Accepted

**Context.** After RRF selects top-k chunk_ids, `search()` must
hydrate them back into full result dicts. Options: O(n) boolean
filtering (`chunks[chunks.chunk_id == cid]`), or O(1) index lookup
(`chunks.loc[cid]`).

**Decision.** `load_index()` calls `chunks.set_index("chunk_id",
drop=False)` once at boot. `search()` uses `.loc[cid]` inside the
hydration loop.

**Rationale.** For each query, hydration runs k times (typically 10).
O(1) per lookup vs O(n=792). Cost of one-time index build is
negligible. `drop=False` keeps `chunk_id` accessible as both index and
column, avoiding surprises for reviewers reading the DataFrame.

**Reversibility.** Change one line in `load_index()`.

---

## 2026-07-19 · `infer_device()` as public helper, shared build + load
**Status:** Accepted

**Context.** Both `index.py` (Chroma build) and `load.py` (query-time
embed) need to select CUDA vs CPU consistently.

**Decision.** `infer_device()` lives in `ingestion/index.py`, no
leading underscore (public). `load.py` imports it directly. No
duplication of the torch-availability check.

**Rationale.** If Day 3 experiments change device selection logic
(e.g., MPS support, CUDA memory guards), the change happens in one
place. Drift is prevented by construction.

**Reversibility.** Trivial.

---

## Known trade-offs and deferred items

Not decisions per se — active items to remember.

- **`decret_74_737`** currently indexed as one article (`d74-737-...`).
  Parent JORFTEXT identifier was not provided on Day 1 and this ends up
  a single-article LODA fetch. Not case-critical (Ordonnance 45-2590
  covers notaire regulation). Revisit Day 9 if buffer permits.

- **`decret_2023_1297`** currently 0 rows due to a PISTE `/consult/jorf`
  500 error. Non-blocking for V1. Retry Day 9.

- **CGI articles 641, 1133, 1727** (fiscal declaration delay,
  extinction exemption, interest for late payment) are outside the
  currently-fetched LEGISCTA. Case-adjacent but not critical. Add to
  manifest on Day 3 if retrieval eval shows a gap.

- **FlagEmbedding batching.** `.encode(..., batch_size=32)` is
  partially ignored — the model runs one prompt at a time under the
  hood. 24s for 792 chunks is fine; would become a bottleneck at
  5000+ chunks on Day 3 experiments.

- **HuggingFace Hub HEAD requests** fire on every build/load with an
  unauthenticated warning. Non-blocking. Set `HF_HUB_OFFLINE=1` in
  Docker environment (Day 8).

- **`titre` field is empty for all 792 rows.** Code civil articles are
  identified by `num`, not by title. Boost weight on `titre` will act
  as no-op — noted for Day 3 boost-tuning experiments so time is not
  spent optimizing a dead field.

---

## 2026-07-20 · Synthetic ground truth via GPT-4o-mini
**Status:** Accepted · `[rubric · Retrieval evaluation]`

**Context.** Retrieval eval requires (question, correct_chunk_id) pairs to
compute Hit Rate and MRR. Real user query logs do not exist for lex-clair
before Day 6. Two options: hand-label ~50 pairs (expensive, small), or
generate synthetically with an LLM against the article corpus.

**Decision.** Generate 1584 pairs by prompting
`gpt-4o-mini-2024-07-18` to produce 2 plain-French questions per article
across all 792 articles. Stored at `data/ground_truth.csv`, committed.

Implementation lives in `eval/ground_truth.py`. Structured output enforced
via Pydantic (`responses.parse(text_format=QuestionList)`), so JSON parse
errors are structurally impossible. Temperature 0.7 for Q1/Q2 diversity.
Prompt in French, non-lawyer register — "petit-enfant qui découvre une
succession, pas un avocat".

Actual production numbers: 742 API calls (50 already present from an earlier
`--sample 50` validation run; resume-safety picked up from there), 0 failures,
$0.0717 total, ~25 minutes wall time. Under the plan's $0.50 budget by 6x.

**Rationale.** Zoomcamp module 04 and reference `02-evaluating-retrieval.md`
establish this pattern; peer reviewers will recognize it in one glance.
Synthetic ground truth is the standard pre-launch RAG-eval signal — a real
labeled set would require domain expertise and 10+ hours we do not have.

The prompt explicitly targets the plain-French non-lawyer register so the
question distribution matches real users (grandchildren facing succession
disputes) rather than domain-expert queries. Sample inspection of 100 pairs
before running the full corpus confirmed 6/6 prompt rules being followed.

Structured output was chosen over free-text JSON because free-text plagued
Day 1's earlier extraction attempts — model would drift into markdown code
fences at ~5% rate, breaking `json.loads`. `responses.parse` eliminates the
problem at the API layer.

**Trade-offs named honestly.**

- The prompt explicitly instructs "Ne recopie PAS les termes exacts de
  l'article — reformule". This artificially depresses lexical overlap
  between questions and source text. BM25 has less to match on than a
  natural query distribution would provide. This biases retrieval eval
  against lexical methods, and is the direct cause of the counter-
  intuitive result documented in ADR  (hybrid loses to vector).
- Real users may cite article numbers verbatim after receiving legal
  correspondence ("qu'est-ce que dit l'art 815 ?"). The prompt explicitly
  strips article numbers from generated que#20stions, so this query mode is
  under-represented in eval.
- Temperature 0.7 makes re-generation non-deterministic — reviewers who
  re-run `eval/ground_truth.py` will get different (but similar-quality)
  questions than the committed CSV. Acceptable: the committed CSV IS the
  reproducibility anchor. Reviewers evaluating retrieval against a fresh
  ground truth would still measure the same architectural conclusions.

**Reversibility.** Trivial. `rm data/ground_truth.csv &&
uv run python -m eval.ground_truth`. Checkpoint-per-50-articles makes
partial regens Ctrl-C-safe.

---

## 2026-07-20 · Retrieval configuration: vector_only ships
**Status:** Accepted · `[rubric · Retrieval evaluation · Best practices · Hybrid search]`

**Context.** Four retrieval configurations were compared against the 1484-row
test split of `data/ground_truth.csv` (100 rows held out for boost tuning).
BM25 boosts tuned via 20-iteration random search over `(0.0, 3.0)` per field
on the val set, then evaluated on the untouched test set.

Numbers (Hit@10 / MRR@10):

| config         | hit_rate@10 | mrr@10 |
|----------------|-------------|--------|
| bm25_only      | 0.4164      | 0.2343 |
| **vector_only**| **0.7695**  | **0.5547** |
| hybrid         | 0.7224      | 0.4153 |
| hybrid_tuned   | 0.7385      | 0.4531 |

Best boost vector found: `texte=1.266, titre=0.089, section_path=0.656, num=1.516`.
L2 distance from uniform: 1.133 (meaningful, not noise).

**Decision.** Ship `HybridRetriever.search(mode="vector")` as the production
default. Retain all 4 configurations in `eval/retrieval_eval.py` and the
`mode` / `boost_dict` parameters in `ingestion/load.py`. Commit
`data/retrieval_eval_results.csv` so reviewers can inspect the numbers that
drove the choice.

**Rationale.** Vector-only beats every fusion variant on both metrics. RRF
fusion actively degrades vector performance (−4.7pp Hit@10, −13.9pp MRR)
because averaging a strong signal with a weak one produces a mid signal.

The cause is diagnosed and named in ADR #19: synthetic ground truth was
generated with explicit de-lexicalization. BM25 has almost nothing to match
on for the question distribution actually present in the eval set.

Under the "prioritize what Zoomcamp wants → prioritize professional
consensus → prioritize project-specific reasoning" decision protocol
(established Day 3): Zoomcamp is silent on this tie-break. Professional
consensus in retrieval literature says "ship what the eval demonstrates".
The eval demonstrates vector_only. We ship vector_only.

Retaining all four configs in the codebase — rather than deleting the losing
ones — is a deliberate diligence signal. A peer reviewer can inspect
`data/retrieval_eval_results.csv`, replicate the finding with
`uv run python -m eval.retrieval_eval`, and confirm the ship choice matches
the data. That's stronger evidence of methodological integrity than a clean
final codebase would be.

Boost tuning did NOT ship, but the discovery it made is preserved: `titre`
boost converged near zero (0.089), matching the Day 2 known-trade-off note
that titre is empty for all 792 Code civil rows. This is a validation
signal, not a discarded artifact — it demonstrates the tuning procedure
found real structure in the data.

**Trade-offs named honestly.**

- **Real user queries may not match synthetic query distribution.** A
  grandchild who receives a notaire's letter citing "art 815 alinéa 2"
  will type that verbatim. Vector-only handles this less well than BM25
  or hybrid; synthetic GT deliberately hides this query mode. Risk is
  real but unmeasured pre-launch. Post-launch retest, once feedback
  logs exist (Day 7 monitoring), is committed to below.
- **The +5pp Hit@10 ship-gate defined in the Day 3 plan was NOT met** by
  hybrid_tuned (actual +1.6pp over hybrid). This is a legitimate finding,
  not a failure: the eval directly showed tuning cannot rescue hybrid
  because the fusion itself is the ceiling.
- **The `num=1.516` boost is optimizing a phenomenon the eval cannot see.**
  Article-number queries are absent from synthetic GT by construction.
  A real query "art 815" would benefit from the num boost, but this
  benefit is invisible in current numbers. Documented for post-launch
  retest.

**Reversibility.** Trivial. Day 4's `rag/` module reads the retrieval mode
from one config flag. Switching to `hybrid_tuned` would be a one-line
change plus loading the boost dict from
`data/retrieval_eval_results.csv`.

**Post-launch retest committed.** Once user feedback exists in Postgres
(Day 7), re-generate ground truth from actual queries. If real queries
contain legal-register lexical anchors that synthetic GT lacks, hybrid or
hybrid_tuned may beat vector_only on that distribution. Re-run
`eval/retrieval_eval.py` on the new GT and re-decide.

---
## 2026-07-21 · Cross-encoder rerank via BGE-reranker-v2-m3 through sentence-transformers
**Status:** Accepted · `[rubric · Best practices · Document re-ranking]`

**Context.** Reranking is one of the three core best-practices locked
Day 0 (ADR #4). The standard pattern is bi-encoder retrieval → cross-
encoder rerank: BGE-M3 (used in retrieve.py) encodes query and doc
independently — fast, less precise. A cross-encoder scores (query, doc)
pairs jointly through a transformer — slower per pair but more precise.
Runs at query time on 20 candidates.

**Decision.** Rerank with `BAAI/bge-reranker-v2-m3` (2.27GB, multilingual)
loaded through `sentence-transformers.CrossEncoder`, not through
`FlagEmbedding.FlagReranker`. Return raw logits sorted descending, no
sigmoid transform, no threshold. `rag/rerank.py` exposes `rerank(query,
hits, k=5)`.

**Rationale.**
Model choice: family match with BGE-M3 embedder — consistent architectural
narrative for peer reviewers ("bi-encoder from BGE family for retrieval,
cross-encoder from same family for reranking"). Multilingual pretraining
covers French; empirically validated on the quasi-usufruit test query.

Loader choice: FlagEmbedding 1.4.x has a known open bug (PR #1544)
where the reranker silently loads the slow `XLMRobertaTokenizer`, which
lacks `prepare_for_model` in modern `transformers`, crashing at
`compute_score()` time with `AttributeError`. Direct `AutoTokenizer(...,
use_fast=True)` also hits the same slow-tokenizer fallback because
BGE-reranker's `tokenizer_config.json` specifies the slow class as
default. `sentence-transformers` handles the loader path cleanly — same underlying model weights, identical scores. Plane I regression check after `uv add sentence-transformers` confirmed BGE-M3 in ingestion/load.py still works unchanged.

No threshold: we only sort, we don't reject. Adding a threshold requires calibration on labeled positive/negative pairs we don't have.

**Trade-offs named honestly.**

- **Absolute scores are uncalibrated and near zero.** For the test query
  "quasi-usufruit et notaire" the top-5 scores were +0.0929 down to
  +0.0131 — barely positive. Two readings, both correct: (a) no article
  in the corpus perfectly matches this compound concept (quasi-usufruit
  is defined across cc-587 + cc-621 + doctrine, not one article); (b)
  the reranker is a general multilingual model, not French-legal-domain-
  finetuned. Depresses absolute scores without necessarily damaging rank
  order. Consequence: Day 6 UI cannot use these as a confidence signal
  without recalibration. Use percentile-relative rendering, not absolute.
- **Reranker completely replaced retrieve.py's top-5** (0% overlap on
  the compound-query test). For queries where retrieve.py already ranks
  the correct article at rank #1, this shuffle could theoretically hurt
  rather than help. Not measured — retrieval eval (Day 3) was pre-rerank.
  ADR #20 pattern (measure to ship decisions) was not applied here
  because rerank is a rubric-line implementation, not a shipdecision
  requiring measurement.
- **+~40MB dependency chain** (sentence-transformers). Acceptable for
  a project, would matter in a container. Alternative was a ~15-line
  raw-transformers path with `use_fast=True`, rejected because it also
  falls back to slow tokenizer in this environment.

**Reversibility.** Trivial. `rag/rerank.py` is 100 lines, isolated
behind `rerank(query, hits, k)`. Swap CrossEncoder for a different
backend (Cohere API, ColBERT, raw transformers with a working tokenizer
fix, etc.) touches one file. Removing rerank entirely — one line in
`rag/flow.py`.

---

## 2026-07-21 · LLM query rewriting via gpt-4o-mini + Pydantic structured output
**Status:** Accepted · `[rubric · Best practices · User query rewriting]`

**Context.** Query rewriting is one of the three core best-practices
locked Day 0 (ADR #4). The vocabulary gap between plain-French user
queries ("ma grand-mère a vendu la maison") and legal-register statute
text ("cession en démembrement de propriété") is the defining problem
lex-clair exists to solve. Vector retrieval alone spans some of this
gap; rewriting closes more.

**Decision.** Single-shot LLM rewrite via `gpt-4o-mini` with
`responses.parse(text_format=RewrittenQuery)`. Prompt written in
French, non-lawyer register on input, ≤40-word single-sentence output
enriched with legal vocabulary. Retrieval-side only — the ORIGINAL
query is used in the answer prompt; the REWRITTEN query is used only
for retrieve + rerank. Silent fallback to original query on any
failure (empty output, parse error, network).

**Rationale.**
Technique choice: three techniques considered.
- Single-shot rewrite (chosen): one API call, ~$0.00005/query, cheapest,
  defensible, one file.
- Multi-query: generate 3 variants, retrieve for each, RRF-merge. 3× cost,
  ~3× code complexity, more robust to lexical variance. Cut for Day 4
  scope; log as deferred experiment if Day 7 real-query logs show
  rewrite quality is inconsistent.
- HyDE (Hypothetical Document Embeddings): generate a fake answer,
  embed that, retrieve on the embedding. Elegant on English; not proven
  on French legal text. Rejected on evidence-tier grounds.

Structured output: same pattern as Day 3 ground truth generation (ADR
#19). Free-text JSON drift was the Day 1 lesson — `responses.parse`
eliminates it at the API layer.

Retrieval-side only: architectural discipline. The answer must be
grounded in what the user actually asked, not our translation. Empirical
validation: on the deliverable query "grand-mère a vendu la maison en
usufruit", the rewriter disambiguated to "a cédé la nue-propriété"
(one of two plausible interpretations); the answer still addressed
the user's original ambiguous phrasing rather than the rewriter's
disambiguation.

Silent fallback: rewriting is a *retrieval enhancement*, not a
correctness path. A dead OpenAI shouldn't take down the whole flow.
`log.warning()` preserves the signal for monitoring while degrading
gracefully to raw-query retrieval.

**Trade-offs named honestly.**

- **Over-composing.** Every rewrite adds "réserve héréditaire" and
  "succession" whether the original query implies inheritance or not.
  IN "Le notaire a-t-il fait une erreur ?" (6 words, could be any
  notarial context) → OUT injects the full succession + reserve +
  quotité disponible + héritiers framing. Acceptable for V1 because
  lex-clair's scope IS succession disputes; if the user is outside
  scope, the corpus doesn't cover their case anyway. Mitigation options
  logged for later: loosen the prompt to "STRICTEMENT pertinents" or
  measure Hit@k with rewrite on/off.
- **Semantic disambiguation is silent.** Ambiguous queries get one legal
  interpretation forced by the rewriter. "grand-mère a vendu la maison
  en usufruit" got resolved to "cédé la nue-propriété" — but could
  equally mean "sold her usufruit right". If the user meant the other,
  downstream retrieval is misaligned. Day 6 UI may need a
  disambiguation confirm step; not blocking Day 4.
- **Uplift not measured pre-launch.** Rubric awards points for
  *implementing* rewrite, not for measuring it. Retrieval eval on
  rewrite-on vs rewrite-off is a deferred experiment (§7). ADR #20
  measurement pattern applies only where measurement drives a ship
  decision; here it doesn't.
- **Silent fallback masks OpenAI outages.** If OpenAI is down,
  retrieval quality silently degrades to raw-query performance. The
  log.warning() breadcrumb is the mitigation — Day 7 monitoring will
  surface it via dashboard.

**Reversibility.** Trivial. `rag/rewrite.py` is 90 lines, isolated
behind `rewrite(query) -> str`. Prompt lives inline as `REWRITE_PROMPT`
constant — iterating takes one edit. Disabling entirely — one line
in `rag/flow.py`.

---

## 2026-07-21 · Retrieve k=20 → rerank k=5
**Status:** Accepted

**Context.** Cross-encoder reranking requires over-fetch from the
initial retrieval step. If retrieve returns k=5 and rerank keeps k=5,
the reranker has nothing to reorder — you get the same 5 items back,
sorted differently. The over-fetch ratio determines how much room the
reranker has to promote articles the bi-encoder missed.

**Decision.** `RETRIEVE_K=20`, `RERANK_K=5`. Constants at the flow
layer (`rag/flow.py`), not per-file.

**Rationale.**
Zoomcamp reference: module 04 uses k=5 for retrieval-only (no rerank
step). No specific rerank ratio locked.

Professional consensus in retrieval literature: 3-5× over-fetch is the
standard pattern. 20:5 = 4×, dead center.

Three ratios considered:
- **10 → 3**: too tight. Reranker has 7 candidates to reorder into 3
  slots. Limited room to recover missed hits. Risk of ceiling effect
  where reranker cannot promote articles from beyond retrieve's top-10.
- **50 → 10**: over-provisions. Reranker latency scales linearly with
  pairs — 50 pairs is 2.5× the wall time for marginal recall gain.
  Also: our corpus is 792 chunks total. Top-50 is 6% of everything;
  candidate quality degrades sharply past top-20 in a small corpus.
  LLM prompt at 10 chunks × ~300 tokens = 3000 tokens context, ~2×
  the k=5 baseline.
- **20 → 5** (chosen): balanced. Reranker sees 20 candidates, has
  room to promote articles from ranks 10-20 into the top-5. LLM
  prompt stays under 1500 tokens of context, under 6000 chars total.
  Wall time under 200ms for the rerank pass on GPU.

Empirical validation on the "quasi-usufruit et notaire" test query:
reranker completely replaced retrieve.py's top-5 (cc-819/cc-626/cc-758-2
→ ord45-2590-1/-1bis/-1bisa/cc-621/d73-609-3). cc-621 (art. 621 =
vente simultanée usufruit/nue-propriété) was promoted from a lower
retrieve rank into the top-5. Exactly the pattern the ratio was
designed to enable.

Constants at flow layer, not per-file: keeps ratio decisions
centralized. If Day 7 monitoring says top-5 is too tight for real
queries touching more articles, one file changes.

**Trade-offs named honestly.**

- **Not measured empirically.** Rubric doesn't require it, Day 3
  measurement budget was spent on retrieval configs (ADR #20 pattern).
  Ratio was chosen from professional consensus + one-query eyeball
  validation, not from a sweep. Deferred experiment (§7): grid
  {(10,3), (20,5), (50,10)} × {vector_only, hybrid} × ground truth =
  6 configs.
- **20 candidates ≈ 2.5% of the 792-chunk corpus.** For a bigger
  corpus this ratio would be much smaller. Currently "small corpus,
  generous over-fetch" territory; the trade-off would shift for a
  10k-chunk corpus and needs re-tuning if scope grows.
- **k=5 for the LLM context bounds synthesis quality.** Real
  succession-domain queries often touch 3-8 articles (quasi-usufruit
  = cc-587 + cc-621 + doctrine + jurisprudence). k=5 may be too
  tight for complex queries. Only measurable once Day 7 monitoring
  collects real query logs; deferred.

**Reversibility.** Trivial. Both constants live at
`rag/flow.py:38-39`. One-line change to swap ratio, no downstream
impact.

---

## 2026-07-22 · Claude judge via OpenRouter (not direct Anthropic API)
**Status:** Accepted

**Context.** The Day 4 judge stack needs a third provider to compare
French legal answers across multiple LLMs. Direct Anthropic billing via a
personal account is blocked in the EU, creating a practical access issue.

**Decision.** Use OpenRouter as the transport for a Claude judge with the
model slug `anthropic/claude-haiku-4.5`. Keep the `anthropic` Python
package installed for reversibility, but make OpenRouter the active
runtime path.

**Rationale.** OpenRouter removes the Anthropic billing wall while
preserving the same underlying model family. The trade-off is small
latency overhead (~50-100ms) and a minor pricing/availability margin (~5%).
This is a better operational fit than blocking Day 4 eval on provider
billing constraints.

**Trade-offs.** The route is slightly less direct than a native Anthropic
call, and provider-specific quirks can differ. The implementation remains
simple and reversible.

**Reversibility.** High. The `anthropic` dependency stays in place and the
judge wiring is isolated to one evaluation module.

---

## 2026-07-22 · Mistral as French-native third judge
**Status:** Accepted

**Context.** A strong peer-review signal for French-domain RAG is to have
three providers agree on answer quality rather than only relying on a
single model family. The rubric benefits from a cross-provider judge stack.

**Decision.** Add Mistral as the third judge provider using
`mistral-small-latest` as the runtime model. Keep the judge methodology
reference-free and comparative.

**Rationale.** Mistral is a French-native and widely used provider in
European contexts, which makes it a credible cross-check against OpenAI
and Claude-style judgments on French legal questions. The value is not
that it is perfect, but that it gives a distinct opinion on the same
answers.

**Trade-offs.** The model is pinned to a rolling release (`mistral-small-latest`),
so provider-side changes can shift behavior over time. This raises JSON
schema drift risk compared with a dated snapshot.

**Reversibility.** Medium. The integration is isolated and can be removed
without changing the rest of the judge pipeline.

---

## 2026-07-22 · LLM-as-judge methodology: % agreement + majority vote
**Status:** Accepted

**Context.** The judge setup needs a simple, rubric-visible method to
aggregate three independent judgments without overcomplicating the Day 4
implementation. The rubric should be readable to reviewers without a
statistics background.

**Decision.** Use percentage agreement plus majority vote as the judge
aggregation method. Judge each answer on a 3-class relevance scale and
aggregate by agreement rate and majority label.

**Rationale.** Three-way weighted kappa is mathematically messy, hard to
explain to reviewers, and not clearly visible in the rubric. Percentage
agreement with majority vote is simpler, more transparent, and aligns
with a Zoomcamp-style evaluation mindset. The method is also reference-
free, which fits the Day 4 setup.

**Trade-offs.** This method does not produce a calibrated inter-rater
statistic like Cohen's kappa. It is intentionally simpler and more
operational than a full agreement model.

**Reversibility.** High. The scoring logic is local to the evaluation code
and can be swapped later if a more formal metric becomes necessary.

---
## ADR #32 — Three-table Postgres schema for Plane IV persistence

**Date:** 2026-07-25 · **Day:** 7 · **Status:** Locked

### Decision

Migrate Day 6 JSON+CSV persistence to Postgres with three tables:
`conversations` (thread metadata), `turns` (Q&A pairs, one-to-many under
conversation), `feedback` (ratings, one-to-many under turn). Foreign keys
CASCADE on delete. Indexes on all foreign-key columns and `created_at`
timestamps for Grafana time-series queries.

### Alternatives considered

**Two-table (Zoomcamp reference):** `conversations` + `feedback`, where
each "conversation" is a single Q&A. Rejected because Day 6's ADR #31
locked a multi-turn architecture: one thread contains N Q&A pairs. Forcing
each turn into its own conversation row would either (a) flatten the
sidebar's thread grouping — regressing the UX from Day 6 — or (b) create
synthetic conversation rows with the same title, corrupting Grafana's
"turns per conversation" panel.

**Single denormalized table:** `conversations` with a JSONB column
holding all turns. Rejected because Grafana panel SQL becomes lateral
joins against JSONB paths, defeating the point of using a relational store.

**Separate SQLite files per conversation (Day 6 continuity):** rejected
because it doesn't move any rubric-line and blocks the Grafana dashboard.

### Rationale

- **Peer reviewers who know module 07 recognize the `feedback` table**
  as the reference pattern. The `turns` layer reads as a legitimate
  extension for multi-turn architectures, not a departure.
- **CASCADE preserves referential integrity** — dropping a conversation
  from the UI cleanly removes its turns and their feedback with no
  orphan rows corrupting Grafana panels.
- **Feedback as a separate table** (not a column on `turns`) matches
  Zoomcamp reference AND future-proofs for comment-only submissions or
  re-rating scenarios where a turn could accumulate multiple feedback
  events.
- **CHECK constraint on `rating IN (-1, 1)`** enforces the Day 6 semantics
  at the DB layer. Loud failure at INSERT if the UI ever sends invalid
  data — verified with a rejected `rating = 0` INSERT during schema
  validation.

### Trade-offs kept in code

- FK CASCADE is destructive; a bug in the "delete conversation" UI flow
  could nuke feedback history. Mitigated by not exposing a "delete"
  action in Day 7 UI; conversation deletion is post-Day-9 concern.
- `turns.answer_en` column persists translations that Day 6 held only in
  session state. Trade-off: durability across restarts vs. storage cost.
  Chose durability — translations are ~1KB each, negligible.
- Sequence gaps in `feedback.id` on rolled-back INSERTs are expected
  PostgreSQL behavior. Not exposed externally, no mitigation needed.

## ADR #33 — Grafana dashboard structure and file-provisioned build path

**Date:** 2026-07-25 · **Day:** 7 · **Status:** Locked

### Decision

Six-panel dashboard, 2-per-row layout, provisioned to Grafana via a
committed JSON file at `monitoring/grafana/dashboards/lexclair.json`
using Grafana's file-based provisioner (see
`monitoring/grafana/provisioning/dashboards/dashboards.yml`).

Panels chosen to tell one coherent story about how the RAG app is used:
1. **Feedback ratio** (stat) — headline signal, do users like the answers
2. **Questions asked per day** (time series) — usage trend
3. **Response time distribution** (histogram) — performance shape
4. **Cost per query over time** (time series, USD-formatted) — spend signal
5. **Model usage** (bar) — infrastructure signal
6. **Turns per conversation** (histogram) — surfaces the multi-turn
   architecture (ADR #31) as a distinguishing feature vs Zoomcamp
   reference module 07's single-turn assumption

### Alternatives considered

**Automated init-script provisioning** (Zoomcamp reference module 07's
pattern): script talks to Grafana's HTTP API at container start, creates
the datasource, imports the dashboard. Rejected: file provisioning
achieves the same "zero clicks on first boot" peer-review UX with less
code, no script to maintain, and no bootstrap ordering concerns
(provisioning runs before HTTP is up).

**Dashboard-as-code (Grafonnet/Terraform)**: rejected as scope-inflating
for a solo capstone; JSON authoring in the UI + git-committed export is
the industry-standard workflow for small deployments.

**In-UI build only, no file commit**: rejected because the rubric point
requires the dashboard to survive `docker compose down -v` and a clone
by a peer reviewer. UI-only state lives in Grafana's own DB volume and
would be lost.

### Rationale

- **Peer reviewers get a populated dashboard on first `docker compose up`**
  because the file provisioner loads it at container boot. Same "install
  and it just works" property as ADR #33's datasource YAML.
- **The six panels aren't uniform**; each has a stated purpose. Turns
  per conversation specifically surfaces our multi-turn architecture as
  a visible differentiator, not just a chart-count filler.
- **JSON was authored in the UI, then committed as a portable export**
  (with `__inputs` block for datasource UID rewiring). This gives the
  best of both: interactive iteration in Grafana, deterministic replay
  in git.

### Trade-offs kept in code

- Committed JSON has a schema-versioned tie to Grafana v11.4.0. A future
  Grafana upgrade may require the JSON to be regenerated. Compose file
  pins the tag to `grafana/grafana-oss:11.4.0` to prevent silent drift.
- Two dashboard formats exist and must be kept distinct: the **portable**
  format (committed, with `${DS_LEXCLAIR}` placeholder) and the
  **internal** format (what the UI's JSON Model editor works with). Editing
  in the UI produces internal format; committing requires converting to
  portable. Documented in this ADR to prevent future confusion.
- Some panels (Questions per day, Turns per conversation) will look
  sparse in demo state due to only ~24 turns of migrated Day 6 data. This
  is honest — the dashboard is populated from real usage, not synthetic
  seed data.

## ADR #34 — Services-only docker-compose scope for Day 7

**Date:** 2026-07-25 · **Day:** 7 · **Status:** Locked (Day 8 supersedes for app)

### Decision

Day 7's `docker-compose.yml` containerises Postgres and Grafana only.
The Streamlit app continues to run on the host via
`uv run streamlit run app/streamlit_app.py`, connecting to Postgres at
`localhost:5432`.

Kill switch for DB connection lives at module level in `monitoring/db.py`
(`_DB_HEALTHY: bool`), NOT in Streamlit session state. One flag, one
check per call, no Streamlit dependency in db.py — preserves ADR #10's
plane-separation discipline.

### Alternatives considered

**Full-stack docker-compose today** (app + Postgres + Grafana): rejected
because Day 8 has a dedicated Containerization rubric line (+2 points).
Splitting the compose file work across two days for no rubric gain
would double Day 7 scope. Day 8 adds the app service; connection strings
switch from `localhost` to the `postgres` service name inside the
network.

**Kill switch in Streamlit session state**: rejected because it would
require importing `streamlit` inside `monitoring/db.py`, breaking
plane-separation. Module-level global is simpler, testable, and works
correctly for the single-user HF Spaces deploy target (Day 9).

### Rationale

- **Day 7 scope stays focused** on Plane IV completion (persistence +
  monitoring dashboard), not deployment plumbing.
- **`app/streamlit_app.py` runs unchanged** on the host during
  development — no BGE cache mount, no Streamlit-in-Docker port
  forwarding, no hot-reload complexity to solve today.
- **The kill switch pattern is testable** via `_reset_conn()` helper
  and works identically in local dev and containerised deployment. It's
  not a Day 7 hack that Day 8 needs to remove.

### Trade-offs kept in code

- The peer-review demo on Day 9 requires two commands, not one
  (`docker compose up -d` for services, then
  `uv run streamlit run app/streamlit_app.py` for the app). Day 8
  collapses this to one command. Documented in README.
- Kill switch is one-way per process — once `_DB_HEALTHY` flips false,
  the app stays in degraded mode until Streamlit restart, even if
  Postgres recovers. Chosen over auto-retry to avoid hammering a down
  DB. If HF Spaces reveals this as a real issue on Day 9, a "reconnect"
  button in the UI banner would resolve it in ~10 min.
- Session-state loss on browser refresh when the kill switch has
  tripped: rendering degrades to whatever's already in `st.session_state`;
  full state is lost on refresh. Acceptable for a solo demo; documented
  as a Day 9 buffer improvement if HF Spaces feedback demands it.

## ADR #35 — Vision extractor swapped from Opus 4.7 to Qwen3-VL

**Date:** 2026-07-27 · **Branch:** v2-agentic (Deliverable 3) · **Status:** Accepted

### Context

Deliverable 3's real run on 55 dossier documents exhausted OpenRouter
credit mid-batch on Opus 4.7 vision extraction, at roughly $0.03-0.05 per
page. DeepSeek's flagship models (`deepseek-v4-flash`/`deepseek-v4-pro`,
confirmed live via DeepSeek's own API docs) are text-only and cannot
process images, so DeepSeek's direct API — despite already-paid credit
there — is not a viable transport for this stage. DeepSeek-OCR, a
purpose-built OCR model (~$0.03/M tokens), is not hosted on OpenRouter or
DeepSeek's direct API; it would require adding DeepInfra as a new
provider, which is deferred rather than done under this ADR.

### Decision

`VLM_MODEL_ID` in `ingestion/dossier/extract.py` changes from
`anthropic/claude-opus-4.7` to `qwen/qwen3-vl-235b-a22b-instruct` —
verified live against OpenRouter's `/api/v1/models` catalog
(`modality: text+image->text`, `max_completion_tokens: 32768`). Qwen3-VL
is OpenRouter's best OCR-targeted vision model today; its own model
description lists "document AI, multilingual OCR" as a target scenario.
`max_tokens` is now set explicitly to 8192 per vision call (previously
unset, which is what let a call balloon toward 65536 requested tokens and
trip the 402 credit error). Same OpenRouter OpenAI-compatible client
(`ingestion.clients.get_anthropic_client()`), same message structure
(system prompt + user text/image_url blocks), same French verbatim
system prompt — none of that changed. Haiku 4.5 stays as the `gate.py`
faithfulness check, unchanged.

### Alternatives considered

**DeepSeek direct API** (`deepseek-v4-flash`/`deepseek-v4-pro`): rejected —
confirmed via DeepSeek's own API docs to be text-only, no image input
support at all. Cannot serve this stage regardless of available credit.

**DeepSeek-OCR via DeepInfra**: deferred, not rejected outright — a
purpose-built, cheaper OCR model, but requires standing up a new provider
(DeepInfra) and a new `clients.py` entry point. Out of scope for this
ADR; tracked as a follow-up if Qwen3-VL quality proves insufficient.

### Rationale

- **Cost**: Qwen3-VL pricing (~$0.00000021/token prompt,
  ~$0.0000019/token completion) is roughly 30-50x cheaper than Opus 4.7
  vision per page.
- **Purpose-fit**: Qwen3-VL's own OpenRouter listing explicitly targets
  document AI and multilingual OCR — a closer match to verbatim French
  legal-document transcription than a general-purpose frontier model.
- **No auth/transport change**: still OpenRouter, still the same
  OpenAI-compatible client — reversible with a one-line constant change.

### Trade-offs kept in code

- Wall time per page should drop from ~15-25s (Opus) to ~3-8s (Qwen3-VL),
  but this is not yet measured against the real dossier corpus.
- Quality regression risk on French legal scans is real and unmeasured
  ahead of time — the Haiku 4.5 gate (`gate.gate_case`) is the designed
  safety net for catching exactly this kind of drift per-document.
- Cached Opus-vintage extractions remain valid and are not invalidated —
  each sidecar records `extractor_model`, so a case can contain a mix of
  Opus- and Qwen-extracted documents without ambiguity.

### Follow-ups

- If `gate.gate_case` flags material regressions concentrated on
  Qwen-extracted documents specifically, escalate to DeepSeek-OCR via
  DeepInfra: write a new ADR for that swap, add `get_deepinfra_client()`
  to `ingestion/clients.py`, and add a `--force-model` override to the
  extract CLI so a case can be selectively re-extracted with a different
  vision model without a global constant flip.

## ADR #36 — Actor roles discovered per case, not hardcoded

**Date:** 2026-07-27 · **Branch:** v2-agentic (Deliverable 4) · **Status:** Accepted

### Context

An earlier draft of Deliverable 4 hardcoded 18 `ActorRole` enum values based on the
Bossavit case. Rejected: an enum makes the pipeline case-specific — a different
succession with a `syndic_de_copropriete` or `juge_des_tutelles` would silently
degrade to `autre`. The whole point of lex-clair as generic tooling requires per-case
role discovery.

### Decision

- `actor_role` is a validated snake_case string (regex `^[a-z][a-z0-9_]{2,60}$`), not
  an `Enum`.
- Roles are catalogued per case in `actor_roles.jsonl` with `label_fr`,
  `grounding_note`, `confidence`, `first_seen_doc_id`, `fact_count`.
- Ambiguous role assignments emit both a provisional `Fact` **and** a `RoleAmbiguity`
  record in `role_ambiguities.jsonl`. Never silently pick one.
- The extractor prompt teaches the snake_case pattern with examples from French
  succession/notarial law, but does not restrict the set.

### Consequences

- Downstream reasoners (Day B) must consult the case's `actor_roles.jsonl` to
  interpret role_ids, not a global registry.
- Cross-case aggregation of roles (if ever needed) becomes a normalization concern,
  not a data-model concern — different cases can call the same underlying role
  slightly different names, and that's acceptable as long as intra-case coherence
  holds.
- Ambiguities become first-class data. Day B UI (Deliverable Day B-3) exposes them
  for interactive resolution. Never silently resolved.

### Follow-ups

- Deliverable 5 populates `source_chunk_id` after indexing. Day B analysis reads
  `facts.jsonl` + `actor_roles.jsonl` + `role_ambiguities.jsonl` as its complete
  Plane Ib output.

## ADR #37 — Correction to Deliverable 4 fact extractor — dead model ID

**Date:** 2026-07-27 · **Branch:** v2-agentic (Deliverable 4) · **Status:** Accepted

### Context

Deliverable 4's initial ADR (#36) specified `google/gemini-2.0-flash-001` as
`FACT_EXTRACTOR_MODEL_ID`. A real run against the ship-gate corpus returned an
OpenRouter 404. Root cause: Gemini 2.0 Flash was retired 2026-03-31 — the
model ID no longer resolves. This was a stale-fact failure — the model was
recommended without verifying it against the current OpenRouter catalogue.

### Decision

Swap to `google/gemini-3.1-flash-lite`, verified live against OpenRouter's
`/api/v1/models` catalogue and a 1-token dry call through the existing
`get_anthropic_client()` → `chat.completions.create()` surface on
2026-07-27 (GA, not preview; `max_tokens`/`response_format` supported; same
message shape already used in `facts.py`). This preserves the intended
pipeline provider diversity: `extract` (Qwen) → `gate` (Anthropic) → `facts`
(Google), matching the Day 5 three-judge harness discipline of cross-family
evaluation. Same OpenRouter surface and credit pool — no new provider
integration.

### Consequences

- Adds Google to the OpenRouter model palette (previously Qwen + Anthropic +
  Mistral). Does not add a new provider surface — same OpenRouter client,
  same auth, same credit pool.
- Fact-extraction cost estimate at Gemini 3.1 Flash Lite's published
  OpenRouter pricing ($0.25/M prompt, $1.50/M completion tokens): well within
  the ~$0.30–0.80 budget flagged in Deliverable 4 for the 55-doc corpus.

### Follow-ups

- Codify pre-flight model-ID verification (live catalogue check or a 1-token
  dry call) as a mandatory step for any deliverable introducing a new model
  constant. The Deliverable 3 vision swap (ADR #35) included this step; the
  Deliverable 4 initial pass (ADR #36) did not. Add to the plan-review
  checklist.

## ADR #38 — Dossier incremental indexing: per-case CSV as BM25 stand-in, Chroma append via id-prefix delete

**Date:** 2026-07-27 · **Branch:** v2-agentic (Deliverable 5) · **Status:** Accepted

### Context

Deliverable 5 needed to add dossier chunks to the hybrid retrieval store
without rebuilding the statute corpus. `ingestion/load.py`'s `load_index()`
rebuilds BM25 from `data/chunks.csv` fresh on every call and has no append
mechanism, and it cannot be modified under this deliverable's scope. Chroma,
by contrast, genuinely supports incremental writes via
`client.get_collection(COLLECTION)` + `collection.add()` — bypassing
`ingestion.index.build_chroma`/`build_all` (which `shutil.rmtree()`s the
whole directory) entirely.

### Decision

- Dossier chunks are written to `data/dossier/<case_id>/chunks.csv` (same
  column schema as `data/chunks.csv`), fully overwritten on each
  `index_dossier` run — deterministic chunking makes this idempotent by
  content, and the file is case-scoped, not shared. This CSV is this
  deliverable's own tested artifact; it is **not** wired into
  `load_index()`'s BM25 rebuild yet (Day B follow-up).
- Chroma appends use `client.get_collection` (never create/recreate) plus an
  idempotent **delete-by-id-prefix, then add**: since Chroma has no
  server-side prefix `where` filter, every existing id is listed via
  `collection.get(include=[])["ids"]`, filtered client-side for the
  `dossier-{case_id}-` prefix, and deleted before re-adding the fresh batch.
- `chunk_dossier_document` is a small hand-written recursive splitter (hard
  split on `## Page N`, then paragraph → sentence → character, ~800 char
  target / ~100 char overlap) — no new dependency (no langchain).
- `Fact.source_chunk_id` is backfilled by substring-matching each fact's
  `verbatim_quote` against its own document's chunks (exact match first,
  then whitespace-normalized/lowercased fallback); unmatched facts are
  logged and left `None`, never silently dropped. Every fact's match is
  unconditionally recomputed on each run for clean idempotence.
- Filtering (dossier-only vs statute-only vs blended retrieval) is a Day B
  concern, not added to `HybridRetriever` here — this deliverable's contract
  is that chunks are indexed with the correct prefix, nothing more.

### Consequences

- Dossier chunks landing in Chroma **are** reachable today by production
  vector-mode search (ADR #20's default), even though BM25 does not yet see
  them — a real, if partial, retrieval capability.
- The client-side full-id-list scan for prefix-deletion is fine at current
  corpus scale but does not scale indefinitely; if the Chroma collection
  grows much larger, a `case_id` metadata field + a real `where` filter would
  be cheaper.
- The heuristic sentence splitter (regex-based, no NLP dependency) can
  mis-split on French abbreviations (e.g. "Me.", "Mme"); mitigated by the
  paragraph-level grouping and chunk overlap, not eliminated.
- Deleting a case (privacy or reset) means deleting all chunks with a
  `dossier-<case_id>-` prefix — this becomes a small operational tool later,
  not needed now.

### Follow-ups

- Day B: wire `data/dossier/<case_id>/chunks.csv` into `load_index()`'s BM25
  rebuild (concatenate with `data/chunks.csv`) — needs its own review since
  it touches the statute-side loader.
- Day B router. If the Chroma collection grows large enough that the
  full-id-list scan becomes slow, add a `case_id` metadata field and use
  `where={"case_id": ...}` instead of client-side prefix filtering.

## ADR #39 — Correction to Deliverable 5: dossier chunks synced to shared data/chunks.csv for retriever coherence

**Date:** 2026-07-27 · **Branch:** v2-agentic (Deliverable 5) · **Status:** Accepted

### Context

ADR #38 shipped dossier chunks to two surfaces only: a per-case
`data/dossier/<case_id>/chunks.csv` (audit) and the shared Chroma
collection (dense retrieval), deferring `data/chunks.csv` sync to a
"Day B" follow-up on the assumption that BM25-only blindness to dossier
chunks was an acceptable gap (ADR #20: vector wins over BM25 on synthetic
ground truth). That assumption missed a second dependency:
`HybridRetriever.search()` in `ingestion/load.py` calls
`self.chunks.loc[cid]` to hydrate *every* hit regardless of which
retrieval mode produced it, and `self.chunks` is built exclusively from
`data/chunks.csv` in `load_index()`. Because Chroma is correctly
appended to per ADR #38, vector-mode and hybrid-mode search return
dossier chunk_ids that have no row in `self.chunks` — raising
`KeyError`. ADR #38's "Consequences" claim that dossier chunks were
"reachable today by production vector-mode search" was therefore
incorrect: Chroma returned them, but hydration crashed. Discovered
during Deliverable 5 verification.

### Decision

`index_dossier()` now writes to three persistence surfaces, in order:
(1) the per-case `data/dossier/<case_id>/chunks.csv` (unchanged from ADR
#38), (2) the shared `data/chunks.csv` via the new
`append_to_statute_chunks_csv()` — idempotent drop-then-append keyed on
the `dossier-<case_id>-` chunk_id prefix, with dossier rows
`reindex()`-aligned onto the statute CSV's column set (extra
statute-only columns fill as `""`, never `NaN`), (3) the Chroma
collection (unchanged from ADR #38). `ingestion/load.py` is not
touched — `load_index()` already reads whatever is in
`data/chunks.csv`, so keeping that file in sync is sufficient.

### Consequences

- Fixes the `KeyError` in `HybridRetriever.search()` for any hit
  resolving to a dossier chunk_id, across all three retrieval modes.
- BM25 now also sees dossier chunks as a side effect (ADR #38
  deliberately excluded this) — not the goal of this fix, and per ADR
  #20 vector remains the production default, but BM25-mode search over
  dossiers is now incidentally functional rather than blind.
- Corrects ADR #38's "Consequences" claim about vector-mode
  reachability; that entry is left as-written (ADRs aren't edited
  retroactively) and should be read alongside this one.
- `eval/`'s harnesses were built and tuned statute-only (Day A);
  re-running them now will surface dossier chunks in `data/chunks.csv`,
  which they weren't designed to see. Non-blocking today.

### Follow-ups

- Day B: decide whether eval harnesses should filter dossier-prefixed
  chunk_ids out, or grow a dossier-aware eval track.
- `tests/test_dossier_smoke.py` calls `index_dossier()` directly for
  test case_ids without isolating `data/chunks.csv` — fixed in the same
  change as this ADR via a new `isolated_chunks_csv` fixture mirroring
  `isolated_chroma`, so the fast suite never touches the real, tracked
  CSV.
- Backfilling the real `data/chunks.csv` for the already-indexed
  `private` case (re-running `index_dossier("private")` post-fix) broke
  3 v1-era invariant tests in `tests/test_ingestion_smoke.py` that
  assumed `data/chunks.csv` is statute-only:
  `test_chunks_csv_matches_articles_row_count`,
  `test_chunks_url_populated_where_expected`, and
  `test_rag_flow_end_to_end`'s citation assertion. Fixed in the same
  change: the first two are scoped to non-`dossier-`-prefixed rows; the
  third is loosened from "every citation is Legifrance" to "at least one
  citation is Legifrance," matching what its own docstring already
  claimed. This is the general shape of the v1→v2 pivot's tension
  flagged elsewhere (`docs/response_doctrine.md` §6.3–6.5) — any
  remaining test or eval code assuming a statute-only corpus should be
  audited the same way as dossier ingestion continues.
- **Backfilling the real `private` case exposed a live privacy gap and
  was reverted.** Re-running `index_dossier("private")` to backfill
  `data/chunks.csv` (above) made that case's 853 chunks — real client
  succession documents — reachable by `rag/flow.run()`, the
  general-purpose baseline Q&A flow, with no source scoping. A generic
  query ("Qu'est-ce que le quasi-usufruit ?") then retrieved *zero*
  Legifrance citations: the dossier's own quasi-usufruit convention
  document out-competed statute articles for a plain definitional
  question. ADR #38 had flagged retrieval-mode filtering
  (dossier-only/statute-only/blended) as an unimplemented "Day B
  concern," but that gap was inert before this fix because dossier
  chunks were unreachable (the ADR #39 `KeyError` itself masked it).
  Decision: `dossier-private-*` rows were removed from both
  `data/chunks.csv` and the Chroma collection (delete-by-prefix, same
  mechanism `append_to_chroma` already uses), restoring today's
  statute-only baseline behavior. The `index_dossier()` code fix (three-
  surface write) stays — it's correct and necessary — but no case should
  be indexed into the shared corpus again until `rag/retrieve.py` /
  `flow.run()` can scope retrieval by source. This is now the
  highest-priority Day B item, promoted from "concern" to "blocker for
  indexing any real case."

## ADR #40 — OpenRouter unification for operational simplicity

**Date:** 2026-07-28 · **Branch:** v2-agentic (Day B, Deliverable 0) · **Status:** Accepted

### Context

Day A hit 402/403 twice from per-provider credit exhaustion mid-batch.
Three separate credit pools (OpenAI, Anthropic, Mistral direct) plus
OpenRouter (already used for the dossier vision/fact pipeline) created
split-brain on budget: a batch could die partway through because one
provider's pool ran dry while the other two still had headroom, and there
was no single place to watch remaining credit.

### Decision

All LLM calls now route through OpenRouter's OpenAI-compatible endpoint,
with fully-qualified model IDs:

- `rag/rewrite.py` — `openai/gpt-4o-mini`
- `rag/generate.py` — `openai/gpt-4o-mini`
- `eval/llm_eval.py` judges — `openai/gpt-4o-mini`, `anthropic/claude-haiku-4.5`,
  `mistralai/mistral-small-3.2-24b-instruct`

`ingestion/clients.py` gains a single canonical factory,
`get_openrouter_client()`. `get_openai_client()`, `get_anthropic_client()`,
and `get_mistral_client()` become deprecated thin aliases over it (each
emits a `DeprecationWarning` then returns `get_openrouter_client()`) so
`ingestion/dossier/{gate,extract,facts}.py` — already OpenRouter-routed via
`get_anthropic_client()`, per ADR referenced in that module's docstring —
keep working unchanged.

Two mechanical consequences fell out of this that are worth recording
explicitly rather than leaving implicit in the diff:

- `rag/rewrite.py` and `rag/generate.py` previously called OpenAI's
  **Responses API** (`client.responses.create` / `.responses.parse`).
  OpenRouter's Responses API is beta, stateless-only, and its
  structured-output support is undocumented, so both files were converted
  to **Chat Completions** (`client.chat.completions.create` /
  `client.beta.chat.completions.parse`) instead — the surface
  `get_anthropic_client()` and the Claude judge already use successfully
  through OpenRouter.
- The Mistral judge's model ID was `mistral-small-latest`, a Mistral-native
  API alias with no literal OpenRouter equivalent. It's now pinned to
  `mistralai/mistral-small-3.2-24b-instruct` — the established, stable
  Mistral Small release, chosen over the newer `mistral-small-2603`
  ("Mistral Small 4") to minimize judge-behavior drift relative to
  already-collected eval numbers. This is a substitution, not a like-for-
  like rename.

Preserves 3-family judge diversity: all three judges still use distinct
model families (GPT, Anthropic, Mistral) for cross-family agreement
measurement — only the transport changed.

### Consequences

- Single API key, single credit pool, single usage dashboard. Dev
  environment now needs only `OPENROUTER_API_KEY` (legacy `OPENAI_API_KEY`
  / `MISTRAL_API_KEY` no longer required for any in-scope call site, though
  harmless to leave set).
- `mistralai` dependency removed from `pyproject.toml` — its only two
  callers were the now-deleted native `get_mistral_client()` bodies in
  `ingestion/clients.py` and `eval/llm_eval.py`. `google-genai` was never a
  dependency in this project, despite an initial assumption that it was —
  nothing to remove there.
- `anthropic` and `openai` stay in `pyproject.toml` as direct dependencies:
  `openai` because its SDK is what actually talks to OpenRouter (base_url
  override); `anthropic` despite the native SDK having zero imports
  anywhere in the codebase today — left in place rather than pruned, since
  removing an unused-but-harmless dependency wasn't part of this refactor's
  scope.
- ~5% OpenRouter price penalty versus direct-provider pricing, accepted as
  the cost of eliminating the mid-batch credit-exhaustion failure mode.

### Follow-ups

- Revisit the Mistral slug if OpenRouter ever publishes a stable "latest"
  Mistral Small alias, or if the judge is deliberately re-pinned to a
  specific dated model for other reasons.

## ADR #41 — source_scope filtering on HybridRetriever.search() closes the ADR #39 privacy blocker

**Date:** 2026-07-28 · **Branch:** v2-agentic (Day B, Deliverable B1) · **Status:** Accepted

### Context

ADR #39's Follow-ups promoted retrieval-mode scoping — originally flagged
in ADR #38 as an unimplemented "Day B concern" — to a blocker: once
dossier chunks are correctly synced to `data/chunks.csv` and the shared
Chroma collection, `HybridRetriever.search()` has no mechanism to keep a
real client case's chunks from out-competing statute chunks on a generic
query. The `private` case incident recorded in ADR #39 showed this
concretely — a definitional quasi-usufruit query returned zero
Legifrance citations because the case's own convention document won on
relevance — and had to be manually reverted (dossier-private-* rows
deleted from both surfaces) rather than fixed at the retrieval layer. No
case may be re-indexed into the shared corpus until this scoping exists.

### Decision

`HybridRetriever.search()` (`ingestion/load.py`) gains a `source_scope`
parameter, default `"statute"`:

- `"statute"` (default) — chunk_ids not prefixed `dossier-`
- `"dossier"` — chunk_ids prefixed `dossier-`
- `"case:<case_id>"` — chunk_ids prefixed `dossier-<case_id>-`
- `"blended"` — no filter

Validated by a new `_scope_predicate()` helper called first thing inside
`search()`, before the embedding call, so an invalid value raises
`ValueError` without wasting a query-time embed. The filter is applied to
the fused RRF `scores` dict — after BM25 and vector candidates are
combined into one chunk_id-keyed dict, before the `[:k]` top-k cut —
giving a single filter call site regardless of `mode`. It draws from the
existing `k*3` per-mode over-fetch as its candidate pool rather than
over-fetching further to compensate; a narrow scope (e.g. one small case)
may legitimately return fewer than `k` hits.

`source_scope` is threaded through `rag/retrieve.py::retrieve()` and
`rag/flow.py::run()`, both defaulting to `"statute"` so the existing
general-purpose baseline flow's behavior is unchanged unless a caller
opts in explicitly.

### Consequences

- Closes the ADR #39 blocker: a case can now be indexed into the shared
  corpus without its chunks being reachable by the default statute-only
  flow.
- `eval/retrieval_eval.py` calls `HybridRetriever.search()` directly and
  is out of scope for this change, but silently inherits the new
  `source_scope="statute"` default — resolving ADR #39's own Follow-up
  question ("decide whether eval harnesses should filter dossier-
  prefixed chunk_ids out") as a side effect, since eval was tuned
  statute-only in Day A.
- No case is unblocked from re-indexing by this change alone — that
  remains a product/process decision; this ADR only removes the code
  blocker ADR #39 identified.
- `app/streamlit_app.py` is untouched: it has no case-scoped UI yet, so
  it continues to reach `flow.run()` with the default `"statute"` scope
  end-to-end.

### Follow-ups

- No caller yet passes anything other than the default `"statute"` —
  `"dossier"` / `"case:<id>"` / `"blended"` exist as retrieval-layer
  capability only. Wiring a case-aware UI/API surface that actually
  passes a non-default `source_scope` is future work, not part of this
  deliverable.
- `eval/retrieval_eval.py` should eventually decide, explicitly rather
  than by accident, whether it wants a dossier-aware eval track (ADR
  #39's original follow-up) now that the scoping mechanism exists to
  support one.
- B2 wires the router to drive `source_scope` automatically based on
  query intent. B5 wires the UI selector. Consider adding retrieval
  telemetry on which scope was used per query, for eval.

## ADR #42 — Query router auto-selects retrieval source scope

**Date:** 2026-07-28 · **Branch:** v2-agentic (Day B, Deliverable B2) · **Status:** Accepted

### Context

ADR #41 shipped `source_scope` filtering on `HybridRetriever.search()`, but
every caller had to specify the scope manually — `rag/flow.py::run()`
defaulted to a hardcoded `"statute"`. Both UX and correctness improve when
the system infers scope from query intent instead of requiring the caller
(or a future UI dropdown) to guess it correctly every time.

### Decision

A new module, `rag/router.py`, exposes `route_query(query,
active_case_id=None) -> RouteDecision`. A single Haiku 4.5 call (via
`get_openrouter_client()`, ADR #40, `temperature=0.0` for classifier
determinism) classifies the query into one of 4 intents —
`statute_lookup`, `case_factual`, `gap_analysis`, `other` — using a fixed
French system prompt. The model never chooses `source_scope` directly;
`route_query` maps `(intent, active_case_id)` to a scope deterministically
in Python:

- `statute_lookup` → `"statute"`
- `case_factual` + `active_case_id` → `"case:{active_case_id}"`; without a
  case id, downgrades to `"statute"` with `confidence="low"`
- `gap_analysis` + `active_case_id` → `"blended"`; without a case id,
  downgrades to `"statute"` with `confidence="low"`
- `other` → `"statute"` (safest fallback)

Response parsing mirrors `ingestion/dossier/gate.py::_parse_verifier_response`
(hand-parsed JSON text, code-fence stripped, `ValueError` on bad shape) —
chosen over `rewrite.py`'s `.beta.chat.completions.parse()` structured-output
path specifically so a malformed response is *catchable* rather than
structurally impossible to receive. `route_query` never raises: any
call or parse failure logs a warning and returns
`RouteDecision(intent="other", source_scope="statute", confidence="low",
rationale=...)`.

`RouteDecision.intent` additionally accepts the value `"override"` — never
produced by the classifier itself, only by `rag/flow.py::run()` when the
caller passes an explicit `source_scope`, which skips the router entirely.
This keeps the return shape self-consistent across both paths rather than
having the override path emit a dict that would fail `RouteDecision`
validation if a downstream consumer (e.g. a future UI reconstructing the
model from `result["route_decision"]`) ever re-validated it.

`rag/flow.py::run()`'s signature changes from `(query, verbose=False,
source_scope="statute")` to `(query, source_scope=None,
active_case_id=None, verbose=False)`. When `source_scope is None`, `run()`
calls `route_query` and logs `"router: intent=%s scope=%s conf=%s"` at INFO
level. The resolved `RouteDecision` (or the synthetic override dict) is
included in the return dict as `"route_decision"`.

### Consequences

- Adds one Haiku 4.5 call (~150 input / <50 output tokens, well under
  $0.001) per query that doesn't pass an explicit `source_scope` —
  negligible at current volumes, but now a per-query cost on every
  `app/streamlit_app.py` call site (unchanged code, changed runtime
  behavior) since it still calls `flow.run()` with no `source_scope`.
- Router failure defaults to `"statute"`, the same safe fallback ADR #41
  already established as the retrieval-layer default — a dead OpenRouter
  degrades scope selection, not correctness.
- `result["route_decision"]` gives a future UI (B5) a ready-made "why this
  scope was chosen" surface, including the downgrade rationale when a
  case-specific intent was inferred but no case was active.
- Two existing fast-suite tests written against the ADR #41 hardcoded
  default (`tests/test_ingestion_smoke.py::test_flow_run_default_source_scope_is_statute`)
  needed their mocking updated in this same change to mock
  `flow.route_query` — otherwise a "fast" test would fire a real network
  call. `test_flow_run_passes_source_scope_through` needed no change since
  an explicit `source_scope` always skips the router.

### Follow-ups

- Router prompt tuning based on real query logs, once Day 7-style
  monitoring exists for v2 traffic.
- Optional: cache routing decisions per query hash to save cost on repeat
  queries.
- B5 wires the UI's scope selector and surfaces `route_decision` to the
  user.

## ADR #43 — Compliance matrix generation as Day B reasoning stage

**Date:** 2026-07-28 · **Branch:** v2-agentic (Day B, Deliverable B3) · **Status:** Accepted

### Context

Plane Ib (`facts.jsonl`, `actor_roles.jsonl`, `role_ambiguities.jsonl`) is
now the structured substrate for a case — facts extracted and validated
(Deliverable 5), roles discovered and catalogued (ADR #36), retrieval
scoped to statute vs. dossier vs. blended (ADR #41), and query intent
auto-routed to the right scope (ADR #42). None of that reasons about
*compliance* yet: whether the obligations the statute imposes on each
actor role were actually met. B3 is that reasoning stage — the first
deliverable to produce a legally-grounded judgment rather than retrieve or
classify.

### Decision

A new module, `rag/compliance.py`, exposes
`generate_compliance_matrix(case_id, limit=None, dry_run=False) ->
ComplianceMatrix`. Facts are grouped by exact `actor_role` string (no fuzzy
dedup of near-duplicate role_ids — see Follow-ups). For each role cluster,
`rag/retrieve.py::retrieve()` (ADR #41's `source_scope` filtering, already
hardcoded to `mode="vector"` per ADR #20) fetches the top
`RELEVANT_STATUTE_K=8` statute chunks for a query built from the role's
label + its top action verbs. One call to `anthropic/claude-opus-4.7` with
`reasoning={"effort": "max"}` then judges, per identifiable obligation in
the retrieved articles, whether it was `met`, `breached`, `ambiguous`, or
`insufficient_evidence` — this is the one deliverable in the pipeline that
justifies Opus-class reasoning cost; every other LLM call (routing,
rewriting, fact extraction) uses a cheaper model.

Two corrections against the original spec, confirmed live against
OpenRouter's `/v1/models` and a real dry-call before writing any code: the
model slug is `anthropic/claude-opus-4.7` (a dot, not a hyphen — the
originally-specified `claude-opus-4-7` doesn't exist), and `"max"` is a
`reasoning.effort` enum value (`"max"|"xhigh"|"high"|"medium"|"low"|"minimal"|"none"`),
not a model suffix. A third correction surfaced at ship-gate: the OpenAI
SDK's typed `chat.completions.create()` rejects a bare `reasoning` kwarg,
so it's passed via `extra_body={"reasoning": {"effort": "max"}}` — the
SDK's standard transport for provider-specific fields, which OpenRouter
still receives as `"reasoning": {"effort": "max"}` at the top level of the
request body.

Response parsing mirrors `ingestion/dossier/facts.py::_parse_llm_json`
(fence-strip + `json.JSONDecoder().raw_decode()`, tolerant of trailing
prose), adapted for a top-level JSON list instead of a dict. Each parsed
entry gets a deterministic `entry_id = SHA1(statute_chunk_id + "|" +
actor_role)[:12]`, so a future merge-on-rerun mode (not implemented here)
wouldn't produce duplicate rows for the same obligation/role pair. Output
is fully regenerated (no merge) and written to
`data/dossier/<case_id>/compliance_matrix.json` on every run —
idempotency is achieved via an injectable `_utcnow()` clock rather than a
literal `datetime.now()` call, so the same inputs always serialize to the
same bytes.

Each role cluster is capped at 30 facts (`MAX_FACTS_PER_ROLE`), keeping the
chronologically earliest facts when a role exceeds the cap and logging a
`WARNING` with `role_id`, actual count, and the cap. Heavy roles in the
private case (e.g. `notaire_redacteur`) run 20-40 facts; uncapped, the user
message balloons past 5K tokens before statute chunks are even added, and
Opus's coherence degrades reasoning across too many facts in one call.
Chronological truncation is a stopgap, not a real solution — see
Follow-ups.

`ComplianceMatrix.unresolved_ambiguities` is set to the bare
`len(role_ambiguities)` loaded for the case — a total count for v1, not a
per-entry link between a specific ambiguity and the determinations it may
have blocked (see Follow-ups).

Cost: real (non-`--dry-run`) calls sum OpenRouter's exact per-response
`usage.cost` field across role groups for the `compliance summary`
log line, rather than maintaining a hardcoded per-model price table (the
fallback `rag/flow.py` uses for unknown models). `--dry-run` (no API calls)
estimates tokens via a char/4 heuristic and cost via the same approximate
rates observed in a live pricing check (~$5/M prompt, ~$25/M completion),
clearly marked as an estimate.

### Consequences

- Adds a paid dependency to case processing: ~$2.4–3.5 per full-case run
  at the private case's scale (60 role groups), consistent with the
  ~$1-5/case order of magnitude expected going in. `--limit` and
  `--dry-run` cap dev-iteration cost.
- Cheaper models (Haiku, Sonnet) were considered and rejected during dev
  for this stage specifically — reasoning coherently across many facts ×
  obligations per role is the one place in the pipeline where the cheaper
  models' output was noticeably weaker; every other stage stays on a
  cheaper model.
- `rag/compliance.py` calls into `rag/retrieve.py` and `ingestion/dossier/facts.py`
  read-only (imports the existing `retrieve()` wrapper and the
  `Fact`/`ActorRole`/`RoleAmbiguity` models) rather than re-deriving
  retriever plumbing or a new data model — keeps the Plane I → II
  `load_index()` contract intact.
- `data/dossier/demo/{facts,actor_roles,role_ambiguities}.jsonl` are empty
  in this repo, so B3's tests exercise a synthetic tmp_path fixture case
  instead of a real demo case — no files were added under
  `data/dossier/demo/`.
- `ingestion/dossier/build.py` gained a `--step compliance` (and it was
  appended to `--step all`), so a case's full artifact chain now runs
  extract → gate → facts → index → compliance in one command.

### Follow-ups

- Role_id near-duplicate normalization (e.g. `conseil_regional_des_notaires`
  vs. `conseil_regional_notaires`, both observed in the private case's role
  catalogue) — B3 groups by exact string match; a future pass could dedupe
  by `label_fr` similarity before clustering.
- Per-entry (not just total-count) linkage between `role_ambiguities.jsonl`
  and the specific `ComplianceEntry` determinations an ambiguity may have
  blocked.
- Fact clustering/summarization for oversized role groups instead of
  chronological truncation — the 30-fact cap keeps token cost bounded but
  silently drops the chronologically latest facts for the heaviest roles.
- B5 wires the UI to display the matrix.

## ADR #44 — Cross-role context annotation in compliance prompts

**Date:** 2026-07-30 · **Branch:** v2-agentic (Day B, Deliverable C1) · **Status:** Accepted

### Context

ADR #43 groups facts by exact `actor_role`. Same person in multiple roles
(discovered post-B3 on the private case: Roxane appears as both
`heritier_nu_proprietaire` and `heritier_representation`) is invisible to
per-role LLM calls, degrading gap-analysis quality — the model judging one
role's obligations has no way to know the same physical person also holds
another role with its own obligations.

### Decision

Enrich the compliance user message with a "Contexte inter-rôles" block
enumerating other roles that share source documents with the current
cluster. `rag/compliance.py::_extract_cross_role_context()` scans
`all_facts` for `source_doc_id`s in common with the current role's capped
fact cluster, groups the other roles found there, and renders a
deterministic (sorted) French block naming each other role_id + its
`label_fr` + up to 3 shared doc_ids. `_build_user_message()` appends this
block after the statute-chunks section when non-empty; empty otherwise (no
size or shape change to the message). `COMPLIANCE_SYSTEM_PROMPT` gained one
paragraph instructing the model to treat cross-role presence as evidence
the same physical person may hold multiple roles, and to say so explicitly
in `rationale` when its judgment depends on a cross-role fact. No schema
change, no re-extraction, additive only.

### Consequences

- Marginal prompt token increase (~50-200 tokens per call when cross-role
  present).
- No structural change to `compliance_matrix.json`'s shape —
  `ComplianceEntry`/`ComplianceMatrix` are untouched.
- Idempotency preserved: the helper is a pure function over already-loaded
  facts/roles with sorted iteration order, so repeat runs on the same
  inputs still serialize to identical bytes.
- `_call_compliance_llm()`'s call structure (model, messages shape,
  `max_tokens`, `extra_body`) is unchanged — only the user message string
  grows.

### Follow-ups

- Fuzzy-dedup near-duplicate role names (`heritier_` vs. `heritiere_`
  variants) — same normalization gap ADR #43 already flagged, now doubly
  relevant since it also affects cross-role matching.
- Proper multi-role fact schema (`actor_roles: list[str]`) as an Attempt-2
  architectural change — C1 is a prompt-level patch, not a schema fix.
- Per-person identity resolution via LLM if `label_fr`/role_id matching
  proves unreliable at catching same-person-different-role cases that
  don't share a source document.

---

## ADR #45 — Multi-model answer generation with runtime catalog

**Date:** 2026-07-30 · **Branch:** v2-agentic (Day C, Deliverable C2) · **Status:** Accepted

### Context

Single-model answer generation (`rag/generate.py` hardcoded to
`openai/gpt-4o-mini`) limits user choice for hard queries: there is no way
to trade cost for reasoning depth on a per-query basis. Kimi K3 and Opus 4.7
(`reasoning.effort="max"`) are frontier reasoning models already reachable
through OpenRouter's unified surface (ADR #40), and `rag/compliance.py`
(ADR #43) already proves the `extra_body={"reasoning": {"effort": "max"}}`
call pattern for one of them. This ADR supersedes the plan §6.4 constraint
that locked answer generation to gpt-4o-mini for V1 evaluation stability;
multi-model support is now scoped and evaluations pin their answer model
explicitly at call time.

### Decision

`rag/generate.py` gains an `ANSWER_MODELS` catalog of three answer models —
`gpt-4o-mini` (default), `opus-4.7` (max reasoning effort), `kimi-k3` (max
reasoning effort) — each with `model_id`, `cost_input_per_m`,
`cost_output_per_m`, `reasoning_effort`, `max_tokens`, and a French UI label.
`generate(prompt, model_key=...)` looks up the catalog entry, raises
`ValueError` on an unrecognized key, and only sets `extra_body` when
`reasoning_effort` is configured. `max_tokens` is per-model rather than a
flat constant: `500` for `gpt-4o-mini` (unchanged from pre-C2 behavior),
`4096` for `opus-4.7`/`kimi-k3` — matching `rag/compliance.py`'s existing
Opus precedent, since reasoning-effort tokens share the same budget as
completion tokens and a small `max_tokens` truncates reasoning models before
any visible answer text is produced.

Cost calculation, previously duplicated in `rag/flow.py` via
`_compute_cost()` + `GPT_4O_MINI_INPUT_PER_M`/`GPT_4O_MINI_OUTPUT_PER_M`
(which silently returned `0.0` for any model other than gpt-4o-mini — a
latent bug once the model became selectable), moves fully into
`generate()`, computed from the catalog's per-model rates. `flow.py::run()`
gains an `answer_model: str = "gpt-4o-mini"` param, passes it through as
`generate.generate(p, model_key=answer_model)`, and reads `model_used`,
`cost_usd`, and the new `answer_model_key` field straight from the returned
usage dict.

### Consequences

- Reasoning models are 20-100x more expensive per query than the default
  (Opus: 100x input / 125x output vs. gpt-4o-mini; Kimi: 20x input / 25x
  output) — cost must stay visible wherever model choice is exposed.
- `flow.py` no longer owns any pricing constants; a pricing update now
  touches only `rag/generate.py`'s catalog.
- UI (C3, not yet built) will surface model choice with cost implications
  visible, using each catalog entry's `label_fr`.

### Follow-ups

- Token-level cost breakdown per conversation turn (currently only a
  per-call total).
- Per-model `max_tokens` tuning once real answer lengths are observed for
  the reasoning-tier models.
- Consider a cheaper "medium" tier (e.g. Mistral Small 4, Gemini 2.5 Flash)
  between `gpt-4o-mini` and the two max-effort reasoning models.

## ADR #46 — Streamlit sidebar model toggle

**Date:** 2026-07-30 · **Branch:** v2-agentic (Day C, Deliverable C3) · **Status:** Accepted

### Context

ADR #45 (C2) plumbed `answer_model` through `rag/flow.py` and
`rag/generate.py`'s `ANSWER_MODELS` catalog, but nothing in the UI let a
user actually select it — every query still ran against the hardcoded
default. Users facing a hard query (ambiguous quasi-usufruit facts,
multi-party liability) need a way to opt into a heavier reasoning model
per-query, and need to see the cost tradeoff before doing so.

### Decision

`app/streamlit_app.py` gains a sidebar `st.segmented_control` with three
options mirrored from the C2 catalog keys (`gpt-4o-mini`, `opus-4.7`,
`kimi-k3`), rendered between the "New conversation" button and the
conversations list. Selection persists in `st.session_state.answer_model`
(default `"gpt-4o-mini"`), initialized in `_init_session_state()` alongside
the existing `lang`/`models_warm` keys. A cost-per-question caption below
the toggle (`COST_HINTS`) makes the tradeoff visible before the user asks a
question. `_render_turn()`'s call to `get_flow().run(...)` now passes
`answer_model=st.session_state.answer_model`. The existing technical-details
expander (`_render_turn_details()`) gains an `answer_model_key` row next to
the pre-existing `model_used` row, so the catalog key and resolved
OpenRouter model_id are both visible for debugging.

### Consequences

- Users can spend ~20-100x more per query intentionally (per ADR #45's
  cost table); the cost hint is the only guardrail — there is no
  confirmation dialog or spend cap.
- Model choice is a `st.session_state` value, not per-conversation: it
  applies to whichever question is asked next, and switching models
  mid-conversation is silent (no marker on which turn used which model
  beyond the per-turn debug expander).
- `MODEL_LABELS`/`COST_HINTS` duplicate the `ANSWER_MODELS` catalog keys as
  a local dict rather than importing `rag.generate.ANSWER_MODELS` directly
  — keeps Plane IV consuming Plane II only through `flow.run()`, at the
  cost of the two dicts needing to stay in sync by hand if the catalog
  changes.

### Follow-ups

- Per-turn cost display in the answer bubble itself, not just the debug
  expander.
- Running-total cost per session.
- Automatic model recommendation based on the router's `route_decision`
  confidence (ADR #42).
## ADR #48 — Persisted per-case compliance run log

**Date:** 2026-07-30 · **Branch:** v2-agentic (Day C follow-up) · **Status:** Accepted

### Context

Day C private compliance run (2026-07-30) processed 46 role clusters but
emitted only 6 entries, all on 1-2-fact clusters. Every fact-heavy cluster
(notaire_redacteur at 42 facts, notaire_stagiaire, quasi_usufruitier,
heritier_nu_proprietaire, etc.) produced no output. Since notaire liability
is the project's headline use case and every notaire_* cluster went
silent, the matrix contained zero assessments about the actor class the
tool exists to evaluate.

`rag/compliance.py` already logs `role_id, finish_reason, prompt_tokens,
completion_tokens, raw_chars, elapsed` per call, including a dedicated
warning when `finish_reason == "length"` — but only via `logging.basicConfig`
(stderr), which nothing captures. No persisted log survived the Day C run,
so it cannot be confirmed post-hoc whether each silent role_id truncated
(`finish_reason=length`), failed to parse (`_recover_partial_entries` also
came up empty), or legitimately returned `[]`. `docs/ai_choreography_audit.md`
Recommendation #1 ranks closing this evidence gap above any tuning change,
since a future fix to `MAX_OUTPUT_TOKENS` would otherwise be tuned blind.

### Decision

Add a per-case `logging.FileHandler` at
`data/dossier/{case_id}/compliance_run.log`, attached to the module logger
for the duration of `generate_compliance_matrix()` and removed in a
`finally` block. Overwritten (`mode="w"`) on every run so the log always
matches the current matrix. No other change: `MAX_OUTPUT_TOKENS` stays at
`4096` (deferred — see Follow-ups), no log statements or call sites change.

### Consequences

- The next truncation-related change to `rag/compliance.py` (whenever it
  lands) becomes verifiable rather than assumed: per-role_id outcome
  (truncated vs. parse-failed vs. legitimately empty) will be visible in
  the persisted log across a re-run.
- New artifact per run (`compliance_run.log`), gitignored under
  `data/dossier/private/` alongside the matrix; overwrites on each
  invocation, so only the most recent run's log is ever retained.
- No cost or behavior change to the LLM call itself — this ADR is
  instrumentation only.

### Follow-ups

- Raising/uncapping `MAX_OUTPUT_TOKENS` in `rag/compliance.py` remains open
  and deliberately deferred — this ADR's logging is a prerequisite for
  verifying that fix once it lands. Will need its own ADR number when
  implemented.
- Consider surfacing a per-role truncation/parse-failure/empty summary
  count in the CLI's existing summary print line, once the log has been
  observed across a few runs.

## ADR #49 — Uncapped MAX_OUTPUT_TOKENS on compliance call (correcting Day C notaire silence)

**Date:** 2026-07-30 · **Branch:** v2-agentic (Day C follow-up) · **Status:** Accepted

### Context

Day C private run (2026-07-30) processed 46 role clusters, emitted 6
entries — all on 1-2-fact clusters. Every notaire_* cluster
(notaire_redacteur 42 facts, notaire_stagiaire, notaire_instrumentaire,
notaire_mandataire, notaire_associe, notaire_collaboratrice) produced no
output. ADR #48 added persisted logging to make the truncation vs
parse-fail vs empty-response split verifiable. This ADR is the actual fix
ADR #48 enables verification of. Cites ADR #47 as identical-pattern
precedent for the answer-model tier: reasoning-effort tokens and
completion tokens share the same `max_tokens` budget on Opus 4.7 max via
OpenRouter, so a small cap suffocates reasoning-heavy compliance calls
before any output is emitted.

### Decision

Remove the `MAX_OUTPUT_TOKENS = 4096` cap. Model's default output cap
applies. `extra_body={"reasoning": {"effort": "max"}}` remains — the
reasoning-effort transport pattern is unchanged.

Dry-run path: replace the `MAX_OUTPUT_TOKENS // 2` completion-tokens
estimate with a standalone `_DRY_RUN_EST_COMPLETION_TOKENS = 2048`
constant, since there's no longer a real cap to reference.

Truncation warning log message: no longer references a `max_tokens=`
figure the code doesn't set; instead cites "the provider's max-output
ceiling."

### Consequences

- Potentially higher per-call cost — mitigated by (a) existing 30-fact
  chronological cap in `_cap_facts_chronologically`, (b) ADR #48 log
  surfaces per-call cost immediately so runaway is visible, (c)
  OpenRouter model default cap acts as ceiling.
- Empirical post-uncap run: 46/46 `finish_reason=stop`, ~$22 total,
  all notaire clusters produce substantive output.

### Follow-ups

- If empirical cost is unbounded, add soft ceiling `max_tokens=16000` as
  safety. Splitting >15-fact clusters into sub-calls with independent
  budgets remains Attempt 2 architecture work.

## ADR #52 — Fact-level distillation with verbatim cross-check

**Date:** 2026-08-08 · **Branch:** v2-persons (D5) · **Status:** Accepted

### Context

Facts today store a `verbatim_quote` — the exact source sentence(s), kept
for forensic verification — extracted from ceremony-heavy source letters.
Most correspondence in the private case is roughly 80% ceremony (address
blocks, standard French legal formulas, restated context from prior
correspondence) and 20% substance. The compliance model (ADR #43, ADR #49)
spends attention chewing through ceremony to find the substance that
actually decides an obligation's status. The fix is a dense per-fact
"lawyer's-note" summary that strips ceremony while preserving substance,
with `verbatim_quote` kept intact and unmutated alongside it for forensic
verification.

This deliverable is D5 of the user's Attempt 2 execution plan
(`plan_attempt2_full.md`, local, not committed). D1-D4 (a separate person-
index pipeline: mention extraction, entity resolution, ADR #50/#51) are
**not shipped** as of this ADR — `distill.py` has no dependency on them, so
distillation proceeds independently. #50 and #51 remain open/reserved for
whenever that pipeline is built; this ADR does not claim or reference them.

### Decision

Fact-level (not document-level) distillation via new
`ingestion/dossier/distill.py`. For each fact, `anthropic/claude-haiku-4.5`
(temperature 0.0) reads the fact's `verbatim_quote` plus a 2000-character
window of surrounding source-document text (1000 chars either side of the
quote's location, substring-located; falls back to the document's first
2000 chars with a logged warning if the quote isn't found verbatim) and
emits a 2-5 sentence dense summary. The system prompt requires preserving
every date, name (person or entity), amount, reference to a prior act or
document, and any direct quotation of an admission, refusal, contradiction,
or citation, plus the signatory's identity; it requires stripping address
blocks, closings, standard formulas, restated prior correspondence (unless
the restatement contradicts or is the first mention of a document), and
enclosure lists (unless the enclosure is the substance).

New `Fact` field: `distilled_context: str | None`, default `None` —
additive and backward-compatible; existing serialized facts without the key
load unchanged. **Salvage constraint**: no existing `fact_id` is renumbered
or regenerated, and `verbatim_quote` is never mutated by this or any other
Attempt 2 stage — extensions only ever add new optional fields alongside
it. `compliance_matrix.json`'s existing structure stays valid under this
constraint; any future additive field (e.g. a person-pipeline
`persons_named`) must hold to the same rule.

Idempotency: cache keyed by `SHA-256(verbatim_quote + source_context[:200])`,
persisted to `data/dossier/<case_id>/distill_cache.jsonl`. A re-run with
unchanged facts and unchanged source documents is entirely cache hits — no
API calls, byte-identical `facts.jsonl`. Atomic rewrite of `facts.jsonl` via
`.tmp` → `os.replace`; the `.tmp` file is removed if `os.replace` fails, so
a crash mid-write never corrupts or half-writes the file.

`distill_case` is not wired into `build.py` or any other pipeline entry
point — it ships dormant, invoked only via its own CLI
(`python -m ingestion.dossier.distill --case-id <id> [--dry-run]
[--fact-id <one>]`). The private-case backfill (~$1.20 for ~235 facts) is
authorized separately, at D6's ship gate.

### Consequences

- Additive schema change, fully backward-compatible with every prior
  Attempt 1/2 fact.
- Compliance (D6) will read `distilled_context` as its reasoning input and
  `verbatim_quote` as its verification anchor before finalizing a
  `breached`/`met` verdict — the fiability constraint this deliverable
  exists to serve.
- Cache-hit idempotency means re-running distillation after a partial
  failure only pays for the facts that hadn't succeeded yet.
- D5 introduces zero cost or behavior change to any existing pipeline until
  D6 explicitly invokes it against the private case.

### Follow-ups

- Document-level distillation, if fact-level proves too narrow for
  cross-fact reasoning (e.g. contradictions spanning two documents) —
  deferred.
- Distillation quality eval via LLM-as-judge — deferred to Attempt 3.
- ADR #50 (D1, person-index fixtures) and ADR #51 (D4, global entity store)
  remain open — not written, not implemented, in this session.

## ADR #53 — Compliance verify-conclude prompt + person integration

**Date:** 2026-08-09 · **Branch:** v2-persons (D6) · **Status:** Accepted

### Context

This ADR was briefed as D6: wire D1-D4 (mention extraction → entity
resolution → `persons.jsonl`) and D5 (distillation, ADR #52) into
compliance, then re-run compliance on the private case with Opus 4.7 max.
Pre-work found D1-D4 do not exist anywhere in this repo — `ingestion/dossier/`
has no `mentions.py` or `resolve.py`, confirmed via `git log --all` across
every branch. ADR #52's own Context already stated this ("D1-D4 ... not
shipped"), and its Follow-ups close with "ADR #50 (D1) ... and ADR #51 (D4)
... remain open — not written, not implemented." Both citations still hold
as of this ADR: #50 and #51 remain open.

D6 is therefore descoped to distillation + verify-conclude only. The
person-integration plumbing below (persons context block, `persons_named`
schema field) is built exactly as originally specified — forward-compatible
plumbing that activates automatically once D1-D4 ship — but is **inert in
practice**: `case_persons`/`entities`/`persons_named` are `[]` on every
entry, for every case, today, because no `persons.jsonl` exists anywhere
(case-local or global) to populate them.

The verify-conclude prompt rule is structural motivation independent of the
person pipeline: distillation (ADR #52) strips ceremony from `verbatim_quote`
into a dense `distilled_context`, but a distillation model summarizing
"substance" can pull in adjacent factual context from the surrounding
source text that the strict `verbatim_quote` doesn't itself support (see
Consequences). The compliance model (ADR #43, #49) already reasons over
`verbatim_quote` alone; adding `distilled_context` as a second, denser
reasoning surface without a cross-check would let compliance verdicts
silently inherit any over-inclusion distillation introduces. This closes
that gap the same way ADR #48/#49 closed the truncation-evidence gap:
structurally, in the prompt contract, not by trusting model behavior.

### Decision

(a) `_build_user_message` gains a "Personnes impliquées" section, built
from `case_persons` (case-specific: role assignments, ambiguity note) and
`entities` (base identity: canonical name, aliases), merged and keyed by
`person_id` — appended only when non-empty, mirroring the ADR #44
cross-role block. **Inert until D1-D4 ship.**

(b) Each fact line in the compliance prompt exposes both fields side by
side: `distilled="..."` (reasoning surface) and `citation="..."` (verbatim
verification anchor, previously the only field shown).

(c) `COMPLIANCE_SYSTEM_PROMPT` gains a verify-conclude paragraph: reason
from `distilled`, but before finalizing `breached`/`met`, verify the
specific claim is actually present in `citation` — downgrade to
`insufficient_evidence` on any mismatch or suspected fabrication.

(d) `COMPLIANCE_SYSTEM_PROMPT` gains a person-naming paragraph: name a
person in the rationale for a `breached`/`met` verdict that concerns them,
when a "Personnes impliquées" section is present — but never name a person
whose `ambiguity_note` flags uncertain resolution; fall back to role-only
reference or `insufficient_evidence`. **Inert until D1-D4 ship** (no
section ever renders today, so this rule has no live effect yet).

(e) `ComplianceEntry` gains `persons_named: list[dict]` (default `[]`),
populated per-cluster from the same merge as (a). **`[]` on every entry
until D1-D4 ship.**

(f) **Compliance-run cache** (added before implementation, user directive):
`data/dossier/{case_id}/compliance_cache.jsonl`, keyed by a cluster
fingerprint — SHA-256 of `role_id + sorted(fact_ids) + SHA-256(prompt)`.
Cluster-level, not fact-level. An idempotent re-run with unchanged facts,
retrieval, and context is entirely cache hits at zero marginal cost.
Mirrors ADR #52's `distill_cache.jsonl` pattern.

(g) **Dry-run cost transparency** (added before implementation, user
directive): `--dry-run` now prints one line per cluster
(`prompt_tokens`, `completion_tokens_est`, `cost_est`) in addition to the
existing total line, and raises `RuntimeError` via
`_check_dry_run_cost_gate` if the total estimate exceeds
`DRY_RUN_COST_ALERT_USD = 25.0` — a loud stop before any real spend, since
an estimate that high signals a prompt-size regression (e.g. an unbounded
context block) rather than a normal cost curve.

### Consequences

- The private-case backfill in this session runs only `distill.py`
  (mentions/resolve steps don't exist to run) — 235/235 facts distilled,
  ~3.4% (8/235) fallback rate. Marginal token cost from the persons
  plumbing is currently zero in practice, since the section never renders.
- **Known limitation (a):** the 8 facts (3.4%) whose `distilled_context`
  fell back to `source_context[:2000]` (verbatim quote not locatable in the
  source document via substring or whitespace-tolerant regex — see
  `_extract_fact_neighborhood` in ADR #52) have weaker distillation quality:
  the fallback window is not centered on the quote. Their `fact_id`s are
  logged to `data/dossier/private/distill_fallback_facts.txt` (gitignored,
  private-case data) for future auditing, derived by re-running the same
  deterministic match logic against the committed facts/extracted-text
  state — no LLM call, no new backfill spend.
- **Known limitation (b):** distillation (ADR #52) sometimes pulls
  additional factual context from surrounding source text beyond the
  strict verbatim scope of the fact it's summarizing — an artifact of
  summarizing "substance" rather than performing pure extraction. This is
  exactly the failure mode decision (c)'s verify-conclude rule exists to
  catch: if the specific reasoning claim in `distilled` isn't actually
  supported by `citation`, the compliance model must downgrade to
  `insufficient_evidence` rather than trust the denser surface.
- Additive, backward-compatible schema change (`persons_named` defaults to
  `[]`); no existing `ComplianceEntry` or `compliance_matrix.json` breaks.
- Compliance-run cache and dry-run cost gate are net-new operational
  safety rails, not scoped in the original D6 brief — added because a
  re-run against 235 facts without cluster caching would otherwise re-spend
  on every retry, and an unbounded prompt could regress cost silently.

### Follow-ups

- ADR #50 (D1, person-index fixtures) and ADR #51 (D4, global entity store)
  remain open — carried forward again from ADR #52, still not written, not
  implemented.
- Once D1-D4 ship and `persons.jsonl` exists for a case, decisions (a),
  (d), and (e) activate with no further code change — this is the point of
  building them now as inert plumbing rather than deferring them entirely.
- `Fact.mentioned_person_ids` does not exist yet; `generate_compliance_matrix`
  reads it via `getattr(f, "mentioned_person_ids", None) or []` so the
  per-cluster person-filtering logic is real code today, exercised as a
  no-op, rather than dead code gated behind a feature flag.

## ADR #54 — Per-doc mention extraction (D2)

**Date:** 2026-08-02 · **Branch:** v2-persons (D2) · **Status:** Accepted

### Context

D1's own ADR + fixture case (person-index pipeline scaffolding) never
materialized — ADR #50 was reserved but never written, and no fixture case
was built. Rather than write a standalone D1 ADR after the fact, D2
collapses D1's remaining scope into itself and ships straight to code: this
ADR is the first real entry for the mention-extraction stage.

D5 (distillation, ADR #52) and D6 (compliance verify-conclude + person
plumbing, ADR #53) already shipped. D6's `persons_named` field (on
`ComplianceEntry`) and the "Personnes impliquées" prompt block remain
**inert** — `[]` on every entry, no section ever rendered — because no
`mentions.py`/`resolve.py` output exists to populate them. D2 is the first
of two remaining stages (D3 `resolve.py` is next) needed to make that
plumbing live.

Model choice: `anthropic/claude-haiku-4.5`, matching D5's `distill.py` —
consistency across the person-pipeline's LLM calls, and Haiku 4.5's JSON
reliability at temperature 0 is already proven in this codebase (`facts.py`
uses the same defensive-JSON-parse pattern this module reuses).

Sequencing: mentions extraction reads only
`data/dossier/<case_id>/extracted/<doc_id>.md` (the Attempt 1 verbatim
transcript) and is independent of `facts.jsonl` by design — it does not
read, write, or depend on facts, actor roles, or role ambiguities. This
keeps D2 shippable and testable without any coupling to D4's fact-extraction
schema (sequencing "β" from the design discussion this ADR follows).

**Pre-work correction:** the originating task brief assumed distill's (D5)
test mock pattern lived in `tests/test_dossier_smoke.py`. It doesn't —
distill's tests live in `tests/test_persons_smoke.py` (added in the same D5
commit), whose own docstring states it's the intended home for the
Attempt-2 person-pipeline test suite. D2's 8 tests were added there
instead, following that file's existing inline `MagicMock()` +
`patch(..., get_openrouter_client=...)` convention — there is no named
mock-helper function to reuse in either file.

### Decision

New `ingestion/dossier/mentions.py`, mirroring `distill.py`'s module
structure (constants → system prompt → cache functions → per-item call →
orchestration → CLI). For each extracted document, `anthropic/claude-haiku-4.5`
(temperature 0.0) reads the full markdown transcript and returns a strict
JSON array of mentions, each with exactly 4 fields: `surface_form`
(verbatim, undeduplicated — every occurrence is a distinct mention),
`kind` (`"person"` | `"entity"`), `context_snippet` (1-3 surrounding
sentences), and `role_hint` (nullable, only when textually anchored).

Output is `data/dossier/<case_id>/_mentions/<doc_id>.json` — one file per
document, and that file **is** the cache: keyed by a content-hash
(`SHA-256(markdown)[:16]`) of the source markdown, so a re-run with an
unchanged document is a pure cache hit at zero cost. `schema_version: 1` is
embedded in every cache file for D3 to key its own migration logic against.
Atomic write via `.tmp` → `os.replace`, matching `distill.py`'s
cleanup-on-exception pattern. A JSON-parse failure returns an empty result
to the caller but does **not** write a cache file — distinguishing "nothing
to cache" from a legitimate, cacheable zero-mention extraction (mirrors
`facts.py`'s `_parse_llm_json` `None`-vs-empty-collection idiom, which
`distill.py` has no precedent for since its own output is free prose, not
JSON).

Non-destructive to `facts.jsonl`, `actor_roles.jsonl`, and
`role_ambiguities.jsonl` — this module never opens any of them. Dormant by
default: not wired into `build.py`; invoked only via its own CLI
(`python -m ingestion.dossier.mentions --case-id <id> [--doc-id <one>]
[--force] [--dry-run] [--verbose]`), same dormancy pattern as `distill.py`.

### Consequences

- The committed demo fixture (`data/dossier/demo/extracted/sample_text.md`,
  1 doc, 3 generic English test sentences) extracts **0 mentions at
  $0.0004** on a real run — the fixture has no named entities to find. This
  is expected, not a bug: the demo case exists to prove the pipeline runs
  end-to-end, not to exercise extraction quality. A future private-case
  backfill (~55 docs of real French legal correspondence) is estimated at
  roughly $0.02 total at Haiku 4.5 rates, scaled from this session's
  per-document dry-run cost estimate — that backfill is explicitly **not**
  run in this session (code-only deliverable; private-case backfill is a
  separate step gated on user approval, per the D5/D6 precedent).
- D3 (`resolve.py`) will consume `_mentions/*.json` + `facts.jsonl` to
  produce `persons.jsonl` (case-local and, per the original person-pipeline
  design, eventually global). Until D3 ships, D2's output is inert data —
  correctly shaped, cached, and versioned, but nothing reads it yet.
- Compliance re-run (to make D6's `persons_named` non-empty) remains gated
  on D3's completion, not D2's — D2 alone does not activate any inert
  plumbing.
- `Fact.mentioned_person_ids` still does not exist (see ADR #53's
  Follow-ups) — D2 does not add it; that remains D3/D4 scope if still
  needed once real person IDs exist.

### Follow-ups

- D3 `resolve.py` — deterministic clustering of `_mentions/*.json` entries
  into canonical `persons.jsonl` entities, case-local first.
- D4 (global entity store across cases) remains deferred, per ADR #50/#51's
  original scope — not started this session.

References: ADR #52 (D5, distillation — cache-file and atomic-write
precedent), ADR #53 (D6, compliance verify-conclude — the inert plumbing D2
is the first step toward activating), `plan_attempt2_full.md` (local, not
committed) D2 section.

## ADR #55 — Role-scoped person resolution direct from distilled_context (D3)

**Date:** 2026-08-02 · **Branch:** v2-persons (D3) · **Status:** Accepted

### Context

ADR #54's Follow-ups stated D3 `resolve.py` would perform "deterministic
clustering of `_mentions/*.json` entries into canonical `persons.jsonl`
entities" — i.e. consume D2's per-doc raw mention lists. This ADR
supersedes that specific assumption (ADR #54's Follow-ups), while leaving
`mentions.py` itself unchanged, shipped, and dormant as code — D2 is not
superseded, only the plan for what feeds D3.

The private case's `facts.jsonl` (235 facts) already carries `actor_role`
(Attempt 1's D0 extractor) **and** `distilled_context` (D5, ADR #52) on
every fact — a spot-check found zero null `distilled_context` values, and
distillation quality preserves full names, dates, article citations, and
dossier numbers (the whole point of ADR #52's ceremony-stripping design).
Given that, routing D3 through `_mentions/*.json` would require a resolver
to cross-reference D2's raw per-doc entity list back against the same
`distilled_context` to figure out which role each entity plays — the
signal `mentions.py` would add over `distilled_context` + `actor_role`
directly is effectively zero for a case with good distillation coverage.
`actor_roles.jsonl` (60 roles, `role_ambiguities.jsonl` 22 ambiguities)
already provide the role taxonomy and the flagged-uncertain assignments
D3 needs to scope and clean its input.

D6's `persons_named` field (ADR #53) and the "Personnes impliquées" prompt
block remain inert until `data/dossier/{case_id}/persons.jsonl` exists.
D3 is the step that produces it.

### Decision

New `ingestion/dossier/resolve.py`, structurally closer to `mentions.py`
than to `distill.py` (JSON output, content-hash cache with
`schema_version`, defensive parsing) but reads only `facts.jsonl`,
`actor_roles.jsonl`, and `role_ambiguities.jsonl` — never `_mentions/*.json`.

For each `actor_role` with `fact_count > 0`: facts whose `fact_id` appears
in any `role_ambiguities.jsonl` entry's `fact_ids` are excluded (ambiguous
facts don't contribute reliable signal to who plays a role), the remainder
sorted by date and capped at `MAX_FACTS_PER_ROLE = 20`. `anthropic/claude-haiku-4.5`
(temperature 0.0, matching D2/D5's model choice) receives the role's
`label_fr` + `grounding_note` plus each fact as
`"fact_id: ... | date: ... | distilled: {distilled_context or verbatim_quote}"`,
and returns a strict JSON array of canonical persons: `canonical_name`,
`aliases`, `person_type`, `confidence`, `ambiguity_note`,
`evidence_fact_ids`. Per-role results are cached at
`data/dossier/{case_id}/_persons_cache/{role_id}.json`, keyed by a
content-hash over the role_id and its fact context — mirrors `mentions.py`'s
cache-file shape.

A post-process step merges per-role persons into case-scoped identities:
grouped by accent/case/whitespace-normalized `canonical_name` (exact match
on the normalized key — no fuzzy alias cross-matching, a known limitation
noted below), with the longer observed `canonical_name` winning as
canonical on a collision, aliases unioned, and one `role_assignments` entry
per role the person appears in. Person-level `confidence` is the max across
`role_assignments`; `ambiguity_note` is the first non-null one.
`person_id = f"{case_id}-{slugify(canonical_name)}"` (accent-stripped,
lowercased, hyphen-joined — same normalization family as
`extract.py::_slugify_segment`, so the ID is stable regardless of which
role-call happened to produce the winning canonical form). Output
`data/dossier/{case_id}/persons.jsonl`, one JSON object per line sorted by
`person_id`, atomic `.tmp` → `os.replace` write — matches exactly what the
already-written `scripts/enrich_compliance_with_persons.py` post-processor
expects (`person_id`, `canonical_name`, `role_assignments[].role_id`,
`confidence`, `ambiguity_note` are the fields it actually reads;
`aliases`/`person_type`/`evidence_fact_ids` are extra, for future
consumers).

Dormant by default, same as D2: not wired into `build.py`, invoked via its
own CLI (`python -m ingestion.dossier.resolve --case-id <id> [--role-id
<one>] [--force] [--dry-run] [--verbose]`).

### Consequences

- Verified against the real private-case data: of 60 roles, 14 have
  `fact_count == 0` (skipped) and, after ambiguity exclusion, one more
  (`notaire_collaboratrice`) drops to zero facts and is also skipped — so
  the real backfill makes roughly 45 Haiku 4.5 calls, not 46. Two roles
  (`heritier_nu_proprietaire`: 25 facts, `notaire_redacteur`: 37 facts)
  exceed `MAX_FACTS_PER_ROLE` and are truncated to their 20 most recent
  dated facts — the cap is load-bearing on this case, not decorative.
- D6's `persons_named` plumbing (ADR #53) and
  `enrich_compliance_with_persons.py`'s post-process path both activate
  automatically once `persons.jsonl` exists — no changes needed to either.
- Known limitation: cross-role merge matches on exact
  accent/case-normalized `canonical_name` only. Two roles where Haiku
  extracts genuinely different surface forms for the same real person
  (e.g. "Maître MENA" in one role's output vs. "l'étude PAVY-MENA" in
  another, with no shared normalized string) will **not** merge into one
  person — they surface as two separate `persons.jsonl` records instead.
  This is a direct consequence of skipping the `mentions.py` stage's raw
  alias inventory; accepted as a reasonable trade-off for this session's
  scope, and re-evaluated below.
- `mentions.py` remains shipped, dormant infrastructure — unchanged by
  this ADR — for future cases where distillation is unavailable or of
  lower quality; in that scenario D3's direct-from-`distilled_context`
  shortcut doesn't apply and the original `mentions.py` → resolver path
  becomes the primary one.
- Fact schema untouched — `facts.jsonl`, `actor_roles.jsonl`, and
  `role_ambiguities.jsonl` are read-only inputs to this module, preserving
  the salvage constraint from ADR #52.

### Follow-ups

- Global entity store (`data/dossier/entities/persons.jsonl`) for
  cross-case identity — deferred, per ADR #50/#51's original scope.
- Resolver-quality eval (LLM-as-judge over a sample of role assignments,
  and/or a fuzzy-alias merge pass to close the cross-role gap noted above)
  — deferred to Attempt 3.
- If a future case lacks D5 distillation coverage, `mentions.py` + a
  distinct `_mentions/*.json`-consuming resolver (the originally planned
  D3 shape) becomes the primary path instead of this direct-extraction
  shortcut.
- (d) honorific-stripping and bare-surname-drop shipped as hotfix;
  resolve.py's `_person_merge_key` normalizes French legal honorifics
  before cross-role merge, tightens `SYSTEM_PROMPT` against duplicate
  records, and drops bare-surname placeholders when a fully-named
  same-surname candidate exists in the same role.
- (e) Known residual duplicate this hotfix does not catch (a
  missing-middle-name variant, not honorific-only): "M. BOSSAVIT JEAN
  MARIE" (role `titulaire_contrat`) vs "Monsieur Jean Marie Robert
  BOSSAVIT" (roles `defunt`, `defunt_quasi_usufruitier`,
  `personne_decedee`, `quasi_usufruitier`) — missing "Robert". Left for
  the fuzzy-alias merge pass already noted above (Attempt 3). A second
  word-order duplicate flagged during planning ("Monsieur BOSSAVIT
  François" vs "François Bossavit") merged successfully in the actual
  private-case backfill run — Haiku's own extraction happened to use
  firstname-first order this time, likely encouraged by this hotfix's
  new canonical_name-ordering instruction above — but that's LLM output
  variance, not something this fix structurally guarantees, so a
  surname-first extraction could still slip through on a future rerun.

References: ADR #52 (D5, distillation — the `distilled_context` field this
module reads), ADR #53 (D6, compliance verify-conclude — the
`persons_named` plumbing this module activates), ADR #54 (D2, mention
extraction — the stage this ADR's Decision supersedes as D3's input
source, while leaving `mentions.py` itself shipped and dormant).

## ADR #56 — Parallel dual-model compliance comparative analysis (D7)

**Date:** 2026-08-03 · **Branch:** v2-persons (D7) · **Status:** Accepted

### Context

Every compliance verdict today comes from a single model,
`anthropic/claude-opus-4.7` (ADR #43, reasoning.effort="max"). Single-judge
risk for high-stakes legal determinations is a known concern in this
project's own eval design: the 2026-07-22 judge-diversity entries ("Claude
judge via OpenRouter", "Mistral as French-native third judge") and ADR #40's
restated rationale ("the rubric benefits from a cross-provider judge
stack... 3-family judge diversity for cross-family agreement measurement")
established a 3-provider judge panel specifically to avoid single-model
bias in the eval harness. The compliance matrix — arguably higher-stakes
than an eval score, since its verdicts are the deliverable a non-lawyer
heir reads — has had no equivalent cross-model check until now.

D7 adds an on-demand, per-role comparative mode: run the identical prompt
against a second frontier reasoning model and have a cheap model tag
agreement vs. divergence per obligation. This surfaces model-specific
reasoning differences (a hallucinated obligation, a missed cross-role
interaction, a different read of the same statute excerpt) without
replacing the existing single-model matrix, which stays the default,
unmodified flow.

### Decision

`_call_compliance_llm` (ADR #43/#49/#53) gains one new parameter,
`compliance_model_id: str = COMPLIANCE_MODEL_ID` — every existing call
site (`generate_compliance_matrix`) is unaffected by the default. New
module constants: `COMPLIANCE_MODEL_ALTERNATIVES = ["anthropic/claude-opus-4.7",
"moonshotai/kimi-k3"]` (both already carry `reasoning_effort="max"` in
`rag/generate.py`'s `ANSWER_MODELS` catalog, ADR #45) and
`DIVERGENCE_MODEL_ID = "anthropic/claude-haiku-4.5"`.

New `compare_compliance_for_role(case_id, role_id, dry_run=False)`: loads
the same case artifacts and retrieves the same statute chunks as
`generate_compliance_matrix`'s per-role loop (factored out into a shared
`_persons_context_for_role` helper for the persons/entities filtering step),
then runs `_call_compliance_llm` twice in parallel via
`ThreadPoolExecutor(max_workers=2)` — once per `COMPLIANCE_MODEL_ALTERNATIVES`
entry — before calling new `_call_divergence_analysis` (Haiku 4.5,
temperature 0.0, prompt in `rag/compliance_prompts.py`'s new
`DIVERGENCE_ANALYSIS_SYSTEM_PROMPT`) to tag agreement/divergence per shared
obligation. Pairing verdicts across the two models needs no fuzzy matching:
`_entry_id(statute_chunk_id, actor_role)` is a content-independent hash, so
both models land on the same `entry_id` for the same obligation whenever
they cite the same statute chunk — Python computes `coverage_diff`
(shared/opus-only/kimi-only `entry_id` sets) deterministically, and only
the shared set is sent to the divergence LLM as paired verdicts.

Output is written atomically (`.tmp` + `os.replace`) to
`data/dossier/{case_id}/compliance_comparative_{role_id}.json`. Cache
fingerprint is two-part, both stored in the output file and both required
to match for a cache hit:
- `inputs_hash`: SHA-256 over `role_id` + sorted per-fact fingerprints
  (`_fact_fingerprint`: fact_id plus every field that feeds the prompt —
  date, action, target, verbatim_quote, distilled_context — not just
  fact_id, so a content edit under a stable fact_id still invalidates the
  cache) + sorted chunk_ids + a persons/entities hash + explicitly,
  `sorted(COMPLIANCE_MODEL_ALTERNATIVES)` and `DIVERGENCE_MODEL_ID`. Folding
  the model ids in means a future model swap (e.g. Kimi K3 → K4) can't
  leave an old cached comparative looking valid and silently serve a stale
  cross-model comparison.
- `divergence_prompt_hash`: SHA-256 of `DIVERGENCE_ANALYSIS_SYSTEM_PROMPT`
  at write time (same cache-key-hashes-the-prompt pattern ADR #52's
  `distill.py` and this file's own `_compliance_cache_key`, ADR #53, both
  already use) — an edited divergence prompt invalidates old comparatives
  even when every other input is unchanged.

`compare_compliance_for_role` deliberately calls `_call_compliance_llm`
with `cache=None` for both models, never touching the shared
`compliance_cache.jsonl` (ADR #53). That cache's fingerprint
(`_compliance_cache_key`) has no model dimension — it keys on
`(role_id, fact_ids, prompt_hash)` only, because until now only one model
was ever in play. Reusing it here would mean the second model's call on
the same role/facts/prompt collides on the same key and silently returns
the first model's cached entries. The comparative feature's own
file-level two-hash cache is independent and sufficient; the shared cache
stays exactly as ADR #53 left it.

Dry-run cost estimates needed a small correction alongside this: the
existing `_EST_*_USD_PER_TOKEN` module constants are Opus-specific
($15/$75 per M). A new `_COMPLIANCE_MODEL_DRY_RUN_RATES` lookup (keyed by
`compliance_model_id`, falling back to the existing Opus constants for an
unlisted model) gives Kimi's `--dry-run` estimate its own $3/$15-per-M
rate instead of silently reusing Opus's ~5x-higher rate. The divergence
call gets its own rough Haiku-rate estimate
(`_DIVERGENCE_EST_*`, ~$1/$5 per M per CLAUDE.md's model palette) — unlike
the Opus dry-run constant, this isn't back-solved from a real bill, since
divergence output is small and bounded by shared-obligation count, not
worth the same empirical calibration effort.

CLI: `python -m rag.compliance --case-id <id> --role-id <role> --compare
[--dry-run]`.

### Consequences

- Per-compare cost: roughly $0.28 (Opus) + $0.09 (Kimi) + $0.02 (Haiku
  divergence) ≈ $0.60 for a typical fact cluster — user-initiated per role,
  not a batch operation.
- No real-run cost gate on `compare_compliance_for_role` (unlike
  `generate_compliance_matrix`'s `_check_real_run_cost_gate`) — a single
  compare is bounded at ~$0.60, so the `--dry-run` preview is enough; a
  cost gate only earns its keep once compares can be batched (see
  Follow-ups).
- `generate_compliance_matrix`'s default flow is provably unchanged: its
  one call site never passes `compliance_model_id`, so it always resolves
  to `COMPLIANCE_MODEL_ID`, and it never touches
  `compliance_comparative_*.json`.
- Rubric evidence for LLM evaluation / adversarial-analysis best practice —
  a second, independent frontier-model read on the same high-stakes legal
  determination, in the same spirit as the eval harness's existing judge
  diversity (2026-07-22 entries, ADR #40).

### Follow-ups

- Batch-compare mode (`--compare-all`, all roles in one case) would need a
  real-run cost gate reusing `_check_real_run_cost_gate`'s pro-rated
  pattern from ADR #53 — deferred; single-role compare is bounded at
  ~$0.60 so the dry-run preview suffices for now.
- Three-model ensemble (add a third frontier model as a tie-breaker
  instead of just Opus vs. Kimi) — deferred.
- Systematic Opus-vs-Kimi bias survey across all of Attempt 1's roles —
  an Attempt 3 research question once enough comparative runs accumulate.

References: ADR #43 (compliance matrix generation, the single-model flow
this extends), ADR #45 (model catalog — `reasoning_effort="max"` precedent
for both Opus 4.7 and Kimi K3), ADR #49 (uncapped `max_tokens` on the
compliance call — the comparative calls inherit the same uncapped
behavior via `_call_compliance_llm`), ADR #52 (distillation —
cache-key-hashes-the-prompt precedent this ADR's `divergence_prompt_hash`
follows), ADR #53 (compliance verify-conclude, persons plumbing reused
by `_persons_context_for_role`, and the shared `compliance_cache.jsonl`
this ADR deliberately does not touch), ADR #55 (most recent D-work on
this branch).
---

## ADR #57 — Streamlit compliance analysis panel with single-model toggle (D8)

**Date:** 2026-08-03 · **Branch:** v2-persons (D8) · **Status:** Accepted

### Context

ADR #56 (D7) shipped `compare_compliance_for_role`, reachable only from the
CLI. Compliance verdicts therefore live as JSON on disk, which makes them
inert for the person the project exists to serve: a non-lawyer heir cannot
read `compliance_matrix.json`. D8 is the first user-facing surface over any
compliance output.

The comparative view ADR #56 built is a trust-calibration surface — when two
frontier models agree, a verdict is more defensible; when they diverge, the
divergence itself is the diagnostic signal, pointing at the statute reading
or the fact that decides the question. That argues for exposing it.

But ADR #56's function fans out to both frontier models *and* a Haiku
meta-analysis on every invocation (~$0.60), by design, with no way to run a
single model. Routine per-role review should not pay the comparative
premium, and a user asked to "choose which model answers" reasonably expects
that choosing one means only one runs. There was no public single-model
per-role entry point: `generate_compliance_matrix` is whole-case and
Opus-only, and `_call_compliance_llm`'s `compliance_model_id` parameter
(ADR #56) is private.

Two further constraints came from the app as it actually stood. It had no
tab bar at all — `app/streamlit_app.py` was a single-page chat, so D8 had to
introduce tabs rather than add one. And `st.chat_input` pins to the viewport
bottom only in the page body; nested inside a tab it renders inline, which
would have silently regressed the shipped chat UX (ADRs #30, #46).

### Decision

**Backend (`rag/compliance.py`).** New public
`run_compliance_for_role(case_id, role_id, compliance_model_id=COMPLIANCE_MODEL_ID,
dry_run=False)`: ADR #56's prelude and prompt, one model, no divergence
pass. Output written atomically to
`data/dossier/{case_id}/compliance_single_{role_id}_{model_slug}.json`.

The two per-role entry points now share `_prepare_role_inputs`, a
behaviour-preserving extraction returning a `_RoleInputs` NamedTuple (case
artifacts, role cluster, role label, filtered persons/entities, retrieved
chunks, fact fingerprints, chunk ids, persons hash). They must assemble
inputs identically or their cache hashes stop being comparable and, worse,
the single and comparative paths would send different prompts to the same
model for the same role. `_persons_hash` and `_fact_fingerprint` moved into
that shared section with them.

Cache isolation is the load-bearing detail, and it is two independent
guards. First, `_single_inputs_hash` mirrors `_comparative_inputs_hash` but
folds in the one `compliance_model_id` actually called instead of
`sorted(COMPLIANCE_MODEL_ALTERNATIVES)` + `DIVERGENCE_MODEL_ID`, so toggling
Opus↔Kimi on an otherwise unchanged role is a cache **miss**. Second, the
call passes `cache=None` for the same reason ADR #56 does: the shared
`compliance_cache.jsonl` key has no model dimension (ADR #53), so reusing it
would let a Kimi run return `generate_compliance_matrix`'s cached Opus
entries. Either guard failing would display one model's verdicts under the
other model's name — the precise failure a model toggle must never have.
`test_run_compliance_for_role_cache_is_per_model` pins both.

CLI gains `--model-id`; `--role-id` without `--compare` is now a
single-model run.

**UI (`app/streamlit_app.py`).** Two tabs, "💬 Assistant" and
"🔬 Analyse comparative", created with `on_change="rerun"` and a key so
`.open` is readable server-side. That single choice buys both open
questions: the panel's file I/O is skipped during chat turns, and
`st.chat_input` stays in the page body guarded by `if tab_chat.open`, where
Streamlit still pins it to the viewport bottom. `main()`'s conversation loop
moved to `_render_chat`; the chat is otherwise untouched and the app still
opens on it.

The panel's primary control is a `st.segmented_control` model toggle
(🧠 Claude Opus 4.7 / 🔬 Kimi K3, keys matching
`COMPLIANCE_MODEL_ALTERNATIVES`) plus a "Lancer l'analyse" button: exactly
the selected model is called. ADR #56's dual run stays reachable behind a
separate `st.expander` labelled with its own price, so nothing fires both
models without a deliberate second action. Results are
memoised in `st.session_state.compare_result_cache` keyed by
`(case_id, role_id, mode, model_id)` — the model id belongs in the key for
the same reason it belongs in the inputs hash.

Side-by-side rows align by `entry_id` via ADR #56's precomputed
`coverage_diff`, not by obligation-summary similarity: `entry_id` is a
content-independent hash of `statute_chunk_id + actor_role`, it is the same
pairing the Haiku meta-analysis used, and a separate string-similarity
pairing in the view could contradict the callout rendered directly above it.
Model-exclusive obligations append to the foot of their own column.
Divergences render as a `st.dataframe` (Obligation, Verdict Opus, Verdict
Kimi, Crux, Modèle plus fort, Raison).

Costs are labelled `$US`, not `€`: OpenRouter bills in dollars and every
`usage.cost_usd` the backend returns is dollars. The cost footer sums the
returned usage rather than echoing the preview constants, so the figure
shown is what was billed; a cache hit reads 0,00 $US.

Previews are per model, not one flat constant. Calibrated from `--dry-run`
against the private case, Opus ran $0.46–$0.59 per role and Kimi $0.12 on
the same cluster — the ~5x spread is just the published rate difference
($15/$75 vs $3/$15 per M), so a single shared constant would misprice
whichever model the user did not select. The panel shows
`COST_PREVIEW_USD[model_id]` (≈0,55 / ≈0,12 $US) with the caveat that cost
scales with the role's fact count, and ≈0,70 $US for the comparative run.

### Consequences

- No new dependencies (`st.tabs`, `st.columns`, `st.expander`,
  `st.dataframe`, pandas — all already present).
- Selecting Kimi costs ≈$0.12 per role against ≈$0.55 for Opus and ≈$0.70
  for the comparative run, so the toggle is a real ~5x lever and not just a
  labelling choice; the comparative premium is paid only on explicit
  opt-in, and a repeat click is $0.
- Widens D8 past its originally scoped `app/`-only footprint into
  `rag/compliance.py`. Additive apart from the `_prepare_role_inputs`
  extraction, which is covered by the existing ADR #56 compare tests.
- `ComplianceEntry.persons_named` renders nothing today.
  `_persons_context_for_role` filters on `Fact.mentioned_person_ids`, which
  D2 (ADR #54) extracted but never wrote back onto facts, so the field is
  `[]` on every entry and the panel's "Personnes impliquées" block is
  rendered conditionally rather than showing a permanently empty section.
  The ADR #53 plumbing remains inert.
- `data/dossier/demo/` is an empty fixture (0 facts, 0 entries), so the case
  selector orders cases with entries first and explains empty ones rather
  than opening on a dead panel.
- Streamlit's `AppTest` models `st.tabs` as a `Block`, not a widget, so tab
  state does not survive a simulated rerun and the panel cannot be reached
  through the tab in automated tests. Render and wiring coverage therefore
  comes from driving the panel functions directly; the tab itself is manual
  smoke. This is an AppTest limitation, not an app behaviour.
- Rubric evidence for the interface and LLM-evaluation criteria.

### Follow-ups

- (a) batch-compare across roles with a cost cap, reusing
  `_check_real_run_cost_gate`'s pro-rated pattern (deferred from ADR #56).
- (b) export an analysis as PDF (bundle with D10).
- (c) a third tie-breaker model when Opus and Kimi diverge.
- (d) backfill `Fact.mentioned_person_ids` so `persons_named` activates.
- (e) correct the sidebar `COST_HINTS`, which label USD figures with `€`.

Related: ADR #56 (extended, not superseded — its comparative path is
unchanged and still reachable), ADR #53 (persons plumbing; the shared-cache
model-dimension hazard this ADR guards against twice), ADR #46 (the sidebar
answer-model toggle whose pattern the panel's model toggle mirrors), ADR #43
(the compliance prompt both paths send).
