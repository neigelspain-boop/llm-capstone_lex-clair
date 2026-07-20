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

*Last updated: 2026-07-19 · Day 2 close*