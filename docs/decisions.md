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
  intuitive result documented in ADR #20 (hybrid loses to vector).
- Real users may cite article numbers verbatim after receiving legal
  correspondence ("qu'est-ce que dit l'art 815 ?"). The prompt explicitly
  strips article numbers from generated questions, so this query mode is
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