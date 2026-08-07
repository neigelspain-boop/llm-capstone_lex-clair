<!-- WORKING FILE — delete when Phase 2 is complete; the durable record is ADR #70 + docs/investigator-spec.md -->

# Plane V build checklist

Working file. Tick items as they land. **Not** the documentation — that is
`docs/investigator-spec.md` (contracts) and ADR #70 (rationale). Deleting this
file is the final commit of Phase 2.

Approved 2026-08-07. Phase 1 target 11 Aug (deterministic, zero LLM calls).
Phase 2 target 12-14 Aug (local qwen3 adjudication layers).

---

## Invariants — violate none of these

| # | Invariant | Where enforced |
|---|---|---|
| 1 | `claim` text is identity. No counts, dates, amounts, fact_ids, day-deltas in a claim — they go in `evidence`, which is rewritten on every upsert. | `test_claims_carry_no_volatile_numbers` |
| 2 | A partial pass returns `complete=False` and **never** reconciles. Budget cutoff must stay distinguishable from "fixed". | `orchestrator._record` |
| 3 | `store.save_all` writes tmp + `os.replace`. The forked original truncates first; this harness is designed to be killed. | `test_save_all_is_atomic` |
| 4 | `confounders` / `calibration` / `tier_history` live **outside** `upsert_many`'s update set, so a re-check cannot erase adversarial work. | `test_confounders_survive_reupsert` |
| 5 | A `None` model verdict is "no verdict", never "no", and is never cached. | Phase 2, `ollama.py` callers |
| 6 | `gap` ≠ `unverifiable`. Absence is only a finding when the scope was actually covered. | `check.evaluate` |
| 7 | One externalisation gate function, one caller. `private` is never outbound-eligible. | `test_no_second_outbound_path` |
| 8 | Model rates are defined only in `ingestion/clients.py` (ADR #68). | `slimming` pass flags violations |
| 9 | Deterministic first. No LLM call may answer a question a predicate answers. | review |
| 10 | Never re-extract. `fact_id` is index-positional; re-extraction dangles 212 `evidence_fact_ids` edges. | `graph.py` is load-only |

---

## Phase 1 — deterministic engine

### Day 0 — 7 Aug · paper, zero code risk

- [x] `investigator/PLAN.md` (this file)
- [x] `docs/investigator-spec.md` — module contracts, catalog schema, worked entries, pass contracts, tier table, loop invariants
- [x] ADR #70 in `docs/decisions.md`
- [x] `CLAUDE.md` — five-plane heading, Plane V bullet, Key-docs pointer, Current-state entry
- [x] Commit: `docs: ADR #70 + Plane V investigator spec`

**Verified during Day 0, into the spec and ADR:**

- `cc-730-4` is a **single alinéa** (`LEGIARTI000006430899`, `VIGUEUR`, 281 chars)
  that *enables* release "dans la proportion indiquée à l'acte". No "al. 2", no
  unanimity. The unanimity limb is `cc-815-3` in fine (`LEGIARTI000006432378`).
- Absent from the 792-chunk corpus: `cc-1204`, `cc-1344`, `cc-494-12`,
  `cc-1231-1`. These need `chunk_id: null` + `legiarti_id`.
- `vitrine`: 235 facts, 28 persons, **no `coverage.jsonl`**, **0 sidecars**.
  `private`: 235 facts, 34 persons, 93 coverage rows, 55 sidecars.
  `demo`: 0 facts, **no `persons.jsonl` file at all**.
- Person→fact inversion on `vitrine`: 212 edges, 172/235 facts covered,
  **0 dangling**.

### Day 1 — 8 Aug · skeleton, inert

- [ ] `investigator/__init__.py` — charter docstring
- [ ] `investigator/config.py` — `CasePaths`, `PROMPT_VERSIONS`, budget ceilings, `ALLOW_OUTBOUND_CASES = {"vitrine", "demo"}`
- [ ] `investigator/schema.py` — `Obligation`, `PassResult`, `RunContext`, `Evaluation`, `TIER_ORDER`, `assign_tier()`, `externalisable_findings()`
- [ ] `investigator/store.py` — fork of `scripts/local_audit/findings.py`, path-parameterised, **atomic save**, `attach()`
- [ ] `investigator/cache.py` — fork of `scripts/local_audit/cache.py`, path-parameterised
- [ ] `investigator/catalog.py` — YAML load + validate + merge generic/overlay
- [ ] `investigator/graph.py` — `CaseGraph` loader + fact→persons inversion
- [ ] `investigator/lexicon.py` — antonym table, money/date regexes (NBSP-aware), normalisation helpers
- [ ] `investigator/budget.py` — caps + USD ceilings via `ingestion.clients.estimate_cost_usd`
- [ ] `pyproject.toml` — add `"investigator"` to wheel `packages`
- [ ] `scripts/local_audit/config.py` — add `"investigator"` to `SCAN_DIRS`, `PLANE_DIRS`, `CROSS_PLANE_ALLOWED_IMPORTERS`. **Do not widen `CROSS_PLANE_ALLOWED_SYMBOLS`.**
- [ ] `.gitignore` — `data/dossier/*/investigation/cache.sqlite3`, placed **after** the vitrine negations
- [ ] `tests/test_investigator_smoke.py` — groups 2 (fork equivalence), 5 (tiers), 6 (gate), 7 (resumability), 9 (graph adapter)

### Day 2 — 9 Aug · deterministic end-to-end on vitrine

- [ ] `investigator/passes/__init__.py`
- [ ] `investigator/passes/extract.py` — `PASS_NAME = "graph"`, integrity findings, closed claim enum
- [ ] `investigator/passes/check.py` — `PASS_NAME = "check"`, `evaluate()` pure, `adjudicate: none` path only
- [ ] `investigator/passes/search.py` — `PASS_NAME = "search"`, Layer 0 citation validation; PISTE default-off
- [ ] `investigator/orchestrator.py` — `run_cycle()`, `_record()`, pass order `graph → search → check → contradict → attack`
- [ ] `investigator/render.py` — per-case `DIGEST.md`; the only outbound writer
- [ ] `uv run python -m investigator.orchestrator --case-id vitrine` produces `data/dossier/vitrine/investigation/{findings.jsonl, DIGEST.md, cache.sqlite3}`
- [ ] Tests groups 1 (identity), 4 (predicate semantics)

### Day 3 — 10 Aug · catalogs + remaining passes

- [ ] `investigator/catalogs/generic_fr_succession.yaml` — ~12-14 statute/deontology entries, every `chunk_id` verified against `data/chunks.csv`
- [ ] `investigator/catalogs/README.md` — authoring guide + the person-free rule
- [ ] `data/dossier/private/obligations.yaml` — ~8 contract-clause entries (gitignored)
- [ ] `data/dossier/vitrine/obligations.yaml` — anonymised twin (committed; doc_id globs differ, `anonymize.py` scrubs stems)
- [ ] `investigator/passes/contradict.py` — `PASS_NAME = "contradict"`, deterministic candidate generator, stable order, `CONTRADICT_MAX_PAIRS = 40`
- [ ] `investigator/passes/attack.py` — `PASS_NAME = "attack"`, catalog seed attachment via `store.attach`
- [ ] `investigator/watch.py` — artifact-hash-triggered loop
- [ ] Tests groups 3 (catalog integrity), 8 (contradict determinism)

### Day 4 — 11 Aug am · buffer

- [ ] Full run on `vitrine`, then `private`
- [ ] `README.md` — Plane V section + digest excerpt
- [ ] ADR #70 Follow-ups updated with what actually shipped
- [ ] `uv run pytest tests/ -v` green (251 + new)
- [ ] Privacy grep clean, before and after push

---

## Phase 2 — local qwen3 layers · 12-14 Aug

Each is one additive commit over a pass that already works. None changes a
deterministic verdict's meaning. Each is independently revertible.

- [ ] `investigator/ollama.py` — fork of `scripts/local_audit/ollama_client.py`; `call_json()`, `call_self_consistency()`; inherit `OLLAMA_KEEP_ALIVE="2m"` (12 GB card is shared with BGE-M3)
- [ ] `check` rescue tier — `qwen3:14b`, `verdict_key="atteste"`, n=3, threshold 2. Consumes `Evaluation.candidate_fact_ids`. **Can only flip `gap → satisfied`, never create a gap.** Cache subject `{obligation_id}@{rule_version}|{case_id}|{fact_id}`. Flip catalog entries to `adjudicate: llm_local` one at a time.
- [ ] `contradict` adjudication — `qwen3:30b`, `think=True`, `verdict_key="incompatible"`, over the already-stable pair list
- [ ] `attack` generative confounders — `qwen3:30b`, extending catalog seeds
- [ ] Delete this file

**Known risk:** `scripts/local_audit/passes/duplication.py` documents qwen3:30b
failing a comparable pairwise-judgment task 3/3, both model sizes. Mitigation is
designed in — `contradict` already emits Layer-0 collisions on its own, so the
adjudication layer can be reverted with nothing else lost.

---

## Deferred — ADR #70 Follow-ups

(d) live PISTE validation · (e) `rag.retrieve` for *unanchored* obligation
discovery · (f) multi-case watch · (g) any Streamlit surface · (h) calibration-log
UI. Backfilling `Fact.mentioned_person_ids` stays open under ADR #53 follow-up (d)
— worked around here by inverting `persons.jsonl`.

---

## Verification

```bash
uv run pytest tests/ -v
uv run pytest tests/test_investigator_smoke.py -v

uv run python -m investigator.orchestrator --case-id vitrine
uv run python -m investigator.orchestrator --case-id demo      # empty-graph tolerance
uv run python -m investigator.orchestrator --case-id private

cat data/dossier/vitrine/investigation/DIGEST.md
```

End-to-end checks:

1. A deliberately wrong catalog citation produces a `search` finding and no `check`
   gaps derived from it.
2. `SIGKILL` mid-cycle, then rerun → `findings.jsonl` byte-identical modulo
   `last_seen`.
3. Two runs, no input change → zero new ids, zero reconciliations.
4. `externalisable_findings(..., case_id="private")` raises.

Privacy grep before every commit (CLAUDE.md), and again against `origin/v2-persons`
after push:

```bash
git diff --cached --name-only | grep "data/dossier/" | grep -vE "data/dossier/(demo|vitrine)/"
# must print nothing
```
