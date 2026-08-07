# Plane V — investigator specification

Contracts for `investigator/`. Rationale lives in ADR #70 (`docs/decisions.md`),
per ADR #69: a docstring — and a spec — states the contract; the ADR states why.

Written 2026-08-07, before implementation, per CLAUDE.md's "spec first, code
second". Phase 1 (deterministic, zero LLM calls) targets 11 Aug; Phase 2 (local
qwen3 adjudication layers) 12-14 Aug.

---

## 1. What this plane is for

Plane II answers questions. Plane V **generates them**.

A RAG flow is reactive: a query arrives, context is retrieved, an answer is
synthesised. It cannot surface an obligation nobody asked about. Plane V compares
*what should be in the documents* — per statute, contract, and deontology — against
*what is*, and reports the difference. Its primary signal is **absence**.

That inverts the usual failure mode. A retrieval system's risk is returning the
wrong passage; an absence-driven system's risk is claiming something did not happen
when the record simply was not examined. Section 5 is entirely about not doing
that.

Plane V is **report-only**. It writes `data/dossier/{case_id}/investigation/` and
nothing else. It never mutates a Plane Ib artifact, never re-extracts, and never
sends anything outward except through the single gate in §6.

---

## 2. Module contracts

```
investigator/
  __init__.py       charter: report-only; writes only inside the case dir
  config.py         CasePaths, PROMPT_VERSIONS, budget ceilings, ALLOW_OUTBOUND_CASES
  schema.py         Obligation, PassResult, RunContext, Evaluation, assign_tier(),
                    externalisable_findings()          <- the gate
  store.py          findings store: id-keyed upsert, reconcile, attach; ATOMIC save
  cache.py          SQLite verification cache, (pass, prompt_version, subject, hash)
  catalog.py        YAML load + validate + merge generic/overlay
  graph.py          CaseGraph: frozen loader over Plane Ib artifacts + fact->persons inversion
  lexicon.py        person-free FR legal lexicon: antonyms, money/date regexes, normalisation
  budget.py         call caps + USD ceilings via ingestion.clients
  orchestrator.py   run_cycle(); _record() is the only framework seam
  watch.py          long-running loop, artifact-hash triggered
  render.py         per-case DIGEST.md; the ONLY writer outside the case dir
  passes/{extract,check,contradict,attack,search}.py
  catalogs/generic_fr_succession.yaml
```

`store.py`, `cache.py`, and (in Phase 2) `ollama.py` are **forks** of the
corresponding `scripts/local_audit/` modules, not imports — see ADR #70 for why
importing is impossible and generalising is out of scope. `make_id` and `make_key`
stay formula-identical to their originals and a test asserts that.

**Imported, never copied:**

| Symbol | From | Why |
|---|---|---|
| `estimate_cost_usd`, `extract_usage` | `ingestion.clients` | ADR #68 — model rates may not be redefined outside that module |
| `strip_json_fences`, `parse_json_list` | `ingestion.clients` | ADR #67 |
| `Fact`, `ActorRole`, `RoleAmbiguity`, `DOSSIER_DIR` | `ingestion.dossier.facts` | the graph's input contract |
| `PisteClient` | `ingestion.piste` | live citation validation, default-off |

`rag.retrieve` is deliberately **not** imported. The catalog carries its statute
anchor as an explicit `chunk_id`, so statute text is a `pd.read_csv` lookup over
`data/chunks.csv` — deterministic, free, and complete. Semantic retrieval would
also pull BGE-M3 onto a 12 GB card already contended by the local model.

### `graph.py` — the input contract

`load_graph(case_id) -> CaseGraph` (frozen). Reads, all optional except `facts`:

| File | Into | Absent → |
|---|---|---|
| `facts.jsonl` | `facts: dict[fact_id, Fact]` | empty graph, no raise |
| `actor_roles.jsonl` | `roles: dict[role_id, ActorRole]` | `{}` |
| `role_ambiguities.jsonl` | `ambiguities` | `()` |
| `persons.jsonl` | `persons`, `fact_persons` | `{}` — `demo` has no such file |
| `coverage.jsonl` | `coverage: dict[doc_id, status]`, `coverage_known: bool` | `coverage_known = False` — `vitrine` has no such file |
| `data/chunks.csv` | `statute: dict[chunk_id, row]` | raise (it is committed) |

`fact_persons` is built by **inverting** `persons.jsonl`'s
`role_assignments[].evidence_fact_ids`. Measured on `vitrine`: 235 facts,
28 persons, 212 edges, 172 facts covered, 0 dangling. This is why
`Fact.mentioned_person_ids` is not needed and why ADR #53 follow-up (d) can stay
open.

**Never re-extract.** `fact_id` is index-positional (`f"{doc_id}-f{index:03d}"`,
`ingestion/dossier/facts.py:138`). Re-extraction renumbers, dangling every one of
those 212 edges plus every `ComplianceEntry.evidence_fact_ids` in
`compliance_matrix.json`. Re-extraction is not a refresh; it is a silent
referential-integrity break across three committed artifacts.

---

## 3. The obligation catalog

An obligation is a **codified expected behaviour with a machine-checkable
predicate**. The catalog is the fixed denominator that makes absence meaningful.

Three files, and the split is a privacy rule:

| File | Committed | Contents | Excerpt verified against |
|---|---|---|---|
| `investigator/catalogs/generic_fr_succession.yaml` | yes, `person_free: true` | statute + deontology | `data/chunks.csv` |
| `data/dossier/vitrine/obligations.yaml` | yes (vitrine whitelist) | anonymised contract clauses | `data/dossier/vitrine/chunks.csv` |
| `data/dossier/private/obligations.yaml` | **no** (ADR #66 ignore-by-default) | real clauses, real doc_id globs | `data/dossier/private/chunks.csv` |

A contract-clause obligation cannot be verified without a case, so contract clauses
live only in case overlays. `catalog.load(case_id)` merges generic + overlay by
`obligation_id`, overlay winning.

### Entry schema

```yaml
catalog_id: generic_fr_succession
catalog_version: v1
person_free: true                # test-asserted; gates commit eligibility
obligations:
  - obligation_id: <stable, permanent — renaming re-mints every finding>
    rule_version: v1             # rides in the cache subject key
    title_fr: ...
    severity: critical|high|medium|low|info
    externalisable: true|false
    source:
      kind: statute|contract_clause|deontology|jurisprudence
      ref: "<human citation; appears in claim text>"
      chunk_id: <chunks.csv chunk_id, or null>
      legiarti_id: <LEGIARTI…, or null>
      doc_id_pattern: "<glob over source_doc_id, for non-statute sources>"
      excerpt_fr: |
        <verbatim; for kind==statute MUST appear in that chunk's texte>
    also_anchored: [{chunk_id: ..., note_fr: ...}]
    bearer:
      actor_roles: [<snake_case role_ids matched against Fact.actor_role>]
      bearer_note_fr: ...
    window:
      from_event: <fact_match predicate, or null for standing duties>
      deadline_days: <int|null>
      deadline_note_fr: ...
    foreach: <null | {kind: role_instances, actor_roles: [...], key: person_id}>
    expected_evidence:           # all_of / any_of / none_of over fact_match leaves
      all_of:
        - kind: fact_match
          actor_roles: [...]
          any_terms_fr: [...]
          match_fields: [action, target, verbatim_quote, distilled_context]
          after_trigger: true
          min_count: 1
    evidence_scope:
      doc_id_patterns: [...]
      require_gate_status: ok
    absence_tier_cap: T3
    presence_tier_cap: T1
    adjudicate: none             # Phase 1 ships `none` everywhere
    claim_template_fr: "..."
    confounders_seed: ["..."]
```

### Five commitments the schema encodes

1. **The predicate is closed-world evaluable.** Every leaf is a `fact_match` over a
   finite candidate set drawn from the `CaseGraph`, so `matches == 0` is a
   *decidable fact about the graph*, not a model judgment. That is the whole basis
   for treating absence as a confident negative.
2. **`evidence_scope` is the honesty gate** — see §5.
3. **`absence_tier_cap`** encodes that absence evidence is structurally weaker than
   presence evidence. A gap can never reach T1.
4. **`claim_template_fr` interpolates only catalog-stable placeholders**:
   `{obligation_id}`, `{source_ref}`, `{title_fr}`, `{foreach_role}`,
   `{foreach_key}`. Counts, dates, amounts, fact_ids and day-deltas go in
   `evidence`, which is rewritten on every upsert. Claim text is identity.
5. **`foreach`** makes universally-quantified obligations per-instance, so
   satisfying one institution's notification does not resolve the others.

### Anchors absent from the corpus

The statute corpus is 792 chunks across 9 sources
(`cc_successions`, `cc_usufruit`, `cc_liberalites`, `cc_responsabilite`,
`cgi_dettes_defunt`, `ca_assurances_responsabilite`,
`cp_appropriations_frauduleuses`, `decret_73_609`, `ord_45_2590`). Verified absent:
`cc-1204` (porte-fort), `cc-1344` (mise en demeure), `cc-494-12` (habilitation
familiale), `cc-1231-1`.

An obligation whose anchor is absent ships with `chunk_id: null` and
`legiarti_id` set. `search` Layer 0 then emits a `citation_unverifiable` finding
rather than the catalog test failing — the test keeps its teeth without blocking
authoring.

### Worked example — the `foreach` case

```yaml
  - obligation_id: qu-notif-extrait-etablissements
    rule_version: v1
    title_fr: "Délivrance par le notaire rédacteur d'un extrait de la convention à chaque établissement gestionnaire"
    severity: critical
    externalisable: true
    source:
      kind: contract_clause
      ref: "Convention de quasi-usufruit — clause de notification aux tiers détenteurs"
      chunk_id: null
      doc_id_pattern: "*convention*quasi*usufruit*"
      excerpt_fr: |
        Le notaire soussigné remettra un extrait de la présente convention à chacun
        des établissements dépositaires ou gestionnaires des actifs démembrés.
    bearer:
      actor_roles: [notaire_redacteur]
      bearer_note_fr: "Obligation personnelle du rédacteur, non déléguée."
    window:
      from_event:
        kind: fact_match
        actor_roles: [notaire_redacteur, notaire_instrumentaire]
        any_terms_fr: ["convention de quasi-usufruit"]
        match_fields: [action, target, verbatim_quote]
        min_count: 1
      deadline_days: 30
    foreach:
      kind: role_instances
      actor_roles: [gestionnaire_scpi, etablissement_bancaire]
      key: person_id
    expected_evidence:
      all_of:
        - kind: fact_match
          bind_to_foreach: true
          any_terms_fr: ["extrait", "notification", "signification", "copie de la convention"]
          match_fields: [action, target, verbatim_quote, distilled_context]
          after_trigger: true
          min_count: 1
    evidence_scope:
      doc_id_patterns: ["*courrier*", "*mail*", "*lettre*", "*scpi*", "*banc*"]
      require_gate_status: ok
    absence_tier_cap: T3
    presence_tier_cap: T1
    adjudicate: none
    claim_template_fr: >
      Obligation {obligation_id} ({source_ref}) : aucun fait du dossier n'atteste
      la délivrance de l'extrait de convention au détenteur d'actifs
      {foreach_role}/{foreach_key}.
    confounders_seed:
      - "La remise a pu être verbale ou par voie non archivée."
      - "L'établissement a pu être informé par le quasi-usufruitier lui-même, ce qui n'exonère pas le rédacteur."
```

`{foreach_key}` is a `person_id`, never a canonical name. Claim text is identity
and must never carry a name onto an outbound path; the gate substitutes
`{actor_role}#{ordinal}`, and only the local digest resolves display labels.

### Worked example — a statute anchor, verified

```yaml
  - obligation_id: succ-deblocage-proportion-acte
    rule_version: v1
    title_fr: "Libération de fonds successoraux limitée à la proportion indiquée à l'acte de notoriété"
    severity: critical
    externalisable: true
    source:
      kind: statute
      ref: "Code civil, art. 730-4 ; art. 815-3 in fine"
      chunk_id: cc-730-4
      legiarti_id: LEGIARTI000006430899
      excerpt_fr: |
        Les héritiers désignés dans l'acte de notoriété ou leur mandataire commun sont réputés, à l'égard des tiers détenteurs de biens de la succession, avoir la libre disposition de ces biens et, s'il s'agit de fonds, la libre disposition de ceux-ci dans la proportion indiquée à l'acte.
    also_anchored:
      - chunk_id: cc-815-3
        note_fr: >
          Le consentement de tous les indivisaires est requis pour tout acte ne
          ressortissant pas à l'exploitation normale des biens indivis.
```

Both texts were read out of `data/chunks.csv` and are `etat=VIGUEUR`. Note what
this entry does **not** say: there is no "art. 730-4 al. 2", the article is a
single alinéa, and it *enables* release rather than requiring unanimity — the
unanimity limb is `cc-815-3` in fine. An earlier draft of this catalog asserted
the opposite, and `search`'s Layer 0 is what catches that class of error, with
zero network calls. See ADR #70.

---

## 4. Pass contracts

Passes are plain module-level functions hard-wired by name in the orchestrator —
no base class, no registry — mirroring `scripts/local_audit`. Each returns
`PassResult(findings: list[dict], complete: bool)`.

```python
class PassResult(NamedTuple):
    findings: list[dict]
    complete: bool      # False => orchestrator skips reconcile_pass

@dataclass(frozen=True)
class RunContext:
    case_id: str
    paths: CasePaths
    graph: CaseGraph
    catalog: Catalog
    budget: Budget
    sha: str | None
```

`complete` replaces `slimming.LAST_DIVERGENCE_COMPLETE`, a module global that is
not reentrant and would break per-case concurrent runs.

Order is fixed: `graph → search → check → contradict → attack`. `search` precedes
`check` so a bad citation is known before its obligation produces gaps; `attack`
is last because it reads what the others wrote.

| | `extract` | `check` | `contradict` | `attack` | `search` |
|---|---|---|---|---|---|
| `PASS_NAME` | `"graph"` | `"check"` | `"contradict"` | `"attack"` | `"search"` |
| Phase 1 LLM calls | 0 | 0 | 0 | 0 | 0 |
| Phase 2 model | — | qwen3:14b | qwen3:30b | qwen3:30b | — |
| Network | none | none | none | none | PISTE, default-off |

### `extract` — `PASS_NAME = "graph"`

A thin loader, never re-extraction (§2). Emits only **integrity** findings, and it
earns its slot because a defect in the substrate invalidates every downstream
negative: you may not claim "absent" over a corrupt graph.

Closed claim enum — `fact_ids_orphelins_dans_persons`, `role_non_catalogué`,
`document_sans_fait`, `couverture_absente`. Ids and counts go in `evidence`.
Always `complete=True` (the pass is total by construction). No cache: the whole
pass is a few hundred milliseconds of dict work.

### `check` — `PASS_NAME = "check"`

```python
def evaluate(obligation: Obligation, graph: CaseGraph,
             foreach_key: str | None = None) -> Evaluation:   # pure, no I/O

@dataclass(frozen=True)
class Evaluation:
    status: Literal["satisfied", "gap", "unverifiable", "not_triggered", "window_breach"]
    matched_fact_ids: tuple[str, ...]
    candidate_fact_ids: tuple[str, ...]     # deterministically ranked, <=8
    scope_doc_ids: tuple[str, ...]
    scope_covered: bool
    trigger_fact_id: str | None
    window_breach_days: int | None
```

Term matching is NFKD-normalised, accent-folded and apostrophe-normalised across
`[action, target, verbatim_quote, distilled_context]`. Window arithmetic runs
**only when both dates are non-null** — 174 of vitrine's 235 facts are dated.

`candidate_fact_ids` is unused in Phase 1. It exists so the Phase 2 rescue tier
plugs in without touching any other module.

### `contradict` — `PASS_NAME = "contradict"`

The deterministic candidate generator *is* the pass in Phase 1. The model never
sees the corpus. Four buckets over `lexicon.py`:

| Bucket | Candidate rule |
|---|---|
| `montant` | NBSP-aware `(\d[\d\s .,]*)\s*(€\|EUR\|euros?)`, bucketed by integer euro value; ≥2 facts differing in `date` or `actor_role` |
| `date_instrument` | same normalised instrument token, different `date` for the same action lemma |
| `sens_action` | same `(actor_role, normalised target)`, action verbs on opposite sides of an authored antonym table |
| `denombrement` | party-count collisions, `(\d+)\s+(héritiers?\|indivisaires\|nus-propriétaires)` |

Pairs are ordered `(bucket, fact_id_a, fact_id_b)` with `fact_id_a`
lexicographically first, capped at `CONTRADICT_MAX_PAIRS = 40`. **The cap must
truncate a stable order** — an unstable cap re-mints finding ids every cycle and
destroys the resolve-on-fix signal.

Layer 0 output is itself a finding, at `confidence="low"`, `tier=T5`. So the pass
degrades gracefully if the Phase 2 adjudication layer proves unreliable —
`scripts/local_audit/passes/duplication.py` documents qwen3:30b failing a
comparable pairwise task 3/3 at both model sizes.

### `attack` — `PASS_NAME = "attack"`

Runs over the **store**, not the graph. Targets open `check`/`contradict` findings
at tier ≥ T3. Phase 1 attaches each obligation's `confounders_seed` with
`source: "catalog_seed"` and zero calls; for a well-authored catalog that is most
of the value.

Writes via `store.attach(finding_id, confounders=…, calibration_entry=…)`, which
mutates only fields **outside** `upsert_many`'s update set. That set is exactly
`{severity, confidence, related_files, line, evidence, verified_at_sha,
prompt_version, last_seen, status}`. So a later `check` re-run **cannot erase
adversarial work** — the same mechanism that already preserves `first_seen`. This
is an invariant and it is tested.

### `search` — `PASS_NAME = "search"`

Validates *catalog citations*, not case facts. **Zero LLM calls by rule** — every
question here is a string comparison, and no LLM pass may re-derive what a
deterministic check answers.

Layer 0, no network: does `source.chunk_id` exist in `data/chunks.csv`; is
`etat == "VIGUEUR"`; does `excerpt_fr` appear in that chunk's `texte` after
normalisation; does `legiarti_id` match.

Layer 1, default-off: `PisteClient.get_article(legiarti_id)` confirms the article
is still in force today and its text has not drifted since the corpus was built.
Cache subject `piste|{legiarti_id}`, `code_hash` over the corpus `texte`, so an
article is fetched once per corpus version. **Missing `PISTE_*` credentials → Layer
0 only and `complete=False`**: a missing API key must never resolve real findings.

---

## 5. `gap` is not `unverifiable`

The single most important distinction in the design.

One predicate produces two different outcomes, and conflating them is how an
absence-driven system starts hallucinating:

- **`gap`** — trigger fired, scope covered, zero matches. The finding is
  *"the obligation was not performed."*
- **`unverifiable`** — trigger fired, scope **not** covered. The finding is
  *"the record needed to test this obligation is not in the dossier."*

The second is not a failure mode, it is a deliverable: a document-request list. It
is also the anti-hallucination device, because it makes it structurally impossible
for the engine to upgrade "we didn't look" into "it didn't happen".

Coverage comes from `coverage.jsonl`, written by `ingestion/dossier/gate.py`,
and it is **graded rather than binary**:

| Scope | Status | Tier |
|---|---|---|
| every document verified | `gap` | T3 |
| some verified, some unknown | `gap` | T4 |
| none verified, or no coverage file | `unverifiable` | T5 |

The middle row is load-bearing. On the real corpus 8 of 55 documents carry an
unparsed gate verdict, so an all-or-nothing rule discarded 28 confirmed
documents because of 3 unknown ones and collapsed *every* obligation to
`unverifiable` — the engine could never assert a breach at all. A wholly
unverified scope still says nothing; a mostly-verified one says something
weaker, and the tier carries that instead of the status discarding it.

**`vitrine` has no `coverage.jsonl`** (only `private` does, 93 rows), so
`coverage_known=False` there and every gap degrades to T5. The consequence is
worth stating plainly: the anonymised case yields **no gap findings at all**,
while the real case yields nine. Anonymisation drops the coverage file, and
without it absence cannot be distinguished from a collection failure.

---

## 6. Finding schema, tiers, and the gate

`store.normalize()` produces the 15 keys of `scripts/local_audit/findings.py`
unchanged, plus:

```python
"case_id": str,  "obligation_id": str | None,
"tier": Literal["T1","T2","T3","T4","T5"],  "tier_basis": str,
"externalisable": bool,
"confounders": list[dict],       # {text_fr, source, dispositive, added_at}
"evidence_pointers": {"fact_ids": [], "doc_ids": [], "chunk_ids": [],
                      "statute_refs": [], "person_ids": []},
"calibration": list[dict],       # append-only {ts, tier, confidence, reason, actor}
```

Tiers are **mechanically derived by one pure function, never asserted**:

| Tier | Condition |
|---|---|
| T1 | presence; ≥2 distinct `source_doc_id`; all from `coverage.status == "ok"` docs; no integrity defect touching them |
| T2 | presence; ≥1 verbatim-quoted evidence fact from a gate-passed doc |
| T3 | **absence over a fully covered scope** — the strongest a gap may ever be |
| T4 | adjudicated verdict with consensus, no deterministic corroboration (Phase 2) |
| T5 | absence over partially or unknown-covered scope; any supporting fact touched by an integrity defect |

Final tier is `min(computed, absence_tier_cap | presence_tier_cap)`.

### The gate

```python
OUTBOUND_MIN_TIER = "T2"

def externalisable_findings(findings, case_id, min_tier=OUTBOUND_MIN_TIER) -> list[dict]:
    """The ONLY path from the findings store to any outbound artifact."""
```

Drops anything below `min_tier`; anything `externalisable=False` on the finding or
its obligation; anything carrying a dispositive confounder. Redacts `person_id`s to
`{actor_role}#{ordinal}`. Raises if `case_id not in config.ALLOW_OUTBOUND_CASES`,
which is `{"vitrine", "demo"}` — **`private` is deliberately excluded**.

Enforced three ways: `render.render_outbound()` is the only function writing
outside the case dir; a source-scan test asserts no other module names
`config.OUTBOUND_DIR`; a monkeypatch-sentinel test proves there is no second path.

This is the schema-level home for the case doctrine that T5 material is never
externalised.

---

## 7. Resumability — why a kill costs at most one unit of work

The unit of work is **one (pass, subject) evaluation whose verdict is committed
before the next begins**. Everything else in a cycle is recomputation from files,
which is free.

1. Finding ids derive from `(pass, subject, claim)` — no timestamp — so
   `upsert_many` is idempotent and `first_seen` is never mutated.
2. `store.save_all` writes tmp + `os.replace`. The forked original uses
   `write_text`, which truncates before writing; `local_audit` survives that
   because nobody kills it, and this harness is designed to be killed. **Without
   this, point 1 is a lie.**
3. A truncated pass returns `complete=False`, so `reconcile_pass` never runs and
   nothing is falsely resolved. Budget cutoff must stay distinguishable from
   "fixed".
4. `last_completed_pass` in the state file resumes at a pass boundary.
5. Cache writes `commit()` then `close()` per write, and keys carry no timestamp,
   so a restart re-derives the same key and hits.
6. A `None` model verdict is never cached — "no verdict" is not "no".

State file `data/dossier/{case_id}/investigation/.watch_state.json` holds
`cycle_seq`, timestamps, `artifact_hashes` (`facts.jsonl`, `persons.jsonl`,
`actor_roles.jsonl`, `coverage.jsonl`, each catalog, `data/chunks.csv`),
`last_completed_pass`, `incomplete_passes`, and the USD counters.

**The watch trigger is artifact hashes, not `git rev-parse HEAD`.**
`scripts/local_audit/watch.py` polls git because its subject is code; this plane's
subject is data.

**Cold-start backlog mode is the "runs as long as it needs" property**, and it
falls out of budget-cap + cache with no extra machinery: the loop does not sleep
while the last cycle reported incomplete. In Phase 1 a full deterministic cycle
over `vitrine` is a few seconds and $0.00, so a cycle never truncates; the
machinery exists so Phase 2 inherits it unchanged.

### Budget

```python
@dataclass
class Budget:
    local_calls: int = 400        # wall-clock gate, not a money gate
    cloud_calls: int = 20
    piste_calls: int = 20
    usd_cycle_ceiling: float = 1.00
    usd_session_ceiling: float = 10.00
```

Pre-flight charge via `estimate_cost_usd` (which raises `KeyError` on an unknown
model rather than silently zeroing — ADR #45/#68), settled post-flight with
`extract_usage`, which prefers the provider's own reported cost. Two ceilings,
mirroring `rag/compliance.py`'s dry-run/real-run pair.

---

## 8. Testing stance

`tests/test_investigator_smoke.py`, fast suite, **zero LLM calls, zero network**,
file-local fixtures (there is no `conftest.py`). Mirrors
`tests/test_local_audit_smoke.py`: assert **invariants, never counts**, because
counts are supposed to change as the catalog grows and findings resolve.

Groups: identity · fork equivalence with `local_audit` · catalog integrity
(including "every statute `excerpt_fr` appears verbatim in its chunk", so a bad
citation cannot be committed) · predicate semantics on synthetic graphs · tier
assignment · the gate · resumability · contradict determinism · graph adapter on
real `vitrine`/`demo` data.

Explicitly not asserted: finding counts, tier distributions, "the catalog finds N
gaps in vitrine".

---

## 9. Relationship to `rag/compliance.py`

Both assess obligations against case facts. They are not the same system, and
Plane V does not replace Plane II's compliance surface — ADR #43 stands.

| | `rag/compliance.py` | `investigator/passes/check.py` |
|---|---|---|
| Obligation source | LLM discovers them inside 8 retrieved chunks per role | authored, versioned, `rule_version`-invalidated catalog |
| Statute anchor | semantic retrieval, `RELEVANT_STATUTE_K = 8` | explicit `chunk_id`, exact CSV lookup |
| Coverage semantics | an obligation whose article did not retrieve is **silently absent** | fixed denominator; "not evaluated" ≠ "satisfied" |
| Persistence | full regeneration per run; no `first_seen`, no resolve-on-fix | id-keyed upsert, `reconcile_pass`, calibration history |
| Adversarial | none | `confounders`, seeded by catalog, extended in Phase 2 |
| First-run cost | ~$22 (46 clusters, Opus 4.7 max — ADR #49's anchor) | $0.00 |

The decisive row is the third. Compliance cannot distinguish "evaluated and
satisfied" from "never evaluated", because a missing obligation produces no output
at all. A catalog is a fixed denominator, and that is the entire epistemic basis
for treating absence as a signal.
