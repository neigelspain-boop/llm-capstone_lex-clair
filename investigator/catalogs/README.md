# Authoring obligation catalogs

An obligation is a **codified expected behaviour with a machine-checkable
predicate**. The catalog is the fixed denominator that makes absence a signal:
because an obligation nobody wrote produces no output at all, what is not in
here is invisible to the engine.

Full schema: `docs/investigator-spec.md` §3. Rationale: ADR #70.

## Where an entry goes

| File | Committed | For |
|---|---|---|
| `generic_fr_succession.yaml` | yes | statute and deontology, verifiable against `data/chunks.csv` |
| `data/dossier/{case}/obligations.yaml` | only `demo`/`vitrine` | contract clauses, verifiable against that case's documents |

A contract clause cannot be citation-verified without a case, so clauses never
go in the generic file. Everything committed must be **person-free** — no party
name, address, amount, or identifier. A test asserts it against the vitrine
roster, because a leak here would leak into every future case.

## The two rules that are not negotiable

**Never invent a citation.** Every `kind: statute` excerpt must appear verbatim
in its chunk's `texte`; the `search` pass checks all of them on every cycle and
a test blocks the commit. The corpus is 792 chunks across 9 sources, and real
articles fall outside it — `cc-1204`, `cc-1344`, `cc-494-12` and `cc-1231-1` all
do. Anchor those with a `legiarti_id` **confirmed against Légifrance**, leaving
`chunk_id: null`; `search` then reports them as unverifiable-here rather than
wrong. A guessed LEGIARTI is worse than no anchor: it looks verified.

**`claim_template_fr` is identity.** It may interpolate only
`{obligation_id}`, `{source_ref}`, `{title_fr}`, `{foreach_role}`,
`{foreach_key}`. Counts, dates, amounts and fact ids belong in `evidence`,
which is rewritten every cycle. A claim carrying a count re-mints its finding id
whenever the case changes, so the finding can never be tracked, resolved, or
argued against across runs.

## Getting a predicate right

These are the mistakes that actually happened while authoring the first
catalog, each of which produced confident and wrong output:

- **Leaving an evidence leaf's `actor_roles` empty and meaning "anyone".** It
  means *the bearer*. Performance is conduct by the party who owes it; without
  that default one well-worded sentence from an unrelated third party reports
  the duty performed. That is a false `satisfied`, and no later adjudication
  layer can repair it — the Phase 2 rescue only ever flips `gap → satisfied`.
  Use `any_actor: true` when you genuinely mean any party, as for a trigger.

- **Leaving `trigger_select` at its default for a state-change trigger.** A
  dossier recites earlier successions as background. `earliest` is right for a
  duty that starts on a *request*; `latest` for one that starts on an
  *extinction* or a *death*.

- **Assuming a deadline is testable.** It is not, when the corpus disagrees
  about when the trigger happened — this one matches 45 dated facts across
  three different deaths. The engine refuses to compute a breach from an
  ambiguous trigger and says so in `evidence`. Narrow the trigger's
  `actor_roles` and terms if you want a real deadline test.

- **Writing broad `any_terms_fr`.** Recall costs you nothing here; precision is
  everything. `"passif"` matches half the dossier. `"inscription au passif"`
  and `"créance de restitution au passif"` match the thing you meant.

## Confounders are part of the entry, not an afterthought

`confounders_seed` is the strongest argument *against* the finding your
obligation will produce, written by the person best placed to know it — you,
while the clause is in front of you. The `attack` pass attaches them with zero
model calls, and a finding that reaches an outbound artifact carries them.

Write the ones a competent opponent would actually make: the act performed by
another channel, the clause read as an obligation of means, the deadline not
yet expired, the duty owed by someone else.

## Before you commit

```bash
uv run pytest tests/test_investigator_smoke.py -q     # catalog integrity + citations
uv run python -m investigator.orchestrator --case-id vitrine
```

`search` must report zero findings. Any it does report are defects in the
catalog, not in the case.
