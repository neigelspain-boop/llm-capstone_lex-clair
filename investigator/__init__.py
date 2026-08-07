"""Plane V — obligation-driven case investigation.

Where Plane II answers a query, Plane V generates the questions: it compares
what *should* be in a case's documents (per statute, contract, and deontology)
against what *is*, and reports the difference. Its primary signal is absence.

Report-only, and the guarantee is structural rather than conventional: every
write lands under `data/dossier/{case_id}/investigation/`, and the only path
from the findings store to a shareable artifact is
`investigator.schema.externalisable_findings()`. Plane Ib artifacts are read
and never mutated; nothing here re-extracts, because `fact_id` is
index-positional and re-extraction would dangle every `evidence_fact_ids` edge
in `persons.jsonl` and `compliance_matrix.json`.

Contracts: `docs/investigator-spec.md`. Rationale: ADR #70.
"""
