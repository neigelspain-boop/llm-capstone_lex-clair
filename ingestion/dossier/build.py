"""End-to-end dossier ingestion pipeline: extract → gate → facts → index
(Plane I · offline).

CLI orchestrator for a single case's dossier ingestion — the dossier
analogue of ingestion/build.py's fetch → parse → chunk → index pipeline.
Runs the four dossier stages in sequence for one case_id and reports a
summary, including any coverage warnings from gate.py.

Inputs:  --case-id <id>, --step {extract,gate,facts,index,all}, --limit N,
         and --raw-dir <path> (conditionally required — see below)  (CLI args)
Outputs: full dossier artifact tree under data/dossier/<case_id>/:
             extracted/<doc_id>.md + .json   (extract.py)
             coverage.jsonl                  (gate.py)
             facts.jsonl                     (facts.py, backfilled by index.py)
             chunks.csv                      (index.py)
         plus appended rows in the shared Chroma collection (index.py)

Deliverable 5 status: extract, gate, facts, and index are all implemented.
--step all runs the full extract → gate → facts → index pipeline.

--raw-dir is only required for steps that read raw source files (extract,
all) — argparse enforces this after parsing (see STAGES_REQUIRING_RAW_DIR),
since it can't express "required unless --step is X" declaratively. gate and
facts operate on already-extracted artifacts under data/dossier/<case_id>/
and never need --raw-dir: each sidecar written by extract.py now carries
source_relpath, so gate.gate_case(case_id) locates every document's original
source file by itself, and facts.extract_case_facts(case_id) reads only the
.md transcripts under extracted/.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ingestion import build as statute_build
from ingestion.dossier import anonymize, distill, extract, gate, facts, index, resolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Steps that read raw source files and therefore require --raw-dir.
STAGES_REQUIRING_RAW_DIR = {"extract", "all"}


# ========== stage runner ==========

# Shared with Plane I's statute pipeline — this file previously carried a
# byte-identical copy, docstring included. Both are `ingestion.*`, so the
# import is within one plane namespace, not a cross-plane path.
_run_stage = statute_build._run_stage


# ========== pipeline orchestration ==========

def run_pipeline(
    case_id: str,
    raw_dir: Path | None = None,
    step: str = "all",
    limit: int | None = None,
    source_case_id: str = anonymize.DEFAULT_SOURCE_CASE_ID,
    use_llm: bool = False,
) -> dict:
    """Run the requested pipeline step(s) for one case; return a summary dict.

    step="anonymize" derives case_id's extracted/*.md AND its analytical
    artifacts from source_case_id's (ADRs #59, #62), then verifies the result,
    raising if any known identifier survives. Deliberately excluded from
    step="all": it is a one-off derivation from a DIFFERENT case, not a stage
    of this case's own build.

    step="anonymize-investigation" does the same for the Plane V transcript —
    investigation/findings.jsonl and DIGEST.md — through the ROLE register
    rather than the persona roster (ADR #78). Excluded from step="all" for
    the same reason, and separate from step="anonymize" because it derives a
    different plane's artifacts from a different vocabulary: publishing the
    dossier and publishing the investigation over it are two decisions.

    step="scrub" re-applies the structured-PII regexes to an already-derived
    case in place, for when a pattern is strengthened after the case was
    built. It takes no source case: it repairs, it does not re-derive.

    use_llm gates the anonymiser's residual sweep and defaults OFF, which is
    the opposite of anonymize_case's own default. Two reasons, both learned
    from the first showcase build. The sweep rewrites the markdown but cannot
    be replayed identically over a fact's verbatim_quote, so quotes stop
    matching chunks and the index backfill degrades. And it is generative: it
    reconstructed an email address that the structured-PII layer had already
    redacted, which the gate then caught as a leak. Deterministic-only keeps
    markdown, facts and chunks in agreement and keeps the output reproducible.

    step="extract" runs only extract.extract_case and returns
    {case_id, step, doc_count, docs} — requires raw_dir. step="gate" runs
    gate.gate_case(case_id) (no raw_dir needed — see module docstring),
    prints a one-line summary, and returns
    {case_id, step, total, ok, warnings, source_missing, parse_failed, elapsed}.
    step="facts" runs facts.extract_case_facts(case_id) (no raw_dir needed),
    prints a one-line summary, and returns
    {case_id, step, docs_processed, facts_extracted, unique_roles,
    ambiguities, parse_failed_docs, elapsed}.
    step="index" runs index.index_dossier(case_id) (no raw_dir needed),
    prints a one-line summary, and returns
    {case_id, step, **DossierIndexResult.model_dump()}.
    step="all" runs extract → gate → facts → index in sequence (requires
    raw_dir); any stage failure aborts the remaining stages.
    """
    if step == "anonymize":
        summary = _run_stage(
            "ANONYMIZE", anonymize.anonymize_case, case_id,
            source_case_id=source_case_id, use_llm=use_llm,
        )

        residual = summary["residual_proper_nouns"]
        top = sorted(residual.items(), key=lambda kv: -kv[1])[:15]
        artifacts = " ".join(f"{k}={v}" for k, v in sorted(summary["artifacts"].items()))
        print(
            f"anonymize summary · case_id={case_id} source={summary['source_case_id']} "
            f"docs_written={summary['docs_written']} entities_mapped={summary['entities_mapped']} "
            f"files_verified={summary['files_scanned']} "
            f"residual_proper_nouns={len(residual)} elapsed={summary['elapsed']:.1f}s"
        )
        if artifacts:
            print(f"  artifacts translated: {artifacts}")
        if top:
            # The gate proves the KNOWN identifiers are gone; it cannot prove
            # an unknown one is. These need a human read before publishing.
            print("  unrecognised proper nouns to review: " + ", ".join(f"{t}({n})" for t, n in top))

        return {"case_id": case_id, "step": step, **summary}

    if step == "anonymize-investigation":
        summary = _run_stage(
            "ANONYMIZE-INVESTIGATION", anonymize.anonymize_investigation, case_id,
            source_case_id=source_case_id,
        )
        residual = summary["residual_proper_nouns"]
        top = sorted(residual.items(), key=lambda kv: -kv[1])[:15]
        print(
            f"anonymize-investigation summary · case_id={case_id} "
            f"source={summary['source_case_id']} "
            f"findings={summary['findings_translated']} ids_mapped={summary['ids_mapped']} "
            f"files_verified={summary['files_scanned']} "
            f"residual_proper_nouns={len(residual)} elapsed={summary['elapsed']:.1f}s"
        )
        if top:
            print("  unrecognised proper nouns to review: " + ", ".join(f"{t}({n})" for t, n in top))
        return {"case_id": case_id, "step": step, **summary}

    if step == "scrub":
        changed = _run_stage("SCRUB", anonymize.scrub_case_pii, case_id)
        total = sum(changed.values())
        print(
            f"scrub summary · case_id={case_id} files_changed={len(changed)} "
            f"records_changed={total}"
        )
        for name, count in sorted(changed.items()):
            print(f"  {name}: {count}")
        return {"case_id": case_id, "step": step, "files_changed": len(changed),
                "records_changed": total, "changed": changed}

    if step == "extract":
        results = _run_stage("EXTRACT", extract.extract_case, case_id, raw_dir, limit=limit)
        return {
            "case_id": case_id,
            "step": step,
            "doc_count": len(results),
            "docs": [r.doc_id for r in results],
        }

    if step == "gate":
        t0 = time.time()
        reports = _run_stage("GATE", gate.gate_case, case_id)
        elapsed = time.time() - t0

        ok = sum(
            1 for r in reports
            if r.status == "ok" and not r.missing_facts and not r.mistranscriptions
        )
        warnings_ = sum(
            1 for r in reports
            if r.status == "ok" and (r.missing_facts or r.mistranscriptions)
        )
        source_missing = sum(1 for r in reports if r.status == "source_missing")
        parse_failed = sum(1 for r in reports if r.status == "parse_failed")
        total = len(reports)

        print(
            f"gate summary · case_id={case_id} total={total} ok={ok} "
            f"warnings={warnings_} source_missing={source_missing} "
            f"parse_failed={parse_failed} elapsed={elapsed:.1f}s"
        )

        return {
            "case_id": case_id,
            "step": step,
            "total": total,
            "ok": ok,
            "warnings": warnings_,
            "source_missing": source_missing,
            "parse_failed": parse_failed,
            "elapsed": elapsed,
        }

    if step == "facts":
        summary = _run_stage("FACTS", facts.extract_case_facts, case_id)

        print(
            f"facts summary · case_id={case_id} docs_processed={summary['docs_processed']} "
            f"facts={summary['facts_extracted']} unique_roles={summary['unique_roles']} "
            f"ambiguities={summary['ambiguities']} parse_failed_docs={summary['parse_failed_docs']} "
            f"elapsed={summary['elapsed']:.1f}s"
        )

        return {"case_id": case_id, "step": step, **summary}

    if step == "distill":
        summary = _run_stage("DISTILL", distill.distill_case, case_id)

        print(
            f"distill summary · case_id={case_id} total_facts={summary['total_facts']} "
            f"distilled={summary['facts_distilled']} cache_hits={summary['cache_hits']} "
            f"cost=${summary['total_cost']:.4f}"
        )

        return {"case_id": case_id, "step": step, **summary}

    if step == "resolve":
        summary = _run_stage("RESOLVE", resolve.resolve_case, case_id)

        print(
            f"resolve summary · case_id={case_id} "
            f"roles_processed={summary['total_roles_processed']} "
            f"persons={summary['total_persons_resolved']} "
            f"cache_hits={summary['cache_hits']} cost=${summary['total_cost']:.4f}"
        )

        return {"case_id": case_id, "step": step, **summary}

    if step == "index":
        result = _run_stage("INDEX", index.index_dossier, case_id)

        print(
            f"index summary · case_id={case_id} docs_indexed={result.docs_indexed} "
            f"chunks_created={result.chunks_created} facts_backfilled={result.facts_backfilled} "
            f"facts_unmatched={result.facts_unmatched} elapsed={result.elapsed:.1f}s"
        )

        return {"case_id": case_id, "step": step, **result.model_dump()}

    if step == "all":
        # Order is load-bearing, not alphabetical:
        #   distill after facts, because it rewrites facts.jsonl in place;
        #   index after distill, because its source_chunk_id backfill rewrites
        #     the same rows and would otherwise drop distilled_context;
        #   resolve last, because it clusters persons from distilled_context
        #     per role (ADR #55).
        # distill and resolve were outside `all` until now, which meant a case
        # built through this pipeline had null distilled_context — weakening
        # every downstream term match — and no persons.jsonl at all, so Plane
        # V's `foreach` obligations produced nothing. mentions.py stays out:
        # ADR #55 supersedes it.
        extract_summary = run_pipeline(case_id, raw_dir=raw_dir, step="extract", limit=limit)
        gate_summary = run_pipeline(case_id, step="gate")
        facts_summary = run_pipeline(case_id, step="facts")
        distill_summary = run_pipeline(case_id, step="distill")
        index_summary = run_pipeline(case_id, step="index")
        resolve_summary = run_pipeline(case_id, step="resolve")

        print(
            f"all summary · case_id={case_id} "
            f"docs_extracted={extract_summary['doc_count']} "
            f"gate_ok={gate_summary['ok']} gate_warnings={gate_summary['warnings']} "
            f"facts={facts_summary['facts_extracted']} "
            f"distilled={distill_summary['facts_distilled']} "
            f"chunks_created={index_summary['chunks_created']} "
            f"facts_backfilled={index_summary['facts_backfilled']} "
            f"persons={resolve_summary['total_persons_resolved']}"
        )

        return {
            "case_id": case_id,
            "step": step,
            "extract": extract_summary,
            "gate": gate_summary,
            "facts": facts_summary,
            "distill": distill_summary,
            "index": index_summary,
            "resolve": resolve_summary,
        }

    raise ValueError(
        f"unknown --step {step!r}; expected 'extract', 'gate', 'facts', "
        f"'distill', 'index', 'resolve', or 'all'"
    )


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.build --case-id <id> [--raw-dir <path>] [--step extract|gate|facts|index|all] [--limit N]."""
    parser = argparse.ArgumentParser(
        description="Build one case's dossier artifacts: extract → gate → facts → index."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--source-case-id", type=str, default=anonymize.DEFAULT_SOURCE_CASE_ID,
        help="source case for --step anonymize (ADR #59); ignored by every other step. "
        f"Default: {anonymize.DEFAULT_SOURCE_CASE_ID}",
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=False, default=None,
        help="directory of raw dossier documents for this case "
             "(required for --step extract/all; optional for --step gate/facts/index)",
    )
    parser.add_argument(
        "--step",
        choices=["anonymize", "anonymize-investigation", "scrub", "extract", "gate",
                 "facts", "distill", "index", "resolve", "all"],
        default="all",
        help="which pipeline step(s) to run (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of documents processed (extract step only)",
    )
    parser.add_argument(
        "--llm-sweep", action="store_true",
        help="enable the LLM residual sweep during --step anonymize (default: off). "
             "The sweep is generative: it degrades fact/chunk agreement and has "
             "reconstructed redacted PII. Use it to hunt residuals, not to publish.",
    )
    args = parser.parse_args()

    if args.step in STAGES_REQUIRING_RAW_DIR and args.raw_dir is None:
        parser.error(f"--raw-dir is required for --step {args.step}")

    t_start = time.time()
    log.info(
        "dossier build starting · case_id=%s raw_dir=%s step=%s limit=%s",
        args.case_id, args.raw_dir, args.step, args.limit,
    )
    summary = run_pipeline(
        args.case_id, args.raw_dir, step=args.step, limit=args.limit,
        source_case_id=args.source_case_id, use_llm=args.llm_sweep,
    )
    log.info("dossier build complete · %s · total elapsed %.1fs", summary, time.time() - t_start)


if __name__ == "__main__":
    main()
