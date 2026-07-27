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

from ingestion.dossier import extract, gate, facts, index

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Steps that read raw source files and therefore require --raw-dir.
STAGES_REQUIRING_RAW_DIR = {"extract", "all"}


# ========== stage runner ==========

def _run_stage(name: str, func, *args, **kwargs):
    """Run one pipeline stage with elapsed-time logging."""
    log.info("=" * 70)
    log.info("STAGE START · %s", name)
    log.info("=" * 70)
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception:
        log.error("STAGE FAILED · %s (after %.1fs)", name, time.time() - t0)
        raise
    log.info("STAGE DONE · %s (%.1fs)", name, time.time() - t0)
    return result


# ========== pipeline orchestration ==========

def run_pipeline(
    case_id: str, raw_dir: Path | None = None, step: str = "all", limit: int | None = None
) -> dict:
    """Run the requested pipeline step(s) for one case; return a summary dict.

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

    if step == "index":
        result = _run_stage("INDEX", index.index_dossier, case_id)

        print(
            f"index summary · case_id={case_id} docs_indexed={result.docs_indexed} "
            f"chunks_created={result.chunks_created} facts_backfilled={result.facts_backfilled} "
            f"facts_unmatched={result.facts_unmatched} elapsed={result.elapsed:.1f}s"
        )

        return {"case_id": case_id, "step": step, **result.model_dump()}

    if step == "all":
        extract_summary = run_pipeline(case_id, raw_dir=raw_dir, step="extract", limit=limit)
        gate_summary = run_pipeline(case_id, step="gate")
        facts_summary = run_pipeline(case_id, step="facts")
        index_summary = run_pipeline(case_id, step="index")

        print(
            f"all summary · case_id={case_id} "
            f"docs_extracted={extract_summary['doc_count']} "
            f"gate_ok={gate_summary['ok']} gate_warnings={gate_summary['warnings']} "
            f"facts={facts_summary['facts_extracted']} "
            f"chunks_created={index_summary['chunks_created']} "
            f"facts_backfilled={index_summary['facts_backfilled']}"
        )

        return {
            "case_id": case_id,
            "step": step,
            "extract": extract_summary,
            "gate": gate_summary,
            "facts": facts_summary,
            "index": index_summary,
        }

    raise ValueError(
        f"unknown --step {step!r}; expected 'extract', 'gate', 'facts', 'index', or 'all'"
    )


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.build --case-id <id> [--raw-dir <path>] [--step extract|gate|facts|index|all] [--limit N]."""
    parser = argparse.ArgumentParser(
        description="Build one case's dossier artifacts: extract → gate → facts → index."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--raw-dir", type=Path, required=False, default=None,
        help="directory of raw dossier documents for this case "
             "(required for --step extract/all; optional for --step gate/facts/index)",
    )
    parser.add_argument(
        "--step", choices=["extract", "gate", "facts", "index", "all"], default="all",
        help="which pipeline step(s) to run (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of documents processed (extract step only)",
    )
    args = parser.parse_args()

    if args.step in STAGES_REQUIRING_RAW_DIR and args.raw_dir is None:
        parser.error(f"--raw-dir is required for --step {args.step}")

    t_start = time.time()
    log.info(
        "dossier build starting · case_id=%s raw_dir=%s step=%s limit=%s",
        args.case_id, args.raw_dir, args.step, args.limit,
    )
    summary = run_pipeline(args.case_id, args.raw_dir, step=args.step, limit=args.limit)
    log.info("dossier build complete · %s · total elapsed %.1fs", summary, time.time() - t_start)


if __name__ == "__main__":
    main()
