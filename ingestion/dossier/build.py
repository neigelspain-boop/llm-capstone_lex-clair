"""End-to-end dossier ingestion pipeline: extract → gate → facts → index
(Plane I · offline).

CLI orchestrator for a single case's dossier ingestion — the dossier
analogue of ingestion/build.py's fetch → parse → chunk → index pipeline.
Runs the four dossier stages in sequence for one case_id and reports a
summary, including any coverage warnings from gate.py.

Inputs:  --case-id <id>, --step {extract,gate,all}, --limit N, and
         --raw-dir <path> (conditionally required — see below)  (CLI args)
Outputs: full dossier artifact tree under data/dossier/<case_id>/:
             extracted/<doc_id>.md + .json   (extract.py)
             coverage.jsonl                  (gate.py)
             facts.jsonl                     (facts.py)
             chunks.csv                      (index.py)
         plus appended rows in the shared Chroma collection (index.py)

Deliverable 3 status: extract and gate steps are implemented. --step all
(full extract → gate → facts → index) is not implemented yet — raises
NotImplementedError until facts/index land in Deliverable 5.

--raw-dir is only required for steps that read raw source files (extract,
all) — argparse enforces this after parsing (see STAGES_REQUIRING_RAW_DIR),
since it can't express "required unless --step is X" declaratively. gate
(and future facts/index) operate on already-extracted artifacts under
data/dossier/<case_id>/ and never need --raw-dir: each sidecar written by
extract.py now carries source_relpath, so gate.gate_case(case_id) locates
every document's original source file by itself.
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
    step="all" (full extract → gate → facts → index) is not implemented
    yet — raises NotImplementedError until facts/index land in Deliverable 5.
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

    if step == "all":
        raise NotImplementedError("all steps pending — implement in later deliverables")

    raise ValueError(f"unknown --step {step!r}; expected 'extract', 'gate', or 'all'")


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.build --case-id <id> [--raw-dir <path>] [--step extract|gate] [--limit N]."""
    parser = argparse.ArgumentParser(
        description="Build one case's dossier artifacts: extract → gate → facts → index."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--raw-dir", type=Path, required=False, default=None,
        help="directory of raw dossier documents for this case "
             "(required for --step extract/all; optional for --step gate)",
    )
    parser.add_argument(
        "--step", choices=["extract", "gate", "all"], default="all",
        help="which pipeline step(s) to run (default: all — not yet implemented)",
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
