"""End-to-end dossier ingestion pipeline: extract → gate → facts → index
(Plane I · offline).

CLI orchestrator for a single case's dossier ingestion — the dossier
analogue of ingestion/build.py's fetch → parse → chunk → index pipeline.
Runs the four dossier stages in sequence for one case_id and reports a
summary, including any coverage warnings from gate.py.

Inputs:  --case-id <id>, --raw-dir <path>, --step {extract,all}, --limit N  (CLI args)
Outputs: full dossier artifact tree under data/dossier/<case_id>/:
             extracted/<doc_id>.md + .json   (extract.py)
             coverage.jsonl                  (gate.py)
             facts.jsonl                     (facts.py)
             chunks.csv                      (index.py)
         plus appended rows in the shared Chroma collection (index.py)

Deliverable 2 status: only the extract stage is implemented. --step extract
runs extract.extract_case and returns a summary; --step all (the default)
raises NotImplementedError until gate/facts/index land in later
deliverables — it does not call any of them yet.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ingestion.dossier import extract, gate, facts, index

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


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
    case_id: str, raw_dir: Path, step: str = "all", limit: int | None = None
) -> dict:
    """Run the requested pipeline step(s) for one case; return a summary dict.

    step="extract" runs only extract.extract_case and returns
    {case_id, step, doc_count, docs}. step="all" (full extract → gate →
    facts → index) is not implemented yet — raises NotImplementedError until
    gate/facts/index land in later deliverables.
    """
    if step == "extract":
        results = _run_stage("EXTRACT", extract.extract_case, case_id, raw_dir, limit=limit)
        return {
            "case_id": case_id,
            "step": step,
            "doc_count": len(results),
            "docs": [r.doc_id for r in results],
        }

    if step == "all":
        raise NotImplementedError("all steps pending — implement in later deliverables")

    raise ValueError(f"unknown --step {step!r}; expected 'extract' or 'all'")


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.build --case-id <id> --raw-dir <path> [--step extract] [--limit N]."""
    parser = argparse.ArgumentParser(
        description="Build one case's dossier artifacts: extract → gate → facts → index."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="directory of raw dossier documents for this case",
    )
    parser.add_argument(
        "--step", choices=["extract", "all"], default="all",
        help="which pipeline step(s) to run (default: all — not yet implemented)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of documents processed (extract step only)",
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info(
        "dossier build starting · case_id=%s raw_dir=%s step=%s limit=%s",
        args.case_id, args.raw_dir, args.step, args.limit,
    )
    summary = run_pipeline(args.case_id, args.raw_dir, step=args.step, limit=args.limit)
    log.info("dossier build complete · %s · total elapsed %.1fs", summary, time.time() - t_start)


if __name__ == "__main__":
    main()
