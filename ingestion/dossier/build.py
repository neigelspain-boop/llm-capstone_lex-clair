"""End-to-end dossier ingestion pipeline: extract → gate → facts → index
(Plane I · offline).

CLI orchestrator for a single case's dossier ingestion — the dossier
analogue of ingestion/build.py's fetch → parse → chunk → index pipeline.
Runs the four dossier stages in sequence for one case_id and reports a
summary, including any coverage warnings from gate.py.

Inputs:  --case-id <id>, --raw-dir <path>  (CLI args)
Outputs: full dossier artifact tree under data/dossier/<case_id>/:
             extracted/<doc_id>.md + .json   (extract.py)
             coverage.jsonl                  (gate.py)
             facts.jsonl                     (facts.py)
             chunks.csv                      (index.py)
         plus appended rows in the shared Chroma collection (index.py)
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

def run_pipeline(case_id: str, raw_dir: Path) -> dict:
    """Run extract → gate → facts → index for one case; return a summary dict."""
    raise NotImplementedError(
        "spec: call _run_stage('EXTRACT', extract.extract_case, case_id, "
        "raw_dir), _run_stage('GATE', gate.gate_case, case_id), "
        "_run_stage('FACTS', facts.extract_case_facts, case_id), "
        "_run_stage('INDEX', index.index_case, case_id) in sequence. Return "
        "a summary dict: {case_id, doc_count, fact_count, chunk_count, "
        "coverage_warnings: [records from gate_case where verdict == "
        "'incomplete']}. A stage failure must abort the remaining stages "
        "(mirrors ingestion/build.py's _run_stage re-raise behavior)."
    )


# ========== CLI entrypoint ==========

def main() -> None:
    """Command-line entrypoint: python -m ingestion.dossier.build --case-id <id> --raw-dir <path>."""
    parser = argparse.ArgumentParser(
        description="Build one case's dossier artifacts: extract → gate → facts → index."
    )
    parser.add_argument("--case-id", type=str, required=True, help="case identifier")
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="directory of raw dossier documents for this case",
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("dossier build starting · case_id=%s raw_dir=%s", args.case_id, args.raw_dir)
    summary = run_pipeline(args.case_id, args.raw_dir)
    log.info("dossier build complete · %s · total elapsed %.1fs", summary, time.time() - t_start)


if __name__ == "__main__":
    main()
