"""End to end: a folder of PDFs in, an argued report out.

One command for the whole chain, because the three-step version is easy to get
wrong — running `discover` before `facts` exist, or `orchestrator` before
`resolve`, produces a thin report and no error.

    raw/*.pdf
      └─ build      extract → gate → facts → distill → index → resolve
      └─ discover   the acts that stipulate something → obligations.proposed.yaml
      └─ REVIEW     ← stops here by default
      └─ investigate graph → search → check → contradict → attack → RAPPORT.md

**It stops at the review gate.** Discovery verifies every excerpt word-for-word
against its own source document, so invented text cannot survive — but the
bearer, the deadline and the match terms are the model's reading and nobody has
checked them. A wrong bearer produces a confident accusation against the wrong
party, in a report aimed at an insurer. `--accept-proposed` skips the pause for
an unattended run; it is opt-in for that reason.

Stages are invoked as subprocesses rather than imported, following
`investigator/corpus.py`'s precedent for reaching Plane I: each stage keeps its
own CLI contract, its own logging, and its own exit code, and a crash in one
cannot take the runner's process with it.

Usage:
    uv run python -m investigator.run_case --case-id mycase --raw-dir path/to/pdfs
    uv run python -m investigator.run_case --case-id mycase --skip-build --local-llm
    uv run python -m investigator.run_case --case-id mycase --accept-proposed --local-llm
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from investigator import config

log = logging.getLogger(__name__)

STAGES = ("build", "discover", "investigate")


def _run(argv: list[str], label: str) -> float:
    """Run one stage, streaming its output. Raises on failure."""
    log.info("=" * 70)
    log.info("STAGE · %s", label)
    log.info("=" * 70)
    started = time.time()
    result = subprocess.run(argv, cwd=config.PROJECT_ROOT, check=False)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    log.info("STAGE DONE · %s (%.1fs)", label, elapsed)
    return elapsed


def run_case(
    case_id: str,
    raw_dir: Path | None = None,
    skip_build: bool = False,
    skip_discover: bool = False,
    accept_proposed: bool = False,
    local_llm: bool = False,
    model: str | None = None,
    outbound: bool = False,
) -> dict:
    """Build, discover, and investigate one case. Returns per-stage timings."""
    paths = config.CasePaths.for_case(case_id)
    timings: dict[str, float] = {}
    python = [sys.executable]

    # ---------- build ----------
    if skip_build:
        log.info("run_case: --skip-build, using the existing case artifacts")
    else:
        if raw_dir is None:
            raise ValueError("--raw-dir is required unless --skip-build is given")
        timings["build"] = _run(
            python + ["-m", "ingestion.dossier.build", "--case-id", case_id,
                      "--raw-dir", str(raw_dir), "--step", "all"],
            "build (extract → gate → facts → distill → index → resolve)",
        )

    # ---------- discover ----------
    if skip_discover:
        log.info("run_case: --skip-discover, using the existing catalog")
    else:
        timings["discover"] = _run(
            python + ["-m", "investigator.discover", "--case-id", case_id],
            "discover (obligations from this case's own acts)",
        )

    # ---------- the review gate ----------
    proposal = paths.case_catalog_proposal
    if proposal.exists() and not paths.case_catalog.exists():
        if not accept_proposed:
            log.warning("=" * 70)
            log.warning("REVIEW REQUIRED — stopping before the investigation")
            log.warning("=" * 70)
            print(
                f"\nProposition d'obligations : {proposal}\n\n"
                "Chaque extrait a été vérifié mot pour mot contre son document source.\n"
                "En revanche le débiteur, le délai et les termes de preuve sont des\n"
                "lectures du modèle et n'ont été vérifiés par personne.\n\n"
                "  1. relire et corriger le fichier\n"
                f"  2. mv {proposal.name} {paths.case_catalog.name}\n"
                f"  3. uv run python -m investigator.run_case --case-id {case_id} "
                "--skip-build --skip-discover"
                + (" --local-llm" if local_llm else "")
                + "\n"
            )
            return {"case_id": case_id, "stopped_at": "review", "timings": timings}

        # Explicitly asked for: install the proposal unreviewed.
        proposal.rename(paths.case_catalog)
        log.warning(
            "run_case: --accept-proposed — %s installed without review; findings "
            "derived from it rest on an unverified bearer and deadline",
            paths.case_catalog.name,
        )

    # ---------- investigate ----------
    argv = python + ["-m", "investigator.orchestrator", "--case-id", case_id]
    if local_llm:
        argv.append("--local-llm")
    if model:
        argv += ["--model", model]
    if outbound:
        argv.append("--outbound")
    timings["investigate"] = _run(argv, "investigate (graph → search → check → contradict → attack)")

    total = sum(timings.values())
    print(
        "\n" + "=" * 70 + f"\nrun_case · case_id={case_id} · "
        + " ".join(f"{k}={v / 60:.1f}min" for k, v in timings.items())
        + f" · total={total / 60:.1f}min\n"
        + f"  rapport : {paths.investigation_dir / 'RAPPORT.md'}\n"
        + f"  digest  : {paths.digest_md}\n"
    )
    return {"case_id": case_id, "stopped_at": None, "timings": timings}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, discover and investigate one case, end to end.",
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="folder of source PDFs; required unless --skip-build")
    parser.add_argument("--skip-build", action="store_true",
                        help="the case is already built")
    parser.add_argument("--skip-discover", action="store_true",
                        help="use the existing obligations.yaml, propose nothing")
    parser.add_argument("--accept-proposed", action="store_true",
                        help="install obligations.proposed.yaml without review and keep "
                             "going — unattended runs only; the bearer and deadline in a "
                             "proposed obligation have been checked by nobody")
    parser.add_argument("--local-llm", action="store_true",
                        help="enable local adjudication during the investigation")
    parser.add_argument("--model", default=None)
    parser.add_argument("--outbound", action="store_true",
                        help="also write the gated outbound extract")
    args = parser.parse_args()

    try:
        run_case(
            args.case_id,
            raw_dir=args.raw_dir,
            skip_build=args.skip_build,
            skip_discover=args.skip_discover,
            accept_proposed=args.accept_proposed,
            local_llm=args.local_llm,
            model=args.model,
            outbound=args.outbound,
        )
    except (RuntimeError, ValueError) as exc:
        log.error("run_case: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
