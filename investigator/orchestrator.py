"""One investigation cycle: load, run every pass, record, render.

`_record` is the entire framework. There is no pass base class, no registry and
no discovery — passes are plain functions listed by name below, exactly as in
`scripts/local_audit/orchestrator.py`. Adding a pass means importing it and
adding a line, which is the point: the sequence is readable in one screen and
its ordering constraints are visible.

Order is fixed and load-bearing:

    graph → search → check → contradict → attack

`graph` first because a defect in the substrate invalidates every later
negative. `search` before `check` so a bad citation is known before obligations
derived from it start producing gaps. `attack` last because it reads the store
the others just wrote.

Usage:
    uv run python -m investigator.orchestrator --case-id vitrine
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from investigator import budget as budget_mod
from investigator import catalog as catalog_mod
from investigator import config, graph as graph_mod, ollama, render, store
from investigator.report import render_brief
from investigator.report import render_brief as report_mod_render
from investigator.passes import attack, check, contradict, extract, search
from investigator.schema import PassResult, RunContext

log = logging.getLogger(__name__)

# Order matters; see the module docstring. `attack` is last because it reads
# the store the others just wrote.
PASSES = (
    (extract.PASS_NAME, extract.run, None),
    (search.PASS_NAME, search.run, "search_offline"),
    (check.PASS_NAME, check.run, None),
    (contradict.PASS_NAME, contradict.run, "contradict"),
    (attack.PASS_NAME, attack.run, "attack"),
)


@dataclass
class CycleReport:
    case_id: str
    elapsed_s: float = 0.0
    obligations: int = 0
    facts: int = 0
    per_pass: dict[str, tuple[int, bool]] = field(default_factory=dict)
    open_findings: int = 0
    budget: dict = field(default_factory=dict)
    digest_path: str = ""
    brief_path: str = ""
    brief_path: str = ""

    def summary(self) -> str:
        passes = " ".join(
            f"{name}={n}{'' if complete else '(partiel)'}"
            for name, (n, complete) in self.per_pass.items()
        )
        return (
            f"investigator · case_id={self.case_id} obligations={self.obligations} "
            f"facts={self.facts} {passes} open={self.open_findings} "
            f"local_calls={self.budget.get('local_used', 0)} "
            f"elapsed={self.elapsed_s:.1f}s cost=${self.budget.get('usd_spent_cycle', 0.0):.4f}"
        )


# ========== the seam ==========


def _record(
    paths: config.CasePaths,
    pass_name: str,
    result: PassResult,
    sha: str | None,
    prompt_version: str | None = None,
) -> None:
    """Upsert a pass's findings, and reconcile only if it finished.

    An incomplete pass must never reconcile. A budget cutoff, a truncated
    sweep, a missing credential — at this layer each is indistinguishable from
    every unreached finding having been fixed, so resolving them would quietly
    erase real findings and report it as progress.
    """
    store.upsert_many(paths, result.findings, verified_at_sha=sha, prompt_version=prompt_version)
    if not result.complete:
        log.warning("orchestrator: %s incomplete — skipping reconciliation", pass_name)
        return
    current = {store.make_id(f["pass"], f["subject"], f["claim"]) for f in result.findings}
    store.reconcile_pass(paths, pass_name, current)


def _current_sha() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return out.stdout.strip() or None


# ========== the cycle ==========


def run_cycle(
    case_id: str,
    dossier_dir: Path | None = None,
    budget: budget_mod.Budget | None = None,
    write_outbound: bool = False,
    local_model: str | None = None,
) -> CycleReport:
    """Run every pass over one case and rewrite its digest."""
    started = time.time()
    paths = config.CasePaths.for_case(case_id, dossier_dir=dossier_dir)
    paths.investigation_dir.mkdir(parents=True, exist_ok=True)

    case_graph = graph_mod.load_graph(case_id, dossier_dir=dossier_dir)
    case_catalog = catalog_mod.load(case_id, dossier_dir=dossier_dir)
    health = extract.graph_health(case_graph)

    ctx = RunContext(
        case_id=case_id,
        paths=paths,
        graph=case_graph,
        catalog=case_catalog,
        budget=budget or budget_mod.Budget(),
        health=health,
        sha=_current_sha(),
        local_model=local_model,
    )
    if local_model:
        log.info("orchestrator: local adjudication enabled (%s)", local_model)

    report = CycleReport(
        case_id=case_id, obligations=len(case_catalog), facts=len(case_graph.facts)
    )

    for pass_name, run_pass, version_key in PASSES:
        result = run_pass(ctx)
        _record(
            paths,
            pass_name,
            result,
            ctx.sha,
            config.PROMPT_VERSIONS.get(version_key) if version_key else None,
        )
        report.per_pass[pass_name] = (len(result.findings), result.complete)

    final = store.load_all(paths)
    report.open_findings = sum(1 for f in final.values() if f.get("status") == "open")
    report.digest_path = render.render_digest(paths, final, case_catalog)
    report.brief_path = render_brief(paths, final, case_graph, case_catalog)
    report.brief_path = report_mod_render(paths, final, case_graph, case_catalog)
    if write_outbound:
        render.render_outbound(paths, case_id, final, case_catalog)
    report.budget = ctx.budget.snapshot()
    report.elapsed_s = time.time() - started
    return report


# ========== CLI ==========


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Plane V investigation cycle.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="enable the local-model adjudication tiers (Ollama on localhost). "
        "Deterministic verdicts are computed first either way; a model can only "
        "move each one in the single direction its pass documents.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"override the local model for every tier, e.g. {config.OLLAMA_MODEL_JUDGMENT}. "
        f"Default: {config.OLLAMA_MODEL_RESCUE} for the high-volume rescue tier, "
        f"{config.OLLAMA_MODEL_JUDGMENT} for comparative judgment.",
    )
    parser.add_argument(
        "--local-calls",
        type=int,
        default=config.LOCAL_CALLS_PER_CYCLE,
        help="per-cycle cap on local model calls; the cap is what makes a cycle "
        "terminate, and hitting it marks the pass partial rather than resolving "
        "what it never reached.",
    )
    parser.add_argument(
        "--outbound",
        action="store_true",
        help="also write the gated outbound extract (refused for a case that is "
        "not in ALLOW_OUTBOUND_CASES)",
    )
    args = parser.parse_args()

    local_model = None
    if args.local_llm or args.model:
        local_model = args.model or ollama.AUTO
        if not ollama.is_available():
            parser.error(
                "Ollama is not reachable at "
                f"{config.OLLAMA_URL} — start it, or drop --local-llm to run "
                "the deterministic tier only."
            )

    report = run_cycle(
        args.case_id,
        write_outbound=args.outbound,
        budget=budget_mod.Budget(local_calls=args.local_calls),
        local_model=local_model,
    )
    print(report.summary())
    print(f"digest:  {report.digest_path}")
    print(f"rapport: {report.brief_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
