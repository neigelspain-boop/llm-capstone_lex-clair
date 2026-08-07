"""Leave-it-running entry point — the "as long as it needs" surface.

Polls the case's artifacts rather than `git rev-parse HEAD`:
`scripts/local_audit/watch.py`'s subject is code, this plane's is data. A cycle
fires when any watched artifact's content hash moves, when a catalog changes,
when the statute corpus changes, or on the daily full-sweep timer.

Nothing here makes the loop resumable — that property comes from the store and
the cache, and this file only decides *when* to start a cycle. A `SIGKILL` at
any point costs at most one unit of work, because every verdict is committed
before the next begins and finding ids carry no timestamp.

The backlog behaviour is the part that matters and it needs no machinery: while
the last cycle reported an incomplete pass, the loop does not sleep. Phase 1's
deterministic cycle finishes in under a second, so this only bites once the
budgeted Phase 2 layers land — which is exactly when it should.

Usage:
    uv run python -m investigator.watch --case-id vitrine
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from investigator import config, orchestrator
from investigator.cache import content_hash_for_file

log = logging.getLogger(__name__)


# ========== state ==========


def _load_state(paths: config.CasePaths) -> dict:
    if paths.watch_state.exists():
        try:
            return json.loads(paths.watch_state.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("watch: unreadable state file, starting fresh")
    return {}


def _save_state(paths: config.CasePaths, state: dict) -> None:
    paths.investigation_dir.mkdir(parents=True, exist_ok=True)
    paths.watch_state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def artifact_hashes(paths: config.CasePaths) -> dict[str, str]:
    """Content hash of everything a cycle's verdicts depend on.

    A missing artifact hashes to `"missing"` rather than being omitted, so a
    file appearing or disappearing is itself a change that triggers a cycle.
    """
    hashes = {
        name: content_hash_for_file(paths.case_dir / name)
        for name in config.WATCHED_CASE_ARTIFACTS
    }
    hashes["catalog:generic"] = content_hash_for_file(config.GENERIC_CATALOG)
    hashes["corpus:chunks.csv"] = content_hash_for_file(config.STATUTE_CHUNKS_CSV)
    return hashes


# ========== the loop ==========


def loop(case_id: str, dossier_dir: Path | None = None, once: bool = False) -> None:
    paths = config.CasePaths.for_case(case_id, dossier_dir=dossier_dir)
    state = _load_state(paths)

    while True:
        now = time.time()
        current = artifact_hashes(paths)
        stale = current != state.get("artifact_hashes")
        due = now - state.get("last_full_sweep_ts", 0) > config.FULL_SWEEP_INTERVAL_SECONDS
        backlog = bool(state.get("incomplete_passes"))

        if stale or due or backlog or "artifact_hashes" not in state:
            reason = (
                "artefacts modifiés" if stale
                else "reliquat du cycle précédent" if backlog
                else "balayage périodique"
            )
            log.info("watch: cycle (%s)", reason)
            report = orchestrator.run_cycle(case_id, dossier_dir=dossier_dir)
            print(report.summary())
            state = {
                "case_id": case_id,
                "cycle_seq": state.get("cycle_seq", 0) + 1,
                "last_cycle_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "last_full_sweep_ts": now,
                "artifact_hashes": current,
                "incomplete_passes": [
                    name for name, (_, complete) in report.per_pass.items() if not complete
                ],
                "open_findings": report.open_findings,
                "budget": report.budget,
            }
            _save_state(paths, state)

        if once:
            return
        if state.get("incomplete_passes"):
            # Work remains that this cycle could not reach. Do not sleep — the
            # cache makes the next pass cheap and the budget bounds it again.
            continue
        time.sleep(config.WATCH_POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plane V continuously for one case.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args()
    loop(args.case_id, once=args.once)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("watch: starting, polling every %ds", config.WATCH_POLL_SECONDS)
    try:
        main()
    except KeyboardInterrupt:
        log.info("watch: stopped")
