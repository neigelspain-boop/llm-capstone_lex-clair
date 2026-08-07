"""Leave-it-running-nonstop entry point.

Polls `git rev-parse HEAD` on an interval; on a new commit, diffs against
the last processed SHA and runs an incremental cycle scoped to what changed.
A full sweep (all passes, full scope) still runs at least once a day
regardless of diffs, since duplication candidates can go stale without any
single file changing (a sibling function elsewhere in the embedding
neighborhood may have moved).

Usage:
    uv run python -m scripts.local_audit.watch
"""
from __future__ import annotations

import json
import logging
import subprocess
import time

from scripts.local_audit import config, orchestrator

log = logging.getLogger(__name__)


def _load_state() -> dict:
    if config.WATCH_STATE.exists():
        try:
            return json.loads(config.WATCH_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(state: dict) -> None:
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    config.WATCH_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _current_sha() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    return out.stdout.strip() or None


def _changed_files(old_sha: str, new_sha: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", old_sha, new_sha, "--",
         "*.py", "CLAUDE.md", "docs/*.md"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def loop() -> None:
    state = _load_state()

    if "last_sha" not in state:
        log.info("watch: no prior state — running initial full sweep")
        sha = orchestrator.run_full_sweep()
        state = {"last_sha": sha, "last_full_sweep_ts": time.time()}
        _save_state(state)

    while True:
        time.sleep(config.WATCH_POLL_SECONDS)

        now = time.time()
        if now - state.get("last_full_sweep_ts", 0) > config.FULL_SWEEP_INTERVAL_SECONDS:
            log.info("watch: daily full-sweep timer elapsed")
            sha = orchestrator.run_full_sweep()
            state["last_sha"] = sha
            state["last_full_sweep_ts"] = now
            _save_state(state)
            continue

        current = _current_sha()
        if current is None or current == state.get("last_sha"):
            continue

        changed = _changed_files(state["last_sha"], current)
        log.info("watch: HEAD moved %s -> %s (%d file(s) changed)", state["last_sha"], current, len(changed))
        orchestrator.run_incremental(changed)
        state["last_sha"] = current
        _save_state(state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("watch: starting, polling every %ds, full sweep every %ds",
              config.WATCH_POLL_SECONDS, config.FULL_SWEEP_INTERVAL_SECONDS)
    try:
        loop()
    except KeyboardInterrupt:
        log.info("watch: stopped")
