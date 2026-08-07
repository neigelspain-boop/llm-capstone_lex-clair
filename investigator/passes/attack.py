"""Adversarial pass: attach the counter-narrative to every strong finding.

The discipline this encodes is the one that makes a hand-built case file
trustworthy — every conclusion carries the best argument against it, recorded
next to it rather than discovered later by the other side.

Phase 1 attaches each obligation's authored `confounders_seed`, with zero model
calls. For a well-authored catalog that is most of the value: the person who
wrote the obligation is the person best placed to say how its apparent breach
could be innocent. Phase 2 adds generated confounders on top of the same seeds.

This pass reads the **store**, not the graph, so it must run last. It is also
the only pass that works by side effect: it emits no findings of its own in
Phase 1, and instead writes onto the findings the other passes just recorded.
`store.attach` touches only fields outside `upsert_many`'s update set, so a
later `check` re-run cannot erase the result — the same mechanism that
preserves `first_seen`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from investigator import store
from investigator.schema import TIER_ORDER, PassResult, RunContext

log = logging.getLogger(__name__)

PASS_NAME = "attack"

# Findings weak enough that a counter-narrative adds nothing: a T5 hypothesis is
# already flagged as unreliable, and a confounder would only dress it up.
MIN_TIER_TO_ATTACK = "T4"

ATTACKABLE_PASSES = frozenset({"check", "contradict"})


def select_targets(store_: dict[str, dict], min_tier: str = MIN_TIER_TO_ATTACK) -> list[dict]:
    """Open findings worth arguing against, in a stable order."""
    limit = TIER_ORDER[min_tier]
    targets = [
        f
        for f in store_.values()
        if f.get("status") == "open"
        and f.get("pass") in ATTACKABLE_PASSES
        and TIER_ORDER.get(f.get("tier", "T5"), 99) <= limit
    ]
    return sorted(targets, key=lambda f: (TIER_ORDER[f["tier"]], f["id"]))


def run(ctx: RunContext) -> PassResult:
    """Seed confounders onto every attackable finding.

    Idempotent: `store.attach` deduplicates on `text_fr`, so re-running adds
    nothing and the calibration log does not grow on a no-op cycle.
    """
    current = store.load_all(ctx.paths)
    targets = select_targets(current)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    attached = 0

    for finding in targets:
        obligation = ctx.catalog.get(finding.get("obligation_id") or "")
        if obligation is None or not obligation.confounders_seed:
            continue
        already = {c.get("text_fr") for c in finding.get("confounders", [])}
        seeds = [
            {
                "text_fr": text,
                "source": "catalog_seed",
                "dispositive": False,
                "added_at": now,
            }
            for text in obligation.confounders_seed
            if text not in already
        ]
        if not seeds:
            continue
        ok = store.attach(
            ctx.paths,
            finding["id"],
            confounders=seeds,
            calibration_entry={
                "ts": now,
                "tier": finding["tier"],
                "confidence": finding.get("confidence", "low"),
                "reason": f"{len(seeds)} contre-argument(s) du catalogue attaché(s)",
                "actor": PASS_NAME,
            },
        )
        if ok:
            attached += 1

    log.info("attack: seeded confounders on %d of %d target(s)", attached, len(targets))
    # No findings of its own in Phase 1. Reconciliation over an empty list is
    # correct and a no-op: this pass has never emitted anything to resolve.
    return PassResult(findings=[], complete=True)
