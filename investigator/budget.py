"""Per-cycle and per-session spending limits.

Two kinds of scarcity, deliberately counted separately. Local model calls cost
nothing in USD but seconds each, so their cap is what makes a cycle *terminate*
— without it a cold start on a large case runs until something is killed.
Cloud calls cost money, so they carry both a call cap and two USD ceilings,
mirroring `rag/compliance.py`'s dry-run/real-run pair.

Rates are never defined here. `ingestion.clients.estimate_cost_usd` is the only
source (ADR #68) and it raises `KeyError` on an unlisted model rather than
returning 0.0, so a typo in a model slug surfaces as a failure instead of a
silently free run.

Phase 1 calls no model at all, so nothing here is exercised beyond the PISTE
counter. It ships now because the pass contracts already reference
`may_escalate`, and a budget bolted on after the fact tends to have a path
around it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ingestion.clients import estimate_cost_usd, extract_usage
from investigator import config
from investigator.schema import TIER_ORDER

log = logging.getLogger(__name__)

# Tiers strong enough to be worth spending cloud money attacking. A T3 finding
# is an absence over a covered scope — real, but the confounders that matter
# for it are usually the authored ones, not generated prose.
ESCALATION_TIERS = frozenset({"T1", "T2"})


@dataclass
class Budget:
    """Mutable per-cycle counters. One instance per `run_cycle`."""

    local_calls: int = config.LOCAL_CALLS_PER_CYCLE
    cloud_calls: int = config.CLOUD_CALLS_PER_CYCLE
    piste_calls: int = config.PISTE_CALLS_PER_CYCLE
    usd_cycle_ceiling: float = config.USD_CYCLE_CEILING
    usd_session_ceiling: float = config.USD_SESSION_CEILING
    usd_spent_cycle: float = 0.0
    usd_spent_session: float = 0.0
    cloud_enabled: bool = False

    _local_used: int = field(default=0, repr=False)
    _cloud_used: int = field(default=0, repr=False)
    _piste_used: int = field(default=0, repr=False)

    # ========== call counters ==========

    def local_exhausted(self) -> bool:
        return self._local_used >= self.local_calls

    def piste_exhausted(self) -> bool:
        return self._piste_used >= self.piste_calls

    def take_local(self) -> bool:
        """Claim one local call. False when the cap is reached.

        A caller that gets False must return `complete=False` from its pass, or
        the cutoff is indistinguishable from every remaining finding having
        been fixed.
        """
        if self.local_exhausted():
            return False
        self._local_used += 1
        return True

    def take_piste(self) -> bool:
        if self.piste_exhausted():
            return False
        self._piste_used += 1
        return True

    # ========== cloud ==========

    def may_escalate(self, finding: dict) -> bool:
        """Whether this finding is worth a paid call.

        Every condition must hold: cloud enabled at all, the finding is
        externalisable, its tier is strong enough to be worth attacking, and
        both the call cap and both USD ceilings have room.
        """
        if not self.cloud_enabled:
            return False
        if not finding.get("externalisable", False):
            return False
        if finding.get("tier") not in ESCALATION_TIERS:
            return False
        if self._cloud_used >= self.cloud_calls:
            return False
        return (
            self.usd_spent_cycle < self.usd_cycle_ceiling
            and self.usd_spent_session < self.usd_session_ceiling
        )

    def charge_cloud(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Pre-flight estimate, booked before the call is made.

        Booking first means a crash mid-call cannot make a cycle look cheaper
        than it was. `settle_cloud` corrects the estimate afterwards.
        """
        cost = estimate_cost_usd(model_id, prompt_tokens, completion_tokens)
        self._cloud_used += 1
        self.usd_spent_cycle += cost
        self.usd_spent_session += cost
        return cost

    def settle_cloud(self, response, model_id: str, estimated: float) -> float:
        """Replace an estimate with the provider's own reported cost."""
        usage = extract_usage(response, model_id)
        actual = usage.get("cost_usd", estimated)
        delta = actual - estimated
        self.usd_spent_cycle += delta
        self.usd_spent_session += delta
        return actual

    # ========== reporting ==========

    def snapshot(self) -> dict:
        return {
            "local_used": self._local_used,
            "cloud_used": self._cloud_used,
            "piste_used": self._piste_used,
            "usd_spent_cycle": round(self.usd_spent_cycle, 6),
            "usd_spent_session": round(self.usd_spent_session, 6),
        }


def strongest_first(findings: list[dict]) -> list[dict]:
    """Order findings so the scarcest budget is spent on the best evidence."""
    return sorted(findings, key=lambda f: (TIER_ORDER.get(f.get("tier", "T5"), 99), f["id"]))
