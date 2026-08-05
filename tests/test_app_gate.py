"""Dossier access gate (ADR #58).

The authorisation decision is deliberately pure (_resolve_active_case_id,
_filter_accessible) so it can be tested without a Streamlit runtime. These
are the two enforcement points; both must independently deny while locked.
"""
from app.streamlit_app import (
    NO_CASE_LABEL,
    PUBLIC_CASE_IDS,
    _case_is_public,
    _filter_accessible,
    _resolve_active_case_id,
)


# ========== enforcement point 2: active_case_id for the chat ==========

def test_locked_session_never_yields_a_protected_case() -> None:
    """The load-bearing assertion: with no case id reaching flow.run, the
    ADR #42 router can only resolve source_scope="statute", so a locked
    session cannot retrieve dossier text no matter what it asks."""
    assert _resolve_active_case_id("private", unlocked=False) is None


def test_locked_session_may_still_use_a_public_case() -> None:
    assert _resolve_active_case_id("demo", unlocked=False) == "demo"


def test_unlocked_session_yields_the_protected_case() -> None:
    assert _resolve_active_case_id("private", unlocked=True) == "private"


def test_no_selection_yields_none() -> None:
    for value in (None, "", NO_CASE_LABEL):
        assert _resolve_active_case_id(value, unlocked=True) is None


def test_stale_protected_selection_is_denied_after_relock() -> None:
    """Re-locking must revoke a case already chosen while unlocked — the
    widget value survives the rerun, so the gate cannot trust it."""
    assert _resolve_active_case_id("private", unlocked=True) == "private"
    assert _resolve_active_case_id("private", unlocked=False) is None


# ========== enforcement point 1: case list for the compliance panel ==========

def test_locked_session_sees_only_public_cases() -> None:
    """The D8 panel reads facts and matrices from disk directly, so gating
    only the chat would leave the larger surface open."""
    assert _filter_accessible(["private", "demo", "acme"], unlocked=False) == ["demo"]


def test_unlocked_session_sees_every_case() -> None:
    cases = ["private", "demo", "acme"]
    assert _filter_accessible(cases, unlocked=True) == cases


def test_unknown_cases_are_protected_by_default() -> None:
    """A new case directory must be protected without anyone remembering to
    add it to a denylist — only the explicit PUBLIC_CASE_IDS allowlist opens
    a case up.

    ADR #66 added vitrine to the allowlist. That is a statement of fact — it
    is anonymised and committed (ADR #65) — and deliberately does NOT relax
    the rule this test guards: an unnamed case is still closed.
    """
    assert _filter_accessible(["brand_new_client"], unlocked=False) == []
    assert not _case_is_public("brand_new_client")
    assert PUBLIC_CASE_IDS == {"demo", "vitrine"}


# ========== session-owned cases (ADR #66) ==========

def test_session_owned_case_is_reachable_by_its_creator() -> None:
    """A dossier created through the app must be usable immediately.

    Its creator has no passphrase — on a fresh checkout none is even
    configured — so without this the new-dossier flow would build a case the
    user could never open.
    """
    assert _resolve_active_case_id("mine", unlocked=False, owned={"mine"}) == "mine"
    assert _filter_accessible(["mine", "private"], unlocked=False, owned={"mine"}) == ["mine"]


def test_session_owned_case_is_not_reachable_by_another_session() -> None:
    """Ownership is per-session state and never persisted, so a second
    visitor sees the same case as protected."""
    assert _resolve_active_case_id("mine", unlocked=False, owned=set()) is None
    assert _filter_accessible(["mine"], unlocked=False, owned=set()) == []


def test_ownership_cannot_launder_the_protected_case() -> None:
    """Ownership grants access to what this session created, not to anything
    it names — `private` is protected regardless of what the set claims."""
    assert _filter_accessible(["private"], unlocked=False, owned={"mine"}) == []


# ========== case id validation for the new-dossier flow (ADR #66) ==========

def test_case_id_slug_rejects_traversal_and_bad_shapes() -> None:
    """case_id becomes a directory name AND the `dossier-<id>-` chunk-id
    prefix, so anything outside the slug set either escapes data/dossier/ or
    corrupts scope parsing."""
    from app.streamlit_app import _valid_case_id

    for good in ("vitrine", "dupont_2026", "a1", "cas-01"):
        assert _valid_case_id(good), good
    for bad in ("../etc/passwd", "Dupont", "a", "has space", "", "x" * 70, "-lead", "/abs"):
        assert not _valid_case_id(bad), bad
