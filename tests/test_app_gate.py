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
    a case up."""
    assert _filter_accessible(["brand_new_client"], unlocked=False) == []
    assert not _case_is_public("brand_new_client")
    assert PUBLIC_CASE_IDS == {"demo"}
