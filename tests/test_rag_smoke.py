"""Plane II fast smoke tests: query router (Day B, Deliverable B2, ADR #42).

Router unit tests mock rag.router.get_openrouter_client so no network call
happens. Flow integration tests mock rag.flow.route_query, rag.flow.rewrite,
rag.flow.retrieve, rag.flow.generate — same monkeypatch style as the B1
tests in tests/test_ingestion_smoke.py.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


# ========== helpers ==========

def _mock_openrouter_client(content: str) -> MagicMock:
    """Build a MagicMock client whose chat.completions.create(...) returns
    a response with the given raw text as message.content."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = response
    return client


def _classifier_json(intent: str, confidence: str = "high", rationale: str = "test") -> str:
    return json.dumps({"intent": intent, "confidence": confidence, "rationale": rationale})


# ========== router unit tests ==========

def test_router_statute_lookup(monkeypatch) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client(_classifier_json("statute_lookup")),
    )

    decision = router.route_query("qu'est-ce que le quasi-usufruit ?", None)

    assert decision.intent == "statute_lookup"
    assert decision.source_scope == "statute"


def test_router_case_factual_with_case_id(monkeypatch) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client(_classifier_json("case_factual")),
    )

    decision = router.route_query("quand le notaire a-t-il envoye la mise en demeure ?", "private")

    assert decision.intent == "case_factual"
    assert decision.source_scope == "case:private"


def test_router_case_factual_without_case_id_downgrades(monkeypatch) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client(_classifier_json("case_factual")),
    )

    decision = router.route_query("quand le notaire a-t-il envoye la mise en demeure ?", None)

    assert decision.source_scope == "statute"
    assert decision.confidence == "low"
    assert "case_factual" in decision.rationale or "dossier actif" in decision.rationale


def test_router_gap_analysis_with_case_id(monkeypatch) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client(_classifier_json("gap_analysis")),
    )

    decision = router.route_query("le notaire a-t-il manque a son obligation ?", "private")

    assert decision.intent == "gap_analysis"
    assert decision.source_scope == "blended"


def test_router_gap_analysis_without_case_id_downgrades(monkeypatch) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client(_classifier_json("gap_analysis")),
    )

    decision = router.route_query("le notaire a-t-il manque a son obligation ?", None)

    assert decision.source_scope == "statute"
    assert decision.confidence == "low"


def test_router_other_defaults_statute(monkeypatch) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client(_classifier_json("other")),
    )

    decision = router.route_query("bonjour, comment ca va ?", None)

    assert decision.intent == "other"
    assert decision.source_scope == "statute"


def test_router_parse_failure_returns_safe_default(monkeypatch, caplog) -> None:
    from rag import router

    monkeypatch.setattr(
        router, "get_openrouter_client",
        lambda: _mock_openrouter_client("this is not json"),
    )

    with caplog.at_level("WARNING"):
        decision = router.route_query("qu'est-ce que le quasi-usufruit ?", None)

    assert decision.intent == "other"
    assert decision.source_scope == "statute"
    assert decision.confidence == "low"
    assert any("router" in rec.message for rec in caplog.records)


# ========== flow integration tests ==========

def test_flow_run_uses_router_when_scope_none(monkeypatch) -> None:
    from rag import flow
    from rag.router import RouteDecision

    mock_decision = RouteDecision(
        intent="statute_lookup", source_scope="statute", confidence="high", rationale="test",
    )
    mock_route_query = MagicMock(return_value=mock_decision)
    mock_retrieve = MagicMock(return_value=[])
    monkeypatch.setattr(flow, "route_query", mock_route_query)
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", mock_retrieve)
    monkeypatch.setattr(
        flow.generate, "generate",
        lambda p: ("mocked answer", {"prompt_tokens": 0, "completion_tokens": 0}),
    )

    result = flow.run("Qu'est-ce que le quasi-usufruit ?", source_scope=None)

    mock_route_query.assert_called_once()
    assert mock_retrieve.call_args.kwargs["source_scope"] == "statute"
    assert result["route_decision"]["source_scope"] == "statute"


def test_flow_run_skips_router_when_scope_provided(monkeypatch) -> None:
    from rag import flow

    mock_route_query = MagicMock()
    mock_retrieve = MagicMock(return_value=[])
    monkeypatch.setattr(flow, "route_query", mock_route_query)
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", mock_retrieve)
    monkeypatch.setattr(
        flow.generate, "generate",
        lambda p: ("mocked answer", {"prompt_tokens": 0, "completion_tokens": 0}),
    )

    result = flow.run("Qu'est-ce que le quasi-usufruit ?", source_scope="blended")

    mock_route_query.assert_not_called()
    assert mock_retrieve.call_args.kwargs["source_scope"] == "blended"
    assert result["route_decision"]["intent"] == "override"
    assert result["route_decision"]["source_scope"] == "blended"
