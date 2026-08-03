"""Plane II fast smoke tests: query router (Day B, Deliverable B2, ADR #42),
compliance matrix generator (Day B, Deliverable B3, ADR #43), cross-role
context annotation (Day B, Deliverable C1, ADR #44), the answer model
catalog (Day C, Deliverable C2, ADR #45), and the D6 verify-conclude
prompt + person-integration plumbing + compliance-run cache + dry-run cost
gate (ADR #53).

Router unit tests mock rag.router.get_openrouter_client so no network call
happens. Flow integration tests mock rag.flow.route_query, rag.flow.rewrite,
rag.flow.retrieve, rag.flow.generate — same monkeypatch style as the B1
tests in tests/test_ingestion_smoke.py. Compliance tests mock
rag.compliance.get_openrouter_client and rag.compliance.retrieve, and use a
synthetic tmp_path fixture case (data/dossier/demo/*.jsonl are empty, so
there's no real demo case to run an end-to-end test against). Generate unit
tests mock rag.generate.get_openrouter_client the same way the router tests
do.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.dossier.facts import ActorRole, Fact, RoleAmbiguity


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


def _mock_generate(prompt: str, model_key: str | None = None) -> tuple[str, dict]:
    """Stand-in for rag.generate.generate used by flow tests — mirrors the
    real usage dict shape (ADR #45) so flow.run()'s tokens["model_id"] /
    tokens["cost_usd"] / tokens["model_key"] reads don't KeyError."""
    key = model_key or "gpt-4o-mini"
    from rag.generate import ANSWER_MODELS

    return "mocked answer", {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "model_id": ANSWER_MODELS[key]["model_id"],
        "model_key": key,
    }


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
    monkeypatch.setattr(flow.generate, "generate", _mock_generate)

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
    monkeypatch.setattr(flow.generate, "generate", _mock_generate)

    result = flow.run("Qu'est-ce que le quasi-usufruit ?", source_scope="blended")

    mock_route_query.assert_not_called()
    assert mock_retrieve.call_args.kwargs["source_scope"] == "blended"
    assert result["route_decision"]["intent"] == "override"
    assert result["route_decision"]["source_scope"] == "blended"


# ========== answer model catalog tests (C2, ADR #45) ==========

def test_flow_run_default_answer_model_is_gpt4o_mini(monkeypatch) -> None:
    from rag import flow

    mock_generate = MagicMock(side_effect=_mock_generate)
    monkeypatch.setattr(flow, "route_query", MagicMock())
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(flow.generate, "generate", mock_generate)

    flow.run("Qu'est-ce que le quasi-usufruit ?", source_scope="statute")

    assert mock_generate.call_args.kwargs["model_key"] == "gpt-4o-mini"


def test_flow_run_passes_answer_model_through(monkeypatch) -> None:
    from rag import flow

    mock_generate = MagicMock(side_effect=_mock_generate)
    monkeypatch.setattr(flow, "route_query", MagicMock())
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(flow.generate, "generate", mock_generate)

    flow.run(
        "Qu'est-ce que le quasi-usufruit ?",
        source_scope="statute",
        answer_model="opus-4.7",
    )

    assert mock_generate.call_args.kwargs["model_key"] == "opus-4.7"


def test_flow_run_return_dict_includes_answer_model_key(monkeypatch) -> None:
    from rag import flow

    monkeypatch.setattr(flow, "route_query", MagicMock())
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(flow.generate, "generate", _mock_generate)

    result = flow.run(
        "Qu'est-ce que le quasi-usufruit ?",
        source_scope="statute",
        answer_model="kimi-k3",
    )

    assert result["answer_model_key"] == "kimi-k3"
    assert result["model_used"] == "moonshotai/kimi-k3"


def test_generate_invalid_model_key_raises() -> None:
    from rag.generate import generate

    with pytest.raises(ValueError, match="unknown model_key"):
        generate("prompt", model_key="not-a-real-model")


def test_generate_default_model_uses_gpt4o_mini(monkeypatch) -> None:
    from rag import generate

    client = _mock_openrouter_client("answer text")
    client.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50,
    )
    monkeypatch.setattr(generate, "get_openrouter_client", lambda: client)

    generate.generate("prompt")

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert "extra_body" not in call_kwargs


def test_generate_opus_uses_reasoning_effort(monkeypatch) -> None:
    from rag import generate

    client = _mock_openrouter_client("answer text")
    client.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50,
    )
    monkeypatch.setattr(generate, "get_openrouter_client", lambda: client)

    generate.generate("prompt", model_key="opus-4.7")

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-opus-4.7"
    assert call_kwargs["extra_body"] == {"reasoning": {"effort": "max"}}


def test_generate_kimi_uses_reasoning_effort(monkeypatch) -> None:
    from rag import generate

    client = _mock_openrouter_client("answer text")
    client.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50,
    )
    monkeypatch.setattr(generate, "get_openrouter_client", lambda: client)

    generate.generate("prompt", model_key="kimi-k3")

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "moonshotai/kimi-k3"
    assert call_kwargs["extra_body"] == {"reasoning": {"effort": "max"}}


def test_generate_cost_calculation_from_catalog(monkeypatch) -> None:
    from rag import generate

    prompt_tokens, completion_tokens = 1000, 500

    for model_key, cfg in generate.ANSWER_MODELS.items():
        client = _mock_openrouter_client("answer text")
        client.chat.completions.create.return_value.usage = MagicMock(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )
        monkeypatch.setattr(generate, "get_openrouter_client", lambda c=client: c)

        _, usage = generate.generate("prompt", model_key=model_key)

        expected_cost = (
            prompt_tokens * cfg["cost_input_per_m"] / 1_000_000
            + completion_tokens * cfg["cost_output_per_m"] / 1_000_000
        )
        assert usage["cost_usd"] == pytest.approx(expected_cost)
        assert usage["model_id"] == cfg["model_id"]
        assert usage["model_key"] == model_key


# ========== compliance matrix tests (B3, ADR #43) ==========

def _compliance_entries_json() -> str:
    return json.dumps([
        {
            "statute_chunk_id": "cc-587",
            "statute_excerpt": "L'usufruitier doit conserver la substance des choses.",
            "obligation_summary": "Conserver la substance des biens quasi-usufruits.",
            "status": "met",
            "evidence_fact_ids": ["doc1-f001"],
            "rationale": "Les faits montrent la remise de la convention. Aucune irrégularité constatée.",
        },
    ])


def _mock_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "cc-587", "num": "587", "titre": "Code civil", "section_path": "Livre II",
            "texte": "L'usufruitier doit conserver la substance des choses.",
            "source": "legifrance", "source_label": "Code civil", "url": "https://example.test/cc-587",
            "rrf_score": 0.9,
        },
    ]


def _mock_compliance_client(content: str, cost: float = 0.005) -> MagicMock:
    """Like _mock_openrouter_client, but with usage.prompt_tokens/
    completion_tokens/cost set to real numbers (not MagicMock attributes) —
    rag.compliance sums these across role groups, which breaks on an
    un-configured MagicMock auto-attribute."""
    client = _mock_openrouter_client(content)
    client.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50, cost=cost,
    )
    return client


def _write_fixture_case(base_dir: Path, case_id: str) -> None:
    """Write a small synthetic case (2 roles, 3 facts, 1 ambiguity) to
    base_dir/case_id/*.jsonl. data/dossier/demo/*.jsonl are empty in this
    repo, so there's no real demo case to test against."""
    case_dir = base_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    facts = [
        Fact(
            fact_id="doc1-f001", date="2024-01-10", actor_role="notaire_redacteur",
            action="recevoir une convention de quasi-usufruit", target="convention",
            verbatim_quote="Le notaire reçoit la convention de quasi-usufruit.",
            source_doc_id="doc1", source_chunk_id="dossier-testcase-doc1-c001",
        ),
        Fact(
            fact_id="doc1-f002", date="2024-01-15", actor_role="notaire_redacteur",
            action="informer les heritiers de leurs droits", target="heritiers",
            verbatim_quote="Le notaire informe les héritiers de leurs droits.",
            source_doc_id="doc1", source_chunk_id="dossier-testcase-doc1-c002",
        ),
        Fact(
            fact_id="doc2-f001", date="2024-02-01", actor_role="heritier_nu_proprietaire",
            action="signer la convention", target="convention",
            verbatim_quote="L'héritier signe la convention de quasi-usufruit.",
            source_doc_id="doc2", source_chunk_id="dossier-testcase-doc2-c001",
        ),
    ]
    roles = [
        ActorRole(
            role_id="notaire_redacteur", label_fr="Notaire rédacteur",
            grounding_note="Notaire ayant rédigé l'acte.", first_seen_doc_id="doc1",
            fact_count=2, confidence="high",
        ),
        ActorRole(
            role_id="heritier_nu_proprietaire", label_fr="Héritier nu-propriétaire",
            grounding_note="Héritier titulaire de la nue-propriété.", first_seen_doc_id="doc2",
            fact_count=1, confidence="high",
        ),
    ]
    ambiguities = [
        RoleAmbiguity(
            ambiguity_id="doc1-a001", source_doc_id="doc1",
            verbatim_quote="Maître X a signé.",
            candidate_role_ids=["notaire_associe", "notaire_stagiaire"],
            note="Statut du signataire incertain.", fact_ids=["doc1-f001"],
        ),
    ]

    with (case_dir / "facts.jsonl").open("w", encoding="utf-8") as f:
        for fact in facts:
            f.write(fact.model_dump_json() + "\n")
    with (case_dir / "actor_roles.jsonl").open("w", encoding="utf-8") as f:
        for role in roles:
            f.write(role.model_dump_json() + "\n")
    with (case_dir / "role_ambiguities.jsonl").open("w", encoding="utf-8") as f:
        for amb in ambiguities:
            f.write(amb.model_dump_json() + "\n")


def test_compliance_entry_schema_validates() -> None:
    from rag.compliance import ComplianceEntry

    entry = ComplianceEntry(
        entry_id="abc123def456",
        statute_chunk_id="cc-587",
        statute_excerpt="Le quasi-usufruitier doit conserver la substance des biens.",
        obligation_summary="L'usufruitier doit conserver la substance des biens.",
        actor_role="quasi_usufruitier",
        status="met",
        evidence_fact_ids=["doc1-f001"],
        rationale="Les faits montrent que l'usufruitier a respecté cette obligation. Aucune preuve contraire.",
    )

    assert entry.status == "met"
    assert entry.entry_id == "abc123def456"


def test_compliance_entry_id_deterministic() -> None:
    from rag.compliance import _entry_id

    id1 = _entry_id("cc-587", "notaire_redacteur")
    id2 = _entry_id("cc-587", "notaire_redacteur")

    assert id1 == id2
    assert len(id1) == 12
    assert id1 != _entry_id("cc-587", "heritier_nu_proprietaire")


def test_compliance_handles_json_fence_wrapper() -> None:
    from rag.compliance import _parse_compliance_response

    wrapped = "```json\n" + _compliance_entries_json() + "\n```"
    parsed = _parse_compliance_response(wrapped, role_id="notaire_redacteur")

    assert len(parsed) == 1
    assert parsed[0]["statute_chunk_id"] == "cc-587"


def test_compliance_handles_trailing_prose() -> None:
    from rag.compliance import _parse_compliance_response

    raw = _compliance_entries_json() + "\n\nCeci est une analyse basée sur les faits fournis."
    parsed = _parse_compliance_response(raw, role_id="notaire_redacteur")

    assert len(parsed) == 1
    assert parsed[0]["statute_chunk_id"] == "cc-587"


def test_compliance_parse_recovers_leading_entries_on_trailing_truncation(caplog) -> None:
    from rag.compliance import _parse_compliance_response

    good_entry = json.loads(_compliance_entries_json())[0]
    truncated_tail = (
        '{"statute_chunk_id": "cc-601", "statute_excerpt": "Le nu-propri\\u00e9taire doit'
    )
    raw = "[" + json.dumps(good_entry) + ",\n" + truncated_tail

    with caplog.at_level("WARNING"):
        parsed = _parse_compliance_response(raw, role_id="notaire_redacteur")

    assert len(parsed) == 1
    assert parsed[0]["statute_chunk_id"] == "cc-587"
    assert any("partial parse recovered 1" in rec.message for rec in caplog.records)


def test_compliance_matrix_idempotent(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(
        compliance, "get_openrouter_client",
        lambda: _mock_compliance_client(_compliance_entries_json()),
    )
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())
    fixed_time = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(compliance, "_utcnow", lambda: fixed_time)

    compliance.generate_compliance_matrix("testcase")
    out_path = tmp_path / "testcase" / "compliance_matrix.json"
    content1 = out_path.read_bytes()

    compliance.generate_compliance_matrix("testcase")
    content2 = out_path.read_bytes()

    assert content1 == content2


def test_compliance_matrix_end_to_end_demo_with_mocked_llm(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "demo")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(
        compliance, "get_openrouter_client",
        lambda: _mock_compliance_client(_compliance_entries_json()),
    )
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.generate_compliance_matrix("demo")

    out_path = tmp_path / "demo" / "compliance_matrix.json"
    assert out_path.exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    loaded = compliance.ComplianceMatrix.model_validate(data)

    assert loaded.total_entries == len(loaded.entries)
    assert len(loaded.entries) == 2  # one entry per role group in the fixture
    entry_ids = [e.entry_id for e in loaded.entries]
    assert len(entry_ids) == len(set(entry_ids))
    assert loaded.unresolved_ambiguities == 1
    for entry in loaded.entries:
        assert entry.status in {"met", "breached", "ambiguous", "insufficient_evidence"}


def test_compliance_cli_dry_run_no_api_calls(tmp_path, monkeypatch, capsys) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)

    def _fail_if_called():
        raise AssertionError("get_openrouter_client must not be called in --dry-run")

    monkeypatch.setattr(compliance, "get_openrouter_client", _fail_if_called)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())
    monkeypatch.setattr("sys.argv", ["rag.compliance", "--case-id", "testcase", "--dry-run"])

    compliance.main()

    captured = capsys.readouterr()
    assert "compliance dry-run" in captured.out
    assert "est_prompt_tokens" in captured.out
    assert not (tmp_path / "testcase" / "compliance_matrix.json").exists()


def test_compliance_cli_limit_caps_role_groups(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(
        compliance, "get_openrouter_client",
        lambda: _mock_compliance_client(_compliance_entries_json()),
    )
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    call_log: list[str] = []
    original_call = compliance._call_compliance_llm

    def _tracking_call(role_id, *args, **kwargs):
        call_log.append(role_id)
        return original_call(role_id, *args, **kwargs)

    monkeypatch.setattr(compliance, "_call_compliance_llm", _tracking_call)
    monkeypatch.setattr("sys.argv", ["rag.compliance", "--case-id", "testcase", "--limit", "1"])

    compliance.main()

    assert len(call_log) == 1


# ========== cross-role context tests (C1, ADR #44) ==========

def _write_fixture_case_cross_role(base_dir: Path, case_id: str) -> None:
    """Two role clusters sharing a source_doc_id, so the C1 cross-role block
    is triggered — unlike _write_fixture_case, whose two roles sit on
    different docs (doc1 vs doc2) and so never trigger it."""
    case_dir = base_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    facts = [
        Fact(
            fact_id="doc1-f001", date="2024-01-10", actor_role="heritier_nu_proprietaire",
            action="signer la convention", target="convention",
            verbatim_quote="L'héritier nu-propriétaire signe la convention.",
            source_doc_id="doc1", source_chunk_id="dossier-crosscase-doc1-c001",
        ),
        Fact(
            fact_id="doc1-f002", date="2024-01-12", actor_role="heritier_representation",
            action="recevoir notification", target="notification",
            verbatim_quote="L'héritier par représentation reçoit notification.",
            source_doc_id="doc1", source_chunk_id="dossier-crosscase-doc1-c002",
        ),
    ]
    roles = [
        ActorRole(
            role_id="heritier_nu_proprietaire", label_fr="Héritier nu-propriétaire",
            grounding_note="Héritier titulaire de la nue-propriété.", first_seen_doc_id="doc1",
            fact_count=1, confidence="high",
        ),
        ActorRole(
            role_id="heritier_representation", label_fr="Héritier par représentation",
            grounding_note="Héritier venant en représentation d'un héritier prédécédé.",
            first_seen_doc_id="doc1", fact_count=1, confidence="high",
        ),
    ]

    with (case_dir / "facts.jsonl").open("w", encoding="utf-8") as f:
        for fact in facts:
            f.write(fact.model_dump_json() + "\n")
    with (case_dir / "actor_roles.jsonl").open("w", encoding="utf-8") as f:
        for role in roles:
            f.write(role.model_dump_json() + "\n")


def test_extract_cross_role_context_no_shared_docs_returns_empty() -> None:
    from rag.compliance import _extract_cross_role_context

    cluster_facts = [
        Fact(
            fact_id="doc_x-f001", date=None, actor_role="role_a", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]
    all_facts = cluster_facts + [
        Fact(
            fact_id="doc_y-f001", date=None, actor_role="role_b", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_y", source_chunk_id=None,
        ),
    ]

    result = _extract_cross_role_context("role_a", cluster_facts, all_facts, [])

    assert result == ""


def test_extract_cross_role_context_single_shared_role() -> None:
    from rag.compliance import _extract_cross_role_context

    cluster_facts = [
        Fact(
            fact_id="doc_x-f001", date=None, actor_role="role_a", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]
    all_facts = cluster_facts + [
        Fact(
            fact_id="doc_x-f002", date=None, actor_role="role_b", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]
    actor_roles = [
        ActorRole(
            role_id="role_b", label_fr="Rôle B", grounding_note="note",
            first_seen_doc_id="doc_x", fact_count=1, confidence="high",
        ),
    ]

    result = _extract_cross_role_context("role_a", cluster_facts, all_facts, actor_roles)

    assert "role_b" in result
    assert "Rôle B" in result


def test_extract_cross_role_context_multiple_shared_roles() -> None:
    from rag.compliance import _extract_cross_role_context

    cluster_facts = [
        Fact(
            fact_id="doc_x-f001", date=None, actor_role="role_a", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_x", source_chunk_id=None,
        ),
        Fact(
            fact_id="doc_y-f001", date=None, actor_role="role_a", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_y", source_chunk_id=None,
        ),
    ]
    all_facts = cluster_facts + [
        Fact(
            fact_id="doc_x-f002", date=None, actor_role="role_b", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_x", source_chunk_id=None,
        ),
        Fact(
            fact_id="doc_y-f002", date=None, actor_role="role_c", action="agir",
            target=None, verbatim_quote="quote", source_doc_id="doc_y", source_chunk_id=None,
        ),
    ]

    result = _extract_cross_role_context("role_a", cluster_facts, all_facts, [])

    assert "role_b" in result
    assert "role_c" in result


def test_extract_cross_role_context_caps_doc_list_at_3() -> None:
    from rag.compliance import _extract_cross_role_context

    cluster_facts = [
        Fact(
            fact_id=f"doc{i}-f001", date=None, actor_role="role_a", action="agir",
            target=None, verbatim_quote="quote", source_doc_id=f"doc{i}", source_chunk_id=None,
        )
        for i in range(1, 6)
    ]
    shared_facts = [
        Fact(
            fact_id=f"doc{i}-f002", date=None, actor_role="role_b", action="agir",
            target=None, verbatim_quote="quote", source_doc_id=f"doc{i}", source_chunk_id=None,
        )
        for i in range(1, 6)
    ]
    all_facts = cluster_facts + shared_facts

    result = _extract_cross_role_context("role_a", cluster_facts, all_facts, [])

    assert "doc1" in result and "doc2" in result and "doc3" in result
    assert "doc4" not in result and "doc5" not in result
    assert "(…)" in result


def test_build_user_message_includes_cross_role_when_present() -> None:
    from rag.compliance import _build_user_message

    facts_a = [
        Fact(
            fact_id="doc_x-f001", date="2024-01-01", actor_role="role_a",
            action="signer", target="doc", verbatim_quote="Signature.",
            source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]
    facts_b = [
        Fact(
            fact_id="doc_x-f002", date="2024-01-02", actor_role="role_b",
            action="notifier", target="doc", verbatim_quote="Notification.",
            source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]
    all_facts = facts_a + facts_b
    actor_roles = [
        ActorRole(
            role_id="role_b", label_fr="Rôle B", grounding_note="note",
            first_seen_doc_id="doc_x", fact_count=1, confidence="high",
        ),
    ]

    message = _build_user_message("role_a", "Rôle A", facts_a, _mock_chunks(), all_facts, actor_roles)

    assert "Contexte inter-rôles" in message


def test_build_user_message_omits_cross_role_when_empty() -> None:
    from rag.compliance import _build_user_message

    facts_a = [
        Fact(
            fact_id="doc_x-f001", date="2024-01-01", actor_role="role_a",
            action="signer", target="doc", verbatim_quote="Signature.",
            source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]

    message = _build_user_message("role_a", "Rôle A", facts_a, _mock_chunks(), facts_a, [])

    assert "Contexte inter-rôles" not in message


def test_generate_compliance_matrix_end_to_end_with_cross_role_mocked(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case_cross_role(tmp_path, "crosscase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)

    captured_messages: list[str] = []

    def _tracking_client() -> MagicMock:
        client = _mock_compliance_client(_compliance_entries_json())
        original_create = client.chat.completions.create

        def _create(*args, **kwargs):
            captured_messages.append(kwargs["messages"][1]["content"])
            return original_create(*args, **kwargs)

        client.chat.completions.create = _create
        return client

    monkeypatch.setattr(compliance, "get_openrouter_client", _tracking_client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.generate_compliance_matrix("crosscase")

    out_path = tmp_path / "crosscase" / "compliance_matrix.json"
    assert out_path.exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    loaded = compliance.ComplianceMatrix.model_validate(data)

    assert loaded.total_entries == len(loaded.entries)
    for entry in loaded.entries:
        assert entry.status in {"met", "breached", "ambiguous", "insufficient_evidence"}
    assert any("Contexte inter-rôles" in msg for msg in captured_messages)


# ========== D6 persons + distillation + cache + cost-gate tests (ADR #53) ==========

def test_compliance_user_message_includes_persons_when_present() -> None:
    from rag.compliance import _build_user_message

    facts_a = [
        Fact(
            fact_id="doc_x-f001", date="2024-01-01", actor_role="role_a",
            action="signer", target="doc", verbatim_quote="Signature.",
            source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]
    case_persons = [
        {"person_id": "p1", "role_assignments": ["heritier_nu_proprietaire"]},
    ]
    entities = [
        {"person_id": "p1", "canonical_name": "Marie Dupont", "aliases": []},
    ]

    message = _build_user_message(
        "role_a", "Rôle A", facts_a, _mock_chunks(), facts_a, [],
        case_persons=case_persons, entities=entities,
    )

    assert "Personnes impliquées" in message
    assert "Marie Dupont" in message


def test_compliance_user_message_includes_distilled_context_when_present() -> None:
    from rag.compliance import _build_user_message

    facts_a = [
        Fact(
            fact_id="doc_x-f001", date="2024-01-01", actor_role="role_a",
            action="signer", target="doc", verbatim_quote="Signature.",
            source_doc_id="doc_x", source_chunk_id=None,
            distilled_context="Résumé dense de la signature.",
        ),
    ]

    message = _build_user_message("role_a", "Rôle A", facts_a, _mock_chunks(), facts_a, [])

    assert 'distilled="Résumé dense de la signature."' in message


def test_compliance_user_message_omits_persons_when_empty() -> None:
    """Also the regression check for the real private-case/demo state today
    (no persons.jsonl exists anywhere, D1-D4 unshipped)."""
    from rag.compliance import _build_user_message

    facts_a = [
        Fact(
            fact_id="doc_x-f001", date="2024-01-01", actor_role="role_a",
            action="signer", target="doc", verbatim_quote="Signature.",
            source_doc_id="doc_x", source_chunk_id=None,
        ),
    ]

    message = _build_user_message("role_a", "Rôle A", facts_a, _mock_chunks(), facts_a, [])

    assert "Personnes impliquées" not in message


def test_compliance_matrix_entry_has_persons_named_field() -> None:
    """Direct unit test of _build_entries' persons_named plumbing — not
    exercised through generate_compliance_matrix end-to-end, since that path
    depends on Fact.mentioned_person_ids, which doesn't exist yet."""
    from rag.compliance import _build_entries

    raw_entries = json.loads(_compliance_entries_json())
    entries = _build_entries(
        raw_entries, "notaire_redacteur",
        persons_named=[{"person_id": "p1", "canonical_name": "Marie Dupont"}],
    )

    assert len(entries) == 1
    assert entries[0].persons_named == [{"person_id": "p1", "canonical_name": "Marie Dupont"}]


def test_compliance_prompt_falls_back_on_low_confidence_person() -> None:
    """Prompt-content check, not a code-layer behavioral test — the verdict
    downgrade on ambiguous person identity is an LLM-prompt instruction, no
    code-layer logic implements it in this deliverable."""
    from rag.compliance_prompts import COMPLIANCE_SYSTEM_PROMPT

    assert "Ne nommez PAS" in COMPLIANCE_SYSTEM_PROMPT
    assert "insufficient_evidence" in COMPLIANCE_SYSTEM_PROMPT.split("Ne nommez PAS", 1)[1]


def test_compliance_matrix_cache_avoids_llm_call_on_rerun(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compliance_client(_compliance_entries_json())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.generate_compliance_matrix("testcase")
    first_call_count = client.chat.completions.create.call_count
    assert first_call_count == 2  # one per role group in the fixture

    compliance.generate_compliance_matrix("testcase")
    second_call_count = client.chat.completions.create.call_count

    assert second_call_count == first_call_count  # rerun is entirely cache hits
    assert (tmp_path / "testcase" / "compliance_cache.jsonl").exists()


def test_compliance_cli_dry_run_reports_per_cluster_tokens(tmp_path, monkeypatch, capsys) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)

    def _fail_if_called():
        raise AssertionError("get_openrouter_client must not be called in --dry-run")

    monkeypatch.setattr(compliance, "get_openrouter_client", _fail_if_called)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())
    monkeypatch.setattr("sys.argv", ["rag.compliance", "--case-id", "testcase", "--dry-run"])

    compliance.main()

    captured = capsys.readouterr()
    assert "compliance dry-run cluster · role_id=" in captured.out
    assert captured.out.count("compliance dry-run cluster") == 2  # one per role group in the fixture


def test_check_dry_run_cost_gate_raises_above_threshold() -> None:
    from rag.compliance import DRY_RUN_COST_ALERT_USD, _check_dry_run_cost_gate

    with pytest.raises(RuntimeError, match="exceeds"):
        _check_dry_run_cost_gate(DRY_RUN_COST_ALERT_USD + 0.01)


def test_check_dry_run_cost_gate_allows_below_threshold() -> None:
    from rag.compliance import DRY_RUN_COST_ALERT_USD, _check_dry_run_cost_gate

    _check_dry_run_cost_gate(DRY_RUN_COST_ALERT_USD - 0.01)  # no raise


def test_check_real_run_cost_gate_raises_when_pacing_over_budget() -> None:
    from rag.compliance import REAL_RUN_COST_ALERT_USD, _check_real_run_cost_gate

    # Full budget already spent after only 1 of 2 clusters — badly off pace.
    with pytest.raises(RuntimeError, match="pro-rated"):
        _check_real_run_cost_gate(1, 2, REAL_RUN_COST_ALERT_USD)


def test_check_real_run_cost_gate_allows_when_pacing_under_budget() -> None:
    from rag.compliance import REAL_RUN_COST_ALERT_USD, _check_real_run_cost_gate

    _check_real_run_cost_gate(1, 46, REAL_RUN_COST_ALERT_USD / 46 - 0.01)  # no raise


def test_check_real_run_cost_gate_noop_before_any_clusters_done() -> None:
    from rag.compliance import _check_real_run_cost_gate

    _check_real_run_cost_gate(0, 46, 1000.0)  # nothing attributable yet — no raise


def test_compliance_matrix_real_run_aborts_and_preserves_cache(tmp_path, monkeypatch) -> None:
    """A real run that breaches the pro-rated pace aborts, but the cluster(s)
    already completed are persisted to compliance_cache.jsonl before the
    raise — a rerun after fixing the underlying issue resumes at zero cost
    for them, since caching is per-cluster append-only (ADR #53)."""
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(
        compliance, "get_openrouter_client",
        lambda: _mock_compliance_client(_compliance_entries_json(), cost=20.0),
    )
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    with pytest.raises(RuntimeError, match="exceeds the pro-rated"):
        compliance.generate_compliance_matrix("testcase")

    cache_path = tmp_path / "testcase" / "compliance_cache.jsonl"
    assert cache_path.exists()
    cached_lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(cached_lines) == 1  # first cluster's result persisted before the abort
    assert not (tmp_path / "testcase" / "compliance_matrix.json").exists()


# ========== comparative dual-model analysis tests (ADR #56, D7) ==========

def _compare_model_responses(opus_status: str = "met", kimi_status: str = "breached") -> dict[str, str]:
    """Canned Opus/Kimi/divergence JSON responses for compare-mode tests.
    Opus and Kimi agree on statute_chunk_id=cc-587 for notaire_redacteur, so
    both land on the same entry_id (a content-independent hash of
    statute_chunk_id + actor_role) — the pairing key the divergence
    analysis relies on."""
    obligation_summary = "Conserver la substance des biens quasi-usufruits."
    opus_json = json.dumps([{
        "statute_chunk_id": "cc-587", "statute_excerpt": "x",
        "obligation_summary": obligation_summary,
        "status": opus_status, "evidence_fact_ids": ["doc1-f001"], "rationale": "Opus rationale.",
    }])
    kimi_json = json.dumps([{
        "statute_chunk_id": "cc-587", "statute_excerpt": "x",
        "obligation_summary": obligation_summary,
        "status": kimi_status, "evidence_fact_ids": ["doc1-f001"], "rationale": "Kimi rationale.",
    }])
    if opus_status == kimi_status:
        shared_obligations = [{
            "obligation_summary": obligation_summary,
            "opus_verdict": opus_status, "kimi_verdict": kimi_status,
        }]
        divergent_obligations = []
    else:
        shared_obligations = []
        divergent_obligations = [{
            "obligation_summary": obligation_summary,
            "opus_verdict": opus_status, "kimi_verdict": kimi_status,
            "crux": "test crux", "stronger_side": "opus", "why": "test why",
        }]
    divergence_json = json.dumps({
        "shared_obligations": shared_obligations,
        "divergent_obligations": divergent_obligations,
        "meta_summary": "Résumé de test.",
    })
    return {
        "anthropic/claude-opus-4.7": opus_json,
        "moonshotai/kimi-k3": kimi_json,
        "anthropic/claude-haiku-4.5": divergence_json,
    }


def _mock_compare_client(model_to_content: dict[str, str], cost: float = 0.005) -> MagicMock:
    """Like _mock_compliance_client, but dispatches on the `model` kwarg —
    compare_compliance_for_role's three calls (Opus, Kimi, Haiku divergence)
    each need their own canned response, unlike the single-model matrix
    flow's one-content-fits-all mock."""
    client = MagicMock()

    def _create(*, model, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=model_to_content[model]))]
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, cost=cost)
        return response

    client.chat.completions.create.side_effect = _create
    return client


def _mutate_fact_verbatim(base_dir: Path, case_id: str, fact_id: str, new_verbatim: str) -> None:
    """Rewrite one fact's verbatim_quote in facts.jsonl, keeping its fact_id
    unchanged — used to verify compare_compliance_for_role's cache
    invalidates on fact content changes, not just fact-id membership."""
    facts_path = base_dir / case_id / "facts.jsonl"
    rewritten = []
    for line in facts_path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj["fact_id"] == fact_id:
            obj["verbatim_quote"] = new_verbatim
        rewritten.append(json.dumps(obj, ensure_ascii=False))
    facts_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def test_compare_compliance_calls_both_models_with_distinct_model_ids(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.compare_compliance_for_role("testcase", "notaire_redacteur")

    called_models = [
        call.kwargs.get("model") for call in client.chat.completions.create.call_args_list
    ]
    assert called_models.count("anthropic/claude-opus-4.7") == 1
    assert called_models.count("moonshotai/kimi-k3") == 1
    assert called_models.count("anthropic/claude-haiku-4.5") == 1


def test_compare_compliance_writes_comparative_json_with_expected_schema(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    result = compliance.compare_compliance_for_role("testcase", "notaire_redacteur")

    out_path = tmp_path / "testcase" / "compliance_comparative_notaire_redacteur.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))

    for key in (
        "case_id", "role_id", "role_label", "generated_at", "inputs_hash",
        "divergence_prompt_hash", "models", "coverage_diff",
        "divergence_analysis", "divergence_model_id", "cache_hit",
    ):
        assert key in on_disk, f"missing key: {key}"

    assert on_disk["models"]["opus"]["model_id"] == "anthropic/claude-opus-4.7"
    assert on_disk["models"]["kimi"]["model_id"] == "moonshotai/kimi-k3"
    assert on_disk["coverage_diff"]["shared_entry_ids"] == result["coverage_diff"]["shared_entry_ids"]
    assert len(on_disk["coverage_diff"]["shared_entry_ids"]) == 1
    assert on_disk["divergence_analysis"]["divergent_obligations"]
    assert on_disk["cache_hit"] is False


def test_compare_compliance_is_idempotent_via_inputs_hash(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    first = compliance.compare_compliance_for_role("testcase", "notaire_redacteur")
    first_call_count = client.chat.completions.create.call_count
    assert first_call_count == 3  # opus + kimi + haiku divergence
    assert first["cache_hit"] is False

    second = compliance.compare_compliance_for_role("testcase", "notaire_redacteur")

    assert client.chat.completions.create.call_count == first_call_count  # no new calls
    assert second["cache_hit"] is True
    assert second["inputs_hash"] == first["inputs_hash"]


def test_compare_compliance_invalidates_cache_on_fact_change(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.compare_compliance_for_role("testcase", "notaire_redacteur")
    first_call_count = client.chat.completions.create.call_count

    # Same fact_id, different content — inputs_hash must not treat this as
    # a cache hit (a plain fact_id-only fingerprint would miss this).
    _mutate_fact_verbatim(
        tmp_path, "testcase", "doc1-f001",
        "Le notaire reçoit la convention de quasi-usufruit (version modifiée).",
    )

    result = compliance.compare_compliance_for_role("testcase", "notaire_redacteur")

    assert client.chat.completions.create.call_count == first_call_count + 3  # re-invoked
    assert result["cache_hit"] is False


def test_compare_compliance_invalidates_cache_on_divergence_prompt_change(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.compare_compliance_for_role("testcase", "notaire_redacteur")
    first_call_count = client.chat.completions.create.call_count

    # Facts/chunks/persons/models all unchanged — only the divergence
    # prompt text changed (e.g. a prompt-wording fix) — must still miss.
    monkeypatch.setattr(
        compliance, "DIVERGENCE_ANALYSIS_SYSTEM_PROMPT",
        compliance.DIVERGENCE_ANALYSIS_SYSTEM_PROMPT + "\n(edited)",
    )

    result = compliance.compare_compliance_for_role("testcase", "notaire_redacteur")

    assert client.chat.completions.create.call_count == first_call_count + 3  # re-invoked
    assert result["cache_hit"] is False


def test_divergence_analysis_produces_valid_schema(monkeypatch) -> None:
    from rag import compliance

    divergence_json = json.dumps({
        "shared_obligations": [
            {"obligation_summary": "o1", "opus_verdict": "met", "kimi_verdict": "met"},
        ],
        "divergent_obligations": [
            {
                "obligation_summary": "o2", "opus_verdict": "met", "kimi_verdict": "breached",
                "crux": "c", "stronger_side": "opus", "why": "w",
            },
        ],
        "meta_summary": "Résumé.",
    })
    client = _mock_compare_client({"anthropic/claude-haiku-4.5": divergence_json})
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)

    opus_entry = compliance.ComplianceEntry(
        entry_id="e1", statute_chunk_id="cc-587", statute_excerpt="x",
        obligation_summary="o1", actor_role="notaire_redacteur", status="met",
        evidence_fact_ids=["doc1-f001"], rationale="r",
    )
    kimi_entry = compliance.ComplianceEntry(
        entry_id="e1", statute_chunk_id="cc-587", statute_excerpt="x",
        obligation_summary="o1", actor_role="notaire_redacteur", status="met",
        evidence_fact_ids=["doc1-f001"], rationale="r",
    )

    divergence, usage = compliance._call_divergence_analysis(
        "notaire_redacteur", "Notaire rédacteur", [opus_entry], [kimi_entry], {"e1"},
    )

    assert set(divergence) == {"shared_obligations", "divergent_obligations", "meta_summary"}
    assert divergence["shared_obligations"][0]["opus_verdict"] == "met"
    assert divergence["divergent_obligations"][0]["stronger_side"] == "opus"
    assert usage["estimated"] is False


def test_compare_compliance_does_not_pollute_shared_compliance_cache(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.compare_compliance_for_role("testcase", "notaire_redacteur")

    assert not (tmp_path / "testcase" / "compliance_cache.jsonl").exists()


# ========== single-model per-role analysis (ADR #57, D8) ==========

def test_run_compliance_for_role_calls_only_the_selected_model(tmp_path, monkeypatch) -> None:
    """The whole point of the UI model toggle: selecting one model must fire
    exactly that model once, and neither the other frontier model nor the
    divergence model at all."""
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.run_compliance_for_role(
        "testcase", "notaire_redacteur", compliance_model_id="moonshotai/kimi-k3",
    )

    called_models = [
        call.kwargs.get("model") for call in client.chat.completions.create.call_args_list
    ]
    assert called_models == ["moonshotai/kimi-k3"]


def test_run_compliance_for_role_cache_is_per_model(tmp_path, monkeypatch) -> None:
    """Opus and Kimi results for the same role must not share a cache slot.

    A shared slot would let the panel display one model's verdicts under the
    other model's name — the exact failure _single_inputs_hash and the
    per-model filename exist to prevent.
    """
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses(opus_status="met", kimi_status="breached"))
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    opus = compliance.run_compliance_for_role(
        "testcase", "notaire_redacteur", compliance_model_id="anthropic/claude-opus-4.7",
    )
    assert opus["cache_hit"] is False
    assert client.chat.completions.create.call_count == 1

    # Toggling the model must MISS, not serve the Opus result back.
    kimi = compliance.run_compliance_for_role(
        "testcase", "notaire_redacteur", compliance_model_id="moonshotai/kimi-k3",
    )
    assert kimi["cache_hit"] is False
    assert client.chat.completions.create.call_count == 2
    assert kimi["inputs_hash"] != opus["inputs_hash"]
    assert kimi["model"]["model_id"] == "moonshotai/kimi-k3"
    assert kimi["model"]["entries"][0]["status"] == "breached"
    assert opus["model"]["entries"][0]["status"] == "met"

    # Both results coexist on disk under model-scoped filenames.
    case_dir = tmp_path / "testcase"
    assert (case_dir / "compliance_single_notaire_redacteur_anthropic_claude-opus-4.7.json").exists()
    assert (case_dir / "compliance_single_notaire_redacteur_moonshotai_kimi-k3.json").exists()

    # Re-selecting the first model serves its own cache, with no new call.
    again = compliance.run_compliance_for_role(
        "testcase", "notaire_redacteur", compliance_model_id="anthropic/claude-opus-4.7",
    )
    assert again["cache_hit"] is True
    assert client.chat.completions.create.call_count == 2
    assert again["model"]["entries"][0]["status"] == "met"


def test_run_compliance_for_role_does_not_pollute_shared_compliance_cache(tmp_path, monkeypatch) -> None:
    """Mirrors the compare-mode guard: the shared compliance_cache.jsonl key
    has no model dimension, so this path must never write to it."""
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    client = _mock_compare_client(_compare_model_responses())
    monkeypatch.setattr(compliance, "get_openrouter_client", lambda: client)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    compliance.run_compliance_for_role("testcase", "notaire_redacteur")

    assert not (tmp_path / "testcase" / "compliance_cache.jsonl").exists()


def test_run_compliance_for_role_raises_on_unknown_role(tmp_path, monkeypatch) -> None:
    from rag import compliance

    _write_fixture_case(tmp_path, "testcase")
    monkeypatch.setattr(compliance, "DOSSIER_DIR", tmp_path)
    monkeypatch.setattr(compliance, "retrieve", lambda *a, **k: _mock_chunks())

    with pytest.raises(ValueError, match="role_id"):
        compliance.run_compliance_for_role("testcase", "role_qui_nexiste_pas")


# ========== answer prompt: statute vs dossier vs blended (ADR #62) ==========

def _statute_chunk() -> dict:
    return {
        "chunk_id": "cc-587-1", "source_label": "Code civil", "num": "587",
        "titre": "", "section_path": "Livre II > Titre III",
        "url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006428859",
        "texte": "Si l'usufruit comprend des choses dont on ne peut faire usage...",
    }


def _dossier_chunk() -> dict:
    return {
        "chunk_id": "dossier-vitrine-note_de_presentation-c001",
        "source_label": "", "num": "note_de_presentation", "titre": "Page 1",
        "section_path": "", "url": "data/dossier/vitrine/extracted/note.md",
        "texte": "Mme MARTIN a mis en demeure Maître DUBOIS le 30 juin 2026.",
    }


def test_dossier_question_does_not_get_the_statute_prompt() -> None:
    """The statute template labels its context "ARTICLES DE LOI" and demands
    "art. {num} du {source_label} ({url})" — fields a case document has no
    real values for. Applied to a dossier it produced a hedged summary citing
    "art. non spécifié" at a fabricated "https://data/dossier/..." URL."""
    from rag import prompt

    p = prompt.build("quel est le litige dans ce dossier ?", [_dossier_chunk()])

    assert "PIÈCES DU DOSSIER" in p
    assert "ARTICLES DE LOI PERTINENTS" not in p
    assert "https://" not in p, "a private document must never be given a web URL"


def test_dossier_prompt_demands_named_concrete_facts() -> None:
    """The whole point of the case-file surface: a reply that would describe
    any succession is a failure, not a safe answer."""
    from rag import prompt

    p = prompt.build("quel est le litige ?", [_dossier_chunk()])

    assert "nomme les personnes" in p
    assert "généralités juridiques" in p
    assert "p. {page}" in p or "p. Page 1" in p or "Page: Page 1" in p


def test_statute_question_keeps_the_article_prompt() -> None:
    from rag import prompt

    p = prompt.build("qu'est-ce que le quasi-usufruit ?", [_statute_chunk()])

    assert "ARTICLES DE LOI PERTINENTS" in p
    assert "PIÈCES DU DOSSIER" not in p
    assert "legifrance.gouv.fr" in p


def test_blended_retrieval_gets_the_template_that_separates_the_two() -> None:
    """A gap-analysis query routes to "blended", so both kinds of chunk arrive
    together and the reply has to say which claim came from which."""
    from rag import prompt

    p = prompt.build("le notaire a-t-il manqué à ses obligations ?",
                     [_statute_chunk(), _dossier_chunk()])

    assert "Distingue toujours ce qui vient de la LOI" in p
    assert "legifrance.gouv.fr" in p
    assert "Mme MARTIN" in p


def test_template_is_chosen_from_the_chunks_not_the_caller() -> None:
    """Retrieval is what actually decides what the model is looking at;
    reading it here keeps the prompt and the context from disagreeing."""
    from rag import prompt

    assert prompt._is_dossier(_dossier_chunk())
    assert not prompt._is_dossier(_statute_chunk())
