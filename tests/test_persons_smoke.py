"""Smoke tests for Attempt 2, Deliverable D5 — fact-level distillation
(ingestion/dossier/distill.py).

Named test_persons_smoke.py to match the Attempt 2 execution plan's
deliverable numbering (D1's smoke suite, extended here for D5), even though
D1-D4 (the actual person-index pipeline: mentions.py, resolve.py) are
unshipped as of this session — see ADR #52's Context for why D5 doesn't
depend on them.

Synthetic fixtures only (a small facts.jsonl + one extracted/*.md built in
tmp_path) — data/dossier/demo/ has no real fact content to distill against
yet, so these tests don't depend on it. Mirrors
tests/test_dossier_smoke.py's DOSSIER_DIR monkeypatch + MagicMock
client-patch conventions.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingestion.dossier import distill


def _fake_chat_response(content: str):
    """Minimal stand-in for an OpenAI-shaped chat.completions.create() response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


_DISTILLED_TEXT = (
    "Maître MENA reçoit la convention de quasi-usufruit le 8 mars 2024, "
    "portant sur des parts de SCPI EDISSIMMO et NOTAPIERRE. Trois "
    "petits-enfants disposent d'une créance de restitution de 195 572 €."
)

_FACT = {
    "fact_id": "doc-f001",
    "date": "2024-03-08",
    "actor_role": "notaire_redacteur",
    "action": "recevoir une convention de quasi-usufruit",
    "target": "parts de SCPI",
    "verbatim_quote": (
        "Trois petits-enfants sont titulaires d'une creance de restitution de "
        "195 572 EUR nee d'une convention de quasi-usufruit du 8 mars 2024."
    ),
    "source_doc_id": "doc",
    "source_chunk_id": "dossier-acme-doc-c001",
    "distilled_context": None,
}

_SOURCE_MD = (
    "## Page 1\n\n"
    "Maitre Eve Marie MENA\nNotaire a Paris\n\n"
    "Madame, Monsieur,\n\n"
    "Je vous prie de bien vouloir prendre connaissance des elements suivants. "
    + _FACT["verbatim_quote"]
    + " Cette creance est nee d'une convention recue par l'etude PAVY-MENA.\n\n"
    "Je vous prie d'agreer, Madame, Monsieur, l'expression de mes salutations "
    "distinguees.\nMaitre MENA"
)


@pytest.fixture
def distill_case_dir(tmp_path, monkeypatch):
    """Redirect distill.DOSSIER_DIR to an isolated tmp dir with one case's facts + extracted doc."""
    monkeypatch.setattr(distill, "DOSSIER_DIR", tmp_path)
    case_dir = tmp_path / "acme"
    extracted_dir = case_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "doc.md").write_text(_SOURCE_MD, encoding="utf-8")

    facts_path = case_dir / "facts.jsonl"
    facts_path.write_text(json.dumps(_FACT, ensure_ascii=False) + "\n", encoding="utf-8")
    return case_dir


def test_distill_fact_produces_valid_output_on_fixture(distill_case_dir) -> None:
    """distill_fact returns non-empty dense text from a mocked client on a cache miss."""
    source_text = (distill_case_dir / "extracted" / "doc.md").read_text(encoding="utf-8")
    source_context = distill._extract_fact_neighborhood(source_text, _FACT["verbatim_quote"])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_chat_response(_DISTILLED_TEXT)

    with patch("ingestion.dossier.distill.get_openrouter_client", return_value=mock_client):
        text, usage = distill.distill_fact(dict(_FACT), source_context, cache={}, dry_run=False)

    assert isinstance(text, str)
    assert text.strip() != ""
    assert usage["cache_hit"] is False
    mock_client.chat.completions.create.assert_called_once()


def test_extract_fact_neighborhood_matches_across_nbsp_whitespace_drift() -> None:
    """Regex fallback finds a verbatim_quote (regular spaces) against source
    text carrying NBSP in place of a space (e.g. in a number like "195 572"),
    instead of silently returning the source_text[:window] fallback."""
    verbatim_quote = "amount of 195 572 euros"
    source_text = (
        "some preceding filler text " * 5
        + "amount of 195 572 euros"
        + " some trailing filler text" * 5
    )

    result = distill._extract_fact_neighborhood(source_text, verbatim_quote, window=40)

    assert result != source_text[:40]
    assert "195 572" in result


def test_extract_fact_neighborhood_matches_across_newline_count_drift() -> None:
    """Regex fallback finds a verbatim_quote containing a single \\n where the
    source text has \\n\\n (different line-break handling), instead of
    falling back to source_text[:window]."""
    verbatim_quote = "premiere ligne\ndeuxieme ligne du fait juridique invoque ici"
    source_text = (
        "filler " * 10
        + "premiere ligne\n\ndeuxieme ligne du fait juridique invoque ici"
        + " filler" * 10
    )

    result = distill._extract_fact_neighborhood(source_text, verbatim_quote, window=40)

    assert result != source_text[:40]
    assert "deuxieme ligne du fait juridique invoque ici" in result


def test_distill_case_populates_all_facts(distill_case_dir) -> None:
    """distill_case backfills distilled_context on every fact in the case."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_chat_response(_DISTILLED_TEXT)

    with patch("ingestion.dossier.distill.get_openrouter_client", return_value=mock_client):
        summary = distill.distill_case("acme")

    assert summary["total_facts"] == 1
    assert summary["facts_distilled"] == 1

    facts_path = distill_case_dir / "facts.jsonl"
    lines = facts_path.read_text(encoding="utf-8").strip().splitlines()
    fact = json.loads(lines[0])
    assert fact["distilled_context"] == _DISTILLED_TEXT


def test_distill_case_idempotent_on_rerun(distill_case_dir) -> None:
    """Second run on unchanged input is all cache hits; facts.jsonl is byte-identical."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_chat_response(_DISTILLED_TEXT)

    facts_path = distill_case_dir / "facts.jsonl"

    with patch("ingestion.dossier.distill.get_openrouter_client", return_value=mock_client):
        distill.distill_case("acme")
        bytes_after_first = facts_path.read_bytes()

        summary2 = distill.distill_case("acme")
        bytes_after_second = facts_path.read_bytes()

    assert summary2["cache_hits"] == 1
    assert summary2["facts_distilled"] == 0
    assert mock_client.chat.completions.create.call_count == 1  # never called on the rerun
    assert bytes_after_first == bytes_after_second


def test_distill_cache_key_deterministic() -> None:
    """_cache_key is a pure function: same inputs produce the same key, every call."""
    quote = "Trois petits-enfants sont titulaires d'une creance."
    context = "## Page 1\n\nContexte quelconque autour de la citation."

    key1 = distill._cache_key(quote, context)
    key2 = distill._cache_key(quote, context)

    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 64  # sha256 hex digest length


def test_distill_atomic_write_under_failure(distill_case_dir) -> None:
    """A failing os.replace leaves facts.jsonl untouched and no .tmp file behind."""
    facts_path = distill_case_dir / "facts.jsonl"
    original_bytes = facts_path.read_bytes()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_chat_response(_DISTILLED_TEXT)

    with patch("ingestion.dossier.distill.get_openrouter_client", return_value=mock_client), \
         patch("ingestion.dossier.distill.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            distill.distill_case("acme")

    assert facts_path.read_bytes() == original_bytes
    assert not (distill_case_dir / "facts.jsonl.tmp").exists()


def test_distill_preserves_existing_fact_fields(distill_case_dir) -> None:
    """Every field except distilled_context is unchanged after a distillation run."""
    facts_path = distill_case_dir / "facts.jsonl"
    original = json.loads(facts_path.read_text(encoding="utf-8").strip())

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_chat_response(_DISTILLED_TEXT)

    with patch("ingestion.dossier.distill.get_openrouter_client", return_value=mock_client):
        distill.distill_case("acme")

    updated = json.loads(facts_path.read_text(encoding="utf-8").strip())

    original_minus = {k: v for k, v in original.items() if k != "distilled_context"}
    updated_minus = {k: v for k, v in updated.items() if k != "distilled_context"}
    assert original_minus == updated_minus
    assert updated["distilled_context"] == _DISTILLED_TEXT
