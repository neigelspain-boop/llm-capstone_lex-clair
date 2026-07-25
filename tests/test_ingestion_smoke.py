"""Smoke tests for the ingestion pipeline (Plane I) · articles + chunks integrity.

Runs in <3 seconds. Guards against corpus schema drift and silent data loss
between fetch → parse → chunk. Does NOT test BM25 or Chroma — those are
integration concerns that cost real time and GPU. Smoke tests must stay fast
enough that they get run on every save.

The 18 critical articles asserted below are the load-bearing legal references
for the lex-clair case domain (quasi-usufruit, succession, notaire liability,
fiscal anti-abuse). If any is missing, retrieval eval on Day 3 cannot pass.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest


# Smoke-test artifact paths for the canonical ingestion corpus.
ROOT = Path(__file__).resolve().parents[1]
ARTICLES_CSV = ROOT / "data" / "articles.csv"
CHUNKS_CSV = ROOT / "data" / "chunks.csv"
GROUND_TRUTH_CSV = ROOT / "data" / "ground_truth.csv"

# Exact article IDs that encode the core legal case domain for retrieval.
# If any of these disappear, downstream eval cannot achieve target coverage.
CRITICAL_ARTICLES = [
    # Successions
    "cc-720",
    # Usufruit + quasi-usufruit
    "cc-578", "cc-587", "cc-600", "cc-601",
    # Libéralités + réserve héréditaire
    "cc-893", "cc-912", "cc-913", "cc-914-1", "cc-920", "cc-924", "cc-1094-1",
    # Responsabilité délictuelle
    "cc-1240", "cc-1241",
    # Anti-abus fiscal 2024
    "cgi-774bis",
    # Action directe assureur
    "ca-l124-3", "ca-l124-5",
    # Abus de confiance
    "cp-314-1",
]


# --- fixtures -----------------------------------------------------------------


def test_import_smoke_streamlit_app() -> None:
    """Import smoke — catches typos, bad imports, missing deps."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", "import app.streamlit_app; print('imports ok')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "imports ok" in result.stdout


# ========== Day 6: Streamlit app smoke tests ==========

def test_streamlit_app_imports():
    """UI module imports cleanly + Day 7 contract intact.

    Guards: import chain, LABELS shape, Day 7 db_offline_banner labels,
    and the three wrappers (_save_conversation, _load_all_conversations,
    _append_feedback) that the render code calls into. Any refactor that
    renames or drops a wrapper breaks silently at rendertime without this.
    """
    import importlib

    module = importlib.import_module("app.streamlit_app")

    assert hasattr(module, "LABELS"), "LABELS dict missing"
    assert "fr" in module.LABELS and "en" in module.LABELS
    assert "db_offline_banner" in module.LABELS["fr"], (
        "Day 7 fr db_offline_banner label missing"
    )
    assert "db_offline_banner" in module.LABELS["en"], (
        "Day 7 en db_offline_banner label missing"
    )

    for name in ("_save_conversation", "_load_all_conversations", "_append_feedback"):
        assert hasattr(module, name), f"{name} wrapper missing"
        assert callable(getattr(module, name)), f"{name} not callable"


def test_feedback_schema_columns_match_day7_contract(db_conn):
    """Postgres feedback table has exactly the columns db.append_feedback
    reads from the row dict, plus DB-managed columns (id, created_at).

    Original intent preserved: guard against schema drift between what
    the app writes and what the persistence layer expects. Day 7 moves
    the check from a CSV header constant to a real DB introspection query.
    """
    expected_columns = {
        "id",              # SERIAL PK, DB-managed
        "conversation_id", # from row["conversation_id"]
        "turn_id",         # from row["turn_id"]
        "rating",          # from row["rating"]
        "comment",         # from row.get("comment")
        "created_at",      # from either DEFAULT NOW() or migrate override
    }
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'feedback'
        """)
        actual = {r[0] for r in cur.fetchall()}

    assert actual == expected_columns, (
        f"feedback schema drift — expected {expected_columns}, got {actual}"
    )


def test_conversation_wrapper_roundtrip(db_conn):
    """Wrapper roundtrip: _save_conversation → _load_all_conversations.

    Complements test_db_conversation_roundtrip by exercising the app-layer
    wrappers, including _save_conversation's updated_at mutation which the
    sidebar sort depends on. A refactor that drops the mutation would
    silently regress "most-recent-first" ordering — this test catches it.
    """
    import time
    from app import streamlit_app as ui

    conv = ui._new_conversation("Wrapper roundtrip test")
    original_updated = conv["updated_at"]

    # Sleep enough that datetime.now differs on any reasonable clock resolution
    time.sleep(0.05)

    ui._save_conversation(conv)

    # updated_at mutation contract — sidebar sort depends on this
    assert conv["updated_at"] != original_updated, (
        "_save_conversation must mutate conv['updated_at'] to NOW "
        "(sidebar sort-by-recency depends on this)"
    )
    assert conv["updated_at"] > original_updated, (
        "updated_at moved backwards, clock issue or bug"
    )

    # Roundtrip through Postgres via the wrapper (not db.* directly)
    loaded = ui._load_all_conversations()
    assert conv["id"] in loaded, "wrapper didn't persist"
    assert loaded[conv["id"]]["title"] == "Wrapper roundtrip test"

    # Cleanup — CASCADE nukes any turns
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM conversations WHERE id = %s", (conv["id"],))
    db_conn.commit()


@pytest.fixture(scope="module")
def articles() -> pd.DataFrame:
    """Load articles.csv with the canonical ingestion read config."""
    assert ARTICLES_CSV.exists(), f"missing artifact: {ARTICLES_CSV}"
    return pd.read_csv(ARTICLES_CSV, keep_default_na=False)


@pytest.fixture(scope="module")
def chunks() -> pd.DataFrame:
    """Load chunks.csv with the canonical ingestion read config."""
    assert CHUNKS_CSV.exists(), f"missing artifact: {CHUNKS_CSV}"
    return pd.read_csv(CHUNKS_CSV, keep_default_na=False)


# --- articles.csv invariants --------------------------------------------------


def test_articles_csv_row_count_in_expected_range(articles: pd.DataFrame) -> None:
    """Corpus size drifts either up (over-enumeration) or down (missing sections).

    Bounds chosen to be tight enough to catch either drift but loose enough
    to survive adding one small manifest source without failing.
    """
    assert 700 <= len(articles) <= 900, f"unexpected corpus size: {len(articles)}"


def test_articles_chunk_id_unique(articles: pd.DataFrame) -> None:
    """Duplicate chunk_ids break BM25 fit and Chroma add. Namespacing bug guard."""
    dupes = articles["chunk_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate chunk_ids in articles.csv"


def test_articles_texte_non_null(articles: pd.DataFrame) -> None:
    """Empty texte means nothing to embed. Parse.py should have dropped these."""
    nulls = articles["texte"].isnull().sum()
    assert nulls == 0, f"{nulls} null texte values in articles.csv"


def test_articles_all_critical_present(articles: pd.DataFrame) -> None:
    """Every case-critical article must be indexed. Wrong LEGISCTA guard."""
    present = set(articles["chunk_id"])
    missing = [cid for cid in CRITICAL_ARTICLES if cid not in present]
    assert not missing, f"critical articles missing from articles.csv: {missing}"


def test_articles_only_vigueur(articles: pd.DataFrame) -> None:
    """Non-VIGUEUR articles should not surface in the retrieval corpus."""
    bad = articles[~articles["etat"].isin(["VIGUEUR", "VIGUEUR_DIFF"])]
    assert bad.empty, f"{len(bad)} non-VIGUEUR articles present"


# --- chunks.csv invariants ----------------------------------------------------


def test_chunks_csv_matches_articles_row_count(
    articles: pd.DataFrame, chunks: pd.DataFrame
) -> None:
    """V1 chunk.py is identity — row count must be preserved."""
    assert len(chunks) == len(articles), (
        f"row-count drift: articles={len(articles)}, chunks={len(chunks)}"
    )


def test_chunks_chunk_id_unique(chunks: pd.DataFrame) -> None:
    """Duplicate chunk_ids in chunks.csv would collide in the Chroma primary key."""
    dupes = chunks["chunk_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate chunk_ids in chunks.csv"


def test_chunks_texte_non_null(chunks: pd.DataFrame) -> None:
    """Guards against the NaN round-trip bug (keep_default_na=False regression)."""
    nulls = chunks["texte"].isnull().sum()
    assert nulls == 0, f"{nulls} null texte values in chunks.csv"


def test_chunks_all_critical_present(chunks: pd.DataFrame) -> None:
    """Critical articles must survive the chunking transform."""
    present = set(chunks["chunk_id"])
    missing = [cid for cid in CRITICAL_ARTICLES if cid not in present]
    assert not missing, f"critical articles missing from chunks.csv: {missing}"


def test_chunks_required_columns_present(chunks: pd.DataFrame) -> None:
    """Downstream contract with index.py. If a column drops, index build fails."""
    required = {"chunk_id", "source", "source_label", "num", "titre",
                "section_path", "texte", "url"}
    missing = required - set(chunks.columns)
    assert not missing, f"required columns missing: {missing}"


def test_chunks_url_populated_where_expected(chunks: pd.DataFrame) -> None:
    """URL is what powers verifiable citations on Day 4.

    Not every row has a URL (LODA articles sometimes lack legiarti_id), but
    the vast majority should. Threshold guards against a schema regression
    dropping url from parse.py.
    """
    with_url = (chunks["url"].str.startswith("http")).sum()
    assert with_url >= 0.9 * len(chunks), (
        f"only {with_url}/{len(chunks)} chunks have URLs — expected ≥90%"
    )


# --- ground_truth.csv invariants ----------------------------------------------


@pytest.fixture(scope="module")
def ground_truth() -> pd.DataFrame:
    """Load ground_truth.csv with the canonical ingestion read config."""
    assert GROUND_TRUTH_CSV.exists(), f"missing artifact: {GROUND_TRUTH_CSV}"
    return pd.read_csv(GROUND_TRUTH_CSV, keep_default_na=False)


def test_ground_truth_shape(ground_truth: pd.DataFrame) -> None:
    """Ground truth CSV must satisfy retrieval_eval.py's input contract."""
    assert len(ground_truth) > 0, "ground_truth.csv is empty"
    required = {"chunk_id", "question"}
    missing = required - set(ground_truth.columns)
    assert not missing, f"ground_truth missing columns: {missing}"
    assert (ground_truth["question"].str.len() > 0).all(), "empty questions present"


def test_ground_truth_ids_valid(
    chunks: pd.DataFrame, ground_truth: pd.DataFrame
) -> None:
    """Every chunk_id in ground_truth must resolve to a chunk. Orphans → silent Hit@k=0."""
    valid_ids = set(chunks["chunk_id"])
    gt_ids = set(ground_truth["chunk_id"])
    orphans = gt_ids - valid_ids
    assert not orphans, f"ground_truth chunk_ids not in chunks.csv: {sorted(orphans)[:5]}"


# --- Plane II end-to-end ------------------------------------------------------


@pytest.mark.slow
def test_rag_flow_end_to_end() -> None:
    """Full pipeline: query → French answer with at least one Legifrance citation.

    Slow (~10-25s cold, ~3-4s warm). Opt-in via `pytest -m slow`.
    Not in the default smoke suite because it hits the OpenAI API and
    loads two ~2GB models into VRAM.
    """
    from rag import flow

    result = flow.run("Qu'est-ce que le quasi-usufruit ?")

    # answer is non-empty French text
    assert result["answer"], "empty answer from flow.run"
    assert len(result["answer"]) > 100, "answer suspiciously short"

    # citations link to Legifrance
    assert len(result["citations"]) >= 1, "no citations returned"
    assert all(c["url"].startswith("http") for c in result["citations"]), \
        "non-URL citation slipped through"
    assert all("legifrance" in c["url"] for c in result["citations"]), \
        "citation not pointing at Legifrance"

    # cost + timing sanity
    assert result["cost_usd"] < 0.01, f"cost too high: ${result['cost_usd']}"
    assert result["elapsed_seconds"] < 60, f"too slow: {result['elapsed_seconds']}s"

    # pipeline shape
    assert result["chunks_retrieved"] == 20
    assert result["chunks_reranked"] == 5
    assert result["model_used"] == "gpt-4o-mini"


# --- llm_eval_results.csv invariants (Day 5) ----------------------------------


@pytest.fixture(scope="module")
def llm_eval() -> pd.DataFrame:
    """Load llm_eval_results.csv produced by eval/llm_eval.py."""
    path = ROOT / "data" / "llm_eval_results.csv"
    assert path.exists(), f"missing artifact: {path}"
    return pd.read_csv(path, keep_default_na=False)


def test_llm_eval_row_count(llm_eval: pd.DataFrame) -> None:
    """N samples × 3 judges = 3N rows. Guards partial runs / mid-flight state."""
    n = len(llm_eval)
    assert n % 3 == 0, f"row count {n} not divisible by 3 judges"
    assert n >= 300, f"suspicious sample size: {n // 3} queries (expected >=100)"


def test_llm_eval_all_judges_present(llm_eval: pd.DataFrame) -> None:
    """Every query must have all 3 judgments. No partial evaluations."""
    per_query = llm_eval.groupby("query_id").size()
    incomplete = per_query[per_query != 3]
    assert incomplete.empty, (
        f"{len(incomplete)} queries have !=3 judgments: "
        f"{incomplete.head().to_dict()}"
    )


def test_llm_eval_verdicts_valid(llm_eval: pd.DataFrame) -> None:
    """No JSON drift from any judge model.

    UNKNOWN is a permitted recorded-failure verdict from the silent-fallback
    contract (see eval/llm_eval.py:_parse_judge_response). If UNKNOWN counts
    are high, the run's kill switch should have tripped -- the invariant here
    is contract-shape, not failure-rate.
    """
    valid = {"RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT", "UNKNOWN"}
    bad = llm_eval[~llm_eval["verdict"].isin(valid)]
    assert bad.empty, (
        f"{len(bad)} rows with invalid verdicts: "
        f"{bad['verdict'].unique().tolist()}"
    )

# ========== db.py Postgres integration smoke tests (Day 7) ==========


@pytest.fixture
def db_conn():
    """Yield a live Postgres connection. Skip if unreachable.

    Fast suite must not require `docker compose up`. On skip, other tests
    still run. Does NOT close the connection — db.py owns the singleton.
    """
    import psycopg2  # noqa: F401 — imported for OperationalError catch
    from monitoring import db

    db._reset_conn()  # clear any kill-switch trip from prior test
    conn = db.get_conn()
    if conn is None:
        pytest.skip("Postgres unavailable — expected without docker compose up")
    yield conn
    # deliberately do not close: db.py manages the singleton lifecycle


def test_db_schema_init_idempotent(db_conn):
    """init_schema() is safe to call multiple times.

    Verifies CREATE TABLE IF NOT EXISTS contract: first call creates,
    subsequent calls no-op. Also asserts all three tables land as
    expected shape.
    """
    from monitoring import db

    db.init_schema()
    db.init_schema()  # second call MUST NOT raise
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        tables = {r[0] for r in cur.fetchall()}
    assert {"conversations", "turns", "feedback"}.issubset(tables), (
        f"missing tables: expected superset of "
        f"{{'conversations', 'turns', 'feedback'}}, got {tables}"
    )


def test_db_conversation_roundtrip(db_conn):
    """Save via db.save_conversation, load back, verify shape preservation.

    Catches ORM drift on Day 6 dict → Postgres columns → reconstructed dict.
    Uses app.streamlit_app helpers (same code path the UI uses) to build
    the input, so any breakage in _new_conversation/_new_turn also fails
    this test.
    """
    from app import streamlit_app as ui
    from monitoring import db

    # Build a conversation matching Day 6's ADR #31 shape
    conv = ui._new_conversation("Day 7 smoke test question")
    conv["turns"].append(ui._new_turn("Day 7 smoke test question"))
    conv["turns"][0]["result"] = {
        "answer": "smoke test answer",
        "citations": [
            {"chunk_id": "cc-test", "num": "999", "url": "https://example.com"}
        ],
        "cost_usd": 0.001,
        "elapsed_seconds": 5.0,
        "model_used": "gpt-4o-mini",
        "chunks_retrieved": 20,
        "chunks_reranked": 5,
        "rewritten_query": "smoke",
    }

    # Save
    db.save_conversation(conv)

    # Load back and verify shape
    loaded = db.load_all_conversations()
    assert conv["id"] in loaded, "conversation not persisted"
    reloaded = loaded[conv["id"]]
    assert reloaded["title"] == conv["title"]
    assert len(reloaded["turns"]) == 1
    reloaded_turn = reloaded["turns"][0]
    assert reloaded_turn["question"] == "Day 7 smoke test question"

    # Result dict was reconstructed correctly
    reloaded_result = reloaded_turn["result"]
    assert reloaded_result is not None
    assert reloaded_result["model_used"] == "gpt-4o-mini"
    assert float(reloaded_result["cost_usd"]) == 0.001
    assert len(reloaded_result["citations"]) == 1
    assert reloaded_result["citations"][0]["chunk_id"] == "cc-test"

    # Cleanup — CASCADE nukes the turn too
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM conversations WHERE id = %s", (conv["id"],))
    db_conn.commit()