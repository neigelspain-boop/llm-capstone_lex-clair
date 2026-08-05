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
    """V1 chunk.py is identity — row count must be preserved for the statute
    corpus. Scoped to non-dossier rows: chunks.csv also carries dossier
    chunks (v2, ADR #39) that have no articles.csv counterpart.
    """
    statute_chunks = chunks[~chunks["chunk_id"].str.startswith("dossier-")]
    assert len(statute_chunks) == len(articles), (
        f"row-count drift: articles={len(articles)}, chunks={len(statute_chunks)}"
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
    dropping url from parse.py. Scoped to non-dossier rows: dossier chunks
    (v2, ADR #39) use local extraction file paths as url by design, not
    Legifrance links.
    """
    statute_chunks = chunks[~chunks["chunk_id"].str.startswith("dossier-")]
    with_url = (statute_chunks["url"].str.startswith("http")).sum()
    assert with_url >= 0.9 * len(statute_chunks), (
        f"only {with_url}/{len(statute_chunks)} statute chunks have URLs — expected ≥90%"
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

    # at least one citation links to Legifrance — the corpus is now
    # dual (statute + dossier, ADR #39), so a definitional query can
    # legitimately also surface a non-Legifrance dossier citation; the
    # docstring's contract is "at least one", not "every citation"
    assert len(result["citations"]) >= 1, "no citations returned"
    legifrance_citations = [
        c for c in result["citations"]
        if c["url"].startswith("http") and "legifrance" in c["url"]
    ]
    assert legifrance_citations, "no Legifrance citation among the results"

    # cost + timing sanity
    assert result["cost_usd"] < 0.01, f"cost too high: ${result['cost_usd']}"
    assert result["elapsed_seconds"] < 60, f"too slow: {result['elapsed_seconds']}s"

    # pipeline shape
    assert result["chunks_retrieved"] == 20
    assert result["chunks_reranked"] == 5
    assert result["model_used"] == "openai/gpt-4o-mini"


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


# ========== Day B / Deliverable B1: source_scope filtering (ADR #41) ==========

# Shared vocabulary across all three groups (statute, dossier-demo, dossier-acme)
# is deliberate: a single query must activate BM25 + vector signal for every
# group, or the blended/case-scope tests below would be trivially true.
_MIXED_CORPUS_QUERY = "quasi-usufruit et restitution aux nu-propriétaires"


def _build_mixed_corpus() -> pd.DataFrame:
    """6-row synthetic corpus: 2 statute + 2 dossier-demo + 2 dossier-acme rows.

    Schema matches ingestion/index.py::REQUIRED_COLUMNS. Entirely synthetic --
    never touches real data/chunks.csv or real case content.
    """
    rows = [
        {"chunk_id": "cc-587", "source": "cc_successions", "source_label": "Code civil",
         "num": "587", "titre": "Du quasi-usufruit", "section_path": "Livre III",
         "texte": "L'usufruitier jouit des choses consomptibles à charge de rendre, "
                  "à la fin de l'usufruit, des choses de même quantité et qualité, "
                  "ou leur valeur estimée : ceci définit le quasi-usufruit.",
         "url": ""},
        {"chunk_id": "cc-758", "source": "cc_successions", "source_label": "Code civil",
         "num": "758", "titre": "Droits du conjoint survivant", "section_path": "Livre III",
         "texte": "Les droits du conjoint survivant dans la succession incluent l'usufruit "
                  "ou la quasi-usufruit sur les biens du défunt selon la convention successorale.",
         "url": ""},
        {"chunk_id": "dossier-demo-conv-c001", "source": "dossier-demo", "source_label": "Dossier demo",
         "num": "", "titre": "Convention de quasi-usufruit", "section_path": "",
         "texte": "Convention de quasi-usufruit signée entre les héritiers du dossier demo : "
                  "la quasi-usufruitière conserve la libre disposition des sommes, à charge de "
                  "restitution de la valeur équivalente aux nu-propriétaires au terme de la convention.",
         "url": ""},
        {"chunk_id": "dossier-demo-conv-c002", "source": "dossier-demo", "source_label": "Dossier demo",
         "num": "", "titre": "Convention de quasi-usufruit", "section_path": "",
         "texte": "Le notaire du dossier demo a rédigé la convention de quasi-usufruit précisant "
                  "les modalités de restitution dues par le quasi-usufruitier aux nu-propriétaires héritiers.",
         "url": ""},
        {"chunk_id": "dossier-acme-conv-c001", "source": "dossier-acme", "source_label": "Dossier acme",
         "num": "", "titre": "Convention de quasi-usufruit", "section_path": "",
         "texte": "Convention de quasi-usufruit du dossier acme : le quasi-usufruitier s'engage "
                  "à restituer aux nu-propriétaires la valeur des sommes reçues en quasi-usufruit.",
         "url": ""},
        {"chunk_id": "dossier-acme-conv-c002", "source": "dossier-acme", "source_label": "Dossier acme",
         "num": "", "titre": "Convention de quasi-usufruit", "section_path": "",
         "texte": "Le dossier acme comporte une clause de quasi-usufruit conventionnel définissant "
                  "les droits du quasi-usufruitier et les garanties dues aux nu-propriétaires.",
         "url": ""},
    ]
    return pd.DataFrame(rows, columns=[
        "chunk_id", "source", "source_label", "num", "titre", "section_path", "texte", "url",
    ])


def _embed_into_chroma(chunks: pd.DataFrame, chroma_dir: Path) -> None:
    """Manually embed + write chunks into an isolated Chroma collection.

    Mirrors ingestion/dossier/index.py::append_to_chroma's manual
    PersistentClient + BGEM3FlagModel.encode + collection.add pattern --
    NOT ingestion.index.build_chroma, which unconditionally shutil.rmtree()s
    the shared CHROMA_DIR global. That's unsafe to reuse here even behind a
    monkeypatch: this fixture must never risk touching real data/chroma/.
    """
    import chromadb
    from FlagEmbedding import BGEM3FlagModel
    from ingestion import index as ingestion_index

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=ingestion_index.COLLECTION, metadata={"hnsw:space": "cosine"},
    )
    model = BGEM3FlagModel(ingestion_index.EMBED_MODEL_ID, use_fp16=False, device="cpu")
    texts = chunks["texte"].astype(str).tolist()
    dense_vecs = model.encode(
        texts, return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )["dense_vecs"]
    metadatas = (
        chunks[["source", "source_label", "num", "titre", "section_path", "url"]]
        .astype(str)
        .to_dict(orient="records")
    )
    collection.add(
        ids=chunks["chunk_id"].tolist(),
        embeddings=[[float(v) for v in vec] for vec in dense_vecs],
        documents=texts,
        metadatas=metadatas,
    )


@pytest.fixture(scope="module")
def mixed_corpus_retriever(tmp_path_factory):
    """Real BM25 + real Chroma + real BGE-M3 retriever over an isolated,
    fully synthetic 6-row corpus (2 statute + 2 dossier-demo + 2 dossier-acme
    chunks). Never touches data/chunks.csv or data/chroma/ -- ADR #39's
    `private`-case incident is exactly the risk this isolation avoids.
    """
    from ingestion import index as ingestion_index
    from ingestion.load import load_index

    tmp_dir = tmp_path_factory.mktemp("mixed_corpus")
    chunks = _build_mixed_corpus()
    csv_path = tmp_dir / "chunks.csv"
    chunks.to_csv(csv_path, index=False)

    chroma_dir = tmp_dir / "chroma"
    _embed_into_chroma(chunks, chroma_dir)

    return load_index(
        src=csv_path,
        chroma_dir=chroma_dir,
        collection_name=ingestion_index.COLLECTION,
        device="cpu",
    )


@pytest.mark.slow
def test_source_scope_statute_filters_out_dossier(mixed_corpus_retriever) -> None:
    """source_scope="statute" (the new default) must exclude every dossier chunk."""
    hits = mixed_corpus_retriever.search(_MIXED_CORPUS_QUERY, k=6, source_scope="statute")
    assert hits, "expected at least one statute hit"
    dossier_hits = [h["chunk_id"] for h in hits if h["chunk_id"].startswith("dossier-")]
    assert not dossier_hits, f"statute scope leaked dossier chunks: {dossier_hits}"


@pytest.mark.slow
def test_source_scope_case_returns_only_that_case(mixed_corpus_retriever) -> None:
    """source_scope="case:demo" must return only dossier-demo-* chunks, never acme or statute."""
    hits = mixed_corpus_retriever.search(_MIXED_CORPUS_QUERY, k=6, source_scope="case:demo")
    assert hits, "expected at least one dossier-demo hit"
    assert all(h["chunk_id"].startswith("dossier-demo-") for h in hits), (
        f"case:demo scope leaked non-demo chunks: {[h['chunk_id'] for h in hits]}"
    )


@pytest.mark.slow
def test_source_scope_blended_returns_mix(mixed_corpus_retriever) -> None:
    """source_scope="blended" applies no filter -- statute and dossier chunks both surface."""
    hits = mixed_corpus_retriever.search(_MIXED_CORPUS_QUERY, k=6, source_scope="blended")
    ids = [h["chunk_id"] for h in hits]
    assert any(not cid.startswith("dossier-") for cid in ids), f"no statute chunk in blended results: {ids}"
    assert any(cid.startswith("dossier-") for cid in ids), f"no dossier chunk in blended results: {ids}"


def test_source_scope_invalid_raises() -> None:
    """Bad source_scope raises ValueError before any BM25/Chroma/embed work happens."""
    from ingestion.load import HybridRetriever

    retriever = HybridRetriever(bm25=None, vectors=None, embed_model=None, chunks=None)
    with pytest.raises(ValueError, match="invalid source_scope"):
        retriever.search("x", source_scope="typo")


def test_flow_run_default_source_scope_is_statute(monkeypatch) -> None:
    """flow.run(query) with no source_scope arg must retrieve with 'statute', not 'blended'.

    Since ADR #42 (Deliverable B2), a None source_scope routes through
    rag.router.route_query instead of a hardcoded default, so the router
    must be mocked here too — otherwise this "fast" test would fire a real
    network call to Haiku via OpenRouter.
    """
    from unittest.mock import MagicMock

    from rag import flow
    from rag.router import RouteDecision

    mock_route_query = MagicMock(
        return_value=RouteDecision(
            intent="statute_lookup", source_scope="statute", confidence="high", rationale="test",
        )
    )
    mock_retrieve = MagicMock(return_value=[])
    monkeypatch.setattr(flow, "route_query", mock_route_query)
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", mock_retrieve)
    monkeypatch.setattr(
        flow.generate, "generate",
        lambda p, model_key=None: ("mocked answer", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "model_id": "openai/gpt-4o-mini", "model_key": model_key or "gpt-4o-mini"}),
    )

    flow.run("Qu'est-ce que le quasi-usufruit ?")

    assert mock_retrieve.call_args.kwargs["source_scope"] == "statute"


def test_flow_run_passes_source_scope_through(monkeypatch) -> None:
    """flow.run(query, source_scope=...) must pass the value through to retrieve.retrieve."""
    from unittest.mock import MagicMock

    from rag import flow

    mock_retrieve = MagicMock(return_value=[])
    monkeypatch.setattr(flow.rewrite, "rewrite", lambda q: q)
    monkeypatch.setattr(flow.retrieve, "retrieve", mock_retrieve)
    monkeypatch.setattr(
        flow.generate, "generate",
        lambda p, model_key=None: ("mocked answer", {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "model_id": "openai/gpt-4o-mini", "model_key": model_key or "gpt-4o-mini"}),
    )

    flow.run("Qu'est-ce que le quasi-usufruit ?", source_scope="case:demo")

    assert mock_retrieve.call_args.kwargs["source_scope"] == "case:demo"

# ========== dossier chunk merge at load time (ADR #58) ==========

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _statute_columns() -> list[str]:
    return ["chunk_id", "source", "source_label", "num", "titre",
            "section_path", "texte", "url", "etat", "legiarti_id"]


def test_read_dossier_chunks_merges_every_case(tmp_path) -> None:
    from ingestion.load import _read_dossier_chunks

    _write_csv(tmp_path / "demo" / "chunks.csv", [
        {"chunk_id": "dossier-demo-d1-c001", "source": "dossier-demo", "texte": "a"},
    ])
    _write_csv(tmp_path / "acme" / "chunks.csv", [
        {"chunk_id": "dossier-acme-d1-c001", "source": "dossier-acme", "texte": "b"},
        {"chunk_id": "dossier-acme-d1-c002", "source": "dossier-acme", "texte": "c"},
    ])

    frames = _read_dossier_chunks(_statute_columns(), tmp_path)
    merged = pd.concat(frames, ignore_index=True)

    assert len(merged) == 3
    assert set(merged["chunk_id"]) == {
        "dossier-demo-d1-c001", "dossier-acme-d1-c001", "dossier-acme-d1-c002",
    }


def test_read_dossier_chunks_aligns_columns_without_nan(tmp_path) -> None:
    """Statute-only columns must fill as "" for dossier rows, never NaN
    (project-wide pandas convention)."""
    from ingestion.load import _read_dossier_chunks

    _write_csv(tmp_path / "demo" / "chunks.csv", [
        {"chunk_id": "dossier-demo-d1-c001", "source": "dossier-demo", "texte": "a"},
    ])

    merged = pd.concat(_read_dossier_chunks(_statute_columns(), tmp_path), ignore_index=True)

    assert list(merged.columns) == _statute_columns()
    assert merged["etat"].iloc[0] == ""
    assert merged["legiarti_id"].iloc[0] == ""
    assert not merged.isna().any().any()


def test_read_dossier_chunks_skips_empty_and_missing(tmp_path) -> None:
    from ingestion.load import _read_dossier_chunks

    assert _read_dossier_chunks(_statute_columns(), tmp_path / "nope") == []

    (tmp_path / "zerobyte").mkdir(parents=True, exist_ok=True)
    (tmp_path / "zerobyte" / "chunks.csv").write_text("", encoding="utf-8")
    _write_csv(tmp_path / "empty" / "chunks.csv", [])
    _write_csv(tmp_path / "real" / "chunks.csv", [
        {"chunk_id": "dossier-real-d1-c001", "source": "dossier-real", "texte": "x"},
    ])
    frames = _read_dossier_chunks(_statute_columns(), tmp_path)
    assert len(frames) == 1
    assert frames[0]["chunk_id"].iloc[0] == "dossier-real-d1-c001"


def test_merged_dossier_rows_keep_scope_predicate_prefix(tmp_path) -> None:
    """The merged rows must still satisfy _scope_predicate, which filters on
    the dossier-<case_id>- chunk_id prefix — this is what keeps statute scope
    from leaking dossier text once the two corpora share one row table."""
    from ingestion.load import _read_dossier_chunks, _scope_predicate

    _write_csv(tmp_path / "demo" / "chunks.csv", [
        {"chunk_id": "dossier-demo-d1-c001", "source": "dossier-demo", "texte": "a"},
    ])
    _write_csv(tmp_path / "acme" / "chunks.csv", [
        {"chunk_id": "dossier-acme-d1-c001", "source": "dossier-acme", "texte": "b"},
    ])
    merged = pd.concat(_read_dossier_chunks(_statute_columns(), tmp_path), ignore_index=True)
    ids = list(merged["chunk_id"])

    assert [cid for cid in ids if _scope_predicate("statute")(cid)] == []
    assert [cid for cid in ids if _scope_predicate("case:demo")(cid)] == ["dossier-demo-d1-c001"]
    assert len([cid for cid in ids if _scope_predicate("dossier")(cid)]) == 2


def test_dossier_text_never_enters_the_tracked_statute_csv() -> None:
    """The whole point of ADR #58: data/chunks.csv is git-tracked, so a single
    dossier row in it would put real client names one `git add` from
    publication. Guards against the retired append_to_statute_chunks_csv
    being reintroduced."""
    import ingestion.dossier.index as dossier_index
    from ingestion.index import CHUNKS_CSV

    assert not hasattr(dossier_index, "append_to_statute_chunks_csv")

    if CHUNKS_CSV.exists():
        tracked = pd.read_csv(CHUNKS_CSV, keep_default_na=False)
        leaked = [c for c in tracked["chunk_id"] if str(c).startswith("dossier-")]
        assert not leaked, f"dossier rows found in tracked {CHUNKS_CSV}: {leaked[:5]}"


def test_chroma_scope_filter_prevents_candidate_starvation() -> None:
    """Scope filtering must be pushed into Chroma, not applied after a fixed
    candidate pull.

    Once statute and dossier share one collection, a dossier-flavoured query
    can fill every one of the k*3 candidate slots with dossier chunks, and
    source_scope="statute" — the DEFAULT — then filters them all out and
    returns nothing. Server-side filtering makes that impossible (ADR #58).
    """
    from ingestion.load import HybridRetriever

    chunks = pd.DataFrame([
        {"chunk_id": "cc-587", "source": "cc_usufruit", "texte": "a"},
        {"chunk_id": "dossier-private-d1-c001", "source": "dossier-private", "texte": "b"},
        {"chunk_id": "dossier-acme-d1-c001", "source": "dossier-acme", "texte": "c"},
    ]).set_index("chunk_id", drop=False)
    r = HybridRetriever(bm25=None, vectors=None, embed_model=None, chunks=chunks)

    assert r._dossier_sources() == ["dossier-acme", "dossier-private"]
    assert r._statute_sources() == ["cc_usufruit"]
    # Allowlist, not denylist (ADR #63) — see the fail-closed test below.
    assert r._chroma_scope_filter("statute") == {"source": {"$in": ["cc_usufruit"]}}
    assert r._chroma_scope_filter("dossier") == {
        "source": {"$in": ["dossier-acme", "dossier-private"]}
    }
    assert r._chroma_scope_filter("case:private") == {"source": "dossier-private"}
    assert r._chroma_scope_filter("blended") is None


def test_chroma_scope_filter_excludes_sources_the_row_table_does_not_know() -> None:
    """An unrecognised `source` must be excluded from every narrow scope,
    not admitted into the default one (ADR #63).

    The denylist this replaces derived BOTH lists from the row table, so a
    vector whose case had been removed or renamed was absent from
    dossier_sources and the {"$nin": dossier_sources} clause therefore let it
    through — into source_scope="statute", the default. That is how 824
    orphaned `dossier-vitrine` vectors ended up displacing real statute hits
    inside the k*3 budget and starving the reranker's candidate pool.

    An allowlist inverts the failure: a source nobody declared is a source
    nobody retrieves.
    """
    from ingestion.load import HybridRetriever

    chunks = pd.DataFrame([
        {"chunk_id": "cc-587", "source": "cc_usufruit", "texte": "a"},
        {"chunk_id": "dossier-private-d1-c001", "source": "dossier-private", "texte": "b"},
    ]).set_index("chunk_id", drop=False)
    r = HybridRetriever(bm25=None, vectors=None, embed_model=None, chunks=chunks)

    # "dossier-ghost" is indexed in Chroma but absent from the row table.
    statute_where = r._chroma_scope_filter("statute")
    assert "dossier-ghost" not in statute_where["source"]["$in"]
    assert statute_where["source"]["$in"] == ["cc_usufruit"]

    dossier_where = r._chroma_scope_filter("dossier")
    assert "dossier-ghost" not in dossier_where["source"]["$in"]

    # A case that was never indexed must match nothing, not everything.
    assert r._chroma_scope_filter("case:ghost") == {"source": "dossier-ghost"}


def test_search_drops_orphan_candidates_instead_of_raising() -> None:
    """A chunk_id in Chroma but not in the row table must not crash a query.

    Hydration does `self.chunks.loc[cid]`, so an orphan raised KeyError and
    took down the whole request — reproducibly on source_scope="blended",
    which by design sends no server-side filter. Orphans are dropped before
    the top-k cut so a full k is still returned when candidates exist.
    """
    import numpy as np
    from ingestion.load import HybridRetriever

    chunks = pd.DataFrame([
        {"chunk_id": f"cc-{i}", "source": "cc_usufruit", "texte": "a", "num": str(i),
         "titre": "t", "section_path": "s", "source_label": "Code civil", "url": "u"}
        for i in range(3)
    ]).set_index("chunk_id", drop=False)

    class _Model:
        def encode(self, *a, **k):
            return {"dense_vecs": [np.zeros(4)]}

    class _Vectors:
        # Two real ids sandwiching an orphan, as a desynced collection returns.
        def query(self, **kwargs):
            return {"ids": [["cc-0", "dossier-ghost-d1-c001", "cc-1", "cc-2"]]}

    r = HybridRetriever(
        bm25=None, vectors=_Vectors(), embed_model=_Model(), chunks=chunks,
    )

    hits = r.search("q", k=3, mode="vector", source_scope="blended")
    assert [h["chunk_id"] for h in hits] == ["cc-0", "cc-1", "cc-2"]


def test_reconcile_chroma_reports_and_deletes_orphans_only_when_applied() -> None:
    """reconcile must be a dry run by default — Chroma is real state."""
    from ingestion.index import reconcile_chroma

    class _Collection:
        def __init__(self):
            self.deleted: list[str] = []

        def get(self, include=None):
            return {
                "ids": ["cc-0", "dossier-ghost-c001"],
                "metadatas": [{"source": "cc_usufruit"}, {"source": "dossier-ghost"}],
            }

        def delete(self, ids):
            self.deleted.extend(ids)

    row_ids = {"cc-0", "cc-unvectorised"}

    dry = _Collection()
    result = reconcile_chroma(row_ids, dry, apply=False)
    assert result["orphans"] == 1
    assert result["orphans_by_source"] == {"dossier-ghost": 1}
    assert result["unvectorised"] == 1
    assert result["deleted"] == 0
    assert dry.deleted == [], "dry run must not delete"

    live = _Collection()
    applied = reconcile_chroma(row_ids, live, apply=True)
    assert applied["deleted"] == 1
    assert live.deleted == ["dossier-ghost-c001"]


def test_chroma_scope_filter_on_a_statute_only_corpus() -> None:
    """With no dossier indexed, statute scope still sends its allowlist.

    Before ADR #63 this asserted that no `where` clause was sent at all. That
    compatibility guarantee is exactly what let stray dossier vectors reach
    the default scope, so the contract is now the allowlist — inert in
    effect on a clean corpus, protective on a desynced one.
    """
    from ingestion.load import HybridRetriever

    chunks = pd.DataFrame([
        {"chunk_id": "cc-587", "source": "cc_usufruit", "texte": "a"},
    ]).set_index("chunk_id", drop=False)
    r = HybridRetriever(bm25=None, vectors=None, embed_model=None, chunks=chunks)

    assert r._dossier_sources() == []
    assert r._chroma_scope_filter("statute") == {"source": {"$in": ["cc_usufruit"]}}
    # No dossier sources to allow: fall through to the chunk_id predicate
    # rather than send an empty $in, which some backends read as match-all.
    assert r._chroma_scope_filter("dossier") is None
    assert r._chroma_scope_filter("blended") is None
