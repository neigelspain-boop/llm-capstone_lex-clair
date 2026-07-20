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

from pathlib import Path

import pandas as pd
import pytest


# Smoke-test artifact paths for the canonical ingestion corpus.
ROOT = Path(__file__).resolve().parents[1]
ARTICLES_CSV = ROOT / "data" / "articles.csv"
CHUNKS_CSV = ROOT / "data" / "chunks.csv"

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