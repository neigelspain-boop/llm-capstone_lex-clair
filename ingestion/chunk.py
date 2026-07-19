"""articles.csv → chunks.csv  (Plane I · offline · identity transform for V1);
    in fetch → parse → CHUNK → index. """

import argparse
import logging
from pathlib import Path
from typing import Literal
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Module constants for repo-local article and chunk CSV paths.
ROOT = Path(__file__).resolve().parents[1]
ARTICLES_CSV = ROOT / "data" / "articles.csv"
CHUNKS_CSV = ROOT / "data" / "chunks.csv"

# chunk_articles: pure article-level chunking with schema normalization and validation.

def chunk_articles(
    articles: pd.DataFrame, granularity: Literal["article", "alinea"] = "article"
) -> pd.DataFrame:
    """Return chunk data at the requested granularity.

    Currently only article-level chunking is implemented. The function is
    intentionally pure: it validates schema invariants and returns a copy of
    the input DataFrame so callers can mutate the result safely.
    """
    if granularity == "article":
        result = articles.copy()

        OPTIONAL_STRING_COLUMNS = ["num", "titre", "section_path", "url"]
        for col in OPTIONAL_STRING_COLUMNS:
            if col in result.columns:
                result[col] = result[col].fillna("")
    elif granularity == "alinea":
        raise NotImplementedError("alinea chunking is not supported yet")
    else:
        raise ValueError(
            f"granularity must be 'article' or 'alinea', got {granularity!r}"
        )

    if "chunk_id" not in result.columns:
        raise ValueError("articles DataFrame must contain a chunk_id column")
    if result["chunk_id"].duplicated().any():
        dupes = int(result["chunk_id"].duplicated().sum())
        raise ValueError(f"chunk_id contains {dupes} duplicate values")

    if "texte" not in result.columns:
        raise ValueError("articles DataFrame must contain a texte column")
    if result["texte"].isnull().any():
        nulls = int(result["texte"].isnull().sum())
        raise ValueError(f"texte column contains {nulls} null values")

    return result

# build_chunks: read articles.csv, build chunks, write chunks.csv, and preserve row identity.

def build_chunks(
    src: Path = ARTICLES_CSV,
    out: Path = CHUNKS_CSV,
    granularity: Literal["article", "alinea"] = "article",
) -> pd.DataFrame:
    """Read articles CSV, apply chunking, and write the resulting chunks CSV."""
    log.info("reading articles from %s", src)
    articles = pd.read_csv(src, keep_default_na=False)
    chunks = chunk_articles(articles, granularity=granularity)

    if granularity == "article" and len(chunks) != len(articles):
        log.error(
            "identity chunking changed row count from %d to %d",
            len(articles),
            len(chunks),
        )
        raise AssertionError(
            "article-level chunking must preserve row count for identity mode"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(out, index=False)
    log.info("wrote %d chunk rows to %s", len(chunks), out)
    return chunks


# main: CLI entrypoint for chunk CSV generation.

def main() -> None:
    """Command-line entrypoint for chunk generation."""
    parser = argparse.ArgumentParser(description="Build chunk CSV from articles CSV.")
    parser.add_argument("--src", type=Path, default=ARTICLES_CSV, help="source articles CSV")
    parser.add_argument("--out", type=Path, default=CHUNKS_CSV, help="output chunks CSV")
    parser.add_argument(
        "--granularity",
        choices=["article", "alinea"],
        default="article",
        help="chunking granularity",
    )
    args = parser.parse_args()
    build_chunks(args.src, args.out, granularity=args.granularity)


# STEP 6 — if __name__ == "__main__": main()

if __name__ == "__main__":
    main()