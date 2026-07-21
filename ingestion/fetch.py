"""Fetch corpus articles from PISTE and write raw JSON files to disk."""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.piste import PisteClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Manifest path, raw output dir, and fetched-at stamp.
MANIFEST_PATH = Path("data/corpus_manifest.yaml")
RAW_DIR = Path("data/raw")
STAMP = RAW_DIR / "fetched_at.txt"


# manifest helpers

def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Read corpus manifest YAML and return parsed manifest data."""
    with path.open() as f:
        return yaml.safe_load(f)


# fetch helpers

def enumerate_ids(client: PisteClient, source_key: str, spec: dict) -> list[str]:
    """Choose the fetch strategy for a source and return article ids."""
    strategy = spec["fetch_strategy"]
    if strategy == "section":
        ids = client.list_articles_in_section(spec["section_id"], spec["parent_text_id"])
    elif strategy == "loda":
        ids = client.list_articles_in_loda(spec["text_id"])
    elif strategy == "jorf":
        ids = client.list_articles_in_jorf(spec["text_id"])
    elif strategy == "article":
        ids = [spec["article_id"]]
    else:
        raise ValueError(f"unknown fetch_strategy: {strategy}")
    log.info("  %s: %d article(s)", source_key, len(ids))
    return ids


def fetch_source(
    client: PisteClient,
    source_key: str,
    ids: list[str],
    out_dir: Path,
    limit: int | None = None,
) -> None:
    """Fetch articles by id and save each response as a raw JSON file."""
    src_dir = out_dir / source_key
    src_dir.mkdir(parents=True, exist_ok=True)
    to_fetch = ids[:limit] if limit else ids
    for aid in tqdm(to_fetch, desc=source_key, unit="art"):
        dest = src_dir / f"{aid}.json"
        if dest.exists():
            continue
        try:
            body = client.get_article(aid)
        except Exception as e:
            log.warning("  failed %s: %s", aid, e)
            continue
        dest.write_text(json.dumps(body, ensure_ascii=False, indent=2))


# command-line entrypoint

def main() -> None:
    """CLI entrypoint: load env, manifest, enumerate ids, fetch raw articles, and write a stamp."""
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", help="subset of manifest keys")
    ap.add_argument("--limit", type=int, default=None, help="per-source article cap")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    sources = args.sources or list(manifest["sources"].keys())

    client = PisteClient.from_env()
    log.info("PISTE env: %s", client.env)

    for key in sources:
        spec = manifest["sources"][key]
        try:
            ids = enumerate_ids(client, key, spec)
        except Exception as e:
            log.error("enumeration failed for %s: %s", key, e)
            continue
        fetch_source(client, key, ids, args.out, limit=args.limit)

    args.out.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(datetime.now(timezone.utc).isoformat())
    log.info("done. stamp: %s", STAMP)


if __name__ == "__main__":
    main()