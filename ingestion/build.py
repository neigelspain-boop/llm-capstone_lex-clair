"""End-to-end ingestion pipeline: fetch → parse → chunk → index."""
from __future__ import annotations

import argparse
import logging
import time

from dotenv import load_dotenv

# Import offline-only ingestion stages. Do not import load here.
from ingestion import fetch, parse, chunk, index

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# stage runner for pipeline steps

def _run_stage(name: str, func, *args, **kwargs):
    """Run one pipeline stage with elapsed-time logging."""
    log.info("=" * 70)
    log.info("STAGE START · %s", name)
    log.info("=" * 70)
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception:
        log.error("STAGE FAILED · %s (after %.1fs)", name, time.time() - t0)
        raise
    log.info("STAGE DONE · %s (%.1fs)", name, time.time() - t0)
    return result


# fetch stage without parse-argv indirection

def _run_fetch(sources: list[str] | None, limit: int | None) -> None:
    """Enumerate manifest sources and download raw article JSON."""
    load_dotenv()
    manifest = fetch.load_manifest()
    keys = sources or list(manifest["sources"].keys())

    client = fetch.PisteClient.from_env()
    log.info("PISTE env: %s", client.env)

    for key in keys:
        spec = manifest["sources"][key]
        try:
            ids = fetch.enumerate_ids(client, key, spec)
        except Exception as e:
            log.error("enumeration failed for %s: %s", key, e)
            continue
        fetch.fetch_source(client, key, ids, fetch.RAW_DIR, limit=limit)

    fetch.RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch.STAMP.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


# CLI orchestrator for the full offline pipeline.

def main() -> None:
    """Command-line entrypoint for the end-to-end ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Build the lex-clair corpus from scratch: fetch → parse → chunk → index."
    )
    parser.add_argument("--skip-fetch", action="store_true", help="skip PISTE fetch stage")
    parser.add_argument("--skip-parse", action="store_true", help="skip raw JSON → articles.csv")
    parser.add_argument("--skip-chunk", action="store_true", help="skip articles.csv → chunks.csv")
    parser.add_argument("--skip-index", action="store_true", help="skip BM25 + Chroma build")
    parser.add_argument(
        "--sources", nargs="*", default=None, help="subset of manifest keys for fetch"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="per-source article cap for fetch smoke tests"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="embedding device; inferred from torch.cuda.is_available() if omitted",
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("build starting · flags: skip_fetch=%s skip_parse=%s skip_chunk=%s skip_index=%s",
             args.skip_fetch, args.skip_parse, args.skip_chunk, args.skip_index)

    if not args.skip_fetch:
        _run_stage("FETCH", _run_fetch, args.sources, args.limit)
    else:
        log.info("SKIP · FETCH")

    if not args.skip_parse:
        _run_stage("PARSE", parse.parse_all)
    else:
        log.info("SKIP · PARSE")

    if not args.skip_chunk:
        _run_stage("CHUNK", chunk.build_chunks)
    else:
        log.info("SKIP · CHUNK")

    if not args.skip_index:
        _run_stage("INDEX", index.build_all, device=args.device)
    else:
        log.info("SKIP · INDEX")

    log.info("build complete · total elapsed %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()

