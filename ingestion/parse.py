"""Parse raw JSON articles into a normalized articles CSV."""
from __future__ import annotations
import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Paths for corpus manifest, raw JSON input, and normalized output.
MANIFEST_PATH = Path("data/corpus_manifest.yaml")
RAW_DIR = Path("data/raw")
OUT_CSV = Path("data/articles.csv")

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")

CHUNK_ID_PREFIX = {
    "cc_successions": "cc",
    "cc_usufruit": "cc",
    "cc_responsabilite": "cc",
    "cc_liberalites": "cc",
    "cgi_dettes_defunt": "cgi",
    "ca_assurances_responsabilite": "ca",
    "cp_appropriations_frauduleuses": "cp",
    "ord_45_2590": "ord45-2590",
    "decret_73_609": "d73-609",
    "decret_74_737": "d74-737",
    "decret_2023_1297": "d2023-1297",
}


# text cleaning helpers

def _strip_html(s: str) -> str:
    """Remove HTML markup and collapse whitespace from text fields."""
    if not s:
        return ""
    import html
    s = html.unescape(s)
    s = _HTML_TAG.sub(" ", s)
    return _MULTI_WS.sub(" ", s).strip()


def _ms_to_iso(ms) -> str:
    """Normalize millisecond timestamps to ISO date strings."""
    if ms in (None, "", 0, "0"):
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return ""


def _slugify_num(num: str) -> str:
    """Turn an article number into a compact lowercase identifier."""
    return re.sub(r"\s+", "", num or "").replace(".", "").lower()


def _section_path(article: dict) -> str:
    """Build breadcrumb using PISTE's pre-computed fullSectionsTitre.

    Falls back to walking context.titresTM if the pre-computed string is absent
    (some LODA/JORF responses may not have it).
    """
    import html

    fst = article.get("fullSectionsTitre")
    if fst:
        # decode &gt; → >, &amp; → & etc.
        return html.unescape(fst).strip()

    ctx = article.get("context")
    if isinstance(ctx, dict):
        trail = ctx.get("titresTM", [])
        if isinstance(trail, list) and trail:
            labels = [x.get("titre", "") for x in trail if isinstance(x, dict)]
            return " > ".join(l for l in labels if l).strip()
    return ""


def _titre(article: dict) -> str:
    """Article title — key varies by PISTE endpoint.

    Code articles (Code civil, CGI, etc.) usually don't have titles — they're
    identified by 'num'. LODA articles sometimes do. Empty is the right value
    when neither key is populated.
    """
    for key in ("titre", "titreArt"):
        val = article.get(key)
        if val and isinstance(val, str):
            return val.strip()
    return ""


# article parsing helpers

def parse_one(raw: dict, source_key: str, source_label: str) -> dict | None:
    """Parse one raw JSON object into a normalized article row, or skip it."""
    article = raw.get("article") or raw
    if not isinstance(article, dict):
        return None

    etat = article.get("etat", "")
    if etat and etat not in ("VIGUEUR", "VIGUEUR_DIFF"):
        return None

    num = article.get("num", "") or ""
    texte = _strip_html(article.get("texte", ""))
    if not texte:
        return None

    aid = article.get("id", "")
    prefix = CHUNK_ID_PREFIX.get(source_key, source_key)
    chunk_id = f"{prefix}-{_slugify_num(num)}" if num else f"{prefix}-{aid[-10:].lower()}"

    return {
        "chunk_id": chunk_id,
        "source": source_key,
        "source_label": source_label,
        "num": num,
        "section_path": _section_path(article),
        "titre": _titre(article),
        "texte": texte,
        "etat": etat or "VIGUEUR",
        "date_debut": _ms_to_iso(article.get("dateDebut")),
        "date_fin": _ms_to_iso(article.get("dateFin")),
        "legiarti_id": aid,
        "url": f"https://www.legifrance.gouv.fr/codes/article_lc/{aid}" if aid else "",
    }


# CSV generation

def parse_all(manifest_path=MANIFEST_PATH, raw_dir=RAW_DIR, out_csv=OUT_CSV) -> pd.DataFrame:
    """Load manifest and raw JSON files, normalize articles, and write a CSV."""
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)

    rows: list[dict] = []
    dropped = {"non_vigueur": 0, "empty": 0, "malformed": 0}

    for source_key, spec in manifest["sources"].items():
        src_dir = raw_dir / source_key
        if not src_dir.exists():
            log.warning("no raw dir for %s, skipping", source_key)
            continue
        label = spec["label"]
        for path in tqdm(sorted(src_dir.glob("*.json")), desc=source_key, unit="art"):
            try:
                raw = json.loads(path.read_text())
            except json.JSONDecodeError:
                dropped["malformed"] += 1
                continue
            row = parse_one(raw, source_key, label)
            if row is None:
                article = raw.get("article") or raw
                if article.get("etat") and article["etat"] != "VIGUEUR":
                    dropped["non_vigueur"] += 1
                else:
                    dropped["empty"] += 1
                continue
            rows.append(row)

    df = pd.DataFrame(rows)
    log.info("parsed %d rows (dropped: %s)", len(df), dropped)

    dup = df["chunk_id"].duplicated().sum()
    if dup:
        log.error("%d duplicate chunk_ids", dup)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)
    return df


# command-line entrypoint

def main() -> None:
    """Command-line entrypoint for parsing raw JSON into the CSV output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    parse_all(args.manifest, args.raw, args.out)


if __name__ == "__main__":
    main()