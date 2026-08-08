"""Corpus top-up — teach the engine the law it does not have.

The local corpus is 792 chunks across 9 sources and it is demonstrably
incomplete: `cc-1204` (porte-fort), `cc-1344` (mise en demeure) and `cc-494-12`
(habilitation familiale) are all outside it, and `data/corpus_manifest.yaml`
carries a standing TODO naming CGI 641 among others. An obligation anchored
outside the corpus can never be verified, which caps what the engine may
conclude — so the gap is a limit on the analysis, not a cosmetic one.

**The corpus is manifest-driven, and that is the whole answer.** Each source is
declared in `data/corpus_manifest.yaml` with a `fetch_strategy`, and
`ingestion.build` runs fetch → parse → chunk → index from it.
`fetch_strategy: article` already exists for single articles, so covering one
missing article is a four-line entry.

**What this module must not do.** It must never append to `data/chunks.csv` or
keep a private supplement. That CSV feeds BM25 and Chroma through
`load_index()`, and `ingestion/load.py` already carries a desync detector for
exactly that condition — rows in one and not the other. Extending the manifest
and re-running the existing pipeline regenerates both together, keeps one
authoritative corpus, and improves Plane II's retrieval as a side effect. Plane
I stays the only writer of statute data; this module only ever proposes to it.

Three levels, increasingly committed:

    python -m investigator.corpus --case-id X             report gaps
    python -m investigator.corpus --case-id X --propose    write a manifest patch
    python -m investigator.corpus --case-id X --apply      merge, fetch, reindex

`--apply` is behind its own flag because it mutates a Plane I artifact and
triggers a re-index.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from investigator import catalog as catalog_mod
from investigator import config, graph as graph_mod
from investigator.schema import Obligation

log = logging.getLogger(__name__)

MANIFEST = config.PROJECT_ROOT / "data" / "corpus_manifest.yaml"
PROPOSAL = config.PROJECT_ROOT / "data" / "corpus_manifest.proposed.yaml"

# Chunk-id prefix -> (official code name, LEGITEXT id). The name must be the
# one Légifrance uses, because it goes into the `NOM_CODE` search facet: the
# manifest's own short label ("CGI") matches nothing there. The LEGITEXT id is
# what the manifest needs once an entry is written. An unmapped prefix is
# reported, never guessed at.
CODE_TEXT_IDS = {
    "cc": ("Code civil", "LEGITEXT000006070721"),
    "cgi": ("Code général des impôts", "LEGITEXT000006069577"),
    "ca": ("Code des assurances", "LEGITEXT000006073984"),
    "cp": ("Code pénal", "LEGITEXT000006070719"),
}


@dataclass(frozen=True)
class AnchorGap:
    """One catalog anchor with no chunk in the local corpus."""

    obligation_id: str
    chunk_id: str | None
    legiarti_id: str | None
    ref: str
    kind: str  # "source" | "also_anchored" | "penal"

    @property
    def manifest_key(self) -> str:
        base = (self.chunk_id or self.ref).lower()
        return "topup_" + "".join(ch if ch.isalnum() else "_" for ch in base).strip("_")

    @property
    def code_prefix(self) -> str | None:
        return self.chunk_id.split("-")[0] if self.chunk_id else None


# ========== detection ==========


def anchors_of(obligation: Obligation) -> list[tuple[str, str | None, str | None]]:
    """Every anchor an obligation declares, as `(kind, chunk_id, legiarti_id)`.

    One definition, imported by `passes/search.py`, so the two cannot drift on
    what counts as an anchor.
    """
    out: list[tuple[str, str | None, str | None]] = [
        ("source", obligation.source.chunk_id, obligation.source.legiarti_id)
    ]
    out += [("also_anchored", a.chunk_id, None) for a in obligation.also_anchored]
    out += [("penal", a.chunk_id, None) for a in obligation.penal_anchors]
    return out


def detect_gaps(catalog: catalog_mod.Catalog, statute: dict[str, dict]) -> list[AnchorGap]:
    """Anchors the local corpus cannot resolve, deterministically ordered.

    A statute source with no `chunk_id` counts as a gap even though it is
    legitimately declared that way: the point of the report is to list what the
    engine cannot currently verify, whatever the reason.
    """
    gaps: list[AnchorGap] = []
    for obligation in catalog.ordered():
        for kind, chunk_id, legiarti_id in anchors_of(obligation):
            if chunk_id and chunk_id in statute:
                continue
            if chunk_id is None and obligation.source.kind != "statute":
                continue  # a contract clause needs no corpus anchor
            if chunk_id is None and legiarti_id is None:
                continue
            gaps.append(
                AnchorGap(
                    obligation_id=obligation.obligation_id,
                    chunk_id=chunk_id,
                    legiarti_id=legiarti_id,
                    ref=obligation.source.ref,
                    kind=kind,
                )
            )
    return sorted(gaps, key=lambda g: (g.obligation_id, g.kind, g.chunk_id or ""))


# ========== resolution ==========


def resolve(gap: AnchorGap, client=None) -> str | None:
    """The LEGIARTI id for a gap, or None if it cannot be established.

    An anchor that already carries one needs no network. Otherwise PISTE is
    asked, and **a failure returns None**: a synthesised identifier is worse
    than a missing one because it looks verified, and fabricated citations are
    a failure this project has a documented history of.
    """
    if gap.legiarti_id:
        return gap.legiarti_id
    if client is None or not gap.chunk_id:
        return None

    prefix = gap.code_prefix
    if prefix not in CODE_TEXT_IDS:
        log.warning("corpus: no LEGITEXT mapped for prefix %r (%s)", prefix, gap.chunk_id)
        return None
    code_name, _ = CODE_TEXT_IDS[prefix]
    number = gap.chunk_id.split("-", 1)[1]
    try:
        return client.resolve_article(code_name, number)
    except Exception as exc:  # network, auth, shape — all "unresolved"
        log.warning("corpus: PISTE could not resolve %s (%s): %s", gap.chunk_id, number, exc)
        return None


# ========== proposal ==========


def build_proposal(gaps: list[AnchorGap], client=None) -> tuple[dict, list[AnchorGap]]:
    """`(manifest fragment, still-unresolved gaps)`.

    Article-level entries by default. Pulling the whole "Des obligations"
    section to reach `cc-1204` would add well over a hundred articles nobody
    asked for; promoting an entry to `fetch_strategy: section` stays a human
    edit in the manifest.
    """
    sources: dict[str, dict] = {}
    unresolved: list[AnchorGap] = []
    for gap in gaps:
        article_id = resolve(gap, client)
        if not article_id:
            unresolved.append(gap)
            continue
        label = CODE_TEXT_IDS.get(gap.code_prefix or "", ("", ""))[0] or gap.ref
        sources[gap.manifest_key] = {
            "label": label,
            "fetch_strategy": "article",
            "article_id": article_id,
            "note": f"top-up requis par l'obligation {gap.obligation_id} ({gap.ref})",
        }
    return {"sources": sources}, unresolved


def write_proposal(fragment: dict, path: Path = PROPOSAL) -> str:
    path.write_text(
        "# Proposition de complément de corpus — générée par investigator.corpus.\n"
        "# À relire avant fusion : `--apply` fusionne ce fichier dans\n"
        "# data/corpus_manifest.yaml puis relance ingestion.build sur ces clés.\n\n"
        + yaml.safe_dump(fragment, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return str(path)


# ========== apply ==========


def apply_proposal(proposal_path: Path = PROPOSAL, manifest_path: Path = MANIFEST) -> list[str]:
    """Merge the proposal into the manifest and fetch only the new keys.

    Returns the source keys added. Existing keys are never overwritten — the
    manifest is hand-maintained and a top-up must not silently redefine a
    section someone chose deliberately.
    """
    if not proposal_path.exists():
        raise FileNotFoundError(
            f"{proposal_path} does not exist — run with --propose first and review it"
        )
    proposal = yaml.safe_load(proposal_path.read_text(encoding="utf-8")) or {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest.setdefault("sources", {})

    added = []
    for key, spec in (proposal.get("sources") or {}).items():
        if key in manifest["sources"]:
            log.info("corpus: %s already in the manifest, left alone", key)
            continue
        manifest["sources"][key] = spec
        added.append(key)

    if not added:
        log.info("corpus: nothing to add")
        return []

    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    log.info("corpus: added %d source(s) to the manifest: %s", len(added), ", ".join(added))

    # Re-run the existing pipeline for the new keys only. This regenerates
    # chunks.csv AND the Chroma index together, which is the whole reason the
    # top-up goes through the manifest rather than appending to the CSV.
    cmd = ["uv", "run", "python", "-m", "ingestion.build", "--sources", *added]
    log.info("corpus: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=config.PROJECT_ROOT, check=True)
    return added


# ========== CLI ==========


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and close statute-corpus gaps.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--propose", action="store_true", help="write a manifest patch")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="merge the reviewed patch and re-run ingestion for the new sources",
    )
    parser.add_argument(
        "--offline", action="store_true", help="skip PISTE; report only what is already known"
    )
    args = parser.parse_args()

    case_graph = graph_mod.load_graph(args.case_id)
    case_catalog = catalog_mod.load(args.case_id)
    gaps = detect_gaps(case_catalog, case_graph.statute)

    print(f"corpus · case_id={args.case_id} · {len(case_catalog)} obligation(s) · "
          f"{len(case_graph.statute)} chunks locaux · {len(gaps)} ancre(s) non résolue(s)")
    for gap in gaps:
        print(f"  - {gap.obligation_id} [{gap.kind}] {gap.chunk_id or gap.ref} "
              f"legiarti={gap.legiarti_id or '?'}")

    if not (args.propose or args.apply):
        return

    client = None
    if not args.offline:
        try:
            from ingestion.piste import PisteClient

            client = PisteClient.from_env()
        except Exception as exc:
            log.warning("corpus: PISTE unavailable (%s) — resolving from catalog ids only", exc)

    fragment, unresolved = build_proposal(gaps, client)
    path = write_proposal(fragment)
    print(f"\nproposition : {path} ({len(fragment['sources'])} source(s))")
    for gap in unresolved:
        print(f"  non résolu : {gap.obligation_id} {gap.chunk_id or gap.ref} "
              "— identifiant à confirmer sur Légifrance, jamais deviné")

    if args.apply:
        added = apply_proposal()
        print(f"fusionné et réindexé : {', '.join(added) if added else 'rien à ajouter'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
