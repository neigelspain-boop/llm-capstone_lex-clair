"""Obligation catalog loading, validation, and generic/overlay merge.

A catalog is the fixed denominator that makes absence a signal. Because a
missing obligation produces no output at all, a catalog that fails to load must
fail loudly and before a cycle starts — never partially, and never by silently
dropping the entry that would have found something.

Three files feed one merged catalog, and the split is a privacy rule rather
than a preference:

- `investigator/catalogs/generic_fr_succession.yaml` — committed,
  `person_free: true`, statute and deontology only.
- `data/dossier/{case_id}/obligations.yaml` — the per-case overlay carrying
  contract clauses, which cannot be citation-verified without a case. Ignored
  by default for every case but `demo`/`vitrine` (ADR #66).

Overlay wins on `obligation_id` collision, so a case may tighten a generic
obligation without forking the generic file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from investigator import config
from investigator.schema import Obligation

log = logging.getLogger(__name__)


# ========== file model ==========


class CatalogFile(BaseModel):
    """One YAML catalog file, as authored."""

    catalog_id: str
    catalog_version: str
    jurisdiction: str = "FR"
    # Asserted by the file, verified by a test. Only a person-free catalog is
    # eligible to be committed outside a case directory.
    person_free: bool = False
    obligations: list[Obligation] = Field(default_factory=list)


# ========== merged catalog ==========


@dataclass(frozen=True)
class Catalog:
    """Every obligation in force for one case, with provenance."""

    obligations: dict[str, Obligation]
    origin: dict[str, str]
    files: tuple[Path, ...]

    def ordered(self) -> list[Obligation]:
        """Obligations in a stable order.

        Iteration order is part of the cycle's determinism budget: two runs
        over an unchanged case must evaluate the same obligations in the same
        sequence, or a budget cutoff truncates a different tail each time.
        """
        return [self.obligations[oid] for oid in sorted(self.obligations)]

    def get(self, obligation_id: str) -> Obligation | None:
        return self.obligations.get(obligation_id)

    def __len__(self) -> int:
        return len(self.obligations)


# ========== loading ==========


def load_file(path: Path) -> CatalogFile:
    """Parse and validate one catalog file. Raises on any defect."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not valid YAML — {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    try:
        return CatalogFile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid catalog — {exc}") from exc


def load(
    case_id: str,
    dossier_dir: Path | None = None,
    generic_path: Path | None = None,
) -> Catalog:
    """Merge the generic catalog with this case's overlay.

    A missing generic catalog is tolerated only while it does not yet exist;
    a missing overlay is normal — a case may run on generic obligations alone.
    Neither absence is inferred from a parse failure, which always raises.
    """
    paths = config.CasePaths.for_case(case_id, dossier_dir=dossier_dir)
    generic = generic_path or config.GENERIC_CATALOG

    obligations: dict[str, Obligation] = {}
    origin: dict[str, str] = {}
    loaded: list[Path] = []

    for path in (generic, paths.case_catalog):
        if not path.exists():
            log.info("catalog: %s absent, skipping", path)
            continue
        catalog_file = load_file(path)
        for obligation in catalog_file.obligations:
            oid = obligation.obligation_id
            if oid in obligations:
                log.info(
                    "catalog: %s overridden by %s", oid, path.name
                )
            obligations[oid] = obligation
            origin[oid] = str(path)
        loaded.append(path)

    if not loaded:
        log.warning("catalog: no catalog file found for case %s", case_id)

    return Catalog(obligations=obligations, origin=origin, files=tuple(loaded))
