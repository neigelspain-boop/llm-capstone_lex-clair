"""Fictional persona vocabulary for the anonymiser (Plane Ib · offline, ADR #62).

anonymize.py replaces every resolved entity with a pseudonym. Which pseudonym
is a presentation decision, and this module owns it.

The original vocabulary was "Personne A" / "Organisme B" — deliberately
obvious placeholders, chosen so a reader could never mistake a substitute for
a real person (ADR #59). It works as a privacy device and fails as a document:
a succession argument written in letters cannot be read, and the showcase case
opened with lines like "Nus-propriétaires Personne W ép. [nom] · Personne N".
Nobody can follow who owes what to whom.

Named fictional characters restore readability **without weakening** the
original property, and arguably strengthen it. "Personne A" is unmistakable
but unreadable; a plausible substitute name like "Marc Dupont" would be
readable but invites a reader to believe it and can collide with a real
person. A character from a well-known cartoon is both readable AND
unmistakably fictional — no reader will take "Maître Beerus, notaire" for a
real notaire, and no real French notaire is named Beerus.

**This module contains no real identifiers and must never contain any.**
The bindings — which real person_id becomes which character — are a
re-identification key, and live with the other keys under the gitignored
source case directory (`_persona_roster.json`), never here. This file holds
only the fiction and the loader.

Two things the roster must get right, both learned from the real corpus:

  1. **Merging.** The resolver splits one real person across several
     person_ids (a bare given name, a full civil-status form, a married
     name). Left alone, the three grandchildren of the showcase case render
     as eight different characters and the family tree reads as nonsense.
     `merge_groups` collapses them onto one character each.
  2. **Gate safety.** anonymize.py's verification gate re-runs the
     structured-PII regexes over its own output, so a replacement that looks
     like real PII fails the build. Fictional addresses are therefore shaped
     to be regex-inert — no "<number> <street type>" opening, no bare 5-digit
     run. See assert_gate_safe().
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.dossier.index import DOSSIER_DIR


# ========== module constants ==========

ROSTER_FILENAME = "_persona_roster.json"

# Characters held back for the LLM residual sweep, so a name it introduces is
# visibly distinct from a name the deterministic layers placed. Never assigned
# to a person_id by the roster.
RESIDUAL_NATURAL_POOL = ("Krilin", "Ten Shin Han", "Chaozu", "Yajirobé", "Dendé")
RESIDUAL_LEGAL_POOL = ("Organisation Pilaf", "Patrouilleurs Galactiques")

# Fallback pools for entities the roster does not name. Ordered, and consumed
# deterministically, so an unrostered person still gets a stable character
# rather than being left in place — silence here would be a leak.
FALLBACK_NATURAL_POOL = (
    "Yamcha", "Piccolo", "Tenshinhan", "Bulma", "Chichi", "Videl", "Bardock",
    "Raditz", "Nappa", "Freezer", "Cell", "Broly", "Zarbon", "Dodoria",
    "Ginyu", "Jeice", "Burter", "Recoome", "Guldo", "Tarblé", "Paragus",
    "Kaiô Shin", "Kibito", "Mister Satan", "Boubou", "Pan", "Maron", "Puar",
)
FALLBACK_LEGAL_POOL = (
    "Capsule Corporation", "Pilaf Gestion", "Banque du Mont Paozu",
    "Caisse Céleste des Dépôts", "Caisse de retraite Kamé",
    "Compagnie Kaiô", "Réseau Kaiô", "Tour de Karin", "Temple de Dendé",
    "Armée du Ruban Rouge", "Team Ginyu", "Corp. Namek",
)

# Places. Communes and planets both name "where a thing is", which is what
# makes the substitution read naturally in a French deed.
# Deliberately excludes the canonical Saiyan homeworld: its name contains a
# real surname from the source corpus as a substring. The gate's word-anchored
# matching makes that harmless, but a fiction that reads as a hit on a naive
# substring search is not worth keeping when the pool is this large.
PLACE_POOL = (
    "Namek", "Yardrat", "Tsufuru", "Kanassa", "Konats", "Mont Paozu",
)


# ========== gate-safety contract ==========

# Replacement shapes that anonymize.py's own verification gate would flag as
# surviving PII. A fictional address must not match these, or the build fails
# on text this module produced.
_UNSAFE_SHAPES: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"\b\d{1,4}\s?(?:bis|ter)?\s*"
            r"(?:bd|boulevard|av|avenue|rue|place|impasse|chemin|route|quai|allee|allée)\b",
            re.IGNORECASE,
        ),
        "reads as a street address (<number> <street type>)",
    ),
    (re.compile(r"\b\d{5}\b"), "contains a bare 5-digit run (reads as a postal code)"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "reads as an email address"),
    (re.compile(r"\b[A-Z]{2}\d{2}"), "reads as an IBAN/ICS prefix"),
    (re.compile(r"<<"), "reads as a passport machine-readable zone"),
)


def assert_gate_safe(values) -> None:
    """Raise if any replacement value would be re-detected as PII.

    The gate re-runs the structured-PII patterns over the anonymiser's own
    output (anonymize.verify_anonymization), so "12 avenue Kamé, 77000 Namek"
    is not a charming fictional address — it is a build failure. Fictional
    addresses must be written in shapes the regexes cannot match:
    "Résidence Kamé, Namek", "secteur 4, Namek".

    Called at roster load time so the failure surfaces at the roster, where
    it can be fixed, rather than 54 documents later in the gate.
    """
    for value in values:
        for pattern, why in _UNSAFE_SHAPES:
            if pattern.search(value):
                raise ValueError(
                    f"persona value {value!r} is not gate-safe: it {why}. "
                    "The verification gate re-scans anonymised output, so this "
                    "replacement would fail the build it is meant to pass."
                )


# ========== roster ==========

def roster_path(source_case_id: str) -> Path:
    """Where the person_id -> character bindings live.

    Under the SOURCE case directory, which is gitignored, for the same reason
    _anonymization_map.json is: the bindings map a character back to a real
    person, so they are a re-identification key.
    """
    return DOSSIER_DIR / source_case_id / ROSTER_FILENAME


@dataclass
class Roster:
    """One source case's persona bindings.

    `bindings` is expanded across merge groups at load time, so a lookup by
    ANY person_id in a group returns the group's character — callers never
    need to canonicalise first to get the right name.

    `canonical` maps every person_id in a merge group to the group's first id.
    Callers need it for the other half of merging: collapsing several
    person records into one output record.

    `token_overrides` maps a folded bare word-token to a replacement, and
    exists for the shared family surname. Grouping by person_id cannot decide
    what a surname owned by nine different people should become, so the
    generated scheme degrades it to "[nom]" — which is exactly how the
    showcase ended up reading "Personne W ép. [nom]". An override lets the
    family share one fictional surname and the tree read as a tree.
    """

    bindings: dict[str, str] = field(default_factory=dict)
    canonical: dict[str, str] = field(default_factory=dict)
    token_overrides: dict[str, str] = field(default_factory=dict)

    def character_for(self, person_id: str) -> str | None:
        return self.bindings.get(person_id)

    def head_of(self, person_id: str) -> str:
        return self.canonical.get(person_id, person_id)


def load_roster(source_case_id: str) -> Roster:
    """Load the persona roster for one source case.

    Returns an empty Roster when no roster file exists. That is a legitimate
    state, not an error: anonymize.py falls back to its generated pools and
    still anonymises everything. A roster is a readability upgrade, never a
    privacy prerequisite — which is why this loader is permissive where
    load_extra_identifiers is strict.

    Raises on a roster that exists but is unusable: a value that would fail
    the gate, or one character bound to two people who are not declared to be
    the same person (which would silently merge a notaire with an heir in the
    published case).
    """
    path = roster_path(source_case_id)
    if not path.exists():
        return Roster()

    raw = json.loads(path.read_text(encoding="utf-8"))
    bindings: dict[str, str] = {
        k: v for k, v in (raw.get("bindings") or {}).items() if not k.startswith("_")
    }
    token_overrides: dict[str, str] = {
        k.lower(): v for k, v in (raw.get("token_overrides") or {}).items()
        if not k.startswith("_")
    }
    assert_gate_safe([*bindings.values(), *token_overrides.values()])

    canonical: dict[str, str] = {}
    for group in raw.get("merge_groups") or []:
        if not group:
            continue
        head = group[0]
        for person_id in group:
            canonical[person_id] = head

    # Expand a group's binding to every member, so the roster can name a
    # person once instead of repeating the character for each of the three
    # person_ids the resolver split them into.
    for person_id, head in canonical.items():
        character = bindings.get(head) or bindings.get(person_id)
        if character:
            bindings[person_id] = character

    # Two people sharing one character is the failure this module exists to
    # avoid — it is how the generated scheme collapsed a decedent and a
    # surviving heir onto one label. Check it at load, not at read.
    seen: dict[str, str] = {}
    for person_id, character in bindings.items():
        head = canonical.get(person_id, person_id)
        if character in seen and seen[character] != head:
            raise ValueError(
                f"persona {character!r} is bound to two unmerged people "
                f"({seen[character]!r} and {head!r}). Add them to the same "
                "merge_groups entry if they are the same person, or give one "
                "of them a different character."
            )
        seen[character] = head

    return Roster(bindings=bindings, canonical=canonical, token_overrides=token_overrides)
