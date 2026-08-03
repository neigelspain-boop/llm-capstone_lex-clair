"""Fictional persona vocabulary for the anonymiser (Plane Ib · offline, ADR #62).

anonymize.py replaces every resolved entity with a pseudonym. Which pseudonym
is a presentation decision, and this module owns it.

The convention here is the one courts and anonymised casebooks use: **ordinary,
neutral French names**. It was reached by eliminating the two alternatives.

"Personne A" / "Organisme B" was the first attempt (ADR #59) — deliberately
obvious placeholders, chosen so a reader could never mistake a substitute for
a real person. It works as a privacy device and fails as a document: a
succession argument written in letters cannot be read, and the showcase case
opened with "Nus-propriétaires Personne W ép. [nom] · Personne N".

Well-known cartoon characters were the second. They are readable in isolation
and unmistakably fictional, which looked like a strict improvement. In a
55-document succession file they are worse than the letters: eight family
members sharing one invented surname plus five exotic names for the étude is
more to hold in your head, not less, for a human reader and for the model
reasoning over the corpus.

Ordinary names are what the domain already does. "Mme Camille MARTIN a mis en
demeure Maître Claire DUBOIS le 30 juin 2026" parses on first read because it
is shaped exactly like the sentence it replaced. The cost is real and is paid
elsewhere: a plausible name invites a reader to believe it, so the published
case MUST carry a visible notice that every name is fictional. That notice is
part of this decision, not a nicety attached to it.

**This module contains no real identifiers and must never contain any.**
The bindings — which real person_id becomes which name — are a
re-identification key, and live with the other keys under the gitignored
source case directory (`_persona_roster.json`), never here. This file holds
only the fictional vocabulary and the loader.

Three things the roster must get right, all learned from the real corpus:

  1. **Merging.** The resolver splits one real person across several
     person_ids (a bare given name, a full civil-status form, a married
     name). Left alone, the three grandchildren of the showcase case render
     as eight different people and the family tree reads as nonsense.
     `merge_groups` collapses them onto one name each.
  2. **Token overrides.** A person maps to a full form ("Camille MARTIN"),
     but documents also use bare tokens — a lone surname, a lone given name.
     Mapping a bare given name to the full form yields "Camille MARTIN
     MARTIN"; mapping the shared family surname by ownership yields "[nom]",
     since nine people own it. `token_overrides` names each bare token
     explicitly, which is the only thing that produces a readable family.
  3. **Gate safety.** anonymize.py's verification gate re-runs the
     structured-PII regexes over its own output, so a replacement that looks
     like real PII fails the build. Fictional locations are therefore shaped
     to be regex-inert — no "<number> <street type>" opening, no bare 5-digit
     run. See assert_gate_safe().

Note that no civility title is ever baked into a name here. The source
documents supply their own "Maître" / "Madame", and a title inside the
replacement prints twice ("Maître Maître DUBOIS" — observed 13 times).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.dossier.index import DOSSIER_DIR


# ========== module constants ==========

ROSTER_FILENAME = "_persona_roster.json"

# Names held back for the LLM residual sweep, so a name it introduces is
# visibly distinct from a name the deterministic layers placed. Never assigned
# to a person_id by the roster.
RESIDUAL_NATURAL_POOL = ("Yves DELMAS", "Anne PERROT", "Marc VIDAL")
RESIDUAL_LEGAL_POOL = ("Société Ardennes", "Groupe Solane")

# Fallback pools for entities the roster does not name. Ordered, and consumed
# deterministically, so an unrostered person still gets a stable name rather
# than being left in place — silence here would be a leak.
#
# Surnames are drawn from the ordinary French stock a casebook would use.
# Deliberately excluded: any surname that is also a common French word
# ("Petit", "Leblanc", "Roy"), because replacement is word-anchored on the
# REAL token, not this one, but the residual-identifier report reads the
# OUTPUT — and a pseudonym that doubles as vocabulary makes that report
# unreadable in exactly the way this whole change is meant to fix.
FALLBACK_NATURAL_POOL = (
    "MERCIER", "FONTAINE", "CHEVALIER", "GIRARD", "LEFEVRE", "ROUSSEAU",
    "VINCENT", "MULLER", "LEMAIRE", "DUPUIS", "MARCHAND", "GAUTIER",
    "PERRIN", "MORIN", "NICOLAS", "HENRY", "CLEMENT", "RENAUD", "COLIN",
    "BRUNET", "BARBIER", "SCHMITT", "ARNAUD", "PICARD", "CARON", "GUERIN",
    "BRETON", "AUBERT", "OLIVIER", "REY",
)
# Only used when the roster does NOT set keep_real_legal_persons. Neutral
# trading names with no real-world referent.
FALLBACK_LEGAL_POOL = (
    "Gérimmo Patrimoine", "Valorim Gestion", "Banque Régionale du Centre",
    "Caisse Nationale de Dépôts", "Caisse de retraite Interpro",
    "Compagnie Générale d'Énergie", "Réseau Distribution", "Société Ardennes",
    "Groupe Solane", "Mutuelle du Levant", "Cabinet Verriers", "Foncière Adrets",
)

# Communes. A short, closed set: the demo reads better when the geography is
# three places a reader can keep straight than when every commune in the file
# gets its own invented name.
PLACE_POOL = (
    "Villeneuve", "Beaumont", "Port-Louis",
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

    `token_overrides` maps a folded bare word-token to a replacement. It
    exists for the shared family surname and for bare given names: grouping
    by person_id cannot decide what a surname owned by nine people should
    become, so the generated scheme degrades it to "[nom]" — which is exactly
    how the showcase came to read "Personne W ép. [nom]" — while a bare given
    name resolves to its owner's FULL pseudonym and yields "Camille MARTIN
    MARTIN". Naming each token explicitly is what makes the family readable.

    `keep_real_legal_persons` leaves companies, banks and public bodies under
    their real names. They are not what the anonymisation protects: a national
    bank or a tax office does not identify a private family, and naming them
    makes the published case markedly more concrete. It is a deliberate
    narrowing of the search space for someone already close to the case, and
    it is why the geography in the roster is fictional even when the brand is
    not — otherwise "SIP <commune>" would re-publish the exact locality the
    address redaction just removed.
    """

    bindings: dict[str, str] = field(default_factory=dict)
    canonical: dict[str, str] = field(default_factory=dict)
    token_overrides: dict[str, str] = field(default_factory=dict)
    keep_real_legal_persons: bool = False

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

    return Roster(
        bindings=bindings,
        canonical=canonical,
        token_overrides=token_overrides,
        keep_real_legal_persons=bool(raw.get("keep_real_legal_persons", False)),
    )
