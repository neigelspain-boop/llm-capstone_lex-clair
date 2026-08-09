"""Role-designation vocabulary for the anonymiser (Plane Ib · offline, ADR #78).

A second substitution convention beside personas.py. Where that module
replaces a real person with an ordinary French name ("Camille MARTIN"), this
one replaces them with the office they hold in the case
("heritier_nu_proprietaire_1", "notaire_instrumentaire").

Why two conventions rather than one. They are answers to different questions.
The published *dossier* is meant to be read as a succession file, and a file
whose parties are labelled by function does not read as one — that is exactly
the failure ADR #59 recorded for "Personne A". The published *investigation
transcript* is the opposite: it is already written in role language
throughout ("le notaire soussigné", "le nu-propriétaire", "l'établissement
teneur du compte"), and its findings turn on function rather than identity —
`conv-remise-extrait-au-gestionnaire` yields two separate findings precisely
because one asset holder is a bank and the other an SCPI manager. Naming
those two holders "Banque Régionale du Centre" and "Valorim Gestion" would
add invented specificity to a document that needs none, and would invite a
reader to believe a fiction. A role designation is not an identifier
(ADR #77), so it needs no fiction notice and cannot be mistaken for a party.

Consequences of that difference, all of them deliberate:

  * `keep_real_legal_persons` is FALSE here. The persona convention keeps
    banks and public bodies under their real names because a national bank
    does not identify a private family. In role mode there is no reason to
    keep them: the transcript reasons about "the asset holder", never about
    which one, and the combination of a named SCPI, a named bank and a named
    étude re-identifies the case on its own.
  * Fallbacks are ordinal ("personne_1"), not lettered. The whole register is
    lowercase-underscore slugs, and a stray "Personne A" in that company
    reads as a different kind of thing — which it is.
  * Gate safety is free. A slug has no digit-space-word shape, so none of
    personas.assert_gate_safe's PII patterns can fire on one.

**This module contains no real identifiers and must never contain any.** The
bindings live with the other re-identification keys under the gitignored
source case directory (`_role_roster.json`, `_role_identifiers.json`), for
the same reason `_persona_roster.json` does.

Both files are DERIVED, not hand-written from scratch — see
`derive_role_register`. The derivation is a proposal that must be read before
use, in the same spirit as `obligations.proposed.yaml` (ADR #74): it can tell
that a person carries the role `heritier`, but not that two spellings of a
surname are one person. That judgement is already recorded in
`_persona_roster.json`'s merge groups, so the derivation reuses those rather
than guessing at them again.

CLI: python -m ingestion.dossier.roles --case-id private [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from ingestion.dossier import personas
from ingestion.dossier.index import DOSSIER_DIR


# ========== module constants ==========

ROLE_ROSTER_FILENAME = "_role_roster.json"
ROLE_IDENTIFIERS_FILENAME = "_role_identifiers.json"

# The vocabulary a Roster carries in role mode. Empty pools on purpose: there
# is no finite stock of role names to draw from, so every unrostered entity
# goes straight to the ordinal fallback.
ROLE_VOCABULARY = {
    "natural_pool": (),
    "legal_pool": (),
    "natural_label": "personne",
    "legal_label": "organisme",
    "natural_placeholder": "[nom]",
    "legal_placeholder": "[organisme]",
    "ordinal_fallback": True,
}

# Role ids that describe a position in the case rather than the office a party
# holds, and are therefore a poor label when a person carries something more
# specific. "heritier" is worn by six people here; "heritier_nu_proprietaire"
# is worn by three and is the one the obligations are written against.
_WEAK_ROLE_IDS = {
    "heritier", "mandant", "mandataire", "personne_decedee", "titulaire_contrat",
    "heritier_successoral", "heritier_ayant_droit", "heritier_beneficiaire",
    "heritier_representant", "heritier_petit_enfant", "representant_legal",
    "debit_successoral", "plaignant", "donateur",
}

# Generic place words that must not become the distinctive half of a commune
# slug ("SEINE-ET-MARNE" -> "commune", not "seine").
_PLACE_MARKERS = ("commune", "voie", "lieu")


# ========== text helpers ==========

def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _fold(value)).strip("_")


# ========== register loading ==========

def roster_path(source_case_id: str) -> Path:
    return DOSSIER_DIR / source_case_id / ROLE_ROSTER_FILENAME


def identifiers_path(source_case_id: str) -> Path:
    return DOSSIER_DIR / source_case_id / ROLE_IDENTIFIERS_FILENAME


def load_roster(source_case_id: str) -> personas.Roster:
    """Load the role roster, or an empty one carrying the role vocabulary.

    Permissive about absence for the same reason personas.load_roster is: a
    roster is a readability upgrade, and the ordinal fallback still anonymises
    every entity without one. What it is NOT permissive about is
    keep_real_legal_persons — see below.
    """
    path = roster_path(source_case_id)
    if not path.exists():
        return personas.Roster(**ROLE_VOCABULARY)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("keep_real_legal_persons"):
        raise ValueError(
            f"{path} sets keep_real_legal_persons — role mode must not. The "
            "persona convention keeps a bank's real name because it does not "
            "identify a private family; in role mode the named SCPI, bank and "
            "étude together re-identify the case, and nothing in the "
            "transcript needs them."
        )
    return personas.parse_roster(raw, **ROLE_VOCABULARY)


def load_identifiers(source_case_id: str) -> dict[str, str]:
    """The role-mode counterpart of anonymize.load_extra_identifiers.

    Raises when absent, for that function's reason exactly: an empty table
    would silently disable a whole deterministic layer while still reporting
    success. Derive it with `derive_role_register` rather than writing it by
    hand.
    """
    path = identifiers_path(source_case_id)
    if not path.exists():
        raise RuntimeError(
            f"no {ROLE_IDENTIFIERS_FILENAME} for case {source_case_id!r} — refusing "
            "to anonymise without the role identifier table. Derive it with "
            f"`python -m ingestion.dossier.roles --case-id {source_case_id}`."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ========== derivation ==========

def _role_label(role_ids: list[str]) -> str | None:
    """Pick the role designation that best names a person.

    Prefers a specific office over a generic position: the resolver attaches
    both "heritier" and "heritier_nu_proprietaire" to the same person, and
    only the second one is what the obligations are written against. Ties
    break alphabetically so the derivation is deterministic.
    """
    strong = sorted(r for r in role_ids if r and r not in _WEAK_ROLE_IDS)
    if strong:
        return strong[0]
    weak = sorted(r for r in role_ids if r)
    return weak[0] if weak else None


def _derive_bindings(persons: list[dict], canonical: dict[str, str]) -> dict[str, str]:
    """person_id -> role slug, numbered within each role.

    Numbering is by sorted head id rather than file order, so re-deriving
    against a re-run persons.jsonl does not renumber the case.
    """
    roles_by_head: dict[str, set[str]] = defaultdict(set)
    types: dict[str, str] = {}
    for record in persons:
        person_id = record.get("person_id", "")
        head = canonical.get(person_id, person_id)
        for assignment in record.get("role_assignments") or []:
            if assignment.get("role_id"):
                roles_by_head[head].add(assignment["role_id"])
        types.setdefault(head, record.get("person_type", "natural_person"))

    label_of: dict[str, str] = {}
    for head in sorted(roles_by_head):
        label = _role_label(sorted(roles_by_head[head]))
        if label:
            label_of[head] = _slug(label)

    counts: dict[str, int] = defaultdict(int)
    for label in label_of.values():
        counts[label] += 1

    seen: dict[str, int] = defaultdict(int)
    bindings: dict[str, str] = {}
    for head in sorted(label_of):
        label = label_of[head]
        # A role held by exactly one party needs no ordinal — "notaire_
        # instrumentaire" is more readable than "notaire_instrumentaire_1"
        # and is what the office is actually called.
        if counts[label] == 1:
            bindings[head] = label
        else:
            seen[label] += 1
            bindings[head] = f"{label}_{seen[label]}"
    return bindings


def _derive_identifiers(extras: dict[str, str], token_overrides: dict[str, str]) -> dict[str, str]:
    """Re-target the persona identifier table onto role slugs.

    Works by equivalence class, not by key: `_extra_identifiers.json` already
    groups its 161 keys onto ~54 replacement values, and those groupings are a
    reviewed human decision about which real things are the same thing. Every
    key sharing a persona value keeps sharing one role slug.

    Markers already in bracket form ("[email]", "[BIC]") pass through — they
    are not personas and there is nothing to re-target.
    """
    by_value: dict[str, list[str]] = defaultdict(list)
    for key, value in extras.items():
        by_value[value].append(key)

    # Which persona values name a place. Detected from the persona place pool
    # rather than guessed from the key, so a commune keeps its class even when
    # the key is a département or an island group.
    place_values = {v for v in by_value if any(p in v for p in personas.PLACE_POOL)}

    out: dict[str, str] = {}
    place_n = 0
    person_n = 0
    org_n = 0
    for value in sorted(by_value):
        keys = by_value[value]
        if value.startswith("["):
            slug = value
        elif value in place_values:
            place_n += 1
            slug = f"commune_{place_n}"
        elif value in extras and extras[value] == value:
            # A key mapped to itself: a named financial product the persona
            # run deliberately kept (SCPI NOTAPIERRE, EDISSIMMO). Role mode
            # does not keep them — a named SCPI plus a named bank is most of
            # the way to the case.
            org_n += 1
            slug = f"produit_{org_n}"
        elif any(m in _fold(value) for m in _PLACE_MARKERS) or value.startswith("des ") \
                or value.startswith("du ") or value.startswith("archipel"):
            place_n += 1
            slug = f"voie_{place_n}"
        elif value.isupper() or " " in value:
            person_n += 1
            slug = f"personne_{person_n}"
        else:
            person_n += 1
            slug = f"personne_{person_n}"
        for key in keys:
            out[key] = slug

    # Bare tokens the persona roster names explicitly. They are not in the
    # extras table but must resolve in role mode too, or a lone given name
    # survives into the transcript.
    for token, persona in token_overrides.items():
        if token.startswith("_"):
            continue
        out.setdefault(token, out.get(persona, "[nom]"))
    return out


def derive_role_register(source_case_id: str, force: bool = False) -> dict:
    """Write `_role_roster.json` and `_role_identifiers.json` for one case.

    Reuses `_persona_roster.json`'s merge groups verbatim. Those record which
    person_ids are one real person — a judgement made by reading the corpus,
    which no amount of re-derivation recovers, and getting it wrong is how a
    decedent and a surviving heir end up on one label.

    Refuses to overwrite an existing register unless `force`, because the
    derivation is a PROPOSAL: roles come from persons.jsonl, so the labels are
    only as good as the resolver's role assignments, and the intended workflow
    is derive once, read, hand-correct, then keep the corrected file.
    """
    case_dir = DOSSIER_DIR / source_case_id
    persons_path = case_dir / "persons.jsonl"
    if not persons_path.exists():
        raise RuntimeError(f"no persons.jsonl for case {source_case_id!r}")

    persona_roster = personas.load_roster(source_case_id)
    raw_persona = json.loads(
        (case_dir / personas.ROSTER_FILENAME).read_text(encoding="utf-8")
    ) if (case_dir / personas.ROSTER_FILENAME).exists() else {}

    persons = [
        json.loads(line) for line in persons_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bindings = _derive_bindings(persons, persona_roster.canonical)

    from ingestion.dossier.anonymize import load_extra_identifiers  # local: avoids a cycle

    identifiers = _derive_identifiers(
        load_extra_identifiers(source_case_id), persona_roster.token_overrides,
    )

    roster_doc = {
        "_readme": (
            "Role designations for the investigation transcript (ADR #78). Derived by "
            "`python -m ingestion.dossier.roles`; hand-correct freely, it is not "
            "regenerated unless --force. merge_groups is copied from "
            "_persona_roster.json and is the reviewed grouping decision."
        ),
        "keep_real_legal_persons": False,
        "merge_groups": raw_persona.get("merge_groups") or [],
        "token_overrides": {
            k: identifiers.get(k, "[nom]")
            for k in persona_roster.token_overrides
        },
        "bindings": bindings,
    }
    identifiers_doc = {
        "_readme": (
            "Role-mode counterpart of _extra_identifiers.json (ADR #78). Keys are REAL "
            "identifiers — this file is a re-identification key and never leaves the "
            "gitignored source case directory."
        ),
        **identifiers,
    }

    for path, doc in ((roster_path(source_case_id), roster_doc),
                      (identifiers_path(source_case_id), identifiers_doc)):
        if path.exists() and not force:
            raise RuntimeError(
                f"{path} already exists — refusing to overwrite a reviewed register. "
                "Pass --force to regenerate."
            )
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "source_case_id": source_case_id,
        "bindings": len(bindings),
        "token_overrides": len(roster_doc["token_overrides"]),
        "identifiers": len(identifiers),
    }


# ========== CLI ==========

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the role-designation register for one source case (ADR #78)."
    )
    parser.add_argument("--case-id", required=True, help="source case to derive from")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing register (it is meant to be hand-corrected)",
    )
    args = parser.parse_args()

    summary = derive_role_register(args.case_id, force=args.force)
    print(
        f"role register · case_id={summary['source_case_id']} "
        f"bindings={summary['bindings']} token_overrides={summary['token_overrides']} "
        f"identifiers={summary['identifiers']}"
    )
    print(f"  {roster_path(args.case_id)}")
    print(f"  {identifiers_path(args.case_id)}")
    print("  Read both before publishing — the role labels come from the resolver.")


if __name__ == "__main__":
    main()
