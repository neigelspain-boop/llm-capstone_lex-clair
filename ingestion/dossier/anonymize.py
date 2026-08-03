"""Derive an anonymised, publishable case from a private dossier
(Plane Ib · offline, ADR #59).

Reads DOSSIER_DIR/<source_case_id>/extracted/*.md and writes
DOSSIER_DIR/<target_case_id>/extracted/*.md with every identifier removed,
so the downstream pipeline (facts -> index -> compliance) can run over it
unchanged and produce a committable public showcase.

Anonymisation happens HERE, at the source markdown, and not later: facts.py
already forbids names in extracted facts (actors must be role_ids), but
Fact.verbatim_quote copies source sentences literally, which is how real
names reached the private compliance rationales. Every downstream artifact
inherits whatever this module leaves behind.

Four layers, strongest first — the LLM is the last resort, not the first:

  1. Deterministic replacement of every canonical_name/alias in the source
     case's persons.jsonl (ADR #55). 100% recall on resolved entities.
  2. Structured-PII regexes: emails, phones, IBAN, SIREN/SIRET, postal
     addresses, notarial reference numbers.
  3. Deterministic transforms of the re-identifying *combination*: a fixed
     date offset, a fixed amount scale, and generalisation of named
     financial products. Names alone are not the whole risk — an exact sum
     plus exact dates plus a named SCPI identifies a case on its own.
  4. An LLM sweep for residual free-text identifiers the first three
     missed (an unresolved person, an identifying place or job title).

Then verify_anonymization() re-reads every produced artifact and RAISES on
any surviving identifier. It also reports unrecognised proper nouns for
human review, because the known-entity list cannot prove an unknown
identifier is absent — see its docstring for the limits of that claim.

The pseudonym mapping is a re-identification key. It is written under the
PRIVATE case directory (gitignored), never under the target case, and
persisting it is what makes reruns idempotent: the same person maps to the
same pseudonym across all documents and across runs.

CLI: python -m ingestion.dossier.build --case-id vitrine --step anonymize
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from ingestion.clients import get_openrouter_client
from ingestion.dossier.index import DOSSIER_DIR

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ========== module constants ==========

DEFAULT_SOURCE_CASE_ID = "private"

SWEEP_MODEL_ID = "anthropic/claude-haiku-4.5"
SWEEP_MAX_TOKENS = 8192
SWEEP_TEMPERATURE = 0.0

# Fixed, deterministic transforms. Backwards in time so the showcase reads as
# a historical matter rather than dated in the future. Amounts scale by a
# single factor so ratios between figures in the dossier stay coherent; share
# counts are deliberately NOT scaled (they are not monetary and scaling them
# would break "N parts" arithmetic against the scaled totals).
DATE_OFFSET_DAYS = -829
AMOUNT_SCALE = 1.27

# The minimum length at which a known entity string is safe to replace
# literally. Shorter strings ("CDC", "SCP") collide with ordinary words and
# are handled by the alias list only when they are distinctive.
MIN_ENTITY_LEN = 3

# Filename of the extra-identifier table inside the SOURCE case directory.
EXTRA_IDENTIFIERS_FILENAME = "_extra_identifiers.json"

# Proper nouns that are legitimate legal/domain vocabulary and must NOT be
# flagged as residual identifiers by the verification report.
ALLOWED_PROPER_NOUNS = {
    "code", "civil", "assurances", "cgi", "legifrance", "france", "scpi",
    "sci", "scp", "selarl", "sarl", "sas", "sa", "ue", "rgpd", "insee",
    "page", "article", "articles", "titre", "livre", "section", "chapitre",
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout",
    "septembre", "octobre", "novembre", "decembre", "lundi", "mardi",
    "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "monsieur", "madame", "maitre", "me", "mme", "mm", "m", "dr",
    "personne", "organisme", "cabinet", "etude", "ville", "commune", "rue",
    "region", "notaire", "notaires", "usufruit", "quasi", "succession",
    "tribunal", "cour", "cassation", "chambre", "conseil", "president",
    "euro", "euros", "eur", "objet", "note", "guide", "lecture", "affaire",
    "annexe", "annexes", "piece", "pieces", "dossier", "sommaire",
    # Legal / accounting / numeric vocabulary that is SHOUTED in these
    # documents (headers, signature blocks, amounts written out in words) and
    # would otherwise drown the residual-identifier report.
    "mandant", "mandataire", "proprietaire", "proprietaires", "usufruitier",
    "nu", "nue", "propriete", "actif", "passif", "total", "neant", "bien",
    "biens", "droits", "donateurs", "donataires", "heritier", "heritiers",
    "affirmation", "sincerite", "attention", "acte", "actes", "arrondissement",
    "iban", "siret", "siren", "cedex", "pacs", "ehpad", "francs", "francais",
    "un", "une", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "vingt", "trente", "cent", "cents", "mille", "million",
    "millions", "date", "montant", "objet", "signature", "fait", "lu",
    "approuve", "convention", "successions", "gestion", "part", "parts",
    "abattement", "message", "courrier", "email", "web", "tel", "ttc", "ht",
    "tva", "les", "des", "nous", "vous", "pour", "votre", "notre", "cette",
    "ces", "par", "non", "oui", "bonjour", "madame", "cordialement",
}


# ========== structured PII patterns ==========

# Ordered: more specific patterns first, so an email is not partially eaten
# by the phone pattern.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Machine-readable zone of a scanned passport / ID card ("P<FRA<surname><<
    # <given names>..."). The surname sits inside an unbroken alphanumeric
    # run, so no name-matching layer can reach it. Redact the whole zone
    # rather than trying to parse it.
    (re.compile(r"[A-Z0-9<]*<<[A-Z0-9<]*"), "[MRZ]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{3}\d{4,}\b"), "[ICS]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}\b"), "[IBAN]"),
    (re.compile(r"\b(?:\+33|0)\s?[1-9](?:[\s.-]?\d{2}){4}\b"), "[telephone]"),
    (re.compile(r"\b\d{3}[ ]?\d{3}[ ]?\d{3}[ ]?\d{5}\b"), "[SIRET]"),
    (re.compile(r"\b\d{3}[ ]?\d{3}[ ]?\d{3}\b(?=\s*(?:RCS|SIREN))"), "[SIREN]"),
    # Street line: number + type + name, up to a comma or end of line.
    (
        re.compile(
            r"\b\d{1,4}\s?(?:bis|ter)?\s*"
            r"(?:bd|boulevard|av|avenue|rue|place|impasse|chemin|route|quai|allee|allée)\b"
            r"[^,\n]{0,40}",
            re.IGNORECASE,
        ),
        "[adresse]",
    ),
    # French postal code + commune.
    (re.compile(r"\b\d{5}\s+[A-ZÀ-Ý][\w'\-]+(?:\s+[A-ZÀ-Ý][\w'\-]+)?"), "[code postal] [ville]"),
    # Notarial / invoice reference numbers.
    (re.compile(r"\b[Nn]°\s?[A-Z]?\d[\w/.-]{5,}\b"), "[réf.]"),
    (re.compile(r"\bacte\s+n[°o]\s?\d[\w/.-]*", re.IGNORECASE), "acte n° [réf.]"),
]

_MONTHS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}
_MONTHS_FR_REV = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
    7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre",
    12: "décembre",
}

_DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})(?:er)?\s+(" + "|".join(_MONTHS_FR) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DATE_COMPACT_RE = re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b")

# French money: "195 572 €", "195572,50 EUR". Narrow/non-breaking spaces count
# as thousands separators in the extracted markdown.
_AMOUNT_RE = re.compile(
    r"\b(\d{1,3}(?:[   ]\d{3})+|\d{4,})(?:,(\d{1,2}))?\s?(€|EUR\b|euros\b)",
    re.IGNORECASE,
)


# ========== models ==========

@dataclass
class AnonymizationReport:
    """Outcome of verify_anonymization for one case."""

    case_id: str
    files_scanned: int = 0
    leaks: list[tuple[str, str]] = field(default_factory=list)       # (file, identifier)
    residual_proper_nouns: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.leaks


# ========== text normalisation ==========

def _fold(text: str) -> str:
    """Lowercase + strip accents, so a SHOUTED accented surname, its
    title-case form and its lowercase form all compare equal. Used for both
    replacement matching and leak detection."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


# ========== source-case identifier tables ==========

def load_extra_identifiers(source_case_id: str) -> dict[str, str]:
    """Identifiers that are NOT resolved entities in persons.jsonl and are
    therefore invisible to layer 1 — places, financial products, firm
    fragments, and real people the persons pipeline never resolved.

    Measured against the real corpus, persons.jsonl covered 10 of 18 sampled
    identifiers, so this table closes the deterministic gap rather than
    leaving it to the LLM.

    It lives in the SOURCE case directory, not in this module, because its
    KEYS are real identifiers. Holding them in tracked source would publish
    exactly what this module exists to remove — permanently, in git history —
    and the project's `grep private` pre-commit convention would not catch it,
    because the leak would be in a source file rather than under
    data/dossier/private/.

    Raises when the table is absent. An empty-and-fine result would silently
    disable a whole deterministic layer while still reporting success, the
    same failure mode _load_known_entities refuses for persons.jsonl.
    """
    path = DOSSIER_DIR / source_case_id / EXTRA_IDENTIFIERS_FILENAME
    if not path.exists():
        raise RuntimeError(
            f"no {EXTRA_IDENTIFIERS_FILENAME} for case {source_case_id!r} — refusing to "
            "anonymise without the extra-identifier table, which covers the "
            "places, products and unresolved people persons.jsonl cannot see"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Underscore keys are documentation for whoever hand-edits the file.
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ========== layer 1: known-entity replacement ==========

def _load_known_entities(source_case_id: str) -> list[tuple[str, str, str]]:
    """Every (name_string, person_type, person_id) triple from the source
    case's persons.jsonl — canonical_name plus every alias (ADR #55).

    person_id is carried because it is the ONLY reliable way to group a
    person's aliases. Grouping by string overlap instead merges everyone who
    shares a surname: on the real corpus that collapsed the deceased father,
    the quasi-usufructuary and all three grandchildren into one pseudonym,
    which reads as though a single person were simultaneously the decedent
    and every heir.

    Returns [] when persons.jsonl is absent; callers must treat that as a
    hard error rather than an empty-and-fine result, since it would silence
    the entire strongest layer.
    """
    path = DOSSIER_DIR / source_case_id / "persons.jsonl"
    if not path.exists():
        return []

    triples: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        person_type = record.get("person_type", "natural_person")
        person_id = record.get("person_id") or record.get("canonical_name", "")
        names = [record.get("canonical_name")] + list(record.get("aliases") or [])
        for name in names:
            if name and len(name) >= MIN_ENTITY_LEN:
                triples.append((name, person_type, person_id))
    return triples


# Word-tokens that are structural rather than identifying, so they must never
# become map keys on their own ("immobilier" would rewrite ordinary prose).
_GENERIC_NAME_TOKENS = {
    "immobilier", "immobiliere", "societe", "notaire", "notaires", "etude",
    "cabinet", "conseil", "caisse", "banque", "assurance", "assurances",
    "mutuel", "mutuelle", "regionale", "regional", "national", "nationale",
    "generale", "general", "depots", "consignations", "retraite", "gestion",
    "france", "francaise", "groupe", "compagnie", "credit", "agricole",
    "monsieur", "madame", "maitre", "veuve", "epouse",
}


def _distinctive_tokens(name: str) -> set[str]:
    """Identifying word-tokens of a name — the parts that leak when the full
    string does not appear verbatim.

    persons.jsonl aliases are often only full forms ("<given name> <SURNAME>",
    "<TRADING NAME> <SECTOR>"), while documents use the bare surname or
    trading name. Whole-string matching alone left exactly those in the output
    on the first real run.
    """
    return {
        token for token in re.split(r"[^a-zA-Z0-9À-ÿ]+", _fold(name))
        if len(token) >= 4 and token not in _GENERIC_NAME_TOKENS and not token.isdigit()
    }


def _build_pseudonym_map(entities: list[tuple[str, str, str]]) -> dict[str, str]:
    """Assign one stable pseudonym per person_id, shared by all their aliases.

    Aliases of one entity must collapse to the SAME pseudonym (a bare surname
    and "<given name> <SURNAME>" are not two people), while distinct entities
    must stay distinct — both properties come free from grouping on person_id.

    Bare distinctive tokens are mapped too, since documents use them where
    persons.jsonl only records full forms. A token owned by exactly one
    person maps to that person's pseudonym; a token shared by several (a
    family surname) is inherently ambiguous, so it maps to a neutral
    placeholder rather than being guessed at or, worse, left in place.

    Pseudonyms are deliberately obvious placeholders ("Personne A",
    "Organisme B") rather than plausible substitute names: a realistic fake
    name invites a reader to believe it, and could collide with a real person.
    """
    by_person: dict[str, str] = {}
    counters = {"natural_person": 0, "legal_person": 0}
    for _, person_type, person_id in entities:
        if person_id in by_person:
            continue
        bucket = "legal_person" if person_type == "legal_person" else "natural_person"
        label = "Organisme" if bucket == "legal_person" else "Personne"
        by_person[person_id] = f"{label} {_letter(counters[bucket])}"
        counters[bucket] += 1

    mapping: dict[str, str] = {}
    token_owners: dict[str, set[str]] = {}
    token_type: dict[str, str] = {}

    for name, person_type, person_id in entities:
        mapping[_fold(name)] = by_person[person_id]
        for token in _distinctive_tokens(name):
            token_owners.setdefault(token, set()).add(person_id)
            token_type[token] = person_type

    for token, owners in token_owners.items():
        if token in mapping:
            continue
        if len(owners) == 1:
            mapping[token] = by_person[next(iter(owners))]
        else:
            mapping[token] = (
                "[organisme]" if token_type.get(token) == "legal_person" else "[nom]"
            )
    return mapping


def _letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA. Keeps pseudonyms short and ordered."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _apply_entity_map(text: str, mapping: dict[str, str]) -> str:
    """Replace every known entity string, LONGEST FIRST.

    Order is load-bearing: replacing a bare "<SURNAME>" before
    "<SURNAME> <given name>" would consume the surname and strand the given
    name in the output as a bare, still-identifying first name.
    """
    for key in sorted(mapping, key=len, reverse=True):
        # Word-anchored: without boundaries a short token matches inside a
        # longer word and corrupts ordinary prose — a 4-letter organisation
        # token turned "Correspondance-courante" into
        # "Correspondance-<pseudonym>ante" on the real corpus.
        pattern = re.compile(_word_anchored(key), re.IGNORECASE)
        # Match on the folded text but splice into the original, so accented
        # source spellings are replaced too.
        text = _sub_folded(text, pattern, mapping[key])
    return text


def _word_anchored(key: str) -> str:
    """`key` as a regex that cannot match inside a longer word.

    \b is unreliable here because keys may begin or end with punctuation or
    accented characters after folding, so use explicit alphanumeric
    lookarounds instead.
    """
    return rf"(?<![a-z0-9]){re.escape(_fold(key))}(?![a-z0-9])"


def _sub_folded(text: str, pattern: re.Pattern, replacement: str) -> str:
    """Apply `pattern` against the accent-folded form of `text` while cutting
    the ORIGINAL text at the matched offsets.

    Folding is length-preserving here (NFD marks are dropped, but only after
    combining characters are separated), so offsets in the folded string map
    back one-to-one only when the source has no precomposed accents. To stay
    correct, fold per-character and keep an index map.
    """
    folded_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        for f in _fold(ch) or " ":
            folded_chars.append(f)
            index_map.append(i)
    folded = "".join(folded_chars)

    out: list[str] = []
    cursor = 0
    for match in pattern.finditer(folded):
        start = index_map[match.start()]
        end = index_map[match.end() - 1] + 1
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# ========== layer 2: structured PII ==========

def _apply_structured_pii(text: str) -> str:
    """Emails, phones, IBAN, SIREN/SIRET, street addresses, postal codes and
    reference numbers — none of which appear in persons.jsonl."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ========== layer 3: date / amount / product transforms ==========

def _shift_date(day: int, month: int, year: int) -> date | None:
    try:
        return date(year, month, day) + timedelta(days=DATE_OFFSET_DAYS)
    except ValueError:
        return None


def _shift_dates(text: str) -> str:
    """Apply DATE_OFFSET_DAYS to every recognisable date, preserving the
    written format so "8 mars 2024" stays prose and "09/01/2026" stays
    numeric. A single shared offset keeps every interval in the dossier —
    the gap between a convention and a death, a notice period — exactly as
    it was, which is what the compliance reasoning depends on."""

    def _text_repl(m: re.Match) -> str:
        month = _MONTHS_FR.get(_fold(m.group(2)))
        shifted = _shift_date(int(m.group(1)), month, int(m.group(3))) if month else None
        if shifted is None:
            return m.group(0)
        return f"{shifted.day} {_MONTHS_FR_REV[shifted.month]} {shifted.year}"

    def _num_repl(m: re.Match) -> str:
        shifted = _shift_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if shifted is None:
            return m.group(0)
        return f"{shifted.day:02d}/{shifted.month:02d}/{shifted.year}"

    def _compact_repl(m: re.Match) -> str:
        shifted = _shift_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if shifted is None:
            return m.group(0)
        return f"{shifted.year}{shifted.month:02d}{shifted.day:02d}"

    text = _DATE_TEXT_RE.sub(_text_repl, text)
    text = _DATE_NUM_RE.sub(_num_repl, text)
    return _DATE_COMPACT_RE.sub(_compact_repl, text)


def _scale_amounts(text: str) -> str:
    """Multiply every euro amount by AMOUNT_SCALE, keeping French formatting
    (space thousands separator, comma decimal). Ratios between figures are
    preserved because one factor is used everywhere."""

    def _repl(m: re.Match) -> str:
        whole = re.sub(r"[   ]", "", m.group(1))
        cents = m.group(2) or "0"
        value = float(f"{whole}.{cents.ljust(2, '0')}")
        scaled = round(value * AMOUNT_SCALE, 2)
        integral = int(scaled)
        grouped = f"{integral:,}".replace(",", " ")
        frac = round((scaled - integral) * 100)
        unit = m.group(3)
        if frac:
            return f"{grouped},{frac:02d} {unit}"
        return f"{grouped} {unit}"

    return _AMOUNT_RE.sub(_repl, text)


def _apply_extra_identifiers(text: str, extras: dict[str, str]) -> str:
    """Places, financial products and firm fragments that persons.jsonl does
    not resolve as entities (see load_extra_identifiers).

    Longest key first, so a longer key is consumed whole rather than being
    eaten by a shorter one that is a substring of it.
    """
    for name in sorted(extras, key=len, reverse=True):
        pattern = re.compile(_word_anchored(name), re.IGNORECASE)
        text = _sub_folded(text, pattern, extras[name])
    return text


# ========== layer 4: LLM residual sweep ==========

SWEEP_SYSTEM_PROMPT = """\
Tu es un agent d'anonymisation de documents juridiques français. On te donne \
un document déjà partiellement anonymisé : les noms connus ont été remplacés \
par des étiquettes ("Personne A", "Organisme B"), les adresses, emails et \
téléphones par des marqueurs entre crochets ("[adresse]", "[email]").

Ta tâche : supprimer les identifiants RÉSIDUELS que le traitement automatique \
a manqués — un prénom ou nom de famille isolé, une commune, un lieu-dit, un \
nom d'étude ou de société, un numéro de dossier, une fonction nominative.

Règles strictes :
- Remplace un nom de personne résiduel par "Personne Z", une organisation par \
"Organisme Z", un lieu par "[lieu]", une référence par "[réf.]".
- NE modifie RIEN d'autre : ni les dates, ni les montants, ni les articles de \
loi, ni les termes juridiques (quasi-usufruit, nue-propriété, notaire, \
succession...), ni la structure du document, ni les en-têtes "## Page N".
- Ne reformule pas, ne résume pas, ne corrige pas la langue. Le document doit \
rester mot pour mot identique en dehors des identifiants remplacés.
- Conserve les étiquettes déjà posées telles quelles.

Retourne UNIQUEMENT le document anonymisé, sans préambule ni commentaire.
"""


def _llm_sweep(text: str, doc_id: str) -> str:
    """Last-resort pass for identifiers layers 1-3 could not know about.

    Soft-fails to the input text on any error: the deterministic layers have
    already run and verify_anonymization gates the result regardless, so a
    transient API failure must not abort a 55-document build. The failure is
    logged loudly because it means this document got weaker treatment.
    """
    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=SWEEP_MODEL_ID,
            messages=[
                {"role": "system", "content": SWEEP_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=SWEEP_MAX_TOKENS,
            temperature=SWEEP_TEMPERATURE,
        )
        swept = (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — soft-fail, gate still applies
        log.warning("anonymize: LLM sweep failed for doc_id=%s (%s); keeping deterministic output", doc_id, e)
        return text

    if not swept:
        log.warning("anonymize: LLM sweep returned empty for doc_id=%s; keeping deterministic output", doc_id)
        return text

    # A sweep that loses most of the document has rewritten rather than
    # redacted it — refuse the result instead of silently truncating a case.
    if len(swept) < 0.5 * len(text):
        log.warning(
            "anonymize: LLM sweep shrank doc_id=%s from %d to %d chars; keeping deterministic output",
            doc_id, len(text), len(swept),
        )
        return text
    return swept


# ========== document + case orchestration ==========

def anonymize_text(
    text: str,
    mapping: dict[str, str],
    doc_id: str,
    extras: dict[str, str],
    use_llm: bool = True,
) -> str:
    """Run all four layers over one document's markdown, in order."""
    text = _apply_entity_map(text, mapping)
    # Structured PII BEFORE extra identifiers: generalising a commune name
    # first turns "<code postal> <commune>" into "<code postal> <fiction>",
    # which no longer matches the postal-code pattern and strands the code in
    # the output.
    text = _apply_structured_pii(text)
    text = _apply_extra_identifiers(text, extras)
    text = _shift_dates(text)
    text = _scale_amounts(text)
    if use_llm:
        text = _llm_sweep(text, doc_id)
    return text


def _identifier_tokens(mapping: dict[str, str], extras: dict[str, str]) -> set[str]:
    """Distinctive word-tokens of every known identifier, for filename
    scrubbing.

    Filenames use bare surnames and trading names, not the full canonical
    form recorded in persons.jsonl, so whole-string matching alone leaves
    them in place. Tokens shorter than 4 characters and purely structural
    document words are excluded — over-replacing here only makes a filename
    uglier, while under-replacing publishes a name in every citation, so the
    bias is deliberate.
    """
    generic = {
        "immobilier", "societe", "notaire", "notaires", "etude", "cabinet",
        "conseil", "caisse", "banque", "assurance", "assurances", "mutuel",
        "regionale", "regional", "national", "nationale", "generale",
        "depots", "consignations", "retraite", "gestion", "france",
    }
    tokens: set[str] = set()
    for name in list(mapping) + list(extras):
        for token in re.split(r"[^a-z0-9]+", _fold(name)):
            if len(token) >= 4 and token not in generic:
                tokens.add(token)
    return tokens


def anonymize_doc_id(doc_id: str, mapping: dict[str, str], extras: dict[str, str]) -> str:
    """Anonymise a document's filename stem.

    Filenames carry identifiers too — the real corpus names its documents
    after the family, the counterparty organisation and the invoice number —
    and doc_id flows straight into chunk_id ("dossier-<case>-<doc_id>-cNNN"),
    which is committed in chunks.csv and shown in citations. Anonymising the
    text but not the filename would publish the name in every citation.

    Also the join key for artifact translation: fact_id, source_doc_id and
    source_chunk_id are all built from a doc_id, so every reference to a
    document must be rewritten through this one function to stay consistent.
    """
    folded = _fold(doc_id)

    # Whole identifiers first (longest wins), then bare distinctive tokens.
    for key in sorted(mapping, key=len, reverse=True):
        token = re.sub(r"[^a-z0-9]+", "_", _fold(key)).strip("_")
        if len(token) >= MIN_ENTITY_LEN:
            folded = folded.replace(token, re.sub(r"[^a-z0-9]+", "_", _fold(mapping[key])).strip("_"))
    for name in sorted(extras, key=len, reverse=True):
        token = re.sub(r"[^a-z0-9]+", "_", _fold(name)).strip("_")
        if len(token) >= MIN_ENTITY_LEN:
            folded = folded.replace(
                token, re.sub(r"[^a-z0-9]+", "_", _fold(extras[name])).strip("_")
            )
    for token in sorted(_identifier_tokens(mapping, extras), key=len, reverse=True):
        folded = re.sub(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", "x", folded)

    # 5 digits and up, not 6: a French postal code is exactly 5, and filenames
    # in the real corpus carry them ("..._immeuble_sis_a_<commune>_77590").
    # Nothing in a document stem needs a 5-digit run preserved, so the bias is
    # again towards an uglier filename over a published identifier.
    folded = re.sub(r"\d{5,}", "ref", folded)
    folded = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")
    return folded or "document"


def _map_path(source_case_id: str) -> Path:
    """Where the pseudonym mapping lives.

    Under the PRIVATE case directory, never the target: the mapping links
    pseudonyms back to real people, so it is the one artifact that must never
    ship with the public case. data/dossier/private/ is gitignored wholesale.
    """
    return DOSSIER_DIR / source_case_id / "_anonymization_map.json"


def load_or_build_mapping(source_case_id: str) -> dict[str, str]:
    """Load the persisted pseudonym mapping, extending it for any entity
    added since the last run. Persisting is what makes reruns idempotent."""
    entities = _load_known_entities(source_case_id)
    if not entities:
        raise RuntimeError(
            f"no persons.jsonl for case {source_case_id!r} — refusing to anonymise "
            "without the known-entity list, which is the strongest layer "
            "(run the persons pipeline first)"
        )

    path = _map_path(source_case_id)
    existing: dict[str, str] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

    mapping = _build_pseudonym_map(entities)
    mapping.update(existing)  # persisted assignments win, so pseudonyms never move
    for key, value in _build_pseudonym_map(entities).items():
        mapping.setdefault(key, value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return mapping


def anonymize_case(
    target_case_id: str,
    source_case_id: str = DEFAULT_SOURCE_CASE_ID,
    use_llm: bool = True,
) -> dict:
    """Derive target_case_id/extracted/*.md from source_case_id's extracted
    markdown, then verify. Raises if verification finds a surviving
    identifier — a failed build is the point, not a warning to be skimmed."""
    t0 = time.time()
    src_dir = DOSSIER_DIR / source_case_id / "extracted"
    if not src_dir.exists():
        raise RuntimeError(f"no extracted/ directory for source case {source_case_id!r}")

    mapping = load_or_build_mapping(source_case_id)
    extras = load_extra_identifiers(source_case_id)
    out_dir = DOSSIER_DIR / target_case_id / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for md_path in sorted(src_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        anonymised = anonymize_text(text, mapping, md_path.stem, extras, use_llm=use_llm)
        out_name = anonymize_doc_id(md_path.stem, mapping, extras)
        (out_dir / f"{out_name}.md").write_text(anonymised, encoding="utf-8")
        written += 1
        log.info("anonymize: %s -> %s", md_path.name, f"{out_name}.md")

    report = verify_anonymization(target_case_id, source_case_id=source_case_id)
    elapsed = time.time() - t0

    return {
        "case_id": target_case_id,
        "source_case_id": source_case_id,
        "docs_written": written,
        "entities_mapped": len(mapping),
        "files_scanned": report.files_scanned,
        "residual_proper_nouns": report.residual_proper_nouns,
        "elapsed": elapsed,
    }


# ========== verification gate ==========

def _scan_targets(case_id: str) -> list[Path]:
    """Every artifact a leak could hide in — not just the markdown, because
    facts and compliance re-introduce text through an LLM after this module
    has run."""
    case_dir = DOSSIER_DIR / case_id
    targets = sorted(case_dir.glob("extracted/*.md"))
    for name in ("chunks.csv", "facts.jsonl", "actor_roles.jsonl",
                 "role_ambiguities.jsonl", "compliance_matrix.json",
                 "persons.jsonl"):
        path = case_dir / name
        if path.exists():
            targets.append(path)
    return targets


# High-signal residual-identifier heuristics. A plain "any capitalised word"
# rule returned 1605 hits on the real corpus — almost all sentence-initial
# ordinary French — which buries the handful that matter. These two patterns
# target how names actually appear in this corpus: SHOUTED surnames in letter
# headers and signature blocks, and anything following a civility title.
_SHOUTED_RE = re.compile(r"\b[A-ZÀ-Ý]{4,}(?:[-'][A-ZÀ-Ý]+)*\b")
_TITLED_RE = re.compile(
    r"\b(?:M\.|Mme|Mlle|Ma[îi]tre|Me|Dr|Pr)\s+([A-ZÀ-Ý][\wÀ-ÿ'\-]{2,})"
)
_STREET_WORD_RE = re.compile(
    r"\b(?:avenida|calle|strasse|street|road|via|plaza)\b", re.IGNORECASE
)


def verify_anonymization(
    case_id: str,
    source_case_id: str = DEFAULT_SOURCE_CASE_ID,
    raise_on_leak: bool = True,
) -> AnonymizationReport:
    """Scan every artifact of `case_id` for surviving identifiers.

    Two different strengths of claim, and it matters which is which:

    - LEAKS are proven. Every canonical_name/alias from the source case's
      persons.jsonl and every extra-identifier key is searched for
      (accent-folded, case-insensitive), plus the structured-PII regexes.
      A hit fails the build.
    - RESIDUAL PROPER NOUNS are only a review aid. Capitalised tokens that
      are not recognised legal vocabulary are counted and returned so a human
      can look at them. This CANNOT prove the absence of an identifier that
      was never in persons.jsonl — measured against the real corpus, that
      list covered 10 of 18 sampled identifiers. A clean report is a
      necessary condition for publishing, never a sufficient one.
    """
    report = AnonymizationReport(case_id=case_id)

    needles: list[str] = [name for name, _type, _pid in _load_known_entities(source_case_id)]
    needles.extend(load_extra_identifiers(source_case_id))
    folded_needles = {_fold(n): n for n in needles if len(n) >= MIN_ENTITY_LEN}

    for path in _scan_targets(case_id):
        report.files_scanned += 1
        raw = path.read_text(encoding="utf-8", errors="replace")
        folded = _fold(raw)

        for folded_needle, original in folded_needles.items():
            # Word-anchored, matching the replacement layers: a plain
            # substring test flags a 6-letter firm fragment inside an ordinary
            # French word and fails the build on prose. Identifiers genuinely
            # embedded in an alphanumeric run (a passport MRZ) are handled as
            # structured PII below, not here.
            if re.search(rf"(?<![a-z0-9]){re.escape(folded_needle)}(?![a-z0-9])", folded):
                report.leaks.append((str(path), original))

        for pattern, label in _PII_PATTERNS:
            # The replacement markers themselves are expected in the output.
            for match in pattern.finditer(raw):
                if match.group(0).strip() not in label:
                    report.leaks.append((str(path), f"{label}: {match.group(0)[:40]}"))
                    break

        if path.suffix == ".md":
            candidates = list(_SHOUTED_RE.findall(raw))
            candidates += _TITLED_RE.findall(raw)
            candidates += _STREET_WORD_RE.findall(raw)
            for token in candidates:
                key = _fold(token)
                if key in ALLOWED_PROPER_NOUNS or key.isdigit():
                    continue
                report.residual_proper_nouns[token] = report.residual_proper_nouns.get(token, 0) + 1

    if report.leaks and raise_on_leak:
        preview = "\n".join(f"  {f}: {ident}" for f, ident in report.leaks[:15])
        raise RuntimeError(
            f"anonymisation gate FAILED for case {case_id!r} — "
            f"{len(report.leaks)} surviving identifier(s):\n{preview}"
        )

    log.info(
        "anonymize verify: case_id=%s files=%d leaks=%d residual_proper_nouns=%d",
        case_id, report.files_scanned, len(report.leaks), len(report.residual_proper_nouns),
    )
    return report
