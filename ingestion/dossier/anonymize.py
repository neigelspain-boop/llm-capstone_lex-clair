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
from pathlib import Path

from dotenv import load_dotenv

from ingestion.clients import get_openrouter_client
from ingestion.dossier import personas
from ingestion.dossier.index import DOSSIER_DIR

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ========== module constants ==========

DEFAULT_SOURCE_CASE_ID = "private"

SWEEP_MODEL_ID = "anthropic/claude-haiku-4.5"
SWEEP_MAX_TOKENS = 8192
SWEEP_TEMPERATURE = 0.0

# Dates and amounts pass through UNCHANGED (ADR #62). An earlier version
# shifted every date by a fixed offset and scaled every amount by a fixed
# factor, on the reasoning that an exact sum plus exact dates identifies a
# case even with the names gone. That reasoning still holds and the risk is
# accepted deliberately: the published case exists to be read as a worked
# example, and a worked example whose figures do not add up against the deed
# it quotes is worth less than the residual risk it avoids. The transform
# also had to be replayed identically across markdown, facts and the
# compliance matrix to keep them consistent — three chances to disagree, for
# a protection that only binds against someone who already knows the case.

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
    # Shouted form headings, field labels and body text from the scanned forms
    # (déclaration de succession, avis d'impôt, donation-partage, mandat de
    # vente). Measured: these accounted for the large majority of a 521-token
    # residual report, and every one of them buried a token that mattered.
    "presentation", "projet", "avis", "opere", "paye", "fact", "ssion",
    "autorisation", "deblocage", "soussigne", "soussignes", "soussignee",
    "prient", "portefeuille", "document", "documents", "retourner",
    "imperativement", "prenom", "prenoms", "important", "conditions",
    "paiement", "uniquement", "virement", "identite", "bancaire", "credit",
    "agricole", "immobilier", "immobiliere", "asset", "management",
    "donation", "donations", "partage", "partages", "usage", "representant",
    "legal", "nouvelle", "nationalite", "francaise", "fixe", "mobile",
    "conjoint", "adresse", "profession", "situation", "famille", "regime",
    "matrimonial", "fonds", "titres", "requete", "premiere", "premier",
    "deuxieme", "troisieme", "quatrieme", "devolution", "devolutive",
    "successorale", "successoral", "tiers", "qualites", "qualite",
    "hereditaires", "hereditaire", "composition", "liquidites", "placements",
    "patrimoine", "reconnaissance", "constitution", "conventionnel", "detail",
    "enregistrement", "mention", "donnees", "personnelles", "formalisme",
    "recu", "present", "notoriete", "expose", "quart", "douzieme", "effets",
    "effet", "absence", "inventaire", "aide", "assistance", "sociale",
    "deces", "fichier", "dispositions", "dernieres", "volontes", "sort",
    "dematerialisees", "justificatives", "produites", "informations",
    "information", "acceptation", "pure", "simple", "attestation",
    "obligations", "fiscales", "avertissement", "contrats", "assurance",
    "avant", "compter", "destruction", "associes", "declaration",
    "declarations", "declarant", "declares", "cadre", "cadres", "remplir",
    "deposant", "enfants", "petits", "option", "creance", "observations",
    "preliminaires", "communaute", "espece", "plan", "actions", "mondiale",
    "brut", "restitution", "exigible", "jour", "usufruitiers", "balance",
    "imposables", "imposable", "liquidation", "payer", "acomptes", "verses",
    "reste", "impot", "revenu", "reductions", "credits", "imputations",
    "suite", "jointe", "prelevements", "prelevement", "sociaux", "prel",
    "base", "calcul", "solde", "psol", "tenu", "elements", "sera",
    "rembourse", "remboursement", "automatique", "aucune", "demarche",
    "faire", "complementaires", "complementaire", "source", "autres",
    "autre", "contacter", "clients", "adresser", "cheque", "energie",
    "procuration", "procurations", "protegee", "requerant", "modalites",
    "temporaire", "viager", "autorisations", "faculte", "substitution",
    "applicable", "remuneration", "pluri", "representation", "decharge",
    "dedie", "onze", "anticipe", "presence", "capacite", "mariage",
    "posterite", "rappel", "anterieure", "masse", "partager", "numero",
    "incorporees", "rapportees", "caractere", "relatif", "relative",
    "acquisition", "exercice", "retour", "interdiction", "aliener",
    "hypothequer", "nantir", "mutation", "publicite", "fonciere",
    "normalisee", "repression", "insuffisances", "dissimulations", "copie",
    "authentique", "achat", "maison", "telephone", "favoris", "centre",
    "gare", "commerces", "ecole", "econome", "energivore", "faible",
    "forte", "emission", "identification", "vendre", "engager",
    "financieres", "acquereur", "duree", "renouvellement", "resiliation",
    "mois", "consommation", "particulieres", "mediation", "election",
    "exclusif", "vente", "avec", "delegation", "negociation", "relance",
    "manquantes", "residence", "hameau", "republique", "passeport",
    "passport", "certificado", "registro", "ciudadano", "union", "unión",
    "registration", "certificate", "citizen", "central", "avda", "avenida",
    "agence", "filiere", "disponible", "historique", "evenements",
    "demande", "reponse", "dossiers", "actifs", "passifs", "etat",
    "special", "immeuble", "formulee", "pluralite", "ayants", "encaissement",
    "prealable", "opposition", "distributions", "electronique", "calendrier",
    "previsionnel", "delai", "mise", "disposition", "reserve", "reserves",
    "expresse", "clôture", "cloture", "refus", "prise", "ecris", "lequel",
    "donateur", "donataire", "donnes", "jouissance", "logement", "pouvoirs",
    "droit", "droits", "parties", "partie", "compte", "rendu",
    "interrogation", "ouverte", "matin", "quoi", "dans", "sous", "vent",
    "iles", "via", "documento", "acceder", "seize", "soixante", "quatorze",
    "quarante", "cinquante", "conq", "centj", "vingt-cinq",
    "quatre-vingt-quinze", "trente-six", "bleu-vert", "mari", "tips",
    "chef", "quasi-usufruitier", "quasi-usufruits", "quasi-usufruit",
    "decedee", "decedes", "donation-partage",
    # Public registries, professional bodies and payment schemes. Named in
    # every French succession file; none of them identifies THIS one.
    "ficoba", "ficovie", "ciclade", "carsat", "cpam", "cnil", "anssi",
    "sepa", "adsn", "crpcen", "camca", "cecmc", "fcddv", "spfe", "dasc",
    "agira", "yousign", "lrar", "trve", "cher", "crds",
    # Residual-report hygiene: single words the scan still surfaced after the
    # sweep above. Composites are matched part by part, so a hyphenated term
    # needs each of its parts here, not the joined form.
    "mandat", "releve", "frais", "domicile", "dont", "avez", "designation",
    "viii", "bleu", "vert", "usufruits", "quinze",
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
    # OCR of a 2D barcode / datamatrix on a scanned tax or notarial form,
    # which comes through as an unbroken run of capitals. No French word is
    # 20 letters of solid uppercase, and the run may encode the very fields
    # printed beside it, so it is redacted rather than reviewed.
    (re.compile(r"\b[A-Z]{20,}\b"), "[code-barres]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{3}\d{4,}\b"), "[ICS]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}\b"), "[IBAN]"),
    (re.compile(r"\b(?:\+33|0)\s?[1-9](?:[\s.-]?\d{2}){4}\b"), "[telephone]"),
    # Label-anchored phone numbers. The rule above is format-anchored and so
    # requires the leading 0 / +33 that OCR of a scanned letterhead routinely
    # loses or mangles — "Tél. 40 54 44 44" and "Tel ; 514 813 9053" both
    # survived it. When a document says a number is a telephone, that is
    # better evidence than its shape, so trust the label and redact whatever
    # digit run follows it.
    (
        re.compile(
            r"\b(?:t[ée]l(?:[ée]phone)?|mobile|portable|fax|gsm)\b\s*[.:;]?\s*"
            r"\+?\d[\d .\-]{6,}\d",
            re.IGNORECASE,
        ),
        "Tél. [telephone]",
    ),
    (re.compile(r"\b\d{3}[ ]?\d{3}[ ]?\d{3}[ ]?\d{5}\b"), "[SIRET]"),
    (re.compile(r"\b\d{3}[ ]?\d{3}[ ]?\d{3}\b(?=\s*(?:RCS|SIREN))"), "[SIREN]"),
    # Same 9-digit shape with the marker BEFORE it — "RCS Beaumont 315 429 837",
    # which is the ordering French company footers actually use. The lookahead
    # rule above only fires on "315 429 837 RCS" and so never matched these.
    (
        re.compile(
            r"\b(?:RCS|RCS\s+[A-ZÀ-Ý][\w'\-]+|SIREN|SIRET)\s+\d{3}[ ]?\d{3}[ ]?\d{3}\b"
        ),
        "[SIREN]",
    ),
    # Bare 9-digit triplets: bank account, contract and policy references, which
    # carry no marker at all and were passing through the anonymiser verbatim.
    # A monetary amount has the same shape, so anything reading as currency is
    # excluded — and \b already prevents biting a 9-digit window out of the
    # longer SIRET/account runs handled above.
    (
        re.compile(r"\b\d{3}[ ]\d{3}[ ]\d{3}\b(?!\s*(?:€|EUR\b|euros?\b))"),
        "[réf.]",
    ),
    # Regulator approval numbers — AMF "n° GP 07000033" / "GP 07 0000 33".
    # The generic reference rule below expects at most one letter before the
    # first digit ("[A-Z]?\d"), so a two-letter agrément prefix escaped it.
    # The "n°" is optional but the prefix is enumerated rather than left as
    # [A-Z]{2,3}: any two capitals followed by digits would also swallow
    # ordinary citations ("ADR 63", "SCPI 07 …").
    (
        re.compile(r"\b(?:[Nn]°?\s?)?(?:GP|GC)\s?\d(?:[\d ]{3,}\d|\d{3,})\b"),
        "[réf.]",
    ),
    # Street line: number + type + name, up to a comma, a marker or end of
    # line. The tail stops at "[" because the patterns above have already
    # placed markers on this line — without it a 40-character tail slices
    # through "[SIREN]" and leaves "[adresse]IREN]" in the published text.
    (
        re.compile(
            r"\b\d{1,4}\s?(?:bis|ter)?\s*"
            r"(?:bd|boulevard|av|avenue|rue|place|impasse|chemin|route|quai|allee|allée)\b"
            r"[^,\n\[]{0,40}",
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


def _build_pseudonym_map(
    entities: list[tuple[str, str, str]],
    roster: personas.Roster | None = None,
    pinned: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign one stable pseudonym per person, shared by all their aliases.

    Aliases of one entity must collapse to the SAME pseudonym (a bare surname
    and "<given name> <SURNAME>" are not two people), while distinct entities
    must stay distinct — both properties come free from grouping on person_id,
    and the roster's merge groups extend the first property across the several
    person_ids the resolver splits one real person into.

    Bare distinctive tokens are mapped too, since documents use them where
    persons.jsonl only records full forms. A token owned by exactly one
    person maps to that person's pseudonym; a token shared by several (a
    family surname) is inherently ambiguous, so it maps to a neutral
    placeholder unless the roster names it — never guessed at, never left
    in place.

    Pseudonyms follow the court/casebook convention — ordinary French names,
    readable as the sentence they replace. That readability is bought with a
    real cost: a plausible name invites a reader to believe it, so the
    published case must carry a visible fiction notice. See personas.py for
    why this beats both "Personne A" and unmistakably-fictional alternatives.
    The generated fallback remains "Personne A" / "Organisme B", which is
    unmistakable but unreadable — acceptable because it only ever covers
    entities the roster forgot.
    """
    roster = roster or personas.Roster()
    by_person = assign_personas(entities, roster, pinned=pinned)

    mapping: dict[str, str] = {}
    token_owners: dict[str, set[str]] = {}
    token_type: dict[str, str] = {}

    for name, person_type, person_id in entities:
        head = roster.head_of(person_id)
        # An entity with no assignment is one the roster chose to leave alone
        # (keep_real_legal_persons). It must contribute NO mapping entry and
        # NO token ownership — a token it owned would otherwise be rewritten
        # on its behalf, which is the opposite of keeping the name real.
        if head not in by_person:
            continue
        key = _fold(name)
        # Two person records claiming the same MULTI-WORD name means the
        # resolver split one person in two — the later record silently wins
        # the key and the earlier one's roles end up attributed to a different
        # name. It is invisible in the output (both names are plausible), so
        # it is logged here rather than left to a human read. Fix by adding
        # the ids to a merge group; the shared alias is the evidence.
        #
        # Bare surnames are excluded: in a succession file the whole family
        # lists the same one, which is expected and is what token_overrides
        # exists to resolve.
        shared_bare_surname = len(re.findall(r"[a-z0-9à-ÿ]+", key)) < 2
        if key in mapping and mapping[key] != by_person[head] and not shared_bare_surname:
            log.warning(
                "anonymize: name %r is claimed by two unmerged people (%r and %r) — "
                "one will be attributed to the other; add them to merge_groups",
                name, mapping[key], by_person[head],
            )
        mapping[key] = by_person[head]
        for token in _distinctive_tokens(name):
            token_owners.setdefault(token, set()).add(head)
            token_type[token] = person_type

    for token, owners in token_owners.items():
        if token in mapping:
            continue
        if token in roster.token_overrides:
            # The shared family surname, and bare given names. Owned by
            # several people or resolving to a full form, so the rules below
            # would degrade them to "[nom]" or duplicate the surname; the
            # roster names each token explicitly instead.
            mapping[token] = roster.token_overrides[token]
        elif len(owners) == 1:
            mapping[token] = by_person[next(iter(owners))]
        else:
            mapping[token] = (
                "[organisme]" if token_type.get(token) == "legal_person" else "[nom]"
            )

    # Overrides for tokens no entity name contains — a family surname the
    # resolver only ever saw inside longer strings still has to be nameable.
    for token, replacement in roster.token_overrides.items():
        mapping.setdefault(token, replacement)
    return mapping


def assign_personas(
    entities: list[tuple[str, str, str]],
    roster: personas.Roster,
    pinned: dict[str, str] | None = None,
) -> dict[str, str]:
    """canonical person_id -> pseudonym, in three tiers.

    1. The roster's character for that person. The roster is an explicit
       human decision and outranks everything: editing it must change the
       output, not lose a race with a file the last run happened to write.
    2. `pinned` — assignments persisted from an earlier run, which is what
       keeps reruns idempotent for everyone the roster does NOT name. Kept
       only for person_ids that still exist; a stale entry for a person since
       merged away must not resurrect them, which is the bug that put two
       different people on one label in the first place.
    3. The fallback pools, then the "Personne A" / "Organisme B" letter
       scheme once a pool is exhausted. An unrostered entity is still
       anonymised — leaving it in place would be a leak, and a roster is a
       readability feature, not a privacy one.

    Merged person_ids collapse here: every id in a merge group resolves to
    the group head, so all of them get the same pseudonym.
    """
    pinned = pinned or {}
    by_person: dict[str, str] = {}
    used: set[str] = set()
    counters = {"natural_person": 0, "legal_person": 0}
    pools = {
        "natural_person": list(personas.FALLBACK_NATURAL_POOL),
        "legal_person": list(personas.FALLBACK_LEGAL_POOL),
    }

    ordered: list[tuple[str, str]] = []
    for _name, person_type, person_id in entities:
        head = roster.head_of(person_id)
        if head in by_person:
            continue
        bucket = "legal_person" if person_type == "legal_person" else "natural_person"
        if bucket == "legal_person" and roster.keep_real_legal_persons:
            # No assignment at all, so no mapping entry is created and the
            # name passes through untouched (ADR #62).
            continue
        character = roster.character_for(person_id) or pinned.get(head)
        if character:
            by_person[head] = character
            used.add(character)
        else:
            ordered.append((head, bucket))

    # Unnamed entities are assigned only after every pinned and rostered
    # character is known, so a fallback can never steal a name the roster
    # wanted for someone else.
    for head, bucket in ordered:
        character = next((c for c in pools[bucket] if c not in used), None)
        if character is None:
            label = "Organisme" if bucket == "legal_person" else "Personne"
            while f"{label} {_letter(counters[bucket])}" in used:
                counters[bucket] += 1
            character = f"{label} {_letter(counters[bucket])}"
            counters[bucket] += 1
        by_person[head] = character
        used.add(character)

    return by_person


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


# ========== layer 3: named products, places and firm fragments ==========

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

SWEEP_SYSTEM_PROMPT = f"""\
Tu es un agent d'anonymisation de documents juridiques français. On te donne \
un document déjà anonymisé par un traitement déterministe : les noms réels ont \
été remplacés par des noms de personnages de fiction (Dragon Ball), les lieux \
par des noms de planètes, et les adresses, emails et téléphones par des \
marqueurs entre crochets ("[email]", "[telephone]").

Ces noms de fiction sont VOULUS. Ils ne sont pas des identifiants résiduels : \
tu dois les conserver mot pour mot, sans exception.

Ta tâche : supprimer les identifiants RÉELS RÉSIDUELS que le traitement \
automatique a manqués — un prénom ou nom de famille français isolé, une \
commune, un lieu-dit, un nom d'étude ou de société, un numéro de dossier.

Règles strictes :
- Remplace un nom de personne résiduel par l'un de : \
{", ".join(personas.RESIDUAL_NATURAL_POOL)}. Une organisation par l'un de : \
{", ".join(personas.RESIDUAL_LEGAL_POOL)}. Un lieu par l'un de : \
{", ".join(personas.PLACE_POOL)}. Une référence par "[réf.]".
- N'invente JAMAIS d'adresse email, de téléphone, d'IBAN ni de code postal, \
même à titre de remplacement. Un marqueur déjà posé reste tel quel.
- NE modifie RIEN d'autre : ni les dates, ni les montants, ni les articles de \
loi, ni les termes juridiques (quasi-usufruit, nue-propriété, notaire, \
succession...), ni la structure du document, ni les en-têtes "## Page N".
- Ne reformule pas, ne résume pas, ne corrige pas la langue. Le document doit \
rester mot pour mot identique en dehors des identifiants remplacés.

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
    """Run all four layers over one document's markdown, in order.

    Structured PII runs FIRST, against the raw text, because its patterns
    recognise real formats and a name replaced ahead of them destroys the
    format they match on. Two ways that goes wrong, both observed:
    generalising a commune turns "<code postal> <commune>" into
    "<code postal> <fiction>", which no longer matches the postal pattern and
    strands the code; and where the pattern still fires it consumes the first
    words of the replacement, publishing "[code postal] [ville] des impôts de
    Yardrat" — a marker with the tail of a pseudonym hanging off it.
    """
    text = _apply_structured_pii(text)
    text = _apply_entity_map(text, mapping)
    text = _apply_extra_identifiers(text, extras)
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


def assert_no_replacement_cascade(mapping: dict[str, str], extras: dict[str, str]) -> None:
    """Raise if any replacement VALUE is itself a replacement KEY.

    Replacement is a sequence of passes over one growing string, so a value
    written by an early pass is still visible to every later one. When a
    pseudonym happens to be another entry's key, the later pass rewrites it
    and one person silently takes another's name: "Claire DUBOIS" became
    "Nicole DUBOIS" on the first court-style build, because a different
    person's middle name was "Claire".

    Nothing about that failure is visible in the output — it looks like a
    perfectly ordinary name — so it cannot be left to a human read. Checked
    across mapping AND extras together, since the two are applied in sequence
    over the same text.

    Identity entries (a key deliberately mapped to itself, marking a name the
    policy keeps real) are exempt: rewriting them to themselves is a no-op.
    """
    combined = {**mapping, **{_fold(k): v for k, v in extras.items()}}
    collisions: list[str] = []
    for key, value in combined.items():
        for token in re.split(r"[^a-zA-Z0-9À-ÿ]+", _fold(value)):
            if not token or token not in combined:
                continue
            if _fold(combined[token]) == token:
                continue
            collisions.append(
                f"  {key!r} -> {value!r}, but {token!r} is itself a key -> {combined[token]!r}"
            )
    if collisions:
        preview = "\n".join(sorted(set(collisions))[:10])
        raise RuntimeError(
            "replacement cascade: a pseudonym would be rewritten by a later "
            "pass, silently giving one person another's name. Choose a "
            f"pseudonym that is not also a key:\n{preview}"
        )


def _map_path(source_case_id: str) -> Path:
    """Where the pseudonym mapping lives.

    Under the PRIVATE case directory, never the target: the mapping links
    pseudonyms back to real people, so it is the one artifact that must never
    ship with the public case. data/dossier/private/ is gitignored wholesale.
    """
    return DOSSIER_DIR / source_case_id / "_anonymization_map.json"


def load_or_build_mapping(source_case_id: str) -> dict[str, str]:
    """Build the pseudonym mapping, reusing persisted PERSON assignments.

    What is persisted is `by_person` — person_id -> pseudonym — not the
    flattened string map. That distinction is the whole fix. The previous
    version persisted the flattened map and replayed it with
    `mapping.update(existing)`, so a stale entry beat the fresh assignment
    and two different people ended up on one label while another label went
    unused. Pinning at the person level instead keeps reruns idempotent
    (nobody is renumbered) without letting yesterday's map override today's
    roster.

    Legacy flat maps are discarded rather than migrated: they carry exactly
    the assignments this fix exists to overrule, and the derived case is
    regenerated wholesale anyway.
    """
    entities = _load_known_entities(source_case_id)
    if not entities:
        raise RuntimeError(
            f"no persons.jsonl for case {source_case_id!r} — refusing to anonymise "
            "without the known-entity list, which is the strongest layer "
            "(run the persons pipeline first)"
        )

    roster = personas.load_roster(source_case_id)

    path = _map_path(source_case_id)
    pinned: dict[str, str] = {}
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict) and "by_person" in stored:
            pinned = stored["by_person"]
        else:
            log.warning(
                "anonymize: discarding legacy pseudonym map at %s (no by_person section) "
                "— pseudonyms will be reassigned from the roster", path,
            )

    # Only pin people who still exist, and only under their merged identity.
    # A pin on a person_id that has since been merged away would reinstate the
    # split this roster exists to close.
    live_heads = {roster.head_of(pid) for _n, _t, pid in entities}
    pinned = {pid: name for pid, name in pinned.items() if pid in live_heads}

    by_person = assign_personas(entities, roster, pinned=pinned)
    mapping = _build_pseudonym_map(entities, roster, pinned=pinned)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"by_person": by_person, "mapping": mapping}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return mapping


def anonymize_case(
    target_case_id: str,
    source_case_id: str = DEFAULT_SOURCE_CASE_ID,
    use_llm: bool = True,
    include_artifacts: bool = True,
) -> dict:
    """Derive the target case from the source case, then verify.

    Two halves: the extracted markdown (here) and the analytical artifacts
    facts/roles/persons/compliance (anonymize_artifacts). Both are needed for
    a usable public case — markdown alone gives a readable dossier with no
    analysis behind it, which is what the first showcase build produced.

    Raises if verification finds a surviving identifier — a failed build is
    the point, not a warning to be skimmed.
    """
    t0 = time.time()
    src_dir = DOSSIER_DIR / source_case_id / "extracted"
    if not src_dir.exists():
        raise RuntimeError(f"no extracted/ directory for source case {source_case_id!r}")

    mapping = load_or_build_mapping(source_case_id)
    extras = load_extra_identifiers(source_case_id)
    assert_no_replacement_cascade(mapping, extras)
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

    artifacts: dict[str, int] = {}
    if include_artifacts:
        artifacts = anonymize_artifacts(target_case_id, source_case_id, mapping, extras)

    report = verify_anonymization(target_case_id, source_case_id=source_case_id)
    elapsed = time.time() - t0

    return {
        "case_id": target_case_id,
        "source_case_id": source_case_id,
        "docs_written": written,
        "entities_mapped": len(mapping),
        "artifacts": artifacts,
        "files_scanned": report.files_scanned,
        "residual_proper_nouns": report.residual_proper_nouns,
        "elapsed": elapsed,
    }


# ========== artifact translation ==========

# Field policies. Explicit rather than heuristic, because the obvious
# heuristic ("scrub anything ending in _id") is wrong in both directions here:
# statute_chunk_id ("cc-730-1") and model_id are public references that must
# survive untouched, while `date` carries no "_id" and must still be shifted.
KEEP = "keep"            # copied verbatim
TEXT = "text"            # full free-text layers (names, PII, dates, amounts)
IDENT = "ident"          # a document-derived id, rewritten via anonymize_doc_id
IDENT_LIST = "ident[]"   # a list of the above
DROP = "drop"            # emitted as null, recomputed downstream

ARTIFACT_POLICIES: dict[str, dict[str, str]] = {
    "facts.jsonl": {
        "fact_id": IDENT,
        "source_doc_id": IDENT,
        # Recomputed by index.index_dossier's backfill against the chunks it
        # actually writes. Translating the old value would be a guess that
        # silently breaks every citation it does not happen to match.
        "source_chunk_id": DROP,
        "date": TEXT,
        "actor_role": KEEP,
        "action": TEXT,
        "target": TEXT,
        "verbatim_quote": TEXT,
        "distilled_context": TEXT,
    },
    "actor_roles.jsonl": {
        "role_id": KEEP,
        "label_fr": TEXT,
        "grounding_note": TEXT,
        "first_seen_doc_id": IDENT,
        "fact_count": KEEP,
        "confidence": KEEP,
    },
    "role_ambiguities.jsonl": {
        "ambiguity_id": IDENT,
        "source_doc_id": IDENT,
        "verbatim_quote": TEXT,
        "candidate_role_ids": KEEP,
        "note": TEXT,
        "fact_ids": IDENT_LIST,
    },
}

COMPLIANCE_TOP_POLICY = {
    "case_id": KEEP,          # overwritten with the target case id
    "generated_at": KEEP,
    "model_id": KEEP,
    "total_facts_considered": KEEP,
    "total_entries": KEEP,
    "unresolved_ambiguities": KEEP,
    "entries": KEEP,          # handled element-wise
}

COMPLIANCE_ENTRY_POLICY = {
    "entry_id": KEEP,
    # Statute text is public law and must stay accurate to the letter. Running
    # the date shift over it would silently misdate the articles the whole
    # analysis rests on.
    "statute_chunk_id": KEEP,
    "statute_excerpt": KEEP,
    "obligation_summary": TEXT,
    "actor_role": KEEP,
    "status": KEEP,
    "evidence_fact_ids": IDENT_LIST,
    "rationale": TEXT,
    "persons_named": TEXT,
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _fold(value)).strip("-")


def _rewrite_fact_id(value: str, mapping: dict[str, str], extras: dict[str, str]) -> str:
    """Rewrite a "<doc_id>-fNNN" / "<doc_id>-aNNN" identifier.

    The doc_id half is anonymised; the ordinal suffix is preserved verbatim so
    the id keeps its shape and stays joinable. Splitting on the LAST hyphen is
    what makes that safe — source doc_ids contain hyphens of their own.
    """
    head, sep, suffix = value.rpartition("-")
    if not sep or not re.fullmatch(r"[a-z]\d+", suffix):
        return anonymize_doc_id(value, mapping, extras)
    return f"{anonymize_doc_id(head, mapping, extras)}-{suffix}"


def _apply_policy(value, policy: str, mapping, extras):
    if value is None or policy == KEEP:
        return value
    if policy == DROP:
        return None
    if policy == IDENT:
        return _rewrite_fact_id(str(value), mapping, extras)
    if policy == IDENT_LIST:
        return [_rewrite_fact_id(str(v), mapping, extras) for v in value]
    if policy == TEXT:
        if isinstance(value, list):
            return [_apply_policy(v, TEXT, mapping, extras) for v in value]
        if not isinstance(value, str):
            return value
        return anonymize_text(value, mapping, "artifact", extras, use_llm=False)
    raise ValueError(f"unknown field policy {policy!r}")


def _translate_record(record: dict, policy: dict[str, str], mapping, extras, where: str) -> dict:
    """Apply a field policy to one record, refusing unknown fields.

    Unknown fields fail the build rather than defaulting. Either default is
    wrong on a privacy path: defaulting to KEEP publishes whatever a new field
    holds, and defaulting to TEXT silently corrupts the next structured field
    someone adds. A loud failure is the only option that cannot lose.
    """
    unknown = set(record) - set(policy)
    if unknown:
        raise RuntimeError(
            f"{where}: no anonymisation policy for field(s) {sorted(unknown)}. "
            "Add them to ARTIFACT_POLICIES — a field with no policy is either "
            "an unreviewed leak or a silently corrupted value."
        )
    return {k: _apply_policy(v, policy[k], mapping, extras) for k, v in record.items()}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8",
    )


def _translate_persons(
    records: list[dict], target_case_id: str, roster: personas.Roster, mapping, extras,
) -> list[dict]:
    """Collapse the source case's persons onto one record per real person.

    persons.jsonl is the one artifact a field policy cannot express, because
    translating it is not a per-record rewrite: the resolver split several
    real people across multiple person_ids, and emitting those unchanged would
    show the same character three times in the per-person panel.

    Aliases are NOT translated one by one — every alias of a person maps to
    the same character, so the output would be a list of duplicates. The
    fictional name is the only alias the public case has.
    """
    merged: dict[str, dict] = {}
    for record in records:
        person_id = record.get("person_id", "")
        head = roster.head_of(person_id)
        real_name = record.get("canonical_name", "")
        # No mapping entry means the roster chose to keep this entity's name
        # real (keep_real_legal_persons). The name still has to go through the
        # free-text layers, not straight out: an organisation's name carries
        # the geography this policy generalises, so "SIP <commune>" would
        # otherwise republish the exact locality the address redaction removed
        # — which is precisely what the gate caught on the first run.
        character = mapping.get(_fold(real_name)) or _apply_policy(
            real_name, TEXT, mapping, extras,
        )

        entry = merged.setdefault(head, {
            "person_id": f"{target_case_id}-{_slug(character or head)}",
            "canonical_name": character,
            "aliases": [character] if character else [],
            "person_type": record.get("person_type", "natural_person"),
            "confidence": record.get("confidence", "medium"),
            "ambiguity_note": None,
            "role_assignments": [],
        })

        notes = [
            n for n in (
                entry["ambiguity_note"],
                _apply_policy(record.get("ambiguity_note"), TEXT, mapping, extras),
            ) if n
        ]
        entry["ambiguity_note"] = " · ".join(dict.fromkeys(notes)) or None

        by_role = {a["role_id"]: a for a in entry["role_assignments"]}
        for assignment in record.get("role_assignments") or []:
            role_id = assignment.get("role_id")
            evidence = _apply_policy(
                assignment.get("evidence_fact_ids") or [], IDENT_LIST, mapping, extras,
            )
            existing = by_role.get(role_id)
            if existing is None:
                by_role[role_id] = {
                    "role_id": role_id,
                    "confidence": assignment.get("confidence", "medium"),
                    "ambiguity_note": _apply_policy(
                        assignment.get("ambiguity_note"), TEXT, mapping, extras,
                    ),
                    "evidence_fact_ids": list(dict.fromkeys(evidence)),
                }
            else:
                existing["evidence_fact_ids"] = list(
                    dict.fromkeys([*existing["evidence_fact_ids"], *evidence])
                )
        entry["role_assignments"] = sorted(by_role.values(), key=lambda a: a["role_id"])

    return sorted(merged.values(), key=lambda r: r["person_id"])


def anonymize_artifacts(
    target_case_id: str,
    source_case_id: str,
    mapping: dict[str, str],
    extras: dict[str, str],
) -> dict[str, int]:
    """Derive the target case's analytical artifacts from the source case's.

    The alternative is re-running facts -> mentions -> resolve -> distill ->
    compliance over the anonymised markdown, which costs $25+ on the
    compliance stage alone and produces a DIFFERENT analysis — one nobody has
    reviewed. Translating instead publishes exactly the determinations that
    were paid for and read, with every identifier removed.

    Deterministic throughout (no LLM sweep), which is also what keeps the
    markdown, the facts and the chunks in agreement: a sweep rewrites the
    markdown but cannot be replayed identically over a quote, so quotes stop
    matching chunks and the index backfill degrades.

    verify_anonymization already scans every file written here — it was
    written for this and only ever had the markdown half implemented.
    """
    src_dir = DOSSIER_DIR / source_case_id
    out_dir = DOSSIER_DIR / target_case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = personas.load_roster(source_case_id)
    counts: dict[str, int] = {}

    for name, policy in ARTIFACT_POLICIES.items():
        src = src_dir / name
        if not src.exists():
            continue
        records = [
            _translate_record(r, policy, mapping, extras, f"{name}[{i}]")
            for i, r in enumerate(_read_jsonl(src))
        ]
        _write_jsonl(out_dir / name, records)
        counts[name] = len(records)
        log.info("anonymize artifacts: %s -> %d record(s)", name, len(records))

    src_persons = src_dir / "persons.jsonl"
    if src_persons.exists():
        persons = _translate_persons(
            _read_jsonl(src_persons), target_case_id, roster, mapping, extras,
        )
        _write_jsonl(out_dir / "persons.jsonl", persons)
        counts["persons.jsonl"] = len(persons)
        log.info("anonymize artifacts: persons.jsonl -> %d merged person(s)", len(persons))

    src_matrix = src_dir / "compliance_matrix.json"
    if src_matrix.exists():
        matrix = json.loads(src_matrix.read_text(encoding="utf-8"))
        translated = _translate_record(
            matrix, COMPLIANCE_TOP_POLICY, mapping, extras, "compliance_matrix.json",
        )
        translated["case_id"] = target_case_id
        translated["entries"] = [
            _translate_record(
                e, COMPLIANCE_ENTRY_POLICY, mapping, extras, f"compliance_matrix.entries[{i}]",
            )
            for i, e in enumerate(matrix.get("entries") or [])
        ]
        (out_dir / "compliance_matrix.json").write_text(
            json.dumps(translated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        counts["compliance_matrix.json"] = len(translated["entries"])
        log.info(
            "anonymize artifacts: compliance_matrix.json -> %d entries",
            len(translated["entries"]),
        )

    return counts


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


def _placed_replacement_tokens(source_case_id: str, extras: dict[str, str]) -> set[str]:
    """Folded word-tokens of every replacement this pipeline itself places.

    The residual-proper-noun report exists to surface names nobody chose. A
    name the anonymiser deliberately wrote is the opposite of that, and
    counting it drowns the report — "Maître Beerus" appears on nearly every
    page of the showcase, and _TITLED_RE would flag "Beerus" every time.

    Drawn from the persona pools, the extra-identifier replacements, and the
    persisted person assignments, so this stays correct as the roster changes
    without anyone remembering to update a constant.
    """
    values: list[str] = [
        *personas.RESIDUAL_NATURAL_POOL, *personas.RESIDUAL_LEGAL_POOL,
        *personas.FALLBACK_NATURAL_POOL, *personas.FALLBACK_LEGAL_POOL,
        *personas.PLACE_POOL, *extras.values(),
    ]

    # Names the roster deliberately keeps real are output this pipeline chose
    # to write, exactly like a pseudonym. Counting them buries the report:
    # the counterparty organisations alone accounted for 150+ hits, which is
    # what the report exists to make visible.
    roster = personas.load_roster(source_case_id)
    if roster.keep_real_legal_persons:
        values.extend(
            name for name, person_type, _pid in _load_known_entities(source_case_id)
            if person_type == "legal_person"
        )

    path = _map_path(source_case_id)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            values.extend(str(v) for v in (stored.get("by_person") or {}).values())
            values.extend(str(v) for v in (stored.get("mapping") or {}).values())

    tokens: set[str] = set()
    for value in values:
        for token in re.split(r"[^a-zA-Z0-9À-ÿ]+", _fold(value)):
            if token:
                tokens.add(token)
    return tokens


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

    extras = load_extra_identifiers(source_case_id)
    roster = personas.load_roster(source_case_id)

    # A name the roster deliberately keeps real is not a leak. Searching for
    # it anyway would fail the build on exactly the names it was just told to
    # publish, so the exclusion has to happen here and not only in the mapping.
    needles: list[str] = [
        name for name, person_type, _pid in _load_known_entities(source_case_id)
        if not (person_type == "legal_person" and roster.keep_real_legal_persons)
    ]
    # An extras entry mapped to ITSELF is a deliberate keep, not a needle:
    # it exists to register the token as output this pipeline chose to write,
    # so the residual report stops flagging it. Searching for it here would
    # fail the build on a name the table just said to publish.
    needles.extend(k for k, v in extras.items() if _fold(k) != _fold(v))
    folded_needles = {_fold(n): n for n in needles if len(n) >= MIN_ENTITY_LEN}
    allowed = ALLOWED_PROPER_NOUNS | _placed_replacement_tokens(source_case_id, extras)

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
                # Compare part by part, not whole-string: the shouted pattern
                # captures hyphenated composites ("Whis-Maître"), and a
                # composite built entirely from words we already allow is not
                # a residual identifier — it is our own fiction, hyphenated.
                parts = [p for p in re.split(r"[^a-zA-Z0-9À-ÿ]+", _fold(token)) if p]
                if parts and all(p in allowed or p.isdigit() for p in parts):
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
