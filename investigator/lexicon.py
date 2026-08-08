"""Person-free French legal lexicon and text normalisation.

Committed, and it must stay free of any party name, address, or identifier —
this file is the one piece of matching logic shared by every case, so a name
leaking in here would leak into every future case's catalog too. A test asserts
it against the vitrine roster.

Normalisation is deliberately aggressive and one-way. French dossier text mixes
straight and curly apostrophes, non-breaking spaces inside amounts
(`195 572 €`), and inconsistent accenting; comparing raw strings across
documents from four different institutions produces false negatives that look
exactly like real gaps.
"""
from __future__ import annotations

import re
import unicodedata

# ========== normalisation ==========

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'"})
_WS_RE = re.compile(r"\s+")


def fold(text: str | None) -> str:
    """Accent-fold, lowercase, normalise apostrophes and whitespace.

    NFKD then dropping combining marks, so `créance` and `creance` compare
    equal. Not reversible and not for display — matching only.
    """
    if not text:
        return ""
    text = text.translate(_APOSTROPHES)
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WS_RE.sub(" ", stripped.lower()).strip()


def contains_term(haystack_folded: str, term: str) -> bool:
    """Substring match on already-folded text, folding the needle to match."""
    needle = fold(term)
    return bool(needle) and needle in haystack_folded


# ========== amounts ==========

# The thousands separator in this corpus is usually U+00A0 (non-breaking) and
# sometimes U+202F (narrow no-break), because the source PDFs were typeset in
# French. A regex written with a plain `\s` on a NFKD-folded string catches
# both; on raw text it does not, which is why parsing runs after fold().
_MONEY_RE = re.compile(r"(\d[\d\s.,]*)\s*(?:€|eur\b|euros?\b)")
_COUNT_RE = re.compile(
    r"(\d+)\s+(heritiers?|indivisaires?|nus?-proprietaires?|petits-enfants|ayants? droit)"
)


def parse_amounts_eur(text_folded: str) -> set[int]:
    """Whole-euro amounts mentioned in already-folded text.

    Cents are dropped on purpose: a bucket key of `23477` must group
    `23 477,34 €` with `23.477,34 EUR` and with `23477 euros`. Two facts about
    the same payment written by two institutions are the contradiction signal;
    a two-cent rounding difference is not.
    """
    out: set[int] = set()
    for match in _MONEY_RE.finditer(text_folded):
        digits = re.sub(r"[^\d,.]", "", match.group(1))
        # Strip a decimal tail of exactly one or two digits; everything else is
        # a thousands separator that survived folding.
        digits = re.sub(r"[.,]\d{1,2}$", "", digits)
        digits = re.sub(r"[^\d]", "", digits)
        if digits:
            out.add(int(digits))
    return out


def parse_party_counts(text_folded: str) -> set[tuple[int, str]]:
    """`(count, noun)` pairs like `(3, "heritiers")` from folded text."""
    return {(int(n), noun) for n, noun in _COUNT_RE.findall(text_folded)}


# ========== outbound vocabulary discipline ==========

# Terms that characterise conduct as intentional. These may appear freely in
# internal analysis and must NEVER appear in outbound text.
#
# The reason is not squeamishness, it is coverage. Art. L.113-1 al. 2 of the
# Code des assurances excludes faute intentionnelle ou dolosive from cover, so
# a dolosive framing in a letter to the professional or her insurer voids the
# very pocket the claim is aimed at. The internal register exists precisely so
# the operator can name the conduct without that cost; the gate is what keeps
# the two apart.
OUTBOUND_FORBIDDEN_TERMS = (
    "detournement", "detourne", "detourner",
    "dissimulation", "dissimule",
    "appropriation", "approprie",
    "abus de confiance", "escroquerie", "malversation",
    "soustraction frauduleuse", "frauduleux", "frauduleuse",
    "intentionnel", "intentionnelle", "dolosif", "dolosive",
    "faux et usage de faux",
)


def forbidden_outbound_terms(text: str | None) -> list[str]:
    """Forbidden terms present in `text`, folded so accents cannot hide one."""
    folded = fold(text)
    return [term for term in OUTBOUND_FORBIDDEN_TERMS if term in folded]


# ========== action polarity ==========

# Authored antonym table for the `sens_action` contradiction bucket. Each pair
# is two ways of describing the same event with opposite polarity; a document
# asserting one and another asserting the other, for the same actor and target,
# is a candidate incompatibility. Kept small and legal-specific on purpose —
# a general antonym list produces noise, not candidates.
#
# Stems, not infinitives, because the corpus conjugates: "a refusé", "refusant"
# and "refus" must all reach the same axis. Stems are written out rather than
# derived by chopping a suffix, so each one is a reviewable choice — a
# mechanical `verb[:-2]` turns "omettre" into "omett", which matches nothing
# the documents actually contain.
ANTONYM_AXES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("accepter/refuser", ("accept", "agre"), ("refus", "rejet", "decline")),
    ("verser/retenir", ("vers", "regl", "pai", "credit"), ("reten", "preleve", "debit")),
    ("remettre/conserver", ("remis", "remet", "deliv", "transmis"), ("conserv", "gard")),
    ("debloquer/bloquer", ("debloqu", "liber"), ("bloqu", "gel", "consign")),
    ("signer/contester", ("sign", "ratifi"), ("contest", "denie", "desavou")),
    ("notifier/omettre", ("notifi", "inform", "avis"), ("omis", "omet", "neglig")),
    ("autoriser/interdire", ("autoris", "habilit"), ("interdi", "oppos")),
    ("inscrire/omettre", ("inscri", "porte au passif"), ("omis", "omet")),
)


def action_polarity(action_folded: str) -> tuple[str, int] | None:
    """`(axis, sign)` when folded action text sits on an antonym axis.

    Returns the first axis matched in declaration order, so the result is
    stable for a given string — the contradiction pass buckets on it, and an
    unstable bucket would re-mint finding ids every cycle.
    """
    for axis, positive, negative in ANTONYM_AXES:
        if any(stem in action_folded for stem in positive):
            return axis, 1
        if any(stem in action_folded for stem in negative):
            return axis, -1
    return None
