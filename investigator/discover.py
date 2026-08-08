"""Read a case's own instruments and propose the obligations they create.

Until now every obligation in the catalog was typed by hand — which is fine for
one case and impossible as a product. A stranger's convention creates duties
nobody has authored, so `check` runs against generic statute obligations and
says little about them specifically. This closes that gap.

**It proposes; it never installs.** Output goes to `obligations.proposed.yaml`
and the human rename to `obligations.yaml` is the trust boundary. That is not
ceremony: these findings name a professional and are aimed at an insurer, and a
machine-written duty with the wrong bearer or an over-broad term list produces a
confident accusation against the wrong party. The verbatim check below catches
invented text; it cannot catch a real clause misread.

Three things do the actual work, and only one of them is the model:

1. **A deterministic pre-filter.** A case is mostly letters and statements;
   only a few instruments stipulate anything. `lexicon.OBLIGATION_MARKERS`
   picks those out — on the live case, 8 documents of 55, with the convention
   ranked first. Running a model over the other 47 would be waste.
2. **The model**, one call per page window, told which `actor_role` ids exist in
   this case so it names a bearer that is real rather than inventing one.
3. **The verbatim check**, which is where correctness is enforced: an
   `excerpt_fr` that does not appear in its own source document is dropped
   before it is ever written. `passes/search.py::_validate_document` then
   re-applies the same predicate on every cycle, so a proposal that survives
   here stays honest later.

Prompt shape follows `ingestion/dossier/facts.py` — French, numbered
constraints, a literal JSON skeleton, an explicit empty-case escape, and its
load-bearing rule: no verbatim quote, no record. Caching follows
`ingestion/dossier/extract.py` rather than `facts.py`, which re-calls the model
on every run; a long contract is expensive to re-read.

Usage:
    uv run python -m investigator.discover --case-id mycase
    uv run python -m investigator.discover --case-id mycase --dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from investigator import cache, config, graph as graph_mod, ollama
from investigator.lexicon import fold, obligation_marker_hits
from investigator.schema import Obligation

log = logging.getLogger(__name__)

PASS_NAME = "discover"

# Pages, because `extract.py` already writes `## Page N` dividers and a page is
# small enough to keep the local model's window comfortable.
_PAGE_RE = re.compile(r"^## Page \d+\s*$", re.MULTILINE)

SEVERITY_ALIASES = {
    "critique": "critical", "critical": "critical",
    "elevee": "high", "élevée": "high", "high": "high",
    "moyenne": "medium", "medium": "medium",
    "faible": "low", "low": "low",
}

SYSTEM_PROMPT = """Tu es un juriste qui relève, dans un acte français, les \
OBLIGATIONS qu'il crée — et rien d'autre.

Une obligation est un engagement de faire ou de ne pas faire, mis à la charge \
d'une partie désignée. Ce n'est pas un fait, pas une déclaration, pas une \
description de ce qui s'est passé.

Règles impératives :

1. Chaque obligation doit porter un extrait VERBATIM : la phrase exacte de \
l'acte qui la stipule, copiée sans reformulation, sans coupure au milieu d'un \
mot. Pas d'extrait, pas d'obligation.
2. Le débiteur doit être choisi PARMI LES RÔLES FOURNIS. Si aucun ne convient, \
n'émets pas l'obligation.
3. Les termes de preuve sont les mots que contiendrait une pièce attestant \
l'EXÉCUTION de l'obligation — pas les mots de la clause elle-même. Pour « le \
notaire remettra un extrait », la preuve serait « remise », « extrait », \
« accusé de réception », non « remettra ».
4. Un délai n'est renseigné que s'il est écrit dans la clause. Sinon : null.
5. N'invente rien. N'infère pas d'intention. Ne qualifie pas de faute.
6. Au plus 8 obligations par extrait. Une page qui semble en créer davantage \
signale que tu reformules au lieu d'extraire : retiens les plus nettes.

Si le passage ne crée aucune obligation, retourne {"obligations": []}.

Réponds en JSON strict :
{"obligations": [
  {"titre": "<phrase courte, ce que la clause exige>",
   "extrait_verbatim": "<phrase exacte de l'acte>",
   "debiteur_role_id": "<un des rôles fournis>",
   "termes_preuve": ["<mot>", "<mot>"],
   "delai_jours": null,
   "gravite": "critical|high|medium|low"}
]}"""


@dataclass
class DiscoveryReport:
    case_id: str
    documents_scanned: int = 0
    documents_read: int = 0
    windows: int = 0
    proposed: list[Obligation] = field(default_factory=list)
    dropped_not_verbatim: list[str] = field(default_factory=list)
    dropped_unknown_role: list[str] = field(default_factory=list)
    dropped_invalid: list[str] = field(default_factory=list)
    path: str = ""

    def summary(self) -> str:
        return (
            f"discover · case_id={self.case_id} "
            f"documents={self.documents_read}/{self.documents_scanned} "
            f"windows={self.windows} proposed={len(self.proposed)} "
            f"rejetés: extrait_absent={len(self.dropped_not_verbatim)} "
            f"rôle_inconnu={len(self.dropped_unknown_role)} "
            f"invalides={len(self.dropped_invalid)}"
        )


# ========== selection ==========


def select_documents(
    case_graph: graph_mod.CaseGraph, doc_pattern: str | None = None
) -> list[tuple[str, int]]:
    """`(doc_id, marker_count)` for documents that look like instruments.

    Ordered by marker count then id, so the sequence is stable and the richest
    instrument is read first — useful when a run is interrupted.
    """
    from fnmatch import fnmatch

    out = []
    for doc_id, folded_text in case_graph.doc_text_folded.items():
        if doc_pattern:
            if fnmatch(doc_id, doc_pattern):
                out.append((doc_id, len(obligation_marker_hits(folded_text))))
            continue
        hits = obligation_marker_hits(folded_text)
        if hits:
            out.append((doc_id, len(hits)))
    return sorted(out, key=lambda p: (-p[1], p[0]))


def _windows(markdown: str) -> list[str]:
    """Split on the page dividers `extract.py` writes; whole text if none."""
    parts = [p.strip() for p in _PAGE_RE.split(markdown)]
    return [p for p in parts if p] or ([markdown.strip()] if markdown.strip() else [])


# ========== identity ==========


def _slug(text: str, limit: int = 40) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return slug[:limit].rstrip("-")


def obligation_id_for(doc_id: str, titre: str) -> str:
    """Deterministic and stable across runs, so re-discovery does not re-mint ids.

    Prefixed `dec-` (découvert) so a reader can tell at a glance which entries a
    machine proposed and which a human wrote.
    """
    base = f"dec-{_slug(doc_id, 18)}-{_slug(titre, 34)}".strip("-")
    base = re.sub(r"-+", "-", base)[:60].rstrip("-")
    return base if re.match(r"^[a-z][a-z0-9-]{2,60}$", base) else f"dec-{_slug(titre, 50)}"


# ========== proposal construction ==========


def _to_obligation(
    raw: dict, doc_id: str, doc_markdown_folded: str, known_roles: set[str],
    report: DiscoveryReport,
) -> Obligation | None:
    """One model proposal → a validated `Obligation`, or None with a reason logged."""
    titre = str(raw.get("titre", "")).strip()
    excerpt = str(raw.get("extrait_verbatim", "")).strip()
    role = str(raw.get("debiteur_role_id", "")).strip()
    if not (titre and excerpt and role):
        report.dropped_invalid.append(f"{doc_id}: champ manquant ({titre[:40]!r})")
        return None

    # The guard. An excerpt that is not in its own source document is invented,
    # whatever else it looks like.
    if fold(excerpt) not in doc_markdown_folded:
        report.dropped_not_verbatim.append(f"{doc_id}: {excerpt[:70]!r}")
        return None

    if role not in known_roles:
        report.dropped_unknown_role.append(f"{doc_id}: {role!r} ({titre[:40]!r})")
        return None

    terms = [str(t).strip() for t in raw.get("termes_preuve", []) if str(t).strip()]
    if not terms:
        report.dropped_invalid.append(f"{doc_id}: aucun terme de preuve ({titre[:40]!r})")
        return None

    deadline = raw.get("delai_jours")
    deadline = int(deadline) if isinstance(deadline, (int, float)) and deadline else None

    try:
        return Obligation.model_validate({
            "obligation_id": obligation_id_for(doc_id, titre),
            "rule_version": "v1",
            "title_fr": titre,
            "severity": SEVERITY_ALIASES.get(str(raw.get("gravite", "")).lower(), "medium"),
            "externalisable": True,
            "source": {
                "kind": "contract_clause",
                "ref": f"{doc_id} — {titre}",
                "doc_id_pattern": doc_id,
                "excerpt_fr": excerpt,
            },
            "bearer": {
                "actor_roles": [role],
                "bearer_note_fr": "Débiteur relevé dans l'acte lui-même ; à confirmer.",
            },
            "window": {"deadline_days": deadline} if deadline else {},
            "expected_evidence": {
                "all_of": [{
                    "kind": "fact_match",
                    "any_actor": True,
                    "any_terms_fr": terms,
                    "min_count": 1,
                }]
            },
            "evidence_scope": {"require_gate_status": "ok"},
            "adjudicate": "llm_local",
            "claim_template_fr": (
                "Obligation {obligation_id} ({source_ref}) : aucun fait du dossier "
                "n'atteste l'exécution de cette stipulation."
            ),
            "qualification_interne_fr": (
                "Obligation relevée automatiquement dans l'acte. Débiteur, délai et "
                "termes de preuve restent à vérifier avant tout usage."
            ),
            "gravite_interne": "faute_simple",
            "confounders_seed": [
                "L'exécution a pu intervenir par un canal non versé au dossier.",
                "La portée exacte de la clause reste à confirmer par lecture intégrale de l'acte.",
            ],
        })
    except ValidationError as exc:
        report.dropped_invalid.append(f"{doc_id}: {titre[:40]!r} — {str(exc)[:120]}")
        return None


# ========== the pass ==========


def discover(
    case_id: str,
    doc_pattern: str | None = None,
    dossier_dir: Path | None = None,
    model: str | None = None,
    dry_run: bool = False,
) -> DiscoveryReport:
    paths = config.CasePaths.for_case(case_id, dossier_dir=dossier_dir)
    case_graph = graph_mod.load_graph(case_id, dossier_dir=dossier_dir)
    report = DiscoveryReport(case_id=case_id)
    report.documents_scanned = len(case_graph.doc_text_folded)

    selected = select_documents(case_graph, doc_pattern)
    report.documents_read = len(selected)
    known_roles = set(case_graph.roles) or {f.actor_role for f in case_graph.facts.values()}
    roles_block = ", ".join(sorted(known_roles)) or "(aucun rôle catalogué)"
    judge = model or config.JUDGE_MODELS[0]

    log.info(
        "discover: %d document(s) porteurs de clauses sur %d, %d rôle(s) connus",
        len(selected), report.documents_scanned, len(known_roles),
    )
    if dry_run:
        for doc_id, hits in selected:
            log.info("  %s (%d marqueur(s))", doc_id, hits)
        return report

    seen: set[str] = set()
    for doc_id, _hits in selected:
        md_path = paths.case_dir / "extracted" / f"{doc_id}.md"
        if not md_path.exists():
            continue
        markdown = md_path.read_text(encoding="utf-8")
        folded_doc = fold(markdown)

        for idx, window in enumerate(_windows(markdown)):
            report.windows += 1
            content_hash = cache.content_hash_for_text(window)
            subject = f"{judge}|{case_id}|{doc_id}#w{idx:03d}"
            result = cache.get(
                paths, PASS_NAME, config.PROMPT_VERSIONS["discover"], subject, content_hash
            )
            if result is None:
                out = ollama.call_json(
                    SYSTEM_PROMPT,
                    f"Rôles connus de ce dossier : {roles_block}\n\n"
                    f"Extrait de l'acte (doc_id={doc_id}) :\n\n{window}",
                    model=judge,
                    # Extraction, not judgment — so no reasoning trace. The task
                    # is to spot a sentence that stipulates a duty and copy it
                    # verbatim; `facts.py` does the same class of work without
                    # thinking, and the verbatim check verifies the result
                    # deterministically afterwards. Thinking earns its cost in
                    # the check classifier, where the question is whether a
                    # quote *evidences performance*, and it stays on there.
                    #
                    # Measured on one 3.4k-char window of the convention:
                    # think=True exceeded the 900 s timeout and returned
                    # nothing; think=False answered in 38 s with the same six
                    # clauses. Across 77 windows that is the difference between
                    # a working stage and a stalled one.
                    think=False,
                    num_ctx=config.DISCOVER_NUM_CTX,
                )
                if not isinstance(out, dict) or "obligations" not in out:
                    # No verdict is not an empty document; never cached.
                    log.warning("discover: pas de réponse exploitable pour %s#w%d", doc_id, idx)
                    continue
                result = out
                cache.set(
                    paths, PASS_NAME, config.PROMPT_VERSIONS["discover"],
                    subject, content_hash, result,
                )

            for raw in result.get("obligations", []):
                if not isinstance(raw, dict):
                    continue
                obligation = _to_obligation(raw, doc_id, folded_doc, known_roles, report)
                if obligation is None or obligation.obligation_id in seen:
                    continue
                seen.add(obligation.obligation_id)
                report.proposed.append(obligation)

    report.path = write_proposal(paths, report)
    return report


def write_proposal(paths: config.CasePaths, report: DiscoveryReport) -> str:
    """Write the proposal. Never `obligations.yaml` — that rename is the review."""
    payload = {
        "catalog_id": f"decouvert_{paths.case_id}",
        "catalog_version": "v1",
        "jurisdiction": "FR",
        "person_free": False,
        "obligations": [
            o.model_dump(mode="json", exclude_defaults=True) for o in report.proposed
        ],
    }
    header = (
        "# PROPOSITION — relevée automatiquement dans les actes de ce dossier.\n"
        "#\n"
        "# À relire avant usage. Chaque extrait a été vérifié mot pour mot contre le\n"
        "# document source ; en revanche le débiteur, le délai et les termes de preuve\n"
        "# sont des lectures du modèle et n'ont été vérifiés par personne.\n"
        "#\n"
        "# Pour l'activer :  mv obligations.proposed.yaml obligations.yaml\n"
        "# Ce fichier n'est jamais lu par l'analyse tant qu'il porte ce nom.\n\n"
    )
    paths.case_dir.mkdir(parents=True, exist_ok=True)
    paths.case_catalog_proposal.write_text(
        header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return str(paths.case_catalog_proposal)


# ========== CLI ==========


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose obligations from a case's own instruments."
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--doc-pattern", default=None, help="glob over doc_id, overrides the filter")
    parser.add_argument("--model", default=None, help=f"default: {config.JUDGE_MODELS[0]}")
    parser.add_argument("--dry-run", action="store_true", help="list the documents, call nothing")
    args = parser.parse_args()

    report = discover(
        args.case_id, doc_pattern=args.doc_pattern, model=args.model, dry_run=args.dry_run
    )
    print(report.summary())
    for reason, items in (
        ("extrait absent du document", report.dropped_not_verbatim),
        ("rôle inconnu", report.dropped_unknown_role),
        ("invalide", report.dropped_invalid),
    ):
        for item in items[:5]:
            print(f"  rejeté ({reason}) · {item}")
    if report.path:
        print(f"proposition : {report.path}")
        print("  relire, puis : mv obligations.proposed.yaml obligations.yaml")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
