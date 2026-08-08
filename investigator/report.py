"""The readable brief: one argued page per finding.

`render.py` produces the operator's index — every finding, one line each. That
is the wrong shape for the question a finding actually has to answer, which is
not "what did the engine flag" but:

    what was owed, by whom, under which text, what does the file show, what is
    missing, and why does the absence matter?

So this module joins the three things the store deliberately keeps apart —
the finding, the obligation it came from, and the facts it points at — and
renders them as an argument with its sources attached. Statute anchors carry
their Légifrance URL straight from `data/chunks.csv`, so every citation is one
click from the text in force.

It also opens the model's working. For a gap, the closest quotes in the dossier
are listed **with the classifier's verdict on each** — `stipulation`,
`demande`, `manquement` — so "no evidence of performance" is shown rather than
asserted. A reader who disagrees can see exactly which quote to argue about.

Read-only, like everything in this plane. It writes one file inside the case
directory and asserts nothing the store does not already hold.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from investigator import schema
from investigator.catalog import Catalog
from investigator.config import CasePaths
from investigator.graph import CaseGraph
from investigator.store import SEVERITY_ORDER

# Statuses worth arguing. `satisfied` is recorded in the store for completeness
# — it is the other half of the denominator — but it is not a brief.
ACTIONABLE = ("gap", "window_breach", "unverifiable")

SEVERITY_FR = {
    "critical": "critique",
    "high": "élevée",
    "medium": "moyenne",
    "low": "faible",
    "info": "information",
}

ROLE_FR = {
    "execution": "exécution",
    "stipulation": "énoncé de l'obligation",
    "demande": "réclamation de l'exécution",
    "manquement": "constat d'inexécution",
    "autre": "sans rapport",
}

GRAVITE_FR = {
    "faute_simple": "faute simple",
    "faute_caracterisee": "faute caractérisée",
    "faute_lourde": "faute lourde",
    "susceptible_qualification_penale": "susceptible de qualification pénale",
}

TIER_FR = {
    "T1": "documenté, plusieurs sources concordantes",
    "T2": "documenté, source unique",
    "T3": "absence constatée sur un périmètre entièrement vérifié",
    "T4": "absence constatée sur un périmètre partiellement vérifié, ou verdict adjugé",
    "T5": "hypothèse de travail — périmètre non vérifié",
}


# ========== joins ==========


def _classifier_verdicts(paths: CasePaths) -> dict[str, dict]:
    """Every stored classification, keyed by `obligation_id|fact_id`.

    Read from the verification cache rather than recomputed: the point is to
    show what the run actually decided, not what a fresh call would say.
    """
    out: dict[str, dict] = {}
    if not paths.cache_db.exists():
        return out
    conn = sqlite3.connect(paths.cache_db)
    try:
        rows = conn.execute(
            "SELECT subject, result_json FROM verifications WHERE pass_name = 'check'"
        ).fetchall()
    finally:
        conn.close()
    for subject, payload in rows:
        # subject is "{obligation_id}@{rule_version}|{case_id}|{fact_id}"
        parts = subject.split("|")
        if len(parts) != 3:
            continue
        obligation_id = parts[0].split("@")[0]
        try:
            out[f"{obligation_id}|{parts[2]}"] = json.loads(payload)
        except json.JSONDecodeError:
            continue
    return out


def _status_of(finding: dict) -> str:
    head = finding.get("evidence", "")
    return head.split("·")[0].replace("statut=", "").strip()


def _quote_line(graph: CaseGraph, fact_id: str, verdict: dict | None = None) -> list[str]:
    fact = graph.facts.get(fact_id)
    if fact is None:
        return []
    date = fact.date or "date non précisée"
    quote = " ".join(fact.verbatim_quote.split())
    line = [f"- **{date}** · *{fact.actor_role}* — « {quote} »"]
    if verdict is not None:
        role = ROLE_FR.get(verdict.get("role", ""), verdict.get("role", ""))
        motif = " ".join(str(verdict.get("motif", "")).split())
        line.append(f"  - qualification : **{role}**" + (f" — {motif}" if motif else ""))
    line.append(f"  - pièce : `{fact.source_doc_id}`")
    return line


# ========== the brief ==========


def _section(
    index: int,
    finding: dict,
    graph: CaseGraph,
    catalog: Catalog,
    verdicts: dict[str, dict],
) -> list[str]:
    obligation = catalog.get(finding.get("obligation_id") or "")
    status = _status_of(finding)
    pointers = finding.get("evidence_pointers", {})
    out: list[str] = []

    title = obligation.title_fr if obligation else finding["claim"][:90]
    out += [
        f"## {index}. {title}",
        "",
        f"**Gravité** {SEVERITY_FR.get(finding.get('severity', ''), finding.get('severity', ''))} "
        f"· **Fiabilité** {finding['tier']} — {TIER_FR.get(finding['tier'], '')} "
        f"· `{finding['id']}`",
        "",
        f"**Constat.** {' '.join(finding['claim'].split())}",
        "",
    ]

    # ---- the internal register ----
    # Deliberately unrestrained, and deliberately never externalised: the gate
    # strips these fields and refuses any outbound text carrying the vocabulary
    # they use. See ADR #71.
    qualification = finding.get("qualification_interne", "")
    gravite = finding.get("gravite_interne", "")
    penal_refs = finding.get("penal_refs", [])
    if qualification or penal_refs:
        out += ["### Qualification (interne — ne pas externaliser)", ""]
        if qualification:
            out += [" ".join(qualification.split()), ""]
        if gravite:
            out += [f"**Gravité retenue** : {GRAVITE_FR.get(gravite, gravite)}.", ""]
        for chunk_id in penal_refs:
            row = graph.statute.get(chunk_id)
            if row is None:
                continue
            texte = " ".join(str(row.get("texte", "")).split())
            url = str(row.get("url", "")).strip()
            num = str(row.get("num", chunk_id))
            out += [
                f"> {texte[:600]}{'…' if len(texte) > 600 else ''}",
                "",
                f"— **{row.get('source_label', '')} art. {num}**"
                + (f" · [Légifrance]({url})" if url else "")
                + f" · `{row.get('etat', '')}`",
                "",
            ]

    # ---- the legal basis ----
    if obligation is not None:
        src = obligation.source
        out += ["### Fondement", ""]
        excerpt = " ".join(src.excerpt_fr.split())
        out.append(f"> {excerpt}")
        out.append("")
        row = graph.statute.get(src.chunk_id or "")
        if row is not None:
            url = str(row.get("url", "")).strip()
            cite = f"— **{src.ref}**"
            if url:
                cite += f" · [texte en vigueur sur Légifrance]({url})"
            out.append(cite)
        elif src.doc_id_pattern:
            match = sorted(d for d in graph.doc_text_folded if src.doc_id_pattern.strip("*") in d)
            where = f" · pièce `{match[0]}`" if match else ""
            out.append(f"— **{src.ref}**{where}")
        else:
            out.append(f"— **{src.ref}**")
        for anchor in obligation.also_anchored:
            arow = graph.statute.get(anchor.chunk_id)
            aurl = str(arow.get("url", "")).strip() if arow is not None else ""
            note = " ".join(anchor.note_fr.split())
            out.append(
                f"— également : `{anchor.chunk_id}`"
                + (f" · [Légifrance]({aurl})" if aurl else "")
                + (f" — {note}" if note else "")
            )
        out += ["", f"**Débiteur de l'obligation** : {', '.join(obligation.bearer.actor_roles)}."]
        if obligation.bearer.bearer_note_fr:
            out.append(" ".join(obligation.bearer.bearer_note_fr.split()))
        out.append("")

    # ---- what the file does establish ----
    trigger = pointers.get("trigger_fact_id")
    matched = pointers.get("fact_ids", [])
    if trigger or matched:
        out += ["### Ce que le dossier établit", ""]
        if trigger:
            out += _quote_line(graph, trigger)
        for fact_id in matched[:6]:
            if fact_id != trigger:
                out += _quote_line(graph, fact_id)
        out.append("")

    # ---- what it does not ----
    if status in ("gap", "unverifiable"):
        scope = pointers.get("scope_doc_ids", [])
        candidates = pointers.get("candidate_fact_ids", [])
        out += ["### Ce que le dossier ne contient pas", ""]
        if status == "unverifiable":
            out.append(
                "Le périmètre documentaire nécessaire au contrôle n'est pas couvert : "
                "l'absence de pièce ne peut pas, ici, être distinguée d'un défaut de "
                "collecte. **Pièce à réclamer avant toute conclusion.**"
            )
        else:
            out.append(
                f"Aucune pièce du dossier n'atteste l'exécution. Périmètre examiné : "
                f"**{len(scope)} document(s)** rattachés au débiteur de l'obligation."
            )
        out.append("")
        if candidates:
            out += [
                "Pièces les plus proches, et ce qu'elles sont en réalité :",
                "",
            ]
            for fact_id in candidates[:6]:
                verdict = verdicts.get(f"{finding.get('obligation_id')}|{fact_id}")
                out += _quote_line(graph, fact_id, verdict)
            out.append("")

    # ---- why it matters ----
    if obligation is not None and obligation.absence_significance_fr:
        out += [
            "### Portée",
            "",
            " ".join(obligation.absence_significance_fr.split()),
            "",
        ]

    # ---- the other side ----
    confounders = finding.get("confounders", [])
    if confounders:
        out += ["### Objections prévisibles", ""]
        for c in confounders:
            mark = " **(dirimante)**" if c.get("dispositive") else ""
            out.append(f"- {' '.join(str(c.get('text_fr', '')).split())}{mark}")
        out.append("")

    out += ["---", ""]
    return out


def render_brief(
    paths: CasePaths, store_: dict[str, dict], graph: CaseGraph, catalog: Catalog
) -> str:
    """Write `investigation/RAPPORT.md`. Returns its path."""
    paths.investigation_dir.mkdir(parents=True, exist_ok=True)
    verdicts = _classifier_verdicts(paths)

    actionable = [
        f
        for f in store_.values()
        if f.get("status") == "open"
        and f.get("pass") == "check"
        and _status_of(f) in ACTIONABLE
    ]
    actionable.sort(
        key=lambda f: (
            SEVERITY_ORDER.get(f.get("severity", "info"), 99),
            schema.TIER_ORDER.get(f.get("tier", "T5"), 99),
            f.get("obligation_id") or "",
        )
    )

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = [
        f"# Rapport d'analyse — {paths.case_id}",
        "",
        f"Généré le {generated}. **Document de travail interne.** Chaque constat "
        "porte son fondement, les pièces du dossier qui l'établissent, ce qui manque, "
        "et les objections prévisibles. Rien ici n'est une conclusion juridique : "
        "les constats sont à vérifier pièce en main.",
        "",
        f"**{len(actionable)} constat(s) à instruire.**",
        "",
        "| # | Obligation | Gravité | Fiabilité |",
        "|---|---|---|---|",
    ]
    for i, f in enumerate(actionable, 1):
        ob = catalog.get(f.get("obligation_id") or "")
        out.append(
            f"| {i} | {ob.title_fr if ob else f.get('obligation_id')} "
            f"| {SEVERITY_FR.get(f.get('severity', ''), '')} | {f['tier']} |"
        )
    out += ["", "---", ""]

    for i, f in enumerate(actionable, 1):
        out += _section(i, f, graph, catalog, verdicts)

    if not actionable:
        out += ["Aucun constat à instruire.", ""]

    target = paths.investigation_dir / "RAPPORT.md"
    target.write_text("\n".join(out), encoding="utf-8")
    return str(target)
