"""Rendering: the local digest, and the one gated path to a shareable extract.

Two writers with deliberately different rules.

`render_digest` is the operator's view: every open finding, whatever its tier,
including the ones the gate would refuse. That is the point — a T5 hypothesis
is exactly what an investigator needs to see and exactly what must not leave
the machine.

`render_outbound` is the only function in the plane that writes to
`CasePaths.outbound_dir`, and it obtains findings solely from
`schema.externalisable_findings()`. Keeping it to one function with one source
is what makes the property testable rather than aspirational; a second path
would be undiscoverable by inspection.

Both regenerate in full rather than appending, so a resolved finding simply
stops appearing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from investigator import config, schema
from investigator.catalog import Catalog
from investigator.config import CasePaths
from investigator.store import SEVERITY_ORDER

PASS_LABELS = {
    "graph": "Intégrité du graphe",
    "search": "Vérification des citations",
    "check": "Contrôle des obligations",
    "contradict": "Contradictions",
    "attack": "Contre-analyse",
}

TIER_NOTES = {
    "T1": "documenté, plusieurs sources concordantes",
    "T2": "documenté, source unique",
    "T3": "absence constatée sur un périmètre couvert",
    "T4": "verdict adjudiqué sans corroboration déterministe",
    "T5": "hypothèse de travail — périmètre non couvert ou substrat défectueux",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _render_finding(f: dict) -> list[str]:
    lines = [f"- **[{f['tier']}]** `{f['subject']}` — {f['claim']}"]
    if f.get("evidence"):
        lines.append(f"  - constat : {f['evidence']}")
    for confounder in f.get("confounders", []):
        mark = "**neutralisant**" if confounder.get("dispositive") else "contre-argument"
        lines.append(f"  - {mark} : {confounder['text_fr']}")
    lines.append(f"  - id : `{f['id']}`")
    return lines


# ========== the local digest ==========


def render_digest(paths: CasePaths, store_: dict[str, dict], catalog: Catalog | None = None) -> str:
    """Rewrite `investigation/DIGEST.md` from the store. Returns its path."""
    paths.investigation_dir.mkdir(parents=True, exist_ok=True)
    open_findings = [f for f in store_.values() if f.get("status") == "open"]
    resolved = len(store_) - len(open_findings)

    out = [
        f"# Investigation — {paths.case_id}",
        "",
        f"Généré le {_now()}. Vue locale, complète : elle inclut les constatations "
        "de tier T4-T5 que la barrière d'externalisation refuse. Ne pas diffuser.",
        "",
        f"Constatations ouvertes : {len(open_findings)} "
        f"(sur {len(store_)} au total, {resolved} résolue(s)).",
    ]
    if catalog is not None:
        out.append(f"Obligations au catalogue : {len(catalog)}.")
    out.append("")

    if not open_findings:
        out += ["Aucune constatation ouverte.", ""]
        paths.digest_md.write_text("\n".join(out), encoding="utf-8")
        return str(paths.digest_md)

    open_findings.sort(
        key=lambda f: (
            schema.TIER_ORDER.get(f.get("tier", "T5"), 99),
            SEVERITY_ORDER.get(f.get("severity", "info"), 99),
            f["pass"],
            f["subject"],
        )
    )

    by_tier: dict[str, list[dict]] = {}
    for f in open_findings:
        by_tier.setdefault(f.get("tier", "T5"), []).append(f)

    for tier in sorted(by_tier, key=lambda t: schema.TIER_ORDER.get(t, 99)):
        group = by_tier[tier]
        out += [f"## {tier} — {TIER_NOTES.get(tier, '')} ({len(group)})", ""]
        by_pass: dict[str, list[dict]] = {}
        for f in group:
            by_pass.setdefault(f["pass"], []).append(f)
        for pass_name in sorted(by_pass):
            out += [f"### {PASS_LABELS.get(pass_name, pass_name)}", ""]
            for f in by_pass[pass_name]:
                out += _render_finding(f)
            out.append("")

    paths.digest_md.write_text("\n".join(out), encoding="utf-8")
    return str(paths.digest_md)


# ========== the gated outbound extract ==========


def render_outbound(
    paths: CasePaths,
    case_id: str,
    store_: dict[str, dict],
    catalog: Catalog | None = None,
) -> str:
    """Write the gated extract. The ONLY writer to `paths.outbound_dir`.

    Raises for a case that is not outbound-eligible — `externalisable_findings`
    refuses before anything is written, so a refused case leaves no partial
    file behind.
    """
    obligations = catalog.obligations if catalog is not None else None
    kept = schema.externalisable_findings(
        store_.values(), case_id=case_id, obligations=obligations
    )

    paths.outbound_dir.mkdir(parents=True, exist_ok=True)
    target = paths.outbound_dir / "findings.md"

    out = [
        f"# Constatations — {case_id}",
        "",
        f"Généré le {_now()}. Extrait filtré : seules les constatations de tier "
        f"{config.OUTBOUND_MIN_TIER} ou supérieur, non neutralisées par un "
        "contre-argument dirimant, et marquées externalisables.",
        "",
        f"Constatations retenues : {len(kept)}.",
        "",
    ]
    for f in kept:
        out += _render_finding(f)
    out.append("")

    target.write_text("\n".join(out), encoding="utf-8")
    (paths.outbound_dir / "findings.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return str(target)
