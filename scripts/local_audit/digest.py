"""Regenerates audit/DIGEST.md from audit/findings.jsonl.

Full regeneration, not append — fixes the old audit/INDEX.md's inability to
reflect a finding being fixed (resolved findings just drop out of the digest
instead of accumulating forever).
"""
from __future__ import annotations

from datetime import datetime, timezone

from scripts.local_audit import config, findings as findings_store

PASS_LABELS = {
    "doc_drift": "Doc / ADR / CLAUDE.md drift",
    "duplication": "Cross-file semantic duplication",
    "convention": "CLAUDE.md convention compliance",
    "prompt_audit": "LLM prompt-engineering audit",
    "slimming": "Redundancy / dead code / size budgets (deterministic)",
    "slimming_divergence": "Concept-registry divergence (LLM)",
}


def _render_finding(f: dict) -> str:
    loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
    lines = [
        f"- **[{f['confidence']}]** `{loc}` — {f['claim']}",
        f"  - evidence: {f['evidence']}",
    ]
    if f.get("related_files"):
        lines.append(f"  - related: {', '.join(f['related_files'])}")
    lines.append(f"  - id: `{f['id']}`")
    return "\n".join(lines)


def regenerate() -> str:
    store = findings_store.load_all()
    open_findings = [f for f in store.values() if f.get("status") == "open"]
    open_findings.sort(
        key=lambda f: (findings_store.SEVERITY_ORDER.get(f["severity"], 99), f["pass"], f["file"])
    )

    lines = [
        "# Local Audit Digest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Open findings: {len(open_findings)} "
        f"(of {len(store)} total, {len(store) - len(open_findings)} resolved)",
        "",
    ]

    if not open_findings:
        lines.append("No open findings.")
        config.DIGEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(config.DIGEST_MD)

    by_severity: dict[str, list[dict]] = {}
    for f in open_findings:
        by_severity.setdefault(f["severity"], []).append(f)

    for severity in sorted(by_severity, key=lambda s: findings_store.SEVERITY_ORDER.get(s, 99)):
        items = by_severity[severity]
        lines.append(f"## {severity} ({len(items)})")
        lines.append("")
        by_pass: dict[str, list[dict]] = {}
        for f in items:
            by_pass.setdefault(f["pass"], []).append(f)
        for pass_name in sorted(by_pass):
            label = PASS_LABELS.get(pass_name, pass_name)
            lines.append(f"### {label}")
            lines.append("")
            for f in by_pass[pass_name]:
                lines.append(_render_finding(f))
                lines.append("")

    config.DIGEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(config.DIGEST_MD)
