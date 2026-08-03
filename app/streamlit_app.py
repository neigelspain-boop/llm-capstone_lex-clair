"""Streamlit UI for lex-clair — French legal RAG assistant.

Two-level chat architecture: sidebar lists past CONVERSATIONS (threads),
each conversation contains multiple TURNS (Q&A pairs). Standard
ChatGPT/Claude-style pattern with a "New conversation" button, a
conversation list, and per-turn feedback.

Persistence: Postgres via monitoring.db (ADRs #32, #34). Conversations,
turns, and feedback all live in relational tables. Kill switch in
monitoring.db surfaces to the UI as a persistent warning banner when
the DB is unreachable — the app keeps rendering in-memory state.

Two tabs (ADR #57): "Assistant" holds the chat, "Analyse comparative"
holds the per-role compliance panel. Tabs use on_change="rerun" so `.open`
is readable server-side — that gates the panel's file I/O out of chat turns
AND keeps `st.chat_input` in the page body, where it stays pinned to the
viewport bottom (nested in a tab it would render inline instead).

Public surface: Streamlit runs this file directly. Plane II is
consumed via `rag.flow.run(query)` per ADR #10 and, for the compliance
panel, `rag.compliance.run_compliance_for_role` /
`compare_compliance_for_role` per ADRs #57 and #56.
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from monitoring import db


# ========== labels (EN/FR chrome) ==========

LABELS = {
    "fr": {
        "title": "lex-clair · Assistant juridique français",
        "subtitle": "Comprendre vos droits en matière de succession et d'usufruit.",
        "ask_placeholder": "Posez votre question…",
        "welcome": (
            "Posez une question sur la succession, l'usufruit, "
            "ou la responsabilité notariale."
        ),
        "loading_cold": "Chargement des modèles…",
        "loading": "Recherche en cours…",
        "translating": "Traduction en cours…",
        "translation_failed": (
            "Traduction indisponible ; réponse en français ci-dessous."
        ),
        "translation_notice": "",
        "error": "Une erreur s'est produite. Veuillez réessayer.",
        "citations": "Articles cités",
        "helpful": "Utile ?",
        "yes": "👍 Oui",
        "no": "👎 Non",
        "thanks": "Merci",
        "details": "Détails techniques",
        "rewritten": "Requête reformulée",
        "retrieved": "Chunks récupérés",
        "reranked": "Chunks reclassés",
        "model": "Modèle utilisé",
        "model_key": "Clé catalogue",
        "cost": "Coût",
        "elapsed": "Temps de réponse",
        "conv_id_label": "Identifiant de conversation",
        "turn_id_label": "Identifiant du tour",
        "conv_new": "➕ Nouvelle conversation",
        "conv_heading": "💬 Conversations",
        "conv_empty": "Aucune conversation pour l'instant.",
        "docs_heading": "📎 Documents personnels",
        "docs_upload": "Déposez vos documents",
        "docs_future_note": (
            "_Fonctionnalité future — hors périmètre du projet capstone._"
        ),
        "about_heading": "**À propos**",
        "about_body": (
            "Assistant RAG français pour les questions de succession "
            "et d'usufruit. Les réponses citent le Code civil, le Code "
            "des assurances et le CGI via Légifrance.\n\n"
            "Projet capstone LLM Zoomcamp 2026."
        ),
        "db_offline_banner": (
            "⚠️ Base de données indisponible — les conversations et "
            "retours de cette session ne seront pas conservés."
        ),
    },
    "en": {
        "title": "lex-clair · French legal assistant",
        "subtitle": "Understand your rights in inheritance and usufruct matters.",
        "ask_placeholder": "Ask your question (in French)…",
        "welcome": (
            "Ask a question about inheritance, usufruct, or notarial "
            "liability (in French)."
        ),
        "loading_cold": "Loading models…",
        "loading": "Searching…",
        "translating": "Translating…",
        "translation_failed": (
            "Translation unavailable; showing original French answer below."
        ),
        "translation_notice": (
            "🌐 _Machine translation for peer-review accessibility. "
            "The French answer is authoritative._"
        ),
        "error": "An error occurred. Please try again.",
        "citations": "Cited articles",
        "helpful": "Helpful?",
        "yes": "👍 Yes",
        "no": "👎 No",
        "thanks": "Thanks",
        "details": "Technical details",
        "rewritten": "Rewritten query",
        "retrieved": "Retrieved chunks",
        "reranked": "Reranked chunks",
        "model": "Model used",
        "model_key": "Catalog key",
        "cost": "Cost",
        "elapsed": "Response time",
        "conv_id_label": "Conversation ID",
        "turn_id_label": "Turn ID",
        "conv_new": "➕ New conversation",
        "conv_heading": "💬 Conversations",
        "conv_empty": "No conversations yet.",
        "docs_heading": "📎 Personal documents",
        "docs_upload": "Drop your documents",
        "docs_future_note": (
            "_Future iteration — out of course scope._"
        ),
        "about_heading": "**About**",
        "about_body": (
            "French RAG assistant for inheritance and usufruct "
            "questions. Answers cite the Code civil, Code des "
            "assurances, and CGI via Légifrance.\n\n"
            "LLM Zoomcamp 2026 capstone project."
        ),
        "db_offline_banner": (
            "⚠️ Database unavailable — conversations and feedback from "
            "this session will not be persisted."
        ),
    },
}


# ========== paths + constants ==========

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

RATING_UP = 1
RATING_DOWN = -1

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "⚖️"

TRANSLATION_MODEL = "gpt-4o-mini"

TITLE_MAX_LEN = 45  # truncate long questions for sidebar titles

LANG_OPTIONS = ["🇫🇷 FR", "🇬🇧 EN"]

# Answer-model toggle (ADR #46). Keys must match rag.generate.ANSWER_MODELS.
MODEL_LABELS = {
    "gpt-4o-mini": "⚡ Rapide",
    "opus-4.7": "🧠 Opus 4.7",
    "kimi-k3": "🔬 Kimi K3",
}
COST_HINTS = {
    "gpt-4o-mini": "~0,001€/question",
    "opus-4.7": "~0,05€/question",
    "kimi-k3": "~0,03€/question",
}


# ========== compliance analysis panel (ADR #57) ==========

DOSSIER_DIR = PROJECT_ROOT / "data" / "dossier"

TAB_CHAT = "💬 Assistant"
TAB_ANALYSIS = "🔬 Analyse comparative"

# Keys must match rag.compliance.COMPLIANCE_MODEL_ALTERNATIVES — the panel
# offers exactly the two models that module is willing to run.
COMPLIANCE_MODEL_LABELS = {
    "anthropic/claude-opus-4.7": "🧠 Claude Opus 4.7",
    "moonshotai/kimi-k3": "🔬 Kimi K3",
}
DEFAULT_COMPLIANCE_MODEL = "anthropic/claude-opus-4.7"
DIVERGENCE_MODEL_LABEL = "Haiku 4.5"

STATUS_BADGES = {
    "met": "🟢",
    "breached": "🔴",
    "ambiguous": "🟡",
    "insufficient_evidence": "⚪",
}
STATUS_LABELS = {
    "met": "Respectée",
    "breached": "Manquement",
    "ambiguous": "Ambiguë",
    "insufficient_evidence": "Preuves insuffisantes",
}

# Previews shown before the user spends anything. USD, not EUR — OpenRouter
# bills in dollars and every usage.cost_usd the backend returns is dollars.
# (The sidebar COST_HINTS above mislabel USD as €; correcting them is an
# ADR #57 follow-up, not this deliverable.)
#
# Calibrated 2026-08-03 from `--dry-run` on data/dossier/private: Opus ran
# $0.46 (small cluster) to $0.59 (42 facts, capped to 30), Kimi $0.12 on the
# same 30-fact cluster — the ~5x spread is the published rate difference
# ($15/$75 vs $3/$15 per M), so ONE flat constant across both models would
# misprice whichever is not selected. Re-check with --dry-run if the model
# palette or MAX_FACTS_PER_ROLE changes.
COST_PREVIEW_USD = {
    "anthropic/claude-opus-4.7": 0.55,
    "moonshotai/kimi-k3": 0.12,
}
COST_PREVIEW_FALLBACK_USD = 0.55
COST_COMPARE_USD = 0.70  # Opus + Kimi + Haiku meta-analysis

MATRIX_CACHE_TTL_SECONDS = 300


# ========== feedback wrapper (delegates to monitoring.db) ==========

def _append_feedback(row: dict) -> None:
    """Append one feedback row via monitoring.db.

    Kept as a wrapper so the Day 6 render code's call sites don't
    need to change. row is the 10-column dict Day 6 built for the CSV;
    db.append_feedback picks out the 4 keys it stores and ignores the rest.
    """
    db.append_feedback(row)


# ========== conversation persistence (Postgres via monitoring.db) ==========

def _new_conversation(first_question: str) -> dict:
    """Create a new conversation dict with a generated title."""
    now = datetime.now(timezone.utc).isoformat()
    title = first_question.strip()
    if len(title) > TITLE_MAX_LEN:
        title = title[: TITLE_MAX_LEN - 1] + "…"
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "turns": [],
    }


def _new_turn(question: str) -> dict:
    """Create a new turn dict with a pending (None) result."""
    return {
        "turn_id": str(uuid.uuid4()),
        "question": question.strip(),
        "result": None,
        "answer_en": None,
        "feedback": None,
    }


def _save_conversation(conv: dict) -> None:
    """Persist a conversation via monitoring.db.

    Mutates conv['updated_at'] to NOW before delegating — preserves Day 6's
    sidebar sort-by-recency behavior (see _render_sidebar's sort key).
    """
    conv["updated_at"] = datetime.now(timezone.utc).isoformat()
    db.save_conversation(conv)


def _load_all_conversations() -> dict[str, dict]:
    """Load every persisted conversation from Postgres, keyed by id."""
    return db.load_all_conversations()


# ========== flow.run() wrapper (cached) ==========

@st.cache_resource
def get_flow():
    """Cache flow module across sessions. BGE-M3 + reranker stay warm."""
    from rag import flow

    return flow


# ========== translation (Plane IV concern, not Plane II) ==========

@st.cache_resource
def get_openai_client():
    from openai import OpenAI

    return OpenAI()


def _translate_to_english(french_text: str) -> str | None:
    """Machine-translate a French legal answer to English.

    Returns None on any failure — caller shows warning and French fallback.
    """
    if not french_text.strip():
        return ""
    client = get_openai_client()
    prompt = (
        "Translate the following French legal answer to English.\n\n"
        "Requirements:\n"
        "- Preserve ALL markdown formatting exactly: headers (###), "
        "bullet lists, bold (**text**), and links [text](url).\n"
        "- Keep French legal proper nouns untranslated: 'Code civil', "
        "'Code des assurances', 'quasi-usufruit', 'nue-propriété', "
        "'réserve héréditaire', 'usufruitier', 'nu-propriétaire'.\n"
        "- Keep French legal citations verbatim: 'art. 578 du Code civil'.\n"
        "- Do NOT add preamble, explanation, or translator's notes.\n"
        "- Output only the English translation, nothing else.\n\n"
        "French text:\n"
        f"{french_text}"
    )
    try:
        response = client.responses.create(
            model=TRANSLATION_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        return response.output_text
    except Exception:  # noqa: BLE001 — soft-fail with warning
        traceback.print_exc()
        return None


# ========== compliance panel: data access ==========

@st.cache_resource
def get_compliance():
    """Cache the compliance module across sessions. Importing it pulls in
    rag.retrieve (BGE-M3 + reranker), so this must not happen per rerun."""
    from rag import compliance

    return compliance


@st.cache_data(ttl=MATRIX_CACHE_TTL_SECONDS)
def _load_compliance_matrix(case_id: str) -> dict:
    """Read one case's compliance_matrix.json. Cached — the private matrix is
    ~87 KB and would otherwise re-parse on every rerun of the panel."""
    path = DOSSIER_DIR / case_id / "compliance_matrix.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=MATRIX_CACHE_TTL_SECONDS)
def _list_cases_with_matrix() -> list[str]:
    """Case ids that have a compliance_matrix.json on disk.

    Cases with at least one entry sort first, so the panel opens on
    something analysable: data/dossier/demo/ is an empty fixture and would
    otherwise win on alphabetical order and leave the tab looking dead.
    Empty cases still appear rather than silently vanishing — the role
    selector explains why they have nothing to offer.
    """
    if not DOSSIER_DIR.exists():
        return []
    case_ids = [p.parent.name for p in DOSSIER_DIR.glob("*/compliance_matrix.json")]
    return sorted(case_ids, key=lambda c: (not _role_options(c), c))


@st.cache_data(ttl=MATRIX_CACHE_TTL_SECONDS)
def _role_options(case_id: str) -> list[tuple[str, int]]:
    """(role_id, entry_count) for every role that produced matrix entries,
    sorted by role_id. Roles come from the matrix rather than facts.jsonl so
    the dropdown only ever offers roles an analysis has actually covered."""
    matrix = _load_compliance_matrix(case_id)
    counts: dict[str, int] = {}
    for entry in matrix.get("entries") or []:
        role = entry.get("actor_role")
        if role:
            counts[role] = counts.get(role, 0) + 1
    return sorted(counts.items())


# ========== compliance panel: cost + rendering helpers ==========

def _fmt_usd(amount: float) -> str:
    """French decimal comma, explicit USD unit."""
    return f"{amount:.2f} $US".replace(".", ",")


def _result_cost_usd(result: dict) -> float:
    """Total spend for one result, single-model or comparative.

    Reads the usage dicts the backend attaches rather than re-deriving from
    the cost constants, so the figure shown is what was actually billed.
    """
    if "model" in result:
        return (result["model"].get("usage") or {}).get("cost_usd") or 0.0
    models = result.get("models") or {}
    total = 0.0
    for side in models.values():
        total += (side.get("usage") or {}).get("cost_usd") or 0.0
    total += (result.get("divergence_usage") or {}).get("cost_usd") or 0.0
    return total


def _render_cost_footer(result: dict) -> None:
    if result.get("cache_hit"):
        st.caption("Analyse mise en cache — coût 0,00 $US")
    else:
        st.caption(f"Nouvelle analyse — coût {_fmt_usd(_result_cost_usd(result))}")


def _render_entry_card(entry: dict) -> None:
    """One obligation determination: badge + summary, details behind an
    expander. persons_named renders only when non-empty — it is [] on every
    entry today (Fact.mentioned_person_ids was never backfilled, ADR #53),
    and an unconditional section would be permanently blank."""
    status = entry.get("status", "")
    badge = STATUS_BADGES.get(status, "⚪")
    label = STATUS_LABELS.get(status, status)
    chunk_id = entry.get("statute_chunk_id", "")

    with st.container(border=True):
        st.markdown(f"{badge} **{label}** · `{chunk_id}`")
        st.markdown(entry.get("obligation_summary", ""))

        with st.expander("Justification"):
            st.markdown(entry.get("rationale", "") or "_Aucune justification fournie._")

            excerpt = entry.get("statute_excerpt")
            if excerpt:
                st.caption(f"**Extrait de l'article :** {excerpt}")

            fact_ids = entry.get("evidence_fact_ids") or []
            if fact_ids:
                st.caption("**Faits invoqués :** " + ", ".join(f"`{f}`" for f in fact_ids))

            persons = entry.get("persons_named") or []
            if persons:
                st.markdown("**Personnes impliquées**")
                for person in persons:
                    name = person.get("canonical_name") or person.get("person_id", "")
                    st.markdown(f"- {name}")


def _render_single_result(result: dict) -> None:
    model_id = result["model"]["model_id"]
    entries = result["model"].get("entries") or []

    st.markdown(f"#### {COMPLIANCE_MODEL_LABELS.get(model_id, model_id)}")
    st.caption(f"{result.get('role_label', result['role_id'])} · {len(entries)} obligation(s)")

    if not entries:
        st.info("Aucune obligation identifiée pour ce rôle.")
    for idx, entry in enumerate(entries):
        _render_entry_card(entry)

    _render_cost_footer(result)


def _render_divergence_table(divergent: list[dict]) -> None:
    st.markdown("#### Divergences détaillées")
    if not divergent:
        st.success("Aucune divergence : les deux modèles concordent sur toutes les obligations partagées.")
        return
    rows = [
        {
            "Obligation": d.get("obligation_summary", ""),
            "Verdict Opus": STATUS_LABELS.get(d.get("opus_verdict", ""), d.get("opus_verdict", "")),
            "Verdict Kimi": STATUS_LABELS.get(d.get("kimi_verdict", ""), d.get("kimi_verdict", "")),
            "Crux": d.get("crux", ""),
            "Modèle plus fort": d.get("stronger_side", ""),
            "Raison": d.get("why", ""),
        }
        for d in divergent
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_comparative_result(result: dict) -> None:
    """Side-by-side verdicts + divergence meta-analysis.

    Rows align by entry_id via the backend's coverage_diff (ADR #56), not by
    obligation-summary similarity: entry_id is a hash of statute_chunk_id +
    actor_role, so both models land on the same id for the same obligation,
    and it is the same pairing the Haiku meta-analysis above used. Matching
    on string similarity here could pair rows the callout says are unrelated.
    """
    divergence = result.get("divergence_analysis") or {}
    meta_summary = divergence.get("meta_summary")
    if meta_summary:
        st.info(f"**Méta-analyse ({DIVERGENCE_MODEL_LABEL})** — {meta_summary}")

    models = result.get("models") or {}
    opus_by_id = {e["entry_id"]: e for e in (models.get("opus", {}).get("entries") or [])}
    kimi_by_id = {e["entry_id"]: e for e in (models.get("kimi", {}).get("entries") or [])}
    coverage = result.get("coverage_diff") or {}

    col_opus, col_kimi = st.columns([1, 1])
    with col_opus:
        st.markdown("#### 🧠 Opus 4.7 (max)")
    with col_kimi:
        st.markdown("#### 🔬 Kimi K3 (max)")

    # Shared obligations first, one per row, so the two columns stay aligned.
    for entry_id in coverage.get("shared_entry_ids") or []:
        col_opus, col_kimi = st.columns([1, 1])
        with col_opus:
            _render_entry_card(opus_by_id[entry_id])
        with col_kimi:
            _render_entry_card(kimi_by_id[entry_id])

    # Then obligations only one model surfaced, at the foot of its own column.
    opus_only = coverage.get("opus_only_entry_ids") or []
    kimi_only = coverage.get("kimi_only_entry_ids") or []
    if opus_only or kimi_only:
        col_opus, col_kimi = st.columns([1, 1])
        with col_opus:
            if opus_only:
                st.caption("Relevé uniquement par Opus")
                for entry_id in opus_only:
                    _render_entry_card(opus_by_id[entry_id])
        with col_kimi:
            if kimi_only:
                st.caption("Relevé uniquement par Kimi")
                for entry_id in kimi_only:
                    _render_entry_card(kimi_by_id[entry_id])

    st.divider()
    _render_divergence_table(divergence.get("divergent_obligations") or [])
    _render_cost_footer(result)


# ========== compliance panel: orchestration ==========

def _run_and_cache(cache_key: tuple, spinner_msg: str, runner) -> None:
    """Execute one analysis, storing the result under cache_key.

    Errors are surfaced to the user and the traceback printed to the server
    log, matching _render_turn's soft-fail idiom — a failed analysis must not
    take down the panel.
    """
    with st.spinner(spinner_msg):
        try:
            st.session_state.compare_result_cache[cache_key] = runner()
        except Exception:  # noqa: BLE001 — user-facing fallback
            traceback.print_exc()
            st.error(
                "L'analyse a échoué. Vérifiez la connexion au fournisseur de "
                "modèles et réessayez. Détails techniques dans les logs du serveur."
            )


def _render_analysis_panel() -> None:
    st.subheader("Analyse de conformité par rôle")
    st.caption(
        "Évalue les obligations légales d'un rôle d'acteur du dossier à partir "
        "des faits extraits et des articles de loi correspondants."
    )

    cases = _list_cases_with_matrix()
    if not cases:
        st.info(
            "Aucun dossier analysé pour l'instant. Générez d'abord une matrice "
            "de conformité : `python -m rag.compliance --case-id <dossier>`."
        )
        return

    col_case, col_role = st.columns([1, 2])
    with col_case:
        case_id = st.selectbox("Dossier", cases, key="compare_case_id")

    roles = _role_options(case_id)
    if not roles:
        with col_role:
            st.selectbox("Rôle", ["—"], disabled=True, key="compare_role_empty")
        st.info("Aucun rôle analysable pour ce dossier.")
        return

    counts = dict(roles)
    with col_role:
        role_id = st.selectbox(
            "Rôle",
            [r for r, _ in roles],
            format_func=lambda r: f"{r} — {counts[r]} obligation(s)",
            key="compare_selected_role",
        )

    st.divider()

    # --- Primary path: one model, one charge (ADR #57) ---
    model_labels = list(COMPLIANCE_MODEL_LABELS.values())
    label_to_id = {v: k for k, v in COMPLIANCE_MODEL_LABELS.items()}
    chosen_label = st.segmented_control(
        "Modèle",
        options=model_labels,
        default=COMPLIANCE_MODEL_LABELS[DEFAULT_COMPLIANCE_MODEL],
        key="compare_model_selector",
    )
    model_id = label_to_id.get(chosen_label or "", DEFAULT_COMPLIANCE_MODEL)
    st.session_state.compare_model = model_id

    col_run, col_cost = st.columns([1, 3], vertical_alignment="center")
    with col_run:
        launch = st.button("Lancer l'analyse", type="primary", width="stretch")
    with col_cost:
        preview = COST_PREVIEW_USD.get(model_id, COST_PREVIEW_FALLBACK_USD)
        st.caption(
            f"≈ {_fmt_usd(preview)} — seul le modèle sélectionné est interrogé. "
            "Le coût varie avec le nombre de faits du rôle."
        )

    single_key = (case_id, role_id, "single", model_id)
    if launch:
        compliance = get_compliance()
        _run_and_cache(
            single_key,
            f"Analyse en cours ({COMPLIANCE_MODEL_LABELS[model_id]})…",
            lambda: compliance.run_compliance_for_role(
                case_id, role_id, compliance_model_id=model_id
            ),
        )

    single_result = st.session_state.compare_result_cache.get(single_key)
    if single_result:
        st.divider()
        _render_single_result(single_result)

    # --- Opt-in path: both models + meta-analysis (ADR #56) ---
    st.divider()
    with st.expander(f"Comparer les deux modèles — ≈ {_fmt_usd(COST_COMPARE_USD)}"):
        st.caption(
            "Interroge Opus 4.7 **et** Kimi K3 sur les mêmes faits, puis fait "
            "analyser leurs accords et divergences par Haiku 4.5. Utile quand un "
            "verdict doit être défendu : deux modèles d'accord sont plus solides, "
            "et un désaccord signale le point juridique à creuser."
        )
        compare_clicked = st.button(
            "Lancer l'analyse comparative", key="compare_launch", width="stretch"
        )

        compare_key = (case_id, role_id, "compare", None)
        if compare_clicked:
            compliance = get_compliance()
            _run_and_cache(
                compare_key,
                "Analyse en cours (Opus + Kimi + méta-analyse Haiku)…",
                lambda: compliance.compare_compliance_for_role(case_id, role_id),
            )

        compare_result = st.session_state.compare_result_cache.get(compare_key)
        if compare_result:
            st.divider()
            _render_comparative_result(compare_result)


# ========== session state initialisation ==========

def _init_session_state() -> None:
    """Session-state contract.

    conversations: dict[str, dict] — all threads keyed by id
    active_conversation_id: str | None — currently displayed thread
    lang: 'fr' | 'en'
    models_warm: bool
    answer_model: str — ANSWER_MODELS catalog key (ADR #46)
    compare_model: str — selected compliance model id (ADR #57)
    compare_result_cache: dict — analysis results keyed by
        (case_id, role_id, mode, model_id), mode in {'single', 'compare'}.
        The model id is part of the key on purpose: without it, toggling
        Opus -> Kimi would redisplay the Opus result under Kimi's name.
        compare_case_id / compare_selected_role are owned by their widgets.
    """
    if "conversations" not in st.session_state:
        # Idempotent — no-op after first session if Postgres already has
        # the schema. Silently returns if kill switch tripped.
        db.init_schema()
        st.session_state.conversations = _load_all_conversations()
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None
    if "lang" not in st.session_state:
        st.session_state.lang = "fr"
    if "models_warm" not in st.session_state:
        st.session_state.models_warm = False
    st.session_state.setdefault("answer_model", "gpt-4o-mini")
    st.session_state.setdefault("compare_model", DEFAULT_COMPLIANCE_MODEL)
    st.session_state.setdefault("compare_result_cache", {})


# ========== env validation (startup fail-loud) ==========

def _validate_env() -> None:
    load_dotenv(ENV_PATH)
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY manquant dans .env. "
            "L'application ne peut démarrer sans cette clé."
        )
        st.stop()


# ========== top-right language toggle ==========

def _render_lang_toggle() -> None:
    """Prominent language picker in the top-right using segmented_control.

    Wider column allocation (25%) + iOS-style pills = no vertical wrap.
    """
    _spacer, col = st.columns([3, 1])
    with col:
        current = LANG_OPTIONS[0] if st.session_state.lang == "fr" else LANG_OPTIONS[1]
        choice = st.segmented_control(
            "Lang",
            options=LANG_OPTIONS,
            default=current,
            selection_mode="single",
            label_visibility="collapsed",
            key="lang_seg",
        )
    if choice is None:
        return
    new_lang = "en" if "EN" in choice else "fr"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()


# ========== sidebar model toggle (ADR #46) ==========

def _render_model_toggle() -> None:
    """Answer-model selector. Session state persists the ANSWER_MODELS key."""
    current_label = MODEL_LABELS[st.session_state.answer_model]
    choice = st.segmented_control(
        "Modèle",
        options=list(MODEL_LABELS.values()),
        default=current_label,
        key="model_selector",
    )
    if choice is None:
        return
    label_to_key = {v: k for k, v in MODEL_LABELS.items()}
    new_key = label_to_key[choice]
    if new_key != st.session_state.answer_model:
        st.session_state.answer_model = new_key
        st.rerun()
    st.caption(f"Coût estimé : {COST_HINTS[st.session_state.answer_model]}")


# ========== sidebar (new-chat + conversations + docs placeholder + about) ==========

def _render_sidebar(labels: dict) -> None:
    with st.sidebar:
        # --- New conversation button (top priority) ---
        if st.button(
            labels["conv_new"],
            key="new_conv",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.active_conversation_id = None
            st.rerun()

        st.divider()

        # --- Answer-model toggle (ADR #46) ---
        _render_model_toggle()

        st.divider()

        # --- Conversations list ---
        st.markdown("### " + labels["conv_heading"])
        convs = st.session_state.conversations
        if not convs:
            st.caption(labels["conv_empty"])
        else:
            sorted_convs = sorted(
                convs.values(),
                key=lambda c: c.get("updated_at", c.get("created_at", "")),
                reverse=True,
            )
            active_id = st.session_state.active_conversation_id

            title_counts: dict[str, int] = {}
            for conv in sorted_convs:
                title_counts[conv["title"]] = title_counts.get(conv["title"], 0) + 1

            conv_labels: list[str] = []
            conv_ids: list[str] = []
            for idx, conv in enumerate(sorted_convs, start=1):
                conv_ids.append(conv["id"])
                title = conv["title"]
                if title_counts[title] > 1:
                    conv_labels.append(f"{title} ({idx})")
                else:
                    conv_labels.append(title)

            default_index = next(
                (i for i, conv_id in enumerate(conv_ids) if conv_id == active_id),
                0,
            )
            selected_label = st.radio(
                "",
                conv_labels,
                index=default_index,
                key="conv_selector",
                label_visibility="collapsed",
            )
            selected_id = conv_ids[conv_labels.index(selected_label)]
            if selected_id != active_id:
                st.session_state.active_conversation_id = selected_id
                st.rerun()

        st.divider()

        # --- About ---
        st.markdown(labels["about_heading"])
        st.markdown(labels["about_body"])

        st.divider()

        # --- Personal documents placeholder (future scope) ---
        st.markdown("### " + labels["docs_heading"])
        st.file_uploader(
            labels["docs_upload"],
            disabled=True,
            key="docs_placeholder",
            label_visibility="collapsed",
        )
        st.caption(labels["docs_future_note"])


# ========== turn rendering (user + assistant bubbles) ==========

def _render_citations(result: dict, labels: dict) -> None:
    citations = result.get("citations") or []
    if not citations:
        return
    st.markdown("**" + labels["citations"] + "**")
    for cite in citations:
        if isinstance(cite, dict):
            num = cite.get("num")
            chunk_id = cite.get("chunk_id", "")
            url = cite.get("url", "")
            if num:
                text = f"Art. {num}"
                rendered = f"[{text}]({url})" if url else text
            elif chunk_id:
                rendered = (
                    f"[`{chunk_id}`]({url})" if url else f"`{chunk_id}`"
                )
            else:
                rendered = f"[?]({url})" if url else "?"
            st.markdown(f"- {rendered}")
        else:
            st.markdown(f"- {cite}")


def _render_answer_body(turn: dict, conv: dict, labels: dict) -> None:
    """Render answer in current language. Lazy-translate + cache on EN."""
    result = turn["result"]
    french_answer = result.get("answer", "")

    if st.session_state.lang == "fr":
        st.markdown(french_answer)
        return

    if turn.get("answer_en") is None:
        with st.spinner(labels["translating"]):
            translated = _translate_to_english(french_answer)
        if translated is None:
            st.warning(labels["translation_failed"])
            st.markdown(french_answer)
            return
        turn["answer_en"] = translated
        _save_conversation(conv)

    st.caption(labels["translation_notice"])
    st.markdown(turn["answer_en"])


def _render_turn_feedback(
    idx: int, turn: dict, conv: dict, labels: dict
) -> None:
    if turn.get("feedback") is not None:
        icon = "👍" if turn["feedback"] == RATING_UP else "👎"
        st.caption(f"{icon} {labels['thanks']}")
        return

    st.markdown(f"**{labels['helpful']}**")
    col_yes, col_no, _spacer = st.columns([1, 1, 5])
    with col_yes:
        clicked_yes = st.button(
            labels["yes"],
            key=f"fb_yes_{conv['id']}_{idx}",
            use_container_width=True,
        )
    with col_no:
        clicked_no = st.button(
            labels["no"],
            key=f"fb_no_{conv['id']}_{idx}",
            use_container_width=True,
        )

    if not (clicked_yes or clicked_no):
        return

    rating = RATING_UP if clicked_yes else RATING_DOWN
    turn["feedback"] = rating

    # Feedback record: full 10-key Day 6 dict. db.append_feedback stores
    _append_feedback(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": conv["id"],
            "turn_id": turn["turn_id"],
            "question": turn["question"],  
            "answer": turn["result"].get("answer", ""),
            "rating": rating,
            "comment": "",
            "model_used": turn["result"].get("model_used", ""),
            "cost_usd": turn["result"].get("cost_usd", ""),
            "elapsed_seconds": turn["result"].get("elapsed_seconds", ""),
        }
    )
    _save_conversation(conv)
    st.rerun()


def _render_turn_details(turn: dict, conv: dict, labels: dict) -> None:
    with st.expander(labels["details"]):
        st.markdown(
            f"**{labels['conv_id_label']}:** `{conv['id']}`"
        )
        st.markdown(
            f"**{labels['turn_id_label']}:** `{turn['turn_id']}`"
        )
        result = turn["result"]
        fields = [
            ("rewritten_query", labels["rewritten"]),
            ("chunks_retrieved", labels["retrieved"]),
            ("chunks_reranked", labels["reranked"]),
            ("model_used", labels["model"]),
            ("answer_model_key", labels["model_key"]),
            ("cost_usd", labels["cost"]),
            ("elapsed_seconds", labels["elapsed"]),
        ]
        for key, label in fields:
            value = result.get(key, "—")
            if key == "cost_usd" and isinstance(value, (int, float)):
                value = f"${value:.5f}"
            if key == "elapsed_seconds" and isinstance(value, (int, float)):
                value = f"{value:.2f}s"
            st.markdown(f"**{label}:** {value}")


def _render_turn(
    idx: int, turn: dict, conv: dict, labels: dict
) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(turn["question"])

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # Lazy flow execution for pending turns.
        if turn["result"] is None:
            spinner_msg = (
                labels["loading"] if st.session_state.models_warm
                else labels["loading_cold"]
            )
            with st.spinner(spinner_msg):
                try:
                    turn["result"] = get_flow().run(
                        turn["question"],
                        answer_model=st.session_state.answer_model,
                    )
                    st.session_state.models_warm = True
                except Exception:  # noqa: BLE001 — user-facing fallback
                    st.error(labels["error"])
                    traceback.print_exc()
                    turn["result"] = {"_error": True}
            _save_conversation(conv)

        result = turn["result"]
        if result.get("_error"):
            return

        _render_answer_body(turn, conv, labels)
        _render_citations(result, labels)
        _render_turn_feedback(idx, turn, conv, labels)
        _render_turn_details(turn, conv, labels)


# ========== main() orchestration ==========

def _handle_chat_input(user_input: str) -> None:
    """Route a new question: create conversation if none active, append turn."""
    active_id = st.session_state.active_conversation_id
    conv = st.session_state.conversations.get(active_id) if active_id else None

    if conv is None:
        conv = _new_conversation(user_input)
        st.session_state.conversations[conv["id"]] = conv
        st.session_state.active_conversation_id = conv["id"]

    conv["turns"].append(_new_turn(user_input))
    _save_conversation(conv)


def main() -> None:
    st.set_page_config(
        page_title="lex-clair",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _validate_env()
    _init_session_state()

    labels = LABELS[st.session_state.lang]

    # DB health banner — persistent warning while the kill switch is tripped.
    # Kill switch is one-way per process (see ADR #34); recovery requires
    # a Streamlit restart, not just Postgres coming back.
    if not db.is_healthy():
        st.warning(labels["db_offline_banner"])

    _render_sidebar(labels)

    _render_lang_toggle()
    st.title(labels["title"])
    st.caption(labels["subtitle"])

    # on_change="rerun" makes .open readable server-side. That buys two
    # things: the panel's file I/O is skipped during chat turns, and the
    # chat input can stay in the page body (below), where Streamlit pins it
    # to the viewport bottom — nested inside a tab it would render inline.
    tab_chat, tab_analysis = st.tabs(
        [TAB_CHAT, TAB_ANALYSIS], on_change="rerun", key="main_tab",
    )

    with tab_chat:
        _render_chat(labels)

    with tab_analysis:
        if tab_analysis.open:
            _render_analysis_panel()

    if tab_chat.open:
        user_input = st.chat_input(labels["ask_placeholder"])
        if user_input and user_input.strip():
            _handle_chat_input(user_input.strip())
            st.rerun()


def _render_chat(labels: dict) -> None:
    """The conversation view: welcome state, or every turn of the active thread.

    The chat input itself lives in main()'s page body, not here — see the
    tab wiring above.
    """
    active_id = st.session_state.active_conversation_id
    conv = st.session_state.conversations.get(active_id) if active_id else None

    if conv is None or not conv["turns"]:
        st.info(labels["welcome"])
        return

    for idx, turn in enumerate(conv["turns"]):
        _render_turn(idx, turn, conv, labels)


if __name__ == "__main__":
    main()