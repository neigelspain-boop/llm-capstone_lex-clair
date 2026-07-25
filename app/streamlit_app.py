"""Streamlit UI for lex-clair — French legal RAG assistant.

Two-level chat architecture: sidebar lists past CONVERSATIONS (threads),
each conversation contains multiple TURNS (Q&A pairs). Standard
ChatGPT/Claude-style pattern with a "New conversation" button, a
conversation list, and per-turn feedback.

Persistence: Postgres via monitoring.db (ADRs #32, #34). Conversations,
turns, and feedback all live in relational tables. Kill switch in
monitoring.db surfaces to the UI as a persistent warning banner when
the DB is unreachable — the app keeps rendering in-memory state.

Public surface: Streamlit runs this file directly. Plane II is
consumed via `rag.flow.run(query)` per ADR #10.
"""

from __future__ import annotations

import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

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


# ========== session state initialisation ==========

def _init_session_state() -> None:
    """Session-state contract.

    conversations: dict[str, dict] — all threads keyed by id
    active_conversation_id: str | None — currently displayed thread
    lang: 'fr' | 'en'
    models_warm: bool
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
                    turn["result"] = get_flow().run(turn["question"])
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

    user_input = st.chat_input(labels["ask_placeholder"])
    if user_input and user_input.strip():
        _handle_chat_input(user_input.strip())

    # Resolve the active conversation for rendering.
    active_id = st.session_state.active_conversation_id
    conv = st.session_state.conversations.get(active_id) if active_id else None

    if conv is None or not conv["turns"]:
        st.info(labels["welcome"])
        return

    for idx, turn in enumerate(conv["turns"]):
        _render_turn(idx, turn, conv, labels)


if __name__ == "__main__":
    main()