"""
Nepal Parichaya RAG — Streamlit UI

Talks to the Django REST API (`/api/query`, `/api/translate`, `/api/stats`)
instead of importing `rag/` directly, so the UI and backend can deploy
independently.

Config:
  RAG_API_URL  base URL of the API   (default: http://127.0.0.1:8765)
  RAG_API_KEY  optional X-API-Key sent on every API request

Run:
  streamlit run streamlit_app.py
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

import rag_api_client as api
from rag_api_client import APIError

# Default LLM choices shown in the sidebar dropdown — match the rag/ pricing table
# but expose as a static list so the UI has no rag/ import dependency.
MODEL_OPTIONS = [
    "gpt-5.5", "gpt-4o", "gpt-4.1", "gpt-4o-mini",
    "gpt-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo",
]

DEFAULT_SYSTEM_PROMPT_PLACEHOLDER = (
    "(Leave blank to use the server's default system prompt — recommended.)"
)

CORRECTIONS_QUEUE = Path("scripts/corrections_queue.json")


def _save_report(question: str, answer_excerpt: str,
                 wrong_text: str, correct_text: str, description: str):
    reports = []
    if CORRECTIONS_QUEUE.exists():
        with open(CORRECTIONS_QUEUE, encoding="utf-8") as f:
            reports = json.load(f)
    reports.append({
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "question":       question,
        "answer_excerpt": answer_excerpt[:300],
        "wrong_text":     wrong_text.strip(),
        "correct_text":   correct_text.strip(),
        "description":    description.strip(),
        "status":         "pending",
    })
    CORRECTIONS_QUEUE.parent.mkdir(exist_ok=True)
    with open(CORRECTIONS_QUEUE, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def _is_english(text: str) -> bool:
    return not bool(re.search(r'[ऀ-ॿ]', text))


# --- Page Config ---
st.set_page_config(
    page_title="Nepal Parichaya RAG",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Init ---
if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_sources"  not in st.session_state: st.session_state.last_sources  = []
if "total_tokens"  not in st.session_state: st.session_state.total_tokens  = 0
if "total_cost"    not in st.session_state: st.session_state.total_cost    = 0.0


# --- Cached API stats (refreshed every 5 min) ---
@st.cache_data(ttl=300)
def get_stats():
    try:
        return api.stats()
    except APIError:
        return None


# --- Reachability check ---
api_ok = api.health()
server_stats = get_stats() if api_ok else None


# ============================
# LEFT SIDEBAR
# ============================
with st.sidebar:
    st.title("Nepal Parichaya RAG")
    st.caption("Civics Knowledge Assistant")
    st.divider()

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_sources = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost   = 0.0
        st.rerun()

    if st.button("Refresh Server Stats", use_container_width=True):
        get_stats.clear()
        st.rerun()

    st.divider()
    st.subheader("Knowledge Base")
    if server_stats:
        st.write(f"Documents: **{server_stats['documents']:,}**")
        st.write(f"Embedding: `{server_stats['embedding_model']}`")
        st.write(f"LLM: `{server_stats['llm_model']}`")
    else:
        st.write("Documents: **Not loaded**")
    st.write("Source: **Nepal Parichaya**")

    st.divider()
    st.subheader("Status")
    if api_ok:
        st.success(f"API Connected\n\n`{os.environ.get('RAG_API_URL', 'http://127.0.0.1:8765')}`")
    else:
        st.error(
            "Cannot reach RAG API.\n\n"
            f"Set `RAG_API_URL` to point at a running server, "
            f"or start one locally: `python dev.py api`"
        )
    if server_stats and server_stats["documents"] > 0:
        st.success(f"Knowledge base loaded ({server_stats['documents']:,} chunks)")
    elif api_ok:
        st.warning("API reachable but knowledge base empty.")

    st.divider()
    if st.session_state.messages:
        st.download_button(
            "Export Chat",
            data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
            file_name="nepal_rag_chat.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================
# MAIN AREA: Chat + Settings
# ============================
chat_col, settings_col = st.columns([3, 1], gap="large")


# ============================
# RIGHT SETTINGS PANEL
# ============================
with settings_col:
    st.subheader("Settings")

    model_name = st.selectbox("Model", MODEL_OPTIONS, index=1)  # default gpt-4o
    custom_model = st.text_input(
        "Custom model ID (overrides above)",
        placeholder="e.g. gpt-5, gpt-4.5-turbo",
    )
    if custom_model.strip():
        model_name = custom_model.strip()

    temperature = st.slider("Temperature",       0.0,  2.0,  0.2,  0.1)
    top_k       = st.slider("Top-K (retrieval)", 3,    20,   8)
    max_tokens  = st.slider("Max Output Tokens", 256, 4096, 1024, 128)

    st.divider()
    st.subheader("System Prompt (optional override)")
    st.caption(
        "Leave blank to use the server's default system prompt — recommended "
        "for production behavior."
    )
    system_prompt = st.text_area(
        "System prompt override", value="", height=120,
        placeholder=DEFAULT_SYSTEM_PROMPT_PLACEHOLDER,
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Usage")
    c1, c2 = st.columns(2)
    with c1: st.metric("Tokens", f"{st.session_state.total_tokens:,}")
    with c2: st.metric("Cost",   f"${st.session_state.total_cost:.4f}")

    st.divider()
    st.subheader("Retrieved Sources")
    if st.session_state.last_sources:
        for src in st.session_state.last_sources:
            label = src.get("heading") or src["source"]
            with st.expander(f"{label} ({src['similarity']*100:.0f}%)"):
                st.caption(f"Source: {src['source']} | Match: {src.get('match', 'vector')}")
                text = src.get("text", "")
                st.markdown(text[:400] + ("..." if len(text) > 400 else ""))
    else:
        st.caption("Sources will appear here after a query")


# ============================
# CENTER CHAT AREA
# ============================
with chat_col:
    st.info(
        "**Tip:** Ask specific questions for better results.  \n"
        "Good: *\"How many days is Dashain celebrated?\"* | *\"What is the area of Nepal?\"*  \n"
        "Less effective: *\"Dashain\"*, *\"Nepal\"* — short queries give vague answers.  \n"
        "Ask in Nepali, Romanized Nepali, or English. Answers match your language."
    )

    # --- Chat history ---
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Translate button — only shown for English answers
                if msg.get("is_english"):
                    if msg.get("nepali_translation"):
                        with st.expander("नेपाली अनुवाद (Nepali Translation)"):
                            st.markdown(msg["nepali_translation"])
                    else:
                        if st.button("Translate to Nepali", key=f"translate_{i}"):
                            try:
                                with st.spinner("Translating..."):
                                    translation = api.translate(msg["content"])
                                st.session_state.messages[i]["nepali_translation"] = translation
                                st.rerun()
                            except APIError as e:
                                st.error(f"Translation failed: {e}")

                # Report wrong answer
                question_for_msg = ""
                for j in range(i - 1, -1, -1):
                    if st.session_state.messages[j]["role"] == "user":
                        question_for_msg = st.session_state.messages[j]["content"]
                        break

                with st.expander("Report an error in this answer"):
                    with st.form(key=f"report_{i}"):
                        st.caption("Help improve the knowledge base by flagging OCR or factual errors.")
                        wrong   = st.text_input("Wrong text (copy from answer above)",  key=f"wrong_{i}")
                        correct = st.text_input("Correct text",                          key=f"correct_{i}")
                        desc    = st.text_input("Optional: describe the error",          key=f"desc_{i}")
                        submitted = st.form_submit_button("Submit Report")
                        if submitted:
                            if wrong and correct:
                                _save_report(question_for_msg, msg["content"], wrong, correct, desc)
                                st.success("Report saved. Thank you!")
                            else:
                                st.warning("Please fill in both 'Wrong text' and 'Correct text'.")

    # --- Chat input ---
    if prompt := st.chat_input("Ask in Nepali, Romanized Nepali, or English..."):
        if not api_ok:
            st.error(
                "RAG API is not reachable. Start it with `python dev.py api` "
                "or set RAG_API_URL to point at a running server."
            )
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Querying RAG API..."):
                        result = api.query(
                            prompt,
                            model=model_name,
                            top_k=top_k,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt.strip(),
                        )
                except APIError as e:
                    st.error(f"API error: {e}")
                    st.stop()

                answer = result["answer"]
                sources = result.get("sources", [])
                usage = result.get("usage", {})

                st.markdown(answer)

                st.session_state.last_sources = sources
                st.session_state.total_tokens += usage.get("total_tokens", 0)
                st.session_state.total_cost   += usage.get("cost", 0.0)

            st.session_state.messages.append({
                "role":              "assistant",
                "content":           answer,
                "sources":           sources,
                "is_english":        _is_english(answer),
                "nepali_translation": None,
            })
            st.rerun()
