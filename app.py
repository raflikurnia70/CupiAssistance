"""DataSense AI - AI-Powered Data Science Assistant.

Streamlit entry point. Wires together dataset profiling (Pandas), a RAG
knowledge base (LangChain + FAISS), and an LLM chat assistant (Llama via
Groq, or Google Gemini - see src/llm.py) behind a single-page navigation
shell.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import data_analysis, llm, rag, ui
from src.prompts import (
    KB_CONTEXT_TEMPLATE,
    NO_KB_CONTEXT,
    QUICK_PROMPTS,
    SYSTEM_PROMPT,
    WELCOME_SUBTITLE,
    WELCOME_TITLE,
    build_user_prompt,
)

load_dotenv()

st.set_page_config(
    page_title="DataSense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_global_css()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "page": "Overview",
    "df": None,
    "filename": None,
    "profile": None,
    "messages": [],
    "pending_prompt": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# RAG vector store (cached across reruns within the same server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan knowledge base (RAG)...")
def _load_vectorstore():
    return rag.load_or_build_vectorstore()


vectorstore, vs_error = None, None
if rag.is_knowledge_base_available():
    try:
        vectorstore = _load_vectorstore()
    except Exception as exc:  # RAGError or dependency failure
        vs_error = str(exc)
else:
    vs_error = "File data_science_knowledge.pdf tidak ditemukan pada folder data/."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="ds-brand-title">📊 DataSense AI</div>'
            '<div class="ds-brand-sub">AI-Powered Data Science Assistant</div>',
            unsafe_allow_html=True,
        )
        st.radio(
            "Navigation",
            ui.PAGES,
            key="page",
            label_visibility="collapsed",
            format_func=lambda p: f"{ui.PAGE_ICONS[p]}  {p}",
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:0.72rem;color:#8592AD;text-transform:uppercase;'
            'letter-spacing:0.05em;margin-bottom:4px;">AI Status</div>',
            unsafe_allow_html=True,
        )
        ai_ok = llm.is_configured()
        ui.sidebar_status("AI Status", "on" if ai_ok else "off", "Online" if ai_ok else "Offline")

        st.markdown(
            '<div style="font-size:0.72rem;color:#8592AD;text-transform:uppercase;'
            'letter-spacing:0.05em;margin:10px 0 4px 0;">RAG Status</div>',
            unsafe_allow_html=True,
        )
        rag_state = "on" if vectorstore is not None else "off"
        rag_text = "Ready" if vectorstore is not None else "Not Ready"
        ui.sidebar_status("RAG Status", rag_state, rag_text)

        st.markdown(
            '<div style="font-size:0.72rem;color:#8592AD;text-transform:uppercase;'
            'letter-spacing:0.05em;margin:10px 0 4px 0;">Model</div>'
            f'<div style="font-size:0.86rem;color:#E7ECF6;margin-bottom:0.8rem;">{llm.get_provider_display_name()}</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
def render_overview() -> None:
    st.markdown(
        '<div class="ds-hero">'
        '<div class="ds-hero-eyebrow">DataSense AI</div>'
        "<h1>Turn your data questions into actionable insights.</h1>"
        "<p>Analyze your dataset, explore data quality, and ask an AI assistant "
        "for data science guidance.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Upload Dataset", type="primary"):
        st.session_state.page = "Data Analysis"
        st.rerun()

    st.write("")
    if st.session_state.df is None:
        ui.empty_state(
            "🗂️",
            "Your workspace is ready.",
            "Upload a CSV dataset to start exploring your data.",
        )
    else:
        profile = st.session_state.profile
        st.success(
            f"Dataset loaded: **{profile['filename']}** — "
            f"{profile['rows']:,} rows, {profile['columns']} columns."
        )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            ui.feature_card("📊", "Data Analysis", "Understand your dataset with automated profiling."),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            ui.feature_card("💬", "AI Assistant", "Ask questions using natural language."),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            ui.feature_card("📚", "Knowledge Base", "Get answers grounded in Data Science references."),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Data Analysis page
# ---------------------------------------------------------------------------
def render_data_analysis() -> None:
    ui.page_header("Dataset Overview", "Automated dataset profiling powered by Pandas.")

    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            df = data_analysis.load_csv(uploaded)
            profile = data_analysis.profile_dataset(df, uploaded.name)
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            st.session_state.profile = profile
        except data_analysis.DatasetError as exc:
            st.error(str(exc))

    if st.session_state.df is None:
        ui.empty_state(
            "📄",
            "No dataset yet",
            "Upload a CSV file to see automated profiling, data quality checks, and statistics.",
        )
        return

    profile = st.session_state.profile

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(ui.metric_card("Rows", f"{profile['rows']:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(ui.metric_card("Columns", profile["columns"]), unsafe_allow_html=True)
    with c3:
        tone = "bad" if profile["missing_pct"] > 10 else ("warn" if profile["missing_pct"] > 0 else "ok")
        st.markdown(ui.metric_card("Missing Values", f"{profile['missing_pct']}%", tone), unsafe_allow_html=True)
    with c4:
        tone = "warn" if profile["duplicate_rows"] > 0 else "ok"
        st.markdown(ui.metric_card("Duplicate Rows", profile["duplicate_rows"], tone), unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Data Preview")
    st.dataframe(st.session_state.df.head(50), use_container_width=True)

    st.markdown("#### Data Types")
    dtypes_df = pd.DataFrame(profile["column_details"]).rename(
        columns={
            "column": "Column",
            "dtype": "Data Type",
            "missing": "Missing",
            "missing_pct": "Missing %",
            "unique": "Unique",
        }
    )
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    st.markdown("#### Data Quality")
    colA, colB = st.columns(2)
    with colA:
        st.caption("Missing Values")
        st.progress(min(profile["missing_pct"] / 100, 1.0), text=f"{profile['missing_pct']}%")
    with colB:
        st.caption("Duplicate Rows")
        st.progress(min(profile["duplicate_pct"] / 100, 1.0), text=f"{profile['duplicate_pct']}%")

    if profile["outlier_columns"]:
        cols_str = ", ".join(o["column"] for o in profile["outlier_columns"])
        st.warning(f"Potential Issues: possible outliers detected in **{cols_str}**.")
    else:
        st.success("Potential Issues: no strong outlier indication detected.")

    st.markdown("#### Statistical Summary")
    st.dataframe(profile["describe"], use_container_width=True)

    missing_df = pd.DataFrame(profile["column_details"])
    missing_df = missing_df[missing_df["missing"] > 0]
    if not missing_df.empty:
        st.markdown("#### Missing Values by Column")
        try:
            import plotly.express as px

            fig = px.bar(
                missing_df,
                x="column",
                y="missing",
                labels={"column": "Column", "missing": "Missing Values"},
                color_discrete_sequence=["#2E5CE6"],
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.bar_chart(missing_df.set_index("column")["missing"])


# ---------------------------------------------------------------------------
# AI Chat page
# ---------------------------------------------------------------------------
def _build_history(limit: int = 6) -> list[dict]:
    return st.session_state.messages[-limit:]


def _generate_reply(user_text: str) -> str:
    if not user_text.strip():
        return "Silakan tuliskan pertanyaan terlebih dahulu."

    dataset_context = data_analysis.profile_to_context_text(st.session_state.profile)

    kb_context = NO_KB_CONTEXT
    if vectorstore is not None:
        try:
            kb_chunks = rag.retrieve_context(vectorstore, user_text, k=3)
            kb_context = KB_CONTEXT_TEMPLATE.format(kb_chunks=kb_chunks) if kb_chunks else NO_KB_CONTEXT
        except rag.RAGError:
            kb_context = NO_KB_CONTEXT

    user_prompt = build_user_prompt(user_text, dataset_context, kb_context)
    history = _build_history()[:-1] if st.session_state.messages else []

    try:
        return llm.generate_response(SYSTEM_PROMPT, user_prompt, history=history)
    except llm.LLMConfigError as exc:
        return f"⚠️ {exc}"
    except llm.LLMResponseError as exc:
        return f"⚠️ Maaf, terjadi kendala saat menghubungi model AI. Detail: {exc}"


def _process_user_message(text: str) -> None:
    text = text.strip()
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("DataSense AI sedang menganalisis..."):
            reply = _generate_reply(text)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


def render_ai_chat() -> None:
    ui.page_header("DataSense AI", "Your Data Science Copilot")

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        tone = "ok" if vectorstore is not None else "bad"
        st.markdown(ui.badge("● RAG Ready" if vectorstore is not None else "● RAG Not Ready", tone), unsafe_allow_html=True)
    with status_col2:
        tone = "ok" if llm.is_configured() else "bad"
        st.markdown(ui.badge("● Model Connected" if llm.is_configured() else "● Model Not Connected", tone), unsafe_allow_html=True)

    if st.session_state.df is not None:
        profile = st.session_state.profile
        st.markdown(
            f'<div class="ds-context-card"><div class="ds-context-title">Current Dataset</div>'
            f"{profile['filename']}<br/>{profile['rows']:,} rows • {profile['columns']} columns</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    pending = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if not st.session_state.messages and not pending:
        st.markdown(f"##### {WELCOME_TITLE}")
        st.caption(WELCOME_SUBTITLE)
        qc1, qc2 = st.columns(2)
        cols = [qc1, qc2, qc1, qc2]
        for i, prompt_text in enumerate(QUICK_PROMPTS):
            with cols[i]:
                if st.button(prompt_text, key=f"quick_{i}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()
        st.write("")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Tanyakan tentang dataset Anda atau konsep Data Science...")
    final_input = pending or user_input
    if final_input:
        _process_user_message(final_input)


# ---------------------------------------------------------------------------
# Knowledge Base page
# ---------------------------------------------------------------------------
def render_knowledge_base() -> None:
    ui.page_header("Knowledge Base", "Retrieval-Augmented Generation reference material.")

    tone = "ok" if vectorstore is not None else "bad"
    status_text = "Ready" if vectorstore is not None else "Not Ready"
    st.markdown(
        f'<div class="ds-card">'
        f'<p style="margin:0 0 6px 0;"><strong>Status:</strong> {ui.badge(status_text, tone)}</p>'
        f'<p style="margin:0 0 6px 0;"><strong>Document:</strong> data_science_knowledge.pdf '
        f'<span style="color:var(--text-secondary);">(Demonstration Knowledge Base)</span></p>'
        f'<p style="margin:0;color:var(--text-secondary);font-size:0.88rem;">'
        f"DataSense AI uses Retrieval-Augmented Generation (RAG) to retrieve relevant knowledge "
        f"before generating answers.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if vs_error:
        st.warning(f"Knowledge base belum siap: {vs_error}")

    st.write("")
    st.markdown("#### Topics")
    topics = [
        "Data Cleaning", "Exploratory Data Analysis", "Missing Values", "Outliers",
        "Feature Engineering", "Classification", "Regression", "Model Evaluation",
        "Precision and Recall", "Overfitting and Underfitting", "Train/Test Split",
        "Common Data Science Mistakes",
    ]
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            st.markdown(f'<div class="ds-card" style="margin-bottom:0.7rem;padding:0.8rem 1rem;">{topic}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------
def render_settings() -> None:
    ui.page_header("Settings", "AI configuration overview.")

    st.markdown("#### AI Configuration")
    rows = [
        ("Model", llm.get_provider_label()),
        ("Temperature", str(llm.DEFAULT_TEMPERATURE)),
        ("Max Output Tokens", str(llm.get_max_tokens())),
        ("Top P", str(llm.DEFAULT_TOP_P)),
        ("RAG", "Enabled" if vectorstore is not None else "Unavailable"),
        ("Vector Database", "FAISS"),
        ("Embedding Model", rag.EMBEDDING_MODEL),
        ("API Status", "Connected" if llm.is_configured() else "Not Connected"),
    ]
    table_rows = "".join(
        f'<tr><td style="padding:8px 12px;color:var(--text-secondary);">{label}</td>'
        f'<td style="padding:8px 12px;font-weight:600;color:var(--text-primary);">{value}</td></tr>'
        for label, value in rows
    )
    st.markdown(
        f'<div class="ds-card"><table style="width:100%;border-collapse:collapse;">{table_rows}</table></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    with st.expander("What do these parameters mean?"):
        st.markdown(
            "**Temperature** — Controls response creativity/randomness. Lower values produce more "
            "focused, deterministic answers.\n\n"
            "**Max Output Tokens** — Controls the maximum length of a single response.\n\n"
            "**Top P** — Controls token sampling: the model only considers the smallest set of tokens "
            "whose cumulative probability exceeds this value."
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
render_sidebar()

PAGE_RENDERERS = {
    "Overview": render_overview,
    "Data Analysis": render_data_analysis,
    "AI Chat": render_ai_chat,
    "Knowledge Base": render_knowledge_base,
    "Settings": render_settings,
}
PAGE_RENDERERS[st.session_state.page]()
