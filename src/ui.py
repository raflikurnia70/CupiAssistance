"""Custom styling and small reusable UI components for the Streamlit app.

Keeps app.py focused on page logic; this module only renders markup.
"""

from __future__ import annotations

import html

import streamlit as st

PAGES = ["Overview", "Data Analysis", "AI Chat", "Knowledge Base", "Settings"]

PAGE_ICONS = {
    "Overview": "🏠",
    "Data Analysis": "📊",
    "AI Chat": "💬",
    "Knowledge Base": "📚",
    "Settings": "⚙️",
}


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy-900: #0B1220;
            --navy-800: #131C2E;
            --bg: #F5F7FB;
            --card-bg: #FFFFFF;
            --border: #E4E9F2;
            --text-primary: #101828;
            --text-secondary: #667085;
            --accent: #2E5CE6;
            --accent-light: #EAF1FF;
            --success: #16A34A;
            --success-bg: #EAF7EE;
            --warning: #D97706;
            --warning-bg: #FEF6E7;
            --error: #DC2626;
            --error-bg: #FDECEC;
        }

        html, body, [class*="css"] { font-family: "Segoe UI", "Inter", -apple-system, sans-serif; }
        .stApp { background: var(--bg); }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

        section[data-testid="stSidebar"] {
            background: var(--navy-900);
            border-right: 1px solid #060A12;
        }
        section[data-testid="stSidebar"] * { color: #E7ECF6 !important; }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
            margin-bottom: 2px;
        }
        section[data-testid="stSidebar"] .stButton button {
            background: transparent;
            border: 1px solid #2A3652;
            border-radius: 8px;
            color: #E7ECF6 !important;
            width: 100%;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: var(--accent);
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] hr { border-color: #22304A; }

        .ds-brand-title { font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0; letter-spacing: -0.01em; }
        .ds-brand-sub { font-size: 0.78rem; color: #9AA7C2; margin-top: 2px; margin-bottom: 1.1rem; }

        .ds-status-row { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; margin-bottom: 6px; color: #C7D0E4; }
        .ds-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .ds-dot-on { background: #34D399; box-shadow: 0 0 0 3px rgba(52,211,153,0.18); }
        .ds-dot-off { background: #F87171; box-shadow: 0 0 0 3px rgba(248,113,113,0.18); }
        .ds-dot-warn { background: #FBBF24; box-shadow: 0 0 0 3px rgba(251,191,36,0.18); }

        .ds-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }

        .ds-hero {
            background: linear-gradient(135deg, var(--navy-900) 0%, #1B2A4A 100%);
            border-radius: 20px;
            padding: 2.6rem 2.6rem;
            color: #FFFFFF;
            margin-bottom: 1.6rem;
        }
        .ds-hero-eyebrow { color: #9DB4F5; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.6rem; }
        .ds-hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 0.6rem 0; letter-spacing: -0.02em; }
        .ds-hero p { color: #C7D0E4; font-size: 1rem; max-width: 620px; margin: 0; line-height: 1.55; }

        .ds-page-header { margin-bottom: 1.4rem; }
        .ds-page-header h2 { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 0 0 0.2rem 0; letter-spacing: -0.01em; }
        .ds-page-header p { color: var(--text-secondary); font-size: 0.92rem; margin: 0; }

        .ds-feature-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.4rem;
            height: 100%;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .ds-feature-icon {
            width: 40px; height: 40px; border-radius: 10px;
            background: var(--accent-light); color: var(--accent);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.15rem; margin-bottom: 0.8rem;
        }
        .ds-feature-card h4 { margin: 0 0 0.35rem 0; font-size: 0.95rem; color: var(--text-primary); letter-spacing: 0.02em; text-transform: uppercase; }
        .ds-feature-card p { margin: 0; font-size: 0.87rem; color: var(--text-secondary); line-height: 1.5; }

        .ds-metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .ds-metric-label { font-size: 0.78rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem; }
        .ds-metric-value { font-size: 1.65rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; }
        .ds-metric-value.ok { color: var(--success); }
        .ds-metric-value.warn { color: var(--warning); }
        .ds-metric-value.bad { color: var(--error); }

        .ds-empty-state {
            text-align: center;
            padding: 3.2rem 1.5rem;
            background: var(--card-bg);
            border: 1px dashed var(--border);
            border-radius: 16px;
        }
        .ds-empty-state .ds-empty-icon { font-size: 2.1rem; margin-bottom: 0.7rem; }
        .ds-empty-state h3 { margin: 0 0 0.35rem 0; color: var(--text-primary); font-size: 1.15rem; }
        .ds-empty-state p { margin: 0; color: var(--text-secondary); font-size: 0.92rem; }

        .ds-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; }
        .ds-badge-ok { background: var(--success-bg); color: var(--success); }
        .ds-badge-warn { background: var(--warning-bg); color: var(--warning); }
        .ds-badge-bad { background: var(--error-bg); color: var(--error); }

        .ds-context-card {
            background: var(--accent-light);
            border: 1px solid #CFE0FF;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.9rem;
            font-size: 0.87rem;
            color: var(--text-primary);
        }
        .ds-context-card .ds-context-title { font-weight: 700; font-size: 0.78rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }

        .stChatMessage { border-radius: 14px; }
        .stButton button {
            border-radius: 10px;
            border: 1px solid var(--border);
            font-weight: 500;
        }
        .stButton button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        [data-testid="stMetricValue"] { color: var(--text-primary); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status(label: str, state: str, detail: str = "") -> None:
    dot_class = {"on": "ds-dot-on", "off": "ds-dot-off", "warn": "ds-dot-warn"}.get(state, "ds-dot-off")
    text = html.escape(detail) if detail else html.escape(label)
    st.markdown(
        f'<div class="ds-status-row"><span class="ds-dot {dot_class}"></span>'
        f'<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="ds-page-header"><h2>{html.escape(title)}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        f'<div class="ds-metric-card"><div class="ds-metric-label">{html.escape(label)}</div>'
        f'<div class="ds-metric-value{tone_class}">{html.escape(str(value))}</div></div>'
    )


def feature_card(icon: str, title: str, description: str) -> str:
    return (
        f'<div class="ds-feature-card"><div class="ds-feature-icon">{icon}</div>'
        f"<h4>{html.escape(title)}</h4><p>{html.escape(description)}</p></div>"
    )


def empty_state(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ds-empty-state"><div class="ds-empty-icon">{icon}</div>'
        f"<h3>{html.escape(title)}</h3><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "ok") -> str:
    return f'<span class="ds-badge ds-badge-{tone}">{html.escape(text)}</span>'
