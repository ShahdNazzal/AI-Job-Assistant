"""Central CSS design system for the AI Job Assistant UI."""

from __future__ import annotations

import streamlit as st

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

:root {
    --aija-ink: #0f172a;
    --aija-body: #475569;
    --aija-muted: #64748b;
    --aija-primary: #4f46e5;
    --aija-primary-dark: #4338ca;
    --aija-border: #e2e8f0;
    --aija-border-soft: #eef2f7;
    --aija-surface: #ffffff;
    --aija-canvas: #f8fafc;
    --aija-chip-bg: #eef2ff;
    --aija-chip-text: #4338ca;
    --aija-chip-border: #e0e7ff;
    --aija-grad: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #c026d3 100%);
    --aija-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
    --aija-shadow-md: 0 1px 2px rgba(15, 23, 42, 0.04), 0 10px 28px -14px rgba(79, 70, 229, 0.22);
    --aija-shadow-lg: 0 2px 4px rgba(15, 23, 42, 0.05), 0 18px 40px -16px rgba(79, 70, 229, 0.32);
    --aija-radius: 16px;
}

.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select {
    font-family: 'Inter', 'IBM Plex Sans Arabic', -apple-system, 'Segoe UI',
                 system-ui, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3 { letter-spacing: -0.02em; color: var(--aija-ink); }
.stApp h2 { font-weight: 700; }
.stApp h3 { font-weight: 700; }
.stApp p, .stApp li { color: var(--aija-body); }
.stApp a { color: var(--aija-primary); }
.stApp strong { color: var(--aija-ink); }

.block-container { padding-top: 2.1rem; padding-bottom: 4rem; max-width: 1200px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stFooter"] { display: none; }

.stApp {
    background:
        radial-gradient(900px 320px at 85% -120px, rgba(124, 58, 237, 0.07), transparent 65%),
        radial-gradient(700px 300px at 8% -140px, rgba(79, 70, 229, 0.08), transparent 60%),
        var(--aija-surface);
}

[data-testid="stSidebar"] {
    background: var(--aija-canvas);
    border-right: 1px solid var(--aija-border-soft);
}
[data-testid="stSidebar"] * { color: inherit; }
[data-testid="stSidebar"] [data-testid="stPageLink"] { margin: 1px 0; }
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    display: block;
    padding: 0.42rem 0.7rem;
    border-radius: 10px;
    color: #334155;
    font-weight: 500;
    font-size: 0.93rem;
    text-decoration: none;
    transition: background 0.15s ease, color 0.15s ease;
}
[data-testid="stSidebar"] a:hover,
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background: var(--aija-chip-bg);
    color: var(--aija-primary-dark);
}

.stButton > button {
    border-radius: 11px;
    border: 1px solid var(--aija-border);
    background: var(--aija-surface);
    color: var(--aija-ink);
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0.45rem 1.1rem;
    box-shadow: var(--aija-shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease,
                border-color 0.15s ease, background 0.15s ease;
}
.stButton > button:hover {
    border-color: #c7d2fe;
    background: #f5f7ff;
    transform: translateY(-1px);
    box-shadow: var(--aija-shadow-md);
}
.stButton > button[kind="primary"],
.stApp [data-testid="stBaseButton-primary"] {
    background: var(--aija-grad);
    border: none;
    color: #ffffff;
}
.stButton > button[kind="primary"]:hover,
.stApp [data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.06);
    background: var(--aija-grad);
}
.stDownloadButton > button {
    border-radius: 11px;
    border: 1px solid #c7d2fe;
    color: var(--aija-primary-dark);
    font-weight: 600;
}

.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"],
.stApp [data-baseweb="select"] {
    border-radius: 10px;
}
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    border-radius: 10px;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #a5b4fc !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

.stApp code {
    background: var(--aija-chip-bg) !important;
    color: var(--aija-chip-text) !important;
    border: 1px solid var(--aija-chip-border);
    border-radius: 999px;
    padding: 0.18rem 0.7rem 0.22rem;
    font-family: inherit;
    font-size: 0.84rem;
    font-weight: 600;
    margin: 0.12rem 0.1rem;
    display: inline-block;
    white-space: nowrap;
}

[data-testid="stMetric"] {
    background: var(--aija-surface);
    border: 1px solid var(--aija-border);
    border-radius: var(--aija-radius);
    padding: 1rem 1.15rem;
    box-shadow: var(--aija-shadow-sm);
}
[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] { color: var(--aija-muted); font-weight: 500; }

[data-baseweb="tab-list"] { gap: 0.35rem; border-bottom-color: var(--aija-border-soft); }
[data-baseweb="tab"] {
    padding: 0.45rem 0.9rem;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    color: var(--aija-muted);
}
[data-baseweb="tab"][aria-selected="true"] { color: var(--aija-primary-dark); }
[data-baseweb="tab-highlight"] { background: var(--aija-grad); height: 3px; border-radius: 3px; }
[data-baseweb="tab-border"] { display: none; }

[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--aija-border-soft);
    box-shadow: var(--aija-shadow-sm);
}
[data-testid="stExpander"] details {
    border-radius: 14px;
    border: 1px solid var(--aija-border);
    background: var(--aija-surface);
    overflow: hidden;
}
[data-testid="stExpander"] summary:hover { color: var(--aija-primary-dark); }
hr { border-color: var(--aija-border-soft); }

.stProgress > div > div { background: var(--aija-grad); }
.stApp [role="slider"] { background-color: var(--aija-primary) !important; }

.stApp ::-webkit-scrollbar { width: 9px; height: 9px; }
.stApp ::-webkit-scrollbar-thumb {
    background: #cbd5e1; border-radius: 999px; border: 2px solid transparent;
    background-clip: content-box;
}
.stApp ::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }
.stApp ::-webkit-scrollbar-track { background: transparent; }

@keyframes aijaFadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.aija-hero, .aija-cards, .aija-card { animation: aijaFadeUp 0.45s ease both; }
.aija-cards .aija-card:nth-child(2) { animation-delay: 0.07s; }
.aija-cards .aija-card:nth-child(3) { animation-delay: 0.14s; }
@media (prefers-reduced-motion: reduce) {
    .aija-hero, .aija-cards, .aija-card { animation: none; }
}

.aija-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--aija-chip-border);
    border-radius: 22px;
    padding: 2.4rem 2.6rem 2.2rem;
    margin: 0.4rem 0 1.6rem;
    background:
        radial-gradient(420px 180px at 88% -30%, rgba(192, 38, 211, 0.12), transparent 70%),
        radial-gradient(420px 200px at 4% 120%, rgba(79, 70, 229, 0.10), transparent 65%),
        linear-gradient(135deg, #eef2ff 0%, #fbf6ff 55%, #eff6ff 100%);
}
.aija-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image: radial-gradient(rgba(79, 70, 229, 0.10) 1px, transparent 1px);
    background-size: 22px 22px;
    mask-image: radial-gradient(340px 190px at 100% 0%, rgba(0,0,0,0.9), transparent 75%);
    -webkit-mask-image: radial-gradient(340px 190px at 100% 0%, rgba(0,0,0,0.9), transparent 75%);
}
.aija-hero-badge {
    display: inline-block;
    background: var(--aija-surface);
    border: 1px solid var(--aija-chip-border);
    color: var(--aija-primary-dark);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    box-shadow: var(--aija-shadow-sm);
}
[dir="rtl"] .aija-hero-badge { letter-spacing: 0; text-transform: none; }
.aija-hero-title {
    font-size: clamp(2.1rem, 4.2vw, 3.1rem);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.035em;
    margin: 0.55em 0 0.35em;
    background: var(--aija-grad);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
.aija-hero-sub {
    color: var(--aija-body);
    font-size: 1.06rem;
    line-height: 1.65;
    max-width: 62ch;
    margin: 0;
}

.aija-cards {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 0.9rem;
    margin-bottom: 0.4rem;
}
.aija-card {
    background: var(--aija-surface);
    border: 1px solid var(--aija-border);
    border-radius: var(--aija-radius);
    padding: 1.3rem 1.35rem 1.25rem;
    box-shadow: var(--aija-shadow-sm);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.aija-card:hover {
    transform: translateY(-3px);
    border-color: #c7d2fe;
    box-shadow: var(--aija-shadow-lg);
}
.aija-card-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--aija-grad);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.95rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 6px 14px -6px rgba(79, 70, 229, 0.55);
}
.aija-card h3 { margin: 0 0 0.4rem; font-size: 1.02rem; }
.aija-card p { margin: 0; font-size: 0.9rem; line-height: 1.6; color: var(--aija-body); }

@media (max-width: 860px) {
    .aija-cards { grid-template-columns: 1fr; }
    .aija-hero { padding: 1.7rem 1.5rem 1.6rem; }
}
</style>
"""


def inject_theme() -> None:
    """Inject the shared stylesheet. Safe to call once per page render."""
    st.markdown(BASE_CSS, unsafe_allow_html=True)