"""Shared visual design layer (modern React / Next.js look) with light + dark mode.

Usage in every page, as the FIRST thing after set_page_config:

    from src.design import theme_switch, inject_design, page_header, stat_cards, chip, chips, style_fig, is_dark
    theme_switch()       # dark-mode toggle (top of the sidebar)
    render_sidebar()
    apply_rtl()
    inject_design()      # after render_sidebar so it overrides src/theme.py
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import streamlit as st

# ----------------------------------------------------------------------------------
# Theme tokens
# ----------------------------------------------------------------------------------
_LIGHT_VARS = """
    --aj-primary: #4f46e5;
    --aj-primary-2: #7c3aed;
    --aj-accent-ink: #4f46e5;
    --aj-ink: #0f172a;
    --aj-muted: #64748b;
    --aj-line: #e2e8f0;
    --aj-card: #ffffff;
    --aj-surface: #ffffff;
    --aj-input: #ffffff;
    --aj-soft: #eef2ff;
    --aj-soft-line: #c7d2fe;
    --aj-soft-ink: #4338ca;
    --aj-muted-bg: #f1f5f9;
    --aj-bg: #f8fafc;
    --aj-bg-a: #e0e7ff;
    --aj-bg-b: #fae8ff;
    --aj-header: rgba(248, 250, 252, .75);
    --aj-dropzone: #f5f7ff;
    --aj-tabs: #eef2ff;
    --aj-tab-active: #ffffff;
    --aj-ok: #16a34a;
    --aj-ok-bg: #dcfce7;
    --aj-ok-line: #bbf7d0;
    --aj-ok-ink: #166534;
    --aj-miss: #dc2626;
    --aj-miss-bg: #fee2e2;
    --aj-miss-line: #fecaca;
    --aj-miss-ink: #991b1b;
    --aj-warn-bg: #fef3c7;
    --aj-warn-line: #fde68a;
    --aj-warn-ink: #b45309;
    --aj-g1: #4f46e5;
    --aj-g2: #7c3aed;
    --aj-g3: #c026d3;
    --aj-radius: 16px;
    --aj-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 8px 24px rgba(15, 23, 42, .06);
    --aj-scroll: #cbd5e1;
"""

_DARK_VARS = """
    color-scheme: dark;
    /* softer slate palette (not black) */
    --aj-primary: #6366f1;
    --aj-primary-2: #8b5cf6;
    --aj-accent-ink: #a5b4fc;
    --aj-ink: #f1f5f9;
    --aj-muted: #94a3b8;
    --aj-line: #334155;
    --aj-card: #1e293b;
    --aj-surface: #172033;
    --aj-input: #141d30;
    --aj-soft: #2a3160;
    --aj-soft-line: #4a54a0;
    --aj-soft-ink: #c7d2fe;
    --aj-muted-bg: #273449;
    --aj-bg: #0f172a;
    --aj-bg-a: rgba(99, 102, 241, .22);
    --aj-bg-b: rgba(168, 85, 247, .14);
    --aj-header: rgba(15, 23, 42, .78);
    --aj-dropzone: #18213a;
    --aj-tabs: #1a2338;
    --aj-tab-active: #2c3a66;
    --aj-ok: #22c55e;
    --aj-ok-bg: rgba(34, 197, 94, .14);
    --aj-ok-line: rgba(34, 197, 94, .38);
    --aj-ok-ink: #86efac;
    --aj-miss: #ef4444;
    --aj-miss-bg: rgba(239, 68, 68, .14);
    --aj-miss-line: rgba(239, 68, 68, .40);
    --aj-miss-ink: #fca5a5;
    --aj-warn-bg: rgba(245, 158, 11, .14);
    --aj-warn-line: rgba(245, 158, 11, .40);
    --aj-warn-ink: #fcd34d;
    --aj-g1: #4338ca;
    --aj-g2: #6d28d9;
    --aj-g3: #a21caf;
    --aj-shadow: 0 1px 2px rgba(0, 0, 0, .25), 0 10px 28px rgba(0, 0, 0, .28);
    --aj-scroll: #475569;

    /* re-map the variables used by src/theme.py so its rules follow the dark palette */
    --aija-ink: #f1f5f9;
    --aija-body: #cbd5e1;
    --aija-muted: #94a3b8;
    --aija-primary: #818cf8;
    --aija-primary-dark: #c7d2fe;
    --aija-border: #334155;
    --aija-border-soft: #2b3750;
    --aija-surface: #1e293b;
    --aija-canvas: #172033;
    --aija-chip-bg: #2a3160;
    --aija-chip-text: #c7d2fe;
    --aija-chip-border: #4a54a0;
    --aija-shadow-sm: 0 1px 2px rgba(0, 0, 0, .30);
    --aija-shadow-md: 0 10px 28px -14px rgba(0, 0, 0, .60);
    --aija-shadow-lg: 0 18px 40px -16px rgba(0, 0, 0, .70);
"""

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- base ---------- */
html, body, .stApp, .stMarkdown, button, input, textarea, select {
    font-family: 'Inter', 'Cairo', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(1100px 480px at 8% -8%, var(--aj-bg-a) 0%, transparent 60%),
        radial-gradient(900px 420px at 100% 0%, var(--aj-bg-b) 0%, transparent 55%),
        var(--aj-bg);
    background-attachment: fixed;
    color: var(--aj-ink);
}
.block-container { padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- top header bar ---------- */
header[data-testid="stHeader"] {
    background: var(--aj-header);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--aj-line);
}
header[data-testid="stHeader"] :is(button, a, span, p, div, svg) { color: var(--aj-ink); }
div[data-testid="stDecoration"] { display: none; }

h1, h2, h3, h4 { letter-spacing: -0.01em; }
h3 { font-weight: 700 !important; font-size: 1.25rem !important; margin-top: .4rem; }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--aj-line), transparent); margin: 1.6rem 0; }

code {
    background: var(--aj-soft); color: var(--aj-soft-ink); border-radius: 7px;
    padding: .12rem .45rem; font-size: .85em; border: 1px solid var(--aj-soft-line);
}

/* ---------- hero header ---------- */
.aj-hero {
    position: relative; overflow: hidden;
    display: flex; align-items: center; gap: 18px;
    padding: 26px 28px; margin-bottom: 1.4rem;
    border-radius: 22px; color: #fff;
    background: linear-gradient(135deg, var(--aj-g1) 0%, var(--aj-g2) 55%, var(--aj-g3) 100%);
    box-shadow: 0 18px 40px rgba(79, 70, 229, .28);
}
.aj-hero::before, .aj-hero::after {
    content: ""; position: absolute; border-radius: 50%;
    background: rgba(255, 255, 255, .10); pointer-events: none;
}
.aj-hero::before { width: 260px; height: 260px; inset-inline-end: -70px; top: -120px; }
.aj-hero::after  { width: 160px; height: 160px; inset-inline-end: 120px; bottom: -90px; }
.aj-hero-icon {
    width: 58px; height: 58px; flex-shrink: 0; border-radius: 18px;
    background: rgba(255, 255, 255, .18); backdrop-filter: blur(6px);
    display: flex; align-items: center; justify-content: center; font-size: 28px;
}
.aj-hero-text { position: relative; z-index: 1; flex: 1; }
.aj-badge {
    display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .04em;
    padding: 3px 11px; border-radius: 999px; margin-bottom: 8px; color: #fff;
    background: rgba(255, 255, 255, .20); border: 1px solid rgba(255, 255, 255, .28);
}
.aj-hero-title { font-size: 1.7rem; font-weight: 800; line-height: 1.25; color: #fff; }
.aj-hero-sub { font-size: .95rem; opacity: .92; margin-top: 6px; max-width: 760px; color: #fff; }
.aj-progress { display: flex; gap: 6px; margin-top: 14px; max-width: 280px; }
.aj-progress span { flex: 1; height: 5px; border-radius: 999px; background: rgba(255, 255, 255, .28); }
.aj-progress span.on { background: #fff; }

/* ---------- cards (bordered containers) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--aj-card);
    border: 1px solid var(--aj-line) !important;
    border-radius: var(--aj-radius) !important;
    box-shadow: var(--aj-shadow);
}

/* ---------- stat cards ---------- */
.aj-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: .4rem 0 1.2rem; }
.aj-stat {
    background: var(--aj-card); border: 1px solid var(--aj-line); border-radius: var(--aj-radius);
    padding: 16px 18px; box-shadow: var(--aj-shadow); position: relative; overflow: hidden;
}
.aj-stat::before {
    content: ""; position: absolute; inset-inline-start: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--aj-primary), var(--aj-primary-2));
}
.aj-stat.ok::before   { background: var(--aj-ok); }
.aj-stat.miss::before { background: var(--aj-miss); }
.aj-stat.warn::before { background: #f59e0b; }
.aj-stat-val { font-size: 1.9rem; font-weight: 800; line-height: 1.1; color: var(--aj-ink); }
.aj-stat-label { font-size: .82rem; color: var(--aj-muted); margin-top: 4px; font-weight: 500; }

/* ---------- chips ---------- */
.aj-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: .35rem 0 .6rem; }
.aj-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 5px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600;
    border: 1px solid transparent; line-height: 1.3;
}
.aj-chip.neutral { background: var(--aj-soft); color: var(--aj-soft-ink); border-color: var(--aj-soft-line); }
.aj-chip.ok      { background: var(--aj-ok-bg); color: var(--aj-ok-ink); border-color: var(--aj-ok-line); }
.aj-chip.miss    { background: var(--aj-miss-bg); color: var(--aj-miss-ink); border-color: var(--aj-miss-line); }
.aj-chip.warn    { background: var(--aj-warn-bg); color: var(--aj-warn-ink); border-color: var(--aj-warn-line); }
.aj-chip.muted   { background: var(--aj-muted-bg); color: var(--aj-muted); border-color: var(--aj-line); }
.aj-empty { color: var(--aj-muted); font-size: .88rem; }

/* ---------- buttons ---------- */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stPageLink"] a {
    border-radius: 12px; font-weight: 600;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stDownloadButton"] button,
div[data-testid="stPageLink"] a {
    background: var(--aj-card); color: var(--aj-ink); border: 1px solid var(--aj-line);
}
div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-1px); border-color: var(--aj-primary); box-shadow: 0 6px 16px rgba(79, 70, 229, .18);
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--aj-primary), var(--aj-primary-2));
    border: none; box-shadow: 0 8px 20px rgba(79, 70, 229, .30);
}
div[data-testid="stButton"] button[kind="primary"] * { color: #fff !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { box-shadow: 0 12px 26px rgba(79, 70, 229, .40); }

/* ---------- tabs ---------- */
div[data-baseweb="tab-list"] {
    gap: 6px; background: var(--aj-tabs); padding: 5px; border-radius: 14px; width: fit-content;
}
button[data-baseweb="tab"] {
    border-radius: 10px !important; padding: 8px 18px !important; font-weight: 600; height: auto;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--aj-tab-active); box-shadow: 0 2px 8px rgba(15, 23, 42, .10);
}
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--aj-accent-ink) !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

/* ---------- inputs ---------- */
div[data-baseweb="textarea"], div[data-baseweb="input"], div[data-baseweb="base-input"],
div[data-baseweb="select"] > div {
    border-radius: 12px !important; border-color: var(--aj-line) !important; background: var(--aj-input) !important;
}
div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input,
div[data-baseweb="select"] input, div[data-baseweb="select"] div[value] {
    background: transparent !important; color: var(--aj-ink) !important; -webkit-text-fill-color: var(--aj-ink);
}
div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within > div {
    border-color: var(--aj-primary) !important; box-shadow: 0 0 0 3px rgba(99, 102, 241, .20);
}
span[data-baseweb="tag"] {
    background: var(--aj-soft) !important; color: var(--aj-soft-ink) !important;
    border-radius: 999px !important; font-weight: 600;
}
span[data-baseweb="tag"] * { color: var(--aj-soft-ink) !important; }
section[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed var(--aj-soft-line); border-radius: 16px; background: var(--aj-dropzone);
    transition: background .2s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover { background: var(--aj-soft); }
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--aj-card); color: var(--aj-ink); border: 1px solid var(--aj-line);
}

/* dropdown menus / popovers (multiselect list, selectbox, header menu) */
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] div[role="listbox"] {
    background: var(--aj-card) !important; color: var(--aj-ink) !important;
    border: 1px solid var(--aj-line); border-radius: 12px;
}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * { color: var(--aj-ink) !important; background: transparent; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {
    background: var(--aj-soft) !important;
}

/* ---------- metrics ---------- */
div[data-testid="stMetric"] {
    background: var(--aj-card); border: 1px solid var(--aj-line); border-radius: var(--aj-radius);
    padding: 14px 18px; box-shadow: var(--aj-shadow);
}
div[data-testid="stMetricValue"] { font-weight: 800; }
div[data-testid="stMetricLabel"] * { color: var(--aj-muted) !important; font-weight: 500; }

/* ---------- expanders / alerts / tables / charts ---------- */
div[data-testid="stExpander"] {
    background: var(--aj-card); border: 1px solid var(--aj-line) !important;
    border-radius: 14px !important; box-shadow: 0 1px 2px rgba(15, 23, 42, .04); overflow: hidden;
}
div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background: transparent; }
div[data-testid="stExpander"] summary { font-weight: 600; }
div[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--aj-line); }
div[data-testid="stDataFrame"] {
    border: 1px solid var(--aj-line); border-radius: 14px; overflow: hidden; box-shadow: var(--aj-shadow);
}
div[data-testid="stPlotlyChart"] {
    background: var(--aj-card); border: 1px solid var(--aj-line); border-radius: var(--aj-radius);
    padding: 8px; box-shadow: var(--aj-shadow); overflow: hidden;
}

/* ---------- chat input pinned at the bottom ---------- */
div[data-testid="stBottom"], div[data-testid="stBottom"] > div { background: var(--aj-bg); }
div[data-testid="stChatInput"] { background: var(--aj-input); border: 1px solid var(--aj-line); }
div[data-testid="stChatInput"] textarea { color: var(--aj-ink) !important; background: transparent !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--aj-surface); border-inline-end: 1px solid var(--aj-line);
}
section[data-testid="stSidebar"] :is(p, span, label, a, li, h1, h2, h3, h4) { color: var(--aj-ink); }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--aj-muted); }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] { border-radius: 10px; margin: 2px 6px; }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--aj-soft); }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: var(--aj-accent-ink); font-weight: 700;
}
/* the theme switch card */
section[data-testid="stSidebar"] div[data-testid="stToggle"] {
    background: var(--aj-soft); border: 1px solid var(--aj-soft-line);
    border-radius: 14px; padding: 10px 14px; margin: 6px 0 10px;
}

/* ---------- toasts, scrollbar ---------- */
div[data-testid="stToast"] { border-radius: 14px; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--aj-scroll); border-radius: 999px; border: 2px solid transparent; background-clip: content-box; }

@media (max-width: 720px) {
    .aj-hero { padding: 20px; flex-direction: column; align-items: flex-start; }
    .aj-hero-title { font-size: 1.35rem; }
}
"""

_POLISH_CSS = """
/* ---------- polish: React / Next.js feel ---------- */
html { scroll-behavior: smooth; }
::selection { background: rgba(99, 102, 241, .30); }
.stApp { font-feature-settings: "cv11", "ss01"; -webkit-font-smoothing: antialiased; }
.stApp h1, .stApp h2, .stApp h3 { letter-spacing: -0.025em; }
:focus-visible { outline: 2px solid var(--aj-primary); outline-offset: 2px; border-radius: 8px; }

@keyframes ajRise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes ajFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(12px); } }
.block-container { animation: ajRise .45s cubic-bezier(.2, .7, .2, 1) both; }

/* hero: dotted grid + floating glow like modern landing pages */
.aj-hero {
    background:
        radial-gradient(rgba(255, 255, 255, .16) 1px, transparent 1px) 0 0 / 22px 22px,
        linear-gradient(135deg, var(--aj-g1) 0%, var(--aj-g2) 55%, var(--aj-g3) 100%);
}
.aj-hero::before { animation: ajFloat 9s ease-in-out infinite; }
.aj-hero-title { letter-spacing: -0.03em; }

/* cards lift on hover */
div[data-testid="stVerticalBlockBorderWrapper"], .aj-stat, div[data-testid="stMetric"], div[data-testid="stExpander"] {
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.aj-stat:hover, div[data-testid="stMetric"]:hover {
    transform: translateY(-3px); border-color: var(--aj-soft-line);
    box-shadow: 0 16px 34px -14px rgba(79, 70, 229, .40);
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--aj-soft-line) !important; }
div[data-testid="stExpander"]:hover { border-color: var(--aj-soft-line) !important; }

/* chips + buttons micro-interactions */
.aj-chip { transition: transform .15s ease, box-shadow .15s ease; }
.aj-chip:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(79, 70, 229, .18); }
div[data-testid="stButton"] button:active, div[data-testid="stDownloadButton"] button:active {
    transform: translateY(0) scale(.98);
}

/* sidebar navigation */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    transition: background .15s ease, transform .15s ease;
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover,
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { transform: translateX(3px); }
section[data-testid="stSidebar"] div[data-testid="stToggle"] { box-shadow: var(--aj-shadow); }

div[data-testid="stAlert"] { backdrop-filter: blur(6px); }

@media (prefers-reduced-motion: reduce) {
    .block-container, .aj-hero::before { animation: none; }
    * { transition: none !important; }
}
"""

# Streamlit's own widgets take their colours from the (light) base theme, so dark mode
# has to force the text colours and a few native surfaces explicitly.
_DARK_EXTRAS = """
.stApp :is(h1, h2, h3, h4, h5, h6, p, li, label, legend, summary, [data-testid="stWidgetLabel"] *,
           [data-testid="stMarkdownContainer"]) { color: var(--aj-ink); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: var(--aj-muted); }
.stApp div[data-testid="stAlert"] :is(p, span, div, li, a) { color: var(--aj-ink); }
.stApp div[data-testid="stExpander"] :is(summary, summary *) { color: var(--aj-ink); }
.stApp div[data-testid="stChatMessage"] { color: var(--aj-ink); }
/* st.dataframe is canvas-rendered: invert it so it matches the dark surface */
div[data-testid="stDataFrame"] { filter: invert(.92) hue-rotate(180deg); }
pre, .stApp pre code { background: var(--aj-input) !important; color: var(--aj-ink) !important; }
div[data-testid="stVerticalBlockBorderWrapper"], .aj-stat, div[data-testid="stMetric"] {
    box-shadow: var(--aj-shadow), inset 0 1px 0 rgba(255, 255, 255, .04);
}

/* rules from src/theme.py that hard-code light colours */
.stApp [data-testid="stSidebar"] a,
.stApp [data-testid="stSidebar"] [data-testid="stPageLink"] a { color: var(--aija-body); }
.stApp [data-testid="stSidebar"] a:hover,
.stApp [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background: var(--aija-chip-bg); color: var(--aija-primary-dark);
}
.stApp .stButton > button:hover { background: var(--aj-soft); border-color: var(--aj-soft-line); }
.stApp .stDownloadButton > button { border-color: var(--aj-soft-line); color: var(--aj-soft-ink); }
.stApp .aija-hero {
    background:
        radial-gradient(420px 180px at 88% -30%, rgba(192, 38, 211, .20), transparent 70%),
        linear-gradient(135deg, #1e2548 0%, #2a1f4d 55%, #172a4a 100%);
    border-color: var(--aj-soft-line);
}
"""


def is_dark() -> bool:
    """True when the user switched the sidebar toggle to dark mode."""
    return bool(st.session_state.get("aj_dark", False))


def theme_switch() -> bool:
    """Render the dark-mode switch at the top of the sidebar.

    Double-call safe: only the first call in a script run draws the widget;
    any later call in the same run simply reuses its stored value. So it does
    not matter whether a page calls it directly AND render_sidebar() calls it.
    """
    if "aj_dark_toggle" not in st.session_state:
        st.session_state["aj_dark_toggle"] = bool(st.session_state.get("aj_dark", False))
    lang = st.session_state.get("lang", "en")
    label = "🌙 الوضع الداكن" if lang == "ar" else "🌙 Dark mode"
    switch = getattr(st.sidebar, "toggle", None) or st.sidebar.checkbox
    try:
        value = switch(label, key="aj_dark_toggle")
    except Exception:  # duplicate key => toggle already drawn this run
        return bool(st.session_state.get("aj_dark", False))
    st.session_state["aj_dark"] = bool(value)
    return bool(value)


def inject_design() -> None:
    """Inject the global stylesheet.

    Call it AFTER render_sidebar() (which injects src/theme.py) so these rules win.
    """
    css = ":root {" + (_LIGHT_VARS + (_DARK_VARS if is_dark() else "")) + "}" + _BASE_CSS + _POLISH_CSS
    if is_dark():
        css += _DARK_EXTRAS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: str = "",
    *,
    step: int | None = None,
    total: int = 4,
    lang: str = "en",
    icon: str = "",
) -> None:
    """Gradient hero banner that replaces st.title + st.caption."""
    badge = ""
    progress = ""
    if step:
        label = f"الخطوة {step} من {total}" if lang == "ar" else f"Step {step} of {total}"
        badge = f'<span class="aj-badge">{label}</span>'
        dots = "".join(
            f'<span class="{"on" if i < step else ""}"></span>' for i in range(total)
        )
        progress = f'<div class="aj-progress">{dots}</div>'
    icon_html = f'<div class="aj-hero-icon">{icon}</div>' if icon else ""
    sub_html = f'<div class="aj-hero-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="aj-hero">{icon_html}<div class="aj-hero-text">{badge}'
        f'<div class="aj-hero-title">{html.escape(title)}</div>{sub_html}{progress}</div></div>',
        unsafe_allow_html=True,
    )


def stat_cards(items: Sequence[tuple[str, object, str]]) -> None:
    """Row of KPI cards. items = [(label, value, tone)], tone in {"", "ok", "miss", "warn"}."""
    cards = "".join(
        f'<div class="aj-stat {tone}"><div class="aj-stat-val">{html.escape(str(value))}</div>'
        f'<div class="aj-stat-label">{html.escape(str(label))}</div></div>'
        for label, value, tone in items
    )
    st.markdown(f'<div class="aj-stats">{cards}</div>', unsafe_allow_html=True)


def chip(text: object, kind: str = "neutral", title: str = "") -> str:
    """One pill. kind in {"neutral", "ok", "miss", "warn", "muted"}. Returns HTML."""
    return (
        f'<span class="aj-chip {kind}" title="{html.escape(title)}">'
        f"{html.escape(str(text))}</span>"
    )


def chips(items: Iterable[object], kind: str = "neutral", empty: str = "") -> str:
    """A wrapped row of pills. Returns HTML (render with st.markdown(..., unsafe_allow_html=True))."""
    parts = [chip(i, kind) for i in items]
    if not parts:
        return f'<span class="aj-empty">{html.escape(empty)}</span>' if empty else ""
    return f'<div class="aj-chips">{"".join(parts)}</div>'


def style_fig(fig, height: int | None = None):
    """Apply the shared look (light or dark) to a Plotly figure."""
    ink = "#e6e9f5" if is_dark() else "#0f172a"
    grid = "#263248" if is_dark() else "#e2e8f0"
    fig.update_layout(
        font={"family": "Inter, Cairo, sans-serif", "color": ink},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font={"size": 15, "color": ink},
        legend_title_text="",
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig