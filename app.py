"""AI Job Assistant - Streamlit entry point (Home page).

Run from the project root with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.i18n import rtl_wrap
from src.privacy import LLM_DISCLOSURE_ITEMS, LLM_DISCLOSURE_NOTE
from src.ui_helpers import (
    T,
    app_title_footer,
    apply_rtl,
    demo_badge,
    load_demo_job_text,
    load_demo_resume,
    render_sidebar,
)

st.set_page_config(
    page_title="AI Job Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session-state defaults (shared by all pages) -----------------------------
DEFAULTS = {
    "resume_data": None,
    "confirmed_skills": [],
    "job_data": None,
    "job_input": "",
    "match_result": None,
    "recommendations": None,
    "report_lang": "en",
}
for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

render_sidebar()
apply_rtl()

# --- Hero ---------------------------------------------------------------------
lang = st.session_state.get("lang", "en")
st.markdown(
    rtl_wrap(
        f"""
<div class="aija-hero">
  <span class="aija-hero-badge">{T('home_badge')}</span>
  <h1 class="aija-hero-title">{T('hero_title')}</h1>
  <p class="aija-hero-sub">{T('hero_subtitle')}</p>
</div>
""",
        lang,
    ),
    unsafe_allow_html=True,
)

# --- How it works (feature cards) ----------------------------------------------
st.subheader(T("how_title"))
_steps = [
    (T("home_card1_title"), T("home_card1_body")),
    (T("home_card2_title"), T("home_card2_body")),
    (T("home_card3_title"), T("home_card3_body")),
]
_cards_html = "".join(
    f'<div class="aija-card"><span class="aija-card-num">{i}</span>'
    f"<h3>{title}</h3><p>{body}</p></div>"
    for i, (title, body) in enumerate(_steps, start=1)
)
st.markdown(
    rtl_wrap(f'<div class="aija-cards">{_cards_html}</div>', lang),
    unsafe_allow_html=True,
)
st.caption(T("home_cta_hint"))

st.divider()

# --- Demo data loader ---------------------------------------------------------
demo_col1, demo_col2 = st.columns([1, 2])
with demo_col1:
    if st.button(T("load_demo"), key="home_load_demo"):
        try:
            st.session_state.resume_data = load_demo_resume()
            st.session_state.confirmed_skills = [
                s.name for s in st.session_state.resume_data.skills_explicit
            ]
            st.session_state.job_input = load_demo_job_text()
            st.session_state.job_data = None
            st.session_state.match_result = None
            st.session_state.recommendations = None
            st.success(T("demo_loaded"))
            demo_badge()
        except Exception as exc:  # noqa: BLE001
            st.error(T("err_generic", detail=type(exc).__name__))
with demo_col2:
    st.caption(
        "Resume: `data/sample_resume.txt` - Job: `data/sample_job_description.txt`"
    )

st.divider()

# --- Privacy -------------------------------------------------------------------
with st.expander(T("privacy_home_title")):
    st.markdown(T("privacy_home_body"))
    st.markdown(f"**{T('what_is_sent')}**")
    for item in LLM_DISCLOSURE_ITEMS:
        st.markdown(f"- {item}")
    st.caption(LLM_DISCLOSURE_NOTE)

st.info(T("medical_disclaimer"))
app_title_footer()