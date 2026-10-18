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
st.markdown(rtl_wrap(f"# {T('hero_title')}", lang), unsafe_allow_html=True)
st.caption(T("hero_subtitle"))

# --- How it works -------------------------------------------------------------
st.subheader(T("how_title"))
col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown(T("how_step1"))
with col2:
    st.markdown(T("how_step2"))
with col3:
    st.markdown(T("how_step3"))

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
