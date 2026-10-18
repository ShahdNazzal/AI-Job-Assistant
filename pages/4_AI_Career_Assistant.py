"""Page 4 - AI Career Assistant: grounded Q&A (RAG + optional LLM),
resume improvement suggestions, and report downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import SETTINGS
from src.exceptions import ModelNotAvailableError
from src.i18n import rtl_wrap
from src.llm_service import get_llm_service
from src.privacy import LLM_DISCLOSURE_ITEMS, LLM_DISCLOSURE_NOTE
from src.rag_service import build_assistant_prompt, get_cached_rag_index
from src.recommendation_engine import generate_recommendations, rule_based_answer
from src.report_generator import (
    build_report_data,
    generate_csv_report,
    generate_json_report,
    generate_markdown_report,
    generate_pdf_report,
)
from src.ui_helpers import (
    T,
    apply_rtl,
    render_sidebar,
    timestamp_slug,
    download_bytes_button,
)
from src.utils import truncate

st.set_page_config(page_title="AI Career Assistant - AI Job Assistant", layout="wide")

render_sidebar()
apply_rtl()

st.title(T("assistant_title"))
st.caption(T("assistant_intro"))

lang = st.session_state.get("lang", "en")

resume_data = st.session_state.get("resume_data")
job_data = st.session_state.get("job_data")
match = st.session_state.get("match_result")

if not (resume_data and job_data and match):
    st.warning(T("err_need_analysis"))
    st.page_link("pages/2_Job_Matching.py", label=T("go_to_matching"))
    st.stop()

# --- LLM status and disclosure ---------------------------------------------------
llm = get_llm_service()
if llm.is_configured():
    st.info(
        T(
            "llm_box_configured",
            base_url=llm.config.base_url,
            model=llm.config.model,
        )
    )
else:
    st.warning(T("llm_box_missing"))

with st.expander(T("what_is_sent")):
    for item in LLM_DISCLOSURE_ITEMS:
        st.markdown(f"- {item}")
    st.caption(LLM_DISCLOSURE_NOTE)

# --- Knowledge base retrieval (works with or without an LLM) ----------------------
rag_index = None
try:
    rag_index = get_cached_rag_index()
except ModelNotAvailableError:
    st.warning(T("eval_needs_model"))
except FileNotFoundError:
    rag_index = None

confirmed_skills = st.session_state.get("confirmed_skills") or []

# --- Question input ---------------------------------------------------------------
st.subheader(T("suggested_title"))
sq_labels = [
    ("sq_gaps", "What are my three biggest skill gaps for this job?"),
    ("sq_projects", "Which of my projects are most relevant to this role?"),
    ("sq_summary", "How can I improve my professional summary?"),
    ("sq_interview", "What technical topics should I prepare for in an interview?"),
    ("sq_satisfied", "Which requirements do I already satisfy?"),
    ("sq_evidence", "What evidence supports my compatibility score?"),
]
question = ""
clicked_question = None
row_a, row_b, row_c = st.columns(3)
cols = [row_a, row_b, row_c]
for index, (key, english_text) in enumerate(sq_labels):
    with cols[index % 3]:
        if st.button(T(key), key=f"sq_{key}"):
            clicked_question = english_text

user_question = st.text_input(T("ask_placeholder"), key="assistant_question_input")
ask_clicked = st.button(T("ask_btn"), type="primary", key="ask_btn")

question = clicked_question or (user_question.strip() if ask_clicked and user_question.strip() else "")

if question:
    with st.spinner(T("thinking")):
        chunks = []
        if rag_index is not None:
            try:
                chunks = rag_index.retrieve(question, k=SETTINGS.rag_top_k)
            except Exception as exc:  # noqa: BLE001 - retrieval must not break the page
                st.caption(f"RAG retrieval skipped ({type(exc).__name__}).")

        answer = None
        used_llm = False
        if llm.is_configured():
            resume_summary = (
                f"Name: {resume_data.name or 'unknown'}\n"
                f"Confirmed skills: {', '.join(confirmed_skills) or 'none'}\n"
                f"Summary: {truncate(resume_data.summary or 'none', 600)}\n"
                f"Experience excerpt: {truncate(resume_data.experience_text or 'none', 1200)}\n"
                f"Projects excerpt: {truncate(resume_data.projects_text or 'none', 1200)}"
            )
            job_summary = (
                f"Title: {job_data.title or 'unknown'}\n"
                f"Required skills: {', '.join(job_data.required_skill_names) or 'none'}\n"
                f"Preferred skills: {', '.join(job_data.preferred_skill_names) or 'none'}\n"
                f"Responsibilities: {' | '.join(job_data.responsibilities[:6]) or 'none'}\n"
                f"Experience required: {job_data.experience_years_min or 'unspecified'} years"
            )
            comp_lines = "; ".join(
                f"{c.label}={c.value:.2f}" if c.value is not None else f"{c.label}=N/A"
                for c in match.components
            )
            match_summary = (
                f"Overall: {match.overall if match.overall is not None else 'N/A'} "
                f"({'; '.join(comp_lines)}). "
                f"Matched required: {', '.join(match.matched_required) or 'none'}. "
                f"Missing required: {', '.join(match.missing_required) or 'none'}."
            )
            system, user_prompt = build_assistant_prompt(
                question, resume_summary, job_summary, match_summary, chunks, lang
            )
            result = llm.chat(system, user_prompt)
            if result.ok:
                answer = result.content
                used_llm = True
            else:
                st.error(T("llm_failed", detail=result.error))

        if answer is None:
            answer = rule_based_answer(
                question, resume_data, job_data, match, confirmed_skills, chunks
            )

    st.subheader(T("answer_label"))
    if lang == "ar":
        st.markdown(rtl_wrap(answer.replace("\n", "  \n"), lang), unsafe_allow_html=True)
    else:
        st.markdown(answer)
    st.caption(T("ai_label") if used_llm else T("rules_label"))

    if chunks:
        with st.expander(T("kb_sources")):
            for chunk in chunks:
                st.markdown(
                    f"**[{chunk.title} - {chunk.heading}]** (similarity {chunk.score:.2f})"
                )
                st.markdown(f"> {chunk.text[:400]}...")
    elif rag_index is not None:
        st.caption(T("kb_no_match"))

st.divider()

# --- Resume improvement suggestions ------------------------------------------------
st.subheader(T("improve_title"))
st.caption(T("improve_intro"))

recommendations = st.session_state.get("recommendations")
if recommendations is None:
    recommendations = generate_recommendations(
        resume_data, job_data, match, confirmed_skills
    )
    st.session_state.recommendations = recommendations

for item in recommendations.items:
    with st.container(border=True):
        st.markdown(
            rtl_wrap(f"**{item.title}** - _{item.priority}_", lang),
            unsafe_allow_html=True,
        )
        if item.detail:
            st.markdown(rtl_wrap(item.detail, lang), unsafe_allow_html=True)
        if item.before or item.after:
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown(f"**{T('before_label')}**")
                st.markdown(rtl_wrap(f"> {item.before or '-'}", lang), unsafe_allow_html=True)
            with col_after:
                st.markdown(f"**{T('after_label')}**")
                st.markdown(rtl_wrap(f"> {item.after or '-'}", lang), unsafe_allow_html=True)
            st.caption(T("draft_warning"))

st.divider()

# --- Report downloads ---------------------------------------------------------------
st.subheader(T("downloads_title"))
st.caption(T("downloads_intro"))

report_data = build_report_data(resume_data, job_data, match, recommendations, confirmed_skills)
slug = timestamp_slug()

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    download_bytes_button(
        "dl_json",
        generate_json_report(report_data),
        f"ai_job_assistant_report_{slug}.json",
        "application/json",
        key="dl_json_btn",
    )
with col2:
    download_bytes_button(
        "dl_csv",
        generate_csv_report(report_data),
        f"ai_job_assistant_skills_{slug}.csv",
        "text/csv",
        key="dl_csv_btn",
    )
with col3:
    download_bytes_button(
        "dl_md",
        generate_markdown_report(report_data, lang=lang).encode("utf-8"),
        f"ai_job_assistant_report_{slug}.md",
        "text/markdown",
        key="dl_md_btn",
    )
with col4:
    try:
        pdf_bytes = generate_pdf_report(report_data, lang=lang)
        download_bytes_button(
            "dl_pdf",
            pdf_bytes,
            f"ai_job_assistant_report_{slug}.pdf",
            "application/pdf",
            key="dl_pdf_btn",
        )
        st.caption(T("pdf_built"))
    except Exception as exc:  # noqa: BLE001 - font/shaping problems must not crash
        st.error(T("err_generic", detail=f"PDF export: {type(exc).__name__}"))
