"""Page 1 - Resume Analysis: upload, parse, review, confirm skills."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.exceptions import (
    AIJobAssistantError,
    CorruptedFileError,
    EmptyDocumentError,
    EmptyFileError,
    FileTooLargeError,
    InsufficientTextError,
    UnsupportedFileTypeError,
)
from src.i18n import rtl_wrap
from src.resume_parser import parse_resume, parse_resume_text
from src.skill_extractor import taxonomy_skill_options
from src.ui_helpers import (
    T,
    apply_rtl,
    demo_badge,
    load_demo_resume,
    render_sidebar,
)

st.set_page_config(page_title="Resume Analysis - AI Job Assistant", layout="wide")

render_sidebar()
apply_rtl()

st.title(T("resume_title"))
st.caption(T("resume_intro"))

UPLOAD_KEY = "resume_uploader"


def _friendly_parse_error(exc: AIJobAssistantError) -> str:
    if isinstance(exc, UnsupportedFileTypeError):
        ext = Path(st.session_state.get("_last_filename", "file")).suffix or "?"
        return T("err_unsupported", ext=ext)
    if isinstance(exc, FileTooLargeError):
        from config.settings import SETTINGS

        return T("err_too_large", max_mb=SETTINGS.max_upload_mb)
    if isinstance(exc, EmptyFileError):
        return T("err_empty_file")
    if isinstance(exc, CorruptedFileError):
        return T("err_corrupted", kind="PDF/DOCX")
    if isinstance(exc, EmptyDocumentError):
        return T("err_empty_doc")
    if isinstance(exc, InsufficientTextError):
        return T("err_scanned")
    return T("err_generic", detail=type(exc).__name__)


def _store_resume(resume_data, source_label: str) -> None:
    st.session_state.resume_data = resume_data
    st.session_state.confirmed_skills = [s.name for s in resume_data.skills_explicit]
    st.session_state["_last_filename"] = source_label
    # Invalidate downstream results.
    st.session_state.job_data = None
    st.session_state.match_result = None
    st.session_state.recommendations = None
    st.toast(T("parsed_ok"))


# --- Input tabs ----------------------------------------------------------------
tab_upload, tab_paste, tab_demo = st.tabs(
    [T("tab_upload"), T("tab_paste"), T("tab_demo")]
)

with tab_upload:
    uploaded = st.file_uploader(
        T("upload_label", max_mb=10), type=["pdf", "docx", "txt"], key=UPLOAD_KEY
    )
    if uploaded is not None:
        signature = (uploaded.name, uploaded.size)
        if st.session_state.get("_upload_signature") != signature:
            st.session_state["_upload_signature"] = signature
            try:
                with st.spinner(T("parsing")):
                    data_bytes = uploaded.getvalue()
                    resume_data = parse_resume(data_bytes, uploaded.name)
                _store_resume(resume_data, uploaded.name)
            except AIJobAssistantError as exc:
                st.session_state.resume_data = None
                st.error(_friendly_parse_error(exc))
            except Exception as exc:  # noqa: BLE001
                st.session_state.resume_data = None
                st.error(T("err_generic", detail=type(exc).__name__))

with tab_paste:
    pasted = st.text_area(T("paste_label"), height=220, key="paste_area")
    if st.button(T("parse_btn"), key="parse_pasted_btn") and pasted.strip():
        try:
            with st.spinner(T("parsing")):
                resume_data = parse_resume_text(
                    pasted, source_type="paste", source_label="pasted text"
                )
            _store_resume(resume_data, "pasted text")
        except AIJobAssistantError as exc:
            st.error(_friendly_parse_error(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(T("err_generic", detail=type(exc).__name__))

with tab_demo:
    st.caption(T("demo_banner"))
    if st.button(T("load_demo_resume"), key="load_demo_resume_btn"):
        try:
            with st.spinner(T("parsing")):
                resume_data = load_demo_resume()
            _store_resume(resume_data, "sample_resume.txt")
        except Exception as exc:  # noqa: BLE001
            st.error(T("err_generic", detail=type(exc).__name__))

# --- Results -------------------------------------------------------------------
resume_data = st.session_state.get("resume_data")
if not resume_data:
    st.stop()

lang = st.session_state.get("lang", "en")

if resume_data.source_type == "demo":
    demo_badge()

st.success(T("parsed_ok"))
st.caption(T("word_count_label", words=resume_data.word_count))

for warning in resume_data.warnings:
    if warning.startswith("short_text:"):
        st.warning(T("warn_short_text", words=warning.split(":", 1)[1]))

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader(T("candidate_section"))
    name_cell = resume_data.name or T("not_detected")
    if resume_data.name and resume_data.name_confidence == "low":
        name_cell += f" _({T('name_conf_low')})_"
    st.markdown(rtl_wrap(f"**{T('name_label')}:** {name_cell}", lang), unsafe_allow_html=True)

    contact = resume_data.contact
    contact_bits = [
        f"Email: {contact.email}" if contact.email else None,
        f"Phone: {contact.phone}" if contact.phone else None,
        f"LinkedIn: {contact.linkedin}" if contact.linkedin else None,
        f"GitHub: {contact.github}" if contact.github else None,
    ]
    st.markdown(
        rtl_wrap(
            f"**{T('contact_label')}:** "
            + (" | ".join(b for b in contact_bits if b) or T("not_detected")),
            lang,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(f"**{T('summary_label')}**")
    summary_text = resume_data.summary or T("not_detected")
    if resume_data.summary and resume_data.summary_confidence == "low":
        summary_text += f" _({T('name_conf_low')})_"
    st.markdown(rtl_wrap(summary_text, lang), unsafe_allow_html=True)

with right:
    st.subheader(T("sections_label"))
    for section_name, info in resume_data.sections.items():
        if section_name == "header":
            continue
        status = T("section_found") if info.found else T("section_missing")
        marker = "**[x]**" if info.found else "[ ]"
        st.markdown(f"{marker} `{section_name}` - {status}")

st.divider()

# --- Skills: explicit + editable ----------------------------------------------
st.subheader(T("skills_section"))

explicit = resume_data.skills_explicit
inferred = resume_data.skills_inferred

if explicit:
    st.markdown(
        rtl_wrap(
            f"{len(explicit)} " + T("skills_section"),
            lang,
        ),
        unsafe_allow_html=True,
    )
    st.caption(T("skills_explicit_help"))
    for skill in explicit:
        with st.expander(f"**{skill.name}** - {skill.category_label}"):
            if skill.matched_via == "alias":
                st.caption(T("via_alias"))
            for ev in skill.evidence:
                section = f" `[{ev.section}]`" if ev.section else ""
                st.markdown(f"-{section} \"{ev.snippet}\"")
else:
    st.info(T("no_evidence"))

if inferred:
    st.markdown(f"**{T('inferred_badge')}** ({len(inferred)})")
    st.caption(T("skills_inferred_help"))
    inferred_names = ", ".join(s.name for s in inferred)
    st.markdown(rtl_wrap(inferred_names, lang), unsafe_allow_html=True)

st.divider()

st.subheader(T("edit_skills"))
st.caption(T("edit_skills_help"))
options = taxonomy_skill_options()
current = st.session_state.get("confirmed_skills") or []
current = [c for c in current if c in options]
selected = st.multiselect(
    T("edit_skills"),
    options=options,
    default=current,
    key="skills_editor",
    help=T("edit_skills_help"),
)
st.caption(T("selected_count", count=len(selected)))
if st.button(T("save_skills"), key="save_skills_btn"):
    st.session_state.confirmed_skills = list(selected)
    st.session_state.match_result = None
    st.session_state.recommendations = None
    st.success(T("skills_saved", count=len(selected)))

st.divider()

with st.expander(T("view_text")):
    st.text(resume_data.raw_text[:8000])

col_clear, col_next = st.columns([1, 2])
with col_clear:
    if st.button(T("clear_resume"), key="clear_resume_btn"):
        st.session_state.resume_data = None
        st.session_state.confirmed_skills = []
        st.session_state.match_result = None
        st.session_state.recommendations = None
        st.rerun()
with col_next:
    st.markdown(T("next_hint"))
