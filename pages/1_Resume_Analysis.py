"""Page 1 - Resume Analysis: upload, parse, review, confirm skills."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.design import chips, inject_design, theme_switch, page_header, stat_cards
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

theme_switch()  # dark-mode toggle at the top of the sidebar
render_sidebar()
apply_rtl()
inject_design()  # after render_sidebar so it overrides src/theme.py

lang = st.session_state.get("lang", "en")

page_header(T("resume_title"), T("resume_intro"), step=1, lang=lang, icon="📄")

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
    [f"📤 {T('tab_upload')}", f"📝 {T('tab_paste')}", f"✨ {T('tab_demo')}"]
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
    if st.button(T("parse_btn"), key="parse_pasted_btn", type="primary") and pasted.strip():
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
    if st.button(T("load_demo_resume"), key="load_demo_resume_btn", type="primary"):
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

if resume_data.source_type == "demo":
    demo_badge()

explicit = resume_data.skills_explicit
inferred = resume_data.skills_inferred
found_sections = [
    n for n, i in resume_data.sections.items() if n != "header" and i.found
]
total_sections = len([n for n in resume_data.sections if n != "header"])

st.success(T("parsed_ok"))

stat_cards(
    [
        ("الكلمات" if lang == "ar" else "Words", resume_data.word_count, ""),
        (T("skills_section"), len(explicit), "ok"),
        (T("inferred_badge"), len(inferred), "warn" if inferred else ""),
        (T("sections_label"), f"{len(found_sections)}/{total_sections}", ""),
    ]
)

for warning in resume_data.warnings:
    if warning.startswith("short_text:"):
        st.warning(T("warn_short_text", words=warning.split(":", 1)[1]))

left, right = st.columns([2, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader(f"👤 {T('candidate_section')}")
        name_cell = resume_data.name or T("not_detected")
        if resume_data.name and resume_data.name_confidence == "low":
            name_cell += f" _({T('name_conf_low')})_"
        st.markdown(
            rtl_wrap(f"**{T('name_label')}:** {name_cell}", lang), unsafe_allow_html=True
        )

        contact = resume_data.contact
        contact_bits = [
            f"✉️ {contact.email}" if contact.email else None,
            f"📞 {contact.phone}" if contact.phone else None,
            f"in {contact.linkedin}" if contact.linkedin else None,
            f"⌥ {contact.github}" if contact.github else None,
        ]
        present = [b for b in contact_bits if b]
        st.markdown(f"**{T('contact_label')}**")
        st.markdown(
            chips(present, "neutral", empty=T("not_detected")), unsafe_allow_html=True
        )

        st.markdown(f"**{T('summary_label')}**")
        summary_text = resume_data.summary or T("not_detected")
        if resume_data.summary and resume_data.summary_confidence == "low":
            summary_text += f" _({T('name_conf_low')})_"
        st.markdown(rtl_wrap(summary_text, lang), unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.subheader(f"🗂️ {T('sections_label')}")
        ok_names = []
        missing_names = []
        for section_name, info in resume_data.sections.items():
            if section_name == "header":
                continue
            (ok_names if info.found else missing_names).append(section_name)
        st.markdown(
            chips(ok_names, "ok") + chips(missing_names, "muted"), unsafe_allow_html=True
        )
        st.caption(f"{T('section_found')} / {T('section_missing')}")

st.divider()

# --- Skills: explicit + editable ----------------------------------------------
st.subheader(f"🧩 {T('skills_section')}")

if explicit:
    st.markdown(chips([s.name for s in explicit], "ok"), unsafe_allow_html=True)
    st.caption(T("skills_explicit_help"))
    for skill in explicit:
        with st.expander(f"{skill.name} · {skill.category_label}"):
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
    st.markdown(chips([s.name for s in inferred], "warn"), unsafe_allow_html=True)

st.divider()

with st.container(border=True):
    st.subheader(f"✏️ {T('edit_skills')}")
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
        label_visibility="collapsed",
    )
    st.caption(T("selected_count", count=len(selected)))
    if st.button(T("save_skills"), key="save_skills_btn", type="primary"):
        st.session_state.confirmed_skills = list(selected)
        st.session_state.match_result = None
        st.session_state.recommendations = None
        st.success(T("skills_saved", count=len(selected)))

st.divider()

with st.expander(f"📃 {T('view_text')}"):
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
    st.info(T("next_hint"))