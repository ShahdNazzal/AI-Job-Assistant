"""Shared Streamlit UI helpers: language, sidebar, headers, chips, downloads.

Every page calls ``render_sidebar()`` once at the top (after
``st.set_page_config``) so language, privacy status and session controls stay
consistent across the whole app.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import SETTINGS
from src.i18n import RTL_CSS, available_languages, is_rtl_language, translate
from src.privacy import clear_session_state
from src.utils import get_logger

logger = get_logger("ui_helpers")

PAGES = [
    ("app.py", "go_home"),
    ("pages/1_Resume_Analysis.py", "nav_resume"),
    ("pages/2_Job_Matching.py", "nav_matching"),
    ("pages/3_Skill_Gap_Analysis.py", "nav_gap"),
    ("pages/4_AI_Career_Assistant.py", "nav_assistant"),
    ("pages/5_Model_Evaluation.py", "nav_evaluation"),
]


def get_lang() -> str:
    """Current UI language code from session state."""
    return st.session_state.get("lang", "en")


def T(key: str, **kwargs) -> str:
    """Translate ``key`` in the current language."""
    return translate(key, get_lang(), **kwargs)


def apply_rtl() -> None:
    """Inject light RTL CSS when the current language is RTL."""
    if is_rtl_language(get_lang()):
        st.markdown(RTL_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Language selector, LLM status, privacy note, clear-data, navigation."""
    with st.sidebar:
        codes = list(available_languages().keys())
        current = get_lang()
        st.selectbox(
            T("language"),
            options=codes,
            index=codes.index(current) if current in codes else 0,
            format_func=lambda code: available_languages()[code],
            key="lang",
        )
        apply_rtl()

        st.divider()
        from src.llm_service import get_llm_service  # local import (fast UI start)

        service = get_llm_service()
        if service.is_configured():
            st.success(f"{T('llm_status')}: {T('llm_configured')}")
            st.caption(service.describe().replace(".env", "environment"))
        else:
            st.info(f"{T('llm_status')}: {T('llm_not_configured')}")

        st.caption(T("privacy_sidebar"))

        st.divider()
        if st.button(T("clear_data"), key="sidebar_clear_data"):
            clear_session_state()
            st.toast(T("clear_done"))

        st.divider()
        for page_path, label_key in PAGES:
            try:
                st.page_link(page=page_path, label=T(label_key))
            except Exception:  # noqa: BLE001 - page_link API differences
                st.markdown(f"- {T(label_key)}")


def page_header(title_key: str, subtitle_key: str | None = None) -> None:
    """Consistent page title + intro caption."""
    st.title(T(title_key))
    if subtitle_key:
        st.caption(T(subtitle_key))


def chip_list(items: list[str], separator: str = " ") -> str:
    """Render items as inline chips (plain markdown, no HTML injection)."""
    if not items:
        return ""
    return separator.join(f"`{item}`" for item in items)


def demo_badge() -> None:
    st.warning(f"**{T('demo_badge')}** - {T('demo_banner')}")


def score_interpretation_key(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 0.65:
        return "interp_high"
    if value >= 0.40:
        return "interp_medium"
    return "interp_low"


def download_bytes_button(
    label_key: str,
    data: bytes,
    filename: str,
    mime: str,
    key: str,
) -> None:
    st.download_button(
        label=T(label_key),
        data=data,
        file_name=filename,
        mime=mime,
        key=key,
        use_container_width=True,
    )


def timestamp_slug() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def app_title_footer() -> None:
    st.caption(f"{SETTINGS.app_name} v{SETTINGS.app_version} - {T('footer_note')}")


# --- Demo data (always clearly labeled in the UI) ----------------------------


def load_demo_resume():
    """Parse the bundled sample resume; returns a ResumeData."""
    from src.resume_parser import parse_resume_text

    path = PROJECT_ROOT / "data" / "sample_resume.txt"
    return parse_resume_text(
        path.read_text(encoding="utf-8"),
        source_type="demo",
        source_label="sample_resume.txt",
    )


def load_demo_job_text() -> str:
    """Return the bundled sample job description text."""
    path = PROJECT_ROOT / "data" / "sample_job_description.txt"
    return path.read_text(encoding="utf-8")
