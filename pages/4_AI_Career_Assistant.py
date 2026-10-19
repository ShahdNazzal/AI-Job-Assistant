"""Page 4 - AI Career Assistant: chat-style grounded Q&A (RAG + optional LLM),
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
from src.design import chip, inject_design, page_header
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
inject_design()

lang = st.session_state.get("lang", "en")

# --- Chat cosmetics (page-local, layered on top of the shared design) ---------------
CHAT_CSS = """
<style>
/* ---- chat window header ---- */
.aija-chat-header {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 20px; margin-bottom: 14px;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #c026d3);
    color: #ffffff;
    box-shadow: 0 12px 28px rgba(79, 70, 229, .28);
}
.aija-bot-avatar {
    width: 46px; height: 46px; border-radius: 15px;
    background: rgba(255, 255, 255, .18);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.aija-bot-name { font-weight: 700; font-size: 1.05rem; line-height: 1.25; }
.aija-bot-sub { font-size: .8rem; opacity: .9; }
.aija-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: #4ade80; display: inline-block;
    margin-inline-start: 8px; vertical-align: middle;
    box-shadow: 0 0 0 0 rgba(74, 222, 128, .6);
    animation: aijaPulse 1.8s infinite;
}
@keyframes aijaPulse {
    0%   { box-shadow: 0 0 0 0 rgba(74, 222, 128, .55); }
    70%  { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
}
/* ---- chat bubbles ---- */
div[data-testid="stChatMessage"] {
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .06);
    margin-bottom: 12px;
    padding: 14px 16px;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #eef2ff;
    border-color: #c7d2fe;
}
/* ---- chat input ---- */
div[data-testid="stChatInput"] {
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, .10);
}
div[data-testid="stChatInputElement"] textarea { border-radius: 14px !important; }
/* ---- suggestion pills ---- */
div[data-testid="stButton"] button { border-radius: 999px; }
/* ---- recommendation cards ---- */
.aj-reco-title { font-weight: 700; font-size: 1.02rem; }
.aj-before, .aj-after {
    border-radius: 12px; padding: 12px 14px; font-size: .92rem; line-height: 1.6;
}
.aj-before { background: #fef2f2; border: 1px solid #fecaca; }
.aj-after  { background: #f0fdf4; border: 1px solid #bbf7d0; }
.aj-ba-label { font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 6px; }
.aj-before .aj-ba-label { color: #b91c1c; }
.aj-after  .aj-ba-label { color: #15803d; }
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

page_header(T("assistant_title"), T("assistant_intro"), step=4, lang=lang, icon="🤖")

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

with st.expander(f"🔒 {T('what_is_sent')}"):
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

# --- Chat state --------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _render_message(msg: dict) -> None:
    """Render one stored chat message (user or assistant)."""
    role = msg["role"]
    avatar = "🧑‍💼" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            if msg.get("error"):
                st.error(msg["error"])
            st.markdown(
                chip(
                    T("ai_label") if msg.get("used_llm") else T("rules_label"),
                    "neutral" if msg.get("used_llm") else "muted",
                ),
                unsafe_allow_html=True,
            )
        content = msg.get("content", "")
        if lang == "ar":
            st.markdown(
                rtl_wrap(content.replace("\n", "  \n"), lang),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)
        if role == "assistant" and msg.get("chunks"):
            with st.expander(f"📚 {T('kb_sources')}"):
                for c in msg["chunks"]:
                    st.markdown(
                        f"**[{c['title']} - {c['heading']}]** (similarity {c['score']:.2f})"
                    )
                    st.markdown(f"> {c['text']}...")


def _ask(question_text: str) -> None:
    """Run the full RAG + LLM + rule-based pipeline and store both messages."""
    st.session_state.chat_history.append({"role": "user", "content": question_text})

    with st.spinner(T("thinking")):
        chunks = []
        if rag_index is not None:
            try:
                chunks = rag_index.retrieve(question_text, k=SETTINGS.rag_top_k)
            except Exception as exc:  # noqa: BLE001 - retrieval must not break the page
                st.caption(f"RAG retrieval skipped ({type(exc).__name__}).")

        answer = None
        used_llm = False
        error_detail = None
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
                question_text, resume_summary, job_summary, match_summary, chunks, lang
            )
            result = llm.chat(system, user_prompt)
            if result.ok:
                answer = result.content
                used_llm = True
            else:
                error_detail = T("llm_failed", detail=result.error)

        if answer is None:
            answer = rule_based_answer(
                question_text, resume_data, job_data, match, confirmed_skills, chunks
            )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "used_llm": used_llm,
            "error": error_detail,
            "chunks": [
                {
                    "title": c.title,
                    "heading": c.heading,
                    "score": float(c.score),
                    "text": c.text[:400],
                }
                for c in chunks
            ],
        }
    )


# --- Suggested questions ------------------------------------------------------------
st.subheader(f"💡 {T('suggested_title')}")
sq_labels = [
    ("sq_gaps", "What are my three biggest skill gaps for this job?"),
    ("sq_projects", "Which of my projects are most relevant to this role?"),
    ("sq_summary", "How can I improve my professional summary?"),
    ("sq_interview", "What technical topics should I prepare for in an interview?"),
    ("sq_satisfied", "Which requirements do I already satisfy?"),
    ("sq_evidence", "What evidence supports my compatibility score?"),
]
row_a, row_b, row_c = st.columns(3)
cols = [row_a, row_b, row_c]
for index, (key, english_text) in enumerate(sq_labels):
    with cols[index % 3]:
        if st.button(T(key), key=f"sq_{key}", use_container_width=True):
            st.session_state.pending_question = english_text

# --- Chat window --------------------------------------------------------------------
with st.container(border=True):
    bot_name = "مساعدة المسيرة الذكية" if lang == "ar" else "AI Career Coach"
    bot_sub = (
        "متاحة الآن — اسأليني أي شيء عن سيرتك وهذه الوظيفة"
        if lang == "ar"
        else "Online - ask me anything about your resume and this job"
    )
    clear_label = "🗑️ مسح المحادثة" if lang == "ar" else "🗑️ Clear chat"

    st.markdown(
        f"""
        <div class="aija-chat-header">
            <div class="aija-bot-avatar">🤖</div>
            <div>
                <div class="aija-bot-name">{bot_name}<span class="aija-dot"></span></div>
                <div class="aija-bot-sub">{bot_sub}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        welcome_ar = (
            "أهلاً بك! أنا مساعدتك المهنية الذكية. قرأت سيرتك الذاتية ووصف الوظيفة، "
            "وعندي قاعدة معرفة جاهزة عن البحث عن عمل. جرّبي أحد الأسئلة المقترحة أعلاه، "
            "أو اكتبي سؤالك في مربع الكتابة بالأسفل."
        )
        welcome_en = (
            "Hi there! I'm your AI career assistant. I have read your resume and the job "
            "description, and I have a built-in knowledge base about job hunting. "
            "Try one of the suggested questions above, or type your own in the box below."
        )
        with st.chat_message("assistant", avatar="🤖"):
            if lang == "ar":
                st.markdown(rtl_wrap(welcome_ar, lang), unsafe_allow_html=True)
            else:
                st.markdown(welcome_en)

    for msg in st.session_state.chat_history:
        _render_message(msg)

    if st.session_state.chat_history and st.button(clear_label, key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

st.divider()

# --- Resume improvement suggestions ------------------------------------------------
st.subheader(f"✨ {T('improve_title')}")
st.caption(T("improve_intro"))

recommendations = st.session_state.get("recommendations")
if recommendations is None:
    recommendations = generate_recommendations(
        resume_data, job_data, match, confirmed_skills
    )
    st.session_state.recommendations = recommendations

_PRIORITY_KIND = {"high": "miss", "medium": "warn", "low": "muted"}

for item in recommendations.items:
    with st.container(border=True):
        kind = _PRIORITY_KIND.get(str(item.priority).strip().lower(), "neutral")
        head, tag = st.columns([4, 1])
        with head:
            st.markdown(
                rtl_wrap(f'<div class="aj-reco-title">{item.title}</div>', lang),
                unsafe_allow_html=True,
            )
        with tag:
            st.markdown(chip(item.priority, kind), unsafe_allow_html=True)
        if item.detail:
            st.markdown(rtl_wrap(item.detail, lang), unsafe_allow_html=True)
        if item.before or item.after:
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown(
                    rtl_wrap(
                        f'<div class="aj-before"><div class="aj-ba-label">{T("before_label")}</div>'
                        f'{item.before or "-"}</div>',
                        lang,
                    ),
                    unsafe_allow_html=True,
                )
            with col_after:
                st.markdown(
                    rtl_wrap(
                        f'<div class="aj-after"><div class="aj-ba-label">{T("after_label")}</div>'
                        f'{item.after or "-"}</div>',
                        lang,
                    ),
                    unsafe_allow_html=True,
                )
            st.caption(T("draft_warning"))

st.divider()

# --- Report downloads ---------------------------------------------------------------
st.subheader(f"📥 {T('downloads_title')}")
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

# --- Chat input (pinned at the bottom, like a real chat app) ------------------------
prompt = st.chat_input(T("ask_placeholder"))
pending = st.session_state.pop("pending_question", None)
if prompt or pending:
    _ask(prompt or pending)
    st.rerun()