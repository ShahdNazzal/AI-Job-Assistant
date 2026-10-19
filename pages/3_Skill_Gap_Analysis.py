"""Page 3 - Skill Gap Analysis: prioritized gaps with honest guidance."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.design import chips, inject_design, theme_switch, page_header, stat_cards
from src.i18n import translate
from src.models import SkillGapItem
from src.skill_extractor import get_skill_entry
from src.ui_helpers import T, apply_rtl, render_sidebar, timestamp_slug

st.set_page_config(page_title="Skill Gap Analysis - AI Job Assistant", layout="wide")

theme_switch()  # dark-mode toggle at the top of the sidebar
render_sidebar()
apply_rtl()
inject_design()  # after render_sidebar so it overrides src/theme.py

lang = st.session_state.get("lang", "en")

page_header(T("gap_title"), T("gap_intro"), step=3, lang=lang, icon="🧭")

resume_data = st.session_state.get("resume_data")
job_data = st.session_state.get("job_data")
match = st.session_state.get("match_result")

if not (resume_data and job_data and match):
    st.warning(T("err_need_analysis"))
    st.page_link("pages/2_Job_Matching.py", label=T("go_to_matching"))
    st.stop()


def importance_for(requirement: str) -> str:
    if requirement == "required":
        return T("importance_high")
    return T("importance_medium")


def build_gap_items() -> list[SkillGapItem]:
    items: list[SkillGapItem] = []
    role = job_data.title or "this role"
    for requirement, skill_list in (
        ("required", match.missing_required),
        ("preferred", match.missing_preferred),
    ):
        for name in skill_list:
            entry = get_skill_entry(name)
            why = (
                f"Listed as {translate(requirement, lang)} for '{role}'."
                + (f" {entry.learning_hints}" if entry and entry.learning_hints else "")
            )
            learning = (
                entry.learning_hints
                if entry and entry.learning_hints
                else "Study the fundamentals, then apply them in a small, documented project."
            )
            items.append(
                SkillGapItem(
                    skill=name,
                    category=entry.category if entry else "other",
                    category_label=entry.category_label if entry else "Other",
                    requirement=requirement,
                    importance=importance_for(requirement),
                    why_it_matters=why,
                    learning_direction=learning,
                )
            )
    # Required gaps first, then by importance label.
    items.sort(key=lambda i: (i.requirement != "required", i.importance, i.skill))
    return items


gap_items = build_gap_items()
req_items = [g for g in gap_items if g.requirement == "required"]
pref_items = [g for g in gap_items if g.requirement == "preferred"]

stat_cards(
    [
        (T("gap_req_title"), len(req_items), "miss" if req_items else "ok"),
        (T("gap_pref_title"), len(pref_items), "warn" if pref_items else "ok"),
        (T("matched_skills"), len(match.matched_required) + len(match.matched_preferred), "ok"),
    ]
)

# --- Required gaps: one card per skill -------------------------------------------
st.subheader(f"🔴 {T('gap_req_title')}")
if req_items:
    for g in req_items:
        with st.container(border=True):
            head, tag = st.columns([4, 1])
            with head:
                st.markdown(f"#### {g.skill}")
                st.markdown(
                    chips([g.category_label], "neutral"), unsafe_allow_html=True
                )
            with tag:
                st.markdown(
                    chips([g.importance], "miss"), unsafe_allow_html=True
                )
            st.markdown(f"**{T('col_why')}**")
            st.markdown(g.why_it_matters)
            st.markdown(f"**{T('col_learning')}**")
            st.success(g.learning_direction)
else:
    st.success(T("no_gaps"))

# --- Preferred gaps: compact table ----------------------------------------------
if pref_items:
    st.subheader(f"🟡 {T('gap_pref_title')}")
    pref_df = pd.DataFrame(
        [
            {
                T("col_skill"): g.skill,
                T("col_category"): g.category_label,
                T("col_importance"): g.importance,
                T("col_learning"): g.learning_direction,
            }
            for g in pref_items
        ]
    )
    st.dataframe(pref_df, use_container_width=True, hide_index=True)

st.info(T("honesty_note"))

# --- Transferable ---------------------------------------------------------------
if match.transferable:
    st.subheader(f"🔁 {T('transferable_title')}")
    st.caption(T("transferable_note"))
    for t in match.transferable:
        with st.container(border=True):
            st.markdown(
                T(
                    "transferable_item",
                    missing=t.missing_skill,
                    related=", ".join(t.related_skills_you_have),
                )
            )

# --- Skills needing confirmation -------------------------------------------------
inferred = resume_data.skills_inferred
if inferred:
    st.subheader(f"❓ {T('confirm_title')}")
    st.caption(T("confirm_body"))
    st.markdown(chips([s.name for s in inferred], "warn"), unsafe_allow_html=True)
    st.page_link("pages/1_Resume_Analysis.py", label=T("edit_skills"))

# --- Evidence for matched skills --------------------------------------------------
if match.matched_required or match.matched_preferred:
    with st.expander(f"🔎 {T('gap_evidence_title')}"):
        explicit_map = {s.name: s for s in resume_data.skills_explicit}
        for name in match.matched_required + match.matched_preferred:
            skill = explicit_map.get(name)
            if not skill:
                continue
            st.markdown(f"**{skill.name}**")
            for ev in skill.evidence[:3]:
                section = f" `[{ev.section}]`" if ev.section else ""
                st.markdown(f"-{section} \"{ev.snippet}\"")

# --- CSV download ------------------------------------------------------------------
st.divider()
if gap_items:
    csv_df = pd.DataFrame(
        [
            {
                "skill": g.skill,
                "category": g.category_label,
                "requirement": g.requirement,
                "importance": g.importance,
                "why_it_matters": g.why_it_matters,
                "learning_direction": g.learning_direction,
            }
            for g in gap_items
        ]
    )
    st.download_button(
        f"⬇️ {T('download_gap_csv')}",
        data=csv_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"skill_gaps_{timestamp_slug()}.csv",
        mime="text/csv",
        key="gap_csv_download",
        use_container_width=True,
    )