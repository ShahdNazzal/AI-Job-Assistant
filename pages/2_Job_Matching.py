"""Page 2 - Job Matching: parse the JD, compute and visualize the match."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import SETTINGS, ScoringWeights
from src.exceptions import JobDescriptionError, ModelNotAvailableError
from src.i18n import rtl_wrap
from src.job_parser import parse_job_description
from src.matching_engine import MatchingEngine
from src.recommendation_engine import generate_recommendations
from src.ui_helpers import (
    T,
    apply_rtl,
    demo_badge,
    load_demo_job_text,
    render_sidebar,
    score_interpretation_key,
)

st.set_page_config(page_title="Job Matching - AI Job Assistant", layout="wide")

render_sidebar()
apply_rtl()

st.title(T("matching_title"))
st.caption(T("matching_intro"))

resume_data = st.session_state.get("resume_data")
if not resume_data:
    st.warning(T("err_need_resume"))
    st.page_link("pages/1_Resume_Analysis.py", label=T("go_to_resume"))
    st.stop()

lang = st.session_state.get("lang", "en")

# --- Weight controls -----------------------------------------------------------
with st.expander(T("adjust_weights")):
    st.caption(T("adjust_weights_help"))
    base = SETTINGS.scoring_weights
    c1, c2 = st.columns(2)
    with c1:
        w_req = st.slider(T("w_required"), 0.0, 0.8, base.required_skills, 0.05, key="w_required")
        w_sem = st.slider(T("w_semantic"), 0.0, 0.8, base.semantic_similarity, 0.05, key="w_semantic")
    with c2:
        w_pref = st.slider(T("w_preferred"), 0.0, 0.8, base.preferred_skills, 0.05, key="w_preferred")
        w_evid = st.slider(T("w_evidence"), 0.0, 0.8, base.evidence, 0.05, key="w_evidence")


def _current_weights() -> ScoringWeights:
    return ScoringWeights(
        required_skills=st.session_state.get("w_required", 0.45),
        preferred_skills=st.session_state.get("w_preferred", 0.15),
        semantic_similarity=st.session_state.get("w_semantic", 0.25),
        evidence=st.session_state.get("w_evidence", 0.15),
    ).normalized()


# --- Job description input ------------------------------------------------------
st.subheader(T("job_label"))
job_text_value = st.session_state.get("job_input", "")
demo_col, _ = st.columns([1, 2])
with demo_col:
    if st.button(T("load_demo_jd"), key="load_demo_jd_btn"):
        st.session_state["job_input"] = load_demo_job_text()
        st.session_state.job_data = None
        st.session_state.match_result = None
        st.session_state.recommendations = None
        st.rerun()

job_text = st.text_area(
    T("job_label"),
    value=job_text_value,
    height=260,
    key="job_input",
    label_visibility="collapsed",
)

analyze_clicked = st.button(T("analyze_btn"), type="primary", key="analyze_btn")

if analyze_clicked:
    if not job_text or not job_text.strip():
        st.warning(
            T("err_job_short", chars=0, min_chars=SETTINGS.min_job_chars)
        )
    else:
        try:
            with st.spinner(T("computing")):
                st.caption(T("model_loading_hint"))
                job_data = parse_job_description(job_text)
                st.session_state.job_data = job_data
                engine = MatchingEngine()
                match = engine.compute_match(
                    resume_data,
                    job_data,
                    confirmed_skills=st.session_state.get("confirmed_skills") or [],
                    weights=_current_weights(),
                )
                st.session_state.match_result = match
                st.session_state.recommendations = generate_recommendations(
                    resume_data,
                    job_data,
                    match,
                    st.session_state.get("confirmed_skills") or [],
                )
        except JobDescriptionError:
            st.error(
                T(
                    "err_job_short",
                    chars=len(job_text or ""),
                    min_chars=SETTINGS.min_job_chars,
                )
            )
        except ModelNotAvailableError as exc:
            st.error(str(exc))

job_data = st.session_state.get("job_data")
match = st.session_state.get("match_result")
if not job_data or not match:
    st.stop()

# Clearly label demo data when the textarea still holds the bundled sample.
if job_text and job_text.strip() == load_demo_job_text().strip():
    demo_badge()

# --- Job requirements overview --------------------------------------------------
st.divider()
st.subheader(T("job_overview"))

title_cell = job_data.title or T("not_detected")
if job_data.title and job_data.title_confidence == "low":
    title_cell += f" _{T('job_title_low')}_"
st.markdown(rtl_wrap(f"**{T('job_title_label')}:** {title_cell}", lang), unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"**{T('exp_label')}:** "
                + (T("years_tpl", years=job_data.experience_years_min) if job_data.experience_years_min else T("none_found")))
with col_b:
    st.markdown(f"**{T('education_label')}:** " + (", ".join(job_data.education[:3]) or T("none_found")))
with col_c:
    st.markdown(f"**{T('certs_label')}:** " + (", ".join(job_data.certifications[:3]) or T("none_found")))

st.markdown(f"**{T('req_skills_label')}** ({len(job_data.required_skills)})")
st.markdown(rtl_wrap(" ".join(f"`{s.name}`" for s in job_data.required_skills) or T("none_found"), lang), unsafe_allow_html=True)
st.markdown(f"**{T('pref_skills_label')}** ({len(job_data.preferred_skills)})")
st.markdown(rtl_wrap(" ".join(f"`{s.name}`" for s in job_data.preferred_skills) or T("none_found"), lang), unsafe_allow_html=True)
st.caption(T("req_pref_note"))

if job_data.keywords:
    st.markdown(f"**{T('keywords_label')}:** " + ", ".join(job_data.keywords))

if job_data.responsibilities:
    with st.expander(T("resp_label")):
        for item in job_data.responsibilities:
            st.markdown(f"- {item}")

st.divider()

# --- Dashboard ------------------------------------------------------------------
st.subheader(T("score_section"))

interp_key = score_interpretation_key(match.overall)
metric_col, gauge_col = st.columns([1, 2])
with metric_col:
    overall_display = (
        f"{match.overall * 100:.0f}%" if match.overall is not None else "N/A"
    )
    st.metric(T("overall_score"), overall_display)
    if interp_key:
        st.info(T(interp_key))
    for note in match.notes:
        st.caption(f"- {note}")
with gauge_col:
    import plotly.graph_objects as go

    if match.overall is not None:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(match.overall * 100, 1),
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 40], "color": "#FEE2E2"},
                        {"range": [40, 65], "color": "#FEF3C7"},
                        {"range": [65, 100], "color": "#DCFCE7"},
                    ],
                },
            )
        )
        fig.update_layout(
            height=240,
            margin={"t": 30, "b": 10, "l": 30, "r": 30},
            title={"text": T("overall_score")},
        )
        st.plotly_chart(fig, use_container_width=True, key="gauge_overall")

# Component scores chart
import pandas as pd
import plotly.express as px

comp_rows = []
for comp in match.components:
    comp_rows.append(
        {
            "component": T(
                {
                    "required_skills": "w_required",
                    "preferred_skills": "w_preferred",
                    "semantic_similarity": "w_semantic",
                    "evidence": "w_evidence",
                }[comp.key]
            ),
            "score": (comp.value or 0) * 100 if comp.available else 0,
            "status": "ok" if comp.available else "unavailable",
        }
    )
comp_df = pd.DataFrame(comp_rows)
fig_components = px.bar(
    comp_df,
    x="score",
    y="component",
    orientation="h",
    color="status",
    color_discrete_map={"ok": "#2563EB", "unavailable": "#9CA3AF"},
    text_auto=".0f",
    range_x=[0, 100],
    title=T("component_chart"),
    labels={"score": "%", "component": ""},
    height=260,
)
fig_components.update_layout(margin={"t": 40, "b": 10, "l": 10, "r": 10})
st.plotly_chart(fig_components, use_container_width=True, key="components_chart")

for comp in match.components:
    if not comp.available:
        st.caption(f"**{comp.label}**: {comp.note}")

# Skill gap chart
gap_df = pd.DataFrame(
    {
        "type": [T("required"), T("required"), T("preferred"), T("preferred")],
        "state": [T("matched_skills"), T("missing_skills")] * 2,
        "count": [
            len(match.matched_required),
            len(match.missing_required),
            len(match.matched_preferred),
            len(match.missing_preferred),
        ],
    }
)
fig_gap = px.bar(
    gap_df,
    x="type",
    y="count",
    color="state",
    barmode="group",
    color_discrete_map={
        T("matched_skills"): "#16A34A",
        T("missing_skills"): "#DC2626",
    },
    title=T("gap_chart"),
    labels={"count": "", "type": ""},
    height=280,
)
fig_gap.update_layout(margin={"t": 40, "b": 10, "l": 10, "r": 10})
st.plotly_chart(fig_gap, use_container_width=True, key="gap_chart")

# Matched / missing lists
left_col, right_col = st.columns(2)
with left_col:
    st.markdown(f"**{T('matched_skills')}**")
    st.markdown(
        rtl_wrap(" ".join(f"`{s}`" for s in match.matched_required) or T("none_found"), lang),
        unsafe_allow_html=True,
    )
with right_col:
    st.markdown(f"**{T('missing_skills')}**")
    st.markdown(
        rtl_wrap(" ".join(f"`{s}`" for s in match.missing_required) or T("none_found"), lang),
        unsafe_allow_html=True,
    )

if match.transferable:
    with st.expander(T("transferable_title")):
        for t in match.transferable:
            st.markdown(
                T(
                    "transferable_item",
                    missing=t.missing_skill,
                    related=", ".join(t.related_skills_you_have),
                )
            )
            st.caption(t.rationale)

with st.expander(T("score_notes")):
    st.markdown(T("weights_used", w_req=match.weights_used.get("required_skills", 0),
                  w_pref=match.weights_used.get("preferred_skills", 0),
                  w_sem=match.weights_used.get("semantic_similarity", 0),
                  w_evid=match.weights_used.get("evidence", 0)))
    st.markdown(f"- {T('disclaimer_matching')}")
    st.markdown(f"- {T('semantic_limitation')}")
    st.markdown(f"- {T('reanalyze_hint')}")

st.caption(T("reanalyze_hint"))
