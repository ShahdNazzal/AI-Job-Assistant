"""Page 5 - Model Evaluation: live metrics from bundled labeled datasets."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.exceptions import ModelNotAvailableError
from src.evaluation import (
    EvaluationDataError,
    evaluate_matching,
    evaluate_skill_extraction,
)
from src.ui_helpers import T, apply_rtl, render_sidebar

st.set_page_config(page_title="Model Evaluation - AI Job Assistant", layout="wide")

render_sidebar()
apply_rtl()

st.title(T("eval_title"))
st.caption(T("eval_intro"))

# --- Skill extraction evaluation ----------------------------------------------------
st.subheader(T("nav_evaluation"))
st.markdown(f"**{T('run_skill_eval')}**")
st.caption(T("skill_eval_desc"))

if st.button(T("run_skill_eval"), key="run_skill_eval_btn", type="primary"):
    try:
        with st.spinner(T("thinking")):
            result = evaluate_skill_extraction()
    except EvaluationDataError as exc:
        st.error(str(exc))
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(T("precision"), f"{result['precision'] * 100:.1f}%")
        m2.metric(T("recall"), f"{result['recall'] * 100:.1f}%")
        m3.metric(T("f1"), f"{result['f1'] * 100:.1f}%")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("tp"), result["tp"])
        c2.metric(T("fp"), result["fp"])
        c3.metric(T("fn"), result["fn"])
        c4.metric(T("samples_label"), result["samples"])

        with st.expander(T("per_sample")):
            for detail in result["details"]:
                st.markdown(f"**{detail['label']}**")
                st.caption(
                    T(
                        "expected_vs_got",
                        expected=", ".join(detail["expected"]) or "-",
                        missing=", ".join(detail["missing"]) or "-",
                        unexpected=", ".join(detail["unexpected"]) or "-",
                    )
                )

# --- Matching evaluation ---------------------------------------------------------------
st.divider()
st.markdown(f"**{T('run_match_eval')}**")
st.caption(T("match_eval_desc"))

if st.button(T("run_match_eval"), key="run_match_eval_btn", type="primary"):
    try:
        with st.spinner(T("computing")):
            result = evaluate_matching()
    except ModelNotAvailableError:
        st.warning(T("eval_needs_model"))
    except EvaluationDataError as exc:
        st.error(str(exc))
    else:
        if result["all_passed"]:
            st.success(f"**{T('passed')}** - {result['ordering']['message']}")
        else:
            st.error(
                f"**{T('failed')}** - {result['ordering'].get('message', '')}"
            )

        cases_df = pd.DataFrame(
            [
                {
                    "case": case.get("name", ""),
                    "tier": case.get("tier", ""),
                    "overall": (
                        f"{case['overall'] * 100:.0f}%"
                        if case.get("overall") is not None
                        else "N/A"
                    ),
                    "passed": T("passed") if case.get("passed") else T("failed"),
                }
                for case in result["cases"]
            ]
        )
        st.dataframe(cases_df, use_container_width=True, hide_index=True)

        with st.expander(T("expander_details")):
            for case in result["cases"]:
                st.markdown(
                    f"**{case.get('name', '')}** - "
                    + (T("passed") if case.get("passed") else T("failed"))
                )
                if case.get("error"):
                    st.caption(case["error"])
                    continue
                for check in case.get("checks", []):
                    marker = "OK" if check["passed"] else "XX"
                    st.markdown(f"- [{marker}] {check['check']}")

# --- Unit tests note --------------------------------------------------------------------
st.divider()
st.info(T("unit_tests_note"))

# --- Dataset limitations ----------------------------------------------------------------
with st.expander(T("dataset_limits_title")):
    st.markdown(T("dataset_limits_body"))

st.caption(T("eval_disclaimer"))
