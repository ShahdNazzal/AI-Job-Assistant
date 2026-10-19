"""Page 5 - Model Evaluation: live metrics from bundled labeled datasets."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.design import chips, inject_design, theme_switch, page_header, stat_cards
from src.exceptions import ModelNotAvailableError
from src.evaluation import (
    EvaluationDataError,
    evaluate_matching,
    evaluate_skill_extraction,
)
from src.ui_helpers import T, apply_rtl, render_sidebar

st.set_page_config(page_title="Model Evaluation - AI Job Assistant", layout="wide")

theme_switch()  # dark-mode toggle at the top of the sidebar
render_sidebar()
apply_rtl()
inject_design()  # after render_sidebar so it overrides src/theme.py

lang = st.session_state.get("lang", "en")

page_header(T("eval_title"), T("eval_intro"), lang=lang, icon="🧪")

# --- Skill extraction evaluation ----------------------------------------------------
with st.container(border=True):
    st.subheader(f"🏷️ {T('run_skill_eval')}")
    st.caption(T("skill_eval_desc"))

    if st.button(
        f"▶ {T('run_skill_eval')}", key="run_skill_eval_btn", type="primary"
    ):
        try:
            with st.spinner(T("thinking")):
                result = evaluate_skill_extraction()
        except EvaluationDataError as exc:
            st.error(str(exc))
        else:
            stat_cards(
                [
                    (T("precision"), f"{result['precision'] * 100:.1f}%", "ok"),
                    (T("recall"), f"{result['recall'] * 100:.1f}%", "ok"),
                    (T("f1"), f"{result['f1'] * 100:.1f}%", ""),
                ]
            )
            stat_cards(
                [
                    (T("tp"), result["tp"], "ok"),
                    (T("fp"), result["fp"], "warn"),
                    (T("fn"), result["fn"], "miss"),
                    (T("samples_label"), result["samples"], ""),
                ]
            )

            with st.expander(f"📄 {T('per_sample')}"):
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
with st.container(border=True):
    st.subheader(f"🎯 {T('run_match_eval')}")
    st.caption(T("match_eval_desc"))

    if st.button(
        f"▶ {T('run_match_eval')}", key="run_match_eval_btn", type="primary"
    ):
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

            passed_n = sum(1 for c in result["cases"] if c.get("passed"))
            stat_cards(
                [
                    (T("passed"), passed_n, "ok"),
                    (T("failed"), len(result["cases"]) - passed_n, "miss" if passed_n != len(result["cases"]) else ""),
                ]
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
                        "passed": ("✅ " + T("passed"))
                        if case.get("passed")
                        else ("❌ " + T("failed")),
                    }
                    for case in result["cases"]
                ]
            )
            st.dataframe(cases_df, use_container_width=True, hide_index=True)

            with st.expander(f"🔬 {T('expander_details')}"):
                for case in result["cases"]:
                    st.markdown(
                        f"**{case.get('name', '')}** - "
                        + (T("passed") if case.get("passed") else T("failed"))
                    )
                    if case.get("error"):
                        st.caption(case["error"])
                        continue
                    for check in case.get("checks", []):
                        marker = "✅" if check["passed"] else "❌"
                        st.markdown(f"- {marker} {check['check']}")

# --- Unit tests note --------------------------------------------------------------------
st.info(T("unit_tests_note"))

# --- Dataset limitations ----------------------------------------------------------------
with st.expander(f"⚠️ {T('dataset_limits_title')}"):
    st.markdown(T("dataset_limits_body"))

st.caption(T("eval_disclaimer"))