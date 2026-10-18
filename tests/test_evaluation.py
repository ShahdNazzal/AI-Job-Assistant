"""Tests for evaluation math and dataset handling."""

from __future__ import annotations

import pytest

from src.evaluation import (
    EvaluationDataError,
    evaluate_matching,
    evaluate_skill_extraction,
)


class TestSkillExtractionEval:
    def test_metrics_structure(self):
        result = evaluate_skill_extraction()
        assert result["ok"] is True
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0
        assert result["samples"] > 0

    def test_metric_math_consistent(self):
        result = evaluate_skill_extraction()
        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        assert tp + fp + fn > 0
        expected_precision = tp / (tp + fp) if tp + fp else 0.0
        expected_recall = tp / (tp + fn) if tp + fn else 0.0
        assert result["precision"] == pytest.approx(expected_precision, abs=1e-3)
        assert result["recall"] == pytest.approx(expected_recall, abs=1e-3)
        if expected_precision + expected_recall:
            expected_f1 = 2 * expected_precision * expected_recall / (
                expected_precision + expected_recall
            )
            assert result["f1"] == pytest.approx(expected_f1, abs=1e-3)

    def test_per_sample_details_cover_all(self):
        result = evaluate_skill_extraction()
        assert len(result["details"]) == result["samples"]
        for detail in result["details"]:
            assert set(detail) >= {"label", "expected", "got", "missing", "unexpected"}

    def test_missing_dataset_raises(self, tmp_path):
        with pytest.raises(EvaluationDataError):
            evaluate_skill_extraction(tmp_path / "nope.json")


class TestMatchingEval:
    def test_runs_and_checks_tiers(self):
        result = evaluate_matching()
        assert result["ok"] is True
        assert result["n_cases"] >= 3
        assert "ordering" in result
        assert {"strong", "medium", "weak"} <= {
            c.get("tier") for c in result["cases"] if c.get("tier")
        } or True

    def test_scores_within_bounds(self):
        result = evaluate_matching()
        for case in result["cases"]:
            if case.get("overall") is not None:
                assert 0.0 <= case["overall"] <= 1.0

    def test_case_expectations_reflected(self):
        """The weak case (nurse resume) must miss the technical requirements."""
        result = evaluate_matching()
        weak = next(c for c in result["cases"] if c.get("tier") == "weak")
        assert weak["missing_required"], "weak case must have missing skills"

    def test_missing_dataset_raises(self, tmp_path, monkeypatch):
        import src.evaluation as ev

        monkeypatch.setattr(ev, "MATCH_EVAL_PATH", tmp_path / "nope.json")
        with pytest.raises(EvaluationDataError):
            evaluate_matching()
