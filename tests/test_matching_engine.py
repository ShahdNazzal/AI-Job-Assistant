"""Tests for the matching engine using deterministic dummy embeddings."""

from __future__ import annotations

import pytest

from src.exceptions import ModelNotAvailableError
from src.job_parser import parse_job_description
from src.matching_engine import MatchingEngine
from src.models import ResumeData
from src.resume_parser import parse_resume_text
from src.utils import count_words


def _engine(dummy_embedder) -> MatchingEngine:
    return MatchingEngine(embedding_service=dummy_embedder)


class TestComponents:
    def test_coverage_math(self, dummy_embedder, strong_resume_text, matching_job_text):
        resume = parse_resume_text(strong_resume_text)
        job = parse_job_description(matching_job_text)
        # Confirmed skills cover Python, Scikit-learn, SQL, Docker, FastAPI;
        # the job requires Python + Machine Learning + Scikit-learn + SQL + Docker.
        result = _engine(dummy_embedder).compute_match(resume, job)
        required = job.required_skill_names
        expected_cov = len(set(required) & {"Python", "Machine Learning", "Scikit-learn", "SQL", "Docker"}) / len(
            required
        )
        req_comp = next(c for c in result.components if c.key == "required_skills")
        assert req_comp.value == pytest.approx(expected_cov)

    def test_overall_in_range(self, dummy_embedder, strong_resume_text, matching_job_text):
        resume = parse_resume_text(strong_resume_text)
        job = parse_job_description(matching_job_text)
        result = _engine(dummy_embedder).compute_match(resume, job)
        assert result.overall is not None
        assert 0.0 <= result.overall <= 1.0

    def test_matched_and_missing_lists(self, dummy_embedder, strong_resume_text, matching_job_text):
        resume = parse_resume_text(strong_resume_text)
        job = parse_job_description(matching_job_text)
        result = _engine(dummy_embedder).compute_match(resume, job)
        assert "Python" in result.matched_required
        assert "Machine Learning" in result.matched_required  # "Machine learning engineer" in summary
        assert result.missing_required == []
        assert "FastAPI" in result.matched_preferred

    def test_semantic_component_available(self, dummy_embedder, strong_resume_text, matching_job_text):
        resume = parse_resume_text(strong_resume_text)
        job = parse_job_description(matching_job_text)
        result = _engine(dummy_embedder).compute_match(resume, job)
        sem = next(c for c in result.components if c.key == "semantic_similarity")
        assert sem.available and 0.0 <= sem.value <= 1.0


class TestHonestHandling:
    def test_no_preferred_skills_redistributes(self, dummy_embedder, strong_resume_text):
        job = parse_job_description(
            "Job Title: Python Developer\n\n"
            "Responsibilities\n- Write backend services and data tooling.\n\n"
            "Required qualifications\n"
            "- Python programming experience in production.\n"
            "- SQL and relational databases.\n"
        )
        resume = parse_resume_text(strong_resume_text)
        result = _engine(dummy_embedder).compute_match(resume, job)
        pref = next(c for c in result.components if c.key == "preferred_skills")
        assert not pref.available
        assert pref.note
        available_weight = sum(c.weight for c in result.components if c.available)
        assert available_weight == pytest.approx(1.0)
        assert result.overall is not None

    def test_no_required_skills_redistributes(self, dummy_embedder, strong_resume_text):
        job = parse_job_description(
            "Job Title: Community Manager\n\n"
            "About the role\nWe need a friendly communicator with great people "
            "skills for our developer community. You will host community events, "
            "write newsletters, and support members across time zones every week."
        )
        resume = parse_resume_text(strong_resume_text)
        result = _engine(dummy_embedder).compute_match(resume, job)
        req_comp = next(c for c in result.components if c.key == "required_skills")
        assert not req_comp.available
        assert result.overall is not None

    def test_empty_confirmed_skills_zero_coverage(self, dummy_embedder, matching_job_text):
        resume = parse_resume_text("Just some unrelated text without skills. " * 10)
        job = parse_job_description(matching_job_text)
        result = _engine(dummy_embedder).compute_match(resume, job, confirmed_skills=[])
        req_comp = next(c for c in result.components if c.key == "required_skills")
        assert req_comp.value == 0.0
        assert result.missing_required

    def test_weights_respected(self, dummy_embedder, strong_resume_text, matching_job_text):
        from config.settings import ScoringWeights

        resume = parse_resume_text(strong_resume_text)
        job = parse_job_description(matching_job_text)
        weights = ScoringWeights(
            required_skills=1.0, preferred_skills=0.0,
            semantic_similarity=0.0, evidence=0.0,
        )
        result = _engine(dummy_embedder).compute_match(
            resume, job, confirmed_skills=["Python", "Scikit-learn", "SQL", "Docker", "Machine Learning"],
            weights=weights.normalized(),
        )
        assert result.overall == pytest.approx(1.0)


class TestTransferable:
    def test_related_skill_suggestion(self, dummy_embedder):
        resume = parse_resume_text(
            "Tech Skills\nPython, NumPy\n\nWork Experience\nUsed Python daily."
        )
        job = parse_job_description(
            "Job Title: ML Engineer\n\n"
            "About the role\nYou will train tabular models for our product.\n\n"
            "Required qualifications\n"
            "- Solid experience with scikit-learn for modeling.\n"
        )
        result = _engine(dummy_embedder).compute_match(resume, job)
        assert any(t.missing_skill == "Scikit-learn" for t in result.transferable)


class TestMonotonicity:
    def test_strong_beats_weak(self, dummy_embedder, matching_job_text):
        strong = parse_resume_text(
            "Skills\nPython, scikit-learn, SQL, Docker, machine learning\n\n"
            "Work Experience\nBuilt machine learning models with Python, scikit-learn, "
            "SQL and Docker for three years.\n"
        )
        weak = parse_resume_text(
            "Skills\nPainting, pottery, guitar\n\n"
            "Work Experience\nTaught art classes and organized workshops.\n"
        )
        job = parse_job_description(matching_job_text)
        engine = _engine(dummy_embedder)
        strong_result = engine.compute_match(strong, job)
        weak_result = engine.compute_match(weak, job)
        assert strong_result.overall > weak_result.overall

    def test_long_resume_embedding_works(self, dummy_embedder, matching_job_text):
        long_text = ("Python and SQL engineering experience. " * 400)
        resume = parse_resume_text(long_text)
        job = parse_job_description(matching_job_text)
        result = _engine(dummy_embedder).compute_match(resume, job)
        assert result.overall is not None
        assert count_words(resume.raw_text) > 1000


class TestEvidenceHonesty:
    def test_evidence_uses_work_sections(self, dummy_embedder):
        resume = parse_resume_text(
            "Skills\nPython, SQL, Docker, scikit-learn, FastAPI\n\n"
            "Work Experience\nUsed Python, SQL, Docker, scikit-learn and FastAPI in production."
        )
        job = parse_job_description(
            "Job Title: Machine Learning Engineer\n\n"
            "About the role\nYou will train and ship models for our platform.\n\n"
            "Required qualifications\n"
            "- Python.\n- SQL.\n- Docker.\n- scikit-learn.\n"
        )
        result = _engine(dummy_embedder).compute_match(resume, job)
        evidence = next(c for c in result.components if c.key == "evidence")
        assert evidence.available
        assert evidence.value is not None and evidence.value > 0.5

    def test_manually_added_skills_get_no_evidence(self, dummy_embedder):
        resume = parse_resume_text("Skills\nPython\n")
        job = parse_job_description(
            "Job Title: Platform Engineer\n\n"
            "About the role\nYou will maintain our internal tooling and "
            "developer environments.\n\n"
            "Required qualifications\n- Python.\n- Docker.\n"
        )
        result = _engine(dummy_embedder).compute_match(
            resume, job, confirmed_skills=["Python", "Docker"]
        )
        evidence = next(c for c in result.components if c.key == "evidence")
        # Python evidenced in skills section only (not work), Docker manual:
        # evidence ratio should be 0 -> discounted or low.
        assert evidence.value is not None and evidence.value < 0.6
