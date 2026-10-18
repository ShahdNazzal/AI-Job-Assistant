"""Tests for job-description parsing."""

from __future__ import annotations

import pytest

from src.exceptions import JobDescriptionError
from src.job_parser import parse_job_description

FULL_JD = """Job Title: Junior AI Engineer

Responsibilities
- Develop machine learning models for text classification.
- Serve models through REST APIs.

Required qualifications
- Bachelor's degree in Computer Science.
- 2+ years of experience with Python.
- Experience with scikit-learn and SQL.

Preferred qualifications
- Exposure to PyTorch and Docker.
- Knowledge of RAG concepts.
"""

HEADING_LESS_JD = """We are hiring a data scientist to join our analytics team.
You will build dashboards and models. You need Python and SQL experience
with at least one year of practice. Nice extras: Docker and Git.
"""


class TestFullJD:
    def test_title_detected(self):
        job = parse_job_description(FULL_JD)
        assert job.title == "Junior AI Engineer"
        assert job.title_confidence == "high"

    def test_required_vs_preferred_split(self):
        job = parse_job_description(FULL_JD)
        req = {s.name for s in job.required_skills}
        pref = {s.name for s in job.preferred_skills}
        assert "Python" in req
        assert "Scikit-learn" in req
        assert "SQL" in req
        assert "PyTorch" in pref
        assert "Retrieval-Augmented Generation" in pref
        assert "Docker" in pref

    def test_experience_years(self):
        job = parse_job_description(FULL_JD)
        assert job.experience_years_min == 2

    def test_education_detected(self):
        job = parse_job_description(FULL_JD)
        assert any("bachelor" in e.lower() for e in job.education)

    def test_responsibilities_extracted(self):
        job = parse_job_description(FULL_JD)
        assert any("machine learning" in r.lower() for r in job.responsibilities)

    def test_keywords_present(self):
        job = parse_job_description(FULL_JD)
        assert isinstance(job.keywords, list)


class TestValidation:
    def test_too_short_raises(self):
        with pytest.raises(JobDescriptionError):
            parse_job_description("Hiring dev. Apply now.")

    def test_empty_raises(self):
        with pytest.raises(JobDescriptionError):
            parse_job_description("")

    def test_no_heading_defaults_to_required(self):
        job = parse_job_description(HEADING_LESS_JD)
        names = {s.name for s in job.required_skills}
        assert "Python" in names
        assert "SQL" in names
        assert "no_required_heading" in job.warnings

    def test_no_skills_warning(self):
        job = parse_job_description(
            "Job Title: Community Manager\n\n"
            "We need a friendly person with great communication. "
            "You will host events and write newsletters for our community members."
        )
        assert "no_skills" in job.warnings

    def test_no_fabrication_empty_fields(self):
        job = parse_job_description(
            "Job Title: Community Manager\n\n"
            "We need a friendly person with great communication. "
            "You will host events and write newsletters for our community members."
        )
        assert job.experience_years_min is None
        assert job.required_skills == []
