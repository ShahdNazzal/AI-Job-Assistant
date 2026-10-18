"""Tests for report generation: JSON, CSV, Markdown, PDF."""

from __future__ import annotations

import csv
import io
import json

import pytest

from src.embedding_service import DummyEmbeddingService
from src.job_parser import parse_job_description
from src.matching_engine import MatchingEngine
from src.recommendation_engine import generate_recommendations
from src.report_generator import (
    build_report_data,
    generate_csv_report,
    generate_json_report,
    generate_markdown_report,
    generate_pdf_report,
)
from src.resume_parser import parse_resume_text

RESUME = """Dana Haddad
dana@example.com

Professional Summary
Junior data scientist with Python experience.

Technical Skills
Python, sklearn, SQL

Work Experience
Intern - Acme (2024)
- Built models with Python and sklearn.
"""

JOB = """Job Title: Data Scientist

Responsibilities
- Build predictive models and analyses for the business.

Required qualifications
- Python.
- scikit-learn.
- SQL.
- Docker.

Preferred qualifications
- Pandas.
"""


@pytest.fixture(scope="module")
def report_data():
    resume = parse_resume_text(RESUME)
    job = parse_job_description(JOB)
    match = MatchingEngine(embedding_service=DummyEmbeddingService()).compute_match(
        resume, job
    )
    recs = generate_recommendations(resume, job, match, [s.name for s in resume.skills_explicit])
    return build_report_data(resume, job, match, recs, [s.name for s in resume.skills_explicit])


class TestJSON:
    def test_valid_and_complete(self, report_data):
        raw = generate_json_report(report_data)
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["score"]["overall"] is not None
        assert "missing_required" in parsed["skills"]
        assert parsed["tool"].startswith("AI Job Assistant")

    def test_no_secrets(self, report_data):
        raw = generate_json_report(report_data).decode("utf-8")
        assert "api_key" not in raw.lower()
        assert "LLM_API_KEY" not in raw


class TestCSV:
    def test_rows_match_skills(self, report_data):
        raw = generate_csv_report(report_data)
        text = raw.decode("utf-8-sig")
        reader = list(csv.reader(io.StringIO(text)))
        header = reader[0]
        assert header[:4] == ["Skill", "Category", "Requirement", "Status"]
        skills = report_data["skills"]
        expected = (
            len(skills["matched_required"])
            + len(skills["missing_required"])
            + len(skills["matched_preferred"])
            + len(skills["missing_preferred"])
        )
        assert len(reader) - 1 == expected


class TestMarkdown:
    def test_contains_sections(self, report_data):
        md = generate_markdown_report(report_data, lang="en")
        assert "# AI Job Assistant - Analysis Report" in md
        assert "Missing required skills" in md
        assert "heuristic" in md

    def test_arabic_rendering(self, report_data):
        md = generate_markdown_report(report_data, lang="ar")
        assert "تقرير التحليل" in md


class TestPDF:
    def test_english_pdf(self, report_data):
        pdf_bytes = generate_pdf_report(report_data, lang="en")
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 2000

    def test_arabic_pdf_with_shaping(self, report_data):
        pdf_bytes = generate_pdf_report(report_data, lang="ar")
        assert pdf_bytes.startswith(b"%PDF")
        # Arabic text must be present (UTF-16 readable via extraction).
        assert len(pdf_bytes) > 2000

    def test_pdf_extractable_text(self, report_data):
        import fitz

        pdf_bytes = generate_pdf_report(report_data, lang="en")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        assert "AI Job Assistant" in text
        assert "Missing required skills" in text
