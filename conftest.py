"""Pytest configuration: project root on sys.path, shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

# Keep tests independent of local secrets.
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("LLM_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("LLM_MODEL", "test-model")

import pytest

from src.embedding_service import DummyEmbeddingService


@pytest.fixture()
def dummy_embedder() -> DummyEmbeddingService:
    return DummyEmbeddingService()


@pytest.fixture()
def strong_resume_text() -> str:
    return (
        "Alex Rivera\nalex@example.com\n\n"
        "Professional Summary\n"
        "Machine learning engineer with Python experience.\n\n"
        "Technical Skills\n"
        "Python, scikit-learn, SQL, Docker, FastAPI\n\n"
        "Work Experience\n"
        "ML Engineer - Acme (2023 - present)\n"
        "- Built models with Python and scikit-learn.\n"
        "- Wrote SQL and deployed with Docker.\n"
    )


@pytest.fixture()
def matching_job_text() -> str:
    return (
        "Job Title: Machine Learning Engineer\n\n"
        "Responsibilities\n"
        "- Build machine learning models in Python.\n\n"
        "Required qualifications\n"
        "- Python programming.\n"
        "- Experience with scikit-learn.\n"
        "- SQL.\n"
        "- Docker.\n\n"
        "Preferred qualifications\n"
        "- FastAPI.\n"
    )
