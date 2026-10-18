"""Centralized configuration for AI Job Assistant.

All tunable values live here so that business logic never reads
``os.environ`` directly. Optional LLM credentials are read from a local
``.env`` file (see ``.env.example``) and are never hard-coded.

Usage:
    from config.settings import SETTINGS, ScoringWeights
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# Project layout -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
SKILLS_TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
FONTS_DIR = DATA_DIR / "fonts"


def _env_float(key: str, default: float) -> float:
    """Read a float from the environment, falling back on bad values."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Read an int from the environment, falling back on bad values."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ScoringWeights:
    """Weights of the four compatibility-score components.

    The defaults are documented in the README; they can be overridden via
    environment variables and are re-normalized so they always sum to 1.0.
    """

    required_skills: float = 0.45
    preferred_skills: float = 0.15
    semantic_similarity: float = 0.25
    evidence: float = 0.15

    def normalized(self) -> "ScoringWeights":
        """Return a copy whose weights sum to 1.0 (safe on zero sum)."""
        total = (
            self.required_skills
            + self.preferred_skills
            + self.semantic_similarity
            + self.evidence
        )
        if total <= 0:
            return replace(
                self,
                required_skills=0.25,
                preferred_skills=0.25,
                semantic_similarity=0.25,
                evidence=0.25,
            )
        return replace(
            self,
            required_skills=self.required_skills / total,
            preferred_skills=self.preferred_skills / total,
            semantic_similarity=self.semantic_similarity / total,
            evidence=self.evidence / total,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "semantic_similarity": self.semantic_similarity,
            "evidence": self.evidence,
        }

    @classmethod
    def from_env(cls) -> "ScoringWeights":
        return cls(
            required_skills=_env_float("WEIGHT_REQUIRED_SKILLS", 0.45),
            preferred_skills=_env_float("WEIGHT_PREFERRED_SKILLS", 0.15),
            semantic_similarity=_env_float("WEIGHT_SEMANTIC_SIMILARITY", 0.25),
            evidence=_env_float("WEIGHT_EVIDENCE", 0.15),
        ).normalized()


@dataclass(frozen=True)
class LLMConfig:
    """Optional LLM provider settings (OpenAI-compatible chat API)."""

    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "").strip())
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        ).strip().rstrip("/")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "").strip())
    timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 30.0))
    temperature: float = 0.2
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 900))

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
            model=os.getenv("LLM_MODEL", "").strip(),
            timeout=_env_float("LLM_TIMEOUT", 30.0),
            max_tokens=_env_int("LLM_MAX_TOKENS", 900),
        )

    def is_configured(self) -> bool:
        """True only when every required credential field is present."""
        return bool(self.api_key and self.base_url and self.model)

    def safe_description(self) -> str:
        """Human-readable description that never exposes the API key."""
        return f"base_url={self.base_url} | model={self.model or 'not set'} | key={'set' if self.api_key else 'missing'}"


@dataclass(frozen=True)
class AppConfig:
    """General application settings."""

    app_name: str = "AI Job Assistant"
    app_version: str = "1.0.0"

    # Uploads
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 10))
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
    )
    embedding_batch_size: int = 32
    # Long documents are split into paragraph chunks before embedding because
    # MiniLM truncates input at 256 word-piece tokens.
    long_text_chunk_words: int = 180

    # Skill extraction
    semantic_infer_threshold: float = 0.55
    semantic_infer_max: int = 15

    # Input validation thresholds
    min_resume_chars: int = 200
    min_job_chars: int = 150
    scanned_pdf_chars_per_page: int = 40

    # Evidence / retrieval
    evidence_snippet_width: int = 120
    rag_top_k: int = 3
    rag_min_score: float = 0.20

    # Reports
    report_max_context_chars: int = 6000

    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights.from_env)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)


SETTINGS = AppConfig()
