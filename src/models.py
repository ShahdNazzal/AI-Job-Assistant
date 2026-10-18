"""Pydantic data models shared across all AI Job Assistant modules."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A short excerpt showing where a skill was found in a document."""

    snippet: str
    section: str | None = None


class ExtractedSkill(BaseModel):
    """A skill detected in a resume or a job description.

    matched_via:
        "explicit" - canonical name matched with clear word boundaries.
        "alias"    - a known alias (e.g. "sklearn") matched.
        "inferred" - semantically related skill suggested by embeddings
                     (low confidence, requires user confirmation).
    """

    name: str
    category: str = "other"
    category_label: str = "Other"
    matched_via: str = "explicit"
    matched_text: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: str = "high"


class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    location: str | None = None


class SectionInfo(BaseModel):
    """One resume section (summary, experience, education, ...)."""

    name: str
    found: bool = False
    text: str = ""


class ResumeData(BaseModel):
    """Structured result of parsing one resume."""

    raw_text: str
    source_type: str = "txt"          # pdf | docx | txt | paste | demo
    source_label: str = "resume"
    name: str | None = None
    name_confidence: str = "none"     # high | low | none
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str | None = None
    summary_confidence: str = "none"
    sections: dict[str, SectionInfo] = Field(default_factory=dict)
    skills_explicit: list[ExtractedSkill] = Field(default_factory=list)
    skills_inferred: list[ExtractedSkill] = Field(default_factory=list)
    word_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    parsed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def section_text(self, names: list[str]) -> str:
        """Concatenate the text of the given sections (missing ones skipped)."""
        parts = [
            self.sections[n].text
            for n in names
            if n in self.sections and self.sections[n].found and self.sections[n].text
        ]
        return "\n".join(parts).strip()

    @property
    def experience_text(self) -> str:
        return self.section_text(["experience", "internships"])

    @property
    def projects_text(self) -> str:
        return self.section_text(["projects"])


class JobData(BaseModel):
    """Structured result of parsing one job description."""

    raw_text: str
    title: str | None = None
    title_confidence: str = "none"
    required_skills: list[ExtractedSkill] = Field(default_factory=list)
    preferred_skills: list[ExtractedSkill] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    experience_years_min: int | None = None
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    required_section_text: str | None = None
    preferred_section_text: str | None = None
    word_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    parsed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def required_skill_names(self) -> list[str]:
        return [s.name for s in self.required_skills]

    @property
    def preferred_skill_names(self) -> list[str]:
        return [s.name for s in self.preferred_skills]


class ScoreComponent(BaseModel):
    """One component of the weighted compatibility score."""

    key: str
    label: str
    value: float | None = None       # 0..1, None => not evaluable
    weight: float = 0.0              # effective weight used (after renormalizing)
    weighted_contribution: float | None = None
    available: bool = True
    note: str = ""


class TransferableSkill(BaseModel):
    """A missing skill for which the candidate has related evidence."""

    missing_skill: str
    related_skills_you_have: list[str] = Field(default_factory=list)
    rationale: str = ""


class MatchResult(BaseModel):
    """Full output of the matching engine for one resume/job pair."""

    overall: float | None = None
    components: list[ScoreComponent] = Field(default_factory=list)
    weights_used: dict[str, float] = Field(default_factory=dict)
    matched_required: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    matched_preferred: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    transferable: list[TransferableSkill] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SkillGapItem(BaseModel):
    """One missing skill with prioritization and learning guidance."""

    skill: str
    category: str = "other"
    category_label: str = "Other"
    requirement: str = "required"     # required | preferred
    importance: str = "Medium"        # High | Medium | Low
    why_it_matters: str = ""
    learning_direction: str = ""


class RecommendationItem(BaseModel):
    """A single actionable recommendation."""

    kind: str = "improvement"         # priority | improvement | learning | interview
    title: str
    detail: str = ""
    priority: str = "Medium"          # High | Medium | Low
    before: str | None = None
    after: str | None = None
    source: str = "rule-based"        # rule-based | llm-draft


class RecommendationSet(BaseModel):
    items: list[RecommendationItem] = Field(default_factory=list)
    generated_with_llm: bool = False


class RetrievedChunk(BaseModel):
    """One knowledge-base chunk returned by the RAG retriever."""

    title: str
    heading: str = ""
    text: str
    score: float = 0.0
