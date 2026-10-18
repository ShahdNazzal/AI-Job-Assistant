"""Weighted, transparent resume-to-job matching engine.

Score components (defaults configurable via env/UI):
- required_skills      : fraction of job's required skills covered by the
                         user's confirmed skills (default weight 0.45)
- preferred_skills     : same for preferred skills        (default 0.15)
- semantic_similarity  : cosine similarity between full resume and full
                         job-description embeddings       (default 0.25)
- evidence             : how much of the required-skill coverage is backed
                         by experience/projects sections, blended with the
                         semantic similarity of those sections vs the job
                                                          (default 0.15)

Honesty rules:
- A component that cannot be evaluated (e.g. no preferred skills detected)
  is marked unavailable and its weight is redistributed proportionally -
  it is never silently given a fake value.
- The result is a heuristic indicator, not a hiring probability.
"""

from __future__ import annotations

import numpy as np

from config.settings import SETTINGS, ScoringWeights
from src.exceptions import ModelNotAvailableError
from src.embedding_service import EmbeddingService, get_embedding_service
from src.models import (
    JobData,
    MatchResult,
    ResumeData,
    ScoreComponent,
    TransferableSkill,
)
from src.skill_extractor import canonicalize_skill_names, get_skill_entry
from src.utils import get_logger

logger = get_logger("matching_engine")

_WORK_SECTIONS = ("experience", "internships", "projects")


class MatchingEngine:
    """Compute a documented, configurable resume/job compatibility score."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        self._embedding_service = embedding_service

    @property
    def embedder(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    # --- components -----------------------------------------------------

    def _coverage(
        self, targets: list[str], confirmed: set[str]
    ) -> tuple[list[str], list[str], float | None]:
        matched = [s for s in targets if s in confirmed]
        missing = [s for s in targets if s not in confirmed]
        score = (len(matched) / len(targets)) if targets else None
        return matched, missing, score

    def _semantic_similarity(self, resume: ResumeData, job: JobData) -> float:
        res_vec = self.embedder.embed_long_text(resume.raw_text)
        job_vec = self.embedder.embed_long_text(job.raw_text)
        return EmbeddingService.similarity(res_vec, job_vec)

    def _evidence_score(
        self,
        resume: ResumeData,
        job: JobData,
        matched_required: list[str],
        confirmed: set[str],
    ) -> tuple[float | None, list[str]]:
        """Blend (a) share of matched required skills evidenced in work
        sections with (b) semantic similarity of work sections vs the job."""
        notes: list[str] = []
        explicit_map = {s.name: s for s in resume.skills_explicit}

        ratio: float | None = None
        if matched_required:
            evidenced = 0
            for name in matched_required:
                skill = explicit_map.get(name)
                if not skill:
                    continue  # manually added by user: no extractor evidence
                if any(
                    (ev.section or "").lower() in _WORK_SECTIONS
                    for ev in skill.evidence
                ):
                    evidenced += 1
            ratio = evidenced / len(matched_required)
        elif confirmed:
            notes.append(
                "Evidence ratio skipped: no required skills were matched, so "
                "there is nothing to evidence."
            )

        work_text = "\n".join(
            t for t in (resume.experience_text, resume.projects_text) if t
        ).strip()
        section_sim: float | None = None
        if work_text:
            work_vec = self.embedder.embed_long_text(work_text)
            job_vec = self.embedder.embed_long_text(job.raw_text)
            section_sim = EmbeddingService.similarity(work_vec, job_vec)
        else:
            notes.append(
                "No experience/internship/project sections detected, so the "
                "work-experience semantic component was skipped."
            )

        if ratio is not None and section_sim is not None:
            return 0.6 * ratio + 0.4 * section_sim, notes
        if ratio is not None:
            notes.append("Evidence score uses skill-evidence ratio only (x0.8 discount).")
            return 0.8 * ratio, notes
        if section_sim is not None:
            notes.append("Evidence score uses work-section semantic similarity only (x0.8 discount).")
            return 0.8 * section_sim, notes
        return None, notes

    # --- public API -------------------------------------------------------

    def compute_match(
        self,
        resume: ResumeData,
        job: JobData,
        confirmed_skills: list[str] | None = None,
        weights: ScoringWeights | None = None,
    ) -> MatchResult:
        """Compute the full match result for a resume/job pair.

        Raises :class:`ModelNotAvailableError` when embeddings are required
        but the model cannot be loaded.
        """
        weights = (weights or SETTINGS.scoring_weights).normalized()
        confirmed_names = confirmed_skills or [s.name for s in resume.skills_explicit]
        confirmed = set(canonicalize_skill_names(confirmed_names))

        if not confirmed:
            logger.info("No confirmed skills for this resume; coverage may be 0.")

        matched_req, missing_req, req_score = self._coverage(
            job.required_skill_names, confirmed
        )
        matched_pref, missing_pref, pref_score = self._coverage(
            job.preferred_skill_names, confirmed
        )
        semantic = self._semantic_similarity(resume, job)
        evidence_score, evidence_notes = self._evidence_score(
            resume, job, matched_req, confirmed
        )

        raw_components: dict[str, tuple[float | None, str]] = {
            "required_skills": (req_score, "" if req_names_present(job) else "no_required"),
            "preferred_skills": (
                pref_score,
                "" if job.preferred_skill_names else "no_preferred",
            ),
            "semantic_similarity": (semantic, ""),
            "evidence": (evidence_score, ""),
        }

        available = {k: v for k, v in raw_components.items() if v[0] is not None}
        total_available_weight = sum(getattr(weights, k) for k in available)
        notes: list[str] = list(evidence_notes)

        components: list[ScoreComponent] = []
        for key, (value, flag) in raw_components.items():
            base_weight = getattr(weights, key)
            if value is None:
                components.append(
                    ScoreComponent(
                        key=key,
                        label=COMPONENT_LABELS[key],
                        value=None,
                        weight=base_weight,
                        weighted_contribution=None,
                        available=False,
                        note=UNAVAILABLE_NOTES.get(flag, "Not evaluable for this input."),
                    )
                )
            else:
                effective = (
                    base_weight / total_available_weight
                    if total_available_weight > 0 and key in available
                    else 0.0
                )
                components.append(
                    ScoreComponent(
                        key=key,
                        label=COMPONENT_LABELS[key],
                        value=round(value, 4),
                        weight=round(effective, 4),
                        weighted_contribution=round(effective * value, 4),
                        available=True,
                    )
                )

        overall: float | None = None
        if available and total_available_weight > 0:
            overall = sum(
                c.weighted_contribution for c in components if c.available
            )
            overall = round(max(0.0, min(1.0, overall)), 4)

        if not available:
            notes.append("No score component could be evaluated for these inputs.")

        transferable = self._transferable_skills(missing_req, confirmed)

        return MatchResult(
            overall=overall,
            components=components,
            weights_used={c.key: c.weight for c in components},
            matched_required=matched_req,
            missing_required=missing_req,
            matched_preferred=matched_pref,
            missing_preferred=missing_pref,
            transferable=transferable,
            notes=notes,
        )

    def _transferable_skills(
        self, missing_required: list[str], confirmed: set[str]
    ) -> list[TransferableSkill]:
        """Suggest missing skills for which the candidate has related skills."""
        out: list[TransferableSkill] = []
        for missing in missing_required:
            entry = get_skill_entry(missing)
            if not entry or not entry.related:
                continue
            related_owned = [r for r in entry.related if r in confirmed]
            if related_owned:
                out.append(
                    TransferableSkill(
                        missing_skill=missing,
                        related_skills_you_have=related_owned,
                        rationale=(
                            f"You list {', '.join(related_owned)}, which is related "
                            f"to {missing}. This does not replace {missing}, but it "
                            f"can shorten the learning path."
                        ),
                    )
                )
        return out


def req_names_present(job: JobData) -> bool:
    return bool(job.required_skill_names)


COMPONENT_LABELS: dict[str, str] = {
    "required_skills": "Required-skill coverage",
    "preferred_skills": "Preferred-skill coverage",
    "semantic_similarity": "Semantic similarity",
    "evidence": "Relevant evidence",
}

UNAVAILABLE_NOTES: dict[str, str] = {
    "no_required": (
        "No required skills were detected in the job description, so skill "
        "coverage cannot be evaluated. Weight redistributed."
    ),
    "no_preferred": (
        "No preferred skills were detected in the job description. Weight "
        "redistributed to the other components."
    ),
}
