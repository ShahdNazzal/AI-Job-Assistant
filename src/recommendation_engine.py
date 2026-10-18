"""Rule-based, evidence-aware recommendations.

Design rules (enforced here, verified in tests):
- Never suggest adding a skill the candidate does not have; missing skills
  become *learning* suggestions instead.
- Example rewrites use only facts present in the resume (skills, project
  names, section text) and are always marked as drafts.
- No invented numbers: quantification advice only asks the user to add
  their own real metrics.
"""

from __future__ import annotations

import re

from src.models import (
    JobData,
    MatchResult,
    RecommendationItem,
    RecommendationSet,
    ResumeData,
    RetrievedChunk,
)
from src.skill_extractor import get_skill_entry
from src.utils import get_logger

logger = get_logger("recommendation_engine")

VAGUE_PHRASES = (
    "responsible for",
    "worked on",
    "helped with",
    "assisted with",
    "involved in",
    "various tasks",
    "familiar with",
    "exposure to",
    "participated in",
)

_SUMMARY_MIN_CHARS = 240


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _word_present(haystack_lower: str, phrase: str) -> bool:
    pattern = re.escape(phrase.lower())
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", haystack_lower) is not None


def generate_recommendations(
    resume: ResumeData,
    job: JobData,
    match: MatchResult,
    confirmed_skills: list[str],
) -> RecommendationSet:
    """Build the full rule-based recommendation set for the current analysis."""
    items: list[RecommendationItem] = []

    # 1) Prioritized learning paths for missing required skills.
    for skill in match.missing_required[:6]:
        entry = get_skill_entry(skill)
        hint = entry.learning_hints if entry else ""
        related = next(
            (t.related_skills_you_have for t in match.transferable if t.missing_skill == skill),
            [],
        )
        detail = hint or "Study the fundamentals, then apply them in a small, documented project."
        if related:
            detail += f" Your experience with {', '.join(related)} should speed this up."
        items.append(
            RecommendationItem(
                kind="learning",
                title=f"Close the gap: {skill}",
                detail=detail,
                priority="High",
                source="rule-based",
            )
        )

    # 2) Interview preparation topics.
    topics = match.missing_required[:4] + [s for s in match.matched_required[:3]]
    if topics:
        items.append(
            RecommendationItem(
                kind="interview",
                title="Prepare these technical topics for interviews",
                detail=(
                    "Be ready to explain and (where relevant) demonstrate: "
                    + ", ".join(topics)
                    + ". For topics you already know, prepare one concrete example "
                      "from your own projects."
                ),
                priority="High",
            )
        )

    # 3) Professional summary improvement (uses only real resume facts).
    summary = resume.summary or ""
    if len(summary) < _SUMMARY_MIN_CHARS:
        top_skills = confirmed_skills[:6]
        project_names = _project_titles(resume)
        role = job.title or "the target role"
        draft = (
            f"Aspiring AI engineer with hands-on experience in {', '.join(top_skills) or 'Python and machine learning'}. "
            + (
                f"Built projects including {', '.join(project_names[:2])}. "
                if project_names
                else ""
            )
            + f"Seeking to apply these skills as a {role}."
        )
        items.append(
            RecommendationItem(
                kind="improvement",
                title="Strengthen your professional summary",
                detail=(
                    "Your summary is short or missing. A strong summary names your "
                    "target role, your strongest confirmed skills, and one concrete "
                    "project. The draft below uses only facts found in your resume - "
                    "edit and verify before use."
                ),
                priority="Medium",
                before=summary[:300] if summary else "(no summary detected)",
                after=draft,
                source="rule-based",
            )
        )

    # 4) Vague statements in experience/projects.
    work_text = "\n".join(
        t for t in (resume.experience_text, resume.projects_text, resume.raw_text) if t
    )
    lowered = work_text.lower()
    flagged = [p for p in VAGUE_PHRASES if _word_present(lowered, p)]
    if flagged:
        items.append(
            RecommendationItem(
                kind="improvement",
                title="Replace vague statements with concrete, verifiable outcomes",
                detail=(
                    "Phrases found in your resume: "
                    + ", ".join(f"'{p}'" for p in flagged[:5])
                    + ". For each, state what you built, the tools you used, and a "
                      "real, checkable result. Never invent metrics."
                ),
                priority="Medium",
                before=f"Responsible for working on various ML tasks.",
                after=(
                    "Built a text-classification pipeline with Python and scikit-learn; "
                    "describe the dataset, your exact contribution, and one true, "
                    "verifiable outcome (e.g. 'validated with 5-fold cross-validation')."
                ),
                source="rule-based",
            )
        )

    # 5) Quantification nudge - only if experience exists without any numbers.
    experience = resume.experience_text
    if experience and not _has_digit(experience):
        items.append(
            RecommendationItem(
                kind="improvement",
                title="Add real, verifiable numbers to your experience",
                detail=(
                    "Your experience section contains no figures. Where TRUE, add "
                    "dataset sizes, model metric improvements you actually measured, "
                    "or scope (users, documents processed). Only include numbers you "
                    "can defend in an interview."
                ),
                priority="Low",
                source="rule-based",
            )
        )

    # 6) Skills-section organization.
    if confirmed_skills:
        categories = sorted(
            {
                entry.category_label
                for name in confirmed_skills
                if (entry := get_skill_entry(name))
            }
        )
        if len(categories) >= 2:
            items.append(
                RecommendationItem(
                    kind="improvement",
                    title="Group your technical skills by category",
                    detail=(
                        "Organize skills under headings such as "
                        + ", ".join(categories[:5])
                        + ". Recruiters and ATS parsers both scan faster when skills "
                          "are grouped and named exactly as the job ad names them."
                    ),
                    priority="Low",
                    source="rule-based",
                )
            )

    # 7) Job-description keywords missing from the resume (honest framing).
    resume_lower = resume.raw_text.lower()
    missing_keywords = [
        kw
        for kw in job.keywords
        if len(kw) >= 4 and not _word_present(resume_lower, kw)
    ][:5]
    if missing_keywords:
        items.append(
            RecommendationItem(
                kind="improvement",
                title="Cover important job-description keywords - honestly",
                detail=(
                    "These keywords appear in the job description but not in your "
                    "resume: " + ", ".join(missing_keywords) + ". If you genuinely "
                    "have related experience, describe it using the same terminology; "
                    "if not, treat them as learning topics instead of adding them."
                ),
                priority="Medium",
                source="rule-based",
            )
        )

    return RecommendationSet(items=items, generated_with_llm=False)


def _project_titles(resume: ResumeData) -> list[str]:
    """Extract short project titles from 'Project:'-style lines."""
    titles: list[str] = []
    for line in resume.projects_text.split("\n"):
        cleaned = line.strip()
        m = re.match(r"^(?:[-*\u2022]?\s*)?([A-Za-z0-9][A-Za-z0-9 ._'/\-]{3,48}?)(?::|\s[-\u2013]\s)", cleaned)
        if m:
            title = m.group(1).strip()
            if title and title.lower() not in {t.lower() for t in titles}:
                titles.append(title)
    return titles[:5]


# --- Rule-based assistant fallback -------------------------------------------


def rule_based_answer(
    question: str,
    resume: ResumeData,
    job: JobData,
    match: MatchResult,
    confirmed_skills: list[str],
    kb_chunks: list[RetrievedChunk] | None = None,
) -> str:
    """Deterministic fallback answers when no LLM is configured.

    Mirrors the suggested-questions intents so users without an API key still
    get useful, grounded answers.
    """
    q = (question or "").lower()
    job_label = job.title or "this role"
    lines: list[str] = []

    def _head(text: str) -> None:
        lines.append(text)
        lines.append("")

    if any(
        w in q
        for w in ("gap", "missing", "biggest skill", "فجوات", "ناقص", "فجوة")
    ):
        _head("Your biggest skill gaps for " + job_label + ":")
        if match.missing_required:
            for i, skill in enumerate(match.missing_required[:3], start=1):
                entry = get_skill_entry(skill)
                hint = entry.learning_hints if entry else ""
                lines.append(f"{i}. {skill}" + (f" - {hint}" if hint else ""))
        else:
            lines.append("None detected: your confirmed skills cover the requirements found.")
    elif any(
        w in q
        for w in ("project", "relevant experience", "مشاريع", "مشروع")
    ):
        _head("How to judge project relevance for " + job_label + ":")
        projects = resume.projects_text
        if projects:
            overlap = [
                s
                for s in match.matched_required
                if any(
                    s.lower() in line.lower()
                    for line in projects.split("\n")
                )
            ]
            lines.append(
                "Projects that mention required skills carry the most weight. "
                + (
                    "Skills evidenced in your projects: " + ", ".join(overlap) + "."
                    if overlap
                    else "None of your project lines mention the required skills by name."
                )
            )
            lines.append("Tip: name the required technology explicitly in the project bullet where you truly used it.")
        else:
            lines.append("No projects section was detected in your resume.")
    elif any(
        w in q
        for w in ("summary", "improve my professional", "ملخص")
    ):
        _head("Improving your professional summary:")
        lines.append(
            "Name the target role, your strongest confirmed skills, and one "
            "concrete project. Current summary: "
            + ((resume.summary or "none detected")[:220])
        )
    elif any(
        w in q
        for w in ("interview", "prepare", "topics", "مقابلة", "تحضير", "مواضيع")
    ):
        _head("Technical topics to prepare:")
        topics = match.missing_required[:4] + match.matched_required[:3]
        lines.append(
            ", ".join(topics) if topics else "No specific topics detected - review the core skills of the role."
        )
    elif any(
        w in q
        for w in (
            "satisfy",
            "already have",
            "meet the requirements",
            "qualify",
            "أستوفي",
            "متطلبات",
        )
    ):
        _head("Requirements you already satisfy:")
        lines.append(
            ", ".join(match.matched_required)
            if match.matched_required
            else "None of the detected required skills are in your confirmed list yet."
        )
    elif any(
        w in q
        for w in ("score", "evidence", "why", "درجة", "توافق", "أدلة")
    ):
        _head("What supports your compatibility score:")
        for c in match.components:
            status = f"{c.value:.2f}" if c.value is not None else "not evaluable"
            lines.append(f"- {c.label}: {status} (weight {c.weight:.0%})")
        lines.append("")
        lines.append("The overall score is the weighted sum of the evaluable components.")
    else:
        _head("Rule-based assistant (no LLM configured):")
        lines.append(
            "I can answer from your analysis only. Confirmed skills: "
            + (", ".join(confirmed_skills[:10]) or "none")
            + f". Matched required skills: {', '.join(match.matched_required) or 'none'}. "
            f"Missing: {', '.join(match.missing_required) or 'none'}."
        )

    if kb_chunks:
        lines.append("")
        lines.append("Related knowledge-base excerpts:")
        for chunk in kb_chunks[:2]:
            excerpt = chunk.text[:160].replace("\n", " ")
            lines.append(f"- [{chunk.title}] {excerpt}...")

    lines.append("")
    lines.append(
        "Note: configure an LLM provider in .env to get generative, "
        "context-aware answers. These rule-based answers use only your "
        "parsed documents."
    )
    return "\n".join(lines)
