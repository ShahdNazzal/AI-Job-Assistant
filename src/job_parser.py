"""Job-description parsing: title detection, required vs preferred skills,
responsibilities, experience, education, certifications, and keywords.

The parser only reports what is actually present in the text. When the
description does not separate required from preferred qualifications, all
detected skills are treated as required and a note is added.
"""

from __future__ import annotations

import re
from collections import Counter

from config.settings import SETTINGS
from src.exceptions import JobDescriptionError
from src.models import ExtractedSkill, JobData
from src.skill_extractor import extract_skills
from src.utils import count_words, get_logger, normalize_text

logger = get_logger("job_parser")

TITLE_FIELD_RE = re.compile(
    r"^\s*(?:job\s*title|position|role|title)\s*[:\-]\s*(?P<title>.+)$",
    re.IGNORECASE,
)
_TITLE_NOISE = ("we are", "about us", "about the", "apply", "join us", "company")

JD_SECTION_ALIASES: dict[str, list[str]] = {
    "requirements": [
        "requirements", "required qualifications", "qualifications",
        "minimum qualifications", "required skills", "required experience",
        "must have", "must-haves", "what you'll need", "what you will need",
        "who you are", "you have", "skills & qualifications",
        "skills and qualifications", "basic qualifications",
    ],
    "preferred": [
        "preferred qualifications", "preferred skills", "preferred",
        "nice to have", "nice-to-have", "bonus points", "bonus",
        "good to have", "desirable", "plus points", "advantageous",
    ],
    "responsibilities": [
        "responsibilities", "key responsibilities", "your responsibilities",
        "what you'll do", "what you will do", "duties", "the role",
        "about the role", "job responsibilities", "your impact",
    ],
    "benefits": ["benefits", "perks", "what we offer", "we offer"],
}

_EXPERIENCE_RE = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:to|or|-|–)\s*(\d{1,2})\s*\+?\s*(?:\+\s*)?years?"
    r"|(\d{1,2})\s*\+?\s*years?",
    re.IGNORECASE,
)
_EDU_PATTERNS = [
    r"b\.?s\.?c?\b[^.,;\n]*", r"bachelor(?:'s)?(?:\s+degree)?[^.,;\n]*",
    r"m\.?s\.?c?\b[^.,;\n]*", r"master(?:'s)?(?:\s+degree)?[^.,;\n]*",
    r"ph\.?d\.?[^.,;\n]*", r"doctorate[^.,;\n]*",
    r"b\.?a\.?\b[^.,;\n]*", r"m\.?a\.?\b[^.,;\n]*",
    r"associate(?:'s)? degree[^.,;\n]*", r"high school diploma[^.,;\n]*",
    r"bootcamp[^.,;\n]*",
]
_EDU_RE = re.compile("|".join(_EDU_PATTERNS), re.IGNORECASE)
_CERT_RE = re.compile(
    r"(?:aws certified[a-z ]*|azure [a-z][a-z ]*certif[a-z]*|google cloud certified[a-z ]*"
    r"|pmp|cissp|comptia [a-z+]*|cka|itil|scrum master|tensorflow developer certificate"
    r"|professional machine learning engineer)\b",
    re.IGNORECASE,
)
_CERT_WORD_RE = re.compile(r"\bcertif(?:icate|ication|ied)\b", re.IGNORECASE)

_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "are", "will", "that",
    "this", "have", "from", "who", "all", "not", "but", "can", "has", "was",
    "were", "their", "them", "they", "than", "then", "into", "such", "using",
    "use", "work", "team", "role", "job", "years", "year", "experience",
    "including", "ability", "strong", "plus", "preferred", "required",
    "requirements", "responsibilities", "qualifications", "skills", "knowledge",
    "understanding", "candidates", "candidate", "looking", "seeking", "company",
    "opportunity", "benefits", "about", "what", "you'll", "we're", "we", "it",
    "an", "as", "at", "be", "by", "in", "is", "of", "on", "or", "to", "per",
    "other", "etc", "new", "well", "best", "must", "should", "would", "like",
}


def _split_jd_sections(
    text: str,
) -> tuple[dict[str, str], dict[str, tuple[int, int]], bool]:
    """Split a job description into labeled sections.

    Returns ``(sections, spans, has_required_heading)``.
    """
    sections: dict[str, str] = {}
    spans: dict[str, tuple[int, int]] = {}
    current = "header"
    current_start = 0
    buffer: list[str] = []
    offset = 0

    def _flush(end_pos: int) -> None:
        joined = "\n".join(buffer).strip()
        if current in sections:
            if joined:
                sections[current] = sections[current] + "\n" + joined
            spans[current] = (spans[current][0], max(spans[current][1], end_pos))
        else:
            sections[current] = joined
            spans[current] = (current_start, end_pos)

    lines = text.split("\n")
    saw_requirement_heading = False
    for line in lines:
        stripped = line.strip().rstrip(":").strip()
        lowered = stripped.lower()
        heading = None
        if 2 <= len(stripped) <= 60 and re.match(r"^[A-Za-z][A-Za-z&' \-]+$", stripped):
            for label, aliases in JD_SECTION_ALIASES.items():
                if lowered in aliases:
                    heading = label
                    break
        if heading:
            _flush(offset)
            current = heading
            current_start = offset
            buffer = []
            if heading == "requirements":
                saw_requirement_heading = True
        else:
            buffer.append(line)
        offset += len(line) + 1
    _flush(offset)
    return sections, spans, saw_requirement_heading


def _detect_title(text: str, header_text: str) -> tuple[str | None, str]:
    for line in text.split("\n")[:20]:
        m = TITLE_FIELD_RE.match(line)
        if m:
            title = m.group("title").strip().rstrip(".")
            if 2 <= len(title) <= 80:
                return title, "high"
    for line in header_text.split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if len(candidate.split()) <= 10 and not any(n in lowered for n in _TITLE_NOISE):
            if not re.search(r"\d{3,}", candidate) and not candidate.startswith(("http", "www")):
                return candidate, "low"
        break
    return None, "none"


def _extract_experience_years(text: str) -> int | None:
    best: int | None = None
    for m in _EXPERIENCE_RE.finditer(text):
        lo = int(m.group(1) or m.group(3) or 0)
        if best is None or lo < best:
            best = lo
    return best


def _extract_education(text: str) -> list[str]:
    found: list[str] = []
    for m in _EDU_RE.finditer(text):
        snippet = m.group(0).strip()
        if 3 <= len(snippet) <= 90 and snippet.lower() not in {f.lower() for f in found}:
            found.append(snippet)
    return found[:6]


def _extract_certifications(text: str) -> list[str]:
    found: list[str] = []
    for m in _CERT_RE.finditer(text):
        cert = m.group(0).strip()
        if cert.lower() not in {c.lower() for c in found}:
            found.append(cert)
    if not found:
        # Mention of "certification (e.g. X)" without a known vendor name.
        for m in _CERT_WORD_RE.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start: line_end if line_end != -1 else len(text)].strip()
            if line and len(line) <= 120 and line.lower() not in {c.lower() for c in found}:
                found.append(line)
            break
    return found[:8]


def _extract_keywords(text: str, exclude: set[str], limit: int = 12) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z+#.\-]{2,}", text.lower())
    filtered = [
        t for t in tokens
        if t not in _STOPWORDS and t not in exclude and len(t) >= 3
    ]
    counts = Counter(filtered)
    return [word for word, _count in counts.most_common(limit)]


def _classify_skills(
    raw_text: str,
    skills: list[ExtractedSkill],
    spans: dict[str, tuple[int, int]],
) -> tuple[list[ExtractedSkill], list[ExtractedSkill]]:
    """Assign each detected skill to required/preferred by evidence location."""
    required: list[ExtractedSkill] = []
    preferred: list[ExtractedSkill] = []
    lowered = raw_text.lower()
    for skill in skills:
        surface = (skill.matched_text or skill.name).lower()
        position = lowered.find(surface)
        section: str | None = None
        if position >= 0:
            for name, (start, end) in spans.items():
                if start <= position < end:
                    section = name
                    break
        if section == "preferred":
            preferred.append(skill)
        else:
            required.append(skill)
    return required, preferred


def _extract_responsibilities(text: str) -> list[str]:
    items: list[str] = []
    for line in text.split("\n"):
        cleaned = line.strip()
        cleaned = re.sub(r"^[-*\u2022\u00b7\u25cf\u25a0\u25aa]\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        if 15 <= len(cleaned) <= 220 and not cleaned.endswith(":"):
            items.append(cleaned)
        if len(items) >= 12:
            break
    return items


def parse_job_description(text: str) -> JobData:
    """Parse raw job-description text into structured :class:`JobData`."""
    normalized = normalize_text(text)
    if count_words(normalized) < 15 or len(normalized) < SETTINGS.min_job_chars:
        raise JobDescriptionError(
            f"Job description too short ({len(normalized)} characters); "
            f"please provide a more complete description."
        )

    sections, spans, has_req_heading = _split_jd_sections(normalized)
    header_text = sections.get("header", "")
    title, title_conf = _detect_title(normalized, header_text)

    all_skills, _inferred = extract_skills(normalized)
    required, preferred = _classify_skills(normalized, all_skills, spans)

    warnings: list[str] = []
    if title is None:
        warnings.append("no_title")
    if not has_req_heading:
        warnings.append("no_required_heading")
    if not all_skills:
        warnings.append("no_skills")

    responsibilities = _extract_responsibilities(sections.get("responsibilities", ""))
    experience_years = _extract_experience_years(normalized)
    education = _extract_education(normalized)
    certifications = _extract_certifications(normalized)
    exclude = {s.name.lower() for s in all_skills}
    keywords = _extract_keywords(normalized, exclude)

    return JobData(
        raw_text=normalized,
        title=title,
        title_confidence=title_conf,
        required_skills=required,
        preferred_skills=preferred,
        responsibilities=responsibilities,
        experience_years_min=experience_years,
        education=education,
        certifications=certifications,
        keywords=keywords,
        required_section_text=sections.get("requirements") or None,
        preferred_section_text=sections.get("preferred") or None,
        word_count=count_words(normalized),
        warnings=warnings,
    )
