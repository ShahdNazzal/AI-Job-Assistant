"""Resume parsing: file validation, text extraction (PDF/DOCX/TXT), and
lightweight structural analysis (sections, name, contact, summary).

The parser never fabricates information: fields it cannot detect are left
as ``None`` and the UI marks them as "Not detected". Scanned/image-only
PDFs raise :class:`InsufficientTextError` instead of pretending success.
"""

from __future__ import annotations

import io
import re

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from config.settings import SETTINGS
from src.exceptions import (
    CorruptedFileError,
    EmptyDocumentError,
    EmptyFileError,
    FileTooLargeError,
    InsufficientTextError,
    UnsupportedFileTypeError,
)
from src.models import ContactInfo, ExtractedSkill, ResumeData, SectionInfo
from src.skill_extractor import extract_skills
from src.utils import count_words, get_logger, make_snippet, normalize_text

logger = get_logger("resume_parser")

# Canonical section name -> list of heading aliases (matched as whole lines).
SECTION_ALIASES: dict[str, list[str]] = {
    "summary": [
        "professional summary", "summary", "profile", "professional profile",
        "objective", "career objective", "about me", "about",
    ],
    "skills": [
        "skills", "technical skills", "core skills", "key skills",
        "technical proficiency", "competencies", "technologies", "tech stack",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history", "career history",
    ],
    "internships": ["internships", "internship experience", "internship"],
    "projects": [
        "projects", "personal projects", "academic projects",
        "selected projects", "key projects", "project experience",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "education and training", "educational background",
    ],
    "certifications": [
        "certifications", "certificates", "licenses",
        "professional development", "courses & certifications",
    ],
    "languages": ["languages"],
    "awards": ["awards", "honors", "honours", "achievements", "awards & honors"],
    "publications": ["publications", "publications & talks"],
}

_HEADING_LINE_RE = re.compile(
    r"^(?P<title>[A-Za-z][A-Za-z&/ '\-]{1,48})\s*:?\s*$"
)
_NAME_FIELD_RE = re.compile(r"^\s*(?:full[\s-]*)?name\s*[:\-]\s*(?P<name>.+)$", re.IGNORECASE)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{3,4}")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9._%-]+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9._%-]+", re.IGNORECASE)

_NAME_STOPWORDS = {
    "resume", "curriculum", "vitae", "cv", "curriculum vitae",
    "contact", "information", "profile", "summary", "details",
}


# --- Text extraction --------------------------------------------------------


def _validate_file(data: bytes, filename: str) -> str:
    """Validate size/type; return the lower-cased extension."""
    if not data:
        raise EmptyFileError("Uploaded file is empty (0 bytes).")
    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds {SETTINGS.max_upload_mb} MB limit.")
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in SETTINGS.allowed_extensions:
        allowed = ", ".join(SETTINGS.allowed_extensions)
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed: {allowed}"
        )
    return ext


def _extract_pdf(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises various types
        raise CorruptedFileError(f"Cannot open PDF: {exc}") from exc
    try:
        pages = [page.get_text("text") or "" for page in doc]
    except Exception as exc:  # noqa: BLE001
        raise CorruptedFileError(f"Cannot read PDF pages: {exc}") from exc
    finally:
        doc.close()
    total_chars = sum(len(p.strip()) for p in pages)
    n_pages = len(pages)
    if n_pages and total_chars < SETTINGS.scanned_pdf_chars_per_page * n_pages:
        raise InsufficientTextError(
            "PDF contains almost no extractable text (likely a scanned or "
            "image-only document). OCR is not implemented; use a text-based "
            "PDF, DOCX, or paste the text."
        )
    return "\n".join(pages)


def _extract_docx(data: bytes) -> str:
    try:
        document = DocxDocument(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise CorruptedFileError(f"Cannot open DOCX: {exc}") from exc
    parts = [p.text for p in document.paragraphs]
    # Skills are often stored inside tables; read them too.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n".join(parts)


def _extract_txt(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# --- Section splitting --------------------------------------------------------


def _canonical_heading(line: str) -> str | None:
    """Return the canonical section name if ``line`` looks like a heading."""
    stripped = line.strip().rstrip(":").strip()
    if not (2 <= len(stripped) <= 48):
        return None
    match = _HEADING_LINE_RE.match(stripped)
    if not match:
        return None
    title = match.group("title").lower().strip()
    for canonical, aliases in SECTION_ALIASES.items():
        if title in aliases:
            return canonical
    return None


def _split_sections(text: str) -> tuple[dict[str, SectionInfo], dict[str, tuple[int, int]]]:
    """Split resume text into canonical sections.

    Returns ``(sections, spans)`` where ``spans`` maps each canonical name to
    its ``(char_start, char_end)`` range in ``text`` so downstream modules can
    attribute evidence to sections.
    """
    sections: dict[str, SectionInfo] = {}
    spans: dict[str, tuple[int, int]] = {}
    current: str | None = None
    current_start = 0
    buffer: list[str] = []
    offset = 0

    def _flush(end_pos: int) -> None:
        if current is None:
            return
        joined = "\n".join(buffer).strip()
        sections[current] = SectionInfo(name=current, found=bool(joined), text=joined)
        spans[current] = (current_start, end_pos)

    lines = text.split("\n")
    for line in lines:
        heading = _canonical_heading(line)
        if heading:
            _flush(offset)
            current = heading
            current_start = offset
            buffer = []
        else:
            buffer.append(line)
        offset += len(line) + 1
    _flush(offset)

    if not sections:
        # No recognizable headings: treat the whole text as a pseudo header.
        sections["header"] = SectionInfo(name="header", found=bool(text.strip()), text=text.strip())
        spans["header"] = (0, len(text))
    return sections, spans


def _section_of(spans: dict[str, tuple[int, int]], position: int) -> str | None:
    """Return the canonical section containing character ``position``."""
    for name, (start, end) in spans.items():
        if start <= position < end:
            return name
    return None


# --- Field extraction ---------------------------------------------------------


def _extract_name(text: str, spans: dict[str, tuple[int, int]]) -> tuple[str | None, str]:
    """Detect the candidate name. Returns ``(name, confidence)``."""
    for line in text.split("\n")[:15]:
        m = _NAME_FIELD_RE.match(line)
        if m:
            candidate = m.group("name").strip()
            if 2 <= len(candidate) <= 60:
                return candidate, "high"
    # Heuristic: first non-empty line before any section heading.
    header_end = spans.get("summary", (None, None))[0]
    for line in text[: header_end if header_end else len(text)].split("\n"):
        candidate = line.strip()
        if not candidate:
            continue
        words = candidate.split()
        if not (2 <= len(words) <= 5):
            return None, "none"
        if _EMAIL_RE.search(candidate) or any(ch.isdigit() for ch in candidate):
            return None, "none"
        if any(w.lower() in _NAME_STOPWORDS for w in words):
            return None, "none"
        if all(w.isalpha() or w in {"."} or w.endswith(".") for w in words):
            return candidate.title() if candidate.isupper() or candidate.islower() else candidate, "low"
        return None, "none"
    return None, "none"


def _extract_contact(text: str) -> ContactInfo:
    email = _EMAIL_RE.search(text)
    linkedin = _LINKEDIN_RE.search(text)
    github = _GITHUB_RE.search(text)
    phone: str | None = None
    for m in _PHONE_RE.finditer(text):
        # Reject matches that are part of an email or look like a date.
        candidate = m.group(0).strip()
        digits = sum(ch.isdigit() for ch in candidate)
        if 7 <= digits <= 15 and "@" not in candidate:
            phone = candidate
            break
    return ContactInfo(
        email=email.group(0) if email else None,
        phone=phone,
        linkedin=linkedin.group(0) if linkedin else None,
        github=github.group(0) if github else None,
        location=None,  # Location extraction is deliberately conservative.
    )


def _extract_summary(
    text: str, sections: dict[str, SectionInfo]
) -> tuple[str | None, str]:
    info = sections.get("summary")
    if info and info.found and info.text:
        return info.text.strip(), "high"
    # Fallback: first meaningful paragraph, marked low confidence.
    for paragraph in text.split("\n\n"):
        cleaned = paragraph.strip()
        if len(cleaned) >= 80 and not _EMAIL_RE.search(cleaned.split("\n")[0]):
            first_lines = " ".join(cleaned.split("\n")[:4])
            return first_lines[:400], "low"
    return None, "none"


# --- Public API ---------------------------------------------------------------


def parse_resume_text(
    text: str,
    source_type: str = "txt",
    source_label: str = "resume",
    with_semantic_skills: bool = False,
) -> ResumeData:
    """Analyze already-extracted resume text and return structured data."""
    normalized = normalize_text(text)
    words = count_words(normalized)
    warnings: list[str] = []
    if words < 40:
        warnings.append(
            f"short_text:{words}"  # machine-readable; UI translates it.
        )

    sections, spans = _split_sections(normalized)
    name, name_conf = _extract_name(normalized, spans)
    contact = _extract_contact(normalized)
    summary, summary_conf = _extract_summary(normalized, sections)

    explicit, inferred = extract_skills(
        normalized,
        section_spans=spans,
        include_semantic=with_semantic_skills,
    )

    return ResumeData(
        raw_text=normalized,
        source_type=source_type,
        source_label=source_label,
        name=name,
        name_confidence=name_conf,
        contact=contact,
        summary=summary,
        summary_confidence=summary_conf,
        sections=sections,
        skills_explicit=explicit,
        skills_inferred=inferred,
        word_count=words,
        warnings=warnings,
    )


def parse_resume(data: bytes, filename: str, with_semantic_skills: bool = False) -> ResumeData:
    """Validate, extract text from, and parse an uploaded resume file."""
    ext = _validate_file(data, filename)
    if ext == ".pdf":
        text = _extract_pdf(data)
        source_type = "pdf"
    elif ext == ".docx":
        text = _extract_docx(data)
        source_type = "docx"
    else:
        text = _extract_txt(data)
        source_type = "txt"

    if not text or not text.strip():
        raise EmptyDocumentError("Document contains no readable text.")
    logger.info("Parsed resume file: type=%s chars=%d", source_type, len(text))
    return parse_resume_text(
        text, source_type=source_type, source_label=filename,
        with_semantic_skills=with_semantic_skills,
    )


def snippet_from(raw_text: str, start: int, end: int) -> str:
    """Convenience wrapper for building evidence snippets."""
    return make_snippet(raw_text, start, end, SETTINGS.evidence_snippet_width)
