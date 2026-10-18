"""Tests for resume parsing: extraction, validation, and error handling."""

from __future__ import annotations

import io

import fitz
import pytest
from docx import Document

from src.exceptions import (
    CorruptedFileError,
    EmptyDocumentError,
    EmptyFileError,
    FileTooLargeError,
    InsufficientTextError,
    UnsupportedFileTypeError,
)
from src.resume_parser import parse_resume, parse_resume_text

SAMPLE = """Sarah Chen
sarah@example.com | +1 555 210 4477

Professional Summary
ML-focused computer science graduate.

Technical Skills
Python, scikit-learn, SQL, Docker

Work Experience
Intern - Acme (2024)
- Built models with Python.

Projects
Semantic search demo.
"""

SAMPLE_RESUME_PATH = None  # placeholder to keep imports tidy


def _docx_bytes(paragraphs: list[str]) -> bytes:
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _pdf_bytes(text: str | None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    buffer = doc.tobytes()
    doc.close()
    return buffer


class TestTextExtraction:
    def test_txt_roundtrip(self):
        result = parse_resume(b"hello world", "cv.txt")
        assert result.raw_text == "hello world"
        assert result.source_type == "txt"

    def test_docx_extraction(self):
        data = _docx_bytes(["First line", "Second line"])
        result = parse_resume(data, "cv.docx")
        assert "First line" in result.raw_text
        assert "Second line" in result.raw_text
        assert result.source_type == "docx"

    def test_pdf_extraction(self):
        text = (
            "Python developer with SQL experience and Docker deployments. "
            "Built data pipelines and REST APIs for several production teams."
        )
        data = _pdf_bytes(text)
        result = parse_resume(data, "cv.pdf")
        assert "Python" in result.raw_text
        assert result.source_type == "pdf"

    def test_docx_table_content(self):
        document = Document()
        table = document.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "Python, Docker"
        buffer = io.BytesIO()
        document.save(buffer)
        result = parse_resume(buffer.getvalue(), "cv.docx")
        assert "Docker" in result.raw_text


class TestValidation:
    def test_empty_file(self):
        with pytest.raises(EmptyFileError):
            parse_resume(b"", "cv.pdf")

    def test_unsupported_type(self):
        with pytest.raises(UnsupportedFileTypeError):
            parse_resume(b"x", "cv.exe")

    def test_too_large(self):
        import config.settings as cs
        from src.resume_parser import _validate_file

        big = b"x" * (cs.SETTINGS.max_upload_mb * 1024 * 1024 + 1)
        with pytest.raises(FileTooLargeError):
            _validate_file(big, "cv.pdf")

    def test_corrupted_pdf(self):
        with pytest.raises(CorruptedFileError):
            parse_resume(b"this is definitely not a pdf", "cv.pdf")

    def test_empty_document(self):
        with pytest.raises(EmptyDocumentError):
            parse_resume(b"   \n  \n ", "cv.txt")

    def test_scanned_pdf_detected(self):
        # A valid PDF page with no text at all must NOT pass as parsed.
        with pytest.raises(InsufficientTextError):
            parse_resume(_pdf_bytes(None), "scanned.pdf")


class TestStructure:
    def test_sections_detected(self):
        result = parse_resume_text(SAMPLE)
        assert result.sections["summary"].found
        assert result.sections["skills"].found
        assert result.sections["experience"].found
        assert result.sections["projects"].found

    def test_name_and_contact(self):
        result = parse_resume_text(SAMPLE)
        assert result.name == "Sarah Chen"
        assert result.name_confidence in {"high", "low"}
        assert result.contact.email == "sarah@example.com"
        assert result.contact.phone is not None

    def test_skills_extracted_with_evidence(self):
        result = parse_resume_text(SAMPLE)
        names = {s.name for s in result.skills_explicit}
        assert "Python" in names
        assert "Scikit-learn" in names  # canonical name via alias handling
        python_skill = next(s for s in result.skills_explicit if s.name == "Python")
        assert python_skill.evidence
        assert any("Python" in ev.snippet for ev in python_skill.evidence)

    def test_short_text_warning(self):
        result = parse_resume_text("tiny resume")
        assert any(w.startswith("short_text:") for w in result.warnings)

    def test_no_fabricated_fields(self):
        result = parse_resume_text(
            "Just a paragraph with no contact info at all. " * 5
        )
        assert result.contact.email is None
        assert result.summary is None or result.summary_confidence == "low"

    def test_section_text_helper(self):
        result = parse_resume_text(SAMPLE)
        combined = result.section_text(["experience", "projects"])
        assert "Acme" in combined
        assert "Semantic search demo" in combined
