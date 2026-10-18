"""Small shared utilities: text normalization, logging, snippet extraction.

Design notes:
- ``normalize_text`` deliberately preserves paragraph structure; it only
  repairs artifacts introduced by PDF extraction (ligatures, weird bullets,
  runaway blank lines, trailing whitespace).
- Loggers never receive full resumes or contact details; ``mask_sensitive``
  is available for anything that might contain personal data.
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata

# --- Logging ----------------------------------------------------------------

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single sane handler attached."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root = logging.getLogger("ai_job_assistant")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"ai_job_assistant.{name}")


logger = get_logger("utils")

# --- Text normalization -----------------------------------------------------

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

_BULLET_CHARS = "•·▪◦‣∙●○■□–—―"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")


def normalize_text(raw: str) -> str:
    """Clean extractor artifacts while keeping meaningful structure.

    Steps: NFC normalization, ligature repair, smart-quote/dash unification,
    bullet canonicalization, per-line whitespace trimming, collapse of 3+
    consecutive blank lines into one blank line.
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFC", raw)
    for lig, plain in _LIGATURES.items():
        text = text.replace(lig, plain)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u00a0", " ")

    lines: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if stripped and all(ch in _BULLET_CHARS for ch in stripped):
            line = "- " + line.lstrip("".join(_BULLET_CHARS) + " ")
        else:
            line = line.rstrip()
        lines.append(line)
    text = "\n".join(lines)
    # Collapse runs of blank lines (3+ newlines) to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def count_words(text: str) -> int:
    """Approximate word count used for validation and reporting."""
    return len((text or "").split())


# --- Evidence snippets -------------------------------------------------------


def make_snippet(text: str, center_start: int, center_end: int, width: int = 120) -> str:
    """Return a short excerpt of ``text`` centered on a match.

    ``center_start``/``center_end`` are character offsets of the match.
    The result is collapsed to single spaces for safe display in tables.
    """
    if not text:
        return ""
    half = max(width // 2, 20)
    start = max(0, center_start - half)
    end = min(len(text), center_end + half)
    snippet = text[start:end].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


# --- Privacy helpers ---------------------------------------------------------


def mask_sensitive(text: str, keep_domain: bool = True) -> str:
    """Mask emails and phone numbers for log-safe output."""
    if not text:
        return ""

    def _mask_email(m: re.Match) -> str:
        value = m.group(0)
        if keep_domain and "@" in value:
            local, _, domain = value.partition("@")
            shown = local[:2] if len(local) >= 2 else "*"
            return f"{shown}***@{domain}"
        return "***"

    text = _EMAIL_RE.sub(_mask_email, text)
    text = _PHONE_RE.sub(lambda m: "***", text)
    return text


def truncate(text: str, limit: int, marker: str = "... [truncated]") -> str:
    """Hard-truncate long text for prompts/reports, on a word boundary."""
    if not text or len(text) <= limit:
        return text or ""
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit * 0.6:
        cut = cut[:last_space]
    return cut.rstrip() + " " + marker
