"""Custom exception hierarchy for AI Job Assistant.

Every user-facing error maps to one of these types so the UI can show a
clear, actionable message instead of a raw traceback.
"""

from __future__ import annotations


class AIJobAssistantError(Exception):
    """Base class for all application-specific errors."""


# --- Resume parsing ---------------------------------------------------------


class ResumeParseError(AIJobAssistantError):
    """Generic resume parsing failure."""


class UnsupportedFileTypeError(ResumeParseError):
    """The uploaded file has an extension we cannot parse."""


class FileTooLargeError(ResumeParseError):
    """The uploaded file exceeds the configured size limit."""


class EmptyFileError(ResumeParseError):
    """The uploaded file contains zero bytes of data."""


class CorruptedFileError(ResumeParseError):
    """The file exists but cannot be opened (e.g. broken PDF/DOCX)."""


class EmptyDocumentError(ResumeParseError):
    """The document opened fine but yielded no text at all."""


class InsufficientTextError(ResumeParseError):
    """The document yielded too little text (likely a scanned/image-only PDF)."""


# --- Job description --------------------------------------------------------


class JobDescriptionError(AIJobAssistantError):
    """Job description is missing, too short, or unreadable."""


# --- Skills taxonomy --------------------------------------------------------


class TaxonomyError(AIJobAssistantError):
    """skills_taxonomy.json is missing or malformed."""


# --- Embeddings -------------------------------------------------------------


class ModelNotAvailableError(AIJobAssistantError):
    """The Sentence Transformer model could not be loaded."""


# --- LLM --------------------------------------------------------------------


class LLMServiceError(AIJobAssistantError):
    """Base class for LLM adapter errors (never carries the API key)."""


class LLMConfigurationError(LLMServiceError):
    """LLM requested but credentials are missing/incomplete."""


class LLMRequestError(LLMServiceError):
    """Network / HTTP failure while calling the LLM API."""


class LLMResponseError(LLMServiceError):
    """The LLM replied but the response is empty or invalid."""
