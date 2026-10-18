"""Privacy helpers: session clearing, masking, and disclosure text.

Principles implemented here:
- Resumes are processed in memory only; nothing is written to disk.
- Logs never contain full resumes, contact details, or API keys.
- Users get an explicit list of what an LLM feature sends before using it.
- "Clear session data" removes parsed documents and results from Streamlit
  session state. (This cannot guarantee OS-level memory wiping - the UI
  states this honestly.)
"""

from __future__ import annotations

from src.utils import mask_sensitive, truncate


def clear_session_state() -> list[str]:
    """Delete all Streamlit session keys except the language selector.

    Returns the list of cleared keys (useful for tests/debugging).
    """
    import streamlit as st  # local import: only used inside the app

    preserved = {"lang"}
    cleared: list[str] = []
    for key in list(st.session_state.keys()):
        if key in preserved:
            continue
        try:
            del st.session_state[key]
            cleared.append(key)
        except Exception:  # noqa: BLE001 - never crash on cleanup
            continue
    return cleared


def redact_for_logging(text: str, limit: int = 200) -> str:
    """Make a document excerpt safe for logs (mask emails/phones, truncate)."""
    return truncate(mask_sensitive(text or ""), limit)


LLM_DISCLOSURE_ITEMS = [
    "Your question.",
    "A structured summary of your resume: detected name, confirmed skills, and short section excerpts (not the full file).",
    "The parsed job requirements (title, skills, responsibilities).",
    "Your match summary (component scores, matched/missing skills).",
    "Up to 3 retrieved knowledge-base excerpts (project docs, not personal data).",
]

LLM_DISCLOSURE_NOTE = (
    "The full raw resume file is never sent, API keys are never sent, and "
    "the provider is the one you configured in .env. Clearing session data "
    "removes this context from the app (it cannot guarantee provider-side "
    "retention policies)."
)
