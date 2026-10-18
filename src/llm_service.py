"""Optional LLM adapter for OpenAI-compatible chat-completions APIs.

- Credentials come exclusively from environment variables (.env).
- Works with any provider exposing ``POST {base_url}/chat/completions``
  (OpenAI, Azure-style gateways, Groq, Together, OpenRouter, Ollama, vLLM...).
- Never raises into the UI: every failure returns an :class:`LLMResult` with
  a sanitized message. API keys are stripped from any error text.
- ``chat_json`` validates structured replies with Pydantic models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests
from pydantic import BaseModel, ValidationError

from config.settings import LLMConfig
from src.utils import get_logger

logger = get_logger("llm_service")


@dataclass
class LLMResult:
    ok: bool
    content: str = ""
    error: str = ""
    model: str = ""


def _sanitize(text: str, secret: str) -> str:
    """Remove any accidental secret leakage from provider error bodies."""
    if not text:
        return ""
    cleaned = text
    if secret:
        cleaned = cleaned.replace(secret, "***")
    return cleaned[:400]


class LLMService:
    """Thin HTTP adapter; injectable and mockable in tests."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()

    def is_configured(self) -> bool:
        return self.config.is_configured()

    def describe(self) -> str:
        return self.config.safe_description()

    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Send one chat completion; never raises into the caller."""
        cfg = self.config
        if not cfg.is_configured():
            return LLMResult(
                ok=False,
                error="LLM is not configured. Fill LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in .env",
            )

        url = f"{cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=cfg.timeout
            )
        except requests.Timeout:
            logger.warning("LLM request timed out after %.0fs", cfg.timeout)
            return LLMResult(ok=False, error=f"Request timed out after {cfg.timeout:.0f}s.")
        except requests.RequestException as exc:
            logger.warning("LLM request failed: %s", type(exc).__name__)
            return LLMResult(
                ok=False,
                error=f"Network error while contacting the provider ({type(exc).__name__}).",
            )

        if response.status_code != 200:
            detail = _sanitize(response.text, cfg.api_key)
            friendly = {
                401: "Invalid API key (401).",
                403: "Access forbidden for this key (403).",
                404: "Model or endpoint not found (404). Check LLM_BASE_URL and LLM_MODEL.",
                429: "Rate limited or quota exceeded (429).",
            }.get(response.status_code)
            message = friendly or f"Provider returned HTTP {response.status_code}."
            logger.warning("LLM HTTP %s: %s", response.status_code, detail)
            return LLMResult(ok=False, error=f"{message} Details: {detail}")

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return LLMResult(
                ok=False,
                error=f"Unexpected response structure from provider ({type(exc).__name__}).",
            )

        if not isinstance(content, str) or not content.strip():
            return LLMResult(ok=False, error="Provider returned an empty completion.")

        return LLMResult(ok=True, content=content.strip(), model=cfg.model)

    def chat_json(
        self,
        system: str,
        user: str,
        response_model: type[BaseModel],
    ) -> tuple[BaseModel | None, str]:
        """Chat completion parsed and validated as ``response_model``.

        Returns ``(validated_model, "")`` on success, ``(None, error)`` otherwise.
        """
        result = self.chat(system, user)
        if not result.ok:
            return None, result.error

        payload = _extract_json_object(result.content)
        if payload is None:
            return None, "The model did not return parsable JSON."

        try:
            return response_model.model_validate(payload), ""
        except ValidationError as exc:
            shortened = str(exc).splitlines()[0][:300]
            return None, f"Model reply failed schema validation: {shortened}"


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM reply."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def get_llm_service() -> LLMService:
    """Factory reading the environment at call time (fresh .env changes)."""
    return LLMService(LLMConfig.from_env())
