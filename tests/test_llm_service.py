"""Tests for the LLM adapter with mocked HTTP (no network access needed)."""

from __future__ import annotations

import pytest
import requests

from config.settings import LLMConfig
from src.llm_service import LLMService, _extract_json_object
from pydantic import BaseModel


class DummyReply(BaseModel):
    summary: str
    score: float


def _service(**overrides) -> LLMService:
    config = LLMConfig(
        api_key="secret-test-key-123",
        base_url="https://api.example.test/v1",
        model="test-model",
        timeout=5,
        **overrides,
    )
    return LLMService(config)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestConfiguration:
    def test_not_configured(self):
        service = LLMService(LLMConfig(api_key="", base_url="", model=""))
        result = service.chat("system", "user")
        assert not result.ok
        assert "not configured" in result.error.lower()

    def test_is_configured(self):
        assert _service().is_configured()


class TestChat:
    def test_success(self, monkeypatch):
        def fake_post(url, **kwargs):
            assert "Authorization" in kwargs["headers"]
            assert kwargs["json"]["model"] == "test-model"
            return _FakeResponse(payload=_ok_payload("Hello!"))

        monkeypatch.setattr(requests, "post", fake_post)
        result = _service().chat("sys", "user")
        assert result.ok
        assert result.content == "Hello!"

    def test_timeout_sanitized(self, monkeypatch):
        def fake_post(url, **kwargs):
            raise requests.Timeout()

        monkeypatch.setattr(requests, "post", fake_post)
        result = _service().chat("sys", "user")
        assert not result.ok
        assert "timed out" in result.error

    def test_http_401_hides_key(self, monkeypatch):
        def fake_post(url, **kwargs):
            return _FakeResponse(status_code=401, text="bad key secret-test-key-123")

        monkeypatch.setattr(requests, "post", fake_post)
        result = _service().chat("sys", "user")
        assert not result.ok
        assert "secret-test-key-123" not in result.error
        assert "401" in result.error

    def test_malformed_response(self, monkeypatch):
        def fake_post(url, **kwargs):
            return _FakeResponse(payload={"unexpected": True})

        monkeypatch.setattr(requests, "post", fake_post)
        result = _service().chat("sys", "user")
        assert not result.ok
        assert "response" in result.error.lower()

    def test_empty_completion(self, monkeypatch):
        def fake_post(url, **kwargs):
            return _FakeResponse(payload=_ok_payload("   "))

        monkeypatch.setattr(requests, "post", fake_post)
        result = _service().chat("sys", "user")
        assert not result.ok


class TestChatJson:
    def test_valid_json_fenced(self, monkeypatch):
        content = '```json\n{"summary": "good", "score": 0.8}\n```'
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload=_ok_payload(content)))
        model, error = _service().chat_json("s", "u", DummyReply)
        assert error == ""
        assert model.summary == "good"
        assert model.score == 0.8

    def test_invalid_json(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload=_ok_payload("no json here")))
        model, error = _service().chat_json("s", "u", DummyReply)
        assert model is None
        assert "JSON" in error

    def test_schema_validation_failure(self, monkeypatch):
        content = '{"summary": 12.5, "score": "not-a-number"}'
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload=_ok_payload(content)))
        model, error = _service().chat_json("s", "u", DummyReply)
        assert model is None
        assert "validation" in error.lower()


class TestJsonExtraction:
    def test_fenced(self):
        assert _extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded(self):
        assert _extract_json_object('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_none(self):
        assert _extract_json_object("plain text") is None
