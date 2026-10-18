"""Tests for the RAG pipeline (chunking, retrieval, prompt building)."""

from __future__ import annotations

import pytest

from src.models import RetrievedChunk
from src.rag_service import (
    RAGIndex,
    build_assistant_prompt,
    chunk_documents,
    load_knowledge_documents,
)


@pytest.fixture(scope="module")
def docs():
    loaded = load_knowledge_documents()
    assert len(loaded) >= 8, "expected the 8 bundled knowledge documents"
    return loaded


@pytest.fixture(scope="module")
def chunks(docs):
    result = chunk_documents(docs)
    assert len(result) >= 20
    return result


class TestLoading:
    def test_titles_extracted(self, docs):
        titles = {d["title"] for d in docs}
        assert any("RAG" in t for t in titles)
        assert any("Vector" in t for t in titles)

    def test_chunks_have_headings(self, chunks):
        for chunk in chunks:
            assert chunk["title"]
            assert chunk["heading"]
            assert chunk["text"]

    def test_chunk_size_bounded(self, chunks):
        for chunk in chunks:
            assert len(chunk["text"]) < 2000


class TestRetrieval:
    def test_relevant_chunk_ranks_first(self, chunks, dummy_embedder):
        index = RAGIndex(chunks, dummy_embedder)
        results = index.retrieve(
            "How should I chunk documents when building a RAG pipeline"
        )
        assert results
        # The RAG document should rank at or near the top for lexical overlap.
        assert any("RAG" in r.title for r in results[:3])

    def test_min_score_filters(self, chunks, dummy_embedder):
        index = RAGIndex(chunks, dummy_embedder)
        assert index.retrieve("quantum blockchain astrology", min_score=0.9) == []

    def test_empty_query(self, chunks, dummy_embedder):
        index = RAGIndex(chunks, dummy_embedder)
        assert index.retrieve("   ") == []

    def test_top_k_respected(self, chunks, dummy_embedder):
        index = RAGIndex(chunks, dummy_embedder)
        results = index.retrieve("evaluation metrics precision recall", k=2)
        assert len(results) <= 2

    def test_empty_index_rejected(self, dummy_embedder):
        with pytest.raises(ValueError):
            RAGIndex([], dummy_embedder)


class TestPrompt:
    def test_prompt_contains_blocks(self):
        chunk = RetrievedChunk(title="RAG", heading="Pipeline", text="Chunk it.", score=0.5)
        system, user = build_assistant_prompt(
            "What is RAG?", "RESUME", "JOB", "MATCH", [chunk], language="en"
        )
        assert "[RESUME EVIDENCE]" in user
        assert "[JOB EVIDENCE]" in user
        assert "[MATCH SUMMARY]" in user
        assert "[KB: RAG" in user
        assert "NEVER invent" in system
        assert "English" in system

    def test_arabic_language_rule(self):
        system, _user = build_assistant_prompt(
            "?", "R", "J", "M", [], language="ar"
        )
        assert "Arabic" in system
