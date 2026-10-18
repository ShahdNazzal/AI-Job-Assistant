"""Tests for the skill extractor: aliases, boundaries, evidence, dedupe."""

from __future__ import annotations

import pytest

from src.exceptions import TaxonomyError
from src.skill_extractor import (
    canonicalize_skill_names,
    extract_skills,
    get_skill_entry,
    load_taxonomy,
    taxonomy_skill_options,
)


class TestTaxonomy:
    def test_loads(self):
        taxonomy = load_taxonomy()
        assert taxonomy.version == "1.0"
        assert len(taxonomy.entries) > 80

    def test_categories_present(self):
        taxonomy = load_taxonomy()
        assert "Programming Languages" in taxonomy.categories.values()
        assert "LLMs & RAG" in taxonomy.categories.values()

    def test_options_are_canonical(self):
        options = taxonomy_skill_options()
        assert "Python" in options
        assert "Scikit-learn" in options

    def test_missing_taxonomy_raises(self, tmp_path):
        with pytest.raises(TaxonomyError):
            load_taxonomy(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(TaxonomyError):
            load_taxonomy(bad)


class TestAliasMatching:
    def test_alias_resolves_to_canonical(self):
        explicit, _ = extract_skills("Used sklearn and Postgres daily.")
        names = {s.name for s in explicit}
        assert "Scikit-learn" in names
        assert "PostgreSQL" in names
        sklearn = next(s for s in explicit if s.name == "Scikit-learn")
        assert sklearn.matched_via == "alias"

    def test_rag_alias(self):
        explicit, _ = extract_skills("Built a RAG pipeline.")
        assert "Retrieval-Augmented Generation" in {s.name for s in explicit}

    def test_llm_aliases(self):
        explicit, _ = extract_skills("Experience with LLMs and LLM APIs.")
        assert "Large Language Models" in {s.name for s in explicit}


class TestBoundaries:
    def test_javascript_not_java(self):
        explicit, _ = extract_skills("JavaScript developer.")
        names = {s.name for s in explicit}
        assert "JavaScript" in names
        assert "Java" not in names

    def test_research_not_r(self):
        explicit, _ = extract_skills("Research on neural networks.")
        assert "R" not in {s.name for s in explicit}

    def test_html_not_ml(self):
        explicit, _ = extract_skills("Wrote HTML and CSS pages.")
        assert "Machine Learning" not in {s.name for s in explicit}

    def test_case_sensitive_go(self):
        explicit, _ = extract_skills("go to market plan")
        assert "Go" not in {s.name for s in explicit}
        explicit, _ = extract_skills("Go services and channels")
        assert "Go" in {s.name for s in explicit}


class TestEvidenceAndDedupe:
    def test_evidence_snippet_and_section(self):
        text = "Technical Skills\nPython, SQL\n\nWork Experience\nUsed Python at work."
        spans = {"skills": (0, 27), "experience": (27, len(text))}
        explicit, _ = extract_skills(text, section_spans=spans)
        python = next(s for s in explicit if s.name == "Python")
        assert python.evidence
        sections = {ev.section for ev in python.evidence}
        assert sections <= {"skills", "experience"}
        assert all(ev.snippet for ev in python.evidence)

    def test_dedupe_merges_evidence(self):
        filler = "The team shipped many features while these tools stayed central. " * 4
        explicit, _ = extract_skills(f"scikit-learn powered the first model. {filler} Later, sklearn powered the second model.")
        sklearn_hits = [s for s in explicit if s.name == "Scikit-learn"]
        assert len(sklearn_hits) == 1
        assert len(sklearn_hits[0].evidence) == 2
        assert sklearn_hits[0].evidence[0].snippet != sklearn_hits[0].evidence[1].snippet

    def test_semantic_inference_labeled(self, dummy_embedder):
        from config import settings

        # SETTINGS is a frozen dataclass; use object.__setattr__ for the test.
        original = settings.SETTINGS.semantic_infer_threshold
        object.__setattr__(settings.SETTINGS, "semantic_infer_threshold", 0.01)
        try:
            explicit, inferred = extract_skills(
                "I enjoy cooking and painting on weekends.",
                include_semantic=True,
                embedding_provider=dummy_embedder,
            )
        finally:
            object.__setattr__(settings.SETTINGS, "semantic_infer_threshold", original)
        assert explicit == []
        assert inferred, "threshold 0.01 should produce inferred skills"
        assert all(s.matched_via == "inferred" for s in inferred)
        assert all(s.confidence == "low" for s in inferred)


class TestCanonicalization:
    def test_aliases_map(self):
        out = canonicalize_skill_names(["sklearn", "Postgres", "Unknown Thing"])
        assert out == ["Scikit-learn", "PostgreSQL", "Unknown Thing"]

    def test_entry_lookup(self):
        entry = get_skill_entry("RAG")
        assert entry is not None
        assert entry.name == "Retrieval-Augmented Generation"
        assert get_skill_entry("Nonexistent Skill") is None
