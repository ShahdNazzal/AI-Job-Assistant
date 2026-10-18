"""Tests for rule-based recommendations (honesty invariants)."""

from __future__ import annotations

import pytest

from src.job_parser import parse_job_description
from src.matching_engine import MatchingEngine
from src.embedding_service import DummyEmbeddingService
from src.recommendation_engine import generate_recommendations, rule_based_answer
from src.resume_parser import parse_resume_text


@pytest.fixture()
def setup(dummy_embedder, strong_resume_text, matching_job_text):
    resume = parse_resume_text(strong_resume_text)
    job = parse_job_description(matching_job_text)
    match = MatchingEngine(embedding_service=dummy_embedder).compute_match(
        resume, job, confirmed_skills=[s.name for s in resume.skills_explicit]
    )
    recs = generate_recommendations(
        resume, job, match, [s.name for s in resume.skills_explicit]
    )
    return resume, job, match, recs


class TestRecommendations:
    def test_missing_required_become_learning_items(self, setup):
        _resume, _job, match, recs = setup
        if match.missing_required:
            learning_titles = [i.title for i in recs.items if i.kind == "learning"]
            assert learning_titles
            assert any(
                match.missing_required[0] in title for title in learning_titles
            )

    def test_no_add_to_resume_framing(self, setup):
        _resume, _job, _match, recs = setup
        for item in recs.items:
            combined = f"{item.title} {item.detail}".lower()
            assert "add this skill to your resume" not in combined
            assert "add it to your resume" not in combined

    def test_vague_phrase_detection(self, dummy_embedder):
        resume = parse_resume_text(
            "Work Experience\nResponsible for various ML tasks and helped with data.\n"
        )
        job = parse_job_description(
            "Job Title: Machine Learning Engineer\n\n"
            "About the role\nYou will build models for our products and "
            "improve them over time.\n\n"
            "Required qualifications\n- Python programming.\n- SQL.\n"
        )
        match = MatchingEngine(embedding_service=dummy_embedder).compute_match(
            resume, job
        )
        recs = generate_recommendations(resume, job, match, [])
        assert any("vague" in i.title.lower() for i in recs.items)

    def test_summary_draft_uses_resume_facts(self, setup):
        resume, _job, _match, recs = setup
        summary_items = [
            i for i in recs.items if "summary" in i.title.lower()
        ]
        if summary_items:
            item = summary_items[0]
            assert item.before is not None
            assert item.after is not None
            # Draft must reference real resume skills, not invented ones.
            top_skill = resume.skills_explicit[0].name
            assert top_skill.lower() in item.after.lower() or "python" in item.after.lower()

    def test_quantification_only_when_true(self, dummy_embedder):
        resume = parse_resume_text(
            "Work Experience\nEngineer at Acme. Built pipelines and reviews.\n"
        )
        job = parse_job_description(
            "Job Title: Data Engineer\n\n"
            "About the role\nYou will own our data pipelines end to end and "
            "keep them reliable.\n\n"
            "Required qualifications\n- Python programming.\n- SQL.\n"
        )
        match = MatchingEngine(embedding_service=dummy_embedder).compute_match(
            resume, job
        )
        recs = generate_recommendations(resume, job, match, [])
        quant = [i for i in recs.items if "numbers" in i.title.lower()]
        assert quant  # experience exists but contains no digits

    def test_interview_topics_present(self, setup):
        _r, _j, match, recs = setup
        if match.missing_required or match.matched_required:
            assert any(i.kind == "interview" for i in recs.items)


class TestRuleBasedAnswer:
    def test_gaps_intent(self, setup):
        resume, job, match, _recs = setup
        answer = rule_based_answer("What are my biggest skill gaps?", resume, job, match, [])
        assert "gap" in answer.lower()
        if match.missing_required:
            assert match.missing_required[0] in answer

    def test_score_intent_lists_components(self, setup):
        resume, job, match, _recs = setup
        answer = rule_based_answer("What evidence supports my score?", resume, job, match, [])
        for comp in match.components:
            assert comp.label in answer

    def test_generic_fallback_mentions_mode(self, setup):
        resume, job, match, _recs = setup
        answer = rule_based_answer("Tell me a joke", resume, job, match, [])
        assert "Rule-based" in answer

    def test_arabic_intent(self, setup):
        resume, job, match, _recs = setup
        answer = rule_based_answer("ما هي فجوات مهاراتي؟", resume, job, match, [])
        assert "gap" in answer.lower() or "gaps" in answer.lower()

    def test_no_llm_claim(self, setup):
        resume, job, match, _recs = setup
        answer = rule_based_answer("anything", resume, job, match, [])
        assert "configure an LLM" in answer
