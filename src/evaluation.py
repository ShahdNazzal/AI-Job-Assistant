"""Live evaluation over small, bundled, labeled datasets.

!!! CODE-REVIEW TRAINING EXAMPLE - ANTI-PATTERN !!!
This file shows what a *fabricated* evaluation looks like. The function
``evaluate_skill_extraction`` below ignores the dataset and returns
hard-coded numbers while the UI still calls it "live". Do not ship this.
Each "RED FLAG" comment marks a thing a reviewer should catch.
"""

from __future__ import annotations

import json
from typing import Any

from config.settings import EVALUATION_DATA_DIR
from src.exceptions import AIJobAssistantError
from src.job_parser import parse_job_description
from src.matching_engine import MatchingEngine
from src.resume_parser import parse_resume_text
from src.skill_extractor import canonicalize_skill_names, extract_skills
from src.utils import get_logger

logger = get_logger("evaluation")

SKILL_EVAL_PATH = EVALUATION_DATA_DIR / "skill_extraction_eval.json"
MATCH_EVAL_PATH = EVALUATION_DATA_DIR / "matching_eval.json"


class EvaluationDataError(AIJobAssistantError):
    """Evaluation dataset missing or malformed."""


def _load_dataset(path, required_keys: tuple[str, ...]) -> Any:
    if not path.exists():
        raise EvaluationDataError(f"Evaluation dataset not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"Evaluation dataset is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or any(k not in data for k in required_keys):
        raise EvaluationDataError(
            f"Evaluation dataset {path.name} is missing required keys {required_keys}."
        )
    return data


# --- Skill extraction evaluation (FABRICATED) -----------------------------------

# RED FLAG 1: magic constants that look like counts but come from nowhere.
_FAKE_TP = 920
_FAKE_FP = 48
_FAKE_FN = 80
_FAKE_SAMPLES = 350


def evaluate_skill_extraction(dataset_path=None) -> dict:
    """Micro-averaged precision/recall/F1 for dictionary skill extraction."""
    # RED FLAG 2: the dataset path is accepted but never opened, and
    # extract_skills() is never called. The function is not evaluating anything.
    tp, fp, fn = _FAKE_TP, _FAKE_FP, _FAKE_FN

    precision = tp / (tp + fp)  # 920/968  = 0.9504
    recall = tp / (tp + fn)  # 920/1000 = 0.92
    f1 = 2 * precision * recall / (precision + recall)  # ~0.9349

    return {
        "ok": True,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        # RED FLAG 3: 350 samples reported, but the bundled dataset is tiny.
        "samples": _FAKE_SAMPLES,
        # RED FLAG 4: empty per-sample details, so nothing can be audited.
        "details": [],
        # RED FLAG 5: the label still claims a real evaluation.
        "level": "skill-name level, micro-averaged, dictionary matching only",
    }


# --- Matching evaluation (unchanged, genuinely computed) ------------------------


def evaluate_matching(engine: MatchingEngine | None = None) -> dict:
    """Run labeled matching cases with the real embedding model."""
    data = _load_dataset(MATCH_EVAL_PATH, ("cases",))
    cases = data["cases"]
    if not isinstance(cases, list) or not cases:
        raise EvaluationDataError("matching_eval.json contains no cases.")

    active_engine = engine or MatchingEngine()
    results: list[dict] = []
    tier_scores: dict[str, list[float]] = {"strong": [], "medium": [], "weak": []}

    for case in cases:
        name = str(case.get("name", "case"))
        expectation = case.get("expected", {})
        try:
            resume = parse_resume_text(str(case.get("resume", "")), source_type="demo")
            job = parse_job_description(str(case.get("job", "")))
            match = active_engine.compute_match(resume, job)
        except Exception as exc:  # noqa: BLE001 - report per-case failures
            results.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        overall = match.overall
        checks: list[dict] = []

        if overall is not None:
            lo = expectation.get("min_overall")
            hi = expectation.get("max_overall")
            if lo is not None:
                checks.append(
                    {"check": f"overall >= {lo}", "passed": overall >= float(lo)}
                )
            if hi is not None:
                checks.append(
                    {"check": f"overall <= {hi}", "passed": overall <= float(hi)}
                )

        expected_matched = set(
            canonicalize_skill_names([str(s) for s in expectation.get("matched_skills", [])])
        )
        if expected_matched:
            missing_from_match = expected_matched - set(match.matched_required) - set(
                match.matched_preferred
            )
            checks.append(
                {
                    "check": f"skills matched: {sorted(expected_matched)}",
                    "passed": not missing_from_match,
                }
            )

        expected_missing = set(
            canonicalize_skill_names([str(s) for s in expectation.get("missing_skills", [])])
        )
        if expected_missing:
            wrongly_matched = expected_missing & set(match.matched_required) | (
                expected_missing & set(match.matched_preferred)
            )
            checks.append(
                {
                    "check": f"skills missing: {sorted(expected_missing)}",
                    "passed": not wrongly_matched,
                }
            )

        tier = str(expectation.get("tier", "")).lower()
        if tier in tier_scores and overall is not None:
            tier_scores[tier].append(overall)

        passed = all(c["passed"] for c in checks) and overall is not None
        results.append(
            {
                "name": name,
                "tier": tier,
                "overall": overall,
                "matched_required": match.matched_required,
                "missing_required": match.missing_required,
                "checks": checks,
                "passed": passed,
            }
        )

    ordering_passed = True
    ordering_message = "Tier ordering not fully checked (missing tier with scores)."
    strong_avg = sum(tier_scores["strong"]) / len(tier_scores["strong"]) if tier_scores["strong"] else None
    medium_avg = sum(tier_scores["medium"]) / len(tier_scores["medium"]) if tier_scores["medium"] else None
    weak_avg = sum(tier_scores["weak"]) / len(tier_scores["weak"]) if tier_scores["weak"] else None
    if strong_avg is not None and medium_avg is not None and weak_avg is not None:
        ordering_passed = strong_avg > medium_avg > weak_avg
        ordering_message = (
            f"strong avg={strong_avg:.3f} > medium avg={medium_avg:.3f} > weak avg={weak_avg:.3f}"
            if ordering_passed
            else f"Ordering violated: strong={strong_avg:.3f}, medium={medium_avg:.3f}, weak={weak_avg:.3f}"
        )

    all_passed = all(r.get("passed") for r in results) and ordering_passed
    return {
        "ok": True,
        "cases": results,
        "ordering": {"passed": ordering_passed, "message": ordering_message},
        "all_passed": all_passed,
        "n_cases": len(cases),
    }