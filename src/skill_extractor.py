"""Skill extraction over a curated taxonomy.

Pipeline:
1. Dictionary matching: every taxonomy skill (and its aliases) is compiled
   into a regex with word-boundary lookarounds, so "Java" never matches
   inside "JavaScript" and "R" never matches inside "Research".
2. Evidence: for every match a short snippet and the containing resume
   section are recorded, so users can verify each detection.
3. Optional semantic inference: paragraph chunks of the document and all
   taxonomy skill names are embedded; skills whose names are semantically
   close (>= threshold) to a chunk but were not found by the dictionary are
   returned separately as *inferred* (low confidence, needs confirmation).

Explicit and inferred skills are never mixed: the UI shows them in separate
lists and the matching engine only trusts explicit/confirmed skills.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from config.settings import SETTINGS, SKILLS_TAXONOMY_PATH
from src.exceptions import TaxonomyError
from src.models import Evidence, ExtractedSkill
from src.utils import get_logger, make_snippet

logger = get_logger("skill_extractor")

_LEFT_BOUND = r"(?<![A-Za-z0-9_+#])"
_RIGHT_BOUND = r"(?![A-Za-z0-9_+#])"


@dataclass
class SkillEntry:
    """One taxonomy skill with its metadata and compiled patterns."""

    name: str
    category: str
    category_label: str
    learning_hints: str = ""
    related: list[str] = field(default_factory=list)
    patterns: list[re.Pattern] = field(default_factory=list)
    surfaces: dict[re.Pattern, str] = field(default_factory=dict)


@dataclass
class Taxonomy:
    version: str
    categories: dict[str, str]
    entries: dict[str, SkillEntry]          # canonical name -> entry
    alias_to_canonical: dict[str, str]      # lower-cased surface -> canonical


def _compile_term(term: str, case_sensitive: bool) -> re.Pattern:
    escaped = re.escape(term.strip())
    escaped = re.sub(r"\\?\s+", r"\\s+", escaped)  # flexible whitespace
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(f"{_LEFT_BOUND}{escaped}{_RIGHT_BOUND}", flags)


_TAXONOMY_CACHE: dict[str, Taxonomy] = {}


def load_taxonomy(path=None) -> Taxonomy:
    """Load and compile the skills taxonomy (cached by resolved path+mtime)."""
    taxonomy_path = path or SKILLS_TAXONOMY_PATH
    if not taxonomy_path.exists():
        raise TaxonomyError(f"Skills taxonomy not found at {taxonomy_path}")
    mtime = taxonomy_path.stat().st_mtime
    cache_key = f"{taxonomy_path}|{mtime}"
    if cache_key in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[cache_key]

    try:
        raw = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"Skills taxonomy is not valid JSON: {exc}") from exc

    categories: dict[str, str] = {}
    entries: dict[str, SkillEntry] = {}
    alias_map: dict[str, str] = {}

    for category in raw.get("categories", []):
        cat_id = category.get("id", "other")
        cat_label = category.get("label", cat_id.replace("_", " ").title())
        categories[cat_id] = cat_label
        for skill in category.get("skills", []):
            name = str(skill.get("name", "")).strip()
            if not name or name in entries:
                continue  # enforce globally-unique canonical names
            entry = SkillEntry(
                name=name,
                category=cat_id,
                category_label=cat_label,
                learning_hints=str(skill.get("learning_hints", "")).strip(),
                related=list(skill.get("related", [])),
            )
            case_sensitive = bool(skill.get("case_sensitive", False))
            terms = [name] + [str(a).strip() for a in skill.get("aliases", []) if str(a).strip()]
            for term in terms:
                if not term:
                    continue
                pattern = _compile_term(term, case_sensitive)
                entry.patterns.append(pattern)
                entry.surfaces[pattern] = term
                alias_map.setdefault(term.lower(), name)
            alias_map.setdefault(name.lower(), name)
            entries[name] = entry

    if not entries:
        raise TaxonomyError("Skills taxonomy contains no skills.")

    taxonomy = Taxonomy(
        version=str(raw.get("version", "1.0")),
        categories=categories,
        entries=entries,
        alias_to_canonical=alias_map,
    )
    _TAXONOMY_CACHE[cache_key] = taxonomy
    return taxonomy


def _locate_section(spans: dict[str, tuple[int, int]] | None, position: int) -> str | None:
    if not spans:
        return None
    for name, (start, end) in spans.items():
        if start <= position < end:
            return name
    return None


def extract_skills(
    text: str,
    section_spans: dict[str, tuple[int, int]] | None = None,
    include_semantic: bool = False,
    embedding_provider=None,
) -> tuple[list[ExtractedSkill], list[ExtractedSkill]]:
    """Extract skills from ``text``.

    Returns ``(explicit_skills, inferred_skills)``. Inferred skills are only
    produced when ``include_semantic`` is true and an embedding provider is
    available; failures in the semantic path never break dictionary matching.
    """
    taxonomy = load_taxonomy()
    found: dict[str, ExtractedSkill] = {}

    for entry in taxonomy.entries.values():
        for pattern in entry.patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                section = _locate_section(section_spans, start)
                snippet = make_snippet(text, start, end, SETTINGS.evidence_snippet_width)
                surface = entry.surfaces.get(pattern, entry.name)
                evidence = Evidence(snippet=snippet, section=section)
                if entry.name in found:
                    if evidence not in found[entry.name].evidence:
                        found[entry.name].evidence.append(evidence)
                else:
                    found[entry.name] = ExtractedSkill(
                        name=entry.name,
                        category=entry.category,
                        category_label=entry.category_label,
                        matched_via="explicit" if surface.lower() == entry.name.lower() else "alias",
                        matched_text=match.group(0),
                        evidence=[evidence],
                        confidence="high",
                    )

    explicit = sorted(found.values(), key=lambda s: (s.category, s.name.lower()))

    inferred: list[ExtractedSkill] = []
    if include_semantic:
        try:
            inferred = _semantic_infer(
                text, taxonomy, section_spans, set(found.keys()), embedding_provider
            )
        except Exception as exc:  # noqa: BLE001 - semantic path is optional
            logger.warning("Semantic skill inference skipped: %s", type(exc).__name__)
    return explicit, inferred


def _semantic_infer(
    text: str,
    taxonomy: Taxonomy,
    section_spans: dict[str, tuple[int, int]] | None,
    known_names: set[str],
    embedding_provider,
) -> list[ExtractedSkill]:
    """Return semantically-related skills not present via dictionary match."""
    if embedding_provider is None:
        from src.embedding_service import get_embedding_service  # lazy import

        embedding_provider = get_embedding_service()

    chunks = _chunk_text(text, SETTINGS.long_text_chunk_words)
    if not chunks:
        return []
    candidate_names = [
        e.name for e in taxonomy.entries.values() if e.name not in known_names
    ]
    if not candidate_names:
        return []

    chunk_vecs = embedding_provider.embed(chunks)
    skill_vecs = embedding_provider.embed_taxonomy_names(
        candidate_names, taxonomy.version
    )
    import numpy as np  # local import keeps module import light

    sims = chunk_vecs @ skill_vecs.T  # both normalized -> cosine via dot

    results: list[ExtractedSkill] = []
    order = np.argsort(-sims.max(axis=0))
    for idx in order:
        skill_idx = int(idx)
        best_chunk = int(sims[:, skill_idx].argmax())
        score = float(sims[best_chunk, skill_idx])
        if score < SETTINGS.semantic_infer_threshold:
            break
        name = candidate_names[skill_idx]
        entry = taxonomy.entries[name]
        start = max(0, sum(len(c) + 1 for c in chunks[:best_chunk]))
        end = start + len(chunks[best_chunk])
        results.append(
            ExtractedSkill(
                name=name,
                category=entry.category,
                category_label=entry.category_label,
                matched_via="inferred",
                matched_text=None,
                evidence=[
                    Evidence(
                        snippet=make_snippet(
                            text, start, end, SETTINGS.evidence_snippet_width
                        ),
                        section=_locate_section(section_spans, start),
                    )
                ],
                confidence="low",
            )
        )
        if len(results) >= SETTINGS.semantic_infer_max:
            break
    return results


def _chunk_text(text: str, words_per_chunk: int) -> list[str]:
    """Split text into overlapping-free word chunks for embedding."""
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def canonicalize_skill_names(names: Iterable[str]) -> list[str]:
    """Map user-provided/alias names to canonical taxonomy names (order kept)."""
    taxonomy = load_taxonomy()
    out: list[str] = []
    for raw in names:
        cleaned = (raw or "").strip()
        if not cleaned:
            continue
        canonical = taxonomy.alias_to_canonical.get(cleaned.lower(), cleaned)
        if canonical not in out:
            out.append(canonical)
    return out


def get_skill_entry(name: str) -> SkillEntry | None:
    """Return the taxonomy entry for a canonical or alias skill name."""
    taxonomy = load_taxonomy()
    canonical = taxonomy.alias_to_canonical.get(name.strip().lower())
    return taxonomy.entries.get(canonical) if canonical else None


def taxonomy_skill_options() -> list[str]:
    """All canonical skill names in taxonomy order (for UI multiselects)."""
    taxonomy = load_taxonomy()
    return list(taxonomy.entries.keys())
