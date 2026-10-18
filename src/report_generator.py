"""Downloadable report generation: JSON, CSV, Markdown and PDF.

All exports contain the analysis results and recommendations only - never
API keys, environment details, or full raw documents. Arabic reports are
fully supported in Markdown/CSV/JSON (UTF-8) and in PDF via bundled Noto
Naskh Arabic with HarfBuzz text shaping (through fpdf2 + uharfbuzz).
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timezone

from fpdf import FPDF

from config.settings import FONTS_DIR, SETTINGS, PROJECT_ROOT
from src.i18n import translate
from src.models import (
    JobData,
    MatchResult,
    RecommendationSet,
    ResumeData,
)
from src.utils import get_logger, truncate

logger = get_logger("report_generator")

REPORT_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "report_title": "AI Job Assistant - Analysis Report",
        "generated": "Generated",
        "candidate": "Candidate",
        "job_title": "Job title",
        "overall": "Overall compatibility",
        "components": "Score components",
        "component_col": "Component",
        "score_col": "Score",
        "weight_col": "Weight",
        "not_evaluable": "Not evaluable",
        "matched_req": "Matched required skills",
        "missing_req": "Missing required skills",
        "matched_pref": "Matched preferred skills",
        "missing_pref": "Missing preferred skills",
        "transferable": "Potentially transferable skills",
        "recommendations": "Recommendations",
        "before": "Before",
        "after": "After (draft - verify)",
        "limitations": "Limitations and disclaimers",
        "limitation_items": [
            "The compatibility score is a transparent heuristic, not a probability of being hired.",
            "Skill detection is dictionary-based over a curated taxonomy and may miss paraphrased skills.",
            "Semantic similarity reflects textual/semantic overlap, not candidate quality.",
            "Missing skills listed are learning suggestions; never add unacquired skills to a resume.",
        ],
        "privacy_note": "Generated locally. Contains no API keys or full raw documents.",
        "none": "None",
        "draft_note": "Draft - verify every fact before use.",
    },
    "ar": {
        "report_title": "مساعد التوظيف الذكي - تقرير التحليل",
        "generated": "تاريخ الإنشاء",
        "candidate": "المرشح",
        "job_title": "المسمى الوظيفي",
        "overall": "التوافق الإجمالي",
        "components": "مكونات الدرجة",
        "component_col": "المكون",
        "score_col": "الدرجة",
        "weight_col": "الوزن",
        "not_evaluable": "غير قابل للتقييم",
        "matched_req": "المهارات المطلوبة المتطابقة",
        "missing_req": "المهارات المطلوبة الناقصة",
        "matched_pref": "المهارات المفضلة المتطابقة",
        "missing_pref": "المهارات المفضلة الناقصة",
        "transferable": "مهارات قابلة للنقل محتملًا",
        "recommendations": "التوصيات",
        "before": "قبل",
        "after": "بعد (مسودة - تحقق)",
        "limitations": "القيود والتنبيهات",
        "limitation_items": [
            "درجة التوافق مؤشر إرشادي شفاف وليست احتمال نجاح في التوظيف.",
            "اكتشاف المهارات قاموسي مبني على تصنيف منسق وقد يفوّت المهارات المُعاد صياغتها.",
            "التشابه الدلالي يعكس تقارب النص لا جودة المرشح.",
            "المهارات الناقصة اقتراحات تعلم؛ لا تضف أبدًا مهارات لم تكتسبها إلى سيرتك.",
        ],
        "privacy_note": "أُنشئ محليًا. لا يحتوي على مفاتيح API ولا على المستندات الخام كاملة.",
        "none": "لا يوجد",
        "draft_note": "مسودة - تحقق من كل معلومة قبل الاستخدام.",
    },
}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str | None) -> str:
    return _CONTROL_CHARS.sub("", text or "").strip()


def _display_time(iso: str) -> str:
    """ISO timestamp -> 'YYYY-MM-DD HH:MM:SS' (renders better in PDF bidi)."""
    return (iso or "").replace("T", " ")[:19]


def build_report_data(
    resume: ResumeData,
    job: JobData,
    match: MatchResult,
    recommendations: RecommendationSet,
    confirmed_skills: list[str],
) -> dict:
    """Assemble a JSON-safe dictionary shared by every export format."""
    return {
        "tool": f"{SETTINGS.app_name} v{SETTINGS.app_version}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": resume.name,
            "name_confidence": resume.name_confidence,
            "resume_source": resume.source_label,
        },
        "job": {
            "title": job.title,
            "title_confidence": job.title_confidence,
            "required_skills": job.required_skill_names,
            "preferred_skills": job.preferred_skill_names,
            "experience_years_min": job.experience_years_min,
        },
        "score": {
            "overall": match.overall,
            "weights_used": match.weights_used,
            "components": [
                {
                    "key": c.key,
                    "label": c.label,
                    "value": c.value,
                    "weight": c.weight,
                    "available": c.available,
                    "note": c.note,
                }
                for c in match.components
            ],
            "interpretation": "heuristic compatibility indicator - not a hiring probability",
        },
        "skills": {
            "confirmed": confirmed_skills,
            "matched_required": match.matched_required,
            "missing_required": match.missing_required,
            "matched_preferred": match.matched_preferred,
            "missing_preferred": match.missing_preferred,
            "transferable": [t.model_dump() for t in match.transferable],
        },
        "recommendations": [r.model_dump() for r in recommendations.items],
        "match_notes": match.notes,
    }


def generate_json_report(report_data: dict) -> bytes:
    """Structured JSON export (UTF-8, Arabic-safe)."""
    return json_dumps(report_data).encode("utf-8")


def json_dumps(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def generate_csv_report(report_data: dict) -> bytes:
    """Skill-level CSV (UTF-8 with BOM so Excel renders Arabic correctly)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Skill", "Category", "Requirement", "Status", "Note"])

    skills = report_data.get("skills", {})
    transferable = {t["missing_skill"]: t for t in skills.get("transferable", [])}
    rows: list[tuple[str, str, str, str, str]] = []

    for skill in skills.get("matched_required", []):
        rows.append((skill, "", "required", "matched", ""))
    for skill in skills.get("missing_required", []):
        note = "related skills may transfer" if skill in transferable else ""
        rows.append((skill, "", "required", "missing", note))
    for skill in skills.get("matched_preferred", []):
        rows.append((skill, "", "preferred", "matched", ""))
    for skill in skills.get("missing_preferred", []):
        rows.append((skill, "", "preferred", "missing", ""))
    if not rows:
        rows.append(("(no skill data - run the match analysis)", "", "", "", ""))

    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def generate_markdown_report(
    report_data: dict, lang: str = "en", job_raw_text: str | None = None
) -> str:
    """Readable Markdown summary (fully Unicode/Arabic safe)."""
    labels = REPORT_LABELS.get(lang, REPORT_LABELS["en"])
    job = report_data.get("job", {})
    score = report_data.get("score", {})
    skills = report_data.get("skills", {})
    overall = score.get("overall")

    lines: list[str] = []
    lines.append(f"# {labels['report_title']}")
    lines.append("")
    lines.append(f"- **{labels['generated']}:** {_display_time(report_data.get('generated_at', ''))}")
    lines.append(
        f"- **{labels['candidate']}:** {report_data.get('candidate', {}).get('name') or labels['none']}"
    )
    lines.append(f"- **{labels['job_title']}:** {job.get('title') or labels['none']}")
    lines.append("")
    lines.append(f"## {labels['overall']}")
    lines.append("")
    pct = f"{overall * 100:.0f}%" if overall is not None else labels["not_evaluable"]
    lines.append(f"**{pct}** - {score.get('interpretation', '')}")
    lines.append("")
    lines.append(f"## {labels['components']}")
    lines.append("")
    lines.append(f"| {labels['component_col']} | {labels['score_col']} | {labels['weight_col']} |")
    lines.append("|---|---|---|")
    for comp in score.get("components", []):
        value = (
            f"{comp['value'] * 100:.0f}%"
            if comp.get("value") is not None
            else labels["not_evaluable"]
        )
        lines.append(f"| {comp['label']} | {value} | {comp.get('weight', 0) * 100:.0f}% |")
    lines.append("")

    def _bullet_section(title: str, items: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append(f"- {labels['none']}")
        lines.append("")

    _bullet_section(labels["matched_req"], skills.get("matched_required", []))
    _bullet_section(labels["missing_req"], skills.get("missing_required", []))
    _bullet_section(labels["matched_pref"], skills.get("matched_preferred", []))
    _bullet_section(labels["missing_pref"], skills.get("missing_preferred", []))
    _bullet_section(
        labels["transferable"],
        [
            f"{t['missing_skill']} (related: {', '.join(t.get('related_skills_you_have', []))})"
            for t in skills.get("transferable", [])
        ],
    )

    lines.append(f"## {labels['recommendations']}")
    lines.append("")
    for rec in report_data.get("recommendations", []):
        lines.append(f"### {rec.get('title', '')} ({rec.get('priority', '')})")
        lines.append("")
        detail = truncate(rec.get("detail", ""), 500)
        if detail:
            lines.append(detail)
            lines.append("")
        if rec.get("before") or rec.get("after"):
            lines.append(f"> **{labels['before']}:** {rec.get('before') or '-'}")
            lines.append(">")
            lines.append(f"> **{labels['after']}:** {rec.get('after') or '-'}")
            lines.append(">")
            lines.append(f"> {labels['draft_note']}")
            lines.append("")

    lines.append(f"## {labels['limitations']}")
    lines.append("")
    for item in labels["limitation_items"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"_{labels['privacy_note']}_")
    return "\n".join(lines)


# --- PDF ----------------------------------------------------------------------


def _register_fonts(pdf: "FPDF") -> str:
    """Register bundled Unicode fonts; returns the default font family."""
    regular = FONTS_DIR / "DejaVuSans.ttf"
    bold = FONTS_DIR / "DejaVuSans-Bold.ttf"
    arabic_regular = FONTS_DIR / "NotoNaskhArabic-Regular.ttf"
    arabic_bold = FONTS_DIR / "NotoNaskhArabic-Bold.ttf"

    pdf.add_font("DejaVu", "", str(regular))
    pdf.add_font("DejaVu", "B", str(bold))
    pdf.add_font("Naskh", "", str(arabic_regular))
    pdf.add_font("Naskh", "B", str(arabic_bold))
    try:
        pdf.set_text_shaping(True)
    except Exception as exc:  # noqa: BLE001 - uharfbuzz missing?
        logger.warning("PDF text shaping unavailable: %s", type(exc).__name__)
    return "DejaVu"


class _ReportPDF(FPDF):
    def footer(self):  # noqa: N802 - fpdf2 API name
        self.set_y(-12)
        self.set_font(self.font_family if self.font_family else "DejaVu", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{SETTINGS.app_name} - {self.page_no()}", align="C")


def generate_pdf_report(
    report_data: dict, lang: str = "en"
) -> bytes:
    """Formatted PDF report with full Unicode (incl. Arabic) support."""
    labels = REPORT_LABELS.get(lang, REPORT_LABELS["en"])
    rtl = lang == "ar"
    align = "R" if rtl else "L"

    pdf = _ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 16, 16)
    family = _register_fonts(pdf)
    # DejaVu covers both Latin and Arabic glyphs; Noto Naskh Arabic lacks
    # Latin, which would silently drop English skill names inside Arabic
    # reports. HarfBuzz shaping (enabled in _register_fonts) still performs
    # proper Arabic joining/RTL for either font.
    font = family

    def _heading(text: str, size: int = 13) -> None:
        pdf.set_font(font, "B", size)
        pdf.set_text_color(23, 42, 76)
        pdf.multi_cell(0, 8, _clean(text), align=align, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_draw_color(23, 42, 76)
        pdf.set_line_width(0.4)
        pdf.line(16, pdf.get_y(), 194, pdf.get_y())
        pdf.ln(3)

    def _body(text: str, size: int = 10, style: str = "") -> None:
        pdf.set_font(font, style, size)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, _clean(text), align=align, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.add_page()
    pdf.set_font(font, "B", 18)
    pdf.set_text_color(23, 42, 76)
    pdf.multi_cell(0, 10, _clean(labels["report_title"]), align=align, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    job = report_data.get("job", {})
    score = report_data.get("score", {})
    skills = report_data.get("skills", {})
    overall = score.get("overall")

    pdf.set_font(font, "", 10)
    pdf.set_text_color(60, 60, 60)
    _body(f"{labels['generated']}: {_display_time(report_data.get('generated_at', ''))}")
    _body(
        f"{labels['candidate']}: {report_data.get('candidate', {}).get('name') or labels['none']}"
    )
    _body(f"{labels['job_title']}: {job.get('title') or labels['none']}")

    _heading(labels["overall"])
    pct = f"{overall * 100:.0f}%" if overall is not None else labels["not_evaluable"]
    overall_value = overall if overall is not None else 0.0
    score_color = (
        (30, 100, 60) if overall_value >= 0.6
        else (170, 120, 20) if overall_value >= 0.4
        else (150, 40, 40)
    )
    pdf.set_font(font, "B", 26)
    pdf.set_text_color(*score_color)
    pdf.multi_cell(0, 14, _clean(pct), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(font, "", 9)
    pdf.set_text_color(90, 90, 90)
    _body(_clean(score.get("interpretation", "")))

    _heading(labels["components"])
    pdf.set_font(font, "B", 10)
    pdf.set_fill_color(232, 238, 248)
    pdf.set_text_color(23, 42, 76)
    pdf.cell(85, 8, _clean(labels["component_col"]), border=1, fill=True)
    pdf.cell(45, 8, _clean(labels["score_col"]), border=1, fill=True)
    pdf.cell(45, 8, _clean(labels["weight_col"]), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, "", 10)
    pdf.set_text_color(40, 40, 40)
    for comp in score.get("components", []):
        value = (
            f"{comp['value'] * 100:.0f}%"
            if comp.get("value") is not None
            else labels["not_evaluable"]
        )
        pdf.cell(85, 7, _clean(comp["label"])[:60], border=1)
        pdf.cell(45, 7, _clean(value), border=1)
        pdf.cell(45, 7, f"{comp.get('weight', 0) * 100:.0f}%", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    def _skill_section(title: str, items: list[str]) -> None:
        _heading(title, size=12)
        if items:
            pdf.set_font(font, "", 10)
            for item in items:
                pdf.multi_cell(
                    0, 6, _clean(f"• {item}"), align=align,
                    new_x="LMARGIN", new_y="NEXT",
                )
        else:
            _body(labels["none"])
        pdf.ln(1)

    _skill_section(labels["matched_req"], skills.get("matched_required", []))
    _skill_section(labels["missing_req"], skills.get("missing_required", []))
    _skill_section(labels["matched_pref"], skills.get("matched_preferred", []))
    _skill_section(labels["missing_pref"], skills.get("missing_preferred", []))
    _skill_section(
        labels["transferable"],
        [
            f"{t['missing_skill']} (related: {', '.join(t.get('related_skills_you_have', []))})"
            for t in skills.get("transferable", [])
        ],
    )

    _heading(labels["recommendations"])
    for rec in report_data.get("recommendations", [])[:10]:
        _body(f"{rec.get('title', '')} [{rec.get('priority', '')}]", style="B")
        if rec.get("detail"):
            _body(truncate(rec["detail"], 400))
        if rec.get("before") or rec.get("after"):
            _body(f"{labels['before']}: {rec.get('before') or '-'}")
            _body(f"{labels['after']}: {rec.get('after') or '-'}")
            _body(labels["draft_note"], size=8)
        pdf.ln(1)

    pdf.add_page()
    _heading(labels["limitations"])
    for item in labels["limitation_items"]:
        _body(f"• {item}")
    pdf.ln(2)
    pdf.set_font(font, "", 9)
    pdf.set_text_color(110, 110, 110)
    _body(labels["privacy_note"])

    return bytes(pdf.output())
