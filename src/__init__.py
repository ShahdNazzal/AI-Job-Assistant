"""AI Job Assistant - core application logic.

Modules:
    resume_parser       Extract text and structure from PDF/DOCX/TXT resumes.
    job_parser          Parse job-description text into structured data.
    skill_extractor     Dictionary-based (+ optional semantic) skill detection.
    embedding_service   Sentence-Transformer embeddings and cosine similarity.
    matching_engine     Weighted resume-to-job compatibility scoring.
    recommendation_engine  Rule-based, evidence-aware recommendations.
    llm_service         Optional OpenAI-compatible LLM adapter.
    rag_service         Lightweight local RAG over curated knowledge files.
    evaluation          Metrics computed from small labeled datasets.
    report_generator    JSON / CSV / Markdown / PDF report exports.
    privacy             Session clearing and sensitive-data masking helpers.
    i18n                English / Arabic UI strings.
    models              Pydantic data models shared across modules.
    ui_helpers          Shared Streamlit widgets (sidebar, headers, badges).
"""

__version__ = "1.0.0"
