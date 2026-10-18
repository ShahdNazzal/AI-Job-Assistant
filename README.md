# AI Job Assistant

An evidence-based resume analysis and job-matching application for job
seekers, built as a complete AI engineering portfolio project.

Upload a resume (PDF / DOCX / TXT), paste a job description, and get a
transparent, weighted compatibility score, prioritized skill-gap guidance,
honest resume-improvement suggestions, an optional LLM-powered career
assistant grounded in your own documents (RAG), and downloadable reports -
in English and Arabic.

> **Honesty statement.** The compatibility score is a transparent heuristic
> indicator computed from documented components. It is **not** a probability
> of being hired and **not** a scientifically validated recruitment
> prediction. Every metric shown in the app (including evaluation metrics)
> is computed live from defined inputs; nothing is hard-coded.

---
<img width="1896" height="905" alt="image" src="https://github.com/user-attachments/assets/e37c3a66-5a6c-47ca-b600-4c34242fb3f8" />

<img width="1912" height="902" alt="image" src="https://github.com/user-attachments/assets/8468fbd9-5853-496e-a1a4-5d8e30aaba06" />

<img width="1305" height="781" alt="image" src="https://github.com/user-attachments/assets/94bfa748-270f-4f66-8e18-3aad749f7b46" />

<img width="1050" height="896" alt="image" src="https://github.com/user-attachments/assets/248b9572-782b-483a-ae1a-6f629a1f12ee" />

<img width="1076" height="822" alt="image" src="https://github.com/user-attachments/assets/908dede9-7993-418f-9aff-9a1177d43d41" />

<img width="1000" height="291" alt="image" src="https://github.com/user-attachments/assets/8301eb4e-34b1-46d1-80c2-6d290d4676c3" />

## Table of contents

1. [Project overview](#1-project-overview)
2. [Main features](#2-main-features)
3. [Architecture explanation](#3-architecture-explanation)
4. [Technology stack](#4-technology-stack)
5. [Project folder structure](#5-project-folder-structure)
6. [Installation instructions](#6-installation-instructions)
7. [Virtual environment setup](#7-virtual-environment-setup)
8. [How to run the application](#8-how-to-run-the-application)
9. [How to configure the optional LLM](#9-how-to-configure-the-optional-llm)
10. [How the semantic matching algorithm works](#10-how-the-semantic-matching-algorithm-works)
11. [The exact scoring formula and weights](#11-the-exact-scoring-formula-and-weights)
12. [How skill extraction works](#12-how-skill-extraction-works)
13. [How the optional RAG pipeline works](#13-how-the-optional-rag-pipeline-works)
14. [How to run the tests](#14-how-to-run-the-tests)
15. [How evaluation metrics are calculated](#15-how-evaluation-metrics-are-calculated)
16. [Known limitations](#16-known-limitations)
17. [Privacy considerations](#17-privacy-considerations)
18. [Future improvements](#18-future-improvements)
19. [Suggested portfolio and CV description](#19-suggested-portfolio-and-cv-description)

---

## 1. Project overview

Job seekers rarely know *why* their resume does or does not fit a job
posting. Generic "AI resume tools" hide the logic, invent scores, or
require an API key for every feature. AI Job Assistant takes the opposite
approach:

- **Transparency first.** The score is a weighted sum of four documented
  components, shown individually next to the total. Weights are visible and
  adjustable in the UI.
- **Core features are free and local.** Parsing, skill extraction, semantic
  matching, gap analysis, and all report exports run entirely on your
  machine without any LLM.
- **Evidence over vibes.** Every detected skill carries the snippet and
  section where it was found. Missing skills become *learning*
  recommendations - never suggestions to pad your resume.
- **Graceful degradation.** The LLM assistant is optional; without
  credentials the app switches to rule-based answers plus knowledge-base
  retrieval.

The project deliberately demonstrates skills expected of an entry-level AI
engineer: NLP pipelines, embeddings, evaluation discipline, LLM adapter
design, RAG, privacy handling, testing, and packaging.

## 2. Main features

| # | Feature | Where | Works without LLM? |
|---|---------|-------|--------------------|
| 1 | Resume upload & parsing (PDF, DOCX, TXT) with structure, contact, name and summary detection | `pages/1_Resume_Analysis.py` | Yes |
| 2 | Dictionary skill extraction with aliases, word-boundary protection, evidence snippets, and optional semantic inference | `src/skill_extractor.py` | Yes |
| 3 | Job-description analysis: title, required vs preferred skills, responsibilities, experience, education, certifications, keywords | `pages/2_Job_Matching.py` | Yes |
| 4 | Semantic matching engine (Sentence Transformers) with a transparent 4-component weighted score | `src/matching_engine.py` | Yes |
| 5 | Skill-gap analysis with prioritization, learning directions, transferable-skill detection | `pages/3_Skill_Gap_Analysis.py` | Yes |
| 6 | AI career assistant (LLM + RAG, grounded and cited) with rule-based fallback | `pages/4_AI_Career_Assistant.py` | Partial (retrieval + rules) |
| 7 | Lightweight RAG over a curated local knowledge base (8 documents, in-memory vector index) | `src/rag_service.py` | Yes |
| 8 | Resume-improvement suggestions with draft before/after examples (clearly labeled, no fabrication) | `src/recommendation_engine.py` | Yes |
| 9 | Interactive dashboard: gauge, component chart, skill-gap chart, all Plotly | `pages/2_Job_Matching.py` | Yes |
| 10 | Model evaluation page: live precision/recall/F1 and matching-order checks | `pages/5_Model_Evaluation.py` | Mostly (matching eval needs the model) |
| 11 | Report downloads: JSON, CSV, Markdown, PDF (Arabic-capable) | `src/report_generator.py` | Yes |
| 12 | Bilingual UI (English / Arabic with RTL support) | `src/i18n.py` | Yes |

## 3. Architecture explanation

The application separates **UI** (Streamlit pages), **AI logic**
(`src/` modules), **configuration** (`config/settings.py`), and **tests**
(`tests/`). Data flows one way:

```
Uploads / pasted text
        |
        v
resume_parser / job_parser -----------> structured Pydantic models
        |                                        |
        v                                        v
skill_extractor (taxonomy + evidence)   matching_engine (embeddings + weights)
        |                                        |
        v                                        v
recommendation_engine <---------------- MatchResult
        |
        v
Streamlit pages (dashboard, gap analysis, assistant)
        |                    |
        v                    v
report_generator      llm_service + rag_service (optional)
```

Full details and a Mermaid data-flow diagram: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Key design rules:

- Pages never import each other; they communicate through `st.session_state`
  (`resume_data`, `confirmed_skills`, `job_data`, `match_result`,
  `recommendations`).
- Business logic lives in `src/` and is fully importable without Streamlit -
  which is exactly how the pytest suite tests it.
- The embedding model is loaded lazily once per process (Streamlit
  `cache_resource`); no model download happens at startup.
- LLM credentials are read only in `config/settings.py` and never printed.

## 4. Technology stack

| Layer | Technology | Why |
|-------|------------|-----|
| UI | Streamlit | Fast, professional multipage data apps |
| PDF parsing | PyMuPDF (fitz) | Fast, reliable text extraction + scanned-PDF detection |
| DOCX parsing | python-docx | Paragraphs **and** table content |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Small (90 MB), strong semantic performance, CPU-friendly |
| Similarity / math | scikit-learn + NumPy | Cosine similarity, array math |
| Data handling | pandas | Tables for charts and exports |
| Charts | Plotly | Interactive gauges and bar charts |
| Validation | Pydantic v2 | Structured models for parsed data and LLM JSON outputs |
| Config | python-dotenv | `.env` support without leaking secrets |
| LLM adapter | requests | OpenAI-compatible `/chat/completions`, no vendor SDK lock-in |
| PDF reports | fpdf2 + uharfbuzz | Unicode fonts + Arabic text shaping |
| Testing | pytest | Unit + integration tests, offline by design |

## 5. Project folder structure

```
ai-job-assistant/
├── app.py                        # Home page (Streamlit entry point)
├── pages/
│   ├── 1_Resume_Analysis.py      # Upload, parse, review, confirm skills
│   ├── 2_Job_Matching.py         # JD input + dashboard + charts
│   ├── 3_Skill_Gap_Analysis.py   # Prioritized gaps + CSV export
│   ├── 4_AI_Career_Assistant.py  # LLM/RAG Q&A + suggestions + downloads
│   └── 5_Model_Evaluation.py     # Live metrics from labeled datasets
├── src/
│   ├── models.py                 # Pydantic data models
│   ├── exceptions.py             # Typed exception hierarchy
│   ├── utils.py                  # Text normalization, snippets, logging
│   ├── resume_parser.py          # PDF/DOCX/TXT extraction + structure
│   ├── job_parser.py             # JD sections, title, requirements
│   ├── skill_extractor.py        # Taxonomy matching + semantic inference
│   ├── embedding_service.py      # Lazy model wrapper (+ test dummy)
│   ├── matching_engine.py        # Weighted 4-component score
│   ├── recommendation_engine.py  # Rule-based guidance + fallback answers
│   ├── llm_service.py            # OpenAI-compatible adapter
│   ├── rag_service.py            # Chunking, in-memory index, prompts
│   ├── evaluation.py             # P/R/F1 + matching tier checks
│   ├── report_generator.py       # JSON / CSV / Markdown / PDF exports
│   ├── privacy.py                # Session clearing, masking, disclosure
│   ├── i18n.py                   # EN/AR translations + RTL helpers
│   └── ui_helpers.py             # Sidebar, headers, download widgets
├── config/
│   └── settings.py               # All tunables, env parsing, weights
├── data/
│   ├── sample_resume.txt         # Demo data (labeled as demo in the UI)
│   ├── sample_job_description.txt
│   ├── skills_taxonomy.json      # ~130 skills, 18 categories, aliases
│   ├── knowledge_base/*.md       # 8 curated RAG documents
│   ├── evaluation/*.json         # Labeled skill + matching datasets
│   └── fonts/                    # Bundled DejaVu + Noto Naskh (OFL/Bitstream)
├── tests/                        # 9 test files, 107 tests
├── docs/ARCHITECTURE.md
├── requirements.txt
├── .env.example                  # LLM placeholders (no real credentials)
├── .gitignore
├── LICENSE                       # MIT
├── pytest.ini
└── README.md
```

## 6. Installation instructions

Requirements: **Python 3.11+** (tested on 3.12), pip, internet access for
the first model download (~90 MB, cached afterwards).

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # optional - only for the LLM assistant
streamlit run app.py
```

> **CPU-only machines:** pip installs the CUDA build of torch by default
> (~2 GB extra). To save space, install the CPU build first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### First-run notes

- The embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB)
  downloads once from Hugging Face on the first match/assistant use, then
  is cached locally (offline afterwards). Application **startup** never
  downloads anything.
- The app starts and fully works **without an LLM API key**.

## 7. Virtual environment setup

Any standard workflow works. With `venv` (see above) the interpreter lives
in `.venv/`. With `uv`:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

With conda:

```bash
conda create -n aijob python=3.12 -y
conda activate aijob
pip install -r requirements.txt
```

Verify the installation before starting the UI:

```bash
pytest                       # should end with "107 passed"
python -c "import streamlit, fitz, docx, sentence_transformers; print('deps ok')"
```

## 8. How to run the application

From the **project root** (the folder containing `app.py`):

```bash
streamlit run app.py
```

Then:

1. Open the browser at the printed local URL (default
   <http://localhost:8501>).
2. Go to **Resume Analysis** and upload a PDF/DOCX/TXT resume (or load the
   clearly-labeled demo data from the home page).
3. Review the detected skills, add/remove until the list reflects reality,
   and press **Save confirmed skills**.
4. Open **Job Matching**, paste a job description, press **Analyze match**.
5. Explore the dashboard, **Skill Gap Analysis**, and the **AI Career
   Assistant**; download reports from the assistant page.
6. **Model Evaluation** shows live metrics computed from the bundled
   labeled datasets.
7. Use **Language** in the sidebar to switch between English and Arabic
   (RTL). "Clear all session data" wipes parsed documents from memory.

## 9. How to configure the optional LLM

Copy `.env.example` to `.env` and fill in:

```env
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://api.openai.com/v1     # or Groq, Together, OpenRouter,
                                           # Ollama (http://localhost:11434/v1),
                                           # vLLM, LM Studio, Azure gateways...
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT=30
```

- Any provider exposing an OpenAI-style `POST {BASE_URL}/chat/completions`
  endpoint works without code changes.
- Keys are read only by `config/settings.py`; they never appear in logs,
  errors, UI, or exported reports.
- When configured, the assistant sends: your question, a structured resume
  summary, the parsed job requirements, the match summary, and up to 3
  knowledge-base excerpts - exactly what the in-app disclosure box lists.
- When missing or failing (timeout, 401, invalid JSON...), the assistant
  falls back to deterministic rule-based answers; nothing else in the app
  is affected.

## 10. How the semantic matching algorithm works

1. **Text preparation.** Resume and job text are normalized (ligatures,
   whitespace, bullets) while preserving paragraph structure.
2. **Embedding.** Each document is split into ~180-word chunks (MiniLM
   truncates at 256 word-piece tokens), each chunk is embedded, and chunk
   vectors are mean-pooled and L2-normalized, giving one document vector
   per side.
3. **Semantic similarity** = cosine similarity (dot product of normalized
   vectors) between resume and job vectors, clamped to [0, 1].
4. **Skill coverage** = fraction of the job's required (and separately
   preferred) skills present in the user's *confirmed* skill list -
   never the raw resume text.
5. **Evidence score** blends (a) the fraction of matched required skills
   whose extractor evidence sits in the experience/internship/projects
   sections with (b) the cosine similarity between those sections and the
   job description (60/40). If only one signal exists it is used with a
   0.8 discount; if neither exists the component is marked unavailable.
6. **Weighted total** = sum of available components times their effective
   weights. A component that cannot be evaluated (e.g. no preferred skills
   detected) is *excluded and its weight redistributed* - never silently
   defaulted.
7. **Transferable skills.** For each missing required skill, the taxonomy's
   `related` links are checked against confirmed skills ("you know X,
   which is related to Y").

Related concepts are found across wording differences via embeddings
("RAG pipeline" vs "Retrieval-Augmented Generation"), while exact skill
matching remains strict and evidence-backed.

## 11. The exact scoring formula and weights

```
overall = w_req  * required_coverage
        + w_pref * preferred_coverage
        + w_sem  * semantic_similarity
        + w_evid * evidence_score
```

Default weights (configurable via env vars or the in-page sliders; always
re-normalized to sum to 1.0):

| Component | Weight | Env var |
|-----------|--------|---------|
| Required-skill coverage | **0.45** | `WEIGHT_REQUIRED_SKILLS` |
| Preferred-skill coverage | **0.15** | `WEIGHT_PREFERRED_SKILLS` |
| Semantic similarity | **0.25** | `WEIGHT_SEMANTIC_SIMILARITY` |
| Relevant evidence | **0.15** | `WEIGHT_EVIDENCE` |

Rules implemented in `src/matching_engine.py`:

- Components that cannot be evaluated are marked `available=False` and the
  remaining weights are renormalized proportionally. The UI shows a note
  for each excluded component.
- If **no** component is evaluable, `overall` is `None` and the UI says so
  instead of showing a number.
- The result is rounded to 4 decimals and clamped to [0, 1].

Interpretation bands (UI only): >= 65% strong alignment, 40-65% partial,
< 40% limited - with explicit disclaimers.

## 12. How skill extraction works

1. **Taxonomy.** `data/skills_taxonomy.json` defines ~130 skills in 18
   categories with aliases (`sklearn` -> Scikit-learn, `RAG` ->
   Retrieval-Augmented Generation, `Postgres` -> PostgreSQL), learning
   hints, `related` skills, and a `case_sensitive` flag for genuinely
   ambiguous short names (Go, R, C).
2. **Matching.** Every name/alias is compiled to a regex with custom
   boundary lookarounds `(?<![A-Za-z0-9_+#])...(?![A-Za-z0-9_+#])`, so
   "Java" never matches inside "JavaScript", "ML" never matches inside
   "HTML", and "Research" never produces "R".
3. **Evidence.** For every match, a ~120-character snippet and the
   containing resume section are recorded (section attribution comes from
   character-span mapping of the section splitter).
4. **Deduplication.** Matches merge per canonical skill; identical
   snippets collapse; distinct occurrences are all kept as evidence.
5. **Optional semantic inference.** Resume paragraph chunks and all
   taxonomy skill names are embedded (cached); skills with max chunk
   similarity >= 0.55 that were *not* found by the dictionary are returned
   separately as **inferred** (low confidence) and shown in a distinct UI
   list requiring manual confirmation.
6. **User confirmation.** All downstream scoring uses the user-confirmed
   skill list, pre-filled from explicit detections only. Inferred skills
   are never auto-included.

## 13. How the optional RAG pipeline works

1. **Corpus.** Eight curated Markdown documents in
   `data/knowledge_base/` (ML fundamentals, NLP & embeddings, REST APIs,
   RAG, vector search, evaluation, cloud/MLOps, interview prep).
2. **Chunking.** Documents split on `##` headings, then paragraphs are
   merged into ~900-character chunks, keeping title + heading metadata.
3. **Indexing.** Chunks are embedded once with the shared model and kept
   in a NumPy matrix (in-memory; no external vector DB). The index is
   Streamlit-cached and rebuilt only when file signatures change.
4. **Retrieval.** The question is embedded; cosine ranking keeps the top-k
   (default 3) chunks above a minimum score (default 0.20).
5. **Generation.** Retrieved chunks are passed to the LLM inside a strict
   grounding prompt (cite `[KB: title]` / `[Resume]` / `[Job]`, refuse
   when evidence is missing, never invent facts). Answers are labeled
   "AI-generated".
6. **Transparency.** Retrieved sources (title, heading, similarity,
   excerpt) are always displayed - with or without an LLM. Without an
   LLM, the page still shows retrieval results and rule-based answers.

This is a genuine RAG loop (retrieve -> augment -> generate). If the LLM
is unavailable, retrieval alone still runs and is displayed as such.

## 14. How to run the tests

```bash
pytest                 # from the project root
pytest -v              # verbose
pytest tests/test_matching_engine.py -v   # one file
```

The suite (107 tests) covers:

- `test_resume_parser.py` - TXT/PDF/DOCX extraction (documents generated
  in-memory with PyMuPDF/python-docx), scanned-PDF rejection, empty and
  corrupted files, size limits, section/name/contact detection.
- `test_skill_extractor.py` - aliases, boundary protection
  (JavaScript/Java, Research/R, HTML/ML), case-sensitive Go/R, evidence
  sections, dedupe, semantic-inference labeling, taxonomy integrity.
- `test_job_parser.py` - title detection, required/preferred split,
  experience years, education, warnings, too-short rejection.
- `test_matching_engine.py` - component math, weight renormalization,
  empty inputs, transferable skills, evidence honesty, monotonicity
  (strong > weak) - all with a **deterministic dummy embedding service**,
  no model or network needed.
- `test_recommendation_engine.py` - honesty invariants (missing skills
  become learning items, never "add this to your resume"), vague-phrase
  detection, draft summaries using only real resume facts.
- `test_evaluation.py` - P/R/F1 math consistency, dataset error handling,
  matching case structure and score bounds.
- `test_llm_service.py` - mocked HTTP: success, timeout, 401 (key masked),
  malformed payloads, JSON extraction and Pydantic validation.
- `test_rag_service.py` - chunking, retrieval ranking, thresholds,
  prompt blocks and Arabic language rule.
- `test_report_generator.py` - JSON round-trip without secrets, CSV row
  accounting, Markdown (EN/AR), PDF generation and text extraction,
  Arabic PDF with shaping.

## 15. How evaluation metrics are calculated

All metrics are computed **at request time** from
`data/evaluation/*.json` - the page shows numbers only when they were
actually calculated in that session.

**Skill extraction (micro-averaged, skill-name level):**
each labeled sample's expected canonical skills are compared with the
extractor's explicit output. TP = predicted & expected, FP = predicted
only, FN = expected only; `precision = TP/(TP+FP)`,
`recall = TP/(TP+FN)`, `F1 = harmonic mean`. Current bundled dataset
(14 samples): **P = R = F1 = 0.973** (TP=36, FP=1, FN=1) - recomputed
live, not stored.

**Matching evaluation:** five labeled resume/job pairs with expected
tiers (strong / medium / weak), matched/missing skill expectations and
score bounds. The engine (with the real embedding model) runs each case;
per-case checks plus a tier-ordering check
(avg(strong) > avg(medium) > avg(weak)) are reported. On the bundled
dataset all five cases and the ordering check pass
(0.850 > 0.509 > 0.111).

**Limitations of the datasets** (also shown in the app): both are tiny,
authored for demonstration, and sensitive to taxonomy size and text
style. They are **not** benchmarks; production use requires larger,
independently labeled corpora and inter-annotator agreement checks.
Automated unit tests additionally run via `pytest` (see section 14).

## 16. Known limitations

- **Scanned PDFs:** OCR is not implemented. Image-only PDFs are detected
  (chars/page threshold) and clearly rejected with guidance - never
  silently "parsed".
- **Dictionary-bound skill detection:** paraphrased skills ("built neural
  nets") are found only if listed in the taxonomy or caught by the
  optional semantic inference (labeled, low-confidence, manual).
- **Required vs preferred assignment** depends on the job description's
  section headings; unstructured descriptions fall back to
  "everything required" with a visible note.
- **Heuristic document parsing:** name/title detection uses conservative
  patterns and marks low-confidence results; exotic resume layouts may
  defeat it.
- **Semantic similarity ceiling:** cosine similarity of a junior resume
  and a senior job can stay high because domains overlap; that is why it
  contributes only 25% and is displayed with an explicit caveat.
- **Evaluation datasets are toy-sized** (14 texts / 5 pairs) and exist to
  demonstrate methodology, not to certify accuracy.
- **PDF reports** render Arabic via DejaVu + HarfBuzz shaping; complex
  mixed-direction lines (e.g. ISO timestamps) may reorder cosmetically.
  Markdown/JSON/CSV exports are plain UTF-8 and always correct.
- **LLM outputs** can still be wrong; the app enforces citation and
  refusal prompts but cannot guarantee provider behavior. Rule-based
  fallbacks are deterministic and grounded.
- **English JDs parse best.** Non-English job descriptions will extract
  fewer skills because the taxonomy is English; the taxonomy is data, so
  adding categories is straightforward.
- **Privacy:** everything runs locally except explicit LLM calls; Streamlit
  session clearing cannot guarantee OS-level memory wiping (stated in the
  UI).

## 17. Privacy considerations

- Resumes are parsed **in memory only**; nothing is written to disk and
  no analytics exist.
- Logs use a dedicated logger that receives only event metadata; contact
  details and document text are masked (`src/utils.mask_sensitive`) and
  truncated.
- API keys live exclusively in `.env` (git-ignored; `.env.example`
  contains placeholders). Error messages are sanitized through a
  secret-stripping helper before display.
- The LLM feature shows an explicit "what is sent" disclosure before use
  and sends only structured excerpts (never the raw file). Retrieval and
  rule-based modes send nothing anywhere.
- "Clear all session data" (sidebar) deletes every stored session key;
  the UI honestly notes this cannot guarantee provider-side deletion or
  OS-memory wiping.
- Reports contain analysis results only - no keys, no config, no full
  raw documents.

## 18. Future improvements

- OCR support (e.g. Tesseract) for scanned resumes.
- Larger, independently labeled evaluation datasets with inter-annotator
  agreement; bootstrap confidence intervals on metrics.
- Multilingual taxonomy and language detection (Arabic resumes/JDs).
- Persistent (opt-in) analysis history with encryption at rest.
- Smarter JD parsing via an optional LLM structured-extraction mode with
  Pydantic validation and human review.
- Hybrid retrieval (BM25 + vectors) and reranking for the knowledge base.
- Feedback loop: let users correct matched/missing skills and log it for
  taxonomy improvement.
- Deployment recipe: Dockerfile + CI (lint, pytest, evaluation checks)
  and a hosted demo.
- Role-specific weighting presets (research vs applied tracks).

## 19. Suggested portfolio and CV description

**Short version (CV bullet):**

> Built *AI Job Assistant*, a Streamlit application that parses resumes
> (PDF/DOCX/TXT), extracts skills against a 130-entry taxonomy with
> evidence snippets, and scores resume-job fit with a transparent weighted
> formula combining skill coverage and Sentence-Transformer semantic
> similarity; includes a grounded RAG assistant over curated career docs,
> live P/R/F1 evaluation datasets, and JSON/CSV/Markdown/PDF report
> exports in English and Arabic.

**Interview talking points:**

- Why each scoring component exists and how unavailable components are
  renormalized instead of faked.
- Word-boundary regex design for skill matching (JavaScript/Java cases)
  and how evidence snippets build user trust.
- Chunked mean-pooling to work around MiniLM's 256-token limit.
- The decision to make LLM features optional and how fallbacks are tested
  with mocked HTTP.
- Honesty engineering: disclaimers, "not evaluable" states, labeled
  drafts, and live-not-stored metrics.

**Repository labels:** Python, Streamlit, NLP, Sentence Transformers,
Embeddings, RAG, LLM, Pydantic, pytest, Portfolio Project.


