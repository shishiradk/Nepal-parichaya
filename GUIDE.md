# Nepal Parichaya RAG — Complete Project Guide

This guide covers every file and folder in the project: what it does, how to use it, and how to work on it. The project has three separate pipelines — read each section that's relevant to you.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Quick Start](#3-quick-start)
4. [Part A — OpenAI RAG Pipeline (Active)](#part-a--openai-rag-pipeline-active)
   - [Environment Setup](#environment-setup)
   - [Source Data: `markdown_output/`](#source-data-markdown_output)
   - [Step 1: Build Chunks — `scripts/rebuild_chunks.py`](#step-1-build-chunks--scriptsrebuild_chunkspy)
   - [Step 2: Build Vector Store — `nepali_rag_openai.py --build`](#step-2-build-vector-store)
   - [Step 3: Run the App](#step-3-run-the-app)
   - [RAG Module: `rag/`](#rag-module-rag)
     - [`rag/config.py`](#ragconfigpy)
     - [`rag/normalizer.py`](#ragnormalizerpy)
     - [`rag/retriever.py`](#ragretrieverpy)
     - [`rag/generator.py`](#raggeneratorpy)
     - [`rag/store.py`](#ragstorepy)
     - [`rag/__init__.py`](#rag__init__py)
   - [Entry Points](#entry-points)
     - [`nepali_rag_openai.py`](#nepali_rag_openaipythe-cli)
     - [`streamlit_app.py`](#streamlit_apppy--the-ui)
5. [Part B — GCP / Document AI OCR Pipeline (Archived)](#part-b--gcp--document-ai-ocr-pipeline-archived)
   - [What This Pipeline Did](#what-this-pipeline-did)
   - [GCP Prerequisites](#gcp-prerequisites)
   - [File-by-File Reference](#file-by-file-reference)
6. [Part C — Fine-tuning (Future Work)](#part-c--fine-tuning-future-work)
7. [Generated & Gitignored Directories](#generated--gitignored-directories)
8. [Configuration Files](#configuration-files)
9. [Troubleshooting](#troubleshooting)
10. [Extending the Project](#extending-the-project)

---

## 1. Project Overview

This project answers questions about Nepal using the **Nepal Parichaya** book as the sole knowledge source. It uses Retrieval-Augmented Generation (RAG): given a question, it finds the most relevant passages from the book and feeds them to GPT to generate a grounded answer.

```
                    ┌───────────────────────────────────────────────┐
                    │              Nepal Parichaya PDF               │
                    └────────────────────┬──────────────────────────┘
                                         │ Google Document AI (GCP — archived)
                                         ▼
                    ┌───────────────────────────────────────────────┐
                    │        markdown_output/*.md  (43 files)        │
                    │        Raw OCR text, one file per page         │
                    └────────────────────┬──────────────────────────┘
                                         │ scripts/rebuild_chunks.py
                                         ▼
                    ┌───────────────────────────────────────────────┐
                    │   chunked_output/clean_chunks/*.md  (~735)     │
                    │   Topic-aware chunks with frontmatter          │
                    └────────────────────┬──────────────────────────┘
                                         │ nepali_rag_openai.py --build
                                         ▼
                    ┌───────────────────────────────────────────────┐
                    │            chroma_db/  (vector store)          │
                    │   735 embeddings via text-embedding-3-small    │
                    └────────────────────┬──────────────────────────┘
                                         │ at query time
                                         ▼
                    ┌───────────────────────────────────────────────┐
                    │   User Question  ──►  normalize_query()        │
                    │   (Nepali / Romanized / English)               │
                    │         │                                       │
                    │         ▼                                       │
                    │   retrieve()  (hybrid: vector + keyword)       │
                    │         │                                       │
                    │         ▼                                       │
                    │   generate_answer()  →  GPT-4o-mini            │
                    └───────────────────────────────────────────────┘
```

**Two pipelines live in this repo:**

| Pipeline | Folder | Status | Purpose |
|---|---|---|---|
| OpenAI RAG | `rag/`, root | Active | Answering Nepal Parichaya questions |
| GCP Document AI | `gcp/` | Archived | OCR pipeline that produced `markdown_output/` |

---

## 2. Project Structure

```
Nepali-document-processor/
│
│  ── ENTRY POINTS ──────────────────────────────────────────────────────
├── nepali_rag_openai.py       CLI: query the RAG system, build vector store
├── streamlit_app.py           Web UI: Streamlit chat interface
│
│  ── RAG MODULE ─────────────────────────────────────────────────────────
├── rag/
│   ├── __init__.py            Public API re-exports (import from here)
│   ├── config.py              All constants: paths, models, pricing, system prompt
│   ├── normalizer.py          normalize_query(): Romanized/English → Devanagari
│   ├── retriever.py           retrieve(): hybrid vector + keyword search
│   ├── generator.py           generate_answer(): GPT answer generation
│   └── store.py               load/build ChromaDB vector store, load chunks
│
│  ── PREPROCESSING ──────────────────────────────────────────────────────
├── scripts/
│   └── rebuild_chunks.py      Converts markdown_output/ into topic chunks
│
│  ── SOURCE DATA ─────────────────────────────────────────────────────────
├── markdown_output/           43 OCR .md files (one per Nepal Parichaya page)
│   ├── Nepal_Parichaya-0.md
│   ├── Nepal_Parichaya-1.md
│   └── ...  (up to -42.md)
│
│  ── ARCHIVED: GCP PIPELINE ─────────────────────────────────────────────
├── gcp/
│   ├── config.py              GCP config dataclasses (project, processor, storage)
│   ├── batch_ocr.py           Batch Document AI OCR processor class
│   ├── document-ai.py         One-shot Document AI processor
│   ├── RAG.py                 Old Vertex AI RAG pipeline (LangChain + Vertex)
│   ├── query_rag.py           CLI to query the old Vertex AI RAG
│   ├── rag_deploy.py          Deploy RAG corpus to Vertex AI
│   ├── rag_preview.py         Preview Vertex AI RAG results
│   ├── list_available_models.py  List available Vertex AI models
│   ├── test.py / test_config.py / validate.py  GCP credential + config tests
│   ├── docai_sample.json      Sample Document AI response for testing
│   └── config/
│       ├── config.devlopment.json   Dev environment GCP config
│       └── config.production.yaml  Prod environment GCP config
│
│  ── GENERATED (gitignored) ─────────────────────────────────────────────
├── chunked_output/            Created by rebuild_chunks.py
│   └── clean_chunks/*.md      ~735 topic chunks with YAML frontmatter
├── chroma_db/                 Created by nepali_rag_openai.py --build
│
│  ── CONFIG & DOCS ───────────────────────────────────────────────────────
├── requirements.txt           All Python dependencies (pinned versions)
├── .env                       API keys — NOT committed (see .gitignore)
├── .gitignore                 Excludes .claude/, chroma_db/, chunked_output/, etc.
├── .vscode/settings.json      VS Code UTF-8 and Python settings
├── README.md                  Project overview and quick start
└── GUIDE.md                   This file
```

---

## 3. Quick Start

For someone who just cloned the repo and wants to run the chatbot:

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your OpenAI key
echo OPENAI_API_KEY=sk-... > .env

# 4. Build topic-aware chunks from the 43 markdown files
python scripts/rebuild_chunks.py

# 5. Embed chunks and build the ChromaDB vector store (~2 min, ~$0.01)
python nepali_rag_openai.py --build

# 6. Launch the Streamlit UI
streamlit run streamlit_app.py
# → open http://localhost:8501
```

---

## Part A — OpenAI RAG Pipeline (Active)

This is the working RAG system. Everything lives in `rag/`, with two thin entry points at the root.

### Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

The code loads this automatically via `python-dotenv`. Never commit `.env` — it's in `.gitignore`.

---

### Source Data: `markdown_output/`

**What it is:** 43 `.md` files — one per page (approximately) of the Nepal Parichaya book — produced by Google Document AI OCR. These are the raw text inputs for the entire RAG pipeline.

**Files:** `Nepal_Parichaya-0.md` through `Nepal_Parichaya-42.md`

**Format:** Plain Markdown. Each file contains OCR-extracted Devanagari text with some noise (page numbers like `२१८/नेपाल परिचय`, website URLs, standalone numerals). The chunker cleans these out.

**Adding more content:** If you get OCR output for additional pages, drop the `.md` files here and re-run the pipeline from Step 1. The chunker deduplicates by filename stem, so it's safe to add files without removing old ones.

**Do not edit these files manually** — they are the authoritative source. Cleaning happens in the chunker pipeline.

---

### Step 1: Build Chunks — `scripts/rebuild_chunks.py`

**Purpose:** Reads all 43 OCR markdown files, cleans noise, and splits by topic headings into individual chunk `.md` files saved to `chunked_output/clean_chunks/`.

**Why topic-aware chunking matters:**
- Fixed-size chunking: `"दसैं is 10 days... [page break] ...तिहार is 5 days"` → one mixed chunk → bad retrieval
- Topic-aware: `"दसैं"` chunk + `"तिहार"` chunk → clean retrieval

**Usage:**

```bash
# Build chunks (creates chunked_output/clean_chunks/)
python scripts/rebuild_chunks.py

# Preview first 5 chunks without saving
python scripts/rebuild_chunks.py --preview 5

# Show statistics only (no save)
python scripts/rebuild_chunks.py --stats

# Use a different markdown directory
python scripts/rebuild_chunks.py --md-dir path/to/md --out path/to/out
```

**Internal pipeline (6 steps):**

1. **`find_markdown_files()`** — Recursively finds all `.md` files. Deduplicates by stem (keeps most-recently-modified if same name appears in multiple subdirs — handles old Document AI batch output structure).
2. **`clean_text()`** — Strips OCR noise line-by-line: page headers (`२१८/नेपाल परिचय`), standalone page numbers, phone numbers, website URLs, short non-Devanagari lines.
3. **`_is_heading()`** — Detects section headings: 2–55 chars, ≥45% Devanagari, no sentence-final punctuation (`।`), followed by ≥60 chars of text. Has a **poetry guard** — if the next 6 lines look like poem lines (short, no `।`), the current line is NOT treated as a heading, preserving national anthem and song text.
4. **`split_by_topics()`** — Walks lines, starts a new chunk at each heading.
5. **`_split_long_chunk()`** — Chunks longer than 1,200 chars are split at paragraph boundaries (double newlines) to keep them within embedding limits.
6. **Quality filter** — Discards chunks with fewer than 120 chars or less than 55% Devanagari content.

**Output format** (`chunked_output/clean_chunks/<chunk_id>.md`):

```
---
chunk_id: Nepal_Parichaya-5_a3f1b2c4
source_file: Nepal_Parichaya-5
heading: दसैं
chunk_size: 487
nepali_ratio: 0.8932
timestamp: 2025-11-01T12:34:56
---

दसैं

दसैं नेपालको सबैभन्दा ठूलो र ...
```

**Tuning constants** (top of file):

| Constant | Default | Effect |
|---|---|---|
| `MIN_CHUNK_CHARS` | 120 | Discard very short chunks |
| `MAX_CHUNK_CHARS` | 1200 | Split chunks longer than this |
| `MIN_NEPALI_RATIO` | 0.55 | Discard chunks with mostly non-Nepali content |

---

### Step 2: Build Vector Store

```bash
python nepali_rag_openai.py --build
```

This calls `rag/store.py:build_vector_store()` which:
1. Reads all `chunked_output/clean_chunks/*.md` files
2. Embeds them in batches of 100 using OpenAI `text-embedding-3-small`
3. Saves to `chroma_db/` using ChromaDB's persistent storage

**Cost:** ~$0.01 for ~735 chunks. Runs in ~2–3 minutes.

**Re-building:** Safe to re-run — the old collection is deleted and rebuilt from scratch each time.

---

### Step 3: Run the App

**Streamlit UI (recommended):**

```bash
streamlit run streamlit_app.py
# → http://localhost:8501

# On Windows with Devanagari terminal issues:
.\venv\Scripts\streamlit.exe run streamlit_app.py
```

**CLI:**

```bash
# Interactive mode (type questions, Ctrl+C to exit)
python nepali_rag_openai.py

# Single query
python nepali_rag_openai.py -q "नेपालको राजधानी कहाँ हो?"

# Single query with custom Top-K
python nepali_rag_openai.py -q "दसैं" --top-k 10

# Windows UTF-8 fix if Devanagari is garbled
python -X utf8 nepali_rag_openai.py
```

---

### RAG Module: `rag/`

All shared logic is in this Python package. Both `nepali_rag_openai.py` and `streamlit_app.py` import from it. Never import directly from individual files (`from rag.retriever import ...`) — always use the package (`from rag import retrieve`).

---

#### `rag/config.py`

All constants in one place. **Change settings here, not in the entry points.**

**Key constants:**

```python
CHROMA_DIR      = Path("chroma_db")          # vector store location
COLLECTION_NAME = "nepal_parichaya"
CHUNKS_DIR                                    # auto-selects clean_chunks if present
EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI embedding model
LLM_MODEL       = "gpt-4o-mini"             # answer generation model
TOP_K           = 6                          # default chunks to retrieve
MIN_SIMILARITY  = 0.30                       # discard results below 30% similarity
EMBEDDING_BATCH_SIZE = 100
```

**`MODEL_PRICING`:** Cost table (USD per 1M tokens) for `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`. Used by the Streamlit UI to display cost.

**`_ROMAN_NEPALI_DICT`:** Fast lookup dictionary for Romanized Nepali → Devanagari. Used as the first (free, no API call) path in `normalize_query()`. To add a new word mapping, add it here.

**`DEFAULT_SYSTEM_PROMPT`:** The 8-rule instruction given to GPT with every query. Rules include: answer only from context, respond in the question's language, quote facts verbatim, preserve line breaks for poems/anthem. **To change GPT behavior, edit rules here.**

---

#### `rag/normalizer.py`

**Function:** `normalize_query(query: str) -> str`

Converts any query to Devanagari before retrieval (the ChromaDB index is 100% Devanagari).

**Three code paths:**

1. **Already Devanagari** (`नेपाल...`) → returned unchanged immediately.
2. **All words in `_ROMAN_NEPALI_DICT`** (`dashain ko din`) → fast dictionary substitution, no API call.
3. **Everything else** → GPT translates:
   - Romanized Nepali → Devanagari: `"nepal ko rastriya gana"` → `"नेपालको राष्ट्रिय गान"`
   - English Nepal queries → Devanagari: `"national anthem of nepal"` → `"नेपालको राष्ट्रिय गान"`
   - Non-Nepal English → returned unchanged: `"what is democracy"` stays as-is

**To extend:** Add words to `_ROMAN_NEPALI_DICT` in `rag/config.py` to avoid GPT calls for common Romanized terms. Add example translations to the GPT system prompt in `normalizer.py` for better English→Nepali accuracy.

---

#### `rag/retriever.py`

**Function:** `retrieve(collection, query, top_k=None) -> list[dict]`

Hybrid retrieval: vector similarity search + Nepali keyword search combined.

**What it returns:** A list of dicts, each with:

```python
{
    "text":       "दसैं नेपालको सबैभन्दा ठूलो...",
    "source":     "Nepal_Parichaya-5",
    "page":       "5",
    "heading":    "दसैं",
    "similarity": 0.7234,
    "match":      "vector"   # or "keyword"
}
```

**How it works:**

1. **Vector search:** Embeds the query, queries ChromaDB for `top_k` nearest chunks by cosine similarity. Filters out anything below `MIN_SIMILARITY` (0.30).
2. **Keyword search:** Extracts specific Nepali words (≥4 chars, not in `_COMMON` stop words like `नेपालको`, `हुन्छ`). Uses the 2 longest specific words. For each, does a ChromaDB `$contains` keyword filter on a wider result set (`max(top_k*2, 15)` results).
3. **Merge:** Combines vector and keyword results, deduplicates, sorts by similarity. Always appends up to 8 keyword-only hits that vector search missed.

**`_COMMON` set:** Nepali words that appear in almost every chunk (like "नेपालको") are skipped as keywords because they match everything and add noise. To tune: add more common words to this set.

**To tune retrieval quality:** Adjust `MIN_SIMILARITY` in `rag/config.py`, or change the `[:2]` limit on specific keywords (more keywords = wider coverage but more API calls).

---

#### `rag/generator.py`

**Function:** `generate_answer(query, contexts, model=None, temperature=0.2, max_tokens=1024, system_prompt=None) -> (str, dict)`

Calls GPT with the retrieved context chunks and returns the answer.

**Parameters:**
- `query` — the normalized Devanagari query
- `contexts` — list of dicts from `retrieve()`
- `model` — defaults to `LLM_MODEL` from config
- `temperature` — 0.2 by default (factual answers, low randomness)
- `max_tokens` — output length limit
- `system_prompt` — defaults to `DEFAULT_SYSTEM_PROMPT`; Streamlit passes its editable version

**Returns:** `(answer: str, usage: dict)` where `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost` (USD float). The CLI ignores usage with `answer, _ = generate_answer(...)`. Streamlit uses it to display token count and cost.

**Context format passed to GPT:**
```
--- Context 1 (Source: Nepal_Parichaya-5, Page: 5, Similarity: 0.7234) ---
दसैं

दसैं नेपालको सबैभन्दा ठूलो पर्व हो...
```

---

#### `rag/store.py`

Three functions for managing the vector store.

**`load_chunks() -> list[dict]`**
Reads all `.md` files from `CHUNKS_DIR`, parses YAML frontmatter, returns a list of chunk dicts with `id`, `text`, `source`, `page`, `heading`.

**`build_vector_store(chunks) -> collection`**
- Deletes any existing ChromaDB collection
- Creates a new one with cosine similarity metric
- Embeds chunks in batches of 100 (progress bar via `tqdm`)
- Saves to `chroma_db/`

**`load_vector_store() -> collection`**
Opens the existing `chroma_db/` and returns the ChromaDB collection object. Exits with an error message if not found (user needs to run `--build` first).

---

#### `rag/__init__.py`

Re-exports everything public. This is the single import surface for entry points:

```python
from rag import normalize_query, retrieve, generate_answer
from rag import load_chunks, build_vector_store, load_vector_store
from rag import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL
from rag import TOP_K, MIN_SIMILARITY, DEFAULT_SYSTEM_PROMPT, MODEL_PRICING
```

**Do not change this file** unless you're adding a new public function to the package.

---

### Entry Points

#### `nepali_rag_openai.py` — the CLI

A thin wrapper over `rag/`. Contains no RAG logic itself.

**Three modes:**

```bash
# Build / rebuild vector store from chunks
python nepali_rag_openai.py --build

# Single query and print answer
python nepali_rag_openai.py -q "नेपालको क्षेत्रफल कति हो?"
python nepali_rag_openai.py -q "nepals area" --top-k 10

# Interactive REPL (type questions until 'quit')
python nepali_rag_openai.py
```

**Functions:**
- `query_rag(collection, query)` — normalizes query, retrieves, generates, prints answer + source preview
- `interactive_mode(collection)` — REPL loop
- `main()` — argparse entry point, loads `.env`, branches to build/query/interactive

**Windows note:** The file reconfigures `stdout/stderr/stdin` to UTF-8 on Windows so Devanagari prints correctly in the terminal.

---

#### `streamlit_app.py` — the UI

Streamlit web interface. Also a thin wrapper over `rag/`, but adds Streamlit-specific state management and caching.

**Layout:**
- Left sidebar: New Chat, Reload Vector Store, document count, status indicators (API key / ChromaDB), Export Chat
- Center: Chat history + disclaimer tip box + chat input
- Right panel: Model settings (model, temperature, Top-K, max tokens), system prompt editor, token/cost metrics, retrieved source chunks, API code preview

**Key Streamlit patterns:**

```python
@st.cache_resource(ttl=3600)
def get_collection():
    ...
```
The ChromaDB collection is cached for 1 hour. If you rebuild `chroma_db/` while Streamlit is running, click **Reload Vector Store** in the sidebar (calls `get_collection.clear()`) to flush the cache and reload.

**Session state keys:**
- `messages` — chat history (list of `{role, content, sources}`)
- `last_sources` — sources from the most recent query (shown in right panel)
- `total_tokens` / `total_cost` — running totals for the session

**Query flow:**
1. User types in chat input
2. `normalize_query(prompt)` converts to Devanagari
3. `retrieve(collection, normalized, top_k=top_k)` fetches relevant chunks
4. `generate_answer(normalized, contexts, ...)` calls GPT
5. Answer + source badges displayed; message appended to history

**To modify the UI:** All styling is inline CSS in the `st.markdown("""<style>...""")` block at the top. The dark color palette uses `#1a1a2e` (background), `#16213e` (panels), `#4fc3f7` (accent).

---

## Part B — GCP / Document AI OCR Pipeline (Archived)

### What This Pipeline Did

The GCP pipeline ran **before** the current RAG system existed. It:
1. Uploaded Nepal_Parichaya.pdf to Google Cloud Storage
2. Ran Google Document AI OCR on it (batch processing)
3. Downloaded the JSON results and extracted Devanagari text
4. An earlier, now-replaced RAG system used Vertex AI's RAG API

The output of this pipeline (`markdown_output/*.md`) is already in the repo. You do **not** need to run the GCP pipeline unless you want to re-OCR new documents or understand how the markdown files were made.

### GCP Prerequisites

To run any GCP script, you need:
- A Google Cloud project with Document AI API enabled
- A service account JSON key (stored locally — never commit it)
- A GCS bucket for input/output
- Set environment variables or use the config files in `gcp/config/`

```bash
# Required environment variables
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
export PROJECT_ID=your-gcp-project-id
export PROCESSOR_ID=your-docai-processor-id
export PROCESSOR_LOCATION=us
export INPUT_GCS_URI=gs://your-bucket/Nepal_Parichaya.pdf
export OUTPUT_GCS_URI=gs://your-bucket/output/
```

All GCP scripts must be run from inside the `gcp/` directory (they import `from config import ...`):

```bash
cd gcp
python batch_ocr.py
```

---

### File-by-File Reference

#### `gcp/config.py`

Dataclass-based configuration system for GCP. Handles multiple environments (development, staging, production) loaded from environment variables or config files.

Key dataclasses:
- `ProcessorConfig` — Document AI processor ID, location, language hints
- `StorageConfig` — GCS input/output URIs
- `OcrConfig` — OCR options (native PDF parsing, image quality scores)
- `ProcessingConfig` — poll interval, wait timeout
- `AppConfig` — combines all the above

`get_config(config_path=None)` — loads config from a JSON/YAML file or falls back to environment variables.

`ConfigManager` — loads environment-specific config from `gcp/config/config.{env}.json/yaml`.

---

#### `gcp/config/config.devlopment.json` and `config.production.yaml`

Environment-specific GCP configuration files. Contain processor IDs, GCS bucket URIs, OCR settings. These are templates — you fill in your own project values.

**Note:** `config.devlopment.json` has a typo in the filename (missing 'e') — left as-is since it's just archived reference material.

---

#### `gcp/batch_ocr.py`

Main batch OCR processor. The `BatchOCRProcessor` class wraps the Document AI batch processing API.

**Key methods:**
- `start_batch_process()` — submits the batch job asynchronously; returns operation name
- `wait_for_completion_with_progress()` — polls with a tqdm progress bar until the job finishes
- `extract_text_with_progress()` — downloads JSON results from GCS, extracts `.text` fields, saves `.txt` files to `extracted_text/`
- `monitor_operation(operation_name)` — monitor a previously-started operation by name

**Usage:**

```bash
cd gcp

# Start batch job and wait for it
python batch_ocr.py --wait

# Start batch job asynchronously (returns operation name)
python batch_ocr.py

# Monitor an existing operation
python batch_ocr.py --monitor "projects/123/locations/us/operations/456"

# Use a specific environment config
python batch_ocr.py --env production
python batch_ocr.py --config config/config.production.yaml
```

---

#### `gcp/document-ai.py`

Simpler, one-shot (synchronous) Document AI processor — processes a single page at a time. Good for testing OCR on a single file rather than running a full batch job.

---

#### `gcp/RAG.py`

The **old RAG pipeline** built with LangChain + Vertex AI RAG API. This was replaced by the current OpenAI+ChromaDB pipeline. Kept for reference.

Used `VertexAI` LLM and the Vertex AI RAG API's `RagCorpus` for vector storage. The current `rag/` package is the replacement.

---

#### `gcp/rag_deploy.py`

Deploys the RAG corpus to Vertex AI. Creates a `RagCorpus`, uploads the text chunks, and saves the deployment info to `rag_deployment_info.json`. Part of the old Vertex AI RAG workflow.

---

#### `gcp/rag_preview.py`

Preview tool for the Vertex AI RAG deployment — test queries against a deployed corpus before integrating.

---

#### `gcp/query_rag.py`

CLI for querying the old Vertex AI RAG system. Replaced by `nepali_rag_openai.py`.

---

#### `gcp/list_available_models.py`

Lists available Vertex AI model versions (Gemini, etc.) in a given GCP project. Useful for checking which model versions are available in your region.

---

#### `gcp/test.py` / `gcp/test_config.py` / `gcp/validate.py`

Diagnostic and validation scripts:
- `test_config.py` — validates that all required environment variables are set and config loads correctly
- `validate.py` — tests that GCP credentials work (can reach Document AI and GCS APIs)
- `test.py` — end-to-end test of the OCR pipeline with a sample document

---

#### `gcp/docai_sample.json`

A saved Document AI API response for a sample document. Used by `test.py` to test text extraction logic without making live API calls. Useful for developing text post-processing code offline.

---

## Generated & Gitignored Directories

These directories are created by the pipeline and excluded from git. You need to generate them locally.

| Directory | Created by | Contents |
|---|---|---|
| `chunked_output/clean_chunks/` | `scripts/rebuild_chunks.py` | ~735 topic chunk `.md` files |
| `chunked_output/clean_chunks_metadata.json` | `scripts/rebuild_chunks.py` | JSON index of all chunks |
| `chroma_db/` | `nepali_rag_openai.py --build` | ChromaDB vector store (embeddings) |

**If you delete `chroma_db/`:** Re-run `python nepali_rag_openai.py --build`.

**If you delete `chunked_output/`:** Re-run `python scripts/rebuild_chunks.py`, then `--build`.

**If you add new `.md` files to `markdown_output/`:** Re-run both steps, then click **Reload Vector Store** in the Streamlit sidebar.

---

## Configuration Files

#### `.env`

Not committed. Create it manually:

```
OPENAI_API_KEY=sk-your-key-here
```

#### `requirements.txt`

Full `pip freeze` output — all packages pinned to exact versions. The active RAG pipeline needs only a subset:

```
openai>=1.0
chromadb>=0.4
streamlit>=1.30
python-dotenv>=1.0
tqdm
```

The rest (torch, transformers, sentence-transformers, google-cloud-*, langchain, etc.) are from the GCP pipeline and fine-tuning work. Install all with `pip install -r requirements.txt`, or create a leaner `requirements-rag.txt` with just the 5 packages above if you want a faster setup.

#### `.gitignore`

Key exclusions:
```
.claude/          ← Claude Code workspace (never commit)
chroma_db/        ← generated vector store
chunked_output/   ← generated chunks
Nepal_Parichaya.pdf ← large binary
.env              ← API keys
gcp/nepali_ocr_data/
gcp/nepal_parichaya_parts/
gcp/corpus_file/
gcp/google_service_acc_creds.json
google_service_acc_creds.json
```

#### `.vscode/settings.json`

Sets `files.encoding: utf8` and `python.defaultInterpreterPath` to the venv. This ensures VS Code displays Devanagari text correctly.

---

## Troubleshooting

**"Vector store not found" / `sys.exit(1)`**
→ Run `python nepali_rag_openai.py --build`

**"OPENAI_API_KEY not set"**
→ Add `OPENAI_API_KEY=sk-...` to your `.env` file in the project root

**Streamlit shows old document count after adding chunks**
→ Click **Reload Vector Store** in the sidebar (flushes the `@st.cache_resource` cache)

**English query like "nepals area" returns "यो जानकारी दिइएको सन्दर्भमा भेटिएन।"**
→ `normalize_query()` should translate it — verify your OpenAI key is set. Try a more specific query: `"area of nepal in square km"`

**Nepali text garbled in Windows terminal**
→ Run `python -X utf8 nepali_rag_openai.py` or use the Streamlit UI instead

**Anthem / poem displayed as bullet points**
→ The system prompt rule 8 prevents this. If it happens, clear the Streamlit session state: click **New Chat** in the sidebar and hard-refresh the browser (Ctrl+Shift+R)

**`ImportError: No module named 'rag'`**
→ Run scripts from the project root, not from inside a subdirectory. The `rag/` package must be on the Python path.

**GCP scripts: `from config import AppConfig` fails**
→ GCP scripts must be run from inside the `gcp/` directory: `cd gcp && python batch_ocr.py`

**Port 8501 already in use**
→ Another Streamlit instance is running. Kill it:
```powershell
# Windows PowerShell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess -Force
```

---

## Extending the Project

**Add a new document to the knowledge base:**
1. Get OCR output as a `.md` file (use `gcp/batch_ocr.py` or any OCR tool)
2. Drop it in `markdown_output/`
3. `python scripts/rebuild_chunks.py`
4. `python nepali_rag_openai.py --build`
5. Click **Reload Vector Store** in Streamlit

**Support a new language in queries:**
Edit `rag/normalizer.py` — add example translations to the GPT system prompt in the `normalize_query()` function.

**Add a new Romanized Nepali word:**
Add it to `_ROMAN_NEPALI_DICT` in `rag/config.py` — this avoids a GPT API call for that word.

**Change the answer model (e.g. use GPT-4o):**
In `rag/config.py`, change `LLM_MODEL = "gpt-4o"`. Or select it in the Streamlit UI model dropdown.

**Improve retrieval for a specific topic:**
Check what chunks exist: `python scripts/rebuild_chunks.py --preview 10`. If the topic isn't covered, check that `markdown_output/` has the relevant page and that the chunk wasn't filtered out by `MIN_NEPALI_RATIO` or `MIN_CHUNK_CHARS`.

**Switch from OpenAI embeddings to local embeddings:**
Replace the OpenAI client call in `rag/retriever.py` and `rag/store.py` with `sentence-transformers` (already installed). Use `paraphrase-multilingual-MiniLM-L12-v2` for Nepali support.
