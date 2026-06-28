# Nepal Parichaya RAG

[![CI](https://github.com/shishiradk/Nepal-parichaya/actions/workflows/ci.yml/badge.svg)](https://github.com/shishiradk/Nepal-parichaya/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Django 6.0](https://img.shields.io/badge/django-6.0-green.svg)](https://www.djangoproject.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](LICENSE)

A multilingual Retrieval-Augmented Generation (RAG) system for the **Nepal Parichaya** civics book — Django REST API + Streamlit UI + reproducible eval suite. Ask questions in **Nepali, Romanized Nepali, or English**; answers are grounded strictly in the source text and returned with citable chunks.

**Headline eval results** (full report in [`eval/EVAL_REPORT.md`](eval/EVAL_REPORT.md)):

| Metric | Original | After eval-driven fixes (final) |
|---|---|---|
| Retrieval Recall@5 | 0.36 | **0.80** *(+122%)* |
| Answer correctness (1–5) | 3.00 | **4.10** |
| Nepali correctness | 2.75 | **4.21** *(was worst, now best)* |
| Language adherence | 94% | **100%** |
| Out-of-scope refusal | 100% | **100%** |
| Cost / query | — | **$0.006** (gpt-4o) |
| Latency (p50 / p95) | — | **4.5 s / 7.6 s** |

Two documented **negative results** (chain attempt, cross-encoder reranker) in [§6 of EVAL_REPORT](eval/EVAL_REPORT.md) — both showed the textbook next moves *hurt* this specific system, which is the kind of finding senior engineers respect.

## Features

- **Hybrid retrieval** — vector similarity (ChromaDB) + Nepali keyword search
- **Query normalization** — Romanized Nepali (`dashain`) and English (`nepals area`) are automatically translated to Devanagari before retrieval
- **REST API** — Django + DRF backend with API-key auth, CORS, and rate-limiting
- **Streamlit UI** — dark-themed chat interface with source attribution and cost tracking
- **CLI mode** — interactive or single-query from the terminal
- **Eval suite** — 50-question test set, retrieval/E2E/baseline runners, full report — see [`eval/EVAL_REPORT.md`](eval/EVAL_REPORT.md)
- **Topic-aware chunking** — Nepal Parichaya is split by headings, not fixed character counts

## Project Structure

```
├── rag/                    # Core RAG module — pipeline shared by every frontend
│   ├── config.py           # Constants, paths, dicts, system prompt
│   ├── normalizer.py       # normalize_query() — Romanized/English → Devanagari
│   ├── retriever.py        # retrieve() — hybrid vector + keyword search
│   ├── generator.py        # generate_answer() — GPT answer generation
│   └── store.py            # ChromaDB load/build, chunk loader
│
├── api/                    # Django REST app (views, serializers, permissions)
├── nepali_rag_api/         # Django project (settings, urls, wsgi)
├── manage.py               # Django entrypoint
├── rag_api_client.py       # HTTP client used by streamlit_app.py
├── streamlit_app.py        # Streamlit UI (calls /api/query)
├── nepali_rag_openai.py    # CLI entry point
│
├── eval/                   # Evaluation suite — test set, runners, EVAL_REPORT
├── scripts/
│   └── rebuild_chunks.py   # Topic-aware chunker (markdown → clean chunks)
│
├── data/                   # Source documents
│   └── Nepal_Parichaya.pdf
├── docs/                   # Extended documentation
│   └── GUIDE.md
│
├── markdown_output/        # OCR markdown from Document AI (43 pages)
├── chunked_output/         # Topic-aware chunks (generated, gitignored)
└── chroma_db/              # Vector store (generated, gitignored)
```

## Quick Start

Two equivalent task runners ship with the repo:

| Runner | When to use |
|---|---|
| `python dev.py <task>` | **Works everywhere** — pure stdlib, no extra install |
| `make <task>` | If you have GNU Make installed (Git Bash on Windows usually doesn't) |

Run `python dev.py help` (or `make help`) to list every task.

### 1. Install dependencies

```bash
python dev.py install        # or:  pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
# Windows
set OPENAI_API_KEY=sk-...

# Or add to .env file
OPENAI_API_KEY=sk-...
```

### 3. Build chunks and vector store

```bash
# Build topic-aware chunks from markdown
python scripts/rebuild_chunks.py

# Build ChromaDB vector store
python nepali_rag_openai.py --build
```

### 4. Run

```bash
# Django REST API + Swagger UI         → http://127.0.0.1:8765/api/docs/
python dev.py api

# Streamlit UI                          → http://localhost:8501
python dev.py ui

# Interactive CLI
python dev.py cli

# Single CLI query
python nepali_rag_openai.py -q "दसैं कति दिन मनाइन्छ?"
```

### 5. Run the eval suite

```bash
python dev.py verify-eval        # check every test substring exists in chunks
python dev.py eval-retrieval     # R@k, MRR  (cheap, embeddings only)
python dev.py eval-e2e           # correctness, faithfulness, cost  (~$0.30)
python dev.py eval-baselines     # no-RAG / BM25 / vector-only  (~$1.00)
python dev.py eval-all           # all of the above
```

## REST API

The same RAG pipeline is exposed as a Django + DRF REST service for production use (frontend apps, mobile clients, integrations).

### Architecture

```
┌─── streamlit_app.py ───┐          ┌─── any other client ───┐
│  (rag_api_client.py)   │          │  curl, app, frontend   │
└────────────┬───────────┘          └───────────┬────────────┘
             │                                  │
             └──────────  HTTP  ────────────────┘
                            │
                ┌─── Django REST API ───┐
                │  api/views.py         │
                │  (thin wrapper)       │
                └───────────┬───────────┘
                            ▼
                ┌─── rag/ (core) ───┐
                │  retrieve         │
                │  generate         │
                │  ChromaDB store   │
                └───────────────────┘
```

The UI talks to the API over HTTP — no direct `rag/` import — so you can deploy each independently.

### Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `GET` | `/api/health` | Liveness check | none |
| `GET` | `/api/stats` | Vector store size + configured models | none |
| `POST` | `/api/query` | RAG question → answer + sources + cost | API key |
| `POST` | `/api/translate` | Translate text to Nepali Devanagari | API key |
| `GET` | `/api/docs/` | Interactive **Swagger UI** (try requests in the browser) | none |
| `GET` | `/api/redoc/` | ReDoc HTML docs | none |
| `GET` | `/api/schema/` | Raw OpenAPI 3.0 schema (YAML) | none |

### Quick start

```bash
# One-time: create Django's small SQLite stub (silences unapplied-migration warning)
python manage.py migrate

# Start server (assumes vector store already built — see Quick Start above)
# On Windows, port 8000 is often reserved by Hyper-V; use 8765 or another port if so.
python manage.py runserver 127.0.0.1:8765

# Health check
curl http://localhost:8765/api/health

# Query
curl -X POST http://localhost:8765/api/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -d '{"question": "नेपालमा कति जिल्ला छन्?"}'

# Or just open the interactive docs in your browser:
# http://localhost:8765/api/docs/
```

Response shape:
```json
{
  "answer": "Nepal has 77 districts.",
  "sources": [
    {"source": "Nepal_Parichaya-21", "page": "0",
     "heading": "...", "similarity": 0.99, "match": "dict"},
    ...
  ],
  "usage": {"prompt_tokens": 4922, "completion_tokens": 23,
            "total_tokens": 4945, "cost": 0.0125}
}
```

### Configuration (env vars)

**Server (Django):**

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI key the server uses |
| `RAG_API_KEY` | unset → open mode | If set, every request must send `X-API-Key: <value>` |
| `API_RATE_LIMIT` | `30/min` | Per-client rate limit (DRF AnonRateThrottle) |
| `DJANGO_SECRET_KEY` | dev-only stub | Set a real secret in production |
| `DJANGO_DEBUG` | `0` | `1` enables Django debug mode |
| `DJANGO_ALLOWED_HOSTS` | `*` | Comma-separated hosts the server accepts |

**UI (Streamlit) → talks to server:**

| Variable | Default | Purpose |
|---|---|---|
| `RAG_API_URL` | `http://127.0.0.1:8765` | Where the Streamlit UI calls for `/api/query`, etc. |
| `RAG_API_KEY` | unset | Sent as `X-API-Key` if the server requires one |

### Production deploy

```bash
gunicorn nepali_rag_api.wsgi:application \
  --bind 0.0.0.0:8000 --workers 2 --timeout 60
```

The vector store loads lazily on the first request (cold-start ~2-3s), then stays in memory for the worker lifetime.

## Example Queries

| Query | Answer |
|---|---|
| `दसैं कति दिन मनाइन्छ?` | १० दिनपर्यन्त |
| `नेपालको क्षेत्रफल कति हो?` | १,४७,१८१ वर्ग कि.मि. |
| `nepals national anthem` | सयौं थुँगा फूलका... |
| `nepal ko rastriya gana` | राष्ट्रिय गान |
| `area of nepal` | नेपालको क्षेत्रफल |

## Stack

- **Embeddings**: `text-embedding-3-small` (OpenAI)
- **LLM**: `gpt-4o-mini` (OpenAI)
- **Vector store**: ChromaDB (local, persistent)
- **UI**: Streamlit
- **OCR source**: Google Document AI (archived in `gcp/`)
