# Nepal Parichaya RAG — developer task runner
#
# Usage: make <target>      e.g. `make api` to start the REST server
#                                `make help` for the full list
#
# Assumes the venv is activated. If not, run:  source venv/Scripts/activate

PYTHON   ?= python
PORT     ?= 8765
HOST     ?= 127.0.0.1
EVAL_DIR  = eval
TEST_SET  = $(EVAL_DIR)/test_set.jsonl
TRICK_SET = $(EVAL_DIR)/trick_questions.jsonl

# Color helpers (Git Bash + most terminals)
B = \033[1m
G = \033[32m
Y = \033[33m
R = \033[0m

.DEFAULT_GOAL := help


# ───────────────────────────────────────────────────────────────
#  Setup
# ───────────────────────────────────────────────────────────────

install:  ## Install Python dependencies into the active venv
	$(PYTHON) -m pip install -r requirements.txt

migrate:  ## One-time Django migration (silences unapplied-migrations warning)
	$(PYTHON) manage.py migrate

build:  ## Build the ChromaDB vector store from chunked_output/
	$(PYTHON) nepali_rag_openai.py --build

chunks:  ## (Re)build the topic-aware chunks from markdown_output/
	$(PYTHON) scripts/rebuild_chunks.py


# ───────────────────────────────────────────────────────────────
#  Run the three frontends
# ───────────────────────────────────────────────────────────────

api: migrate  ## Start the Django REST API server (http://HOST:PORT/api/docs/)
	@echo "$(G)→ API:   http://$(HOST):$(PORT)/api/health$(R)"
	@echo "$(G)→ Docs:  http://$(HOST):$(PORT)/api/docs/$(R)"
	$(PYTHON) manage.py runserver $(HOST):$(PORT)

ui:  ## Start the Streamlit UI (http://localhost:8501)
	streamlit run streamlit_app.py

cli:  ## Interactive CLI chat against the RAG
	$(PYTHON) nepali_rag_openai.py


# ───────────────────────────────────────────────────────────────
#  Evaluation suite
# ───────────────────────────────────────────────────────────────

verify-eval:  ## Verify every test_set.jsonl gold_substring exists in chunks
	$(PYTHON) $(EVAL_DIR)/verify_substrings.py --test $(TEST_SET)

eval-retrieval:  ## Run retrieval eval (R@k, MRR) — cheap, embeddings only
	$(PYTHON) $(EVAL_DIR)/run_retrieval_eval.py \
		--test $(TEST_SET) \
		--out $(EVAL_DIR)/results/retrieval.json

eval-e2e:  ## Run full end-to-end eval (~$0.30) — correctness, faithfulness, cost
	$(PYTHON) $(EVAL_DIR)/run_e2e_eval.py \
		--test $(TEST_SET) \
		--trick $(TRICK_SET) \
		--out $(EVAL_DIR)/results/e2e.json

eval-baselines:  ## Run no-RAG / BM25 / vector-only baselines (~$1.00)
	$(PYTHON) $(EVAL_DIR)/run_baselines.py \
		--test $(TEST_SET) \
		--out $(EVAL_DIR)/results/baselines.json

eval-reranker:  ## Run cross-encoder re-ranker eval (downloads ~280MB model on first run)
	$(PYTHON) $(EVAL_DIR)/run_reranker_eval.py \
		--test $(TEST_SET) \
		--out $(EVAL_DIR)/results/reranker.json

eval-all: verify-eval eval-retrieval eval-e2e eval-baselines  ## Run the full eval suite (skips reranker)


# ───────────────────────────────────────────────────────────────
#  Quality of life
# ───────────────────────────────────────────────────────────────

freeze:  ## Re-freeze requirements.txt (UTF-8) from the active venv
	$(PYTHON) -m pip freeze > requirements.txt

clean:  ## Remove __pycache__ and empty SQLite stub
	find . -type d -name "__pycache__" -not -path "./venv/*" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned __pycache__"

schema:  ## Dump the OpenAPI schema to api-schema.yml
	PYTHONIOENCODING=utf-8 $(PYTHON) manage.py spectacular --file api-schema.yml
	@echo "wrote api-schema.yml"

help:  ## Show this help
	@echo ""
	@echo "$(B)Nepal Parichaya RAG — developer tasks$(R)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} \
	     /^[a-zA-Z_-]+:.*?## / { printf "  $(Y)%-16s$(R) %s\n", $$1, $$2 } \
	     /^# ─{3,}/             { printf "\n" }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  $(G)make api$(R)             start REST API on http://$(HOST):$(PORT)"
	@echo "  $(G)make ui$(R)              start Streamlit UI"
	@echo "  $(G)make eval-retrieval$(R)  run the retrieval eval"
	@echo "  $(G)make PORT=5000 api$(R)   override the default port"
	@echo ""

.PHONY: install migrate build chunks api ui cli \
        verify-eval eval-retrieval eval-e2e eval-baselines eval-reranker eval-all \
        freeze clean schema help
