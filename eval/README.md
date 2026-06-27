# Evaluation Suite — Nepal Parichaya RAG

A reproducible eval to answer every defensible question an ML engineer would ask about this system.

## Files

| File | Purpose |
|---|---|
| `test_set.jsonl` | ~50 hand-crafted Q+gold_answer+gold_substring across 3 languages |
| `trick_questions.jsonl` | ~15 out-of-scope questions that *should* be refused |
| `run_retrieval_eval.py` | Recall@k, MRR — does the retriever find the right chunk? |
| `run_e2e_eval.py` | Answer correctness, faithfulness, language adherence, refusal accuracy, cost, latency |
| `EVAL_REPORT.md` | Final write-up with every metric and baseline — the thing you point reviewers to |
| `results/` | Raw JSON output from the scripts (gitignored is fine) |

## How to run

```bash
# 1. Fill in test_set.jsonl with ~50 questions (the starter has ~10 examples)
#    and ~15 trick questions in trick_questions.jsonl

# 2. Retrieval evals (fast, cheap — embeddings only)
python eval/run_retrieval_eval.py \
    --test eval/test_set.jsonl \
    --out eval/results/retrieval.json

# 3. End-to-end evals (slower — runs the full pipeline + LLM judge)
python eval/run_e2e_eval.py \
    --test eval/test_set.jsonl \
    --trick eval/trick_questions.jsonl \
    --out eval/results/e2e.json

# 4. Copy the numbers from the two result JSONs into EVAL_REPORT.md
```

## Cost expectations

- Retrieval eval: ~$0.05 total (just embeddings on the queries)
- E2E eval on ~65 questions: ~$1.50–$3.00 (RAG run + 2 LLM-judge calls per Q)

## Test-set schema

```json
{"id": "q001",
 "lang": "ne" | "np-roman" | "en",
 "question": "the question text",
 "gold_answer": "the expected answer",
 "gold_substring": "a substring that MUST appear in the correct chunk (used for recall check)",
 "topic": "geography | history | politics | society | culture | administration"}
```

`gold_substring` is the most important field — it's how the retrieval eval knows whether the right context was retrieved. Pick a distinctive phrase or number that only appears in the correct chunk (e.g. `७७ जिल्ला`).

## What this answers

Run both scripts and the resulting numbers answer:

- Recall@5, MRR — "Does the retriever find the right chunk?"
- Answer correctness — "Are the answers actually right?"
- Faithfulness — "What's your hallucination rate?"
- Refusal accuracy — "Does it correctly refuse out-of-scope questions?"
- Language adherence — "Does it answer in the language asked?"
- Cost per query (avg + p95) — "What does this cost to run?"
- Latency p50/p95 — "How fast is it?"
- Per-language breakdown — "Does it work equally across NP/Roman/EN?"

For ablations (chunk-size, embedding model, threshold sweep, hybrid contribution) — those are Tier-2/3, add them later by parameterizing the existing scripts.
