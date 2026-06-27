"""
Baseline Comparison Eval
========================
Runs three baselines on the same test set to populate §5 of EVAL_REPORT.md:

  1. no-RAG     — LLM answers from its own knowledge, no retrieval at all.
                  (Does retrieval even add value?)
  2. BM25       — pure keyword retrieval (no embeddings), then generate.
                  (Can we even beat a 30-year-old algorithm?)
  3. vector-only — embeddings only, no dict/keyword boost.
                  (Is the hybrid logic earning its complexity?)

For each baseline we compute the same headline metrics as the main e2e eval:
  - Retrieval recall@5  (where applicable)
  - Answer correctness  (LLM-as-judge vs gold_answer, 1-5)
  - Cost per query

Usage:
    python eval/run_baselines.py --test eval/test_set.jsonl \
        --out eval/results/baselines.json
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from statistics import mean as _mean

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from rank_bm25 import BM25Okapi

from rag.config import (
    EMBEDDING_MODEL, LLM_MODEL, MODEL_PRICING,
    CHROMA_DIR, COLLECTION_NAME, DEFAULT_SYSTEM_PROMPT, MIN_SIMILARITY,
)
from rag.store import load_vector_store

JUDGE_MODEL = "gpt-4o"


def with_backoff(fn, *args, tries=8, base=3.0, **kwargs):
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, APIConnectionError, APIError) as e:
            if i == tries - 1:
                raise
            wait = base * (2 ** i)
            print(f"    transient ({type(e).__name__}), sleeping {wait:.0f}s ...")
            time.sleep(wait)


CORRECTNESS_PROMPT = """You are evaluating a system answering questions about Nepal.

Question: {q}

Gold answer: {gold}

System answer: {ans}

Score the system answer on factual correctness vs the gold answer:
  5 = fully correct, all key facts match
  4 = mostly correct, minor omission
  3 = partially correct
  2 = mostly wrong but on the right topic
  1 = wrong / refused / off-topic

Return JSON only: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


def judge_correctness(client, q, gold, ans):
    try:
        r = with_backoff(client.chat.completions.create,
            model=JUDGE_MODEL, temperature=0, max_tokens=200,
            messages=[{"role": "user",
                       "content": CORRECTNESS_PROMPT.format(q=q, gold=gold, ans=ans)}])
        m = re.search(r"\{.*\}", r.choices[0].message.content, re.DOTALL)
        return json.loads(m.group())["score"] if m else 0
    except Exception:
        return 0


def cost_of(usage, model=LLM_MODEL):
    rates = MODEL_PRICING.get(model, {"input": 0.15, "output": 0.60})
    return (usage.prompt_tokens / 1e6) * rates["input"] + \
           (usage.completion_tokens / 1e6) * rates["output"]


def normalize_sub(s: str) -> str:
    return " ".join(s.split()).lower()


def rank_of_gold(retrieved_texts: list[str], gold_substring: str) -> int | None:
    g = normalize_sub(gold_substring)
    for i, t in enumerate(retrieved_texts, 1):
        if g in normalize_sub(t):
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: NO-RAG (LLM only, no context)
# ─────────────────────────────────────────────────────────────────────────────
def baseline_no_rag(client, q: str) -> tuple[str, float]:
    """Ask the LLM the question with no retrieved context."""
    resp = with_backoff(client.chat.completions.create,
        model=LLM_MODEL, temperature=0.2, max_tokens=512,
        messages=[
            {"role": "system",
             "content": "Answer questions about Nepal from your knowledge. "
                        "Answer in the same language as the question. "
                        "Say you don't know if you don't."},
            {"role": "user", "content": q},
        ])
    ans = resp.choices[0].message.content
    return ans, cost_of(resp.usage)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: BM25 retrieval + LLM generation
# ─────────────────────────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, chunks_dir: Path):
        files = sorted(chunks_dir.glob("*.md"))
        self.texts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
        # whitespace tokenization — works for Devanagari + Latin uniformly
        self.tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"BM25 indexed {len(self.texts)} chunks")

    def search(self, query: str, k: int = 5) -> list[str]:
        toks = query.lower().split()
        scores = self.bm25.get_scores(toks)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.texts[i] for i in top_idx]


def baseline_bm25(client, retriever: BM25Retriever, q: str) -> tuple[str, list[str], float]:
    top = retriever.search(q, k=5)
    ctx = "\n---\n".join(t[:1200] for t in top)
    resp = with_backoff(client.chat.completions.create,
        model=LLM_MODEL, temperature=0.2, max_tokens=512,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from Nepal Parichaya:\n{ctx}\n\nQuestion: {q}"},
        ])
    return resp.choices[0].message.content, top, cost_of(resp.usage)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: VECTOR-ONLY (no keyword branches, no dict boost)
# ─────────────────────────────────────────────────────────────────────────────
def baseline_vector_only(client, collection, q: str) -> tuple[str, list[str], float]:
    """Pure vector retrieval — no dict or keyword paths."""
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=[q]).data[0].embedding
    res = collection.query(query_embeddings=[emb], n_results=8,
                           include=["documents", "metadatas", "distances"])
    docs, dists = res["documents"][0], res["distances"][0]
    keep = [(d, dist) for d, dist in zip(docs, dists) if (1 - dist) >= MIN_SIMILARITY]
    keep = keep[:6]
    ctx = "\n---\n".join(d[:1200] for d, _ in keep)
    resp = with_backoff(client.chat.completions.create,
        model=LLM_MODEL, temperature=0.2, max_tokens=512,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from Nepal Parichaya:\n{ctx}\n\nQuestion: {q}"},
        ])
    return resp.choices[0].message.content, [d for d, _ in keep], cost_of(resp.usage)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunks-dir", type=Path, default=Path("chunked_output/clean_chunks"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tests = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(tests)} test questions")

    client = OpenAI()
    collection = load_vector_store()
    bm25 = BM25Retriever(args.chunks_dir)

    rows = {"no_rag": [], "bm25": [], "vector_only": []}

    for t in tests:
        q, gold, gold_sub = t["question"], t["gold_answer"], t["gold_substring"]

        # ── no-RAG ──
        ans, c = baseline_no_rag(client, q)
        score = judge_correctness(client, q, gold, ans)
        rows["no_rag"].append({"id": t["id"], "lang": t["lang"],
                               "score": score, "cost": c, "rank": None})
        print(f"  [no-rag    ] {t['id']:5} corr={score} ${c:.4f}")

        # ── BM25 ──
        ans, ctxs, c = baseline_bm25(client, bm25, q)
        score = judge_correctness(client, q, gold, ans)
        rank = rank_of_gold(ctxs, gold_sub)
        rows["bm25"].append({"id": t["id"], "lang": t["lang"],
                             "score": score, "cost": c, "rank": rank})
        print(f"  [bm25      ] {t['id']:5} corr={score} rank={rank} ${c:.4f}")

        # ── vector-only ──
        ans, ctxs, c = baseline_vector_only(client, collection, q)
        score = judge_correctness(client, q, gold, ans)
        rank = rank_of_gold(ctxs, gold_sub)
        rows["vector_only"].append({"id": t["id"], "lang": t["lang"],
                                    "score": score, "cost": c, "rank": rank})
        print(f"  [vector    ] {t['id']:5} corr={score} rank={rank} ${c:.4f}")

    def summarize(rs):
        scores = [r["score"] for r in rs if r["score"] > 0]
        ranks = [r["rank"] for r in rs if r["rank"] is not None]
        n = len(rs)
        return {
            "n": n,
            "correctness_mean": _mean(scores) if scores else 0,
            "pct_5_5": sum(1 for s in scores if s == 5) / max(n, 1),
            "recall@5": sum(1 for r in ranks if r <= 5) / max(n, 1),
            "cost_avg": sum(r["cost"] for r in rs) / max(n, 1),
        }

    summary = {name: summarize(rs) for name, rs in rows.items()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== Baseline comparison ===")
    print(f"{'baseline':14} {'corr':>5} {'5/5':>6} {'R@5':>5} {'$/q':>7}")
    for name, s in summary.items():
        print(f"{name:14} {s['correctness_mean']:5.2f} "
              f"{s['pct_5_5']*100:5.0f}% {s['recall@5']*100:4.0f}% "
              f"${s['cost_avg']:.4f}")
    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
