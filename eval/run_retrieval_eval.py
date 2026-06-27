"""
Retrieval Evaluation
====================
Measures whether the retriever finds the chunk containing the gold answer.

Metrics:
  - Recall@1, @3, @5, @10: fraction of queries where the gold chunk is in top-k
  - MRR (Mean Reciprocal Rank): how high the gold chunk ranks on average
  - Per-language breakdown (ne / np-roman / en)

A retrieved chunk "matches" if it contains the test entry's `gold_substring`
(normalized whitespace, case-insensitive for Latin).

Usage:
    python eval/run_retrieval_eval.py \
        --test eval/test_set.jsonl \
        --out eval/results/retrieval.json
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# Project root on path so `from rag import ...` works when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import normalize_query, retrieve, load_vector_store


def normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def find_rank(retrieved: list[dict], gold_sub: str) -> int | None:
    """Return 1-based rank of first chunk containing gold_sub, or None."""
    g = normalize(gold_sub)
    for i, ctx in enumerate(retrieved, 1):
        if g in normalize(ctx["text"]):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tests = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(tests)} test questions")

    collection = load_vector_store()

    ranks: list[int | None] = []
    per_lang_ranks: dict[str, list[int | None]] = defaultdict(list)
    per_topic_ranks: dict[str, list[int | None]] = defaultdict(list)
    details = []

    for t in tests:
        normalized = normalize_query(t["question"])
        retrieved = retrieve(collection, normalized,
                             top_k=args.top_k,
                             original_query=t["question"])
        rank = find_rank(retrieved, t["gold_substring"])
        ranks.append(rank)
        per_lang_ranks[t.get("lang", "?")].append(rank)
        per_topic_ranks[t.get("topic", "?")].append(rank)
        details.append({
            "id": t["id"], "lang": t["lang"], "topic": t.get("topic", "?"),
            "question": t["question"][:80],
            "rank": rank, "found": rank is not None,
            "n_retrieved": len(retrieved),
        })
        marker = f"#{rank}" if rank else "MISS"
        print(f"  [{t['lang']:9}] {t['id']}: {marker:>5}  {t['question'][:60]}")

    def summarize(rs: list[int | None]) -> dict:
        n = len(rs)
        if n == 0:
            return {}
        hits = [r for r in rs if r is not None]
        return {
            "n": n,
            "recall@1":  sum(1 for r in hits if r <= 1)  / n,
            "recall@3":  sum(1 for r in hits if r <= 3)  / n,
            "recall@5":  sum(1 for r in hits if r <= 5)  / n,
            "recall@10": sum(1 for r in hits if r <= 10) / n,
            "mrr":       sum(1 / r for r in hits) / n,
        }

    summary = {
        "overall": summarize(ranks),
        "per_language": {lang: summarize(rs) for lang, rs in per_lang_ranks.items()},
        "per_topic":    {top:  summarize(rs) for top,  rs in per_topic_ranks.items()},
    }

    out = {"summary": summary, "details": details}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== Overall ===")
    for k, v in summary["overall"].items():
        print(f"  {k:10}: {v:.3f}" if isinstance(v, float) else f"  {k:10}: {v}")
    print("\n=== Per language ===")
    for lang, s in summary["per_language"].items():
        print(f"  {lang:9}  n={s['n']:3}  R@5={s['recall@5']:.2f}  MRR={s['mrr']:.2f}")

    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
