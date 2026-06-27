"""
Cross-Encoder Re-ranker Evaluation
===================================
Re-ranks the top-K candidates from the existing retriever using a multilingual
cross-encoder, then measures the lift in recall@5 / MRR.

Cross-encoder scores each (query, passage) pair *together* — slower than
bi-encoders but more accurate. Used as a second-stage re-rank after vector
retrieval to clean up noisy top-K orderings without changing embeddings.

Model: BAAI/bge-reranker-base — multilingual (incl. Devanagari), ~280MB download
       on first run. CPU inference; ~0.5-1s per query for 20 candidates.

Usage:
    python eval/run_reranker_eval.py --test eval/test_set.jsonl \
        --out eval/results/reranker.json
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from statistics import mean as _mean

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder

from rag import normalize_query, retrieve, load_vector_store

RERANKER_MODEL = "BAAI/bge-reranker-base"


def normalize_sub(s: str) -> str:
    return " ".join(s.split()).lower()


def rank_of_gold(texts: list[str], gold_sub: str) -> int | None:
    g = normalize_sub(gold_sub)
    for i, t in enumerate(texts, 1):
        if g in normalize_sub(t):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k-retrieve", type=int, default=20,
                    help="how many candidates to fetch before re-ranking")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tests = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(tests)} test questions")

    print(f"Loading cross-encoder {RERANKER_MODEL} (first run may download ~280MB)...")
    ce = CrossEncoder(RERANKER_MODEL, max_length=512)

    collection = load_vector_store()

    before_ranks = []
    after_ranks = []
    per_lang = {"ne": ([], []), "en": ([], []), "np-roman": ([], [])}
    details = []

    for t in tests:
        q = t["question"]
        gold = t["gold_substring"]
        normalized = normalize_query(q)
        candidates = retrieve(collection, normalized,
                              top_k=args.top_k_retrieve, original_query=q)
        texts = [c["text"] for c in candidates]

        rank_before = rank_of_gold(texts, gold)

        # Re-rank with cross-encoder
        if texts:
            scores = ce.predict([(q, t[:1500]) for t in texts])
            order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
            reranked = [texts[i] for i in order]
        else:
            reranked = []

        rank_after = rank_of_gold(reranked, gold)

        before_ranks.append(rank_before)
        after_ranks.append(rank_after)
        per_lang[t["lang"]][0].append(rank_before)
        per_lang[t["lang"]][1].append(rank_after)
        details.append({
            "id": t["id"], "lang": t["lang"],
            "rank_before": rank_before, "rank_after": rank_after,
        })
        b = f"#{rank_before}" if rank_before else "MISS"
        a = f"#{rank_after}" if rank_after else "MISS"
        arrow = "↑" if (rank_before or 99) > (rank_after or 99) else (
                "↓" if (rank_before or 99) < (rank_after or 99) else "=")
        print(f"  [{t['lang']:9}] {t['id']}: {b:>5} -> {a:>5} {arrow}")

    def summarize(ranks):
        n = len(ranks)
        hits = [r for r in ranks if r is not None]
        return {
            "n": n,
            "recall@1": sum(1 for r in hits if r <= 1) / max(n, 1),
            "recall@5": sum(1 for r in hits if r <= 5) / max(n, 1),
            "recall@10": sum(1 for r in hits if r <= 10) / max(n, 1),
            "mrr": sum(1 / r for r in hits) / max(n, 1),
        }

    summary = {
        "before": summarize(before_ranks),
        "after_rerank": summarize(after_ranks),
        "per_language_before": {l: summarize(b) for l, (b, a) in per_lang.items()},
        "per_language_after":  {l: summarize(a) for l, (b, a) in per_lang.items()},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n=== Recall lift from re-ranker (top {args.top_k_retrieve} candidates) ===")
    print(f"{'metric':12} {'before':>8} {'after':>8} {'diff':>8}")
    for k in ("recall@1", "recall@5", "recall@10", "mrr"):
        b = summary["before"][k]
        a = summary["after_rerank"][k]
        print(f"{k:12} {b:8.3f} {a:8.3f} {a-b:+8.3f}")

    print("\n=== Per-language Recall@5 ===")
    for lang in ("ne", "en", "np-roman"):
        b = summary["per_language_before"][lang]["recall@5"]
        a = summary["per_language_after"][lang]["recall@5"]
        print(f"  {lang:9}  before={b:.3f}  after={a:.3f}  diff={a-b:+.3f}")
    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
