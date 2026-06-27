"""
End-to-End Evaluation
=====================
Runs the full RAG pipeline on each test question and measures:

  - Answer correctness     (LLM-as-judge vs gold_answer, 1-5)
  - Faithfulness           (LLM-as-judge: is every claim in the retrieved context?)
  - Language adherence     (script of answer matches script of question)
  - Refusal accuracy       (on out-of-scope trick questions)
  - Cost per query         (avg + p95)
  - Latency                (p50 + p95)

Usage:
    python eval/run_e2e_eval.py \
        --test  eval/test_set.jsonl \
        --trick eval/trick_questions.jsonl \
        --out   eval/results/e2e.json
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from statistics import median

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from rag import normalize_query, retrieve, generate_answer, load_vector_store


def with_backoff(fn, *args, tries=8, base=3.0, **kwargs):
    """Retry on rate-limit / transient API errors with exponential backoff."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, APIConnectionError, APIError) as e:
            if i == tries - 1:
                raise
            wait = base * (2 ** i)
            print(f"    transient error ({type(e).__name__}), sleeping {wait:.0f}s ...")
            time.sleep(wait)

JUDGE_MODEL = "gpt-4o"
DEV = re.compile(r"[ऀ-ॿ]")

REFUSAL_MARKERS = [
    "माफ गर्नुहोस्",
    "जानकारी दिइएको सन्दर्भमा भेटिएन",
    "sorry, i couldn't find",
    "couldn't find that in the provided context",
    "not in the provided context",
    "i couldn't find",
]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def answer_script(q: str, a: str) -> str:
    """Returns 'match' if answer-script matches question-script, else 'mismatch'."""
    q_dev = bool(DEV.search(q))
    a_dev = bool(DEV.search(a))
    # English answer may quote a Nepali term — only flag mismatch if it's PRIMARILY off-script
    if q_dev:
        return "match" if a_dev else "mismatch"
    # English/Romanized question — answer should not be primarily Devanagari
    dev_ratio = len(DEV.findall(a)) / max(len(a), 1)
    return "match" if dev_ratio < 0.3 else "mismatch"


CORRECTNESS_PROMPT = """You are evaluating a RAG system answering questions about Nepal.

Question: {q}

Gold answer: {gold}

System answer: {ans}

Score the system answer on factual correctness vs the gold answer:
  5 = fully correct, all key facts match
  4 = mostly correct, minor omission
  3 = partially correct, missing or fuzzy on some facts
  2 = mostly wrong but on the right topic
  1 = wrong or off-topic

Return JSON only: {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""


FAITHFULNESS_PROMPT = """You are checking whether a RAG answer is grounded in its retrieved context.

Retrieved context:
{ctx}

Answer given:
{ans}

Is every factual claim in the answer supported by the retrieved context above?
  5 = every claim supported
  4 = one minor unsupported detail
  3 = some unsupported claims
  2 = mostly unsupported
  1 = entirely unsupported or contradicts context

Return JSON only: {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""


def judge(client: OpenAI, prompt: str) -> dict:
    try:
        r = with_backoff(
            client.chat.completions.create,
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = r.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {"score": 0, "reason": "parse-fail"}
    except Exception as e:
        return {"score": 0, "reason": f"judge-error: {e}"}


def run_pipeline(collection, q: str) -> tuple[str, list[dict], dict, float]:
    """Retrieves contexts, then truncates each chunk before generation so the
    LLM prompt stays under the per-request TPM ceiling (30k for gpt-4o tier).
    Devanagari is token-heavy: ~1.2-1.5 tokens/char. Capping each chunk at
    ~1200 chars × top 6 chunks keeps the entire request comfortably under 25k.
    Note: this evaluates a slightly trimmed configuration vs. production (top_k=8,
    full chunks). Mention in the report.
    """
    t0 = time.time()
    normalized = with_backoff(normalize_query, q)
    contexts = retrieve(collection, normalized, original_query=q)
    trimmed = [dict(c, text=c["text"][:1200]) for c in contexts[:6]]
    answer, usage = with_backoff(generate_answer, q, trimmed)
    return answer, trimmed, usage, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--trick", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tests = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    tricks = ([json.loads(l) for l in open(args.trick, encoding="utf-8") if l.strip()]
              if args.trick else [])

    collection = load_vector_store()
    client = OpenAI()

    results = []
    costs, latencies = [], []
    correctness_scores, faithfulness_scores = [], []
    lang_match = {"match": 0, "mismatch": 0}
    per_lang = {}

    print(f"\n=== Running {len(tests)} test questions ===")
    for t in tests:
        ans, ctxs, usage, latency = run_pipeline(collection, t["question"])
        # Cap each chunk to keep faithfulness-judge prompt under TPM limit (~30k tokens).
        # Devanagari is token-heavy: ~1000 chars ≈ 800-1200 tokens.
        ctx_str = "\n---\n".join(c["text"][:1000] for c in ctxs[:5])

        corr = judge(client, CORRECTNESS_PROMPT.format(
            q=t["question"], gold=t["gold_answer"], ans=ans))
        faith = judge(client, FAITHFULNESS_PROMPT.format(ctx=ctx_str, ans=ans))
        lm = answer_script(t["question"], ans)

        correctness_scores.append(corr.get("score", 0))
        faithfulness_scores.append(faith.get("score", 0))
        lang_match[lm] += 1
        per_lang.setdefault(t["lang"], []).append(corr.get("score", 0))
        costs.append(usage["cost"])
        latencies.append(latency)

        print(f"  [{t['lang']:9}] {t['id']}: "
              f"corr={corr.get('score',0)} faith={faith.get('score',0)} "
              f"lang={lm} cost=${usage['cost']:.4f} t={latency:.1f}s")

        results.append({
            "id": t["id"], "lang": t["lang"], "type": "test",
            "question": t["question"], "gold": t["gold_answer"], "answer": ans,
            "correctness": corr, "faithfulness": faith,
            "lang_match": lm, "cost": usage["cost"], "latency_s": latency,
        })

    refusals_correct = 0
    print(f"\n=== Running {len(tricks)} trick (out-of-scope) questions ===")
    for t in tricks:
        ans, ctxs, usage, latency = run_pipeline(collection, t["question"])
        refused = any(m in ans.lower() for m in REFUSAL_MARKERS)
        refusals_correct += int(refused)
        costs.append(usage["cost"])
        latencies.append(latency)
        print(f"  [{t['lang']:9}] {t['id']}: refused={refused}  "
              f"cost=${usage['cost']:.4f}")
        results.append({
            "id": t["id"], "lang": t["lang"], "type": "trick",
            "question": t["question"], "answer": ans,
            "refused_correctly": refused,
            "cost": usage["cost"], "latency_s": latency,
        })

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    summary = {
        "n_test": len(tests),
        "n_trick": len(tricks),
        "correctness_mean": mean(correctness_scores),
        "correctness_5_pct": sum(1 for s in correctness_scores if s == 5) / max(len(correctness_scores), 1),
        "faithfulness_mean": mean(faithfulness_scores),
        "hallucination_pct": sum(1 for s in faithfulness_scores if s <= 3) / max(len(faithfulness_scores), 1),
        "lang_adherence_pct": lang_match["match"] / max(sum(lang_match.values()), 1),
        "refusal_accuracy_pct": refusals_correct / max(len(tricks), 1) if tricks else None,
        "cost_avg": mean(costs),
        "cost_p95": percentile(costs, 95),
        "latency_p50_s": median(latencies) if latencies else 0,
        "latency_p95_s": percentile(latencies, 95),
        "per_language_correctness": {l: mean(s) for l, s in per_lang.items()},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30}: {v:.3f}")
        else:
            print(f"  {k:30}: {v}")
    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
