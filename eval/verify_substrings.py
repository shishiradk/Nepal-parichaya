"""
Test-set Substring Verifier
============================
Scans test_set.jsonl and warns about any `gold_substring` that does NOT appear
verbatim in any chunk. Prevents the foot-gun where you write a plausible-looking
gold_substring (e.g. "१४७,५१६") that isn't actually in the OCR'd text.

For each miss, it also suggests candidate substrings from chunks that look
similar (share a long token), so you can pick a real one and retry.

Usage:
    python eval/verify_substrings.py --test eval/test_set.jsonl
    python eval/verify_substrings.py --test eval/test_set.jsonl --chunks-dir chunked_output/clean_chunks
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def load_chunks_text(chunks_dir: Path) -> str:
    """Concatenate every chunk's text — we only need substring-search."""
    if not chunks_dir.exists():
        sys.exit(f"chunks dir not found: {chunks_dir}")
    files = sorted(chunks_dir.glob("*.md"))
    if not files:
        sys.exit(f"no .md chunks in {chunks_dir}")
    text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
    return text, len(files)


def suggest_alternatives(gold_sub: str, all_text_norm: str, n: int = 3) -> list[str]:
    """Find candidate phrases in the corpus that share the longest token of gold_sub."""
    tokens = [t for t in gold_sub.split() if len(t) >= 3]
    if not tokens:
        return []
    longest = max(tokens, key=len)
    if normalize(longest) not in all_text_norm:
        return []
    # Pull short context windows around each occurrence
    needle = normalize(longest)
    suggestions = []
    start = 0
    while len(suggestions) < n:
        idx = all_text_norm.find(needle, start)
        if idx < 0:
            break
        snippet = all_text_norm[max(0, idx - 25): idx + len(needle) + 25]
        suggestions.append(snippet.strip())
        start = idx + len(needle)
    return suggestions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--chunks-dir", type=Path, default=Path("chunked_output/clean_chunks"))
    args = ap.parse_args()

    all_text, n_files = load_chunks_text(args.chunks_dir)
    all_text_norm = normalize(all_text)
    print(f"Loaded {n_files} chunks ({len(all_text):,} chars) from {args.chunks_dir}")

    tests = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    print(f"Checking {len(tests)} test entries...\n")

    misses = []
    by_lang = Counter()
    for t in tests:
        sub = t["gold_substring"]
        if normalize(sub) in all_text_norm:
            print(f"  ✓ {t['id']}  [{t['lang']:9}]  '{sub}'")
        else:
            print(f"  ✗ {t['id']}  [{t['lang']:9}]  '{sub}'  ← NOT FOUND")
            misses.append(t)
            by_lang[t["lang"]] += 1

    if not misses:
        print(f"\n✅ All {len(tests)} gold_substrings verified.")
        return

    print(f"\n⚠️  {len(misses)} miss(es) of {len(tests)}  ({dict(by_lang)})")
    print("These entries will under-report retrieval recall. Suggestions:\n")
    for t in misses:
        print(f"  {t['id']}  Q: {t['question'][:80]}")
        print(f"          gold_substring: '{t['gold_substring']}'")
        sugg = suggest_alternatives(t["gold_substring"], all_text_norm)
        if sugg:
            print(f"          ↳ similar phrases in chunks:")
            for s in sugg:
                print(f"              · ...{s}...")
        else:
            print(f"          ↳ no near-matches found — confirm the fact is actually in the book")
        print()


if __name__ == "__main__":
    main()
