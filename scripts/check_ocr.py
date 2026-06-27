"""
OCR Error Scanner for Nepal Parichaya markdown files
=====================================================
Sends each markdown file to GPT and asks it to flag suspicious Devanagari
text that looks like OCR misreads. Outputs a suggested corrections.json patch.

Usage:
    python scripts/check_ocr.py                  # scan all markdown files
    python scripts/check_ocr.py --file Nepal_Parichaya-0.md   # one file
    python scripts/check_ocr.py --dry-run        # print suggestions, don't write

Cost: ~$0.02–0.05 for all 43 files (gpt-4o-mini).
"""

import os
import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

MARKDOWN_DIR     = Path("markdown_output")
CORRECTIONS_FILE = Path("scripts/corrections.json")

_SYSTEM_PROMPT = """\
You are an OCR error checker for Nepali Devanagari text extracted from a PDF \
using Google Document AI.

Your task: read the text and identify words or short phrases that look like OCR \
misreads — where a character was confused with a visually similar one.

Common Devanagari OCR confusions:
  घ ↔ थ   ण ↔ न   ट ↔ त   ढ ↔ ध   ब ↔ व   ह ↔ न   ञ ↔ ज   श ↔ ष
  ँ (chandrabindu) dropped or swapped with ं (anusvara)
  conjunct characters split or merged incorrectly

Rules:
1. Only flag things you are CONFIDENT are OCR errors, not dialect spellings.
2. Return a JSON array. Each item: {"wrong": "...", "correct": "...", "context": "..."}
   where "context" is 5–10 surrounding words so the human can verify.
3. If you find NO errors, return an empty array: []
4. Do NOT flag proper nouns, place names, or technical terms you are uncertain about.
5. Short output only — no explanation outside the JSON array.
"""


def scan_file(client: OpenAI, filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")

    # Send in chunks of ~3000 chars to stay within context and cost limits
    chunk_size = 3000
    all_findings = []
    seen = set()

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": chunk},
                ],
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            # GPT may return {"errors": [...]} or just [...]
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
            for item in items:
                key = (item.get("wrong", ""), item.get("correct", ""))
                if key[0] and key[1] and key not in seen:
                    seen.add(key)
                    all_findings.append(item)
        except Exception as e:
            print(f"  Warning: GPT error on chunk {i//chunk_size + 1}: {e}")

    return all_findings


def load_existing_corrections() -> dict:
    if not CORRECTIONS_FILE.exists():
        return {}
    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def save_corrections(new_entries: dict):
    existing = {}
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    merged = {**existing, **new_entries}
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(new_entries)} new correction(s) to {CORRECTIONS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Scan markdown files for OCR errors")
    parser.add_argument("--file",    type=str, help="Scan a single file (filename only, e.g. Nepal_Parichaya-0.md)")
    parser.add_argument("--dry-run", action="store_true", help="Print suggestions without writing corrections.json")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    # Windows UTF-8
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    client = OpenAI()
    existing = load_existing_corrections()

    if args.file:
        files = [MARKDOWN_DIR / args.file]
    else:
        files = sorted(MARKDOWN_DIR.glob("*.md"))

    print(f"Scanning {len(files)} file(s) for OCR errors...\n")

    all_suggestions: dict[str, str] = {}
    total_found = 0

    for filepath in files:
        print(f"  {filepath.name} ...", end=" ", flush=True)
        findings = scan_file(client, filepath)

        new = [f for f in findings if f.get("wrong") not in existing]
        print(f"{len(findings)} candidate(s), {len(new)} new")

        for f in new:
            wrong   = f.get("wrong",   "").strip()
            correct = f.get("correct", "").strip()
            context = f.get("context", "").strip()
            if wrong and correct and wrong != correct:
                print(f"    OCR error : {wrong!r}")
                print(f"    Should be : {correct!r}")
                print(f"    Context   : ...{context}...")
                print()
                all_suggestions[wrong] = correct
                total_found += 1

    print(f"\n{'='*55}")
    print(f"  Total new suggestions : {total_found}")
    print(f"  Already in corrections: {len(existing)}")
    print(f"{'='*55}")

    if not all_suggestions:
        print("\nNo new OCR errors found.")
        return

    if args.dry_run:
        print("\n[Dry run] Suggested additions to corrections.json:")
        print(json.dumps(all_suggestions, ensure_ascii=False, indent=2))
    else:
        print("\nReview the suggestions above.")
        answer = input("Add all to corrections.json? [y/N] ").strip().lower()
        if answer == "y":
            save_corrections(all_suggestions)
            print("\nNext steps:")
            print("  python scripts/rebuild_chunks.py")
            print("  python nepali_rag_openai.py --build")
        else:
            print("Nothing written. Edit corrections.json manually if needed.")


if __name__ == "__main__":
    main()
