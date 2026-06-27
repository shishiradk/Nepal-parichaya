"""
OCR Correction Queue Reviewer
==============================
Interactive CLI to review user-submitted error reports and approve them
into corrections.json so they are applied on the next rebuild.

Usage:
    python scripts/review_queue.py           # review all pending reports
    python scripts/review_queue.py --all     # show approved/dismissed too
    python scripts/review_queue.py --stats   # summary only

Workflow after approving:
    python scripts/rebuild_chunks.py
    python nepali_rag_openai.py --build
"""

import sys
import json
import argparse
from pathlib import Path

QUEUE_FILE       = Path(__file__).parent / "corrections_queue.json"
CORRECTIONS_FILE = Path(__file__).parent / "corrections.json"

# Windows UTF-8 console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    with open(QUEUE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_queue(reports: list[dict]):
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def load_corrections() -> dict:
    if not CORRECTIONS_FILE.exists():
        return {}
    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_corrections(corrections: dict):
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


def add_correction(wrong: str, correct: str):
    corrections = load_corrections()
    corrections[wrong] = correct
    save_corrections(corrections)
    print(f"  Added: {wrong!r} → {correct!r}")


def print_report(idx: int, total: int, r: dict):
    print(f"\n{'='*60}")
    print(f"  Report {idx}/{total}  [{r['status'].upper()}]  {r['timestamp']}")
    print(f"{'='*60}")
    print(f"  Question    : {r.get('question', '')}")
    print(f"  Wrong text  : {r.get('wrong_text', '')}")
    print(f"  Correct text: {r.get('correct_text', '')}")
    if r.get("description"):
        print(f"  Description : {r['description']}")
    excerpt = r.get("answer_excerpt", "")
    if excerpt:
        print(f"\n  Answer excerpt:\n  ...{excerpt[:200]}...")


def stats(reports: list[dict]):
    pending   = sum(1 for r in reports if r["status"] == "pending")
    approved  = sum(1 for r in reports if r["status"] == "approved")
    dismissed = sum(1 for r in reports if r["status"] == "dismissed")
    print(f"\nQueue: {len(reports)} total  |  {pending} pending  |  {approved} approved  |  {dismissed} dismissed")
    corrections = load_corrections()
    real = {k: v for k, v in corrections.items() if not k.startswith("_")}
    print(f"corrections.json: {len(real)} active correction(s)")


def review_pending(reports: list[dict], show_all: bool = False) -> int:
    targets = [r for r in reports if r["status"] == "pending"] if not show_all else reports
    if not targets:
        print("\nNo pending reports.")
        return 0

    approved_count = 0
    for report in targets:
        idx = reports.index(report) + 1
        print_report(idx, len(reports), report)

        if report["status"] != "pending":
            print("\n  (already reviewed — press Enter to skip)")
            input("  > ")
            continue

        print("\n  Actions: [a] approve  [e] edit+approve  [d] dismiss  [s] skip  [q] quit")
        action = input("  > ").strip().lower()

        if action == "q":
            print("  Quitting. Changes so far are saved.")
            break
        elif action == "a":
            wrong   = report["wrong_text"]
            correct = report["correct_text"]
            add_correction(wrong, correct)
            report["status"] = "approved"
            approved_count += 1
        elif action == "e":
            wrong   = input(f"  Wrong text   [{report['wrong_text']}]: ").strip() or report["wrong_text"]
            correct = input(f"  Correct text [{report['correct_text']}]: ").strip() or report["correct_text"]
            report["wrong_text"]   = wrong
            report["correct_text"] = correct
            add_correction(wrong, correct)
            report["status"] = "approved"
            approved_count += 1
        elif action == "d":
            report["status"] = "dismissed"
            print("  Dismissed.")
        else:
            print("  Skipped.")

    save_queue(reports)
    return approved_count


def main():
    parser = argparse.ArgumentParser(description="Review OCR correction queue")
    parser.add_argument("--all",   action="store_true", help="Show all reports including reviewed")
    parser.add_argument("--stats", action="store_true", help="Print summary and exit")
    args = parser.parse_args()

    reports = load_queue()

    if not reports:
        print("Queue is empty. No reports submitted yet.")
        return

    stats(reports)

    if args.stats:
        return

    approved = review_pending(reports, show_all=args.all)

    print(f"\n{'='*60}")
    print(f"  Done. {approved} correction(s) approved.")
    if approved:
        print("\n  Next steps to rebuild the knowledge base:")
        print("    python scripts/rebuild_chunks.py")
        print("    python nepali_rag_openai.py --build")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
