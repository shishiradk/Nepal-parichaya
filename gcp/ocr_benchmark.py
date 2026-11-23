"""Score saved OCR outputs for PDF pages and images against verified text.

Run: python gcp/ocr_benchmark.py manifest.json --output scores.json

Manifest paths are relative to the manifest. Each sample represents a single
page, image, or crop. All variants must cover the same source region.

{
  "samples": [
    {
      "id": "book-page-1",
      "kind": "pdf_page",
      "reference": "verified/page-1.txt",
      "variants": {
        "document_ai_original": "raw/page-1.json",
        "document_ai_preprocessed": "processed/page-1.json",
        "open_source": "open-source/page-1.txt"
      }
    }
  ]
}

Use one Document AI JSON response per sample (top-level text or document.text).
For multi-page responses, export the matching page text to a UTF-8 file first.
Plain text and Markdown files are scored as-is after normalization.
"""

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize(text):
    """Ignore whitespace formatting and canonical Unicode representation."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference, prediction):
    """Compute character substitutions, insertions and deletions as one total."""
    if len(prediction) > len(reference):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def read_prediction(path):
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() != ".json":
        return text
    document = json.loads(text)
    if isinstance(document, dict):
        document = document.get("document", document)
    if not isinstance(document, dict) or not isinstance(document.get("text"), str):
        raise ValueError(f"{path}: expected an OCR text string in text or document.text")
    if isinstance(document.get("pages"), list) and len(document["pages"]) > 1:
        raise ValueError(f"{path}: export the matching page text for this sample")
    return document["text"]


def summarize(rows):
    errors = sum(row["character_errors"] for row in rows)
    characters = sum(row["reference_characters"] for row in rows)
    exact = sum(row["exact_match"] for row in rows)
    return {
        "samples": len(rows),
        "character_error_rate": errors / characters if characters else None,
        "exact_match_rate": exact / len(rows),
        "character_errors": errors,
        "reference_characters": characters,
    }


def benchmark(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    base = manifest_path.parent
    rows = []
    by_variant = defaultdict(list)
    by_kind = defaultdict(lambda: defaultdict(list))
    seen = set()
    expected_variants = None

    for sample in manifest["samples"]:
        sample_id = sample["id"]
        kind = sample["kind"]
        variants = sample["variants"]
        if sample_id in seen or not variants:
            raise ValueError(f"{sample_id}: expected a unique ID and at least one variant")
        if expected_variants is None:
            expected_variants = set(variants)
        if set(variants) != expected_variants:
            raise ValueError(f"{sample_id}: every sample must include the same variants")
        seen.add(sample_id)
        reference = normalize((base / sample["reference"]).read_text(encoding="utf-8-sig"))
        for variant, path in variants.items():
            prediction = normalize(read_prediction(base / path))
            errors = edit_distance(reference, prediction)
            row = {
                "id": sample_id,
                "kind": kind,
                "variant": variant,
                "reference_characters": len(reference),
                "character_errors": errors,
                "exact_match": errors == 0,
            }
            rows.append(row)
            by_variant[variant].append(row)
            by_kind[kind][variant].append(row)
    if not rows:
        raise ValueError("The manifest must contain at least one sample")
    return {
        "normalization": "NFC and collapsed whitespace; punctuation and digits preserved",
        "overall": {variant: summarize(items) for variant, items in by_variant.items()},
        "by_kind": {
            kind: {variant: summarize(items) for variant, items in variants.items()}
            for kind, variants in by_kind.items()
        },
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="Save the JSON report instead of printing it")
    args = parser.parse_args()
    try:
        report = json.dumps(benchmark(args.manifest), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        else:
            print(report, end="")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(2, f"OCR benchmark: {error}\n")


if __name__ == "__main__":
    main()
