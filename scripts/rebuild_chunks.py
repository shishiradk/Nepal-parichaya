"""
Topic-aware Chunker for Nepal Parichaya
========================================
Reads raw markdown from markdown_output/, cleans OCR noise, then splits
by topic headings — not by character count.

Why this matters:
  Old chunker: "दसैं is 10 days... [page break] ...तिहार is 5 days"
               → ONE chunk mixes two festivals → bad retrieval
  New chunker: "दसैं" chunk  +  "तिहार" chunk  → clean retrieval

Usage:
    python rebuild_chunks.py               # build clean chunks
    python rebuild_chunks.py --stats       # show stats only
    python rebuild_chunks.py --preview 5   # preview first 5 chunks
"""

import re
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
MARKDOWN_DIR     = Path("markdown_output")
OUTPUT_DIR       = Path("chunked_output/clean_chunks")
CORRECTIONS_FILE = Path(__file__).parent / "corrections.json"

# ── Tuning ─────────────────────────────────────────────────────────────────────
MIN_CHUNK_CHARS   = 120    # discard chunks shorter than this
MAX_CHUNK_CHARS   = 1200   # split chunks longer than this at paragraph boundary
MIN_NEPALI_RATIO  = 0.55   # discard chunks with too little Devanagari

# ── OCR noise patterns ─────────────────────────────────────────────────────────
# These exact patterns appear repeatedly in Nepal Parichaya OCR output
_NOISE_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*/\s*नेपाल\s*परिचय\s*$"),   # "२१८/नेपाल परिचय"
    re.compile(r"^\s*नेपाल\s*परिचय\s*/\s*\d{1,4}\s*$"),   # "नेपाल परिचय/२१९"
    re.compile(r"^\s*\d{1,4}\s*$"),                          # standalone page numbers
    re.compile(r"^\s*फोन\s*नं.*\d{6,}"),                    # phone numbers
    re.compile(r"^\s*वेब\s*:\s*www\."),                      # website lines
    re.compile(r"^\s*www\.", re.IGNORECASE),
    re.compile(r"^\s*http[s]?://", re.IGNORECASE),
    re.compile(r"^\s*[:\-–—]\s*\d{1,4}\s*$"),               # "- 23" style fragments
    re.compile(r"^\s*[०-९]{1,4}\s*$"),                       # Devanagari numerals alone
]

# Lines that are clearly image captions / layout artifacts
# (short lines surrounded by non-paragraph context)
_CAPTION_MAX_LEN = 35   # lines ≤ this AND not followed by paragraph are treated as captions


def _is_devanagari(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ"


def _nepali_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _is_devanagari(c)) / len(letters)


def _chunk_id(text: str, prefix: str = "") -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}" if prefix else h


# ── Step 1: Find markdown files ────────────────────────────────────────────────

def find_markdown_files(base_dir: Path) -> list[Path]:
    """
    Recursively find all .md files. Deduplicate by filename stem so that
    if the same file exists in multiple subdirectories (common with
    Document AI batch output), only the most-recently-modified copy is kept.
    """
    all_files = list(base_dir.rglob("*.md"))

    # Keep one file per stem name (prefer most recently modified)
    by_stem: dict[str, Path] = {}
    for fp in all_files:
        stem = fp.stem
        if stem not in by_stem or fp.stat().st_mtime > by_stem[stem].stat().st_mtime:
            by_stem[stem] = fp

    return sorted(by_stem.values(), key=lambda p: p.stem)


# ── Corrections ────────────────────────────────────────────────────────────────

def load_corrections() -> dict:
    """Load known OCR error corrections from scripts/corrections.json."""
    if not CORRECTIONS_FILE.exists():
        return {}
    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def apply_corrections(text: str, corrections: dict) -> str:
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


# ── Step 2: Clean OCR noise from a single file's text ─────────────────────────

def clean_text(raw: str, corrections: dict | None = None) -> str:
    """
    Remove OCR artifacts line by line and apply known corrections.
    Returns cleaned text preserving paragraph structure.
    """
    # Apply known OCR corrections first (whole-text replace, before line processing)
    if corrections:
        raw = apply_corrections(raw, corrections)

    lines = raw.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Remove known noise patterns
        if any(p.match(stripped) for p in _NOISE_PATTERNS):
            continue

        # Remove lines that are purely punctuation / symbols with no Nepali
        if stripped and len(stripped) <= 10:
            if not any(_is_devanagari(c) for c in stripped):
                continue

        cleaned.append(line)

    # Collapse runs of 3+ blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


# ── Step 3: Detect headings ────────────────────────────────────────────────────

def _is_heading(line: str, following_lines: list[str]) -> bool:
    """
    Return True if `line` looks like a topic heading in Nepal Parichaya.

    Heading characteristics:
      - 2 to 55 characters
      - Mostly Devanagari (≥ 45%)
      - No sentence-ending punctuation (।)
      - No trailing comma
      - Followed by substantive paragraph text (≥ 60 chars combined)
    """
    s = line.strip()
    if not s:
        return False

    # Length guard
    if len(s) < 2 or len(s) > 55:
        return False

    # Must contain some Devanagari
    dev_chars = sum(1 for c in s if _is_devanagari(c))
    if dev_chars == 0:
        return False
    if dev_chars / len(s) < 0.45:
        return False

    # Sentence-final punctuation means it is a sentence, not a heading
    if "।" in s or s.endswith(",") or s.endswith(";"):
        return False

    # Pure numbers are not headings
    if re.match(r"^[\d०-९\s\-/]+$", s):
        return False

    # Must be followed by substantive text (rules out lone captions)
    next_text = " ".join(l.strip() for l in following_lines[:5] if l.strip())
    if len(next_text) < 60:
        return False

    # Poetry guard: if 4+ of the next 6 lines look like poem lines, this line
    # might be a poem line too — but only suppress heading detection when the
    # current line is itself long enough (≥25 chars) to plausibly be a lyric.
    # Short titles like "राष्ट्रिय गान" (14 chars) must stay detectable as headings
    # even when followed entirely by anthem lyrics.
    poetry_like = sum(
        1 for l in following_lines[:6]
        if l.strip() and len(l.strip()) < 55 and "।" not in l
        and any(_is_devanagari(c) for c in l)
    )
    if poetry_like >= 4 and len(s) >= 25:
        return False

    return True


# ── Step 4: Split into topic chunks ───────────────────────────────────────────

def split_by_topics(text: str, source_name: str) -> list[dict]:
    """
    Walk through lines. When a heading is detected, close the previous
    chunk and start a new one. Return list of raw chunk dicts.
    """
    lines = text.splitlines()
    chunks: list[dict] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def _flush():
        nonlocal current_heading, current_body
        body = "\n".join(current_body).strip()
        if body:
            text_block = f"{current_heading}\n\n{body}" if current_heading else body
            chunks.append({
                "heading": current_heading or "",
                "body": body,
                "text": text_block,
                "source": source_name,
            })
        current_heading = None
        current_body = []

    for i, line in enumerate(lines):
        following = lines[i + 1 : i + 8]
        if _is_heading(line, following):
            _flush()
            current_heading = line.strip()
        else:
            if line.strip():
                current_body.append(line)
            elif current_body:              # preserve single blank lines inside paragraphs
                current_body.append("")

    _flush()  # last chunk
    return chunks


# ── Step 5: Sub-split oversized chunks at paragraph boundaries ─────────────────

def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split a single string at sentence boundaries when it exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    while len(text) > max_chars:
        # Find last sentence boundary (। or newline) within the limit
        cut = text.rfind("।", 0, max_chars)
        if cut == -1:
            cut = text.rfind("\n", 0, max_chars)
        if cut == -1:
            cut = max_chars
        else:
            cut += 1  # include the boundary character
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _split_long_chunk(chunk: dict, max_chars: int) -> list[dict]:
    """If a chunk exceeds max_chars, split at blank-line paragraph boundaries.
    If a single paragraph itself exceeds max_chars, hard-split it at sentences.
    """
    if len(chunk["text"]) <= max_chars:
        return [chunk]

    paragraphs = re.split(r"\n\n+", chunk["body"])
    # Hard-split any paragraph that's individually over the limit
    flat_paras: list[str] = []
    for para in paragraphs:
        if len(para) > max_chars:
            flat_paras.extend(_hard_split(para, max_chars))
        else:
            flat_paras.append(para)

    parts: list[dict] = []
    current_paras: list[str] = []
    current_len = 0

    for para in flat_paras:
        if current_len + len(para) > max_chars and current_paras:
            body = "\n\n".join(current_paras)
            heading = chunk["heading"] if not parts else f"{chunk['heading']} (cont.)"
            parts.append({
                "heading": heading,
                "body": body,
                "text": f"{heading}\n\n{body}" if heading else body,
                "source": chunk["source"],
            })
            current_paras = [para]
            current_len = len(para)
        else:
            current_paras.append(para)
            current_len += len(para)

    if current_paras:
        body = "\n\n".join(current_paras)
        heading = chunk["heading"] if not parts else f"{chunk['heading']} (cont.)"
        parts.append({
            "heading": heading,
            "body": body,
            "text": f"{heading}\n\n{body}" if heading else body,
            "source": chunk["source"],
        })

    return parts


# ── Step 6: Quality filter ─────────────────────────────────────────────────────

def _passes_quality(chunk: dict) -> bool:
    text = chunk["text"]
    if len(text) < MIN_CHUNK_CHARS:
        return False
    if _nepali_ratio(text) < MIN_NEPALI_RATIO:
        return False
    return True


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_clean_chunks(md_files: list[Path]) -> list[dict]:
    """Run the full pipeline over all markdown files."""
    corrections = load_corrections()
    if corrections:
        print(f"Loaded {len(corrections)} OCR correction(s) from {CORRECTIONS_FILE.name}")

    all_chunks: list[dict] = []

    for filepath in md_files:
        raw = filepath.read_text(encoding="utf-8")
        cleaned = clean_text(raw, corrections)

        # source label: just the filename without directory noise
        source_name = filepath.stem   # e.g. "Nepal_Parichaya-23"

        raw_chunks = split_by_topics(cleaned, source_name)

        # sub-split long ones
        split_chunks: list[dict] = []
        for c in raw_chunks:
            split_chunks.extend(_split_long_chunk(c, MAX_CHUNK_CHARS))

        # quality filter
        good = [c for c in split_chunks if _passes_quality(c)]
        all_chunks.extend(good)

    # Assign unique IDs
    for i, chunk in enumerate(all_chunks):
        cid = _chunk_id(chunk["text"], chunk["source"])
        chunk["chunk_id"] = cid
        chunk["nepali_ratio"] = round(_nepali_ratio(chunk["text"]), 4)
        chunk["chunk_size"] = len(chunk["text"])
        chunk["index"] = i

    return all_chunks


# ── Save output ────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale chunk files from previous runs so old chunks don't accumulate
    for old_file in out_dir.glob("*.md"):
        old_file.unlink()

    metadata = []

    for chunk in chunks:
        cid = chunk["chunk_id"]

        # Individual .md file (same format as old chunks — compatible with nepali_rag_openai.py)
        md_content = (
            f"---\n"
            f"chunk_id: {cid}\n"
            f"source_file: {chunk['source']}\n"
            f"heading: {chunk['heading']}\n"
            f"chunk_size: {chunk['chunk_size']}\n"
            f"nepali_ratio: {chunk['nepali_ratio']}\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"---\n\n"
            f"{chunk['text']}\n"
        )
        (out_dir / f"{cid}.md").write_text(md_content, encoding="utf-8")

        metadata.append({
            "chunk_id": cid,
            "source_file": chunk["source"],
            "heading": chunk["heading"],
            "chunk_size": chunk["chunk_size"],
            "nepali_ratio": chunk["nepali_ratio"],
            "markdown_file": f"clean_chunks/{cid}.md",
        })

    # metadata JSON
    meta_path = out_dir.parent / "clean_chunks_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(chunks)} chunks → {out_dir}")
    print(f"Metadata → {meta_path}")


# ── Stats & preview ────────────────────────────────────────────────────────────

def print_stats(chunks: list[dict]):
    if not chunks:
        print("No chunks generated.")
        return

    sizes = [c["chunk_size"] for c in chunks]
    ratios = [c["nepali_ratio"] for c in chunks]
    headings = [c for c in chunks if c["heading"]]

    print(f"\n{'='*55}")
    print(f"  Clean chunks generated : {len(chunks):,}")
    print(f"  Chunks with heading    : {len(headings):,}")
    print(f"  Avg chunk size         : {sum(sizes)//len(sizes):,} chars")
    print(f"  Min / Max size         : {min(sizes)} / {max(sizes)}")
    print(f"  Avg Nepali ratio       : {sum(ratios)/len(ratios):.1%}")
    print(f"{'='*55}")

    # Show heading distribution (sample)
    sample_headings = [c["heading"] for c in chunks if c["heading"]][:20]
    if sample_headings:
        print("\n  Sample topic headings detected:")
        for h in sample_headings:
            print(f"    • {h}")
    print()


def preview_chunks(chunks: list[dict], n: int = 5):
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n{'─'*60}")
        print(f"Chunk {i+1} | ID: {chunk['chunk_id']} | Source: {chunk['source']}")
        print(f"Heading : {chunk['heading'] or '(none)'}")
        print(f"Size    : {chunk['chunk_size']} chars | Nepali: {chunk['nepali_ratio']:.0%}")
        print(f"{'─'*60}")
        print(chunk["text"][:400])
        if len(chunk["text"]) > 400:
            print("  [...]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    # Windows UTF-8 fix
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Rebuild topic-aware chunks from Nepal Parichaya markdown")
    parser.add_argument("--md-dir", default=str(MARKDOWN_DIR), help="Directory containing markdown files")
    parser.add_argument("--out",    default=str(OUTPUT_DIR),   help="Output directory for clean chunks")
    parser.add_argument("--stats",  action="store_true",        help="Print stats and exit without saving")
    parser.add_argument("--preview", type=int, metavar="N",    help="Preview first N chunks and exit")
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    out_dir = Path(args.out)

    print("\nNepal Parichaya — Topic-aware Chunker")
    print("=" * 55)

    # Find files
    md_files = find_markdown_files(md_dir)
    if not md_files:
        print(f"No .md files found in: {md_dir}")
        print("Check that markdown_output/ exists and contains .md files.")
        return

    print(f"Found {len(md_files)} markdown files in {md_dir}")

    # Build chunks
    print("Cleaning OCR noise and splitting by topic headings...")
    chunks = build_clean_chunks(md_files)

    print_stats(chunks)

    if args.preview:
        preview_chunks(chunks, args.preview)
        return

    if args.stats:
        return

    # Save
    save_chunks(chunks, out_dir)

    print("\nNext step — rebuild the ChromaDB vector store:")
    print("  Set CHUNKS_DIR in nepali_rag_openai.py to: chunked_output/clean_chunks")
    print("  Then run: python nepali_rag_openai.py --build")


if __name__ == "__main__":
    main()
