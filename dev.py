#!/usr/bin/env python
"""Developer task runner — zero deps, pure stdlib.

Same tasks as the Makefile; works on Windows / Git Bash / Linux / macOS
without needing GNU Make installed.

Usage:
    python dev.py <task> [options]
    python dev.py help

Examples:
    python dev.py api              # start REST API on :8765
    python dev.py ui               # start Streamlit
    python dev.py eval-retrieval   # run the retrieval eval
    python dev.py api --port 5000  # override the default port
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so docstrings with arrows/em-dashes print cleanly.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent
PY = sys.executable                       # always use the current venv python
EVAL = REPO / "eval"
TEST_SET = str(EVAL / "test_set.jsonl")
TRICK_SET = str(EVAL / "trick_questions.jsonl")
RESULTS = EVAL / "results"

# ANSI colors — only emit if stdout is a real TTY (so piped/captured output
# stays plain text instead of `[33mtask[0m` literals).
if sys.stdout.isatty():
    B, G, Y, C, R = "\033[1m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
else:
    B = G = Y = C = R = ""


def run(cmd: list[str], **kw) -> int:
    """Run cmd, stream output, return exit code. Cwd is the repo root."""
    print(f"{C}$ {' '.join(cmd)}{R}")
    return subprocess.call(cmd, cwd=REPO, **kw)


# ─── tasks ─────────────────────────────────────────────────────────────────────

def install(args):
    """Install Python dependencies into the active venv."""
    return run([PY, "-m", "pip", "install", "-r", "requirements.txt"])


def migrate(args):
    """One-time Django migration (silences unapplied-migrations warning)."""
    return run([PY, "manage.py", "migrate"])


def build(args):
    """Build the ChromaDB vector store from chunked_output/."""
    return run([PY, "nepali_rag_openai.py", "--build"])


def chunks(args):
    """(Re)build topic-aware chunks from markdown_output/."""
    return run([PY, "scripts/rebuild_chunks.py"])


def api(args):
    """Start the Django REST API. Default: http://127.0.0.1:8765"""
    migrate(args)
    # Collect static so WhiteNoise can serve Swagger UI CSS/JS in any DEBUG mode.
    # Idempotent — `--noinput` skips the prompt; only changed files are copied.
    run([PY, "manage.py", "collectstatic", "--noinput", "--verbosity", "0"])
    host, port = args.host, args.port
    print(f"{G}→ API:   http://{host}:{port}/api/health{R}")
    print(f"{G}→ Docs:  http://{host}:{port}/api/docs/{R}")
    return run([PY, "manage.py", "runserver", f"{host}:{port}"])


def ui(args):
    """Start the Streamlit UI."""
    return run(["streamlit", "run", "streamlit_app.py"])


def cli(args):
    """Interactive CLI chat against the RAG."""
    return run([PY, "nepali_rag_openai.py"])


def verify_eval(args):
    """Verify every test_set.jsonl gold_substring exists in chunks."""
    return run([PY, str(EVAL / "verify_substrings.py"), "--test", TEST_SET])


def eval_retrieval(args):
    """Run retrieval eval (R@k, MRR) — cheap, embeddings only."""
    RESULTS.mkdir(exist_ok=True)
    return run([PY, str(EVAL / "run_retrieval_eval.py"),
                "--test", TEST_SET, "--out", str(RESULTS / "retrieval.json")])


def eval_e2e(args):
    """Run full end-to-end eval — correctness, faithfulness, cost (~$0.30)."""
    RESULTS.mkdir(exist_ok=True)
    return run([PY, str(EVAL / "run_e2e_eval.py"),
                "--test", TEST_SET, "--trick", TRICK_SET,
                "--out", str(RESULTS / "e2e.json")])


def eval_baselines(args):
    """Run no-RAG / BM25 / vector-only baselines (~$1.00)."""
    RESULTS.mkdir(exist_ok=True)
    return run([PY, str(EVAL / "run_baselines.py"),
                "--test", TEST_SET, "--out", str(RESULTS / "baselines.json")])


def eval_reranker(args):
    """Run cross-encoder re-ranker eval (downloads ~280MB model on first run)."""
    RESULTS.mkdir(exist_ok=True)
    return run([PY, str(EVAL / "run_reranker_eval.py"),
                "--test", TEST_SET, "--out", str(RESULTS / "reranker.json")])


def eval_all(args):
    """Run verify → retrieval → e2e → baselines (skips reranker)."""
    for step in (verify_eval, eval_retrieval, eval_e2e, eval_baselines):
        if rc := step(args):
            return rc
    return 0


def freeze(args):
    """Re-freeze requirements.txt (UTF-8) from the active venv."""
    out = subprocess.check_output([PY, "-m", "pip", "freeze"], cwd=REPO, text=True)
    (REPO / "requirements.txt").write_text(out, encoding="utf-8")
    print(f"wrote requirements.txt ({len(out.splitlines())} packages)")
    return 0


def clean(args):
    """Remove __pycache__ directories outside the venv."""
    n = 0
    for p in REPO.rglob("__pycache__"):
        if "venv" in p.parts or ".git" in p.parts:
            continue
        import shutil
        shutil.rmtree(p, ignore_errors=True)
        n += 1
    print(f"cleaned {n} __pycache__ director{'y' if n == 1 else 'ies'}")
    return 0


def schema(args):
    """Dump the OpenAPI schema to api-schema.yml."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    rc = run([PY, "manage.py", "spectacular", "--file", "api-schema.yml"])
    if rc == 0:
        print("wrote api-schema.yml")
    return rc


# ─── dispatcher ────────────────────────────────────────────────────────────────

TASKS = {
    # name              : (function,           help)
    "install":          install,
    "migrate":          migrate,
    "build":            build,
    "chunks":           chunks,
    "api":              api,
    "ui":               ui,
    "cli":              cli,
    "verify-eval":      verify_eval,
    "eval-retrieval":   eval_retrieval,
    "eval-e2e":         eval_e2e,
    "eval-baselines":   eval_baselines,
    "eval-reranker":    eval_reranker,
    "eval-all":         eval_all,
    "freeze":           freeze,
    "clean":            clean,
    "schema":           schema,
}


def show_help():
    print()
    print(f"{B}Nepal Parichaya RAG — developer tasks{R}")
    print()
    print(f"  Usage: {C}python dev.py <task> [options]{R}")
    print()
    sections = [
        ("Setup",        ["install", "migrate", "build", "chunks"]),
        ("Run",          ["api", "ui", "cli"]),
        ("Eval",         ["verify-eval", "eval-retrieval", "eval-e2e",
                          "eval-baselines", "eval-reranker", "eval-all"]),
        ("Quality of life", ["freeze", "clean", "schema"]),
    ]
    for title, names in sections:
        print(f"  {B}{title}{R}")
        for name in names:
            doc = (TASKS[name].__doc__ or "").splitlines()[0]
            print(f"    {Y}{name:<16}{R} {doc}")
        print()
    print(f"  Examples:")
    print(f"    {G}python dev.py api{R}                  start REST API on http://127.0.0.1:8765")
    print(f"    {G}python dev.py api --port 5000{R}      override the default port")
    print(f"    {G}python dev.py eval-retrieval{R}       run the retrieval eval")
    print()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        show_help()
        return 0

    task = sys.argv[1]
    if task not in TASKS:
        print(f"unknown task: {task!r}\n", file=sys.stderr)
        show_help()
        return 2

    # task-specific args (only `api` uses host/port for now)
    p = argparse.ArgumentParser(prog=f"dev.py {task}")
    if task == "api":
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", default="8765")
    args = p.parse_args(sys.argv[2:])
    return TASKS[task](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
