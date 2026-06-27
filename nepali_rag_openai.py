"""Nepal Parichaya RAG — CLI entry point

Usage:
    python nepali_rag_openai.py                  # interactive mode
    python nepali_rag_openai.py --build           # rebuild vector store from chunks
    python nepali_rag_openai.py -q "your query"  # single query
"""
import sys
import os
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from rag import normalize_query, retrieve, generate_answer, load_vector_store, build_vector_store, load_chunks, TOP_K


def query_rag(collection, query):
    normalized = normalize_query(query)
    print(f"\nQuery: {query}")
    if normalized != query:
        print(f"Normalized: {normalized}")
    print("-" * 60)

    contexts = retrieve(collection, normalized, original_query=query)

    print("\nGenerating answer...\n")
    answer, _ = generate_answer(query, contexts)  # original query → answer in same language
    print(answer)
    print("-" * 60)
    return answer


def interactive_mode(collection):
    print("\n=== Nepali RAG - Interactive Mode ===")
    print("Type your question in Nepali or English. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        query_rag(collection, query)
        print()


def main():
    parser = argparse.ArgumentParser(description="Nepali RAG with OpenAI + ChromaDB")
    parser.add_argument("--build", action="store_true", help="Build/rebuild the vector store from chunks")
    parser.add_argument("--query", "-q", type=str, help="Single query mode")
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Use: set OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if args.build:
        chunks = load_chunks()
        build_vector_store(chunks)
        print("\nVector store ready. Query with: python nepali_rag_openai.py -q \"...\"")
        return

    collection = load_vector_store()
    if args.query:
        query_rag(collection, args.query)
    else:
        interactive_mode(collection)


if __name__ == "__main__":
    main()
