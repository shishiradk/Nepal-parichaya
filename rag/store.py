import sys
import json
from tqdm import tqdm
import chromadb
from openai import OpenAI
from .config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    CHUNKS_DIR, METADATA_FILE,
)


def load_chunks():
    """Load all chunk .md files and their metadata from chunked_output/."""
    metadata = []
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata_map = {m["chunk_id"]: m for m in metadata}

    chunks = []
    chunk_files = sorted(CHUNKS_DIR.glob("*.md"))
    print(f"Found {len(chunk_files)} chunk files in {CHUNKS_DIR}")

    for filepath in chunk_files:
        text = filepath.read_text(encoding="utf-8")
        content = text
        meta = {}
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                content = parts[2].strip()
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        meta[key.strip()] = val.strip()

        chunk_id   = meta.get("chunk_id", filepath.stem)
        source     = meta.get("source_file", "unknown")
        page       = meta.get("page_number", "0")
        heading    = meta.get("heading", "")

        if chunk_id in metadata_map:
            m = metadata_map[chunk_id]
            source = m.get("filename", source)
            page   = str(m.get("page_number", page))

        if content.strip():
            chunks.append({"id": chunk_id, "text": content, "source": source, "page": page, "heading": heading})

    print(f"Loaded {len(chunks)} chunks with content")
    return chunks


def build_vector_store(chunks):
    """Embed chunks via OpenAI and persist in ChromaDB."""
    client = OpenAI()
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection, rebuilding...")
    except Exception:
        pass

    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    print(f"Embedding {len(chunks)} chunks in {total} batches...")

    for i in tqdm(range(0, len(chunks), EMBEDDING_BATCH_SIZE), desc="Embedding"):
        batch     = chunks[i: i + EMBEDDING_BATCH_SIZE]
        texts     = [c["text"][:6000] for c in batch]  # hard cap — embedding model limit
        ids       = [c["id"]   for c in batch]
        metas     = [{"source": c["source"], "page": c["page"], "heading": c.get("heading", "")} for c in batch]
        response  = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        embeddings = [item.embedding for item in response.data]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)

    print(f"Vector store built: {collection.count()} documents in '{CHROMA_DIR}'")
    return collection


def load_vector_store():
    """Load existing ChromaDB collection."""
    if not CHROMA_DIR.exists():
        print("Vector store not found. Run with --build first.")
        sys.exit(1)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_collection(name=COLLECTION_NAME)
    print(f"Loaded vector store with {collection.count()} documents")
    return collection
