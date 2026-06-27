import re
from openai import OpenAI
from .config import EMBEDDING_MODEL, MIN_SIMILARITY, TOP_K, _ROMAN_NEPALI_DICT, _DEV_KEYWORD_BOOST

# Common Nepali words that appear in almost every chunk — skip as keywords
_COMMON = {
    "नेपालको", "नेपालमा", "नेपाल", "गरिन्छ", "हुन्छ", "गर्छ",
    "भएको", "भयो", "छन्", "थियो", "गरे", "भने", "हुने", "गर्न",
}


def _dict_keywords(original_query: str) -> list[str]:
    """Extract guaranteed Devanagari keywords from English words via the dict.
    Skips common words that appear in almost every chunk (same filter as from_query).
    """
    terms = []
    for word in re.findall(r'[a-zA-Z]+', original_query.lower()):
        mapped = _ROMAN_NEPALI_DICT.get(word)
        if mapped and re.search(r'[ऀ-ॿ]', mapped) and len(mapped) >= 4 and mapped not in _COMMON:
            terms.append(mapped)
    return list(dict.fromkeys(terms))[:3]  # deduplicated, max 3


def _dev_dict_keywords(original_query: str) -> list[str]:
    """Mirror of _dict_keywords for native Devanagari queries.
    When a Nepali query contains a known content-word, return the specific
    target phrase from _DEV_KEYWORD_BOOST so it gets the same sim=0.99 boost
    the English path enjoys. Closes the NP-query retrieval gap surfaced by eval.
    """
    terms = []
    # Longest keys first so multi-word triggers ("राष्ट्रिय फूल") beat single-word.
    for key in sorted(_DEV_KEYWORD_BOOST.keys(), key=len, reverse=True):
        if key in original_query:
            terms.append(_DEV_KEYWORD_BOOST[key])
    return list(dict.fromkeys(terms))[:3]


def _kw_search(collection, query_embedding, term, n_results, sim_floor, seen_ids, out, match_type):
    try:
        kw = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where_document={"$contains": term},
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(kw["documents"][0], kw["metadatas"][0], kw["distances"][0]):
            key = doc[:100]
            if key not in seen_ids:
                sim = round(1 - dist, 4)
                if sim < sim_floor:
                    continue
                seen_ids.add(key)
                out.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "?"),
                    "heading": meta.get("heading", ""),
                    "similarity": sim,
                    "match": match_type,
                })
    except Exception:
        pass


def retrieve(collection, query, top_k=None, api_key=None, original_query: str = ""):
    """Hybrid retrieval: vector similarity + keyword matching."""
    if top_k is None:
        top_k = TOP_K

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_embedding = response.data[0].embedding

    # 1. Vector similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    seen_ids = set()
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = round(1 - dist, 4)
        if sim < MIN_SIMILARITY:
            continue
        seen_ids.add(doc[:100])
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "heading": meta.get("heading", ""),
            "similarity": sim,
            "match": "vector",
        })

    # 2a. Dict-extracted keywords — processed FIRST, no similarity floor (human-curated = guaranteed relevant)
    # English/Romanized keys → Devanagari target phrases AND Devanagari keys → target phrases.
    # Scan both the original query and the (Devanagari-normalized) query so Romanized
    # inputs catch Devanagari keys after normalization.
    from_dict = (_dict_keywords(original_query)
                 + _dev_dict_keywords(original_query)
                 + _dev_dict_keywords(query))
    from_dict = list(dict.fromkeys(from_dict))  # dedupe across both paths
    dict_hits = []
    for term in from_dict:
        _kw_search(collection, query_embedding, term, max(top_k * 6, 50), 0.0, seen_ids, dict_hits, "dict")

    # 2b. Normalized-query keywords — similarity-gated (0.10 floor)
    nepali_words = re.findall(r'[ऀ-ॿ]+', query)
    from_query = sorted(
        [w for w in nepali_words if len(w) >= 4 and w not in _COMMON],
        key=len, reverse=True,
    )[:2]
    query_hits = []
    for term in from_query:
        _kw_search(collection, query_embedding, term, max(top_k * 6, 50), 0.10, seen_ids, query_hits, "keyword")

    # 3. Merge: top_k best overall + all dict hits (forced) + query extras above MIN_SIMILARITY
    all_hits = retrieved + dict_hits + query_hits
    all_hits.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = all_hits[:top_k]

    top_ids = {c["text"][:100] for c in top_results}
    # Dict hits not already in top_results — boost similarity so they sort to the top
    forced_extras = [
        dict(c, similarity=0.99) for c in dict_hits if c["text"][:100] not in top_ids
    ]
    # Regular keyword extras above MIN_SIMILARITY, capped
    regular_extras = [
        c for c in query_hits
        if c["text"][:100] not in top_ids and c["similarity"] >= MIN_SIMILARITY
    ][:top_k]

    return sorted(top_results + forced_extras + regular_extras, key=lambda x: x["similarity"], reverse=True)
