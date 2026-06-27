import re
from openai import OpenAI
from .config import LLM_MODEL, _ROMAN_NEPALI_DICT


def normalize_query(query: str, api_key: str = None) -> str:
    """Convert any Latin-script query to Nepali Devanagari for better retrieval.

    - Devanagari: returned unchanged
    - All-dictionary Romanized Nepali (fast path): direct substitution
    - Everything else: GPT translates Romanized Nepali AND English Nepal queries to Devanagari
    """
    if re.search(r'[ऀ-ॿ]', query):
        return query

    words = query.lower().split()
    if all(w in _ROMAN_NEPALI_DICT for w in words if w.isalpha()):
        translated = " ".join(_ROMAN_NEPALI_DICT.get(w, w) for w in words)
        if translated != query.lower():
            return translated

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    result = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content":
             "You normalize search queries for Nepal Parichaya, a Nepali-language civics book. "
             "The search index is ENTIRELY in Nepali Devanagari declarative text (not questions), "
             "so always output a SHORT KEYWORD PHRASE in Devanagari — never a full question sentence.\n\n"
             "Rules:\n"
             "1. Romanized Nepali → Devanagari keyword phrase:\n"
             "   'dashain' → 'दसैं'\n"
             "   'nepal ko rastriya gana' → 'राष्ट्रिय गान'\n"
             "2. English question about Nepal → extract the core topic as a Devanagari keyword phrase:\n"
             "   'how many districts are in nepal?' → 'नेपालको जिल्ला संख्या'\n"
             "   'national anthem of nepal' → 'राष्ट्रिय गान'\n"
             "   'area of nepal' → 'नेपालको क्षेत्रफल'\n"
             "   'capital of nepal' → 'नेपालको राजधानी'\n"
             "   'population of nepal' → 'नेपालको जनसंख्या'\n"
             "   'how many provinces' → 'नेपालको प्रदेश संख्या'\n"
             "   'who is the president' → 'नेपालको राष्ट्रपति'\n"
             "   'national flower of nepal' → 'राष्ट्रिय फूल'\n"
             "   'dashain festival' → 'दसैं पर्व'\n"
             "3. English not about Nepal → return unchanged.\n\n"
             "Output ONLY the keyword phrase — no explanation, no punctuation, no question marks."},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=60,
    )
    return result.choices[0].message.content.strip()
