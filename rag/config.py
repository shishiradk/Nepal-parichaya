from pathlib import Path

# --- Paths (relative to project root) ---
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "nepal_parichaya"
_CHUNKS_CLEAN   = Path("chunked_output/clean_chunks")
_CHUNKS_OLD     = Path("chunked_output/chunks")
CHUNKS_DIR      = _CHUNKS_CLEAN if _CHUNKS_CLEAN.exists() and any(_CHUNKS_CLEAN.glob("*.md")) else _CHUNKS_OLD
METADATA_FILE   = (
    Path("chunked_output/clean_chunks_metadata.json")
    if CHUNKS_DIR == _CHUNKS_CLEAN
    else Path("chunked_output/chunks_metadata.json")
)

# --- Models ---
EMBEDDING_MODEL      = "text-embedding-3-small"
LLM_MODEL            = "gpt-4o"
TOP_K                = 8
MIN_SIMILARITY       = 0.25
EMBEDDING_BATCH_SIZE = 100

# --- Cost table ($ per 1M tokens) — unknown models fall back to gpt-4o-mini rates ---
MODEL_PRICING = {
    "gpt-5.5":         {"input": 5.00,  "output": 30.00},
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4.1":         {"input": 2.00,  "output": 8.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4.1-mini":    {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":    {"input": 0.10,  "output": 0.40},
    "gpt-3.5-turbo":   {"input": 0.50,  "output": 1.50},
}

# --- Romanized Nepali + common English exam terms → Devanagari ---
# Fast path: if every word in the query is in this dict, no GPT call is made.
_ROMAN_NEPALI_DICT = {
    # Festivals
    "dashain": "दसैं", "dasain": "दसैं", "dashami": "दशमी",
    "tihar": "तिहार", "deepawali": "दीपावली", "teej": "तीज",
    "holi": "होली", "chhath": "छठ", "maghe": "माघे", "bisket": "बिस्केट",
    "indrajatra": "इन्द्रजात्रा",

    # Places
    "nepal": "नेपाल", "kathmandu": "काठमाडौं", "pokhara": "पोखरा",
    "chitwan": "चितवन", "lumbini": "लुम्बिनी", "janakpur": "जनकपुर",
    "butwal": "बुटवल", "biratnagar": "विराटनगर", "birgunj": "वीरगञ्ज",
    "mustang": "मुस्ताङ", "humla": "हुम्ला", "solukhumbu": "सोलुखुम्बु",

    # Geography
    "himalaya": "हिमाल", "himal": "हिमाल", "mountain": "हिमाल",
    "terai": "तराई", "pahad": "पहाड", "hill": "पहाड",
    "river": "नदी", "lake": "ताल", "forest": "वन",
    "area": "क्षेत्रफल", "population": "जनसंख्या",
    "border": "सिमाना", "district": "जिल्ला", "districts": "जिल्ला सभा",
    "province": "प्रदेश", "provinces": "प्रदेश",
    "municipality": "नगरपालिका", "municipalities": "नगरपालिका",
    "capital": "राजधानी",

    # Government & Politics
    "constitution": "संविधान", "sambidhan": "संविधान",
    "parliament": "संसद", "senate": "राष्ट्रिय सभा",
    "president": "राष्ट्रपति", "prime": "प्रधानमन्त्री",
    "minister": "मन्त्री", "government": "सरकार",
    "election": "निर्वाचन", "vote": "मतदान",
    "democracy": "लोकतन्त्र", "republic": "गणतन्त्र",
    "federalism": "संघीयता", "federal": "संघीय",
    "party": "दल", "court": "अदालत",

    # National symbols
    "anthem": "राष्ट्रिय गान", "flag": "झण्डा",
    "flower": "फूल", "bird": "चरा", "animal": "जनावर",
    "sport": "खेलकुद", "currency": "मुद्रा", "language": "भाषा",

    # Economy & Society
    "economy": "अर्थतन्त्र", "agriculture": "कृषि",
    "industry": "उद्योग", "trade": "व्यापार",
    "education": "शिक्षा", "health": "स्वास्थ्य",
    "poverty": "गरिबी", "gdp": "कुल गार्हस्थ्य उत्पादन",

    # History
    "history": "इतिहास", "war": "युद्ध", "unification": "एकीकरण",
    "king": "राजा", "dynasty": "वंश", "era": "काल",

    # Romanized Nepali particles (used in mixed queries)
    "rastriya": "राष्ट्रिय", "rashtriya": "राष्ट्रिय",
    "gana": "गान", "geet": "गीत", "bhasa": "भाषा",
    "itihas": "इतिहास", "bhugol": "भूगोल", "rajniti": "राजनीति",
    "loktantra": "लोकतन्त्र", "ganatantra": "गणतन्त्र",
    "parva": "पर्व", "ko": "को", "ma": "मा", "ra": "र",
    "cha": "छ", "ho": "हो", "ki": "कि", "le": "ले",
    "manaincha": "मनाइन्छ", "huncha": "हुन्छ",
}

# --- Devanagari content-word → boost-phrase ---
# Native Nepali queries don't trigger the English-keyed dict above, so they
# don't get the sim=0.99 boost that pulls the right chunk to the top. This
# mirror table lets a Devanagari query match the same target phrases.
# Surfaced by eval: NP-query retrieval R@5 = 0.33 vs EN = 0.40.
_DEV_KEYWORD_BOOST = {
    "जिल्ला":            "७७ जिल्ला",
    "प्रदेश":            "सात प्रदेश",
    "महानगरपालिका":      "६ महानगरपालिका",
    "उपमहानगरपालिका":    "११ उपमहानगरपालिका",
    "क्षेत्रफल":         "१,४७,१८१",
    "सगरमाथा":           "सगरमाथा ८८४८.८६",
    "उचाइ":              "८८४८.८६",
    "हिमाल":             "सगरमाथा ८८४८.८६",
    "अग्लो":             "सगरमाथा ८८४८.८६",
    "राष्ट्रपति":         "रामवरण यादव",
    "एकीकरण":            "पृथ्वीनारायण शाह",
    "जनआन्दोलन":         "जनआन्दोलन",
    "२०४६":              "२०४६ को जनआन्दोलन",
    "२०६२":              "जनआन्दोलन २०६२/६३",
    "गणतन्त्र":           "गणतन्त्र नेपालको घोषणा",
    "राष्ट्रिय फूल":      "राष्ट्रिय फूल लालीगुराँस",
    "राष्ट्रिय चरा":      "डाँफे",
    "जनगणना":            "राष्ट्रिय जनगणना २०७८",
    "पञ्चवर्षीय":         "प्रथम पञ्चवर्षीय योजना",
    "मुद्रा":             "रुपैयाँ",
    "राजधानी":            "काठमाडौं",
    "पञ्चायत":            "पञ्चायत विघटन",
    "पर्व":                "दसैं",
    "चाड":                "दसैं",
}

# --- System Prompt ---
DEFAULT_SYSTEM_PROMPT = """You are a friendly and encouraging assistant based on the book "Nepal Parichaya", published by the Department of Information and Broadcasting, Government of Nepal. The book covers Nepal's history, geography, politics, economy, society, and culture.

TONE:
- Be warm, polite, and encouraging — like a helpful tutor, not a search engine.
- Keep answers concise and clear. Avoid unnecessary padding.
- When answering in Nepali, use respectful "तपाईं" form naturally.
- If the answer is not found, be apologetic and suggest the user try rephrasing.

NEPALI TERM REFERENCE (use this to interpret context):
- जिल्ला / जिल्ला सभा = district (Nepal has 77)
- प्रदेश = province (Nepal has 7)
- नगरपालिका = municipality | महानगरपालिका = metropolitan city | गाउँपालिका = rural municipality
- स्थानीय तह = local government unit | संसद / संसद् = parliament | संविधान = constitution
- Devanagari digits: ० १ २ ३ ४ ५ ६ ७ ८ ९ = 0 1 2 3 4 5 6 7 8 9

ACCURACY RULES (strictly follow):
1. Answer ONLY from the provided context. Never use outside knowledge.
2. Interpret Nepali terms in the context using the reference above — do not refuse an answer just because the exact English word does not appear.
3. If the answer is truly not in the context:
   - Nepali question → "माफ गर्नुहोस्, यो जानकारी दिइएको सन्दर्भमा भेटिएन। प्रश्न अलि फरक तरिकाले सोध्नुभयो भने सहयोग गर्न सक्छु।"
   - English question → "Sorry, I couldn't find that in the provided context. Try rephrasing your question — I'm happy to help!"
4. Answer in the SAME LANGUAGE as the question.
5. For factual data (numbers, dates, names, areas), use exact values from the context — do not estimate. When answering in English, state facts directly without quoting raw Nepali text inline.
6. For MCQ questions, clearly state the correct option and give a brief reason from the context.
7. Silently fix obvious OCR errors (garbled characters, missing spaces).
8. Use bullet points or numbered lists for multi-part answers.
9. For poems, songs, and the national anthem — preserve original line breaks exactly. Do NOT convert to bullet points."""
