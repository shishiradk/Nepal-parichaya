import re
from openai import OpenAI
from .config import LLM_MODEL, DEFAULT_SYSTEM_PROMPT, MODEL_PRICING

_HAS_DEVANAGARI = re.compile(r'[ऀ-ॿ]')


def _uses_completion_tokens(model: str) -> bool:
    """Reasoning-style models use max_completion_tokens instead of max_tokens."""
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


def translate_to_nepali(text: str, api_key: str = None) -> str:
    """Translate an answer to Nepali Devanagari."""
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content":
             "Translate the following text to Nepali Devanagari. "
             "Preserve numbers, proper names, and formatting exactly. "
             "Output only the translated text — no explanation."},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def generate_answer(query, contexts, model=None, temperature=0.2, max_tokens=1024, system_prompt=None, api_key=None):
    """Generate answer using GPT with retrieved context.

    Returns (answer: str, usage: dict) where usage has keys:
        prompt_tokens, completion_tokens, total_tokens, cost (USD)
    """
    if model is None:
        model = LLM_MODEL
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    context_str = ""
    for i, ctx in enumerate(contexts, 1):
        context_str += (
            f"\n--- Context {i} "
            f"(Source: {ctx['source']}, Page: {ctx['page']}, Similarity: {ctx['similarity']}) ---\n"
            + ctx["text"] + "\n"
        )

    # If the question is in English, remind GPT explicitly — context is all Nepali
    # and without this hint GPT tends to answer in Nepali regardless.
    # Exception: verbatim content (anthem, poems) must stay in the original Nepali.
    lang_note = (
        "\n\nIMPORTANT: The question is in English. Answer in English. "
        "Exception: if the answer is verbatim text such as the national anthem, a poem, "
        "or a song, provide the original Nepali text as-is with a brief English intro."
        if not _HAS_DEVANAGARI.search(query) else ""
    )

    user_prompt = f"""Context from Nepal Parichaya:
{context_str}
Question: {query}

Answer based on the above context:{lang_note}"""

    token_kwarg = (
        {"max_completion_tokens": max_tokens}
        if _uses_completion_tokens(model)
        else {"max_tokens": max_tokens}
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        **token_kwarg,
    )

    answer = response.choices[0].message.content
    usage = response.usage
    cost = 0.0
    if usage:
        rates = MODEL_PRICING.get(model, {"input": 0.15, "output": 0.60})
        cost = (usage.prompt_tokens / 1_000_000) * rates["input"] + \
               (usage.completion_tokens / 1_000_000) * rates["output"]

    return answer, {
        "prompt_tokens":      usage.prompt_tokens      if usage else 0,
        "completion_tokens":  usage.completion_tokens  if usage else 0,
        "total_tokens":       usage.total_tokens       if usage else 0,
        "cost":               cost,
    }
