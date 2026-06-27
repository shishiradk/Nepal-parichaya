"""REST views — thin Django wrappers around the existing `rag/` package.

The RAG logic itself stays in `rag/`. These views handle:
  - input validation
  - error mapping (OpenAI errors → 502, validation → 400, quota → 429)
  - response shape
  - vector-store warm-up (loaded once per process, not per request)
"""
import logging

from django.conf import settings
from drf_spectacular.utils import OpenApiExample, extend_schema
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
)
from rest_framework.response import Response
from rest_framework.views import APIView

from rag import (
    generate_answer,
    load_vector_store,
    normalize_query,
    retrieve,
)
from rag.config import EMBEDDING_MODEL, LLM_MODEL
from rag.generator import translate_to_nepali

from .serializers import (
    ErrorResponseSerializer,
    HealthResponseSerializer,
    QueryRequestSerializer,
    QueryResponseSerializer,
    StatsResponseSerializer,
    TranslateRequestSerializer,
    TranslateResponseSerializer,
)

log = logging.getLogger(__name__)

# Vector store is expensive to open — cache it for the process lifetime.
_collection = None


def _vector_store():
    global _collection
    if _collection is None:
        log.info("Loading vector store (cold start)…")
        _collection = load_vector_store()
    return _collection


def _openai_error_response(e):
    """Map OpenAI exceptions to HTTP status codes the client can act on."""
    if isinstance(e, AuthenticationError):
        return Response({"error": "Invalid OpenAI API key on server."}, status=502)
    if isinstance(e, RateLimitError):
        msg = str(e).lower()
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            return Response(
                {"error": "Server's OpenAI quota is exhausted."}, status=429
            )
        return Response({"error": "Rate limited; retry later."}, status=429)
    if isinstance(e, (APIConnectionError, APIError)):
        return Response({"error": "Upstream API error."}, status=502)
    log.exception("Unhandled error in RAG pipeline")
    return Response({"error": "Internal server error."}, status=500)


@extend_schema(
    tags=["system"],
    summary="Liveness check",
    description="Fast, dependency-free health probe. Returns 200 if the Django process is up. No upstream calls.",
    responses={200: HealthResponseSerializer},
)
class HealthView(APIView):
    """GET /api/health — fast liveness check, no upstream calls."""

    permission_classes = []  # public

    def get(self, request):
        return Response({"status": "ok", "service": "nepal-parichaya-rag"})


@extend_schema(
    tags=["system"],
    summary="Server stats",
    description="Returns vector-store size and currently-configured models.",
    responses={200: StatsResponseSerializer},
)
class StatsView(APIView):
    """GET /api/stats — knowledge-base size + configured models."""

    permission_classes = []  # public

    def get(self, request):
        try:
            n = _vector_store().count()
        except Exception:
            n = 0
        return Response({
            "documents": n,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
        })


@extend_schema(
    tags=["rag"],
    summary="Ask a question about Nepal Parichaya",
    description=(
        "Runs the full RAG pipeline:\n"
        "1. Normalizes the question (Romanized/English → Devanagari where useful)\n"
        "2. Retrieves the top-K chunks via hybrid vector + keyword search\n"
        "3. Generates a grounded answer with an OpenAI chat model\n\n"
        "Answers in the same language as the question. Refuses (does not invent) "
        "if the answer is not in the source book."
    ),
    request=QueryRequestSerializer,
    responses={
        200: QueryResponseSerializer,
        400: ErrorResponseSerializer,
        429: ErrorResponseSerializer,
        502: ErrorResponseSerializer,
    },
    examples=[
        OpenApiExample(
            "Nepali Devanagari question",
            value={"question": "नेपालमा कति जिल्ला छन्?"},
            request_only=True,
        ),
        OpenApiExample(
            "English question",
            value={"question": "What is the capital of Nepal?", "top_k": 6},
            request_only=True,
        ),
        OpenApiExample(
            "Romanized Nepali question",
            value={"question": "nepal ko rastriya phul k ho?"},
            request_only=True,
        ),
    ],
)
class QueryView(APIView):
    """POST /api/query — main RAG endpoint."""

    def post(self, request):
        s = QueryRequestSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        question = s.validated_data["question"].strip()
        model = s.validated_data.get("model") or None
        top_k = s.validated_data.get("top_k", 8)
        temperature = s.validated_data.get("temperature", 0.2)
        max_tokens = s.validated_data.get("max_tokens", 1024)
        system_prompt = s.validated_data.get("system_prompt") or None

        try:
            normalized = normalize_query(question)
            contexts = retrieve(_vector_store(), normalized,
                                top_k=top_k, original_query=question)
            answer, usage = generate_answer(
                question, contexts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        except Exception as e:
            return _openai_error_response(e)

        return Response({
            "answer": answer,
            "sources": [
                {
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "heading": c.get("heading"),
                    "similarity": c.get("similarity"),
                    "match": c.get("match"),
                    "text": c.get("text", ""),
                }
                for c in contexts
            ],
            "usage": usage,
        })


@extend_schema(
    tags=["utility"],
    summary="Translate text to Nepali Devanagari",
    description="Calls an OpenAI chat model to translate the given text into Nepali Devanagari, preserving numbers and proper names.",
    request=TranslateRequestSerializer,
    responses={
        200: TranslateResponseSerializer,
        400: ErrorResponseSerializer,
        429: ErrorResponseSerializer,
        502: ErrorResponseSerializer,
    },
    examples=[
        OpenApiExample(
            "English to Nepali",
            value={"text": "Nepal has 77 districts and 7 provinces."},
            request_only=True,
        ),
    ],
)
class TranslateView(APIView):
    """POST /api/translate — translate text into Nepali Devanagari."""

    def post(self, request):
        s = TranslateRequestSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        text = s.validated_data["text"].strip()
        try:
            translation = translate_to_nepali(text)
        except Exception as e:
            return _openai_error_response(e)
        return Response({"translation": translation})
