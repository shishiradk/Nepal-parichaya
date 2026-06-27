from rest_framework import serializers


# ── Request shapes ─────────────────────────────────────────────────────────
class QueryRequestSerializer(serializers.Serializer):
    question = serializers.CharField(
        min_length=2, max_length=2000,
        help_text="The user's question, in Nepali Devanagari, Romanized Nepali, or English.",
    )
    model = serializers.CharField(
        required=False, allow_blank=True, default="",
        help_text="OpenAI chat model id. Defaults to the server's configured model (gpt-4o).",
    )
    top_k = serializers.IntegerField(
        required=False, min_value=1, max_value=20, default=8,
        help_text="How many context chunks to retrieve before generation.",
    )
    temperature = serializers.FloatField(
        required=False, min_value=0.0, max_value=2.0, default=0.2,
        help_text="Sampling temperature. Lower = more deterministic.",
    )
    max_tokens = serializers.IntegerField(
        required=False, min_value=64, max_value=4096, default=1024,
        help_text="Maximum tokens to generate in the answer.",
    )
    system_prompt = serializers.CharField(
        required=False, allow_blank=True, default="",
        help_text="Override the default system prompt (leave blank to use the server default).",
    )


class TranslateRequestSerializer(serializers.Serializer):
    text = serializers.CharField(
        min_length=1, max_length=8000,
        help_text="Text to translate into Nepali Devanagari.",
    )


# ── Response shapes (for Swagger / OpenAPI docs only) ──────────────────────
class HealthResponseSerializer(serializers.Serializer):
    status = serializers.CharField(help_text="`ok` if the service is up.")
    service = serializers.CharField(help_text="Service identifier.")


class SourceSerializer(serializers.Serializer):
    source = serializers.CharField(help_text="Source chunk identifier.")
    page = serializers.CharField(help_text="Page number from the source book.")
    heading = serializers.CharField(help_text="Section heading of the chunk.", allow_blank=True)
    similarity = serializers.FloatField(help_text="Retrieval similarity score (0-1).")
    match = serializers.CharField(help_text="Retrieval path that surfaced this chunk: `vector` | `dict` | `keyword`.")
    text = serializers.CharField(help_text="The chunk text (may be long).")


class StatsResponseSerializer(serializers.Serializer):
    documents = serializers.IntegerField(help_text="Number of chunks in the vector store.")
    embedding_model = serializers.CharField()
    llm_model = serializers.CharField()


class UsageSerializer(serializers.Serializer):
    prompt_tokens = serializers.IntegerField()
    completion_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    cost = serializers.FloatField(help_text="Estimated USD cost of this single query.")


class QueryResponseSerializer(serializers.Serializer):
    answer = serializers.CharField(help_text="Generated answer, grounded in the retrieved sources.")
    sources = SourceSerializer(many=True, help_text="Retrieved chunks ranked by similarity (descending).")
    usage = UsageSerializer()


class TranslateResponseSerializer(serializers.Serializer):
    translation = serializers.CharField(help_text="Devanagari translation of the input text.")


class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField(help_text="Human-readable error message.")
