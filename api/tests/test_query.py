"""Tests for /api/query — the full RAG pipeline, with the OpenAI layer mocked."""
from unittest.mock import MagicMock, patch

from django.test import TestCase
from openai import APIError, AuthenticationError, RateLimitError


def _openai_error(cls, message: str, status: int):
    """Construct an OpenAI exception with a mock response that has .request."""
    mock_response = MagicMock()
    mock_response.status_code = status
    mock_response.headers = {}
    return cls(message, response=mock_response,
               body={"error": {"message": message}})


def _fake_contexts():
    return [
        {
            "text": "Nepal has 77 districts spread across 7 provinces.",
            "source": "Nepal_Parichaya-1",
            "page": "1",
            "heading": "Administrative divisions",
            "similarity": 0.99,
            "match": "dict",
        },
        {
            "text": "Kathmandu is the capital of Nepal.",
            "source": "Nepal_Parichaya-2",
            "page": "1",
            "heading": "Capital",
            "similarity": 0.85,
            "match": "vector",
        },
    ]


def _fake_usage():
    return {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cost": 0.001234,
    }


class QueryHappyPathTests(TestCase):
    @patch("api.views.generate_answer")
    @patch("api.views.retrieve")
    @patch("api.views.normalize_query")
    @patch("api.views._vector_store")
    def test_returns_answer_sources_and_usage(self, mock_vs, mock_norm, mock_ret, mock_gen):
        mock_norm.return_value = "नेपालमा कति जिल्ला छन्?"
        mock_ret.return_value = _fake_contexts()
        mock_gen.return_value = ("Nepal has 77 districts.", _fake_usage())

        r = self.client.post(
            "/api/query",
            {"question": "How many districts are in Nepal?"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["answer"], "Nepal has 77 districts.")
        self.assertEqual(len(body["sources"]), 2)
        self.assertEqual(body["sources"][0]["similarity"], 0.99)
        self.assertIn("text", body["sources"][0])
        self.assertEqual(body["usage"]["total_tokens"], 120)
        self.assertEqual(body["usage"]["cost"], 0.001234)

    @patch("api.views.generate_answer")
    @patch("api.views.retrieve")
    @patch("api.views.normalize_query")
    @patch("api.views._vector_store")
    def test_passes_user_params_through_to_pipeline(
        self, mock_vs, mock_norm, mock_ret, mock_gen,
    ):
        """Custom top_k, temperature, max_tokens, model, system_prompt all forwarded."""
        mock_norm.return_value = "test"
        mock_ret.return_value = []
        mock_gen.return_value = ("ok", _fake_usage())

        self.client.post(
            "/api/query",
            {
                "question": "What is Nepal?",
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 256,
                "model": "gpt-4o-mini",
                "system_prompt": "Be terse.",
            },
            content_type="application/json",
        )

        # retrieve should have been called with our top_k=5
        kwargs = mock_ret.call_args.kwargs
        self.assertEqual(kwargs.get("top_k"), 5)
        # generate_answer should have received the model + temp + tokens + prompt
        gen_kwargs = mock_gen.call_args.kwargs
        self.assertEqual(gen_kwargs.get("model"), "gpt-4o-mini")
        self.assertEqual(gen_kwargs.get("temperature"), 0.7)
        self.assertEqual(gen_kwargs.get("max_tokens"), 256)
        self.assertEqual(gen_kwargs.get("system_prompt"), "Be terse.")


class QueryValidationTests(TestCase):
    def test_empty_body_rejected(self):
        r = self.client.post("/api/query", {}, content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_question_too_short_rejected(self):
        r = self.client.post(
            "/api/query", {"question": "x"}, content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_top_k_out_of_range_rejected(self):
        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?", "top_k": 999},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)


class QueryUpstreamErrorTests(TestCase):
    """OpenAI failures should map to clean client-facing status codes."""

    @patch("api.views.generate_answer")
    @patch("api.views.retrieve")
    @patch("api.views.normalize_query")
    @patch("api.views._vector_store")
    def test_quota_exhausted_returns_429(self, mock_vs, mock_norm, mock_ret, mock_gen):
        mock_norm.return_value = "test"
        mock_ret.return_value = []
        mock_gen.side_effect = _openai_error(
            RateLimitError, "exceeded your current quota", 429,
        )

        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 429)
        self.assertIn("error", r.json())

    @patch("api.views.generate_answer")
    @patch("api.views.retrieve")
    @patch("api.views.normalize_query")
    @patch("api.views._vector_store")
    def test_auth_error_returns_502(self, mock_vs, mock_norm, mock_ret, mock_gen):
        mock_norm.return_value = "test"
        mock_ret.return_value = []
        mock_gen.side_effect = _openai_error(AuthenticationError, "bad key", 401)

        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 502)
