"""Tests for APIKeyPermission — public-path whitelist + X-API-Key gating."""
import os
from unittest.mock import patch

from django.test import TestCase, override_settings

# Tests can't run collectstatic, so use the plain (non-manifest) storage
# only when rendering pages that pull in static assets (Swagger UI).
_PLAIN_STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}


class PublicPathTests(TestCase):
    """These paths must work without an API key, even when RAG_API_KEY is set."""

    def setUp(self):
        os.environ["RAG_API_KEY"] = "secret"

    def tearDown(self):
        os.environ.pop("RAG_API_KEY", None)

    def test_health_public(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)

    def test_stats_public(self):
        with patch("api.views._vector_store") as mock_vs:
            mock_vs.return_value.count.return_value = 0
            r = self.client.get("/api/stats")
        self.assertEqual(r.status_code, 200)

    def test_schema_public(self):
        r = self.client.get("/api/schema/")
        self.assertEqual(r.status_code, 200)

    @override_settings(STORAGES=_PLAIN_STORAGES)
    def test_docs_public(self):
        r = self.client.get("/api/docs/")
        self.assertEqual(r.status_code, 200)


class APIKeyGatingTests(TestCase):
    """When RAG_API_KEY is configured, non-public endpoints require the header."""

    def setUp(self):
        os.environ["RAG_API_KEY"] = "secret-token"

    def tearDown(self):
        os.environ.pop("RAG_API_KEY", None)

    def test_query_rejected_without_key(self):
        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 403)

    def test_query_rejected_with_wrong_key(self):
        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?"},
            content_type="application/json",
            HTTP_X_API_KEY="wrong",
        )
        self.assertEqual(r.status_code, 403)

    @patch("api.views.generate_answer")
    @patch("api.views.retrieve")
    @patch("api.views.normalize_query")
    @patch("api.views._vector_store")
    def test_query_accepted_with_correct_key(self, mock_vs, mock_norm, mock_ret, mock_gen):
        mock_norm.return_value = "test"
        mock_ret.return_value = []
        mock_gen.return_value = (
            "ok",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
        )
        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?"},
            content_type="application/json",
            HTTP_X_API_KEY="secret-token",
        )
        self.assertEqual(r.status_code, 200)


class OpenModeTests(TestCase):
    """When RAG_API_KEY is unset, all endpoints work without a key (dev mode)."""

    def setUp(self):
        os.environ.pop("RAG_API_KEY", None)

    @patch("api.views.generate_answer")
    @patch("api.views.retrieve")
    @patch("api.views.normalize_query")
    @patch("api.views._vector_store")
    def test_query_accepted_without_key_when_unconfigured(
        self, mock_vs, mock_norm, mock_ret, mock_gen,
    ):
        mock_norm.return_value = "test"
        mock_ret.return_value = []
        mock_gen.return_value = (
            "ok",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
        )
        r = self.client.post(
            "/api/query",
            {"question": "What is Nepal?"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
