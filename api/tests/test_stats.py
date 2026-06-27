"""Tests for /api/stats — knowledge-base size + configured models."""
from unittest.mock import MagicMock, patch

from django.test import TestCase


class StatsEndpointTests(TestCase):
    @patch("api.views._vector_store")
    def test_returns_documents_and_models(self, mock_vs):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 951
        mock_vs.return_value = mock_collection

        r = self.client.get("/api/stats")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["documents"], 951)
        self.assertTrue(body["embedding_model"])
        self.assertTrue(body["llm_model"])

    @patch("api.views._vector_store")
    def test_handles_vector_store_failure_gracefully(self, mock_vs):
        """If the vector store can't load, return 0 docs rather than 500."""
        mock_vs.side_effect = RuntimeError("chroma not built")

        r = self.client.get("/api/stats")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["documents"], 0)

    def test_is_public(self):
        """Stats should be reachable without an API key."""
        with patch("api.views._vector_store") as mock_vs:
            mock_vs.return_value.count.return_value = 0
            r = self.client.get("/api/stats")
            self.assertEqual(r.status_code, 200)
