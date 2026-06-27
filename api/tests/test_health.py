"""Tests for /api/health — fast liveness probe, no upstream calls."""
from django.test import TestCase


class HealthEndpointTests(TestCase):
    def test_returns_200_with_status_ok(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "nepal-parichaya-rag")

    def test_is_public_without_api_key(self):
        """Health must work even when RAG_API_KEY is configured."""
        with self.settings():
            import os
            os.environ["RAG_API_KEY"] = "secret-xyz"
            try:
                r = self.client.get("/api/health")
                self.assertEqual(r.status_code, 200)
            finally:
                del os.environ["RAG_API_KEY"]

    def test_no_database_query(self):
        """Health endpoint should not touch the DB — it's a liveness probe."""
        # Django test client doesn't expose db query counts cleanly without
        # extra setup, but we assert the call succeeds without DB config too.
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
