"""Tests for /api/translate — OpenAI translation, mocked."""
from unittest.mock import patch

from django.test import TestCase


class TranslateEndpointTests(TestCase):
    @patch("api.views.translate_to_nepali")
    def test_returns_translation(self, mock_translate):
        mock_translate.return_value = "नेपालमा ७७ जिल्ला छन्।"

        r = self.client.post(
            "/api/translate",
            {"text": "Nepal has 77 districts."},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["translation"], "नेपालमा ७७ जिल्ला छन्।")
        mock_translate.assert_called_once_with("Nepal has 77 districts.")

    def test_empty_text_rejected(self):
        r = self.client.post(
            "/api/translate", {"text": ""}, content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_missing_text_rejected(self):
        r = self.client.post(
            "/api/translate", {}, content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    @patch("api.views.translate_to_nepali")
    def test_upstream_failure_returns_5xx(self, mock_translate):
        from openai import APIConnectionError
        mock_translate.side_effect = APIConnectionError(request=None)

        r = self.client.post(
            "/api/translate",
            {"text": "Hello"},
            content_type="application/json",
        )
        # 502 (bad gateway) per the views' error mapping
        self.assertEqual(r.status_code, 502)
        self.assertIn("error", r.json())
