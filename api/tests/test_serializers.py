"""Pure unit tests for request validation — no Django/DB/HTTP needed."""
from django.test import SimpleTestCase

from api.serializers import (
    QueryRequestSerializer,
    TranslateRequestSerializer,
)


class QueryRequestSerializerTests(SimpleTestCase):
    def test_question_is_required(self):
        s = QueryRequestSerializer(data={})
        self.assertFalse(s.is_valid())
        self.assertIn("question", s.errors)

    def test_question_below_min_length_rejected(self):
        s = QueryRequestSerializer(data={"question": "a"})
        self.assertFalse(s.is_valid())
        self.assertIn("question", s.errors)

    def test_question_at_min_length_accepted(self):
        s = QueryRequestSerializer(data={"question": "ab"})
        self.assertTrue(s.is_valid(), s.errors)

    def test_question_over_max_length_rejected(self):
        s = QueryRequestSerializer(data={"question": "x" * 2001})
        self.assertFalse(s.is_valid())

    def test_defaults_when_only_question_provided(self):
        s = QueryRequestSerializer(data={"question": "What is Nepal?"})
        self.assertTrue(s.is_valid(), s.errors)
        v = s.validated_data
        self.assertEqual(v["top_k"], 8)
        self.assertEqual(v["temperature"], 0.2)
        self.assertEqual(v["max_tokens"], 1024)
        self.assertEqual(v["model"], "")
        self.assertEqual(v["system_prompt"], "")

    def test_top_k_below_min_rejected(self):
        s = QueryRequestSerializer(data={"question": "What is Nepal?", "top_k": 0})
        self.assertFalse(s.is_valid())

    def test_top_k_above_max_rejected(self):
        s = QueryRequestSerializer(data={"question": "What is Nepal?", "top_k": 100})
        self.assertFalse(s.is_valid())

    def test_temperature_above_max_rejected(self):
        s = QueryRequestSerializer(data={"question": "What is Nepal?", "temperature": 3.0})
        self.assertFalse(s.is_valid())

    def test_temperature_below_min_rejected(self):
        s = QueryRequestSerializer(data={"question": "What is Nepal?", "temperature": -0.1})
        self.assertFalse(s.is_valid())

    def test_max_tokens_below_min_rejected(self):
        s = QueryRequestSerializer(data={"question": "What is Nepal?", "max_tokens": 10})
        self.assertFalse(s.is_valid())

    def test_devanagari_question_accepted(self):
        s = QueryRequestSerializer(data={"question": "नेपालमा कति जिल्ला छन्?"})
        self.assertTrue(s.is_valid(), s.errors)

    def test_custom_model_override(self):
        s = QueryRequestSerializer(data={"question": "test", "model": "gpt-4o-mini"})
        self.assertTrue(s.is_valid(), s.errors)
        self.assertEqual(s.validated_data["model"], "gpt-4o-mini")


class TranslateRequestSerializerTests(SimpleTestCase):
    def test_text_required(self):
        s = TranslateRequestSerializer(data={})
        self.assertFalse(s.is_valid())
        self.assertIn("text", s.errors)

    def test_empty_text_rejected(self):
        s = TranslateRequestSerializer(data={"text": ""})
        self.assertFalse(s.is_valid())

    def test_short_text_accepted(self):
        s = TranslateRequestSerializer(data={"text": "Hello"})
        self.assertTrue(s.is_valid(), s.errors)

    def test_text_over_max_length_rejected(self):
        s = TranslateRequestSerializer(data={"text": "x" * 8001})
        self.assertFalse(s.is_valid())
