"""Thin HTTP client for the Nepal Parichaya RAG REST API.

Used by streamlit_app.py so the UI talks to the API instead of importing
`rag/` directly. Lets the UI and the backend deploy independently.

Environment variables:
    RAG_API_URL   — base URL of the API.   Default: http://127.0.0.1:8765
    RAG_API_KEY   — optional X-API-Key sent on every request.

All functions raise APIError on non-2xx responses, with a human-readable message
already mapped from the server's error mapping.
"""
from __future__ import annotations

import os
from typing import Any

import requests


class APIError(RuntimeError):
    """Wraps any 4xx/5xx response or transport error from the API."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _base_url() -> str:
    return os.environ.get("RAG_API_URL", "http://127.0.0.1:8765").rstrip("/")


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    key = os.environ.get("RAG_API_KEY", "")
    if key:
        h["X-API-Key"] = key
    return h


def _request(method: str, path: str, *, json: dict | None = None,
             timeout: float = 60.0) -> dict[str, Any]:
    url = f"{_base_url()}{path}"
    try:
        r = requests.request(method, url, json=json, headers=_headers(), timeout=timeout)
    except requests.RequestException as e:
        raise APIError(f"Could not reach RAG API at {url}: {e}") from e

    if r.status_code // 100 != 2:
        # Try to surface the server's error message
        try:
            payload = r.json()
            msg = payload.get("error") or payload.get("detail") or r.text
        except Exception:
            msg = r.text or f"HTTP {r.status_code}"
        raise APIError(msg, status=r.status_code)

    try:
        return r.json()
    except ValueError as e:
        raise APIError(f"API returned non-JSON response: {e}") from e


# ── Public API ────────────────────────────────────────────────────────────

def health() -> bool:
    """Returns True if /api/health responds 200."""
    try:
        _request("GET", "/api/health", timeout=5)
        return True
    except APIError:
        return False


def stats() -> dict:
    """Knowledge-base size + configured models. Returns the parsed JSON dict."""
    return _request("GET", "/api/stats", timeout=10)


def query(question: str, *, model: str = "", top_k: int = 8,
          temperature: float = 0.2, max_tokens: int = 1024,
          system_prompt: str = "") -> dict:
    """Send a question, return the full response dict {answer, sources, usage}."""
    payload = {
        "question": question,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model
    if system_prompt:
        payload["system_prompt"] = system_prompt
    # generation + 2-3 LLM calls can take up to ~20s; give plenty of headroom
    return _request("POST", "/api/query", json=payload, timeout=120)


def translate(text: str) -> str:
    """Translate text to Nepali Devanagari. Returns just the translation string."""
    payload = {"text": text}
    return _request("POST", "/api/translate", json=payload, timeout=60)["translation"]
