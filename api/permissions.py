"""Simple API-key permission.

Set RAG_API_KEY in the environment to require an `X-API-Key` header on every
request. If RAG_API_KEY is unset (dev mode), the API is open — useful while
developing locally; flip to a real key for any deployed instance.

Always-public paths (no key required): /api/health, /api/schema,
/api/docs, /api/redoc — so the Swagger UI is browsable without a key.
"""
import os

from rest_framework.permissions import BasePermission

_PUBLIC_PATHS = ("/api/health", "/api/stats", "/api/schema", "/api/docs", "/api/redoc")


class APIKeyPermission(BasePermission):
    message = "Invalid or missing X-API-Key header."

    def has_permission(self, request, view):
        path = request.path.rstrip("/")
        if any(path.startswith(p) for p in _PUBLIC_PATHS):
            return True

        required = os.environ.get("RAG_API_KEY")
        if not required:
            # No key configured → open mode (dev only).
            return True

        provided = request.headers.get("X-API-Key", "")
        return provided == required
