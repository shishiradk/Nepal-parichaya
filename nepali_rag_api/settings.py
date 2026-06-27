"""Django settings for the Nepal Parichaya RAG API.

Minimal stateless REST API in front of the existing `rag/` package.
No database models required — the vector store (ChromaDB) lives outside Django.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-do-not-use-in-prod")
DEBUG = os.environ.get("DJANGO_DEBUG", "0") == "1"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    # auth + contenttypes are required by DRF internally (request.user) even
    # though we use a custom API-key permission. Run `manage.py migrate` once
    # to create the SQLite stub and silence the unapplied-migrations warning.
    "django.contrib.auth",
    "django.contrib.contenttypes",
    # WhiteNoise's runserver shim — disables Django's default static handler so
    # WhiteNoiseMiddleware can serve /static/ uniformly in both DEBUG and prod.
    # MUST come before django.contrib.staticfiles.
    "whitenoise.runserver_nostatic",
    "django.contrib.staticfiles",
    "rest_framework",
    "drf_spectacular",
    "drf_spectacular_sidecar",
    "corsheaders",
    "api",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    # WhiteNoise serves /static/ files from STATIC_ROOT in both dev and prod —
    # no need for runserver --insecure or a separate web server.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "nepali_rag_api.urls"
WSGI_APPLICATION = "nepali_rag_api.wsgi.application"

# Required so drf-spectacular's bundled Swagger UI / ReDoc templates resolve.
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

# Required by drf-spectacular UI views (uses static files for the JS bundle).
# WhiteNoise serves these from STATIC_ROOT after `collectstatic`.
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# Tiny SQLite stub for Django's required tables (auth/contenttypes).
# Application data lives in ChromaDB, not here.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- DRF ---
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],   # API-key handled via custom permission
    "DEFAULT_PERMISSION_CLASSES": ["api.permissions.APIKeyPermission"],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.AnonRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {
        "anon": os.environ.get("API_RATE_LIMIT", "30/min"),
    },
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

# --- OpenAPI / Swagger (drf-spectacular) ---
SPECTACULAR_SETTINGS = {
    "TITLE": "Nepal Parichaya RAG API",
    "DESCRIPTION": (
        "REST API for the Nepal Parichaya Retrieval-Augmented Generation system. "
        "Ask civics questions in Nepali Devanagari, Romanized Nepali, or English; "
        "answers are grounded strictly in the *Nepal Parichaya* book and returned "
        "with their source chunks and token-cost breakdown."
    ),
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,   # don't list the schema endpoint in itself
    "COMPONENT_SPLIT_REQUEST": True,
    "TAGS": [
        {"name": "system", "description": "Health and metadata"},
        {"name": "rag",    "description": "Question answering against Nepal Parichaya"},
        {"name": "utility", "description": "Helper utilities (translation, etc.)"},
    ],
    "CONTACT": {"name": "Nepal Parichaya RAG"},
    "LICENSE": {"name": "CC BY 4.0"},
    # Use the sidecar to serve Swagger UI / ReDoc assets locally (no CDN)
    "SWAGGER_UI_DIST": "SIDECAR",
    "SWAGGER_UI_FAVICON_HREF": "SIDECAR",
    "REDOC_DIST": "SIDECAR",
}

# --- CORS ---
# Allow Streamlit / browser frontends to call the API
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

USE_TZ = True
TIME_ZONE = "UTC"
LANGUAGE_CODE = "en-us"
