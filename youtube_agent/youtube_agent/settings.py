import os
from pathlib import Path
from decouple import config, Csv

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = config(name, default=str(default))
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on", "debug"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", "release", "production"}:
        return False

    return default

SECRET_KEY = config("DJANGO_SECRET_KEY", default=config("SECRET_KEY", default=""))
if not SECRET_KEY:
    raise ValueError(
        "Missing Django secret key. Set DJANGO_SECRET_KEY or SECRET_KEY in the environment."
    )

DEBUG = env_bool("DEBUG", default=True)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="127.0.0.1,localhost", cast=Csv())

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third-party
    "django_celery_beat",
    "django_celery_results",
    # local
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "youtube_agent.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "youtube_agent.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── Celery ────────────────────────────────────────────────────────────────────
REDIS_URL = config("REDIS_URL", default="redis://localhost:6379/0")

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = "django-db"         # stores results in DB via django-celery-results
CELERY_CACHE_BACKEND = "default"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"
CELERY_WORKER_POOL = "solo" if os.name == "nt" else "prefork"
CELERY_WORKER_CONCURRENCY = 1 if CELERY_WORKER_POOL == "solo" else 4
CELERY_WORKER_PREFETCH_MULTIPLIER = 1

# How long to keep task results in the DB (7 days)
CELERY_RESULT_EXPIRES = 60 * 60 * 24 * 7

# ── Logging ───────────────────────────────────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "shorts_bot.log",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "core": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

# ── API Keys (loaded from .env) ───────────────────────────────────────────────
# GEMINI_API_KEY: used for text/script generation with the high-quality model.
# GEMINI_API2: used for Gemini Vision calls when verifying image relevance.
GEMINI_API_KEY = config("GEMINI_API_KEY", default="")
GEMINI_API2 = config("GEMINI_API2", default="")

# Default text model for script generation.
GEMINI_MODEL = config("GEMINI_MODEL", default="gemini-3.5-flash")
# Default vision model for image relevance checks.
GEMINI_VISION_MODEL = config("GEMINI_VISION_MODEL", default="gemini-3.1-flash-lite")

PEXELS_API_KEY = config("PEXELS_API_KEY", default="")
YOUTUBE_CLIENT_SECRETS_FILE = config("YOUTUBE_CLIENT_SECRETS_FILE", default="client_secrets.json")

# ── App settings ──────────────────────────────────────────────────────────────
# Topics to rotate through for automatic daily uploads
VIDEO_TOPICS = config(
    "VIDEO_TOPICS",
    default="The Night Ronaldo Almost Never Played",
    cast=Csv(),
)
DAILY_UPLOAD_HOUR = config("DAILY_UPLOAD_HOUR", default=9, cast=int)   # 09:00 UTC
