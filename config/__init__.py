import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent / "youtube_agent"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from youtube_agent.celery import app, app as celery_app

__all__ = ("app", "celery_app")
