# test_visuals.py
"""
Standalone test runner for the visuals pipeline.
Run from the project root:
  python test_visuals.py "Leo Messi FC Barcelona" "Cristiano Ronaldo Real Madrid" --job-id manual-test --topic "Messi vs Ronaldo"
"""

import argparse
import logging
import os
import sys
from pathlib import Path




# ── Bootstrap Django before any app imports ───────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_agent.settings")

import django
django.setup()

# ── Now safe to import app modules ────────────────────────────────────────────
from core.pipeline.visuals        import fetch_visuals
from core.pipeline.image_quality  import passes_quality_check
from core.pipeline.image_relevance import passes_relevance_check

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
        )
    ]
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the visuals pipeline.")
    parser.add_argument("keywords", nargs="+", help="Keywords to search for.")
    parser.add_argument("--job-id", default="manual-test")
    parser.add_argument("--topic", default="football", help="Topic for relevance check.")
    args = parser.parse_args()

    logger.info("Testing visuals pipeline with %s keywords", len(args.keywords))
    logger.info("Topic: %s", args.topic)
    logger.info("Job ID: %s", args.job_id)

    scenes = [
        {"keyword": k, "backups": [], "duration": 5}
        for k in args.keywords
    ]

    try:
        paths = fetch_visuals(
            scenes=scenes,
            job_id=args.job_id,
            topic=args.topic,
        )
        logger.info("── RESULTS ─────────────────────────────────")
        logger.info("Total images accepted: %s", len(paths))
        for p in paths:
            logger.info("  %s", p)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise