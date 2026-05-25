"""Text-to-speech generation for YouTube Shorts voiceovers."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

VOICE_NAME = "en-US-GuyNeural"


async def _save_audio(script: str, output_path: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise Exception("edge-tts is not installed. Run `pip install edge-tts`.") from exc

    communicator = edge_tts.Communicate(text=script, voice=VOICE_NAME)
    await communicator.save(str(output_path))


def generate_audio(script: str, job_id: str) -> str:
    """Generate an MP3 voiceover file and return its full local path."""
    if not script or not script.strip():
        raise Exception("Script cannot be empty for TTS generation.")
    if not job_id or not job_id.strip():
        raise Exception("job_id cannot be empty for TTS generation.")

    job_dir = Path(settings.MEDIA_ROOT) / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "audio.mp3"

    logger.info("[%s] Generating audio with voice %s", job_id[:8], VOICE_NAME)

    try:
        asyncio.run(_save_audio(script.strip(), output_path))
    except Exception as exc:
        logger.error("[%s] Audio generation failed: %s", job_id[:8], exc)
        raise Exception(f"Failed to generate audio with edge-tts: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise Exception("edge-tts did not create a valid audio.mp3 file.")

    logger.info("[%s] Audio saved to %s", job_id[:8], output_path)
    return str(output_path)


def _bootstrap_django() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_agent.settings")

    import django

    django.setup()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    _bootstrap_django()

    parser = argparse.ArgumentParser(description="Generate a voiceover MP3 from text.")
    parser.add_argument("script", help="Script text to convert to speech.")
    parser.add_argument(
        "--job-id",
        default="manual-test",
        help="Job ID used to build the media output path.",
    )
    args = parser.parse_args()

    try:
        audio_file = generate_audio(args.script, args.job_id)
    except Exception:
        logger.exception("Standalone audio generation failed.")
        raise SystemExit(1)

    logger.info("Generated audio file: %s", audio_file)
