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


def _preprocess_script(script: str) -> str:
    """
    Prepare script for edge-tts:
    1. Convert ALL CAPS words to Title Case so TTS pronounces
       them as words, not individual letters.
    2. Replace em dashes with commas so TTS pauses naturally
       instead of saying 'dash'.
    """
    import re

    # Step 1 — replace em dash with comma pause
    script = script.replace("—", ", ")

    # Step 2 — find ALL CAPS words and convert to Title Case
    # Matches sequences of 2+ uppercase letters (avoids single letter like "I")
    def title_case_match(match):
        return match.group(0).title()

    script = re.sub(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', title_case_match, script)

    # Step 3 — clean up double spaces and double commas
    script = re.sub(r',\s*,', ',', script)
    script = re.sub(r'\s+', ' ', script).strip()

    return script



def _trim_silence(audio_path: Path, silence_thresh_db: float = -50.0) -> None:
    """
    Remove trailing silence from the generated MP3.
    edge-tts often adds 0.5-1s of silence at the end.
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence

        audio    = AudioSegment.from_mp3(str(audio_path))
        reversed_audio = audio.reverse()

        # Detect how much silence is at the end (reversed = leading)
        trailing_silence_ms = detect_leading_silence(
            reversed_audio,
            silence_threshold=silence_thresh_db,
            chunk_size=10,
        )

        if trailing_silence_ms > 100:   # only trim if more than 100ms
            trimmed = audio[:-trailing_silence_ms]
            trimmed.export(str(audio_path), format="mp3")
            logger.info(
                "Trimmed %.2fs of trailing silence from audio",
                trailing_silence_ms / 1000
            )
        else:
            logger.info("No significant trailing silence detected — keeping as is")

    except Exception as exc:
        logger.warning("Could not trim silence: %s — keeping original audio", exc)


def generate_audio(script: str, job_id: str) -> str:
    if not script or not script.strip():
        raise Exception("Script cannot be empty for TTS generation.")
    if not job_id or not job_id.strip():
        raise Exception("job_id cannot be empty for TTS generation.")

    job_dir = Path(settings.MEDIA_ROOT) / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "audio.mp3"

    logger.info("[%s] Generating audio with voice %s", job_id[:8], VOICE_NAME)

    clean_script = _preprocess_script(script.strip())

    try:
        asyncio.run(_save_audio(clean_script, output_path))
    except Exception as exc:
        raise Exception(f"Failed to generate audio with edge-tts: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise Exception("edge-tts did not create a valid audio.mp3 file.")

    # Trim trailing silence added by edge-tts
    _trim_silence(output_path)

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
