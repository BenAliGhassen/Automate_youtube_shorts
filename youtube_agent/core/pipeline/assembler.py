"""Video assembler — stub for Phase 3."""
import os
import logging
from pathlib import Path
from django.conf import settings
from PIL import Image
import numpy as np
from moviepy import ImageClip, VideoClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip

logger = logging.getLogger(__name__)

# Constants — YouTube Shorts spec
WIDTH  = 1080
HEIGHT = 1920
FPS    = 30
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, "arial.ttf") 

def _prepare_image(image_path: str) -> np.ndarray:
    """
    Open an image, resize and center-crop it to 1080x1920,
    return as a numpy array (what MoviePy works with internally).
    """
    img = Image.open(image_path).convert("RGB")

    # Step 1 — figure out which dimension to scale to fill the frame
    img_ratio    = img.width / img.height
    target_ratio = WIDTH / HEIGHT  # 1080/1920 = 0.5625

    if img_ratio > target_ratio:
        # image is wider than target → scale by height, crop width
        new_height = HEIGHT
        new_width  = int(HEIGHT * img_ratio)
    else:
        # image is taller than target → scale by width, crop height
        new_width  = WIDTH
        new_height = int(WIDTH / img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Step 2 — center crop to exactly 1080x1920
    left   = (new_width  - WIDTH)  // 2
    top    = (new_height - HEIGHT) // 2
    right  = left + WIDTH
    bottom = top  + HEIGHT
    img    = img.crop((left, top, right, bottom))

    return np.array(img)



def _ken_burns_clip(arr: np.ndarray, duration: float) -> ImageClip:
    """
    Take a 1080x1920 numpy image array and return a MoviePy clip
    that slowly zooms from 1.0x to 1.05x over the full duration.
    """
    h, w = arr.shape[:2]   # h=1920, w=1080
    
    # How much extra pixel space the zoom needs at maximum scale
    # at 1.05x zoom on a 1080px wide image:
    # rendered width = 1080 * 1.05 = 1134px
    # so we have 54px extra horizontally, 27px on each side to crop
    zoom_max = 1.05

    def make_frame(t):
        # t goes from 0.0 to duration
        # progress goes from 0.0 to 1.0
        progress = t / duration

        # scale goes from 1.0 (start) to 1.05 (end)
        scale = 1.0 + (zoom_max - 1.0) * progress

        # new dimensions after scaling
        new_w = int(w * scale)
        new_h = int(h * scale)

        # resize the image to the new scaled size
        pil_img = Image.fromarray(arr)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        scaled_arr = np.array(pil_img)

        # center crop back to original 1080x1920
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        cropped = scaled_arr[y1:y1 + h, x1:x1 + w]

        return cropped

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _split_into_chunks(text: str, chunk_duration: float, total_duration: float) -> list[dict]:
    """
    Split a script into timed chunks — one chunk every chunk_duration seconds.
    Returns a list of dicts: [{text, start, end}, ...]
    """
    # Split script into words
    words = text.split()
    total_words = len(words)

    # How many chunks fit in the total duration
    n_chunks = max(1, int(total_duration / chunk_duration))

    # Distribute words evenly across chunks
    words_per_chunk = max(1, total_words // n_chunks)

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_duration
        end   = min((i + 1) * chunk_duration, total_duration)

        word_start = i * words_per_chunk
        word_end   = word_start + words_per_chunk if i < n_chunks - 1 else total_words
        chunk_text = " ".join(words[word_start:word_end])

        if chunk_text.strip():
            chunks.append({"text": chunk_text, "start": start, "end": end})

    return chunks



def _build_slideshow(image_paths: list[str], duration: float) -> object:
    n = len(image_paths)
    segment_duration = duration / n

    logger.info("Building slideshow: %s images, %.2fs each", n, segment_duration)

    clips = []
    for i, path in enumerate(image_paths):
        arr = _prepare_image(path)
        clip = _ken_burns_clip(arr, segment_duration)   # ← only change
        clips.append(clip)
        logger.info("  Prepared image %s/%s: %s", i + 1, n, path)

    return concatenate_videoclips(clips, method="compose")



def _add_overlays(video_clip, title: str, script: str, duration: float) -> CompositeVideoClip:
    """
    Composite title and subtitle text over the video clip.
    """
    layers = [video_clip]

    # ── Title overlay (top, first 3 seconds) ─────────────────────────────────
    try:
        title_clip = (
            TextClip(
                text=title,                  # ← was txt
                font_size=60,               # ← was fontsize
                color="white",
                font=FONT_PATH,               # ← removed Arial-Bold, use font param only
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(WIDTH - 80, None),
                text_align="center",        # ← was align
            )
            .with_duration(3)
            .with_position(("center", 120))
        )
        layers.append(title_clip)
        logger.info("Title overlay created")
    except Exception as exc:
        logger.warning("Could not create title overlay: %s", exc)

    # ── Subtitle overlays (bottom bar, updates every 5 seconds) ──────────────
    chunks = _split_into_chunks(script, chunk_duration=5.0, total_duration=duration)
    logger.info("Creating %s subtitle chunks", len(chunks))

    for chunk in chunks:
        try:
            subtitle_clip = (
                TextClip(
                    text=chunk["text"],      # ← was txt
                    font_size=42,           # ← was fontsize
                    color="white",
                    font=FONT_PATH,               # ← removed Arial-Bold, use font param only
                    stroke_color="black",
                    stroke_width=1,
                    method="caption",
                    size=(WIDTH - 60, None),
                    text_align="center",    # ← was align
                )
                .with_start(chunk["start"])
                .with_duration(chunk["end"] - chunk["start"])
                .with_position(("center", HEIGHT - 300))
            )
            layers.append(subtitle_clip)
        except Exception as exc:
            logger.warning(
                "Could not create subtitle chunk '%s': %s",
                chunk["text"][:30], exc
            )

    return CompositeVideoClip(layers, size=(WIDTH, HEIGHT))




def build_video(job) -> str:
    job_dir    = Path(settings.MEDIA_ROOT) / "jobs" / str(job.id)
    output_path = job_dir / "final.mp4"

    # 1. Collect image paths in order
    visuals_dir  = job_dir / "visuals"
    image_paths  = sorted(visuals_dir.glob("image_*.*"))
    image_paths  = [str(p) for p in image_paths]

    if not image_paths:
        raise Exception(f"No images found in {visuals_dir}")

    logger.info("[%s] Found %s images", str(job.id)[:8], len(image_paths))

    # 2. Load audio to get real duration
    audio_path = job.audio_path
    audio_clip = AudioFileClip(audio_path)
    duration   = audio_clip.duration

    logger.info("[%s] Audio duration: %.2fs", str(job.id)[:8], duration)

    # 3. Build the silent slideshow
    video_clip = _build_slideshow(image_paths, duration)

# 4. Add text overlays
    logger.info("[%s] Adding title and subtitle overlays", str(job.id)[:8])
    video_with_text = _add_overlays(video_clip, job.title, job.script, duration)

    # 5. Attach audio and write to disk
    logger.info("[%s] Attaching audio and rendering", str(job.id)[:8])
    final_clip = video_with_text.with_audio(audio_clip)

    final_clip.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",       # YouTube requires AAC audio
        audio_bitrate="192k",    # good quality for voice
        threads=4,
        preset="fast",
        logger=None,
    )

    audio_clip.close()
    video_clip.close()
    video_with_text.close()
    final_clip.close()

    return str(output_path)





if __name__ == "__main__":
    import os
    import sys
    import django

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_agent.settings")
    django.setup()

    from core.models import VideoJob

    # paste a real job_id from your admin panel here
    JOB_ID = "500419a2431248efb4a0796e69a3656f"

    job = VideoJob.objects.get(id=JOB_ID)
    path = build_video(job)
    print("Output:", path)