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



def _ken_burns_clip(
    arr: np.ndarray,
    duration: float,
    direction: str = "zoom_in",   # zoom_in | zoom_out | pan_left | pan_right
) -> VideoClip:
    """
    Ken Burns effect with configurable direction.
    zoom_in    : slowly zoom toward center (default)
    zoom_out   : start zoomed in, pull back
    pan_left   : zoom in while panning left
    pan_right  : zoom in while panning right
    """
    h, w     = arr.shape[:2]
    zoom_max = 1.05

    def make_frame(t):
        progress = t / duration

        if direction == "zoom_in":
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx    = w / 2
            cy    = h / 2

        elif direction == "zoom_out":
            scale = zoom_max - (zoom_max - 1.0) * progress
            cx    = w / 2
            cy    = h / 2

        elif direction == "pan_left":
            scale = 1.0 + (zoom_max - 1.0) * progress
            # pan from right to left
            cx = w / 2 + (w * 0.03) * (1 - progress)
            cy = h / 2

        elif direction == "pan_right":
            scale = 1.0 + (zoom_max - 1.0) * progress
            # pan from left to right
            cx = w / 2 - (w * 0.03) * (1 - progress)
            cy = h / 2

        else:
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx, cy = w / 2, h / 2

        new_w = int(w * scale)
        new_h = int(h * scale)

        pil_img    = Image.fromarray(arr)
        pil_img    = pil_img.resize((new_w, new_h), Image.LANCZOS)
        scaled_arr = np.array(pil_img)

        # Crop centered on cx, cy
        x1 = max(0, int(cx * scale - w / 2))
        y1 = max(0, int(cy * scale - h / 2))
        x1 = min(x1, new_w - w)
        y1 = min(y1, new_h - h)

        return scaled_arr[y1:y1 + h, x1:x1 + w]

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


DIRECTIONS = ["zoom_in", "zoom_out", "pan_left", "pan_right"]

def _build_slideshow_from_scenes(
    image_paths: list[str],
    scenes: list[dict],
    total_duration: float,
) -> object:
    n_images = len(image_paths)
    n_scenes = len(scenes)

    if n_images == n_scenes:
        durations = [scene["duration"] for scene in scenes]
    else:
        logger.warning(
            "Image count (%s) != scene count (%s) — equal distribution",
            n_images, n_scenes
        )
        durations = [total_duration / n_images] * n_images

    # Normalize to exact audio length
    duration_sum = sum(durations)
    durations    = [d * (total_duration / duration_sum) for d in durations]

    clips = []
    for i, (path, dur) in enumerate(zip(image_paths, durations)):
        arr = _prepare_image(path)

        # Detect duplicates by filename suffix _dup
        is_dup    = "_dup" in Path(path).stem
        # Rotate direction — duplicates get a different direction from their source
        direction = DIRECTIONS[i % len(DIRECTIONS)]

        clip = _ken_burns_clip(arr, dur, direction=direction)
        clips.append(clip)
        logger.info(
            "  Scene %s: %.2fs — %s [%s]%s",
            i + 1, dur, Path(path).name, direction,
            " (duplicate)" if is_dup else ""
        )

    return concatenate_videoclips(clips, method="compose")



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

    # After collecting image_paths, prepend thumbnail
    thumbnail_path = job_dir / "thumbnail.jpg"
    if thumbnail_path.exists():
        image_paths.insert(0, str(thumbnail_path))
        # Insert a 4.5s scene at the start for the thumbnail
        if job.scenes:
            job.scenes.insert(0, {
                "keyword":  "hook thumbnail",
                "backups":  [],
                "duration": 4.5,
            })
        logger.info("[%s] Thumbnail prepended as first scene", str(job.id)[:8])
    else:
        logger.warning("[%s] No thumbnail found — skipping hook frame", str(job.id)[:8])

    # 2. Load audio to get real duration
    audio_path = job.audio_path
    audio_clip = AudioFileClip(audio_path)
    duration   = audio_clip.duration

    logger.info("[%s] Audio duration: %.2fs", str(job.id)[:8], duration)

    # 3. Build the silent slideshow
    scenes     = job.scenes   # [{"keyword": ..., "duration": 7.0}, ...]
    video_clip = _build_slideshow_from_scenes(image_paths, scenes, duration)

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
    JOB_ID = "ac29303d2239459a845cf52c030d9789"

    job = VideoJob.objects.get(id=JOB_ID)
    path = build_video(job)
    print("Output:", path)