"""Video assembler — builds YouTube Shorts from images, audio, and text overlays."""

import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from django.conf import settings
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from PIL import Image

logger = logging.getLogger(__name__)

# ── YouTube Shorts spec ───────────────────────────────────────────────────────
WIDTH  = 1080
HEIGHT = 1920
FPS    = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH  = os.path.join(SCRIPT_DIR, "arial.ttf")

# ── Transition timing ─────────────────────────────────────────────────────────
# Keep transitions snappy — this is what makes it feel like a real Short
TRANSITION_DURATION     = 0.25   # seconds — fast cut feel
MIN_TRANSITION_DURATION = 0.15
MAX_TRANSITION_DURATION = 0.35

# Ken Burns directions — rotate through these
DIRECTIONS = ["zoom_in", "zoom_out", "pan_left", "pan_right"]


# ── Image preparation ─────────────────────────────────────────────────────────

def _prepare_image(image_path: str) -> np.ndarray:
    """Load, resize, and center-crop image to 1080x1920."""
    img = Image.open(image_path).convert("RGB")

    img_ratio    = img.width / img.height
    target_ratio = WIDTH / HEIGHT

    if img_ratio > target_ratio:
        new_height = HEIGHT
        new_width  = int(HEIGHT * img_ratio)
    else:
        new_width  = WIDTH
        new_height = int(WIDTH / img_ratio)

    img  = img.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width  - WIDTH)  // 2
    top  = (new_height - HEIGHT) // 2
    img  = img.crop((left, top, left + WIDTH, top + HEIGHT))

    return np.array(img)


# ── Ken Burns ─────────────────────────────────────────────────────────────────

def _ken_burns_clip(arr: np.ndarray, duration: float, direction: str = "zoom_in") -> VideoClip:
    """Slow zoom/pan effect on a still image."""
    h, w     = arr.shape[:2]
    zoom_max = 1.06

    def make_frame(t):
        progress = t / max(duration, 0.001)

        if direction == "zoom_in":
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx, cy = w / 2, h / 2
        elif direction == "zoom_out":
            scale = zoom_max - (zoom_max - 1.0) * progress
            cx, cy = w / 2, h / 2
        elif direction == "pan_left":
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx = w / 2 + (w * 0.04) * (1 - progress)
            cy = h / 2
        elif direction == "pan_right":
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx = w / 2 - (w * 0.04) * (1 - progress)
            cy = h / 2
        elif direction == "pan_up":
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx = w / 2
            cy = h / 2 + (h * 0.04) * (1 - progress)
        elif direction == "pan_down":
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx = w / 2
            cy = h / 2 - (h * 0.04) * (1 - progress)
        else:
            scale = 1.0 + (zoom_max - 1.0) * progress
            cx, cy = w / 2, h / 2

        new_w = int(w * scale)
        new_h = int(h * scale)

        pil_img    = Image.fromarray(arr)
        pil_img    = pil_img.resize((new_w, new_h), Image.LANCZOS)
        scaled_arr = np.array(pil_img)

        x1 = max(0, min(int(cx * scale - w / 2), new_w - w))
        y1 = max(0, min(int(cy * scale - h / 2), new_h - h))

        return scaled_arr[y1:y1 + h, x1:x1 + w]

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Transitions ───────────────────────────────────────────────────────────────
# Each transition takes two numpy arrays (from_arr, to_arr),
# a duration in seconds, and the output size tuple.
# All transitions are designed for vertical 1080x1920 portrait format.

def _transition_cut(from_arr, to_arr, duration, size):
    """Hard cut — instant switch, no animation. Fast and clean."""
    w, h = size
    half = duration / 2

    def make_frame(t):
        return from_arr if t < half else to_arr

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_flash(from_arr, to_arr, duration, size):
    """White flash cut — flashes bright then reveals next image."""
    w, h  = size
    white = np.full((h, w, 3), 255, dtype=np.uint8)

    def make_frame(t):
        progress = t / duration
        if progress < 0.4:
            # Fade out to white
            alpha = progress / 0.4
            return (from_arr * (1 - alpha) + white * alpha).astype(np.uint8)
        else:
            # Fade in from white
            alpha = (progress - 0.4) / 0.6
            return (white * (1 - alpha) + to_arr * alpha).astype(np.uint8)

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_slide_left(from_arr, to_arr, duration, size):
    """Slide: current image exits left, next enters from right."""
    w, h = size

    def make_frame(t):
        progress = _ease_out(t / duration)
        offset   = int(w * progress)
        frame    = np.zeros((h, w, 3), dtype=np.uint8)
        # from_arr slides out to the left
        if offset < w:
            frame[:, :w - offset] = from_arr[:, offset:]
        # to_arr slides in from the right
        if offset > 0:
            frame[:, w - offset:] = to_arr[:, :offset]
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_slide_right(from_arr, to_arr, duration, size):
    """Slide: current exits right, next enters from left."""
    w, h = size

    def make_frame(t):
        progress = _ease_out(t / duration)
        offset   = int(w * progress)
        frame    = np.zeros((h, w, 3), dtype=np.uint8)
        if offset < w:
            frame[:, offset:] = from_arr[:, :w - offset]
        if offset > 0:
            frame[:, :offset] = to_arr[:, w - offset:]
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_slide_up(from_arr, to_arr, duration, size):
    """Slide: current exits up, next enters from bottom."""
    w, h = size

    def make_frame(t):
        progress = _ease_out(t / duration)
        offset   = int(h * progress)
        frame    = np.zeros((h, w, 3), dtype=np.uint8)
        if offset < h:
            frame[:h - offset, :] = from_arr[offset:, :]
        if offset > 0:
            frame[h - offset:, :] = to_arr[:offset, :]
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_slide_down(from_arr, to_arr, duration, size):
    """Slide: current exits down, next enters from top."""
    w, h = size

    def make_frame(t):
        progress = _ease_out(t / duration)
        offset   = int(h * progress)
        frame    = np.zeros((h, w, 3), dtype=np.uint8)
        if offset < h:
            frame[offset:, :] = from_arr[:h - offset, :]
        if offset > 0:
            frame[:offset, :] = to_arr[h - offset:, :]
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_zoom_punch(from_arr, to_arr, duration, size):
    """
    Zoom punch — current image zooms in aggressively then cuts to next.
    Very viral feel — used heavily in sports content.
    """
    w, h = size

    def make_frame(t):
        progress = t / duration
        if progress < 0.5:
            # Punch zoom on from_arr
            scale    = 1.0 + 0.15 * (progress / 0.5)
            new_w    = int(w * scale)
            new_h    = int(h * scale)
            pil_img  = Image.fromarray(from_arr)
            pil_img  = pil_img.resize((new_w, new_h), Image.LANCZOS)
            arr      = np.array(pil_img)
            x1       = (new_w - w) // 2
            y1       = (new_h - h) // 2
            return arr[y1:y1 + h, x1:x1 + w]
        else:
            # Instant cut to to_arr, slight zoom back
            scale    = 1.15 - 0.15 * ((progress - 0.5) / 0.5)
            new_w    = int(w * scale)
            new_h    = int(h * scale)
            pil_img  = Image.fromarray(to_arr)
            pil_img  = pil_img.resize((new_w, new_h), Image.LANCZOS)
            arr      = np.array(pil_img)
            x1       = (new_w - w) // 2
            y1       = (new_h - h) // 2
            return arr[y1:y1 + h, x1:x1 + w]

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_whip_left(from_arr, to_arr, duration, size):
    """
    Whip pan left — simulates a fast camera whip to the left.
    Most viral transition on YouTube Shorts and TikTok.
    """
    w, h = size

    def make_frame(t):
        progress = _ease_in_out(t / duration)

        # Motion blur effect using horizontal pixel stretch
        blur_amount = int(w * 0.3 * math.sin(math.pi * progress))

        if progress < 0.5:
            # Whip out: from_arr slides hard left with blur
            offset = int(w * progress * 2)
            frame  = np.zeros((h, w, 3), dtype=np.uint8)
            remain = w - offset
            if remain > 0:
                frame[:, :remain] = from_arr[:, offset:offset + remain]
            # Add motion blur by blending with shifted version
            if blur_amount > 0 and remain > blur_amount:
                blur_strip = from_arr[:, min(offset + blur_amount, w - 1):min(offset + blur_amount + 1, w)]
                if blur_strip.shape[1] > 0:
                    frame[:, :remain] = (frame[:, :remain] * 0.7 + np.broadcast_to(blur_strip, (h, remain, 3)) * 0.3).astype(np.uint8)
        else:
            # Whip in: to_arr enters from right
            offset = int(w * (1 - progress) * 2)
            frame  = np.zeros((h, w, 3), dtype=np.uint8)
            if offset < w:
                frame[:, offset:] = to_arr[:, :w - offset]

        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_whip_up(from_arr, to_arr, duration, size):
    """Whip pan upward — fast vertical whip transition."""
    w, h = size

    def make_frame(t):
        progress = _ease_in_out(t / duration)

        if progress < 0.5:
            offset = int(h * progress * 2)
            frame  = np.zeros((h, w, 3), dtype=np.uint8)
            remain = h - offset
            if remain > 0:
                frame[:remain, :] = from_arr[offset:offset + remain, :]
        else:
            offset = int(h * (1 - progress) * 2)
            frame  = np.zeros((h, w, 3), dtype=np.uint8)
            if offset < h:
                frame[offset:, :] = to_arr[:h - offset, :]

        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_glitch(from_arr, to_arr, duration, size):
    """
    Glitch cut — RGB channel shift effect popular in sports edits.
    Splits RGB channels and offsets them for a digital glitch look.
    """
    w, h = size

    def make_frame(t):
        progress = t / duration

        if progress < 0.3:
            # Glitch on from_arr — shift channels
            intensity = int(w * 0.03 * (progress / 0.3))
            frame     = from_arr.copy()
            if intensity > 0:
                frame[:, intensity:, 0]  = from_arr[:, :w - intensity, 0]   # R shifts right
                frame[:, :w - intensity, 2] = from_arr[:, intensity:, 2]     # B shifts left
            return frame

        elif progress < 0.7:
            # Hard cut midpoint
            blend = (progress - 0.3) / 0.4
            return (from_arr * (1 - blend) + to_arr * blend).astype(np.uint8)

        else:
            # Settle into to_arr with slight glitch
            intensity = int(w * 0.02 * (1 - (progress - 0.7) / 0.3))
            frame     = to_arr.copy()
            if intensity > 0:
                frame[:, :w - intensity, 0] = to_arr[:, intensity:, 0]
            return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_fade(from_arr, to_arr, duration, size):
    """Smooth crossfade — subtle and clean."""
    def make_frame(t):
        alpha = t / duration
        return (from_arr * (1 - alpha) + to_arr * alpha).astype(np.uint8)

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


def _transition_spin_zoom(from_arr, to_arr, duration, size):
    """
    Spin zoom — current image spins and shrinks away,
    next image spins in. Energy-intensive, use sparingly.
    """
    w, h = size

    def make_frame(t):
        progress = t / duration

        if progress < 0.5:
            # Spin out
            angle = 180 * (progress / 0.5)
            scale = 1.0 - 0.4 * (progress / 0.5)
            src   = from_arr
        else:
            # Spin in
            angle = 180 - 180 * ((progress - 0.5) / 0.5)
            scale = 0.6 + 0.4 * ((progress - 0.5) / 0.5)
            src   = to_arr

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        pil_img = Image.fromarray(src).resize((new_w, new_h), Image.LANCZOS)
        pil_img = pil_img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Center on black frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x1    = max(0, (w - new_w) // 2)
        y1    = max(0, (h - new_h) // 2)
        x2    = min(w, x1 + new_w)
        y2    = min(h, y1 + new_h)
        arr   = np.array(pil_img)
        frame[y1:y2, x1:x2] = arr[:y2 - y1, :x2 - x1]
        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Easing functions ──────────────────────────────────────────────────────────

def _ease_out(t: float) -> float:
    """Ease out cubic — fast start, slow end. Good for slides."""
    return 1 - (1 - t) ** 3


def _ease_in_out(t: float) -> float:
    """Ease in-out — slow start, fast middle, slow end. Good for whips."""
    if t < 0.5:
        return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2


# ── Transition registry ───────────────────────────────────────────────────────

# Grouped by energy level so we can pick appropriate transitions per format
TRANSITIONS_HIGH_ENERGY = [
    "whip_left",
    "whip_up",
    "zoom_punch",
    "glitch",
    "slide_left",
    "slide_up",
]

TRANSITIONS_MEDIUM_ENERGY = [
    "slide_left",
    "slide_right",
    "slide_up",
    "slide_down",
    "zoom_punch",
    "flash",
]

TRANSITIONS_LOW_ENERGY = [
    "slide_left",
    "slide_right",
    "fade",
    "cut",
]

TRANSITION_BUILDERS = {
    "cut":        _transition_cut,
    "flash":      _transition_flash,
    "slide_left":  _transition_slide_left,
    "slide_right": _transition_slide_right,
    "slide_up":    _transition_slide_up,
    "slide_down":  _transition_slide_down,
    "zoom_punch":  _transition_zoom_punch,
    "whip_left":   _transition_whip_left,
    "whip_up":     _transition_whip_up,
    "glitch":      _transition_glitch,
    "fade":        _transition_fade,
    "spin_zoom":   _transition_spin_zoom,
}

# Format → transition energy level mapping
FORMAT_TRANSITIONS = {
    "comparison":  TRANSITIONS_HIGH_ENERGY,    # fast cuts for vs content
    "factbombs":   TRANSITIONS_MEDIUM_ENERGY,  # punchy but readable
    "didyouknow":  TRANSITIONS_MEDIUM_ENERGY,
    "top5":        TRANSITIONS_MEDIUM_ENERGY,
    "default":     TRANSITIONS_MEDIUM_ENERGY,
}


def _build_transition_sequence(count: int, format_type: str = "default") -> list[str]:
    """
    Build a non-repeating transition sequence appropriate for the content format.
    Comparison format gets high-energy transitions.
    Other formats get medium energy.
    First transition is always a whip or zoom punch for maximum impact.
    """
    pool     = FORMAT_TRANSITIONS.get(format_type, TRANSITIONS_MEDIUM_ENERGY)
    sequence = []
    previous = None

    for i in range(count):
        # First transition hits hardest
        if i == 0:
            openers = ["whip_left", "zoom_punch", "glitch"]
            picked  = random.choice([t for t in openers if t in pool] or pool)
        else:
            choices = [t for t in pool if t != previous]
            picked  = random.choice(choices or pool)

        sequence.append(picked)
        previous = picked

    return sequence


# ── Subtitle helpers ──────────────────────────────────────────────────────────

def _split_into_chunks(text: str, chunk_duration: float, total_duration: float) -> list[dict]:
    """Split script into timed subtitle chunks."""
    words          = text.split()
    total_words    = len(words)
    n_chunks       = max(1, int(total_duration / chunk_duration))
    words_per_chunk = max(1, total_words // n_chunks)
    chunks         = []

    for i in range(n_chunks):
        start      = i * chunk_duration
        end        = min((i + 1) * chunk_duration, total_duration)
        word_start = i * words_per_chunk
        word_end   = word_start + words_per_chunk if i < n_chunks - 1 else total_words
        chunk_text = " ".join(words[word_start:word_end])
        if chunk_text.strip():
            chunks.append({"text": chunk_text, "start": start, "end": end})

    return chunks


def _add_overlays(video_clip, title: str, script: str, duration: float) -> CompositeVideoClip:
    """Add subtitle text overlays to the video."""
    layers = [video_clip]
    chunks = _split_into_chunks(script, chunk_duration=4.0, total_duration=duration)
    logger.info("Creating %s subtitle chunks", len(chunks))

    for chunk in chunks:
        try:
            subtitle_clip = (
                TextClip(
                    text=chunk["text"],
                    font_size=42,
                    color="white",
                    font=FONT_PATH,
                    stroke_color="black",
                    stroke_width=2,
                    method="caption",
                    size=(WIDTH - 60, None),
                    text_align="center",
                )
                .with_start(chunk["start"])
                .with_duration(chunk["end"] - chunk["start"])
                .with_position(("center", HEIGHT - 280))
            )
            layers.append(subtitle_clip)
        except Exception as exc:
            logger.warning("Could not create subtitle chunk '%s': %s", chunk["text"][:30], exc)

    return CompositeVideoClip(layers, size=(WIDTH, HEIGHT))


# ── Scene candidate loading ───────────────────────────────────────────────────

def _load_scene_candidate_paths(visuals_dir: Path, scene_index: int) -> list[str]:
    """Load image paths for a scene from its saved candidate metadata."""
    metadata_file = visuals_dir / f"scene_{scene_index + 1}_candidates.json"
    if not metadata_file.exists():
        return []
    try:
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        paths = [
            item.get("path")
            for item in metadata.get("top_candidates", [])
            if item.get("path")
        ]
        return [str(Path(p)) for p in paths if p]
    except Exception as exc:
        logger.warning("Could not read scene metadata %s: %s", metadata_file.name, exc)
        return []


def _load_scene_candidate_groups(visuals_dir: Path, scenes: list[dict]) -> list[list[str]]:
    """Load all scene candidate groups."""
    return [
        _load_scene_candidate_paths(visuals_dir, idx)
        for idx, _ in enumerate(scenes)
    ]


# ── Slideshow builders ────────────────────────────────────────────────────────

def _build_slideshow_from_scene_groups(
    scene_groups: list[list[str]],
    scenes: list[dict],
    total_duration: float,
    format_type: str = "default",
) -> object:
    """
    Build slideshow with dynamic transitions between every image.
    Each image gets Ken Burns motion. Transitions are format-aware.
    """
    flat_paths: list[str] = []
    flat_durations: list[float] = []

    for scene, group in zip(scenes, scene_groups):
        if not group:
            raise Exception(f"Scene '{scene.get('keyword', 'unknown')}' has no images.")
        scene_duration     = float(scene.get("duration", total_duration / len(scenes)))
        per_image_duration = scene_duration / len(group)
        flat_paths.extend(group)
        flat_durations.extend([per_image_duration] * len(group))

    if not flat_paths:
        raise Exception("No image paths for slideshow.")

    image_count      = len(flat_paths)
    transition_count = max(0, image_count - 1)

    # Calculate transition duration — keep it snappy
    if transition_count > 0:
        # Never let transitions eat more than 20% of total duration
        max_trans = (total_duration * 0.20) / transition_count
        transition_duration = min(
            TRANSITION_DURATION,
            max(MIN_TRANSITION_DURATION, max_trans)
        )
    else:
        transition_duration = TRANSITION_DURATION

    # Normalize image durations to account for transition time
    transition_total = transition_count * transition_duration
    available        = max(total_duration - transition_total, 1.0)
    duration_sum     = sum(flat_durations)
    flat_durations   = [d * (available / duration_sum) for d in flat_durations]

    # Build transition sequence based on content format
    transition_plan = _build_transition_sequence(transition_count, format_type)

    logger.info(
        "Building slideshow: %s images, %s transitions, format=%s, transition_dur=%.2fs",
        image_count, transition_count, format_type, transition_duration
    )

    clips = []
    size  = None

    for i, (path, dur) in enumerate(zip(flat_paths, flat_durations)):
        arr = _prepare_image(path)

        if size is None:
            size = (arr.shape[1], arr.shape[0])

        is_dup    = "_dup" in Path(path).stem
        direction = DIRECTIONS[i % len(DIRECTIONS)]
        clip      = _ken_burns_clip(arr, dur, direction=direction)
        clips.append(clip)

        logger.info(
            "  Image %s/%s: %.2fs [%s] %s%s",
            i + 1, image_count, dur,
            direction, Path(path).name,
            " (dup)" if is_dup else ""
        )

        # Insert transition between images
        if i < image_count - 1:
            next_arr        = _prepare_image(flat_paths[i + 1])
            transition_name = transition_plan[i]
            builder         = TRANSITION_BUILDERS[transition_name]

            try:
                trans_clip = builder(arr, next_arr, transition_duration, size)
                clips.append(trans_clip)
                logger.info(
                    "  Transition %s: %s -> %s (%.2fs)",
                    transition_name, Path(path).name,
                    Path(flat_paths[i + 1]).name, transition_duration
                )
            except Exception as exc:
                logger.warning("Transition '%s' failed: %s — using cut", transition_name, exc)
                clips.append(_transition_cut(arr, next_arr, transition_duration, size))

    return concatenate_videoclips(clips, method="compose")


def _build_slideshow_from_scenes(
    image_paths: list[str],
    scenes: list[dict],
    total_duration: float,
    format_type: str = "default",
) -> object:
    """Fallback slideshow builder from a flat image list."""
    n_images = len(image_paths)
    n_scenes = len(scenes)

    if n_images == n_scenes:
        durations = [float(s.get("duration", total_duration / n_scenes)) for s in scenes]
    else:
        durations = [total_duration / n_images] * n_images

    duration_sum = sum(durations)
    durations    = [d * (total_duration / duration_sum) for d in durations]

    # Wrap as fake scene groups and delegate
    scene_groups = [[p] for p in image_paths]
    fake_scenes  = [
        {"keyword": Path(p).stem, "duration": d}
        for p, d in zip(image_paths, durations)
    ]

    return _build_slideshow_from_scene_groups(
        scene_groups, fake_scenes, total_duration, format_type
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def build_video(job) -> str:
    """Assemble the final MP4 Short from images, audio, and text."""
    job_dir     = Path(settings.MEDIA_ROOT) / "jobs" / str(job.id)
    output_path = job_dir / "final.mp4"
    visuals_dir = job_dir / "visuals"

    # Detect content format for transition selection
    format_type = getattr(job, "format_type", None) or "default"
    if hasattr(job, "scenes") and job.scenes:
        # Try to read format from scenes metadata
        pass
    logger.info("[%s] Content format: %s", str(job.id)[:8], format_type)

    # Collect image paths
    image_paths = sorted(visuals_dir.glob("image_*.*"))
    image_paths = [str(p) for p in image_paths]

    if not image_paths:
        raise Exception(f"No images found in {visuals_dir}")

    logger.info("[%s] Found %s images", str(job.id)[:8], len(image_paths))

    # Load audio
    audio_clip = AudioFileClip(job.audio_path)
    duration   = audio_clip.duration
    logger.info("[%s] Audio duration: %.2fs", str(job.id)[:8], duration)

    # Thumbnail hook frame
    thumbnail_path   = job_dir / "thumbnail.jpg"
    thumbnail_exists = thumbnail_path.exists()

    scenes = list(job.scenes or [])

    # Load scene candidate groups
    candidate_groups = _load_scene_candidate_groups(visuals_dir, scenes)
    use_groups       = all(group for group in candidate_groups)

    if thumbnail_exists:
        if use_groups:
            candidate_groups.insert(0, [str(thumbnail_path)])
        else:
            image_paths.insert(0, str(thumbnail_path))
        scenes.insert(0, {"keyword": "hook thumbnail", "backups": [], "duration": 4.5})
        logger.info("[%s] Thumbnail prepended", str(job.id)[:8])

    # Build slideshow
    if use_groups:
        video_clip = _build_slideshow_from_scene_groups(
            candidate_groups, scenes, duration, format_type
        )
    else:
        video_clip = _build_slideshow_from_scenes(
            image_paths, scenes, duration, format_type
        )

    # Add subtitles
    logger.info("[%s] Adding subtitle overlays", str(job.id)[:8])
    video_with_text = _add_overlays(video_clip, job.title, job.script, duration)

    # Attach audio and render
    logger.info("[%s] Rendering final.mp4", str(job.id)[:8])
    final_clip = video_with_text.with_audio(audio_clip)

    final_clip.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
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
    import django

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_agent.settings")
    django.setup()

    from core.models import VideoJob

    JOB_ID = "396c33b9e4304d54ac21076d702fe153"
    job    = VideoJob.objects.get(id=JOB_ID)
    path   = build_video(job)
    print("Output:", path)