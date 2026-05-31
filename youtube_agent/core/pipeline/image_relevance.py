"""
Module 2 — Content relevance filter.
Two-stage approach:
  Stage 1 — Filename filter (local, free, instant)
  Stage 2 — Gemini Vision (API call, only if stage 1 passes)

Uses 1 Gemini API call per image that passes stage 1.
Budget impact: ~10 calls per video (5-7 images, some filtered early)
"""

import base64
import json
import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings

logger = logging.getLogger(__name__)


# ── Stage 1 — Filename filter (free, local) ───────────────────────────────────

# Words in Wikimedia filenames that indicate wrong content
REJECT_FILENAME_WORDS = [
    # Women's football
    "women", "woman", "female", "ladies", "girls", "feminine",
    "feminin", "mujer", "damen", "frauen", "nwsl", "wsl", "wfc",
    "matildas", "lionesses",

    # Youth / non-first-team
    # "youth", "u21", "u20", "u19", "u18", "u17", "u16", "u15",
    # "junior", "academy", "reserve", "b_team", "under-21",

    # Kits / logos / icons only (no action)
    "kit_only", "shirt_only", "badge_only", "logo_only",
    "crest_only", "icon_only",

    # Clearly off-topic
    "mascot", "cheerleader", "empty_stadium", "construction",
    "architecture", "diagram", "map", "chart",
]

# Words that MUST appear or the image is suspicious
# (relaxed — not all football images have these)
FOOTBALL_POSITIVE_WORDS = [
    "football", "soccer", "fifa", "uefa", "goal", "match",
    "player", "stadium", "messi", "ronaldo", "mbappe", "neymar",
    "fc_", "_fc", "real_madrid", "barcelona", "liverpool",
    "manchester", "juventus", "chelsea", "arsenal", "milan",
    "world_cup", "champions_league", "premier_league", "la_liga",
]


def passes_filename_filter(file_title: str, keyword: str) -> tuple[bool, str]:
    """
    Stage 1 — Check Wikimedia filename for obvious wrong content.
    Fast and free — no API call.
    Returns (passed, reason).
    """
    lower = file_title.lower().replace(" ", "_")

    # Check for rejection words
    for word in REJECT_FILENAME_WORDS:
        if word in lower:
            return False, f"Filename contains rejected word: '{word}'"

    # Check keyword words appear in filename (loose match)
    keyword_words = [w.lower() for w in keyword.split() if len(w) > 3]
    matches = sum(1 for w in keyword_words if w in lower)

    # Also check for any positive football indicator
    has_football_context = any(w in lower for w in FOOTBALL_POSITIVE_WORDS)

    if not has_football_context and matches == 0:
        return False, f"Filename has no football context and no keyword match"

    return True, f"Filename OK — {matches}/{len(keyword_words)} keyword words matched"


# ── Stage 2 — Gemini Vision (API call) ───────────────────────────────────────

def _image_to_base64(image_path: Path, max_size: int = 256) -> tuple[str, str]:
    """Convert image to base64 for vision model input.

    To avoid exceeding the model context, resize large images to a small
    thumbnail (default max dimension `max_size`) and encode as JPEG.
    If Pillow is not available, falls back to returning the original bytes.
    """
    suffix = image_path.suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    try:
        from PIL import Image
        from io import BytesIO

        with Image.open(image_path) as im:
            im.thumbnail((max_size, max_size))
            buf = BytesIO()
            # Save as JPEG to keep size small regardless of original format
            im.convert("RGB").save(buf, format="JPEG", quality=60)
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime_type = "image/jpeg"
            return data, mime_type
    except Exception:
        # Pillow not installed or processing failed — fall back to raw bytes
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return data, mime_type


def passes_gemini_vision_check(
    image_path: Path,
    keyword: str,
    topic: str,
) -> tuple[bool, str]:
    """
    Use Gemini 3.1 Flash Lite to verify image relevance.
    Uses GEMINI_API2 specifically so GEMINI_API_KEY can remain available for other tasks.
    Returns (passed, reason).
    """
    api_key = getattr(settings, "GEMINI_API2", "").strip()
    if not api_key:
        return True, "Vision check skipped (no API key)"

    model = getattr(settings, "GEMINI_VISION_MODEL", "gemini-3.1-flash-lite").strip()
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    try:
        image_data, mime_type = _image_to_base64(image_path)
    except Exception as exc:
        return True, f"Vision check skipped (encoding error: {exc})"

    prompt = (
        "Look at this image. Answer in JSON only.\n"
        f"Expected: men's football image related to '{keyword}'.\n\n"
        "{\n"
        "  \"is_mens_football\": true or false,\n"
        "  \"is_relevant\": true or false,\n"
        "  \"subject_visible\": true or false,\n"
        "  \"reason\": \"one short sentence\"\n"
        "}"
    )

    request_body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_data,
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 150,
        },
    }

    req = Request(
        url=endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "youtube-agent/1.0",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            logger.warning("Gemini Vision rate limited — skipping %s", image_path.name)
            return True, "Vision check skipped (rate limited)"
        logger.warning("Gemini Vision error %s: %s", exc.code, error_body[:200])
        return True, f"Vision check skipped (API error {exc.code})"
    except Exception as exc:
        logger.warning("Gemini Vision failed: %s", exc)
        return True, f"Vision check skipped ({exc})"

    try:
        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning("Gemini Vision: no candidates in response")
            return True, "Vision check skipped (empty response)"

        text = ""
        for part in candidates[0].get("content", {}).get("parts", []):
            text += part.get("text", "")

        text = text.strip()
        if not text:
            logger.warning("Gemini Vision: empty text in response")
            return True, "Vision check skipped (empty text)"

        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "``":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        result = json.loads(text)
    except Exception as exc:
        logger.warning("Gemini Vision parse error: %s — raw: %s", exc, text[:200])
        return True, "Vision check skipped (parse error)"

    is_mens = result.get("is_mens_football", True)
    is_relevant = result.get("is_relevant", True)
    is_visible = result.get("subject_visible", True)
    reason = result.get("reason", "")

    passed = is_mens and is_relevant and is_visible

    if not passed:
        parts = []
        if not is_mens:
            parts.append("not men's football")
        if not is_relevant:
            parts.append("not relevant")
        if not is_visible:
            parts.append("subject not visible")
        full_reason = " | ".join(parts)
        if reason:
            full_reason += f" ({reason})"
        logger.warning("Vision REJECTED %s: %s", image_path.name, full_reason)
        return False, full_reason

    logger.debug("Vision ACCEPTED %s: %s", image_path.name, reason)
    return True, f"Vision passed: {reason}"

# ── Combined check — call this from visuals.py ────────────────────────────────

def passes_relevance_check(
    image_path: Path,
    file_title: str,
    keyword: str,
    topic: str,
) -> tuple[bool, str]:

    # Stage 1 -- filename filter (free, instant)
    passed, reason = passes_filename_filter(file_title, keyword)
    if not passed:
        logger.info("Stage 1 REJECTED %s: %s", image_path.name, reason)
        return False, f"[Filename] {reason}"

    logger.debug("Stage 1 passed for %s -- running Gemini Vision check", image_path.name)

    # Stage 2 -- Gemini 3.1 Flash Lite Vision
    passed, reason = passes_gemini_vision_check(image_path, keyword, topic)
    if not passed:
        logger.info("Stage 2 REJECTED %s: %s", image_path.name, reason)
        return False, f"[Vision] {reason}"

    return True, "Both checks passed"


# End of image relevance filtering.
# This module applies a local filename filter first, then performs an
# optional Gemini Vision validation using GEMINI_VISION_MODEL.
