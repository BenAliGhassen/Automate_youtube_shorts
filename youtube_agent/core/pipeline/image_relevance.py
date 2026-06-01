"""
Module 2 — Content relevance filter.

Two-stage approach:
  Stage 1 — Filename filter (local, free, instant)
  Stage 2 — Gemini Vision (API call, only if stage 1 passes)

This module filters Wikimedia candidate images before they are assembled
into the final video by combining signal from filename heuristics and
optional Gemini Vision scoring.

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


def compute_filename_score(file_title: str, keyword: str) -> float:
    """
    Return a normalized filename score based on keyword coverage.
    """
    lower = file_title.lower().replace(" ", "_")
    keyword_words = [w.lower() for w in keyword.split() if len(w) > 3]
    if not keyword_words:
        return 0.0
    matches = sum(1 for w in keyword_words if w in lower)
    return round(min(1.0, matches / len(keyword_words)) * 100.0, 1)


def _clamp_score(value: float) -> float:
    return round(min(max(value, 0.0), 100.0), 1)


def score_gemini_vision(
    image_path: Path,
    keyword: str,
    topic: str,
) -> tuple[bool, float, str, dict]:
    """
    Use Gemini Vision to score the image for keyword relevance.
    Returns (passed, score, reason, details).
    """
    api_key = getattr(settings, "GEMINI_API2", "").strip()
    if not api_key:
        return True, 0.0, "Vision check skipped (no API key)", {}

    model = getattr(settings, "GEMINI_VISION_MODEL", "gemini-3.1-flash-lite").strip()
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    try:
        image_data, mime_type = _image_to_base64(image_path)
    except Exception as exc:
        return True, 0.0, f"Vision check skipped (encoding error: {exc})", {}

    prompt = (
        "You are an image relevance judge for viral football Shorts.\n"
        f"Evaluate whether this image matches the keyword '{keyword}' and topic '{topic}'.\n"
        "Return JSON only with scores from 0 to 100.\n"
        "{\n"
        "  \"keyword_relevance\": 0-100,\n"
        "  \"subject_visibility\": 0-100,\n"
        "  \"football_confidence\": 0-100,\n"
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
                        "text": prompt,
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
            return True, 0.0, "Vision check skipped (rate limited)", {}
        logger.warning("Gemini Vision error %s: %s", exc.code, error_body[:200])
        return True, 0.0, f"Vision check skipped (API error {exc.code})", {}
    except Exception as exc:
        logger.warning("Gemini Vision failed: %s", exc)
        return True, 0.0, f"Vision check skipped ({exc})", {}

    try:
        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning("Gemini Vision: no candidates in response")
            return True, 0.0, "Vision check skipped (empty response)", {}

        text = ""
        for part in candidates[0].get("content", {}).get("parts", []):
            text += part.get("text", "")

        text = text.strip()
        if not text:
            logger.warning("Gemini Vision: empty text in response")
            return True, 0.0, "Vision check skipped (empty text)", {}

        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        result = json.loads(text)
    except Exception as exc:
        logger.warning("Gemini Vision parse error: %s — raw: %s", exc, text[:200])
        return True, 0.0, "Vision check skipped (parse error)", {}

    keyword_relevance = _clamp_score(float(result.get("keyword_relevance", 0)))
    subject_visibility = _clamp_score(float(result.get("subject_visibility", 0)))
    football_confidence = _clamp_score(float(result.get("football_confidence", 0)))
    reason = str(result.get("reason", "")).strip()

    vision_score = _clamp_score(
        keyword_relevance * 0.55
        + subject_visibility * 0.25
        + football_confidence * 0.20
    )

    passed = keyword_relevance >= 50.0 and subject_visibility >= 45.0 and football_confidence >= 40.0
    details = {
        "keyword_relevance": keyword_relevance,
        "subject_visibility": subject_visibility,
        "football_confidence": football_confidence,
        "vision_score": vision_score,
    }

    if not passed:
        logger.warning(
            "Vision REJECTED %s: keyword=%s, visible=%s, football=%s — %s",
            image_path.name,
            keyword_relevance,
            subject_visibility,
            football_confidence,
            reason,
        )
        return False, vision_score, f"Vision rejected: {reason}", details

    logger.debug(
        "Vision scored %s for %s: %s",
        vision_score,
        image_path.name,
        reason,
    )
    return True, vision_score, f"Vision passed: {reason}", details


def passes_gemini_vision_check(
    image_path: Path,
    keyword: str,
    topic: str,
) -> tuple[bool, str]:
    passed, score, reason, _details = score_gemini_vision(image_path, keyword, topic)
    if passed:
        return True, reason
    return False, reason


# ── Combined check — call this from visuals.py ────────────────────────────────


def score_image_candidate(
    image_path: Path,
    file_title: str,
    keyword: str,
    topic: str,
) -> tuple[bool, str, float, dict]:
    """
    Score an image candidate using filename and Gemini Vision heuristics.
    Returns (passed, reason, total_score, details).
    """
    passed, reason = passes_filename_filter(file_title, keyword)
    if not passed:
        logger.info("Stage 1 REJECTED %s: %s", image_path.name, reason)
        return False, f"[Filename] {reason}", 0.0, {}

    filename_score = compute_filename_score(file_title, keyword)
    logger.debug("Stage 1 passed for %s -- filename score: %s", image_path.name, filename_score)

    passed, vision_score, reason, details = score_gemini_vision(image_path, keyword, topic)
    if not passed:
        logger.info("Stage 2 REJECTED %s: %s", image_path.name, reason)
        return False, f"[Vision] {reason}", 0.0, details

    total_score = round((filename_score * 0.3) + (vision_score * 0.7), 2)
    details.update(
        {
            "filename_score": filename_score,
            "vision_score": vision_score,
            "total_score": total_score,
        }
    )
    logger.debug(
        "Stage 2 passed for %s -- filename=%s vision=%s total=%s",
        image_path.name,
        filename_score,
        vision_score,
        total_score,
    )
    return True, f"[Scored] filename={filename_score} vision={vision_score} total={total_score}", total_score, details


def passes_relevance_check(
    image_path: Path,
    file_title: str,
    keyword: str,
    topic: str,
) -> tuple[bool, str]:

    passed, reason, _, _ = score_image_candidate(image_path, file_title, keyword, topic)
    return passed, reason


# Variable reference table:
# variable_name | type | purpose
# REJECT_FILENAME_WORDS | list[str] | Filename tokens that indicate wrong or irrelevant images.
# FOOTBALL_POSITIVE_WORDS | list[str] | Positive evidence words that make an image more likely football-related.
# _image_to_base64 | func | Converts a local image file into base64 payload for Gemini Vision.
# compute_filename_score | func | Returns a numeric heuristic score from Wikimedia filename matching.
# _clamp_score | func | Clamps evaluation scores to the 0-100 range.
# score_gemini_vision | func | Calls Gemini Vision to validate image relevance and returns a score.
# passes_gemini_vision_check | func | Simplified wrapper only returning pass/fail and reason.
# score_image_candidate | func | Combines filename and vision scores into a total ranking.
# passes_relevance_check | func | Final scoring wrapper used by the visuals pipeline.


# End of image relevance filtering.
# This module applies a local filename filter first, then performs an
# optional Gemini Vision validation using GEMINI_VISION_MODEL.
