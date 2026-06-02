"""Image collection and filtering logic for sports visuals.

This module tries TheSportsDB player imagery first, then falls back to Wikimedia
Commons when the player source does not provide enough usable images.

The code keeps the pipeline deterministic, rate-limit aware, and heavily logged so
it is easy to understand where each image came from.
"""

import json  # Read and write API responses and metadata.
import logging  # Emit progress, warnings, and debugging information.
import math  # Support smooth motion curves in transition animations.
import random  # Pick randomized transitions.
import sys  # Exit cleanly when the module is run directly.
import time  # Add small pauses to respect API rate limits.
from pathlib import Path  # Work with file-system paths safely.
from urllib.error import HTTPError, URLError  # Catch network and HTTP failures.
from urllib.parse import urlencode  # Build query strings safely.
from urllib.request import Request, urlopen  # Perform HTTP requests without extra dependencies.

import numpy as np  # Work with image arrays returned by the preprocessing pipeline.
from django.conf import settings  # Access MEDIA_ROOT from Django settings.
from moviepy import ColorClip, CompositeVideoClip, ImageClip, concatenate_videoclips  # Build the final slideshow.

from .image_quality import passes_quality_check, score_quality, score_resolution  # Image quality scoring.
from .image_relevance import passes_relevance_check, score_image_candidate  # Semantic / relevance scoring.


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

# Make the lower-level scoring loggers verbose so you can debug image decisions.
logging.getLogger("core.pipeline.image_quality").setLevel(logging.DEBUG)
logging.getLogger("core.pipeline.image_relevance").setLevel(logging.DEBUG)

# Module-level logger for this file.
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# API constants and pipeline limits
# -----------------------------------------------------------------------------

# Wikimedia Commons API endpoint used as a fallback image source.
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# TheSportsDB API endpoint.
# The free public API key is commonly used in examples; if your project uses a
# different key, replace this value accordingly.
THESPORTSDB_API = "https://www.thesportsdb.com/api/v1/json/123"

# Player image keys we try from TheSportsDB, in priority order.
THESPORTSDB_PLAYER_IMAGE_KEYS = (
    "strCutout",
    "strThumb",
    "strRender",
    "strBanner",
    "strFanart1",
    "strPoster",
)

# How many Wikimedia search results to inspect per keyword.
SEARCH_RESULTS_PER_KEYWORD = 10

# How many TheSportsDB candidates to keep per keyword.
THESPORTSDB_IMAGES_PER_KEYWORD = 1

# How many Wikimedia candidates to keep per keyword.
WIKIMEDIA_IMAGES_PER_KEYWORD = 2

# Maximum number of final visuals to assemble.
TARGET_IMAGES = 7

# Minimum number of final visuals required.
MIN_IMAGES = 4

# Prevent the pipeline from downloading too many images overall.
VISUAL_COUNT = 12

# Base transition timing.
TRANSITION_DURATION = 0.42
MIN_TRANSITION_DURATION = 0.18
MAX_TRANSITION_DURATION = 0.48

# Competition names that often need a "men" qualifier in Wikimedia search.
COMPETITION_KEYWORDS = [
    "champions league",
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "world cup",
    "euro",
    "copa del rey",
    "fa cup",
    "league cup",
    "europa league",
    "conference league",
]


# -----------------------------------------------------------------------------
# Shared HTTP helpers
# -----------------------------------------------------------------------------


def _http_get_json(url: str, retries: int = 3, timeout: int = 15) -> dict:
    """Fetch JSON from an HTTP endpoint with a small retry policy."""
    for attempt in range(retries):
        try:
            # Build a GET request with a simple user agent.
            req = Request(
                url=url,
                headers={"User-Agent": "youtube-agent/1.0 (educational project)"},
                method="GET",
            )
            # Open the URL and decode the JSON payload.
            with urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            # Retry HTTP 429 with exponential backoff.
            if exc.code == 429:
                wait = 2 ** attempt * 2
                logger.warning(
                    "Rate limited (429) on %s — waiting %ss before retry %s/%s",
                    url,
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                continue

            # For other HTTP errors, log and stop retrying.
            logger.warning("HTTP error on %s: %s", url, exc)
            return {}

        except (URLError, TimeoutError) as exc:
            # Network-level failures are logged and retried.
            logger.warning("Network error on %s: %s", url, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {}

        except Exception as exc:
            # Catch-all for unexpected parsing or transport issues.
            logger.warning("Unexpected request failure on %s: %s", url, exc)
            return {}

    return {}


# -----------------------------------------------------------------------------
# TheSportsDB helpers
# -----------------------------------------------------------------------------


def _thesportsdb_request(url: str, retries: int = 3) -> dict:
    """Fetch JSON from TheSportsDB with retry support."""
    return _http_get_json(url=url, retries=retries, timeout=15)


def _normalize_image_url(url: str) -> str:
    """Convert protocol-relative URLs to HTTPS URLs."""
    if url.startswith("//"):
        return f"https:{url}"
    return url


def _thesportsdb_search_players(keyword: str, limit: int = 5) -> list[dict]:
    """Search TheSportsDB for players that match the keyword."""
    params = urlencode({"p": keyword})
    url = f"{THESPORTSDB_API}/searchplayers.php?{params}"
    data = _thesportsdb_request(url)

    # The API may return the list under different keys depending on the payload.
    players = data.get("player") or data.get("players") or []

    logger.info("TheSportsDB search '%s' -> %s player(s)", keyword, len(players))
    time.sleep(0.6)
    return players[:limit]


def _player_image_urls(player: dict) -> list[str]:
    """Collect usable image URLs from a TheSportsDB player record."""
    urls: list[str] = []

    # Try image fields in priority order so the best portrait-style image comes first.
    for key in THESPORTSDB_PLAYER_IMAGE_KEYS:
        value = player.get(key)
        if value:
            urls.append(_normalize_image_url(value))

    # Remove duplicates while preserving the original order.
    seen = set()
    ordered = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)

    return ordered


# -----------------------------------------------------------------------------
# Wikimedia helpers
# -----------------------------------------------------------------------------


def _wikimedia_request(url: str, retries: int = 3) -> dict:
    """Fetch JSON from Wikimedia Commons with retry support."""
    return _http_get_json(url=url, retries=retries, timeout=15)


def _wikimedia_search(keyword: str, limit: int = 5) -> list[str]:
    """Search Wikimedia Commons and return file titles for the keyword."""
    params = urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": keyword,
            "srnamespace": 6,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        }
    )
    url = f"{WIKIMEDIA_API}?{params}"

    try:
        data = _wikimedia_request(url)
        results = data.get("query", {}).get("search", [])
        titles = [r["title"] for r in results if r.get("title")]
        logger.info("Wikimedia search '%s' -> %s result(s)", keyword, len(titles))
        time.sleep(1.0)
        return titles
    except Exception as exc:
        logger.warning("Wikimedia search failed for '%s': %s", keyword, exc)
        return []


def _get_image_url(file_title: str, width: int = 1080) -> str | None:
    """Resolve a Wikimedia file title into a usable direct image URL."""
    params = urlencode(
        {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": width,
            "format": "json",
            "utf8": 1,
        }
    )
    url = f"{WIKIMEDIA_API}?{params}"

    try:
        data = _wikimedia_request(url)
        pages = data.get("query", {}).get("pages", {})
        time.sleep(0.8)

        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                return imageinfo[0].get("thumburl") or imageinfo[0].get("url")

        return None
    except Exception as exc:
        logger.warning("Could not get URL for '%s': %s", file_title, exc)
        return None


# -----------------------------------------------------------------------------
# File and image validation helpers
# -----------------------------------------------------------------------------


def _download_image(url: str, output_path: Path, retries: int = 3) -> bool:
    """Download an image to disk with retry support for rate limits."""
    for attempt in range(retries):
        try:
            req = Request(
                url=url,
                headers={"User-Agent": "youtube-agent/1.0 (educational project)"},
                method="GET",
            )
            with urlopen(req, timeout=30) as response:
                data = response.read()

            if not data:
                return False

            output_path.write_bytes(data)
            return True

        except HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * 3
                logger.warning(
                    "Download rate limited for %s — waiting %ss (attempt %s/%s)",
                    url,
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
            else:
                logger.warning("Failed to download %s: %s", url, exc)
                return False

        except URLError as exc:
            logger.warning("Failed to download %s: %s", url, exc)
            return False

    logger.warning("Gave up downloading after %s retries: %s", retries, url)
    return False


def _is_usable_image(file_title: str) -> bool:
    """Reject file types that the pipeline should not use directly."""
    lower = file_title.lower()

    # Skip SVG because it often needs separate conversion.
    if lower.endswith(".svg"):
        return False

    # Skip formats that are not simple still images.
    for ext in [".ogg", ".ogv", ".webm", ".pdf", ".tif", ".tiff"]:
        if lower.endswith(ext):
            return False

    return True


def _is_well_framed(image_path: Path) -> bool:
    """Perform a light sanity check on the image's brightness and framing."""
    try:
        from PIL import Image, ImageStat

        # Load the image and convert it to RGB for stable statistics.
        img = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)

        # Reject very dark images because they rarely work well in video.
        brightness = sum(stat.mean) / 3
        if brightness < 30:
            logger.warning("Rejected dark image: %s (brightness=%.1f)", image_path.name, brightness)
            return False

        # Reject images that are too uniform because they are often logos or blanks.
        stddev = sum(stat.stddev) / 3
        if stddev < 15:
            logger.warning("Rejected uniform image: %s (stddev=%.1f)", image_path.name, stddev)
            return False

        # Reject ultra-wide panoramic images because they crop poorly into portrait.
        original_ratio = img.width / img.height
        if original_ratio > 3.0:
            logger.warning("Rejected panoramic image: %s (ratio=%.2f)", image_path.name, original_ratio)
            return False

        return True

    except Exception as exc:
        # If the check cannot be performed, keep the image instead of blocking it.
        logger.warning("Could not verify image %s: %s", image_path.name, exc)
        return True


# -----------------------------------------------------------------------------
# Search-term helpers
# -----------------------------------------------------------------------------


def _build_search_term(keyword: str) -> str:
    """Build a more precise Wikimedia search term for competition names."""
    keyword_lower = keyword.lower()

    # Add a men's qualifier for competition names to reduce unrelated results.
    for comp in COMPETITION_KEYWORDS:
        if comp in keyword_lower:
            return f"{keyword} men final"

    return keyword


# -----------------------------------------------------------------------------
# Candidate metadata
# -----------------------------------------------------------------------------


def _save_scene_candidates(
    visuals_dir: Path,
    scene_index: int,
    keyword: str,
    candidates: list[dict],
) -> None:
    """Persist candidate ranking metadata for a scene as JSON."""
    metadata_path = visuals_dir / f"scene_{scene_index + 1}_candidates.json"

    metadata = {
        "scene_index": scene_index + 1,
        "keyword": keyword,
        "top_candidates": [
            {
                "path": candidate["path"],
                "source": candidate.get("source", "unknown"),
                "score": candidate["score"],
                "reason": candidate["reason"],
                "quality_score": candidate["quality_score"],
                "resolution_score": candidate["resolution_score"],
                "vision_score": candidate["vision_score"],
                "source_title": candidate["source_title"],
            }
            for candidate in candidates[:3]
        ],
        "all_candidates": candidates,
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


# -----------------------------------------------------------------------------
# Shared candidate scoring helper
# -----------------------------------------------------------------------------


def _score_thesportsdb_candidate(
    *,
    image_path: Path,
    player_name: str,
    keyword: str,
    job_id: str,
    image_url: str = "",
) -> dict | None:
    """
    Score a TheSportsDB image.
    Skips relevance check entirely — TheSportsDB images are pre-curated.
    Only runs quality check (brightness, resolution, framing).
    """
    # Quality check only — no filename filter, no Gemini Vision
    quality_ok, quality_reason = passes_quality_check(image_path)
    if not quality_ok:
        logger.info(
            "[%s] [THESPORTSDB] Quality REJECTED: %s — %s",
            job_id[:8],
            player_name,
            quality_reason,
        )
        return None

    resolution_score, resolution_reason = score_resolution(image_path)
    quality_score, quality_reason       = score_quality(image_path)

    # TheSportsDB images get a fixed high relevance score
    # because they are exact player matches by definition
    relevance_score = 90.0

    score = round(
        relevance_score * 0.60
        + quality_score  * 0.25
        + resolution_score * 0.15,
        2,
    )

    return {
        "path":             str(image_path),
        "source":           "thesportsdb",
        "source_title":     player_name,
        "source_url":       image_url,
        "keyword":          keyword,
        "score":            score,
        "resolution_score": resolution_score,
        "quality_score":    quality_score,
        "vision_score":     relevance_score,
        "reason":           "TheSportsDB pre-curated player image — relevance assumed",
        "details": {
            "resolution_reason": resolution_reason,
            "quality_reason":    quality_reason,
            "relevance_details": {"source": "thesportsdb_assumed"},
        },
    }


# -----------------------------------------------------------------------------
# TheSportsDB collection
# -----------------------------------------------------------------------------


def _fetch_thesportsdb_player_candidates(
    keyword: str,
    job_id: str,
    visuals_dir: Path,
    all_downloaded: list,
    seen_urls: set,
    max_images: int = THESPORTSDB_IMAGES_PER_KEYWORD,
    topic: str = "",
) -> list[dict]:
    """Try to fetch player images from TheSportsDB for one keyword."""
    players = _thesportsdb_search_players(keyword, limit=5)

    # Return early if the API found no players.
    if not players:
        return []

    # Prefer exact name matches first when possible.
    normalized_keyword = keyword.strip().lower()
    players.sort(
        key=lambda p: 0 if (p.get("strPlayer", "").strip().lower() == normalized_keyword) else 1
    )

    keyword_images: list[dict] = []

    for player in players:
        if len(keyword_images) >= max_images:
            break

        player_name = player.get("strPlayer") or keyword
        image_urls = _player_image_urls(player)

        # Inspect each available player image field in priority order.
        for image_url in image_urls:
            if len(keyword_images) >= max_images:
                break

            if not image_url or image_url in seen_urls:
                continue

            # Log before the download so the source is visible during progress.
            logger.info(
                "[%s] [THESPORTSDB] Fetching player='%s' image_url='%s'",
                job_id[:8],
                player_name,
                image_url,
            )

            url_clean = image_url.lower().split("?")[0]
            if url_clean.endswith(".png"):
                ext = ".png"
            elif url_clean.endswith(".webp"):
                ext = ".webp"
            else:
                ext = ".jpg"

            global_index = len(all_downloaded) + len(keyword_images) + 1
            output_path = visuals_dir / f"image_{global_index:02d}{ext}"

            # Download the candidate image.
            if not _download_image(image_url, output_path):
                logger.warning("[%s] [THESPORTSDB] Download failed for player='%s'", job_id[:8], player_name)
                continue

            # Build the scored candidate object.
            candidate = _score_thesportsdb_candidate(
                image_path=output_path,
                player_name=player_name,
                keyword=keyword,
                job_id=job_id,
                image_url=image_url,
            )
            
            # Remove the file if it did not pass the scoring gates.
            if candidate is None:
                output_path.unlink(missing_ok=True)
                time.sleep(1.0)
                continue

            # Keep the URL so we do not download it twice.
            seen_urls.add(image_url)
            keyword_images.append(candidate)

            logger.info(
                "[%s] [THESPORTSDB] ACCEPTED %s (score=%.2f) for '%s' (%s)",
                job_id[:8],
                output_path.name,
                candidate["score"],
                keyword,
                player_name,
            )

            # One image per player is enough for this stage.
            break

        # Small pause between players to keep the API usage gentle.
        time.sleep(0.5)

    return keyword_images


# -----------------------------------------------------------------------------
# Wikimedia collection
# -----------------------------------------------------------------------------

def _score_wikimedia_candidate(
    *,
    image_path: Path,
    file_title: str,
    keyword: str,
    job_id: str,
    topic: str,
    image_url: str = "",
) -> dict | None:
    """
    Score a Wikimedia image.
    Runs the full pipeline: quality check + filename filter + Gemini Vision.
    This is the last resort path — relevance checking is mandatory here
    because Wikimedia search returns uncontrolled, uncurated results.
    """
    # Step 1 — quality gate (brightness, resolution, framing)
    quality_ok, quality_reason = passes_quality_check(image_path)
    if not quality_ok:
        logger.info(
            "[%s] [WIKIMEDIA] Quality REJECTED: %s — %s",
            job_id[:8],
            file_title,
            quality_reason,
        )
        return None

    resolution_score, resolution_reason = score_resolution(image_path)
    quality_score, quality_reason       = score_quality(image_path)

    # Step 2 — relevance gate (filename filter + Gemini Vision)
    relevance_passed, relevance_reason, relevance_score, relevance_details = score_image_candidate(
        image_path=image_path,
        file_title=file_title,
        keyword=keyword,
        topic=topic,
    )

    if not relevance_passed:
        logger.info(
            "[%s] [WIKIMEDIA] Relevance REJECTED: %s — %s",
            job_id[:8],
            file_title,
            relevance_reason,
        )
        return None

    # Combine scores — relevance weighted highest for Wikimedia
    score = round(
        relevance_score * 0.60
        + quality_score  * 0.25
        + resolution_score * 0.15,
        2,
    )

    return {
        "path":             str(image_path),
        "source":           "wikimedia",
        "source_title":     file_title,
        "source_url":       image_url,
        "keyword":          keyword,
        "score":            score,
        "resolution_score": resolution_score,
        "quality_score":    quality_score,
        "vision_score":     relevance_score,
        "reason":           relevance_reason,
        "details": {
            "resolution_reason":  resolution_reason,
            "quality_reason":     quality_reason,
            "relevance_details":  relevance_details,
        },
    }




def _fetch_wikimedia_candidates(
    keyword: str,
    job_id: str,
    visuals_dir: Path,
    all_downloaded: list,
    seen_urls: set,
    max_images: int = WIKIMEDIA_IMAGES_PER_KEYWORD,
    topic: str = "",
) -> list[dict]:
    """Fetch Wikimedia Commons images for one keyword as a fallback source."""
    precise_term = _build_search_term(keyword)
    file_titles = _wikimedia_search(precise_term, limit=SEARCH_RESULTS_PER_KEYWORD)
    logger.info("[%s] Search term: '%s'", job_id[:8], precise_term)
    time.sleep(1.0)

    keyword_images: list[dict] = []

    for title in file_titles:
        if len(keyword_images) >= max_images:
            break

        if not _is_usable_image(title):
            continue

        image_url = _get_image_url(title, width=1080)
        if not image_url or image_url in seen_urls:
            continue

        # Log before downloading so the source is visible in the run output.
        logger.info(
            "[%s] [WIKIMEDIA] Fetching file_title='%s' image_url='%s'",
            job_id[:8],
            title,
            image_url,
        )

        url_clean = image_url.lower().split("?")[0]
        if url_clean.endswith(".png"):
            ext = ".png"
        elif url_clean.endswith(".webp"):
            ext = ".webp"
        else:
            ext = ".jpg"

        global_index = len(all_downloaded) + len(keyword_images) + 1
        output_path = visuals_dir / f"image_{global_index:02d}{ext}"

        # Download the image bytes to disk.
        if not _download_image(image_url, output_path):
            logger.warning("[%s] [WIKIMEDIA] Download failed: %s", job_id[:8], title)
            continue

        # Build the scored candidate object.
        candidate = _score_wikimedia_candidate(
            image_path=output_path,
            file_title=title,
            keyword=keyword,
            job_id=job_id,
            topic=topic,
            image_url=image_url,
        )

        # Remove the file if it failed scoring.
        if candidate is None:
            output_path.unlink(missing_ok=True)
            time.sleep(1.0)
            continue

        # Keep the URL so duplicates are not re-downloaded.
        seen_urls.add(image_url)
        keyword_images.append(candidate)

        logger.info(
            "[%s] [WIKIMEDIA] ACCEPTED %s (score=%.2f) for '%s' (%s)",
            job_id[:8],
            output_path.name,
            candidate["score"],
            keyword,
            title,
        )

        # Pause briefly between downloads.
        time.sleep(0.8)

    return keyword_images


# -----------------------------------------------------------------------------
# Keyword orchestration
# -----------------------------------------------------------------------------


def _fetch_images_for_keyword(
    keyword: str,
    job_id: str,
    visuals_dir: Path,
    all_downloaded: list,
    seen_urls: set,
    max_images: int = 2,
    topic: str = "",
) -> list[dict]:
    """
    Fetch images for one keyword.

    Priority:
      1. TheSportsDB — player cutouts, pre-curated, no relevance check needed
      2. Wikimedia   — ONLY if TheSportsDB returned zero accepted images

    Wikimedia is a last resort, not a supplement.
    """
    # ── Step 1: TheSportsDB (always tried first) ──────────────────────────
    logger.info("[%s] [THESPORTSDB] Trying keyword: '%s'", job_id[:8], keyword)

    sportsdb_images = _fetch_thesportsdb_player_candidates(
        keyword=keyword,
        job_id=job_id,
        visuals_dir=visuals_dir,
        all_downloaded=all_downloaded,
        seen_urls=seen_urls,
        max_images=min(THESPORTSDB_IMAGES_PER_KEYWORD, max_images),
        topic=topic,
    )

    if sportsdb_images:
        logger.info(
            "[%s] [THESPORTSDB] Found %s image(s) for '%s' — Wikimedia not needed",
            job_id[:8],
            len(sportsdb_images),
            keyword,
        )
        return sportsdb_images

    # ── Step 2: Wikimedia (only reached if TheSportsDB found nothing) ─────
    logger.warning(
        "[%s] [THESPORTSDB] Zero images found for '%s' — falling back to Wikimedia",
        job_id[:8],
        keyword,
    )

    wikimedia_images = _fetch_wikimedia_candidates(
        keyword=keyword,
        job_id=job_id,
        visuals_dir=visuals_dir,
        all_downloaded=all_downloaded,
        seen_urls=seen_urls,
        max_images=min(WIKIMEDIA_IMAGES_PER_KEYWORD, max_images),
        topic=topic,
    )

    if wikimedia_images:
        logger.info(
            "[%s] [WIKIMEDIA] Found %s image(s) for '%s'",
            job_id[:8],
            len(wikimedia_images),
            keyword,
        )
    else:
        logger.warning(
            "[%s] [WIKIMEDIA] Also found nothing for '%s'",
            job_id[:8],
            keyword,
        )

    return wikimedia_images


# -----------------------------------------------------------------------------
# Transition helpers
# -----------------------------------------------------------------------------


def _build_transition_sequence(count: int) -> list[str]:
    """Build a randomized transition sequence with no immediate repeats."""
    transition_names = [
        "tornado",
        "slide_left",
        "slide_right",
        "slide_up",
        "slide_down",
        "zoom_in",
        "zoom_out",
        "spin_cw",
        "spin_ccw",
    ]

    sequence: list[str] = []
    previous = None

    for _ in range(count):
        choices = [name for name in transition_names if name != previous]
        picked = random.choice(choices)
        sequence.append(picked)
        previous = picked

    return sequence


def _transition_tornado(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Tornado-style spinning transition."""
    w, h = size
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)

    out_clip = (
        ImageClip(from_arr)
        .set_duration(duration)
        .resize(lambda t: 1.0 + (0.22 * (t / duration)))
        .rotate(lambda t: 540 * (t / duration))
        .set_position(lambda t: (
            -int(0.08 * w * math.sin(6 * math.pi * (t / duration))),
            -int(0.08 * h * math.cos(6 * math.pi * (t / duration))),
        ))
    )

    in_clip = (
        ImageClip(to_arr)
        .set_duration(duration)
        .resize(lambda t: 0.72 + (0.28 * (t / duration)))
        .rotate(lambda t: -540 * (1.0 - (t / duration)))
        .set_position(lambda t: (
            int(0.05 * w * math.sin(6 * math.pi * (1.0 - (t / duration)))),
            int(0.05 * h * math.cos(6 * math.pi * (1.0 - (t / duration)))),
        ))
    )

    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_slide_left(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Horizontal slide transition moving left."""
    w, h = size
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = ImageClip(from_arr).set_duration(duration).set_position(lambda t: (-int(w * (t / duration)), 0))
    in_clip = ImageClip(to_arr).set_duration(duration).set_position(lambda t: (int(w * (1 - (t / duration))), 0))
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_slide_right(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Horizontal slide transition moving right."""
    w, h = size
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = ImageClip(from_arr).set_duration(duration).set_position(lambda t: (int(w * (t / duration)), 0))
    in_clip = ImageClip(to_arr).set_duration(duration).set_position(lambda t: (-int(w * (1 - (t / duration))), 0))
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_slide_up(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Vertical slide transition moving up."""
    w, h = size
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = ImageClip(from_arr).set_duration(duration).set_position(lambda t: (0, -int(h * (t / duration))))
    in_clip = ImageClip(to_arr).set_duration(duration).set_position(lambda t: (0, int(h * (1 - (t / duration)))))
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_slide_down(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Vertical slide transition moving down."""
    w, h = size
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = ImageClip(from_arr).set_duration(duration).set_position(lambda t: (0, int(h * (t / duration))))
    in_clip = ImageClip(to_arr).set_duration(duration).set_position(lambda t: (0, -int(h * (1 - (t / duration)))))
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_zoom_in(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Zoom-in transition with a gentle scale-up."""
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = (
        ImageClip(from_arr)
        .set_duration(duration)
        .resize(lambda t: 1.0 + (0.12 * (t / duration)))
        .set_position("center")
    )
    in_clip = (
        ImageClip(to_arr)
        .set_duration(duration)
        .resize(lambda t: 0.86 + (0.14 * (t / duration)))
        .set_position("center")
    )
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_zoom_out(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Zoom-out transition with a gentle scale-down."""
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = (
        ImageClip(from_arr)
        .set_duration(duration)
        .resize(lambda t: 1.0 - (0.14 * (t / duration)))
        .set_position("center")
    )
    in_clip = (
        ImageClip(to_arr)
        .set_duration(duration)
        .resize(lambda t: 1.1 - (0.1 * (t / duration)))
        .set_position("center")
    )
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_spin_cw(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Clockwise spin transition."""
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = (
        ImageClip(from_arr)
        .set_duration(duration)
        .rotate(lambda t: 360 * (t / duration))
        .resize(lambda t: 1.0 + (0.08 * (t / duration)))
        .set_position("center")
    )
    in_clip = (
        ImageClip(to_arr)
        .set_duration(duration)
        .rotate(lambda t: -180 * (1.0 - (t / duration)))
        .resize(lambda t: 0.9 + (0.1 * (t / duration)))
        .set_position("center")
    )
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


def _transition_spin_ccw(from_arr, to_arr, duration: float, size: tuple[int, int]) -> object:
    """Counterclockwise spin transition."""
    bg = ColorClip(size=size, color=(0, 0, 0)).set_duration(duration)
    out_clip = (
        ImageClip(from_arr)
        .set_duration(duration)
        .rotate(lambda t: -360 * (t / duration))
        .resize(lambda t: 1.0 + (0.08 * (t / duration)))
        .set_position("center")
    )
    in_clip = (
        ImageClip(to_arr)
        .set_duration(duration)
        .rotate(lambda t: 180 * (1.0 - (t / duration)))
        .resize(lambda t: 0.9 + (0.1 * (t / duration)))
        .set_position("center")
    )
    return CompositeVideoClip([bg, out_clip, in_clip], size=size).set_duration(duration)


TRANSITION_BUILDERS = {
    "tornado": _transition_tornado,
    "slide_left": _transition_slide_left,
    "slide_right": _transition_slide_right,
    "slide_up": _transition_slide_up,
    "slide_down": _transition_slide_down,
    "zoom_in": _transition_zoom_in,
    "zoom_out": _transition_zoom_out,
    "spin_cw": _transition_spin_cw,
    "spin_ccw": _transition_spin_ccw,
}


# -----------------------------------------------------------------------------
# Duplicate helper for minimum-image fallback
# -----------------------------------------------------------------------------


def _duplicate_best_image(
    downloaded: list[str],
    visuals_dir: Path,
    needed: int,
) -> list[str]:
    """Duplicate existing images when the pipeline did not find enough files."""
    if not downloaded:
        raise Exception("No images to duplicate — cannot fill minimum.")

    # Start with the original list.
    result = list(downloaded)

    # Use the downloaded list as the source pool.
    source = list(downloaded)

    # Counter used to create unique duplicate names.
    dup_num = 1

    while len(result) < needed:
        # Select a source image in round-robin order.
        src_path = Path(source[(dup_num - 1) % len(source)])
        ext = src_path.suffix

        # Build a new destination file name.
        new_index = len(result) + 1
        dest_path = visuals_dir / f"image_{new_index:02d}_dup{dup_num}{ext}"

        import shutil  # Local import keeps the top-level imports tidy.

        shutil.copy2(str(src_path), str(dest_path))
        result.append(str(dest_path))

        logger.info("Duplicated %s -> %s (dup %s)", src_path.name, dest_path.name, dup_num)
        dup_num += 1

    return result


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def fetch_visuals(scenes: list[dict], job_id: str, topic: str = "") -> list[str]:
    """Fetch, score, and select images for every scene in the job."""
    if not scenes:
        raise Exception("Scenes list cannot be empty.")

    if not job_id or not job_id.strip():
        raise Exception("job_id cannot be empty.")

    # Build the directory that will contain the downloaded visuals for this job.
    visuals_dir = Path(settings.MEDIA_ROOT) / "jobs" / job_id / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Keep track of all selected images across the job.
    all_downloaded: list[str] = []

    # Keep track of URLs we already downloaded so we do not fetch duplicates.
    seen_urls: set[str] = set()

    for scene_index, scene in enumerate(scenes):
        # Stop early once we have reached the target number of visuals.
        if len(all_downloaded) >= TARGET_IMAGES:
            logger.info("[%s] Target of %s images reached — stopping early", job_id[:8], TARGET_IMAGES)
            break

        # Read the main keyword and any backup keywords from the scene.
        main_keyword = scene.get("keyword", "")
        backups = scene.get("backups", [])
        all_keywords = [main_keyword] + backups

        logger.info(
            "[%s] Scene %s/%s: '%s' (+ %s backups)",
            job_id[:8],
            scene_index + 1,
            len(scenes),
            main_keyword,
            len(backups),
        )

        # Hold the best images for the current scene.
        scene_images: list[str] = []

        # Try the main keyword first, then backups until we find something usable.
        for attempt, keyword in enumerate(all_keywords):
            if not keyword.strip():
                continue

            label = "main" if attempt == 0 else f"backup {attempt}"
            logger.info("[%s] Trying %s keyword: '%s'", job_id[:8], label, keyword)

            candidates = _fetch_images_for_keyword(
                keyword=keyword,
                job_id=job_id,
                visuals_dir=visuals_dir,
                all_downloaded=all_downloaded,
                seen_urls=seen_urls,
                max_images=VISUAL_COUNT,
                topic=topic,
            )

            # If the current keyword produced candidates, keep the top 3.
            if candidates:
                candidates.sort(key=lambda item: item["score"], reverse=True)
                top_candidates = candidates[:3]

                # Persist the ranking information for debugging and traceability.
                _save_scene_candidates(visuals_dir, scene_index, keyword, candidates)

                # Store only the top candidates from this keyword.
                scene_images.extend([candidate["path"] for candidate in top_candidates])

                # Log which source each selected candidate came from.
                for candidate in top_candidates:
                    logger.info(
                        "[%s] Scene %s selected image from %s (score=%.2f)",
                        job_id[:8],
                        scene_index + 1,
                        candidate.get("source", "unknown"),
                        candidate["score"],
                    )

                logger.info(
                    "[%s] '%s' -> %s candidate(s) collected, %s top selected",
                    job_id[:8],
                    keyword,
                    len(candidates),
                    len(top_candidates),
                )
                break

            # Move on to the next keyword if this one returned nothing.
            logger.warning("[%s] '%s' returned nothing — trying next backup", job_id[:8], keyword)

        # Append selected images for this scene to the full job list.
        if scene_images:
            all_downloaded.extend(scene_images)
        else:
            logger.warning(
                "[%s] Scene %s exhausted all keywords (main + %s backups) — no images found",
                job_id[:8],
                scene_index + 1,
                len(backups),
            )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    total = len(all_downloaded)
    logger.info("[%s] Total images downloaded: %s", job_id[:8], total)

    if total < MIN_IMAGES:
        raise Exception(
            f"Only {total} images downloaded — minimum is {MIN_IMAGES}. "
            f"All keywords and backups were exhausted. "
            f"Try a more popular topic or check your internet connection."
        )

    if total < TARGET_IMAGES:
        logger.warning(
            "[%s] Got %s images — below target of %s but above minimum of %s. Proceeding.",
            job_id[:8],
            total,
            TARGET_IMAGES,
            MIN_IMAGES,
        )
    else:
        logger.info(
            "[%s] Got %s images — within target range. Ready for assembly.",
            job_id[:8],
            total,
        )

    # ── Source summary ────────────────────────────────────────────────────
    # Count how many images came from each source
    sportsdb_count = 0
    wikimedia_count = 0

    for img_path in all_downloaded:
        meta_files = list(visuals_dir.glob("scene_*_candidates.json"))
        for meta_file in meta_files:
            try:
                with meta_file.open(encoding="utf-8") as f:
                    meta = json.load(f)
                for candidate in meta.get("top_candidates", []):
                    if candidate.get("path") == img_path:
                        if candidate.get("source") == "thesportsdb":
                            sportsdb_count += 1
                        else:
                            wikimedia_count += 1
            except Exception:
                pass

    logger.info(
        "[%s] Source breakdown: TheSportsDB=%s Wikimedia=%s Total=%s",
        job_id[:8],
        sportsdb_count,
        wikimedia_count,
        total,
    )

    return all_downloaded


# -----------------------------------------------------------------------------
# Slideshow builder with dynamic transitions
# -----------------------------------------------------------------------------


def _build_slideshow_from_scene_groups(
    scene_groups: list[list[str]],
    scenes: list[dict],
    total_duration: float,
) -> object:
    """Build the slideshow while inserting randomized non-repeating transitions."""
    group_count = sum(len(group) for group in scene_groups)
    if group_count == 0:
        raise Exception("No candidate images available to build video.")

    durations = []
    for scene, group in zip(scenes, scene_groups):
        if not group:
            raise Exception(
                f"Scene '{scene.get('keyword', 'unknown')}' has no candidate images."
            )
        scene_duration = scene.get("duration", total_duration / len(scenes))
        per_image_duration = scene_duration / len(group)
        durations.extend([per_image_duration] * len(group))

    flat_paths = [path for group in scene_groups for path in group]
    if not flat_paths:
        raise Exception("No image paths available for slideshow construction.")

    image_count = len(flat_paths)
    transition_count = max(0, image_count - 1)

    # Choose a transition duration that keeps the final timeline close to total_duration.
    if transition_count > 0:
        max_allowed = max(MIN_TRANSITION_DURATION, (total_duration * 0.30) / transition_count)
        transition_duration = min(TRANSITION_DURATION, MAX_TRANSITION_DURATION, max_allowed)
    else:
        transition_duration = TRANSITION_DURATION

    transition_total = transition_count * transition_duration
    available_image_duration = max(total_duration - transition_total, 0.5)

    # Normalize image durations so the image timeline plus transitions matches the audio.
    duration_sum = sum(durations)
    if duration_sum <= 0:
        raise Exception("Invalid duration plan for slideshow construction.")
    durations = [d * (available_image_duration / duration_sum) for d in durations]

    # Count images first, then build a transition plan for every gap.
    transition_plan = _build_transition_sequence(transition_count)

    clips = []
    size = None

    for i, (path, dur) in enumerate(zip(flat_paths, durations)):
        arr = _prepare_image(path)

        if size is None:
            size = (arr.shape[1], arr.shape[0])

        is_dup = "_dup" in Path(path).stem
        direction = DIRECTIONS[i % len(DIRECTIONS)]

        clip = _ken_burns_clip(arr, dur, direction=direction)
        clips.append(clip)

        logger.info(
            "  Scene image %s: %.2fs — %s [%s]%s",
            i + 1,
            dur,
            Path(path).name,
            direction,
            " (duplicate)" if is_dup else "",
        )

        # Insert one transition between each pair of images, using a different
        # transition type each time without repeating the same one back-to-back.
        if i < image_count - 1:
            next_path = flat_paths[i + 1]
            next_arr = _prepare_image(next_path)
            transition_name = transition_plan[i]
            transition_builder = TRANSITION_BUILDERS[transition_name]

            transition_clip = transition_builder(
                from_arr=arr,
                to_arr=next_arr,
                duration=transition_duration,
                size=size,
            )
            clips.append(transition_clip)

            logger.info(
                "  Transition %s -> %s: %s (%.2fs)",
                Path(path).name,
                Path(next_path).name,
                transition_name,
                transition_duration,
            )

    return concatenate_videoclips(clips, method="compose")


# -----------------------------------------------------------------------------
# Module guard
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # This module is meant to be imported by the pipeline, not executed directly.
    print("Do not run visuals.py directly.")
    print("Use: python test_visuals.py <keywords> --job-id <id> --topic <topic>")
    sys.exit(1)


# End of file.



# End of image collection and filtering logic.
# This module tries keywords and backups in order, downloads image files,
# and rejects images that fail quality or relevance checks.
# Variable reference table:
# variable_name | type | purpose
# WIKIMEDIA_API | str | Wikimedia Commons API base URL.
# IMAGES_PER_KEYWORD | int | How many images are attempted per keyword.
# SEARCH_RESULTS_PER_KEYWORD | int | Number of search results fetched per keyword.
# VISUAL_COUNT | int | Maximum images the pipeline will consider.
# MIN_IMAGES | int | Minimum images required for a valid video.
# TARGET_IMAGES | int | Ideal image count to build a good short.
# _wikimedia_request | func | Handles retries for Wikimedia API calls.
# _wikimedia_search | func | Searches Wikimedia Commons using a keyword.
# _get_image_url | func | Resolves a file title to an image URL.
# _download_image | func | Downloads image bytes to disk.
# _is_usable_image | func | Rejects unsupported media file types.
# _is_well_framed | func | Optional framing and brightness validation.
# _save_scene_candidates | func | Writes scene candidate metadata to JSON.
# _fetch_images_for_keyword | func | Collects and scores images for one keyword.
# _duplicate_best_image | func | Duplicates existing images when there are too few.
# fetch_visuals | func | Main image acquisition flow for all scenes.



# Final variable reference table at EOF:
# variable_name | type | purpose
# logger | logging.Logger | Pipeline logger for visuals downloading and scoring.
# WIKIMEDIA_API | str | Wikimedia Commons API base URL.
# IMAGES_PER_KEYWORD | int | Images fetched per keyword attempt.
# SEARCH_RESULTS_PER_KEYWORD | int | Number of search results pages requested.
# MIN_IMAGES | int | Lower bound for acceptable images to assemble a video.
# TARGET_IMAGES | int | Preferred number of images to collect for a job.
# fetch_visuals | func | Top-level function called by the content pipeline.