"""Wikimedia Commons image fetching — works for football AND tech channels."""
import time
import json
import logging
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from .image_quality    import passes_quality_check, score_quality, score_resolution
from .image_relevance  import passes_relevance_check, score_image_candidate
from django.conf import settings

logging.getLogger("core.pipeline.image_quality").setLevel(logging.DEBUG)
logging.getLogger("core.pipeline.image_relevance").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Wikimedia Commons API constants and visual collection limits.
WIKIMEDIA_API  = "https://commons.wikimedia.org/w/api.php"
IMAGES_PER_KEYWORD = 7   # images fetched per keyword
SEARCH_RESULTS_PER_KEYWORD = 30
VISUAL_COUNT   = 12     # total max images to download
MIN_IMAGES    = 4     # minimum to proceed
TARGET_IMAGES = 7    # sweet spot upper bound

def _wikimedia_request(url: str, retries: int = 3) -> dict:
    """
    Make a Wikimedia API request with automatic retry on 429.
    Waits longer after each failure (exponential backoff).
    """
    for attempt in range(retries):
        try:
            req = Request(
                url=url,
                headers={"User-Agent": "youtube-agent/1.0 (educational project)"},
                method="GET"
            )
            with urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * 2   # 2s, 4s, 8s
                logger.warning("Rate limited (429) — waiting %ss before retry %s/%s", wait, attempt + 1, retries)
                time.sleep(wait)
            else:
                raise
        except Exception as exc:
            raise

    raise Exception(f"Wikimedia request failed after {retries} retries: {url}")



def _wikimedia_search(keyword: str, limit: int = 5) -> list[str]:
    """Search Wikimedia Commons and return file titles for the keyword."""
    params = urlencode({
        "action":      "query",
        "list":        "search",
        "srsearch":    keyword,
        "srnamespace": 6,
        "srlimit":     limit,
        "format":      "json",
        "utf8":        1,
    })
    url = f"{WIKIMEDIA_API}?{params}"

    try:
        data    = _wikimedia_request(url)
        results = data.get("query", {}).get("search", [])
        titles  = [r["title"] for r in results if r.get("title")]
        logger.info("Wikimedia search '%s' -> %s results", keyword, len(titles))
        time.sleep(1.0)   # always wait 1s after a search 
        return titles
    except Exception as exc:
        logger.warning("Wikimedia search failed for '%s': %s", keyword, exc)
        return []


def _get_image_url(file_title: str, width: int = 1080) -> str | None:
    params = urlencode({
        "action":     "query",
        "titles":     file_title,
        "prop":       "imageinfo",
        "iiprop":     "url",
        "iiurlwidth": width,
        "format":     "json",
        "utf8":       1,
    })
    url = f"{WIKIMEDIA_API}?{params}"

    try:
        data  = _wikimedia_request(url)
        pages = data.get("query", {}).get("pages", {})
        time.sleep(0.8)   # ← wait after every URL lookup
        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                return imageinfo[0].get("thumburl") or imageinfo[0].get("url")
        return None
    except Exception as exc:
        logger.warning("Could not get URL for '%s': %s", file_title, exc)
        return None


def _download_image(url: str, output_path: Path, retries: int = 3) -> bool:
    """Download an image with retry on 429."""
    for attempt in range(retries):
        try:
            req = Request(
                url=url,
                headers={"User-Agent": "youtube-agent/1.0 (educational project)"},
                method="GET"
            )
            with urlopen(req, timeout=30) as response:
                data = response.read()
            if not data:
                return False
            output_path.write_bytes(data)
            return True

        except HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * 3   # 3s, 6s, 12s
                logger.warning("Download rate limited — waiting %ss (attempt %s/%s)", wait, attempt + 1, retries)
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
    """
    Filter out SVGs and non-image files — MoviePy can't use them directly.
    Pillow can handle SVG conversion but it's extra complexity for now.
    """
    lower = file_title.lower()
    # Skip SVGs — they need special conversion
    if lower.endswith(".svg"):
        return False
    # Skip audio/video files that end up in file namespace
    for ext in [".ogg", ".ogv", ".webm", ".pdf", ".tif", ".tiff"]:
        if lower.endswith(ext):
            return False
    return True


def _is_well_framed(image_path: Path) -> bool:
    """
    Verify the image is usable after being cropped to 9:16.
    Rejects images that are:
    - Too zoomed in (subject fills >85% of frame — likely a closeup crop)
    - Too dark (average brightness < 30 — probably black)
    - Too uniform (nearly solid color — logo or blank background only)
    """
    try:
        from PIL import Image, ImageStat
        import numpy as np

        img  = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)

        # Check 1 — brightness (mean of all channels)
        brightness = sum(stat.mean) / 3
        if brightness < 30:
            logger.warning("Rejected dark image: %s (brightness=%.1f)", image_path.name, brightness)
            return False

        # Check 2 — uniformity (low stddev = solid color = logo/blank)
        stddev = sum(stat.stddev) / 3
        if stddev < 15:
            logger.warning("Rejected uniform image: %s (stddev=%.1f)", image_path.name, stddev)
            return False

        # Check 3 — aspect ratio sanity
        # After our 9:16 crop, check the original wasn't absurdly wide
        # (very wide images lose too much content when cropped to portrait)
        original_ratio = img.width / img.height
        if original_ratio > 3.0:
            logger.warning("Rejected panoramic image: %s (ratio=%.2f)", image_path.name, original_ratio)
            return False

        return True

    except Exception as exc:
        logger.warning("Could not verify image %s: %s", image_path.name, exc)
        return True  # don't reject if we can't check


# Competition keywords that need "men" qualifier
COMPETITION_KEYWORDS = [
    "champions league", "premier league", "la liga",
    "serie a", "bundesliga", "ligue 1", "world cup",
    "euro", "copa del rey", "fa cup", "league cup",
    "europa league", "conference league",
]

def _build_search_term(keyword: str) -> str:
    """
    Build a precise Wikimedia search term.
    Adds 'men' qualifier for competitions to avoid women's results.
    """
    keyword_lower = keyword.lower()

    for comp in COMPETITION_KEYWORDS:
        if comp in keyword_lower:
            return f"{keyword} men final"

    return keyword


def _save_scene_candidates(
    visuals_dir: Path,
    scene_index: int,
    keyword: str,
    candidates: list[dict],
) -> None:
    """
    Save candidate ranking metadata for a scene.
    """
    metadata_path = visuals_dir / f"scene_{scene_index + 1}_candidates.json"
    metadata = {
        "scene_index": scene_index + 1,
        "keyword": keyword,
        "top_candidates": [
            {
                "path": candidate["path"],
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
    Try to fetch up to max_images for a single keyword.
    Returns a list of scored candidate dictionaries.
    """
    precise_term = _build_search_term(keyword)
    file_titles  = _wikimedia_search(precise_term, limit=SEARCH_RESULTS_PER_KEYWORD)
    logger.info("[%s] Search term: '%s'", job_id[:8], precise_term)
    time.sleep(2.0)

    keyword_images = []

    for title in file_titles:
        if len(keyword_images) >= max_images:
            break
        if not _is_usable_image(title):
            continue

        image_url = _get_image_url(title, width=1080)
        if not image_url or image_url in seen_urls:
            continue

        url_clean = image_url.lower().split("?")[0]
        if url_clean.endswith(".png"):   ext = ".png"
        elif url_clean.endswith(".webp"): ext = ".webp"
        else:                             ext = ".jpg"

        global_index = len(all_downloaded) + len(keyword_images) + 1
        output_path  = visuals_dir / f"image_{global_index:02d}{ext}"

        if _download_image(image_url, output_path):
            quality_ok, quality_reason = passes_quality_check(output_path)
            if not quality_ok:
                output_path.unlink(missing_ok=True)
                logger.info(
                    "[%s] Quality REJECTED: %s — %s",
                    job_id[:8], title, quality_reason
                )
                time.sleep(1.0)
                continue

            resolution_score, resolution_reason = score_resolution(output_path)
            quality_score, quality_reason = score_quality(output_path)
            relevance_passed, relevance_reason, relevance_score, relevance_details = score_image_candidate(
                image_path=output_path,
                file_title=title,
                keyword=keyword,
                topic=topic,
            )
            if not relevance_passed:
                output_path.unlink(missing_ok=True)
                logger.info(
                    "[%s] Relevance REJECTED: %s — %s",
                    job_id[:8], title, relevance_reason
                )
                time.sleep(1.0)
                continue

            score = round(
                relevance_score * 0.60
                + quality_score * 0.25
                + resolution_score * 0.15,
                2,
            )

            seen_urls.add(image_url)
            candidate = {
                "path": str(output_path),
                "source_title": title,
                "keyword": keyword,
                "score": score,
                "resolution_score": resolution_score,
                "quality_score": quality_score,
                "vision_score": relevance_score,
                "reason": relevance_reason,
                "details": {
                    "resolution_reason": resolution_reason,
                    "quality_reason": quality_reason,
                    "relevance_details": relevance_details,
                },
            }
            keyword_images.append(candidate)
            logger.info(
                "[%s] ACCEPTED %s (score=%.2f) for '%s'",
                job_id[:8], output_path.name, score, keyword,
            )
        else:
            logger.warning("[%s] Download failed: %s", job_id[:8], title)

        time.sleep(1.5)

    return keyword_images


def _duplicate_best_image(
    downloaded: list[str],
    visuals_dir: Path,
    needed: int,
) -> list[str]:
    """
    When not enough images were found, duplicate the best existing ones
    with a suffix so the assembler treats them as separate clips.
    Each duplicate will get a different Ken Burns direction in the assembler.
    Returns the full updated list including duplicates.
    """
    if not downloaded:
        raise Exception("No images to duplicate — cannot fill minimum.")

    result   = list(downloaded)
    source   = list(downloaded)   # pool to duplicate from
    dup_num  = 1

    while len(result) < needed:
        # Pick source image in round-robin order
        src_path = Path(source[(dup_num - 1) % len(source)])
        ext      = src_path.suffix

        # New index continues from last downloaded
        new_index  = len(result) + 1
        dest_path  = visuals_dir / f"image_{new_index:02d}_dup{dup_num}{ext}"

        import shutil
        shutil.copy2(str(src_path), str(dest_path))

        result.append(str(dest_path))
        logger.info(
            "Duplicated %s -> %s (dup %s)",
            src_path.name, dest_path.name, dup_num
        )
        dup_num += 1

    return result


def fetch_visuals(scenes: list[dict], job_id: str, topic: str = "") -> list[str]:
    """
    For each scene, try the main keyword then backups until enough images are found.
    Downloads up to IMAGES_PER_KEYWORD candidates per scene, scores them, and selects
    the top 3 candidates per scene. Candidate metadata for each scene is persisted
    as JSON under the visuals directory.

    Minimum 5 images required — raises Exception if not met.
    Target is 6-7 images — stops early if TARGET_IMAGES reached.
    """
    if not scenes:
        raise Exception("Scenes list cannot be empty.")
    if not job_id or not job_id.strip():
        raise Exception("job_id cannot be empty.")

    visuals_dir = Path(settings.MEDIA_ROOT) / "jobs" / job_id / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    all_downloaded = []
    seen_urls      = set()

    for scene_index, scene in enumerate(scenes):
        # Stop if we already hit the target
        if len(all_downloaded) >= TARGET_IMAGES:
            logger.info("[%s] Target of %s images reached — stopping early", job_id[:8], TARGET_IMAGES)
            break

        main_keyword = scene.get("keyword", "")
        backups      = scene.get("backups", [])
        all_keywords = [main_keyword] + backups   # main first, then 3 backups

        logger.info(
            "[%s] Scene %s/%s: '%s' (+ %s backups)",
            job_id[:8], scene_index + 1, len(scenes), main_keyword, len(backups)
        )

        scene_images = []

        # Try main keyword then each backup until we get at least 1 image
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
                max_images=IMAGES_PER_KEYWORD,
                topic=topic,
            )

            if candidates:
                candidates.sort(key=lambda item: item["score"], reverse=True)
                top_candidates = candidates[:3]
                _save_scene_candidates(visuals_dir, scene_index, keyword, candidates)

                scene_images.extend([candidate["path"] for candidate in top_candidates])
                logger.info(
                    "[%s] '%s' -> %s candidate(s) collected, %s top selected",
                    job_id[:8], keyword, len(candidates), len(top_candidates)
                )
                break   # got images — no need to try more backups
            else:
                logger.warning(
                    "[%s] '%s' returned nothing — trying next backup",
                    job_id[:8], keyword
                )

        if scene_images:
            all_downloaded.extend(scene_images)
        else:
            logger.warning(
                "[%s] Scene %s exhausted all keywords (main + %s backups) — no images found",
                job_id[:8], scene_index + 1, len(backups)
            )

    # ── Final validation ──────────────────────────────────────────────────────
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
            job_id[:8], total, TARGET_IMAGES, MIN_IMAGES
        )
    else:
        logger.info(
            "[%s] Got %s images — within target range. Ready for assembly.",
            job_id[:8], total
        )

    return all_downloaded


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





if __name__ == "__main__":
    print("Do not run visuals.py directly.")
    print("Use: python test_visuals.py <keywords> --job-id <id> --topic <topic>")
    sys.exit(1)

# Final variable reference table at EOF:
# variable_name | type | purpose
# logger | logging.Logger | Pipeline logger for visuals downloading and scoring.
# WIKIMEDIA_API | str | Wikimedia Commons API base URL.
# IMAGES_PER_KEYWORD | int | Images fetched per keyword attempt.
# SEARCH_RESULTS_PER_KEYWORD | int | Number of search results pages requested.
# MIN_IMAGES | int | Lower bound for acceptable images to assemble a video.
# TARGET_IMAGES | int | Preferred number of images to collect for a job.
# fetch_visuals | func | Top-level function called by the content pipeline.


    