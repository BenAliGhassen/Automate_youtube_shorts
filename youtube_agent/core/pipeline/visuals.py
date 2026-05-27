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

from django.conf import settings

logger = logging.getLogger(__name__)

WIKIMEDIA_API  = "https://commons.wikimedia.org/w/api.php"
IMAGES_PER_KEYWORD = 2   # images fetched per keyword
VISUAL_COUNT   = 10      # total max images to download


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
        logger.info("Wikimedia search '%s' → %s results", keyword, len(titles))
        time.sleep(1.0)   # ← always wait 1s after a search
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


def fetch_visuals(keywords: list[str], job_id: str) -> list[str]:
    """
    Search Wikimedia Commons for each keyword in order and download matching images.
    Keywords must be ordered to match the script narrative — images will be saved
    in the same order so the assembler displays them in sync with the voiceover.
    Returns a list of local file paths in narrative order.
    """
    if not keywords:
        raise Exception("Keywords list cannot be empty.")
    if not job_id or not job_id.strip():
        raise Exception("job_id cannot be empty.")

    visuals_dir = Path(settings.MEDIA_ROOT) / "jobs" / job_id / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Fetching ordered visuals for %s keywords: %s", job_id[:8], len(keywords), keywords)

    all_downloaded = []   # final ordered list of local paths
    seen_urls      = set()

    for keyword_index, keyword in enumerate(keywords):
        logger.info("[%s] Processing keyword %s/%s: '%s'", job_id[:8], keyword_index + 1, len(keywords), keyword)

        search_term = f"{keyword} football"
        file_titles = _wikimedia_search(search_term, limit=8)
        time.sleep(2.0)   # wait between keyword searches

        keyword_images = []   # images collected for this keyword only

        for title in file_titles:
            # Stop once we have enough images for this keyword
            if len(keyword_images) >= IMAGES_PER_KEYWORD:
                break

            # Skip non-image files (SVG, PDF, video, etc.)
            if not _is_usable_image(title):
                logger.debug("[%s] Skipping non-image: %s", job_id[:8], title)
                continue

            # Get the direct download URL
            image_url = _get_image_url(title, width=1080)

            # Skip if URL lookup failed or already downloaded this image
            if not image_url:
                continue
            if image_url in seen_urls:
                logger.debug("[%s] Skipping duplicate URL: %s", job_id[:8], image_url)
                continue

            # Determine file extension from URL
            url_clean = image_url.lower().split("?")[0]
            if url_clean.endswith(".png"):
                ext = ".png"
            elif url_clean.endswith(".webp"):
                ext = ".webp"
            else:
                ext = ".jpg"

            # Build output path — index is global across all keywords
            # so files are named image_01, image_02... in narrative order
            global_index = len(all_downloaded) + len(keyword_images) + 1
            output_path  = visuals_dir / f"image_{global_index:02d}{ext}"

            # Download immediately — don't collect all URLs first
            # because downloading in order guarantees narrative sync
            success = _download_image(image_url, output_path)
            time.sleep(1.5)   # wait between downloads

            if success:
                seen_urls.add(image_url)
                keyword_images.append(str(output_path))
                logger.info(
                    "[%s] Saved image_%02d for '%s' — %s",
                    job_id[:8], global_index, keyword, title
                )
            else:
                logger.warning(
                    "[%s] Download failed for '%s' image: %s",
                    job_id[:8], keyword, title
                )

        # Log result for this keyword
        if keyword_images:
            logger.info(
                "[%s] Keyword '%s' → %s image(s) saved",
                job_id[:8], keyword, len(keyword_images)
            )
            all_downloaded.extend(keyword_images)
        else:
            logger.warning(
                "[%s] No images found for keyword '%s' — this section will reuse adjacent images",
                job_id[:8], keyword
            )

    # Final validation
    if not all_downloaded:
        raise Exception(
            f"Could not download any images for keywords: {keywords}. "
            "Check your internet connection or try different keywords."
        )

    if len(all_downloaded) < len(keywords):
        logger.warning(
            "[%s] Got %s images for %s keywords — some keywords had no results",
            job_id[:8], len(all_downloaded), len(keywords)
        )

    logger.info(
        "[%s] Done — %s ordered visuals ready for assembly",
        job_id[:8], len(all_downloaded)
    )
    return all_downloaded