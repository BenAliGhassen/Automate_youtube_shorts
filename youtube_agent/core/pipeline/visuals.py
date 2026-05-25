"""Pexels image fetching for YouTube Shorts visuals."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from django.conf import settings

logger = logging.getLogger(__name__)

PEXELS_SEARCH_ENDPOINT = "https://api.pexels.com/v1/search"
VISUAL_COUNT = 5


def _build_http_opener():
    http_proxy = getattr(settings, "OUTBOUND_HTTP_PROXY", "").strip()
    https_proxy = getattr(settings, "OUTBOUND_HTTPS_PROXY", "").strip()

    proxies: dict[str, str] = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

    # Bypass broken system proxy variables unless the Django settings opt in.
    return build_opener(ProxyHandler(proxies))


def _download_bytes(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
    opener = _build_http_opener()
    request = Request(url=url, headers=headers, method="GET")
    with opener.open(request, timeout=60) as response:
        content_type = response.info().get_content_type()
        return response.read(), content_type


def _extension_for_content_type(content_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(content_type.lower(), ".jpg")


def fetch_visuals(topic: str, job_id: str) -> list[str]:
    """Download five Pexels images for the topic and return their full local paths."""
    if not topic or not topic.strip():
        raise Exception("Topic cannot be empty for visuals fetching.")
    if not job_id or not job_id.strip():
        raise Exception("job_id cannot be empty for visuals fetching.")

    api_key = getattr(settings, "PEXELS_API_KEY", "").strip()
    if not api_key:
        raise Exception("PEXELS_API_KEY is not configured in Django settings.")

    visuals_dir = Path(settings.MEDIA_ROOT) / "jobs" / job_id / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": api_key,
        "User-Agent": "youtube-agent/1.0",
    }
    query_string = urlencode(
        {
            "query": topic,
            "per_page": 15,
            "page": 1,
            "orientation": "portrait",
        }
    )
    search_url = f"{PEXELS_SEARCH_ENDPOINT}?{query_string}"

    logger.info("[%s] Fetching visuals for topic: %s", job_id[:8], topic)

    try:
        opener = _build_http_opener()
        request = Request(url=search_url, headers=headers, method="GET")
        with opener.open(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.error("[%s] Pexels API request failed with status %s", job_id[:8], exc.code)
        raise Exception(
            f"Pexels API request failed with status {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        logger.error("[%s] Could not reach the Pexels API: %s", job_id[:8], exc.reason)
        raise Exception(f"Could not reach the Pexels API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        logger.error("[%s] Pexels API returned invalid JSON.", job_id[:8])
        raise Exception("Pexels API returned invalid JSON.") from exc

    photos = payload.get("photos") or []
    if not photos:
        raise Exception(f"Pexels returned no photos for topic '{topic}'.")

    downloaded_files: list[str] = []
    for photo in photos:
        src = photo.get("src") or {}
        image_url = src.get("medium")
        if not image_url:
            continue

        file_index = len(downloaded_files) + 1
        try:
            image_bytes, content_type = _download_bytes(image_url, headers)
        except HTTPError as exc:
            logger.warning(
                "[%s] Skipping Pexels image %s after status %s",
                job_id[:8],
                image_url,
                exc.code,
            )
            continue
        except URLError as exc:
            logger.warning(
                "[%s] Skipping Pexels image %s because it could not be downloaded: %s",
                job_id[:8],
                image_url,
                exc.reason,
            )
            continue

        if not image_bytes:
            logger.warning("[%s] Skipping empty Pexels image response: %s", job_id[:8], image_url)
            continue

        extension = _extension_for_content_type(content_type)
        output_path = visuals_dir / f"image_{file_index:02d}{extension}"
        output_path.write_bytes(image_bytes)
        downloaded_files.append(str(output_path))
        logger.info("[%s] Saved visual %s to %s", job_id[:8], file_index, output_path)

        if len(downloaded_files) == VISUAL_COUNT:
            break

    if not downloaded_files:
        raise Exception(f"Pexels returned no usable images for topic '{topic}'.")
    if len(downloaded_files) < VISUAL_COUNT:
        raise Exception(
            f"Pexels returned only {len(downloaded_files)} usable images for topic '{topic}'."
        )

    logger.info("[%s] Downloaded %s visuals", job_id[:8], len(downloaded_files))
    return downloaded_files


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

    parser = argparse.ArgumentParser(description="Download Pexels visuals for a topic.")
    parser.add_argument("topic", help="Topic to search on Pexels.")
    parser.add_argument(
        "--job-id",
        default="manual-test",
        help="Job ID used to build the media output path.",
    )
    args = parser.parse_args()

    try:
        visual_paths = fetch_visuals(args.topic, args.job_id)
    except Exception:
        logger.exception("Standalone visuals fetch failed.")
        raise SystemExit(1)

    logger.info("Downloaded %s visuals", len(visual_paths))
    for visual_path in visual_paths:
        logger.info("Visual file: %s", visual_path)
