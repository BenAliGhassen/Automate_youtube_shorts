"""
Module 1 — Resolution and quality filter.
Runs locally — no API call needed.
Rejects images that are too small, too dark, or too uniform to use in a Short.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum dimensions to be usable in a 1080x1920 Short
MIN_WIDTH  = 400   # px
MIN_HEIGHT = 300   # px

# Ideal minimum for clean 9:16 crop without upscaling artifacts
IDEAL_MIN_SIDE = 600   # px


def check_resolution(image_path: Path) -> tuple[bool, str]:
    """
    Check if image dimensions are large enough for a Short.
    Returns (passed, reason).
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        img.close()
    except Exception as exc:
        return False, f"Could not read image dimensions: {exc}"

    # Hard reject — too small to use at all
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False, f"Too small: {w}x{h}px (min {MIN_WIDTH}x{MIN_HEIGHT})"

    # Warn but allow — will upscale slightly
    if min(w, h) < IDEAL_MIN_SIDE:
        return True, f"Acceptable but small: {w}x{h}px (ideal min side {IDEAL_MIN_SIDE}px)"

    return True, f"Resolution OK: {w}x{h}px"


def check_quality(image_path: Path) -> tuple[bool, str]:
    """
    Check image brightness and content variance.
    Rejects blank, black, or solid-color images.
    Returns (passed, reason).
    """
    try:
        from PIL import Image, ImageStat
        img  = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        img.close()
    except Exception as exc:
        return False, f"Could not analyze image: {exc}"

    brightness = sum(stat.mean) / 3
    stddev     = sum(stat.stddev) / 3

    if brightness < 20:
        return False, f"Image too dark (brightness={brightness:.1f})"

    if brightness > 250:
        return False, f"Image too bright/blank (brightness={brightness:.1f})"

    if stddev < 12:
        return False, f"Image too uniform — likely icon or solid color (stddev={stddev:.1f})"

    return True, f"Quality OK (brightness={brightness:.1f}, variance={stddev:.1f})"


def check_aspect_ratio(image_path: Path) -> tuple[bool, str]:
    """
    Reject images that are too wide (panoramic) or too tall (banner).
    Very wide images lose most content when cropped to 9:16.
    Returns (passed, reason).
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        img.close()
    except Exception as exc:
        return False, f"Could not read image: {exc}"

    ratio = w / h

    if ratio > 4.0:
        return False, f"Too panoramic: {w}x{h} (ratio={ratio:.2f}) — loses too much in crop"

    if ratio < 0.2:
        return False, f"Too narrow/tall: {w}x{h} (ratio={ratio:.2f})"

    return True, f"Aspect ratio OK ({ratio:.2f})"


def passes_quality_check(image_path: Path) -> tuple[bool, str]:
    """
    Run all local quality checks on a downloaded image.
    Returns (passed, reason).
    All checks must pass for the image to be accepted.
    """
    checks = [
        check_resolution,
        check_quality,
        check_aspect_ratio,
    ]

    for check in checks:
        passed, reason = check(image_path)
        if not passed:
            logger.warning(
                "Quality check FAILED [%s]: %s — %s",
                check.__name__, image_path.name, reason
            )
            return False, reason

    logger.debug("Quality check PASSED: %s", image_path.name)
    return True, "All quality checks passed"