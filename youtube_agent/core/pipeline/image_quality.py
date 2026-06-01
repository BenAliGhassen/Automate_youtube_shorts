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


def score_resolution(image_path: Path) -> tuple[float, str]:
    """
    Score image resolution from 0.0 to 100.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        img.close()
    except Exception as exc:
        return 0.0, f"Could not read image dimensions: {exc}"

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return 0.0, f"Too small: {w}x{h}px"

    side = min(w, h)
    score = min(1.0, side / IDEAL_MIN_SIDE)
    return round(score * 100.0, 1), f"Resolution score: {w}x{h}px"


def score_quality(image_path: Path) -> tuple[float, str]:
    """
    Score image quality from 0.0 to 100 based on brightness and variance.
    """
    try:
        from PIL import Image, ImageStat
        img = Image.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        img.close()
    except Exception as exc:
        return 0.0, f"Could not analyze image: {exc}"

    brightness = sum(stat.mean) / 3.0
    stddev = sum(stat.stddev) / 3.0

    brightness_score = min(max((brightness - 20.0) / 210.0, 0.0), 1.0)
    variance_score = min(max((stddev - 12.0) / 88.0, 0.0), 1.0)
    score = (brightness_score * 0.55) + (variance_score * 0.45)
    return round(score * 100.0, 1), f"Brightness={brightness:.1f}, variance={stddev:.1f}"


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


# Final variable reference table at EOF:
# variable_name | type | purpose
# MIN_WIDTH | int | Minimum allowed image width for short assembly.
# MIN_HEIGHT | int | Minimum allowed image height for short assembly.
# IDEAL_MIN_SIDE | int | Ideal minimum shorter side length for good cropping.
# check_resolution | func | Validate image dimensions against minimum thresholds.
# check_quality | func | Validate brightness and variance to reject blank/dark images.
# check_aspect_ratio | func | Validate that the image is not too panoramic or too tall.
# score_resolution | func | Score the image resolution from 0–100.
# score_quality | func | Score image brightness and texture quality from 0–100.
# passes_quality_check | func | Aggregate all local checks and return pass/fail status.
