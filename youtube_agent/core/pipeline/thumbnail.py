"""
Thumbnail generator for YouTube Shorts hook frame.
Layout:
  - Red/black dramatic gradient background
  - Bold hook text at the top
  - Player image (from TheSportsDB) centered at the bottom
  - Fallback: Pillow-only card if player image unavailable

Pipeline:
  1. Extract player name from topic/scenes
  2. Fetch player cutout from TheSportsDB
  3. Composite player over gradient background
  4. Add hook text at top
  5. Fallback to pure Pillow card if any step fails
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

WIDTH  = 1080
HEIGHT = 1920

SPORTSDB_API = "https://www.thesportsdb.com/api/v1/json/3"

# ── Player/Country/Club/Topic mappings ───────────────────────────────────────

PLAYER_IDS = {
    "messi":       34146370,
    "ronaldo":     34146304,
    "mbappe":      34162098,
    "neymar":      34146371,
    "haaland":     34169116,
    "benzema":     34146309,
    "modric":      34146306,
    "salah":       34145506,
    "lewandowski": 34146705,
    "vinicius":    34161324,
    "bellingham":  34171882,
    "pedri":       34172243,
    "yamal":       34219490,
    "griezmann":   34159231,
    "kane":        34146220,
    "de bruyne":   34155057,
    "ronaldinho":  34159850,
    "zidane":      34161049,
    "cruyff":      34163559,
    "r9":          34161040,
    "r10":         34159850,
    "nazario":     34161040,
    "pele": 34164201,
}

COUNTRY_PLAYER_MAP = {
    "brazil":      34161040,
    "brasil":      34161040,
    "argentina":   34146370,
    "france":      34161049,
    "portugal":    34146304,
    "netherlands": 34163559,
    "holland":     34163559,
    "dutch":       34163559,
    "germany":     34146705,
    "spain":       34146306,
    "england":     34146220,
    "italy":       34146306,
    "croatia":     34146306,
    "senegal":     34145506,
    "africa":      34145506,
    "colombia":    34146370,
    "uruguay":     34146370,
    "mexico":      34162098,
}

CLUB_PLAYER_MAP = {
    "barcelona":          34146370,
    "fc barcelona":       34146370,
    "real madrid":        34146304,
    "manchester united":  34146304,
    "juventus":           34146304,
    "psg":                34162098,
    "paris":              34162098,
    "liverpool":          34145506,
    "manchester city":    34155057,
    "bayern":             34146705,
    "chelsea":            34159231,
    "arsenal":            34146220,
    "inter miami":        34146370,
    "miami":              34146370,
    "dortmund":           34169116,
    "ajax":               34163559,
    "atletico madrid":    34159231,
}

TOPIC_PLAYER_MAP = {
    "world cup":        34161040,
    "champions league": 34146304,
    "ballon d'or":      34146370,
    "ballon dor":       34146370,
    "golden boot":      34146304,
    "hat trick":        34146370,
    "record":           34146304,
    "goat":             34146370,
    "greatest":         34146370,
    "best":             34146370,
    "worst":            34161040,
    "loss":             34161040,
    "defeat":           34161040,
    "curse":            34163559,
    "final":            34146304,
    "legend":           34161040,
    "comeback":         34146370,
    "injury":           34161040,
    "retired":          34163559,
    "forgotten":        34159850,
    "scandal":          34159850,
    "controversial":    34161049,
    "headbutt":         34161049,
    "red card":         34161049,
    "banned":           34159850,
    "penalty":          34146304,
    "free kick":        34146370,
    "dribble":          34159850,
    "skill":            34159850,
}


# ── Player detection ──────────────────────────────────────────────────────────

def _detect_player(topic: str, hook: str, scenes: list) -> tuple[int | None, str]:
    """
    Detect the most relevant player for the thumbnail.
    Priority order:
      1. Known player name directly in text
      2. Country name → iconic player for that country
      3. Club name → iconic player for that club
      4. Topic keyword → thematically relevant player
      5. None — text-only thumbnail

    Returns (player_id, detection_reason) or (None, reason).
    """
    all_text = f"{topic} {hook} {' '.join(s.get('keyword', '') for s in scenes)}".lower()

    # Priority 1 — direct player name match
    for name, pid in PLAYER_IDS.items():
        if name in all_text:
            logger.info("Player detected by name: %s", name)
            return pid, f"name match: {name}"

    # Priority 2 — country match
    for country, pid in COUNTRY_PLAYER_MAP.items():
        if country in all_text:
            logger.info("Player detected by country: %s", country)
            return pid, f"country match: {country}"

    # Priority 3 — club match
    for club, pid in CLUB_PLAYER_MAP.items():
        if club in all_text:
            logger.info("Player detected by club: %s", club)
            return pid, f"club match: {club}"

    # Priority 4 — topic keyword match
    for keyword, pid in TOPIC_PLAYER_MAP.items():
        if keyword in all_text:
            logger.info("Player detected by topic keyword: %s", keyword)
            return pid, f"topic match: {keyword}"

    # Priority 5 — default to R9 as the most universally dramatic player
    logger.info("No match found — defaulting to R9")
    return 34145943, "default: R9"


# ── TheSportsDB player image ──────────────────────────────────────────────────

def _fetch_player_image(player_id: int) -> bytes | None:
    """Fetch player image from TheSportsDB by ID."""
    url = f"{SPORTSDB_API}/lookupplayer.php?id={player_id}"

    try:
        req = Request(
            url=url,
            headers={"User-Agent": "youtube-agent/1.0"},
            method="GET"
        )
        with urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        players = data.get("players") or []
        if not players:
            logger.warning("TheSportsDB: no player found for ID %s", player_id)
            return None

        player = players[0]
        logger.info("TheSportsDB player: %s", player.get("strPlayer"))

        for field in ["strCutout", "strRender", "strThumb"]:
            img_url = player.get(field)
            if not img_url:
                continue
            try:
                img_req = Request(
                    url=img_url,
                    headers={"User-Agent": "youtube-agent/1.0"},
                    method="GET"
                )
                with urlopen(img_req, timeout=20) as img_response:
                    img_bytes = img_response.read()
                if img_bytes and len(img_bytes) > 5000:
                    logger.info("Player image from %s (%s KB)", field, len(img_bytes) // 1024)
                    return img_bytes
            except Exception as exc:
                logger.warning("Failed field %s: %s", field, exc)
                continue

        return None

    except Exception as exc:
        logger.warning("TheSportsDB lookup failed: %s", exc)
        return None


# ── Background generator ──────────────────────────────────────────────────────

def _generate_background() -> "Image":
    """
    Generate a dramatic red/black gradient background.
    Returns a PIL Image.
    """
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for y in range(HEIGHT):
        progress = y / HEIGHT

        if progress < 0.35:
            # Top: deep crimson
            t = progress / 0.35
            r = int(180 * (1 - t) + 120 * t)
            g = int(10  * (1 - t) + 5   * t)
            b = int(10  * (1 - t) + 5   * t)

        elif progress < 0.65:
            # Middle: black
            t = (progress - 0.35) / 0.30
            r = int(120 * (1 - t) + 8 * t)
            g = int(5   * (1 - t) + 5 * t)
            b = int(5   * (1 - t) + 5 * t)

        else:
            # Bottom: very dark red
            t = (progress - 0.65) / 0.35
            r = int(8  * (1 - t) + 40 * t)
            g = int(5  * (1 - t) + 5  * t)
            b = int(5  * (1 - t) + 5  * t)

        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    return img


def _add_vignette(img: "Image") -> "Image":
    """Add dark vignette edges for depth."""
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for i in range(60):
        alpha  = int(200 * (i / 60) ** 2)
        margin = i * 9
        draw.rectangle(
            [margin, margin, WIDTH - margin, HEIGHT - margin],
            outline=(0, 0, 0, alpha)
        )

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _add_diagonal_lines(img: "Image") -> "Image":
    """Add subtle diagonal line pattern for texture."""
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    spacing = 40
    for i in range(-HEIGHT, WIDTH + HEIGHT, spacing):
        draw.line(
            [(i, 0), (i + HEIGHT, HEIGHT)],
            fill=(255, 255, 255, 8),
            width=1
        )

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ── Text helpers ──────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = True):
    """Load best available font at given size."""
    from PIL import ImageFont

    candidates = [
        "arialbd.ttf",
        "Arial_Bold.ttf",
        "ariblk.ttf",           # Arial Black — ultra bold
        "Impact.ttf",           # Impact — classic YouTube thumbnail font
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
    """Wrap text to fit within max_width."""
    words   = text.split()
    lines   = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def _draw_outlined_text(
    draw,
    text: str,
    font,
    y: int,
    fill: tuple,
    outline: tuple,
    outline_width: int = 4,
) -> int:
    """
    Draw text centered horizontally with a thick outline.
    Returns the y position after the text block.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    w    = bbox[2] - bbox[0]
    h    = bbox[3] - bbox[1]
    x    = (WIDTH - w) // 2

    # Draw outline by offsetting in 8 directions
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)

    # Main text
    draw.text((x, y), text, font=font, fill=fill)

    return y + h + 18


# ── Main composite function ───────────────────────────────────────────────────

def _composite_thumbnail(
    hook: str,
    player_bytes: bytes | None,
    output_path: Path,
) -> bool:
    """
    Composite the final thumbnail:
    - Gradient background
    - Hook text at top
    - Player image centered at bottom
    - Decorative elements
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import io

        # ── Background ────────────────────────────────────────────────────
        img = _generate_background()
        img = _add_diagonal_lines(img)

        # ── Player image (bottom center) ──────────────────────────────────
        player_bottom_y = HEIGHT  # track where player starts for text positioning

        if player_bytes:
            try:
                player_img = Image.open(io.BytesIO(player_bytes)).convert("RGBA")

                # Scale player to fill bottom 60% of frame
                target_h  = int(HEIGHT * 0.62)
                ratio     = target_h / player_img.height
                target_w  = int(player_img.width * ratio)

                # Cap width at frame width
                if target_w > WIDTH:
                    target_w = WIDTH
                    target_h = int(player_img.height * (WIDTH / player_img.width))

                player_img = player_img.resize((target_w, target_h), Image.LANCZOS)

                # Position: centered horizontally, flush to bottom
                px = (WIDTH - target_w) // 2
                py = HEIGHT - target_h

                player_bottom_y = py

                # Add subtle shadow under player
                shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow)
                shadow_draw.ellipse(
                    [px + target_w // 4, py + target_h - 60,
                     px + 3 * target_w // 4, py + target_h + 20],
                    fill=(0, 0, 0, 120)
                )
                shadow = shadow.filter(ImageFilter.GaussianBlur(radius=20))
                img    = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")

                # Paste player
                img_rgba = img.convert("RGBA")
                img_rgba.paste(player_img, (px, py), player_img)
                img = img_rgba.convert("RGB")

                logger.info("Player composited at (%s, %s) size %sx%s",
                            px, py, target_w, target_h)

            except Exception as exc:
                logger.warning("Player composite failed: %s", exc)

        img  = _add_vignette(img)
        draw = ImageDraw.Draw(img)

        # ── Top decorative bar ────────────────────────────────────────────
        bar_h = 8
        draw.rectangle([0, 0, WIDTH, bar_h], fill=(220, 20, 20))

        # ── Hook text (top section) ───────────────────────────────────────
        hook_clean = hook.replace("—", "").strip().upper()

        # Use large font for hook
        font_large  = _load_font(size=108)
        font_medium = _load_font(size=86)

        # Wrap hook text
        lines   = _wrap_text(hook_clean, font_large, WIDTH - 80, draw)
        current_y = 60

        # If too many lines switch to medium font
        if len(lines) > 3:
            lines     = _wrap_text(hook_clean, font_medium, WIDTH - 80, draw)
            font_hook = font_medium
        else:
            font_hook = font_large

        for line in lines:
            current_y = _draw_outlined_text(
                draw=draw,
                text=line,
                font=font_hook,
                y=current_y,
                fill=(255, 255, 255),
                outline=(180, 0, 0),
                outline_width=5,
            )
            current_y += 8   # extra line spacing

        # ── Red accent line under text ────────────────────────────────────
        line_y = current_y + 20
        draw.rectangle(
            [60, line_y, WIDTH - 60, line_y + 5],
            fill=(220, 20, 20)
        )

        # ── Bottom label ──────────────────────────────────────────────────
        font_label = _load_font(size=40)
        label      = "WATCH UNTIL THE END"
        bbox       = draw.textbbox((0, 0), label, font=font_label)
        lw         = bbox[2] - bbox[0]
        lh         = bbox[3] - bbox[1]
        lx         = (WIDTH - lw) // 2
        ly         = HEIGHT - 100

        # Only draw if it doesn't overlap the player too much
        if ly > player_bottom_y + 40:
            pad = 16
            draw.rounded_rectangle(
                [lx - pad, ly - pad,
                 lx + lw + pad, ly + lh + pad],
                radius=10,
                fill=(200, 15, 15)
            )
            draw.text((lx, ly), label, font=font_label, fill=(255, 255, 255))

        img.save(str(output_path), "JPEG", quality=95)
        logger.info("Thumbnail composited and saved: %s", output_path.name)
        return True

    except Exception as exc:
        logger.error("Thumbnail composite failed: %s", exc)
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_thumbnail(
    title: str,
    topic: str,
    hook: str,
    job_dir: Path,
    scenes: list = None,
) -> str:
    """
    Generate the hook thumbnail.
    Returns local file path as string.

    Steps:
    1. Detect player ID from topic/hook/scenes
    2. Fetch player image from TheSportsDB by ID
    3. Composite: gradient + player + hook text
    4. Fallback to text-only card if player image unavailable
    """
    output_path = job_dir / "thumbnail.jpg"
    scenes      = scenes or []

    # Step 1 — detect player (returns tuple: (player_id, reason))
    player_id, reason = _detect_player(topic, hook, scenes)
    logger.info("Player selected: ID=%s reason=%s", player_id, reason)

    # Step 2 — fetch player image by ID
    player_bytes = None
    if player_id:
        player_bytes = _fetch_player_image(player_id)
        if not player_bytes:
            logger.warning(
                "No image fetched for player ID %s -- text-only thumbnail",
                player_id
            )

    # Step 3 — composite (works with or without player image)
    success = _composite_thumbnail(hook, player_bytes, output_path)

    if success:
        return str(output_path)

    raise Exception("Thumbnail generation failed completely.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_agent.settings")

    import django
    django.setup()

    parser = argparse.ArgumentParser()
    parser.add_argument("--title",  default="Netherlands World Cup Curse")
    parser.add_argument("--topic",  default="Lionel Messi FC Barcelona career")
    parser.add_argument("--hook",   default="THEY BANNED THIS FROM FOOTBALL HISTORY")
    parser.add_argument("--job-id", default="manual-test")
    args = parser.parse_args()

    from django.conf import settings
    job_dir = Path(settings.MEDIA_ROOT) / "jobs" / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    path = generate_thumbnail(
        title   = args.title,
        topic   = args.topic,
        hook    = args.hook,
        job_dir = job_dir,
        scenes  = [],
    )
    print("Thumbnail saved:", path)

# Final variable reference table at EOF:
# variable_name | type | purpose
# argparse | module | Used by the standalone CLI runner for thumbnail generation.
# WIDTH | int | Output thumbnail width in pixels.
# HEIGHT | int | Output thumbnail height in pixels.
# SPORTSDB_API | str | Base URL for TheSportsDB API lookups.
# PLAYER_IDS | dict[str, int] | Known player names mapped to player IDs.
# COUNTRY_PLAYER_MAP | dict[str, int] | Country keywords mapped to iconic player IDs.
# CLUB_PLAYER_MAP | dict[str, int] | Club keywords mapped to fallback player IDs.
# TOPIC_PLAYER_MAP | dict[str, int] | Topic keywords mapped to fallback player IDs.
# _detect_player | func | Determine the best player for the thumbnail based on text.
# _fetch_player_image | func | Download a player image from TheSportsDB.
# _generate_background | func | Create the red/black gradient thumbnail background.
# _add_vignette | func | Apply a dark vignette effect.
# _add_diagonal_lines | func | Add subtle texture lines.
# _load_font | func | Load an available bold font.
# _wrap_text | func | Wrap hook text to fit the thumbnail width.
# _draw_outlined_text | func | Render centered outlined text.
# _composite_thumbnail | func | Composite the final thumbnail and save it.
# generate_thumbnail | func | Top-level thumbnail generation function used by the pipeline.
