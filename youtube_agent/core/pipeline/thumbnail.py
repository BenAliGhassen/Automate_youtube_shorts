"""
Thumbnail generator for YouTube Shorts hook frame.
Layout:
  - Red/black dramatic gradient background
  - Bold hook text at the top
  - Player/team image (from TheSportsDB) centered at the bottom when available
  - Fallback: Pillow-only card if no SportsDB image is found

Pipeline:
  1. Extract player or team name from topic/scenes
  2. Dynamically search TheSportsDB for players or teams mentioned in the topic/hook/scenes
  3. Fetch player cutout or team-related image from TheSportsDB
  4. Composite the image over the dramatic background
  5. Add hook text at top
  6. Fallback to pure Pillow card if any step fails
"""

import argparse
import json
import logging
import os
import re
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

# ── SportsDB search helpers ───────────────────────────────────────────────
#
# This pipeline dynamically searches TheSportsDB for players and teams using
# generated topic/hook/scene keywords.
# It no longer relies on hard-coded player or topic ID maps for thumbnail selection.

# ── SportsDB entity search helpers ──────────────────────────────────────────

def _sportsdb_request(url: str, retries: int = 3) -> dict:
    """Perform a SportsDB request, retrying on transient failures."""
    for attempt in range(retries):
        try:
            req = Request(
                url=url,
                headers={"User-Agent": "youtube-agent/1.0"},
                method="GET",
            )
            with urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * 2
                logger.warning(
                    "SportsDB rate limited — waiting %ss (attempt %s/%s)",
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                continue
            logger.warning("SportsDB request failed with HTTP %s: %s", exc.code, exc)
            break

        except URLError as exc:
            logger.warning("SportsDB request error: %s", exc)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            break

        except Exception as exc:
            logger.warning("SportsDB request failed: %s", exc)
            break

    return {}


def _normalize_search_term(value: str) -> str:
    """Normalize a text value into a safe SportsDB search query."""
    tokens = re.findall(r"[A-Za-z0-9']+", value)
    return " ".join(tokens).strip()


def _get_category_representative(topic: str, hook: str) -> tuple[str, str] | None:
    """Return a representative player name for category-based or clickbait topics."""
    lower = f"{topic} {hook}".lower()

    # Category-specific anchors: use a known player even when no named player appears.
    if "shortest" in lower:
        return "player", "Lionel Messi"
    if "tallest" in lower:
        return "player", "Erling Haaland"
    if "oldest" in lower:
        return "player", "Cristiano Ronaldo"

    # Broad topic-based clickbait fallback for player-only thumbnails.
    if "world cup" in lower or "fifa" in lower:
        return "player", "Lionel Messi"
    if "champions league" in lower or "ucl" in lower:
        return "player", "Erling Haaland"
    if "portugal" in lower:
        return "player", "Cristiano Ronaldo"
    if "brazil" in lower:
        return "player", "Neymar Jr"
    if "argentina" in lower:
        return "player", "Lionel Messi"
    if "manchester city" in lower or "man city" in lower:
        return "player", "Erling Haaland"
    if "barcelona" in lower:
        return "player", "Lionel Messi"

    return None


def _select_thumbnail_text(title: str, hook: str) -> str:
    """Choose the best headline for the thumbnail between title and hook."""
    title_clean = title.strip()
    hook_clean = hook.strip()

    if title_clean and len(title_clean) <= 45 and len(hook_clean) > 65:
        return title_clean
    if hook_clean and len(hook_clean) <= 85:
        return hook_clean
    return title_clean or hook_clean


def _should_skip_generic_candidate(normalized: str) -> bool:
    """Skip generic or non-entity candidate strings when building SportsDB search terms."""
    generic_terms = {
        "world cup",
        "world cup 2026",
        "fifa world cup",
        "fifa world cup 2026",
        "fifa",
        "world cup",
        "2026",
        "2026 world cup",
        "the 5 tallest players",
        "5 tallest players",
        "tallest players",
        "shortest players",
        "oldest players",
        "tallest player",
        "shortest player",
        "oldest player",
        "player",
        "players",
        "team",
        "teams",
        "football",
        "soccer",
        "goal",
        "goals",
        "record",
        "stats",
        "highest",
        "lowest",
    }

    if normalized in generic_terms:
        return True

    if re.fullmatch(r"\d+(?:st|nd|rd|th)?", normalized):
        return True

    if re.search(r"\b(?:tallest|shortest|oldest|youngest|fastest|biggest|smallest|best|worst|record|goal|stats|championship|final|cup|world|fifa)\b", normalized) and len(normalized.split()) <= 3:
        return True

    return False


def _candidate_is_team_name(candidate: str) -> bool:
    """Simple heuristic to decide whether a normalized candidate looks like a team name."""
    team_indicators = {
        "fc",
        "real",
        "united",
        "city",
        "sporting",
        "atletico",
        "athletic",
        "club",
        "national",
        "america",
        "galatasaray",
        "munich",
        "madrid",
        "barcelona",
        "psg",
        "liverpool",
        "chelsea",
        "arsenal",
        "dortmund",
        "inter",
        "juventus",
        "milan",
        "ac",
    }
    if any(token in candidate.split() for token in team_indicators):
        return True
    if "national" in candidate:
        return True
    return False


def _build_search_candidates(topic: str, hook: str, scenes: list) -> list[str]:
    """Generate candidate player or team search terms from the topic, hook, and scene keywords."""
    seen = set()
    candidates = []
    raw_terms = [topic, hook] + [scene.get("keyword", "") for scene in scenes]

    def add_candidate(term: str) -> None:
        normalized = _normalize_search_term(term)
        if not normalized or normalized in seen or _should_skip_generic_candidate(normalized):
            return
        seen.add(normalized)
        candidates.append(normalized)

    for raw in raw_terms:
        add_candidate(raw)
        normalized = _normalize_search_term(raw)
        if not normalized:
            continue

        for part in re.split(r"\s+(?:vs|versus)\.?\s+", normalized, flags=re.I):
            add_candidate(part)

        words = normalized.split()
        if len(words) > 1:
            max_length = min(4, len(words))
            for length in range(max_length, 1, -1):
                for start in range(len(words) - length + 1):
                    add_candidate(" ".join(words[start:start + length]))

            for length in range(2, min(4, len(words)) + 1):
                add_candidate(" ".join(words[-length:]))

    return candidates[:24]


def _extract_image_url(record: dict, fields: list[str]) -> str | None:
    """Return the first non-empty URL from a SportsDB record."""
    for field in fields:
        url = record.get(field)
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _search_sportsdb_player(name: str) -> tuple[int, str] | None:
    """Search TheSportsDB for a player name and return the first usable result."""
    url = f"{SPORTSDB_API}/searchplayers.php?p={quote(name)}"
    data = _sportsdb_request(url)
    players = data.get("player") or []
    for player in players:
        player_id = player.get("idPlayer")
        image_url = _extract_image_url(
            player,
            ["strCutout", "strRender", "strThumb", "strFanart", "strThumbSmall"],
        )
        if player_id and image_url:
            logger.info("SportsDB player search matched: %s", player.get("strPlayer"))
            return int(player_id), player.get("strPlayer", name)
    return None


def _search_sportsdb_team(name: str) -> tuple[int, str] | None:
    """Search TheSportsDB for a team name and return the first usable result."""
    url = f"{SPORTSDB_API}/searchteams.php?t={quote(name)}"
    data = _sportsdb_request(url)
    teams = data.get("teams") or []
    for team in teams:
        team_id = team.get("idTeam")
        image_url = _extract_image_url(
            team,
            [
                "strTeamBadge",
                "strTeamLogo",
                "strTeamBanner",
                "strTeamFanart1",
                "strTeamFanart2",
                "strTeamJersey",
            ],
        )
        if team_id and image_url:
            logger.info("SportsDB team search matched: %s", team.get("strTeam"))
            return int(team_id), team.get("strTeam", name)
    return None


def _fetch_team_image(team_id: int) -> bytes | None:
    """Fetch a team image from TheSportsDB by team ID."""
    url = f"{SPORTSDB_API}/lookupteam.php?id={team_id}"
    data = _sportsdb_request(url)
    teams = data.get("teams") or []
    if not teams:
        logger.warning("TheSportsDB: no team found for ID %s", team_id)
        return None

    team = teams[0]
    image_url = _extract_image_url(
        team,
        [
            "strTeamBanner",
            "strTeamFanart1",
            "strTeamFanart2",
            "strTeamJersey",
            "strTeamLogo",
            "strTeamBadge",
        ],
    )
    if not image_url:
        logger.warning("No team image URL available for team ID %s", team_id)
        return None

    return _download_image_bytes(image_url)


def _download_image_bytes(url: str, retries: int = 3) -> bytes | None:
    """Download image bytes from a URL with retry support."""
    for attempt in range(retries):
        try:
            req = Request(
                url=url,
                headers={"User-Agent": "youtube-agent/1.0"},
                method="GET",
            )
            with urlopen(req, timeout=30) as response:
                data = response.read()
                if data and len(data) > 5000:
                    return data
                return None

        except HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * 3
                logger.warning(
                    "SportsDB download rate limited — waiting %ss (attempt %s/%s)",
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                continue
            logger.warning("Failed to download SportsDB image %s: %s", url, exc)
            return None

        except URLError as exc:
            logger.warning("Failed to download SportsDB image %s: %s", url, exc)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            return None

        except Exception as exc:
            logger.warning("Failed to download SportsDB image %s: %s", url, exc)
            return None

    return None


# The thumbnail search logic is intentionally dynamic: it prefers SportsDB lookups
# for real player or team names extracted from the topic, hook, and generated scenes.
# Wikimedia Commons is kept as a backup only for scene visuals when SportsDB imagery
# is not available.
def _detect_player_or_team(topic: str, hook: str, scenes: list) -> tuple[str | None, int | None, str]:
    """Detect the best SportsDB player or team to use for the thumbnail."""
    candidates = _build_search_candidates(topic, hook, scenes)
    for candidate in candidates:
        if not candidate or _should_skip_generic_candidate(candidate):
            continue

        if _candidate_is_team_name(candidate):
            logger.info("Candidate looks like a team; trying team search first: %s", candidate)
            team_match = _search_sportsdb_team(candidate)
            if team_match:
                team_id, team_name = team_match
                return "team", team_id, f"SportsDB team search: {team_name}"

            logger.info("Trying SportsDB player search for team-like text: %s", candidate)
            player_match = _search_sportsdb_player(candidate)
            if player_match:
                player_id, player_name = player_match
                return "player", player_id, f"SportsDB player search: {player_name}"

        else:
            logger.info("Trying SportsDB player search for: %s", candidate)
            player_match = _search_sportsdb_player(candidate)
            if player_match:
                player_id, player_name = player_match
                return "player", player_id, f"SportsDB player search: {player_name}"

            logger.info("Trying SportsDB team search for: %s", candidate)
            team_match = _search_sportsdb_team(candidate)
            if team_match:
                team_id, team_name = team_match
                return "team", team_id, f"SportsDB team search: {team_name}"

    # If no direct SportsDB candidate matches, fall back to a strong topic-related player.
    # This ensures a player image is used even when the topic does not name a player.
    representative = _get_category_representative(topic, hook)
    if representative:
        entity_type, name = representative
        logger.info("Category-based thumbnail fallback selected: %s", name)
        if entity_type == "player":
            player_match = _search_sportsdb_player(name)
            if player_match:
                player_id, player_name = player_match
                return "player", player_id, f"category representative: {player_name}"
        elif entity_type == "team":
            team_match = _search_sportsdb_team(name)
            if team_match:
                team_id, team_name = team_match
                return "team", team_id, f"category representative: {team_name}"

    logger.info("No SportsDB player or team found for the current topic/scenes")
    return None, None, "no SportsDB player or team matched"


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
        # Start the title closer to the player image so the top section feels tighter.
        current_y = 42

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
            current_y += 6   # extra line spacing

        # ── Red accent line under text ────────────────────────────────────
        line_y = current_y + 12
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
    1. Detect player or team ID from topic/hook/scenes using SportsDB search
    2. Fetch player or team image from TheSportsDB by ID
    3. Composite: gradient + SportsDB image + hook text
    4. Fallback to text-only card if SportsDB image unavailable
    """
    output_path = job_dir / "thumbnail.jpg"
    scenes      = scenes or []

    # Step 1 — detect SportsDB player or team entity
    entity_type, entity_id, reason = _detect_player_or_team(topic, hook, scenes)
    logger.info(
        "SportsDB entity selected: type=%s id=%s reason=%s",
        entity_type,
        entity_id,
        reason,
    )

    thumbnail_text = _select_thumbnail_text(title, hook)
    logger.info("Selected thumbnail text: %s", thumbnail_text)

    # Step 2 — fetch image by entity type
    entity_bytes = None
    if entity_type == "player" and entity_id:
        entity_bytes = _fetch_player_image(entity_id)
    elif entity_type == "team" and entity_id:
        entity_bytes = _fetch_team_image(entity_id)

    if not entity_bytes and entity_id:
        logger.warning(
            "No SportsDB image fetched for %s ID %s -- text-only thumbnail",
            entity_type,
            entity_id,
        )

    # Step 3 — composite (works with or without SportsDB image)
    success = _composite_thumbnail(thumbnail_text, entity_bytes, output_path)

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


