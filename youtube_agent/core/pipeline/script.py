"""Gemini-powered script generation for YouTube Shorts."""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from django.conf import settings

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_VISION_MODEL = "gemini-3.1-flash-lite"


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


def _extract_candidate_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_parts = [
            part.get("text", "").strip()
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ]
        if text_parts:
            return "\n".join(text_parts).strip()

    prompt_feedback = response_payload.get("promptFeedback")
    if prompt_feedback:
        raise Exception(
            "Gemini returned no content. Prompt feedback: "
            f"{json.dumps(prompt_feedback, ensure_ascii=True)}"
        )

    raise Exception("Gemini returned no content in the response.")


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned


def _parse_script_payload(raw_text: str) -> dict:
    cleaned_text = _strip_code_fences(raw_text)
    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if not match:
            raise Exception("Gemini response was not valid JSON.")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise Exception("Could not parse Gemini JSON response.") from exc

    title  = payload.get("title")
    script = payload.get("script")
    scenes = payload.get("scenes")

    if not isinstance(scenes, list) or len(scenes) < 1:
        raise Exception("Gemini response missing valid 'scenes' array.")

    validated_scenes = []
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise Exception(f"Scene {i} is not a dict.")

        keyword  = scene.get("keyword")
        backups  = scene.get("backups", [])
        duration = scene.get("duration")

        if not isinstance(keyword, str) or not keyword.strip():
            raise Exception(f"Scene {i} missing valid 'keyword'.")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise Exception(f"Scene {i} missing valid 'duration'.")

        # Normalize backups — must be a list of non-empty strings
        if not isinstance(backups, list):
            backups = []
        backups = [str(b).strip() for b in backups if str(b).strip()]

        validated_scenes.append({
            "keyword":  keyword.strip(),
            "backups":  backups,
            "duration": float(duration),
        })

    normalized_title  = re.sub(r"\s+", " ", title).strip()
    normalized_script = re.sub(r"\s+", " ", script).strip()

    return {
        "title":   normalized_title,
        "script":  normalized_script,
        "scenes":  validated_scenes,
    }


def generate_script(topic: str) -> dict:
    """
    Generate a title and a 60-second spoken script for a topic.

    Returns:
        {
            "title": str,
            "script": str,
        }
    """
    if not topic or not topic.strip():
        raise Exception("Topic cannot be empty for script generation.")

    api_key = getattr(settings, "GEMINI_API_KEY", "").strip()
    if not api_key:
        raise Exception("GEMINI_API_KEY is not configured in Django settings.")

    # Use the configured Gemini text model for script generation.
    # This is intentionally separate from the vision model used for image checks.
    model_name = getattr(settings, "GEMINI_MODEL", GEMINI_VISION_MODEL).strip()
    if not model_name:
        model_name = GEMINI_VISION_MODEL

    prompt = (
    "You are an elite YouTube Shorts scriptwriter for a viral football channel.\n"
    "Your scripts consistently achieve 85%+ retention rate across 30-second Shorts.\n"
    "You understand the psychology of scroll-stopping content, pattern interrupts,\n"
    "and the exact sentence rhythm that keeps viewers locked for 30 full seconds.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "RETENTION PROBLEM YOU MUST SOLVE\n"
    "═══════════════════════════════════════════════════════\n"
    "This channel loses 40% of viewers after second 7.\n"
    "The cause: slow build-up, story-driven pacing, predictable sentences.\n"
    "Your job: eliminate every second of dead air, slow setup, and predictable flow.\n"
    "Every single sentence must hit harder than the previous one.\n"
    "The viewer must feel they will miss something critical if they scroll away.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "FAST-PACE WRITING RULES (NON-NEGOTIABLE)\n"
    "═══════════════════════════════════════════════════════\n"
    "1. SENTENCE LENGTH: Maximum 10 words per sentence. No exceptions.\n"
    "   Bad : 'Lionel Messi was born in Rosario Argentina and from a very young age\n"
    "          showed exceptional talent that nobody had ever seen before.'\n"
    "   Good: 'Rosario. 1987. A kid nobody expected. Then everything changed.'\n\n"
    "2. RHYTHM: Alternate between ultra-short (3-5 words) and short (7-10 words).\n"
    "   This creates a heartbeat rhythm that feels energetic and urgent.\n"
    "   Example: 'Six Ballons d'Or. Six. Not two. Not four. Six.'\n\n"
    "3. PATTERN INTERRUPTS: Every 2-3 sentences, break the flow with a shocking stat,\n"
    "   a question, or a one-word sentence. This resets viewer attention.\n"
    "   Examples: 'Wait.' / 'Nobody talks about this.' / 'The number? 91.'\n\n"
    "4. NO TRANSITIONS: Never use 'and then', 'after that', 'moving on', 'next up'.\n"
    "   Cut hard between ideas. Trust the viewer to follow.\n\n"
    "5. NUMBERS OVER ADJECTIVES: Replace every adjective with a specific number.\n"
    "   Bad : 'He scored an incredible amount of goals that season.'\n"
    "   Good: 'That season. 50 goals. 50.'\n\n"
    "6. SECOND-PERSON WEAPONS: Use 'you' to make stats feel personal.\n"
    "   'You will never see this again.' / 'You already know who won.'\n\n"
    "7. FORBIDDEN WORDS: never use these — they kill retention instantly:\n"
    "   'today', 'in this video', 'let me', 'we are going to', 'subscribe',\n"
    "   'amazing', 'incredible', 'unbelievable', 'legendary' (show don't tell)\n\n"

    "═══════════════════════════════════════════════════════\n"
    "CONTENT FORMAT DETECTION\n"
    "═══════════════════════════════════════════════════════\n"
    "Detect the format from the topic and apply the matching template.\n\n"

    "FORMAT A — COMPARISON BATTLE (e.g. 'Messi vs Ronaldo')\n"
    "Template: HOOK → Stat A vs Stat B → Stat A vs Stat B → Stat A vs Stat B\n"
    "          → Pattern interrupt → Final verdict (controversial, not safe)\n"
    "Rhythm: Ultra fast. 2-3s per image. Head-to-head cuts.\n"
    "Rules:\n"
    "- Alternate strictly between the two subjects every sentence.\n"
    "- Never give a safe verdict — take a side, be controversial.\n"
    "- Use direct numbers only: goals, trophies, assists, records.\n"
    "- End on the losing side's best stat to create debate in comments.\n"
    "Example structure:\n"
    "  '700 goals. 800 goals. 5 Ballons d'Or. 5 Ballons d'Or.\n"
    "   One World Cup. Zero World Cups. The debate is already over.'\n\n"

    "FORMAT B — FACT BOMBS (e.g. '5 facts about Messi nobody knows')\n"
    "Template: HOOK → Fact 1 → Fact 2 → Fact 3 → Fact 4 → Fact 5 → Loop close\n"
    "Rhythm: Fast. 3-4s per image. Each fact is one visual.\n"
    "Rules:\n"
    "- Each fact must be genuinely surprising — no Wikipedia top results.\n"
    "- Lead each fact with the number: 'Fact one.' / 'Number two.'\n"
    "- The last fact must be the most shocking — save the best for last.\n"
    "- Each fact is maximum 2 sentences.\n\n"

    "FORMAT C — DID YOU KNOW (e.g. 'Did you know Ronaldo almost quit football?')\n"
    "Template: HOOK (the answer) → Context → Proof → Stakes → Loop close\n"
    "Rhythm: Fast. 3-4s per image.\n"
    "Rules:\n"
    "- Open with the answer, not the question. Reward curiosity immediately.\n"
    "- Then explain WHY it matters in 3-4 rapid sentences.\n"
    "- End with the consequence: what changed because of this moment.\n\n"

    "FORMAT D — TOP 5 COUNTDOWN (e.g. 'Top 5 Messi goals')\n"
    "Template: HOOK → Number 5 → 4 → 3 → 2 → Number 1 (longest) → Loop close\n"
    "Rhythm: Fast. 3-4s per image. Each number gets one image.\n"
    "Rules:\n"
    "- Count DOWN, never up. 5 to 1 builds anticipation.\n"
    "- Each entry: one sentence maximum. Just the fact. No fluff.\n"
    "- Number 1 gets double the time — it is the payoff.\n"
    "- The hook must reference number 1 without revealing it.\n\n"

    "═══════════════════════════════════════════════════════\n"
   "HOOK AND LOOP STRUCTURE\n"
    "═══════════════════════════════════════════════════════\n"
    "The hook sentence is SPLIT INTO TWO HALVES across intro and outro.\n\n"

    "INTRO (first half) — THE BAIT:\n"
    "- Written in ALL CAPS.\n"
    "- Ends with an em dash (—).\n"
    "- Must sound like a RUMOR, a SECRET, or an IMPOSSIBLE CLAIM.\n"
    "- The viewer must think: 'that cannot be true — I need to know more.'\n"
    "- Use exaggeration, controversy, or a fake-sounding fact as the hook.\n"
    "- The hook does not have to be 100% literal — it can dramatize reality.\n"
    "- Good hooks:\n"
    "    'THE DUTCH GOVERNMENT BANNED THIS FOOTAGE FOR 30 YEARS—'\n"
    "    'THIS COUNTRY WINS EVERY FINAL THEY PLAY — EXCEPT THE ONES THAT MATTER—'\n"
    "    'FIFA TRIED TO DELETE THIS RECORD FROM HISTORY—'\n"
    "    'THEY CHANGED FOOTBALL FOREVER AND WERE NEVER ALLOWED TO WIN—'\n"
    "    'THREE FINALS. THREE DEFEATS. ONE CURSE NOBODY CAN EXPLAIN—'\n"
    "- Bad hooks (descriptive, not bait):\n"
    "    'NO OTHER FOOTBALL TEAM SUFFERED THIS CRUEL TRAGEDY—' (tells not baits)\n"
    "    'THIS IS THE GREATEST TEAM EVER—' (generic, no information gap)\n"
    "    'MESSI IS BETTER THAN RONALDO—' (opinion, not a secret)\n\n"

    "OUTRO (second half) — THE PAYOFF:\n"
    "- Starts with — (em dash).\n"
    "- Completes the intro sentence grammatically.\n"
    "- Must feel like the answer to the bait — but still leave something unresolved.\n"
    "- Ends on a STRONG word. Never a filler.\n"
    "- Good outros:\n"
    "    '—and nobody in football has ever explained why.'\n"
    "    '—and they still have not forgiven themselves.'\n"
    "    '—making them the greatest team to never exist on a trophy.'\n"
    "- Bad outros:\n"
    "    '—making them the ultimate losers.' (dismissive, kills emotion)\n\n"

    "LOOP TEST — read outro into intro aloud:\n"
    "  '...making them the greatest team to never exist on a trophy. | "\
    "THEY CHANGED FOOTBALL FOREVER AND WERE NEVER ALLOWED TO WIN—...'\n"
    "  Must sound like one continuous sentence. If it does not — rewrite.\n\n"
    "═══════════════════════════════════════════════════════\n"
    "SCRIPT TECHNICAL RULES\n"
    "═══════════════════════════════════════════════════════\n"
    "- Total duration: 28 to 35 seconds of spoken narration.\n"
    "- Word count: 70 to 95 words.\n"
    "- Every word earns its place — cut anything that does not add tension.\n"
    "- No empty seconds. No padding. Last word = last millisecond.\n"
    "- Plain spoken sentences only. No bullets, emojis, markdown, stage directions.\n"
    "- Tone: urgent, confident, slightly confrontational. Never calm. Never neutral.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "SCENES AND WIKIMEDIA KEYWORDS\n"
    "═══════════════════════════════════════════════════════\n"
    "CONTEXT: Keywords are searched directly on Wikimedia Commons.\n"
    "Wikimedia is a public archive of real photos, match shots, trophy ceremonies.\n"
    "Write keywords exactly like a Wikipedia photo file name — real names and events.\n\n"

    "Each scene has:\n"
    "  keyword  : primary Wikimedia search (max 5 words)\n"
    "  backups  : 3 fallback searches, specific → generic\n"
    "  duration : seconds this image displays (integer)\n\n"

    "KEYWORD FORMULA: [Full name] + [Club or Country] + [Event]\n"
    "  Good: 'Lionel Messi FC Barcelona goal'\n"
    "  Good: 'Cristiano Ronaldo Real Madrid Champions League'\n"
    "  Bad : 'epic football moment' (not on Wikimedia)\n"
    "  Bad : 'Messi best goal ever' (adjectives return nothing)\n\n"

    "DURATION RULES BY FORMAT:\n"
    "  Comparison : 2-3s per image — ultra fast, head-to-head energy\n"
    "  Fact bombs : 3-4s per image — fast but readable\n"
    "  Did you know: 3-4s per image — fast but readable\n"
    "  Top 5      : 3-4s per image, number 1 gets 6-8s\n\n"

    "SCENE NARRATIVE RULES:\n"
    "- Scene 1 (HOOK VISUAL): Dramatic, scroll-stopping. Does not have to match topic.\n"
    "  Its only job: stop the thumb. Use crowd chaos, controversial moment, red card.\n"
    "- Scenes 2-N (BODY): One image per fact/stat/player — strict narrative order.\n"
    "  For comparisons: strictly alternate between the two subjects.\n"
    "- Last scene (LOOP CLOSE): Visually echoes scene 1. Creates the visual loop.\n\n"

    "BACKUP KEYWORD RULES:\n"
    "  backup 1: slightly broader than primary\n"
    "  backup 2: club or country only\n"
    "  backup 3: fully generic football image\n"
    "  Example:\n"
    "    primary : 'Messi FC Barcelona Champions League'\n"
    "    backup 1: 'Messi Barcelona Camp Nou'\n"
    "    backup 2: 'FC Barcelona football match'\n"
    "    backup 3: 'football match crowd stadium'\n\n"

    "SUM RULE: all durations must sum to total script duration exactly.\n"
    "Use 5 to 7 scenes total.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "OUTPUT FORMAT\n"
    "═══════════════════════════════════════════════════════\n"
    "Return ONLY valid JSON. Zero preamble. Zero explanation. Zero markdown fences.\n\n"
    "{\n"
    '  "format": "comparison|factbombs|didyouknow|top5",\n'
    '  "title": "string — under 60 chars, creates information gap, no clickbait adjectives",\n'
    '  "script": "string — ALL CAPS INTRO— body sentences. —outro completion.",\n'
    '  "scenes": [\n'
    "    {\n"
    '      "keyword": "string",\n'
    '      "backups": ["string", "string", "string"],\n'
    '      "duration": integer\n'
    "    }\n"
    "  ]\n"
    "}\n\n"

    "═══════════════════════════════════════════════════════\n"
    "SELF-CHECK BEFORE OUTPUT\n"
    "═══════════════════════════════════════════════════════\n"
    "☑ Every sentence is under 10 words\n"
    "☑ Rhythm alternates short/ultra-short throughout\n"
    "☑ At least 2 pattern interrupts in the body\n"
    "☑ Zero forbidden words used\n"
    "☑ Intro ends with — (em dash)\n"
    "☑ Outro starts with — and completes the intro grammatically\n"
    "☑ Intro + Outro read as one sentence when looped\n"
    "☑ Last word is strong — not a filler\n"
    "☑ All durations match the format speed rule\n"
    "☑ All durations sum to total script duration\n"
    "☑ All keywords follow Wikimedia formula\n"
    "☑ Scene 1 is scroll-stopping\n"
    "☑ Last scene echoes scene 1 visually\n"
    "☑ format field matches the detected content type\n\n"

    f"TOPIC: {topic}\n"
)
    request_body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.8,
            "maxOutputTokens": 1024,
            "thinkingConfig": {
                "thinkingLevel": "minimal",
            },
        },
    }

    request = Request(
        url=f"{GEMINI_API_BASE}/{model_name}:generateContent?key={api_key}",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "youtube-agent/1.0",
        },
        method="POST",
    )

    logger.info("Generating script for topic: %s with model %s", topic, model_name)

    try:
        opener = _build_http_opener()
        with opener.open(request, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "Gemini API request failed with status %s for model %s",
            exc.code,
            model_name,
        )
        raise Exception(
            f"Gemini API request failed with status {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        logger.error("Gemini API request could not reach the server: %s", exc.reason)
        raise Exception(f"Could not reach Gemini API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        logger.error("Gemini API returned invalid JSON.")
        raise Exception("Gemini API returned invalid JSON.") from exc

    raw_text = _extract_candidate_text(response_payload)
    result = _parse_script_payload(raw_text)
    logger.info("Script generated successfully for topic: %s", topic)
    return result


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

    parser = argparse.ArgumentParser(description="Generate a YouTube Shorts script.")
    parser.add_argument("topic", help="Topic to generate a script for.")
    args = parser.parse_args()

    try:
        generated = generate_script(args.topic)
    except Exception:
        logger.exception("Standalone script generation failed.")
        raise SystemExit(1)

    logger.info(
        "Generated script payload: %s",
        json.dumps(generated, ensure_ascii=True, indent=2),
    )
