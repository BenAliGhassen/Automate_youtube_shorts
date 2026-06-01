"""Gemini-powered script generation for YouTube Shorts."""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from django.conf import settings

logger = logging.getLogger(__name__)

# Gemini API endpoint and model defaults.
# The primary model is used first; if it fails, we optionally retry using a fallback.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
GEMINI_MAX_RETRIES = 3
GEMINI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _build_http_opener():
    # Respect Django settings if outbound proxy configuration is explicitly defined.
    # Otherwise, do not inherit potentially broken OS-level proxy environment variables.
    http_proxy = getattr(settings, "OUTBOUND_HTTP_PROXY", "").strip()
    https_proxy = getattr(settings, "OUTBOUND_HTTPS_PROXY", "").strip()

    proxies: dict[str, str] = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy

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


def _normalize_scene(scene: Any, index: int) -> dict:
    if not isinstance(scene, dict):
        raise Exception(f"Scene {index + 1} is not a valid JSON object.")

    keyword = scene.get("keyword")
    backups = scene.get("backups", [])
    duration = scene.get("duration")

    if not isinstance(keyword, str) or not keyword.strip():
        raise Exception(f"Scene {index + 1} must include a non-empty 'keyword'.")

    if backups is None:
        backups = []
    if not isinstance(backups, list):
        raise Exception(f"Scene {index + 1} 'backups' must be a list of strings.")

    normalized_backups = [str(item).strip() for item in backups if isinstance(item, str) and item.strip()]

    if isinstance(duration, str):
        try:
            duration = float(duration.strip())
        except ValueError:
            duration = None

    if not isinstance(duration, (int, float)) or duration <= 0:
        raise Exception(f"Scene {index + 1} must include a positive 'duration'.")

    return {
        "keyword": keyword.strip(),
        "backups": normalized_backups,
        "duration": float(duration),
    }


def _parse_script_payload(raw_text: str) -> dict:
    cleaned_text = _strip_code_fences(raw_text)
    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if not match:
            raise Exception(
                "Gemini response was not valid JSON with 'title', 'script', and 'scenes' fields."
            )
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise Exception(
                "Gemini response contained JSON-like content, but it could not be parsed."
            ) from exc

    title   = payload.get("title")
    script  = payload.get("script")
    scenes  = payload.get("scenes")
    keywords = payload.get("keywords")
    format_type = payload.get("format")

    if not isinstance(title, str) or not title.strip():
        raise Exception("Gemini response did not include a valid 'title' string.")
    if not isinstance(script, str) or not script.strip():
        raise Exception("Gemini response did not include a valid 'script' string.")

    if not isinstance(scenes, list) or len(scenes) < 1:
        if isinstance(keywords, list) and any(str(k).strip() for k in keywords):
            normalized_keywords = [str(k).strip() for k in keywords if str(k).strip()]
            logger.warning(
                "Gemini response did not include 'scenes'; falling back to keywords for scene generation."
            )
            scenes = [
                {"keyword": keyword, "backups": [], "duration": 4.0}
                for keyword in normalized_keywords
            ]
        else:
            raise Exception("Gemini response did not include a valid 'scenes' array.")

    normalized_scenes = [_normalize_scene(scene, idx) for idx, scene in enumerate(scenes)]

    result = {
        "title":    re.sub(r"\s+", " ", title).strip(),
        "script":   re.sub(r"\s+", " ", script).strip(),
        "scenes":   normalized_scenes,
    }

    if isinstance(format_type, str) and format_type.strip():
        result["format"] = format_type.strip()

    if isinstance(keywords, list):
        result["keywords"] = [str(k).strip() for k in keywords if str(k).strip()]

    return result


def _execute_gemini_request(request: Request, model_name: str) -> dict[str, Any]:
    retry_delay = getattr(settings, "GEMINI_TEXT_RETRY_DELAY", 5)

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            opener = _build_http_opener()
            with opener.open(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        # HTTP errors may be transient; retry on common server-side or rate-limit codes.
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            logger.warning(
                "Gemini API request failed with status %s for model %s (attempt %s/%s)",
                exc.code,
                model_name,
                attempt,
                GEMINI_MAX_RETRIES,
            )
            if exc.code in GEMINI_RETRYABLE_STATUS_CODES and attempt < GEMINI_MAX_RETRIES:
                logger.info(
                    "Retrying Gemini request after %s seconds due to status %s",
                    retry_delay,
                    exc.code,
                )
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise Exception(
                f"Gemini API request failed with status {exc.code}: {error_body}"
            ) from exc

        except URLError as exc:
            logger.warning(
                "Gemini API request could not reach the server: %s (attempt %s/%s)",
                exc.reason,
                attempt,
                GEMINI_MAX_RETRIES,
            )
            if attempt < GEMINI_MAX_RETRIES:
                logger.info("Retrying Gemini request after %s seconds", retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            raise Exception(f"Could not reach Gemini API: {exc.reason}") from exc

        except json.JSONDecodeError as exc:
            logger.error("Gemini API returned invalid JSON.")
            raise Exception("Gemini API returned invalid JSON.") from exc

    raise Exception("Gemini request failed after retries.")


def generate_script(topic: str) -> dict:
    """
    Generate a title, script, and visual scene plan for a topic.

    Returns:
        {
            "title": str,
            "script": str,
            "scenes": list[dict],
            "format": str,          # optional
            "keywords": list[str],  # optional
        }
    """
    if not topic or not topic.strip():
        raise Exception("Topic cannot be empty for script generation.")

    api_key = getattr(settings, "GEMINI_API_KEY", "").strip()
    if not api_key:
        raise Exception("GEMINI_API_KEY is not configured in Django settings.")

    model_name = getattr(settings, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_GEMINI_MODEL

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
    "1. SENTENCE LENGTH: Maximum 15 words per sentence. No exceptions.\n"
    "   Bad : 'Lionel Messi was born in Rosario Argentina and from a very young age\n"
    "          showed exceptional talent that nobody had ever seen before.'\n"
    "   Good: 'exceptional talent nobody had ever seen before then this happened '\n\n"
    "3. PATTERN INTERRUPTS: Every 2-3 sentences, break the flow with a shocking stat,\n"
    "   a question, or a one-word sentence. This resets viewer attention.\n"
    "   Examples: 'Wait.' / 'Nobody talks about this.' / 'The number? 91.'\n\n"
    "5. NUMBERS OVER ADJECTIVES: Replace every adjective with a specific number.\n"
    "   Bad : 'He scored an incredible amount of goals that season.'\n"
    "   Good: 'That season he scored 50 goals.'\n\n"
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
    "- Lead each fact with the number: 'one.' / 'two.'\n"
    "- The last fact must be the most shocking — save the best for last.\n"
    "- Each fact is maximum 6 sentences.\n\n"

    "FORMAT C — DID YOU KNOW (e.g. 'Did you know Ronaldo almost quit football?')\n"
    "Template: HOOK  → Context (the answer) → Proof → Stakes →  Loop close\n"
    "Rhythm: Fast. 3-4s per image.\n"
    "Rules:\n"
    "- explain WHY it matters in 5-6 rapid sentences.\n"
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
    "    'THE BRAZILIAN GOVERNMENT DECLARED HIM A NATIONAL TREASURE'\n"
    "- Bad hooks (descriptive, not bait):\n"
    "    'NO OTHER FOOTBALL TEAM SUFFERED THIS CRUEL TRAGEDY—' (tells not baits)\n"
    "    'THIS IS THE GREATEST TEAM EVER—' (generic, no information gap)\n"
    "    'MESSI IS BETTER THAN RONALDO—' (opinion, not a secret)\n\n"

    "OUTRO (second half) — THE PAYOFF:\n"
    "- Starts with — (em dash).\n"
    "- Must feel like the start to the bait.\n"
    "-the outro must grammatically attach to the begining of the intro to trigger the loop.\n"
    "- Good outros:\n"
    "    '—to stop foreign clubs from ever buying him...'\n"
    "    '—And this is how ...'\n"
    "- Bad outros:\n"
    "    '—making them the ultimate losers.' (dismissive, kills emotion)\n\n"

    "LOOP TEST — read outro into intro aloud:\n"
    "  '...to stop foreign clubs from ever buying him. | "\
    "THE BRAZILIAN GOVERNMENT DECLARED HIM A NATIONAL TREASURE...'\n"
    "  Must sound like one continuous sentence. If it does not — rewrite.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "SCRIPT TECHNICAL RULES\n"
    "═══════════════════════════════════════════════════════\n"
    "- Total duration: 28 to 35 seconds of spoken narration.\n"
    "- Word count: 70 to 105 words.\n"
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

    request_data = json.dumps(request_body).encode("utf-8")
    request = Request(
        url=f"{GEMINI_API_BASE}/{model_name}:generateContent?key={api_key}",
        data=request_data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "youtube-agent/1.0",
        },
        method="POST",
    )

    logger.info("Generating script for topic: %s with model %s", topic, model_name)

    fallback_model = getattr(settings, "GEMINI_FALLBACK_MODEL", GEMINI_FALLBACK_MODEL).strip()
    response_payload = None

    try:
        response_payload = _execute_gemini_request(request, model_name)
    except Exception as exc:
        if fallback_model and fallback_model != model_name:
            logger.warning(
                "Primary Gemini model %s failed; trying fallback model %s",
                model_name,
                fallback_model,
            )
            fallback_request = Request(
                url=f"{GEMINI_API_BASE}/{fallback_model}:generateContent?key={api_key}",
                data=request_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "youtube-agent/1.0",
                },
                method="POST",
            )
            try:
                response_payload = _execute_gemini_request(fallback_request, fallback_model)
                model_name = fallback_model
                logger.info("Gemini fallback model %s succeeded", fallback_model)
            except Exception as fallback_exc:
                logger.error(
                    "Fallback Gemini model %s also failed. Original error: %s",
                    fallback_model,
                    str(exc),
                )
                raise fallback_exc from exc
        else:
            raise

    raw_text = _extract_candidate_text(response_payload)
    result = _parse_script_payload(raw_text)
    logger.info("Script generated successfully for topic: %s", topic)
    return result

# Variable reference table:
# variable_name | type | purpose
# GEMINI_API_BASE | str | Base URL for Gemini v1beta generateContent requests.
# DEFAULT_GEMINI_MODEL | str | Default high-quality model for text generation.
# GEMINI_FALLBACK_MODEL | str | Secondary model to use when primary fails.
# GEMINI_MAX_RETRIES | int | Number of retry attempts for API calls.
# GEMINI_RETRYABLE_STATUS_CODES | set[int] | HTTP status codes treated as transient failures.
# _build_http_opener | func | Creates a URL opener using optional proxy settings.
# _extract_candidate_text | func | Extracts raw text content from Gemini response payload.
# _strip_code_fences | func | Removes markdown fences around JSON from Gemini output.
# _normalize_scene | func | Normalizes each scene dict and validates required fields.
# _parse_script_payload | func | Parses Gemini JSON output into title/script/scenes.
# _execute_gemini_request | func | Sends Gemini requests with retry and error handling.
# generate_script | func | Builds the prompt, calls Gemini, and returns the parsed payload.


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

# Final variable reference table at EOF:
# variable_name | type | purpose
# _bootstrap_django | func | Bootstraps Django settings for standalone execution.
# argparse | module | Used for CLI support when running script.py directly.
# generated | dict | The parsed title/script/scenes payload returned by generate_script.
# args | argparse.Namespace | Command-line arguments parsed for the standalone runner.
