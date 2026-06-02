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

    # Preserve the player-only and wiki-only keyword buckets when Gemini returns them.
    if isinstance(payload.get("listSportsDb"), list):
        result["listSportsDb"] = [str(k).strip() for k in payload["listSportsDb"] if str(k).strip()]
    if isinstance(payload.get("listWiki"), list):
        result["listWiki"] = [str(k).strip() for k in payload["listWiki"] if str(k).strip()]

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
    # Note: thumbnail generation uses a player image from TheSportsDB.
    # Scene keywords should be chosen so Wikimedia Commons can provide complementary visuals.
    if not topic or not topic.strip():
        raise Exception("Topic cannot be empty for script generation.")

    api_key = getattr(settings, "GEMINI_API_KEY", "").strip()
    if not api_key:
        raise Exception("GEMINI_API_KEY is not configured in Django settings.")

    model_name = getattr(settings, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_GEMINI_MODEL

    # The pipeline uses two image sources:
    # 1) The thumbnail is generated from a player-specific image fetched from TheSportsDB.
    # 2) The video visuals are assembled from complementary Wikimedia Commons images.
    # Tell Gemini so the scene keywords are selected to match this dual-source workflow.
    prompt = (
    "You are a production-grade AI scriptwriter for an automated YouTube Shorts pipeline.\n"
    "Your single output drives: text-to-speech voiceover, image fetching from TheSportsDB\n"
    "and Wikimedia Commons, video assembly, and thumbnail generation.\n"
    "Every field you return is consumed by code — precision is mandatory.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "ROLE\n"
    "═══════════════════════════════════════════════════════\n"
    "You write in the style of @ftblfacts — the most viral football facts channel on YouTube Shorts.\n"
    "Their formula: one shocking claim → rapid fire facts → loop close.\n"
    "No analysis. No storytelling. No filler. Just facts that hit like punches.\n"
    "Every sentence must make the viewer feel they just learned a secret.\n"
    "Every number must feel impossible. Every claim must demand a reaction.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "PIPELINE CONTEXT (READ BEFORE WRITING)\n"
    "═══════════════════════════════════════════════════════\n"
    "Your output feeds directly into this automated pipeline:\n"
    "  script   → edge-tts text-to-speech voiceover (10-35 seconds)\n"
    "  scenes   → image fetched per keyword: TheSportsDB first, Wikimedia fallback\n"
    "  title    → displayed as bold text on the hook thumbnail\n"
    "  format   → controls image cut speed in the video assembler\n\n"

    "IMAGE SOURCING RULES (critical for keyword writing):\n"
    "  TheSportsDB  : player cutouts and team badges — searched by EXACT player/team name\n"
    "  Wikimedia    : match photos, action shots — searched by descriptive caption\n"
    "  Priority     : TheSportsDB first → Wikimedia backup\n"
    "  NEVER use    : stadium names alone, abstract concepts, adjectives, logos\n"
    "  95% of keywords must be player full names — 5% max for team names\n"
    "  Each keyword must resolve to a real person or team that exists in both databases\n\n"

    "═══════════════════════════════════════════════════════\n"
    "FORMAT DETECTION — APPLY EXACTLY ONE\n"
    "═══════════════════════════════════════════════════════\n"
    "Detect format from topic. If ambiguous, default to FACTBOMBS.\n\n"

    "A — COMPARISON (topic contains 'vs', 'better than', 'compared to')\n"
    "  Structure : HOOK → [A stat] [B stat] → [A stat] [B stat] x3 → verdict → OUTRO\n"
    "  Rule      : strictly alternate A then B every sentence, no exceptions\n"
    "  Verdict   : take a controversial side — never neutral\n"
    "  Cut speed : 2-3s per image\n\n"
    "  length 8-15 seconds\n\n"

    "B — FACTBOMBS (topic contains 'facts', 'things', 'secrets', 'nobody knows')\n"
    "  Structure : HOOK → Fact 1 → Fact 2 → Fact 3 → Fact 4 → [Fact 5] → OUTRO\n"
    "  Rule      : number each fact with 'one.' 'two.' etc — never 'first' or 'firstly'\n"
    "  Rule      : last fact must be most shocking — it carries the loop\n"
    "  Cut speed : 3-4s per image\n\n"
    "  length 15-30 seconds\n\n"

    "C — DID YOU KNOW (topic is a single surprising fact or event)\n"
    "  Structure : HOOK (the answer) → Context → Proof → Stakes → OUTRO\n"
    "  Rule      : open with the answer, not the question — reward curiosity immediately\n"
    "  Rule      : end with the consequence — what changed because of this\n"
    "  Cut speed : 3-4s per image\n\n"
    "  length 10-25 seconds\n\n"

    "D — TOP 5 (topic contains 'top', 'best', 'worst', 'tallest', 'richest', 'fastest')\n"
    "  Structure : HOOK → #5 → #4 → #3 → #2 → #1 (longest) → OUTRO\n"
    "  Rule      : count DOWN — never up\n"
    "  Rule      : each entry is one sentence maximum\n"
    "  Rule      : #1 gets 6-8s — double other scenes — it is the payoff\n"
    "  Rule      : hook references #1 without revealing it\n"
    "  Cut speed : 3-4s per image, #1 gets 6-8s\n\n"
    "  length 8-23 seconds\n\n"

    "═══════════════════════════════════════════════════════\n"
    "SCRIPT WRITING RULES — ALL FORMATS\n"
    "═══════════════════════════════════════════════════════\n"
    "RHYTHM LAW: alternate ultra-short (3-5 words) and short (8-12 words) sentences.\n"
    "This creates a heartbeat that feels urgent even before the brain processes meaning.\n\n"

    "SENTENCE RULES:\n"
    "  - Hard maximum: 12 words per sentence. Rewrite anything longer.\n"
    "  - Use numbers instead of adjectives: not 'incredible' but '91 goals'\n"
    "  - Use 'you' to make facts personal: 'You will never see this again'\n"
    "  - Pattern interrupt every 3-4 sentences: 'Wait.' / 'Think about that.' / 'Nobody talks about this.'\n\n"

    "FORBIDDEN — instant retention killer:\n"
    "  today / in this video / let me / we are going to / subscribe\n"
    "  amazing / incredible / unbelievable / legendary / great\n"
    "  (show the fact — never name the emotion)\n\n"

    "TONE: urgent, confrontational, slightly conspiratorial.\n"
    "Sound like someone who just discovered a secret and cannot believe nobody knows it.\n\n"

    "DURATION: 28 to 35 seconds spoken. 70 to 115 words. Every second has voiceover.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "HOOK AND LOOP STRUCTURE — MANDATORY\n"
    "═══════════════════════════════════════════════════════\n"
    "The hook sentence is physically SPLIT across intro and outro.\n"
    "When the video loops, outro flows into intro as one seamless sentence.\n\n"

    "INTRO — THE BAIT:\n"
    "  - ALL CAPS\n"
    "  - Ends with em dash (—)\n"
    "  - Sounds like a rumor, a banned secret, or an impossible claim\n"
    "  - Creates an information gap the viewer must close\n"
    "  - Does NOT summarize — it baits\n\n"
    "  GOOD:\n"
    "    'THE BRAZILIAN GOVERNMENT DECLARED HIM A NATIONAL TREASURE—'\n"
    "    'FIFA TRIED TO DELETE THIS RECORD FROM HISTORY—'\n"
    "    'ONE PLAYER HELD THIS RECORD FOR 40 YEARS AND NOBODY NOTICED—'\n"
    "    'THIS NUMBER STILL HAUNTS AN ENTIRE NATION—'\n"
    "  BAD:\n"
    "    'MESSI IS THE GREATEST PLAYER EVER—' (opinion, not a secret)\n"
    "    'THIS TEAM HAD AN INCREDIBLE JOURNEY—' (adjective, no gap)\n"
    "    'NO OTHER TEAM SUFFERED THIS TRAGEDY—' (tells, does not bait)\n\n"

    "OUTRO — THE PAYOFF:\n"
    "  - Starts with em dash (—)\n"
    "  - Grammatically completes the intro sentence\n"
    "  - Ends on a strong noun or verb — never a filler word\n"
    "  - Leaves one thing unresolved — gives the viewer a reason to rewatch\n\n"
    "  GOOD:\n"
    "    '—to stop foreign clubs from ever buying him.'\n"
    "    '—and the record has never been touched since.'\n"
    "    '—and nobody in football has ever explained why.'\n"
    "  BAD:\n"
    "    '—making them the ultimate losers.' (dismissive)\n"
    "    '—which is very surprising.' (weak close)\n\n"

    "LOOP VALIDATION (run this before outputting):\n"
    "  Read: [outro] + [intro] aloud.\n"
    "  Must sound like a single grammatically correct sentence.\n"
    "  If it does not — rewrite the outro.\n\n"

    "═══════════════════════════════════════════════════════\n"
    "SCENES — TECHNICAL SPECIFICATION\n"
    "═══════════════════════════════════════════════════════\n"
    "Use 5 to 7 scenes. Scenes map 1-to-1 with images in the video.\n"
    "Duration sum MUST equal total script spoken duration exactly.\n\n"

    "KEYWORD RULES:\n"
    "  Primary keyword  : exact player full name only (e.g. 'Lionel Messi')\n"
    "  95% of keywords  : player names — TheSportsDB resolves these reliably\n"
    "  5% max           : team name only when no player fits the scene\n"
    "  NEVER            : stadiums, concepts, adjectives, logos, abstract nouns\n\n"

    "BACKUP KEYWORDS (3 required, specific → generic):\n"
    "  backup 1: player + club + context  'Lionel Messi FC Barcelona Champions League'\n"
    "  backup 2: player + club            'Messi Barcelona Camp Nou'\n"
    "  backup 3: club or country only     'FC Barcelona'\n\n"

    "SCENE ORDERING:\n"
    "  - Scenes follow strict narrative order of the script\n"
    "  - COMPARISON: strictly alternate A → B → A → B every scene\n"
    "  - TOP 5: one scene per rank, #1 scene last with longest duration\n"
    "  - Last scene keyword must visually echo first scene (creates loop)\n\n"

    "DURATION PER FORMAT:\n"
    "  COMPARISON   : 2-3s per scene\n"
    "  FACTBOMBS    : 3-4s per scene\n"
    "  DID YOU KNOW : 3-4s per scene\n"
    "  TOP 5        : 3-4s scenes #5-#2, 6-8s for scene #1\n\n"

    "═══════════════════════════════════════════════════════\n"
    "OUTPUT FORMAT — STRICT JSON\n"
    "═══════════════════════════════════════════════════════\n"
    "Return ONLY valid JSON. No preamble. No explanation. No markdown fences.\n"
    "Any deviation from this schema will break the pipeline.\n\n"
    "{\n"
    '  "format": "comparison|factbombs|didyouknow|top5",\n'
    '  "title": "max 60 chars — viral bait hook usable as thumbnail text — no adjectives",\n'
    '  "script": "ALL CAPS INTRO— body sentences. —outro completion.",\n'
    '  "scenes": [\n'
    "    {\n"
    '      "keyword": "Exact Player Full Name",\n'
    '      "backups": [\n'
    '        "Player Name Club Context",\n'
    '        "Player Name Club",\n'
    '        "Club or Country Name"\n'
    "      ],\n"
    '      "duration": 3\n'
    "    }\n"
    "  ]\n"
    "}\n\n"

    "═══════════════════════════════════════════════════════\n"
    "SELF-VALIDATION CHECKLIST — RUN BEFORE OUTPUTTING\n"
    "═══════════════════════════════════════════════════════\n"
    "Script:\n"
    "  [ ] Every sentence is 12 words or fewer\n"
    "  [ ] Rhythm alternates ultra-short and short sentences throughout\n"
    "  [ ] At least 2 pattern interrupts in the body\n"
    "  [ ] Zero forbidden words\n"
    "  [ ] Numbers used instead of adjectives everywhere possible\n"
    "  [ ] Total word count is between 70 and 115\n\n"
    "Hook and loop:\n"
    "  [ ] Intro is ALL CAPS and ends with —\n"
    "  [ ] Outro starts with — and completes intro grammatically\n"
    "  [ ] Outro + Intro read as one sentence when looped\n"
    "  [ ] Last word of outro is strong — not a filler\n"
    "  [ ] Hook creates an information gap — does not summarize\n\n"
    "Scenes:\n"
    "  [ ] 5 to 7 scenes\n"
    "  [ ] All durations sum exactly to script spoken duration\n"
    "  [ ] 95%+ keywords are exact player full names\n"
    "  [ ] Each keyword has 3 backups in order specific → generic\n"
    "  [ ] Scene order matches script narrative exactly\n"
    "  [ ] Last scene echoes first scene visually\n"
    "  [ ] COMPARISON format alternates subjects strictly\n"
    "  [ ] TOP 5 scene #1 has 6-8s duration\n\n"
    "Output:\n"
    "  [ ] Valid JSON only — no preamble, no markdown\n"
    "  [ ] format field matches detected content type\n"
    "  [ ] title is under 60 chars and works as thumbnail text\n\n"
    "FINAL RULE: If any check fails — rewrite that section. Do not output until all pass.\n\n"

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
