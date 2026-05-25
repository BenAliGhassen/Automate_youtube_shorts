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
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


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


def _parse_script_payload(raw_text: str) -> dict[str, str]:
    cleaned_text = _strip_code_fences(raw_text)

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if not match:
            raise Exception(
                "Gemini response was not valid JSON with 'title' and 'script' fields."
            )
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise Exception(
                "Gemini response contained JSON-like content, but it could not be parsed."
            ) from exc

    title = payload.get("title")
    script = payload.get("script")

    if not isinstance(title, str) or not title.strip():
        raise Exception("Gemini response did not include a valid 'title' string.")
    if not isinstance(script, str) or not script.strip():
        raise Exception("Gemini response did not include a valid 'script' string.")

    normalized_title = re.sub(r"\s+", " ", title).strip()
    normalized_script = re.sub(r"\s+", " ", script).strip()
    return {"title": normalized_title, "script": normalized_script}


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

    model_name = getattr(settings, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_GEMINI_MODEL

    prompt = (
        "You are writing a YouTube Shorts voiceover.\n"
        "Create a strong title and a spoken script about the topic below.\n"
        "The script should fit roughly 60 seconds of narration.\n\n"
        "Rules:\n"
        '- Return valid JSON only with exactly two keys: "title" and "script".\n'
        "- The title should be catchy and under 80 characters.\n"
        "- The script should be plain spoken sentences only.\n"
        "- Do not include stage directions, bullet points, scene labels, emojis, or markdown.\n"
        "- Aim for about 130 to 170 words.\n\n"
        f"Topic: {topic}"
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
