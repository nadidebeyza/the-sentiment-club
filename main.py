#!/usr/bin/env python3
"""
The Sentiment Club — Instagram Content Automation Pipeline

Generates philosophy/psychology content via Gemini, renders a minimalist
dictionary-style 1080x1350 image, hosts it publicly, and publishes to Instagram.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
GEMINI_RETRY_BASE_SECONDS = int(os.getenv("GEMINI_RETRY_BASE_SECONDS", "5"))
MODELS_TO_TRY = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
RETRY_ON_503_MAX = 3
RETRY_ON_503_DELAY = 45  # seconds
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash",
    ).split(",")
    if model.strip()
]
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "")

GRAPH_API_VERSION = "v26.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
INSTAGRAM_LOGIN_API_VERSION = os.getenv("INSTAGRAM_LOGIN_API_VERSION", "v23.0")
INSTAGRAM_LOGIN_API_BASE = f"https://graph.instagram.com/{INSTAGRAM_LOGIN_API_VERSION}"

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1350
MARGIN_X = 130
DICT_START_Y = 480
COLOR_WHITE = (255, 255, 255)
COLOR_TEXT = (20, 20, 20)
COLOR_WORD = (0x7D, 0x0B, 0x1F)  # #7d0b1f — main headword
COLOR_LINE = (180, 180, 180)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGE = BASE_DIR / "final_post.jpg"
OUTPUT_STORY = BASE_DIR / "final_story.jpg"
STORY_WIDTH = 1080
STORY_HEIGHT = 1920
STORY_MARGIN_X = 120
STORY_START_Y = 700
FONT_PATH = BASE_DIR / "font.ttf"
FONT_DISPLAY_PATHS = (
    BASE_DIR / "FodaDisplay-Regular.otf",
    BASE_DIR / "Foda Display Regular.otf",
    BASE_DIR / "FodaDisplay-Regular.ttf",
    BASE_DIR / "foda-display.ttf",
    BASE_DIR / "font-display.ttf",
)
WATERMARK_TEXT = os.getenv("WATERMARK_TEXT", "@yourbrand")
WATERMARK_FONT_SIZE = 38

POLL_INTERVAL_SECONDS = 5
POLL_MAX_WAIT_SECONDS = 30
WORD_HISTORY_PATH = BASE_DIR / "word_history.json"
WORD_HISTORY_SIZE = int(os.getenv("WORD_HISTORY_SIZE", "30"))
DUPLICATE_CONTENT_RETRIES = int(os.getenv("DUPLICATE_CONTENT_RETRIES", "5"))

CONTENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "word": {
            "type": "string",
            "description": (
                "Romanized dictionary headword ONLY — lowercase hyphenated romaji. "
                "Examples: yugen, wabi-sabi, mono-no-aware. Never IPA, kanji, or English."
            ),
        },
        "subtitle": {
            "type": "string",
            "description": "Kanji or native script for the concept (e.g. 侘寂, 物の哀れ).",
        },
        "definition": {
            "type": "string",
            "description": "Concise, impactful definition — max 12 words.",
        },
        "viral_hook": {
            "type": "string",
            "description": "Legacy unused field — always set to empty string.",
        },
        "image_prompt": {
            "type": "string",
            "description": "Legacy field — always set to an empty string.",
        },
        "caption": {
            "type": "string",
            "description": (
                "Instagram caption in 3-4 short paragraphs separated by blank lines. "
                "Structure: hook paragraph, 1-2 story paragraphs with emojis, CTA paragraph."
            ),
        },
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 15,
            "maxItems": 15,
            "description": "15 lowercase hashtags without # prefix (e.g. philosophy, wabisabi).",
        },
    },
    "required": [
        "word",
        "subtitle",
        "definition",
        "viral_hook",
        "image_prompt",
        "caption",
        "hashtags",
    ],
}

MACRON_TO_ASCII = str.maketrans("āēīōūĀĒĪŌŪ", "aeiouAEIOU")
HEADWORD_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
IPA_QUALITY_MARKERS = frozenset("ːˈˌɑæəɚɜɪɔʊʌθðʃʒŋɲɴɡᵻɾɰ")

BRAND_NAME = os.getenv("BRAND_NAME", "yourbrand")

GEMINI_SYSTEM_PROMPT = f"""You are the creative director for @{BRAND_NAME} — an Instagram brand
that explores philosophy, deep Japanese concepts, psychology, and the hidden
nuances of language.

Generate ONE unique post concept that is:
- High-retention, save-worthy, and share-worthy
- Emotionally resonant without being cliché
- Rooted in authentic cultural or psychological depth

Rules:
- definition MUST be 12 words or fewer
- subtitle MUST be kanji or native script for the concept (e.g. 侘寂, 物の哀れ)
- word MUST be the romanized dictionary headword ONLY — never IPA, kanji, or English
  Format: lowercase hyphenated romaji (yugen, wabi-sabi, mono-no-aware, ikigai)
  WRONG for word: juːɡeɴ, mono no aware (use mono-no-aware), 物の哀れ, Mono No Aware
- viral_hook MUST be an empty string ""
- image_prompt MUST be an empty string ""
- caption MUST be split into 3-4 short paragraphs separated by blank lines (\\n\\n):
  1) one-line hook | 2) story/context with emojis | 3) optional depth line | 4) save/share CTA
- caption MUST NOT be one long single paragraph — keep each block 1-2 sentences max
- caption MUST be plain text only — never use markdown (**bold**, *italic*, etc.)
- caption MUST NOT use ALL CAPS words or lines for emphasis — use sentence case
- hashtags MUST be exactly 15, all lowercase, with no # prefix in the JSON array
- NEVER repeat any concept listed in the "Recently published — DO NOT REUSE" section
- Output ONLY valid JSON matching the schema — no markdown, no commentary
"""

GEMINI_SYSTEM_PROMPT_BASE = GEMINI_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentimentclub")


def _strip_markdown(text: str) -> str:
    """Remove common markdown markers — Instagram captions are plain text only."""
    cleaned = text
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"  +", " ", cleaned).strip()


def _looks_like_ipa(text: str) -> bool:
    """Detect IPA/pronunciation strings that must not be used as the headword."""
    return any(marker in text for marker in IPA_QUALITY_MARKERS) or any(
        symbol in text for symbol in ("ː", "ˈ", "ˌ", "[", "]")
    )


def _normalize_headword(word: str) -> str:
    """Normalize to compact lowercase hyphenated romaji (yugen, mono-no-aware)."""
    headword = word.strip().translate(MACRON_TO_ASCII).lower()
    headword = headword.replace("_", "-")
    headword = re.sub(r"[^a-z0-9\s-]", "", headword)
    headword = re.sub(r"\s+", "-", headword.strip())
    headword = re.sub(r"-+", "-", headword).strip("-")
    return headword


def _is_valid_headword(word: str) -> bool:
    """Headword must be romaji — not IPA, not empty, not punctuation-heavy."""
    if not word or _looks_like_ipa(word):
        return False
    return bool(HEADWORD_PATTERN.fullmatch(word))


def _normalize_hashtag(tag: str) -> str:
    """Force lowercase hashtags — Instagram tags are case-insensitive but look cleaner."""
    return tag.strip().lstrip("#").lower()


def _fix_caption_line_caps(line: str) -> str:
    """Remove shouty ALL CAPS from a single caption line."""
    stripped = line.strip()
    if stripped and stripped.isupper() and any(char.isalpha() for char in stripped):
        return stripped.capitalize()
    return re.sub(
        r"\b([A-Z]{2,})\b",
        lambda match: match.group(1).capitalize(),
        line,
    )


def _split_sentences_into_paragraphs(text: str) -> str:
    """Break a single caption block into readable hook / body / CTA paragraphs."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return text.strip()
    if len(sentences) == 1:
        return sentences[0]

    cta_keywords = ("save", "share", "bookmark", "tag someone", "send this")
    hook = sentences[0]
    body = sentences[1:]
    cta = ""
    if body and any(keyword in body[-1].lower() for keyword in cta_keywords):
        cta = body.pop()

    paragraphs = [hook]
    for index in range(0, len(body), 2):
        chunk = " ".join(body[index : index + 2]).strip()
        if chunk:
            paragraphs.append(chunk)
    if cta:
        paragraphs.append(cta)
    return "\n\n".join(paragraphs)


def _format_caption_paragraphs(caption: str) -> str:
    """Ensure caption uses blank lines between short, readable paragraphs."""
    text = caption.strip()
    if not text:
        return text

    if re.search(r"\n\s*\n", text):
        paragraphs: list[str] = []
        for block in re.split(r"\n\s*\n", text):
            lines = [_fix_caption_line_caps(line) for line in block.splitlines() if line.strip()]
            if lines:
                paragraphs.append("\n".join(lines))
        return "\n\n".join(paragraphs)

    lines = [_fix_caption_line_caps(line) for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        single = lines[0] if lines else _fix_caption_line_caps(text)
        return _split_sentences_into_paragraphs(single)

    cta_keywords = ("save", "share", "bookmark", "tag someone", "send this")
    paragraphs = [lines[0]]
    cta = lines[-1] if any(keyword in lines[-1].lower() for keyword in cta_keywords) else ""
    body = lines[1:-1] if cta else lines[1:]

    for index in range(0, len(body), 2):
        chunk_lines = body[index : index + 2]
        if chunk_lines:
            paragraphs.append("\n".join(chunk_lines))
    if cta:
        paragraphs.append(cta)
    elif lines[-1] != lines[0]:
        paragraphs.append(lines[-1])

    return "\n\n".join(paragraphs)


def _normalize_caption_text(caption: str) -> str:
    """Normalize caption emphasis and enforce paragraph spacing."""
    return _format_caption_paragraphs(_strip_markdown(caption.strip()))


@dataclass
class PostContent:
    word: str
    subtitle: str
    definition: str
    image_prompt: str
    caption: str
    hashtags: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostContent:
        hashtags = [_normalize_hashtag(tag) for tag in data["hashtags"]]
        if len(hashtags) != 15:
            raise ValueError(f"Expected 15 hashtags, got {len(hashtags)}")

        definition_words = data["definition"].split()
        if len(definition_words) > 12:
            raise ValueError(
                f"Definition exceeds 12 words ({len(definition_words)}): {data['definition']}"
            )

        word = _normalize_headword(data["word"])
        if not _is_valid_headword(word):
            raise ValueError(
                f"Invalid headword '{data['word']}'. word must be romaji like yugen or mono-no-aware, not IPA."
            )

        return cls(
            word=word,
            subtitle=_strip_markdown(data["subtitle"].strip()),
            definition=_strip_markdown(data["definition"].strip()),
            image_prompt=data.get("image_prompt", "").strip(),
            caption=_normalize_caption_text(data["caption"].strip()),
            hashtags=hashtags,
        )

    def full_caption(self) -> str:
        tags = " ".join(f"#{_normalize_hashtag(tag)}" for tag in self.hashtags)
        return f"{self.caption}\n\n{tags}"


# ---------------------------------------------------------------------------
# Stage 1 — Content Engine (Gemini)
# ---------------------------------------------------------------------------


def _uses_instagram_login_api() -> bool:
    """Instagram Login tokens (IGAA...) use graph.instagram.com, not graph.facebook.com."""
    token = INSTAGRAM_ACCESS_TOKEN or ""
    return token.startswith(("IGAA", "IGQ", "IGQV"))


def _instagram_api_base() -> str:
    return INSTAGRAM_LOGIN_API_BASE if _uses_instagram_login_api() else GRAPH_API_BASE


def _resolve_instagram_account_id() -> str:
    """Return the IG user id used for publishing endpoints."""
    if not _uses_instagram_login_api():
        if not INSTAGRAM_ACCOUNT_ID:
            raise EnvironmentError("INSTAGRAM_ACCOUNT_ID is required for Facebook Login tokens")
        return INSTAGRAM_ACCOUNT_ID

    logger.info("Resolving Instagram account via Instagram Login API (/me)...")
    data = _graph_request("GET", "me", params={"fields": "id,user_id,username"})
    account_id = data.get("user_id") or data.get("id")
    if not account_id:
        raise RuntimeError("Could not resolve Instagram account id from /me")
    logger.info(
        "Instagram account resolved — @%s (publish id: %s)",
        data.get("username", "unknown"),
        account_id,
    )
    return str(account_id)


PLACEHOLDER_MARKERS = ("your_", "changeme", "replace_me", "xxx", "example")


def _looks_like_placeholder(value: str | None) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def validate_env() -> None:
    required = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "INSTAGRAM_ACCESS_TOKEN": INSTAGRAM_ACCESS_TOKEN,
    }
    if not _uses_instagram_login_api():
        required["INSTAGRAM_ACCOUNT_ID"] = INSTAGRAM_ACCOUNT_ID

    missing = [name for name, value in required.items() if not value]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    placeholders = [name for name, value in required.items() if _looks_like_placeholder(value)]
    if placeholders:
        raise EnvironmentError(
            f"{', '.join(placeholders)} still contain placeholder values in .env. "
            "Edit .env with your real credentials (do not use .env.example values)."
        )

    if not GEMINI_API_KEY.startswith(("AIza", "AQ.")):
        raise EnvironmentError(
            "GEMINI_API_KEY must start with 'AIza' (standard) or 'AQ.' (auth key) from "
            "https://aistudio.google.com/apikey"
        )

    if not _uses_instagram_login_api():
        if not INSTAGRAM_ACCOUNT_ID.isdigit():
            raise EnvironmentError(
                f"INSTAGRAM_ACCOUNT_ID must be a numeric Instagram Business Account ID, "
                f"not '{INSTAGRAM_ACCOUNT_ID}'. Run: python main.py --lookup-ig-id"
            )

        if INSTAGRAM_ACCOUNT_ID == FACEBOOK_PAGE_ID:
            raise EnvironmentError(
                f"{INSTAGRAM_ACCOUNT_ID} is your Facebook Page ID (The Sentiment Club page), "
                "not your Instagram Business Account ID. They are two different numbers. "
                "Instagram IDs usually start with 178414. Run: python main.py --lookup-ig-id"
            )


def _gemini_models_to_try() -> list[str]:
    """Return ordered Gemini models: primary first, then valid fallbacks."""
    if os.getenv("GEMINI_FALLBACK_MODELS"):
        models = [GEMINI_MODEL]
        for model in GEMINI_FALLBACK_MODELS:
            if model not in models:
                models.append(model)
        return models

    models: list[str] = []
    for model in [GEMINI_MODEL, *MODELS_TO_TRY]:
        if model not in models:
            models.append(model)
    return models


def _is_invalid_gemini_api_key_error(exc: Exception) -> bool:
    message = str(exc)
    return "API key not valid" in message or "API_KEY_INVALID" in message


def load_word_history() -> list[dict[str, str]]:
    """Load the rolling list of recently published concepts."""
    if not WORD_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(WORD_HISTORY_PATH.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        if isinstance(entries, list):
            return entries[-WORD_HISTORY_SIZE:]
    except (OSError, json.JSONDecodeError, TypeError):
        logger.warning("Word history file unreadable — starting with empty history")
    return []


def save_word_history(entries: list[dict[str, str]]) -> None:
    """Persist the most recent N published concepts."""
    trimmed = entries[-WORD_HISTORY_SIZE:]
    WORD_HISTORY_PATH.write_text(
        json.dumps({"entries": trimmed}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def is_duplicate_content(content: PostContent, history: list[dict[str, str]]) -> bool:
    """Return True if this headword or subtitle was published recently."""
    headword = _normalize_headword(content.word)
    subtitle = content.subtitle.strip()
    for entry in history:
        if _normalize_headword(entry.get("word", "")) == headword:
            return True
        if entry.get("subtitle", "").strip() == subtitle:
            return True
    return False


def record_published_word(content: PostContent) -> None:
    """Append a successful publish to rolling word history."""
    history = load_word_history()
    history.append(
        {
            "word": content.word,
            "subtitle": content.subtitle,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_word_history(history)
    logger.info(
        "Word history updated (%s/%s tracked): %s",
        min(len(history), WORD_HISTORY_SIZE),
        WORD_HISTORY_SIZE,
        content.word,
    )


def build_gemini_prompt(history: list[dict[str, str]], *, duplicate_retry: bool = False) -> str:
    """Build Gemini prompt with a banned list of recent concepts."""
    prompt = GEMINI_SYSTEM_PROMPT_BASE
    if history:
        prompt += (
            f"\n\nRecently published — DO NOT REUSE any of these last "
            f"{len(history)} concepts:\n"
        )
        for entry in history:
            prompt += f"- {entry.get('word')} ({entry.get('subtitle', '')})\n"
        prompt += "Pick a completely different, fresh concept not on this list."
    if duplicate_retry:
        prompt += (
            "\nYour previous answer duplicated a banned concept. "
            "Generate something entirely new that is NOT similar in meaning or spelling."
        )
    return prompt


def _call_gemini_for_content(client: genai.Client, prompt: str) -> PostContent:
    """Try Gemini models sequentially until one succeeds, with 503 retry logic."""
    config = types.GenerateContentConfig(
        temperature=0.9,
        response_mime_type="application/json",
        response_json_schema=CONTENT_JSON_SCHEMA,
    )
    last_error: Exception | None = None

    for model in _gemini_models_to_try():
        # Retry loop for 503 errors on each model
        for retry_attempt in range(RETRY_ON_503_MAX + 1):
            try:
                if retry_attempt > 0:
                    logger.info("Retry attempt %d/%d for model: %s", retry_attempt, RETRY_ON_503_MAX, model)
                else:
                    logger.info("Attempting content generation with model: %s", model)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                raw = response.text
                if not raw:
                    raise ValueError("Gemini returned an empty response")
                data = json.loads(raw)
                content = PostContent.from_dict(data)
                return content
        except json.JSONDecodeError as exc:
            logger.warning(
                "Model %s returned invalid JSON (%s). Trying next fallback model...",
                model,
                exc,
            )
            last_error = exc
            break  # Move to next model
        except ValueError as exc:
            if "Invalid headword" in str(exc) or "Expected 15 hashtags" in str(exc):
                logger.warning(
                    "Model %s content validation failed (%s). Trying next fallback model...",
                    model,
                    exc,
                )
                last_error = exc
                break  # Move to next model
            raise
        except genai.errors.ClientError as exc:
            if _is_invalid_gemini_api_key_error(exc):
                raise RuntimeError(
                    "Invalid GEMINI_API_KEY. Create a key at "
                    "https://aistudio.google.com/apikey (starts with 'AIza') "
                    "and set it in your .env file."
                ) from exc
            logger.warning(
                "Model %s failed (%s). Trying next fallback model...",
                model,
                exc,
            )
            last_error = exc
            break  # Move to next model
        except genai.errors.ServerError as exc:
            error_str = str(exc)
            is_503 = "503" in error_str or "UNAVAILABLE" in error_str
            if is_503 and retry_attempt < RETRY_ON_503_MAX:
                logger.warning(
                    "Model %s returned 503 (attempt %d/%d). Waiting %ds before retry...",
                    model, retry_attempt + 1, RETRY_ON_503_MAX, RETRY_ON_503_DELAY
                )
                time.sleep(RETRY_ON_503_DELAY)
                continue  # Retry same model
            logger.warning(
                "Model %s failed (%s). Trying next fallback model...",
                model,
                exc,
            )
            last_error = exc
            break  # Move to next model
        except Exception as exc:
            logger.warning(
                "Model %s failed (%s). Trying next fallback model...",
                model,
                exc,
            )
            last_error = exc
            break  # Move to next model

    raise RuntimeError(
        "All Gemini models failed. Wait and retry, or adjust "
        "GEMINI_MODEL / GEMINI_FALLBACK_MODELS in .env."
    ) from last_error


def generate_content() -> PostContent:
    """Call Gemini and return a unique post concept not in recent history."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    history = load_word_history()
    if history:
        logger.info("Avoiding %s recent concepts from word history", len(history))

    for duplicate_attempt in range(1, DUPLICATE_CONTENT_RETRIES + 1):
        prompt = build_gemini_prompt(
            history,
            duplicate_retry=duplicate_attempt > 1,
        )
        content = _call_gemini_for_content(client, prompt)

        if is_duplicate_content(content, history):
            logger.warning(
                "Duplicate concept blocked: %s (%s) — regenerating (%s/%s)",
                content.word,
                content.subtitle,
                duplicate_attempt,
                DUPLICATE_CONTENT_RETRIES,
            )
            continue

        logger.info(
            "[✓] Gemini Content Generated — concept: %s (%s)",
            content.word,
            content.subtitle,
        )
        return content

    raise RuntimeError(
        f"Could not generate a unique concept after {DUPLICATE_CONTENT_RETRIES} attempts. "
        f"Review {WORD_HISTORY_PATH.name} or increase DUPLICATE_CONTENT_RETRIES."
    )


# ---------------------------------------------------------------------------
# Stage 2 — Visual Engine (Dictionary Style / Pillow)
# ---------------------------------------------------------------------------


def _serif_font_candidates(*, italic: bool = False) -> list[Path]:
    """System serif paths for Latin dictionary typography."""
    candidates: list[Path] = []
    if FONT_PATH.exists():
        candidates.append(FONT_PATH)

    if sys.platform == "darwin":
        candidates.extend([
            Path("/System/Library/Fonts/Supplemental/Georgia Italic.ttf" if italic else "/System/Library/Fonts/Supplemental/Georgia.ttf"),
            Path("/Library/Fonts/Georgia.ttf"),
        ])
    elif sys.platform.startswith("linux"):
        candidates.append(
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf" if italic else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
        )
    elif italic:
        candidates.append(Path("C:/Windows/Fonts/georgiai.ttf"))
    else:
        candidates.append(Path("C:/Windows/Fonts/georgia.ttf"))
    return candidates


def _kanji_font_candidates() -> list[tuple[Path, int]]:
    """Japanese/CJK fonts — prefer bold mincho/gothic faces for subtitle kanji."""
    if sys.platform == "darwin":
        return [
            (Path("/System/Library/Fonts/PingFang.ttc"), 6),
            (Path("/System/Library/Fonts/PingFang.ttc"), 5),
            (Path("/System/Library/Fonts/Hiragino Sans GB.ttc"), 2),
        ]
    if sys.platform.startswith("linux"):
        return [
            (Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"), 0),
            (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"), 0),
            (Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"), 0),
        ]
    return [
        (Path("C:/Windows/Fonts/msgothic.ttc"), 1),
        (Path("C:/Windows/Fonts/msmincho.ttc"), 0),
    ]


def _display_font_candidates() -> list[Path]:
    """Foda Display and common local paths for the brand watermark."""
    candidates: list[Path] = []
    env_path = os.getenv("FODA_DISPLAY_FONT_PATH")
    if env_path:
        candidates.append(Path(env_path))

    if FONT_PATH.exists():
        candidates.append(FONT_PATH)
    candidates.extend(FONT_DISPLAY_PATHS)

    if sys.platform == "darwin":
        candidates.extend([
            Path.home() / "Library/Fonts/Foda Display Regular.otf",
            Path.home() / "Library/Fonts/FodaDisplay-Regular.otf",
            Path("/Library/Fonts/Foda Display Regular.otf"),
            Path("/Library/Fonts/FodaDisplay-Regular.otf"),
        ])
    elif sys.platform.startswith("linux"):
        candidates.extend([
            Path.home() / ".local/share/fonts/FodaDisplay-Regular.otf",
            Path("/usr/local/share/fonts/FodaDisplay-Regular.otf"),
        ])
    else:
        candidates.append(Path("C:/Windows/Fonts/FodaDisplay-Regular.otf"))
    return candidates


def _load_custom_font(
    size: int,
    *,
    is_kanji: bool = False,
    italic: bool = False,
    display: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load font.ttf for Latin text, or a CJK-capable system font for subtitle/kanji."""
    if display:
        for path in _display_font_candidates():
            if path.exists():
                try:
                    font = ImageFont.truetype(str(path), size=size)
                    logger.info("Display watermark font loaded: %s", path.name)
                    return font
                except OSError:
                    continue
        logger.warning(
            "Foda Display not found — add FodaDisplay-Regular.otf to %s",
            BASE_DIR,
        )
        return _load_custom_font(size, italic=False)
    if is_kanji:
        for path, index in _kanji_font_candidates():
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size=size, index=index)
                except OSError:
                    continue
        label = "kanji"
    else:
        candidates = _serif_font_candidates(italic=italic)
        label = "italic serif" if italic else "serif"

    if not is_kanji:
        for path in candidates:
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size=size)
                except OSError:
                    continue

    logger.warning("No %s font found — using Pillow default bitmap font", label)
    return ImageFont.load_default()


def _line_height(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), "Ay", font=font)
    return bbox[3] - bbox[1]


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_left_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    *,
    line_spacing: int = 10,
    lowercase: bool = False,
) -> int:
    """Draw left-aligned wrapped text; return y below the block."""
    rendered = text.lower() if lowercase else text
    lines = _wrap_text(rendered, font, max_width)
    line_h = _line_height(font)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_spacing
    return y


def create_post_image(content: PostContent) -> Path:
    """Compose a minimalist dictionary-style Instagram image on pure white."""
    logger.info("Creating dictionary-style post image...")
    try:
        canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), COLOR_WHITE)
        draw = ImageDraw.Draw(canvas)

        font_kanji = _load_custom_font(60, is_kanji=True)
        font_word = _load_custom_font(110, display=True)
        font_definition = _load_custom_font(34)
        font_watermark = _load_custom_font(WATERMARK_FONT_SIZE, display=True)

        max_width = CANVAS_WIDTH - (MARGIN_X * 2)
        y = DICT_START_Y

        # 1) Subtitle / kanji — top of dictionary entry block
        draw.text((MARGIN_X, y), content.subtitle, font=font_kanji, fill=COLOR_TEXT)
        y += _line_height(font_kanji) + 28

        # 2) Main headword — large compact romaji (e.g. yugen, mono-no-aware)
        y = _draw_left_text(
            draw,
            MARGIN_X,
            y,
            content.word,
            font_word,
            COLOR_WORD,
            max_width,
            line_spacing=6,
            lowercase=False,
        )
        y += 40

        # 3) Subtle horizontal rule
        draw.line(
            (MARGIN_X, y, CANVAS_WIDTH - MARGIN_X, y),
            fill=COLOR_LINE,
            width=2,
        )
        y += 32

        # 4) Definition — lowercase, wrapped
        _draw_left_text(
            draw,
            MARGIN_X,
            y,
            content.definition,
            font_definition,
            COLOR_TEXT,
            max_width,
            line_spacing=12,
            lowercase=True,
        )

        # 5) Bottom-left watermark
        draw.text(
            (MARGIN_X, CANVAS_HEIGHT - 100),
            WATERMARK_TEXT,
            font=font_watermark,
            fill=COLOR_TEXT,
        )

        canvas.save(OUTPUT_IMAGE, format="JPEG", quality=95, optimize=True)
        logger.info("[✓] Image Created — saved to %s", OUTPUT_IMAGE)
        return OUTPUT_IMAGE
    except Exception as exc:
        logger.exception("Image creation failed")
        raise RuntimeError("Image creation failed") from exc


# ---------------------------------------------------------------------------
# Stage 3 — Hosting
# ---------------------------------------------------------------------------

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _verify_public_image_url(url: str) -> bool:
    """Return True when URL serves a direct image Instagram can fetch."""
    headers = {"User-Agent": BROWSER_USER_AGENT}
    try:
        response = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
        if response.status_code >= 400:
            response = requests.get(
                url,
                headers={**headers, "Range": "bytes=0-511"},
                timeout=30,
                allow_redirects=True,
            )
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type.startswith("image/"):
            return True
        if "text/html" in content_type:
            return False
    except requests.RequestException as exc:
        if url.startswith(("https://files.catbox.moe/", "https://litter.catbox.moe/")):
            logger.warning(
                "Could not probe %s locally (%s) — trusting Catbox CDN URL",
                url,
                exc,
            )
            return True
        return False
    return False


def upload_to_github_pages(image_path: Path) -> str:
    """Upload image to GitHub Pages (gh-pages branch) and return public URL."""
    import subprocess
    import shutil
    
    github_repo = os.getenv("GITHUB_REPOSITORY")
    
    if not github_repo:
        raise RuntimeError("GITHUB_REPOSITORY not set (not running in GitHub Actions?)")
    
    owner = github_repo.split("/")[0]
    repo_name = github_repo.split("/")[1]
    
    # Generate unique filename with timestamp to avoid caching
    timestamp = int(datetime.now(timezone.utc).timestamp())
    base_name = image_path.stem  # e.g., "final_post"
    unique_filename = f"{base_name}_{timestamp}.jpg"
    
    logger.info("Uploading image to GitHub Pages as %s...", unique_filename)
    
    original_branch = None
    temp_image = None
    
    try:
        original_branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        )
        original_branch = original_branch_result.stdout.strip()
        
        temp_image = Path("/tmp") / unique_filename
        shutil.copy2(image_path, temp_image)
        
        # Remove local file to avoid checkout conflict
        if image_path.exists():
            image_path.unlink()
        
        branch_check = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", "gh-pages"],
            capture_output=True, text=True
        )
        
        if branch_check.stdout.strip():
            subprocess.run(
                ["git", "fetch", "origin", "gh-pages"],
                capture_output=True, text=True, check=True
            )
            subprocess.run(
                ["git", "checkout", "gh-pages"],
                capture_output=True, text=True, check=True
            )
        else:
            subprocess.run(
                ["git", "checkout", "--orphan", "gh-pages"],
                capture_output=True, text=True, check=True
            )
            subprocess.run(
                ["git", "rm", "-rf", "."],
                capture_output=True, text=True
            )
            Path("index.html").write_text("<html><body>Image hosting for Instagram</body></html>")
            subprocess.run(["git", "add", "index.html"], capture_output=True, text=True, check=True)
        
        shutil.copy2(temp_image, Path(unique_filename))
        
        subprocess.run(
            ["git", "add", unique_filename],
            capture_output=True, text=True, check=True
        )
        
        commit_result = subprocess.run(
            ["git", "commit", "-m", f"Add {unique_filename}"],
            capture_output=True, text=True
        )
        
        subprocess.run(
            ["git", "push", "-f", "origin", "gh-pages"],
            capture_output=True, text=True, check=True
        )
        
        subprocess.run(
            ["git", "checkout", original_branch],
            capture_output=True, text=True, check=True
        )
        
        url = f"https://{owner}.github.io/{repo_name}/{unique_filename}"
        logger.info("[✓] Hosted at GitHub Pages — %s", url)
        
        # Wait for GitHub Pages deployment and verify URL is accessible
        logger.info("Waiting for GitHub Pages deployment (max 10 minutes)...")
        max_wait = 600  # 10 minutes
        check_interval = 15
        elapsed = 0
        
        while elapsed < max_wait:
            time.sleep(check_interval)
            elapsed += check_interval
            
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                content_type = response.headers.get("Content-Type", "")
                
                if response.status_code == 200 and "image" in content_type.lower():
                    logger.info("[✓] URL verified accessible after %d seconds", elapsed)
                    return url
                elif response.status_code == 200:
                    logger.info("URL accessible but Content-Type is '%s', waiting...", content_type)
                else:
                    logger.info("URL not ready yet (status %d), waiting... (%ds/%ds)", 
                               response.status_code, elapsed, max_wait)
            except requests.RequestException as e:
                logger.info("URL not accessible yet (%s), waiting... (%ds/%ds)", 
                           str(e)[:50], elapsed, max_wait)
        
        # If we get here, deployment took too long but let's try anyway
        logger.warning("GitHub Pages deployment took longer than expected, proceeding anyway...")
        return url
        
    except subprocess.CalledProcessError as exc:
        logger.exception("GitHub Pages upload failed: %s", getattr(exc, 'stderr', str(exc)))
        if original_branch:
            subprocess.run(["git", "checkout", original_branch], capture_output=True, text=True)
        raise RuntimeError("GitHub Pages upload failed") from exc
    finally:
        if temp_image and temp_image.exists():
            temp_image.unlink()


def host_image(image_path: Path) -> str:
    """Upload image to GitHub Pages and return direct URL for Instagram."""
    return upload_to_github_pages(image_path)


# ---------------------------------------------------------------------------
# Stage 4 — Instagram Graph API Publishing
# ---------------------------------------------------------------------------


def _graph_request(method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
    """Execute a Graph API request and raise on Meta errors."""
    api_label = "Instagram Login API" if _uses_instagram_login_api() else "Facebook Graph API"
    url = f"{_instagram_api_base()}/{endpoint.lstrip('/')}"
    params = kwargs.pop("params", {})
    params["access_token"] = INSTAGRAM_ACCESS_TOKEN

    try:
        response = requests.request(method, url, params=params, timeout=60, **kwargs)
        data = response.json()
    except requests.RequestException as exc:
        logger.exception("%s HTTP error — %s %s", api_label, method, endpoint)
        raise RuntimeError(f"{api_label} request failed: {endpoint}") from exc

    if not response.ok or "error" in data:
        error = data.get("error", {})
        message = error.get("message", response.text)
        logger.error("%s error: %s", api_label, message)
        raise RuntimeError(f"{api_label} error: {message}")
    return data


def create_media_container(image_url: str, caption: str, account_id: str) -> str:
    """Step 1: Create Instagram media container."""
    logger.info("Creating Instagram media container...")
    data = _graph_request(
        "POST",
        f"{account_id}/media",
        data={"image_url": image_url, "caption": caption},
    )
    creation_id = data.get("id")
    if not creation_id:
        raise RuntimeError("Media container response missing creation id")
    logger.info("Media container created — id: %s", creation_id)
    return creation_id


def wait_for_container_ready(container_id: str) -> None:
    """Poll container status until FINISHED or timeout."""
    logger.info("Polling container status (every %ss, max %ss)...", POLL_INTERVAL_SECONDS, POLL_MAX_WAIT_SECONDS)
    deadline = time.time() + POLL_MAX_WAIT_SECONDS
    terminal_error_states = {"ERROR", "EXPIRED"}

    while time.time() < deadline:
        data = _graph_request(
            "GET",
            container_id,
            params={"fields": "status_code,status"},
        )
        status = data.get("status_code", "UNKNOWN")
        logger.info("Container status: %s (%s)", status, data.get("status", ""))

        if status == "FINISHED":
            logger.info("[✓] Container ready for publishing")
            return
        if status in terminal_error_states:
            raise RuntimeError(f"Container entered terminal state: {status}")
        if status == "PUBLISHED":
            logger.info("Container already published")
            return

        time.sleep(POLL_INTERVAL_SECONDS)

    raise RuntimeError(
        f"Container not ready after {POLL_MAX_WAIT_SECONDS}s — aborting publish"
    )


def publish_media(container_id: str, account_id: str) -> str:
    """Step 2: Publish the media container."""
    logger.info("Publishing to Instagram...")
    data = _graph_request(
        "POST",
        f"{account_id}/media_publish",
        data={"creation_id": container_id},
    )
    media_id = data.get("id")
    if not media_id:
        raise RuntimeError("Publish response missing media id")
    logger.info("[✓] Published to Instagram — media id: %s", media_id)
    return media_id


def publish_to_instagram(image_url: str, caption: str) -> str:
    """Full Instagram publishing sequence."""
    try:
        account_id = _resolve_instagram_account_id()
        api_mode = "Instagram Login API" if _uses_instagram_login_api() else "Facebook Graph API"
        logger.info("Using %s for publishing", api_mode)
        container_id = create_media_container(image_url, caption, account_id)
        wait_for_container_ready(container_id)
        return publish_media(container_id, account_id)
    except Exception as exc:
        logger.exception("Instagram publishing failed")
        raise RuntimeError("Instagram publishing failed") from exc


def create_story_media_container(image_url: str, account_id: str) -> str:
    """Step 1: Create Instagram Story media container."""
    logger.info("Creating Instagram Story media container...")
    data = _graph_request(
        "POST",
        f"{account_id}/media",
        data={
            "image_url": image_url,
            "media_type": "STORIES",
        },
    )
    creation_id = data.get("id")
    if not creation_id:
        raise RuntimeError("Story container response missing creation id")
    logger.info("Story container created — id: %s", creation_id)
    return creation_id


def publish_story_to_instagram(image_url: str) -> str:
    """Full Instagram Story publishing sequence."""
    try:
        account_id = _resolve_instagram_account_id()
        api_mode = "Instagram Login API" if _uses_instagram_login_api() else "Facebook Graph API"
        logger.info("Using %s for Story publishing", api_mode)
        container_id = create_story_media_container(image_url, account_id)
        wait_for_container_ready(container_id)
        return publish_media(container_id, account_id)
    except Exception as exc:
        logger.exception("Instagram Story publishing failed")
        raise RuntimeError("Instagram Story publishing failed") from exc


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Execute the full content → image → host → publish pipeline."""
    logger.info("=" * 60)
    logger.info("The Sentiment Club — Instagram Automation Pipeline")
    logger.info("=" * 60)

    validate_env()

    content = generate_content()
    image_path = create_post_image(content)
    public_url = host_image(image_path)
    media_id = publish_to_instagram(public_url, content.full_caption())
    record_published_word(content)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Concept : %s", content.word)
    logger.info("  Image   : %s", image_path.resolve())
    logger.info("  URL     : %s", public_url)
    logger.info("  Media ID: %s", media_id)
    logger.info("=" * 60)


def lookup_instagram_account_id() -> None:
    """Print Instagram Business Account ID(s) for the configured access token."""
    if not INSTAGRAM_ACCESS_TOKEN:
        raise EnvironmentError("INSTAGRAM_ACCESS_TOKEN is required for lookup")

    if _uses_instagram_login_api():
        logger.info("Looking up account via Instagram Login API...")
        data = _graph_request("GET", "me", params={"fields": "id,user_id,username,name,account_type"})
        print("Token type: Instagram Login (IGAA...)")
        print(f"  Username: @{data.get('username', 'unknown')}")
        print(f"  Account type: {data.get('account_type', 'unknown')}")
        print(f"  INSTAGRAM_ACCOUNT_ID={data.get('user_id') or data.get('id')}")
        print("\nThis token uses graph.instagram.com — no Facebook Page token required.")
        return

    print(f"Facebook Page ID (NOT for publishing): {FACEBOOK_PAGE_ID}")
    print("Instagram publishing requires the separate Instagram Business Account ID.\n")
    logger.info("Looking up Instagram Business Account ID...")
    try:
        response = requests.get(
            f"{GRAPH_API_BASE}/me/accounts",
            params={
                "access_token": INSTAGRAM_ACCESS_TOKEN,
                "fields": "id,name,instagram_business_account{id,username,name}",
            },
            timeout=30,
        )
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("Failed to query Meta Graph API") from exc

    if "error" in data:
        error = data["error"]
        message = error.get("message", "Unknown error")
        raise RuntimeError(f"Graph API error: {message}")

    pages = data.get("data", [])
    if not pages:
        print("No Facebook Pages found for this token.")
        return

    found = False
    for page in pages:
        ig = page.get("instagram_business_account") or {}
        ig_id = ig.get("id")
        if ig_id:
            found = True
            username = ig.get("username", "unknown")
            print(f"Page: {page.get('name')} ({page.get('id')})")
            print(f"  INSTAGRAM_ACCOUNT_ID={ig_id}")
            print(f"  Username: @{username}")
        else:
            print(f"Page: {page.get('name')} ({page.get('id')}) — no linked Instagram account visible")

    if not found:
        print(
            "\nCould not read Instagram account IDs with your current token scopes.\n"
            "Regenerate your token in Meta Graph API Explorer with:\n"
            "  - pages_show_list\n"
            "  - pages_read_engagement\n"
            "  - instagram_basic\n"
            "  - instagram_content_publish\n\n"
            "Or find the numeric ID manually:\n"
            "  Meta Business Suite → Settings → Instagram accounts → Account details"
        )


def show_word_history() -> None:
    """Print recently published concepts tracked for duplicate avoidance."""
    history = load_word_history()
    if not history:
        print(f"No entries in {WORD_HISTORY_PATH.name} yet.")
        return
    print(f"Last {len(history)} published concepts (max {WORD_HISTORY_SIZE}):")
    for index, entry in enumerate(history, start=1):
        print(
            f"  {index:2}. {entry.get('word')} ({entry.get('subtitle', '')}) "
            f"— {entry.get('published_at', 'unknown')}"
        )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--lookup-ig-id":
        load_dotenv()
        try:
            lookup_instagram_account_id()
            return 0
        except (EnvironmentError, RuntimeError) as exc:
            logger.error("%s", exc)
            return 1

    if len(sys.argv) > 1 and sys.argv[1] == "--show-history":
        load_dotenv()
        show_word_history()
        return 0

    try:
        run_pipeline()
        return 0
    except (EnvironmentError, RuntimeError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
