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
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",")
    if model.strip()
]
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "1296052766920644")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")

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
COLOR_LINE = (180, 180, 180)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGE = BASE_DIR / "final_post.jpg"
FONT_PATH = BASE_DIR / "font.ttf"
FONT_DISPLAY_PATHS = (
    BASE_DIR / "FodaDisplay-Regular.otf",
    BASE_DIR / "Foda Display Regular.otf",
    BASE_DIR / "FodaDisplay-Regular.ttf",
    BASE_DIR / "foda-display.ttf",
    BASE_DIR / "font-display.ttf",
)
WATERMARK_TEXT = "@thesentimentclub.co"
WATERMARK_FONT_SIZE = 38

POLL_INTERVAL_SECONDS = 5
POLL_MAX_WAIT_SECONDS = 30

CONTENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "word": {
            "type": "string",
            "description": "Main keyword or concept (e.g. Kintsugi, Mono no aware).",
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
            "description": (
                "Dictionary phonetic line for the image, e.g. \"[wa:bɪ 'sa:bɪ] noun • Japanese\" "
                "using valid IPA inside brackets, part of speech, and source language."
            ),
        },
        "image_prompt": {
            "type": "string",
            "description": "Legacy field — always set to an empty string.",
        },
        "caption": {
            "type": "string",
            "description": (
                "Instagram caption in plain text only (no markdown): scroll-stopping hook, "
                "3-4 line story with emojis, and a save/share CTA."
            ),
        },
        "hashtags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 15,
            "maxItems": 15,
            "description": "15 hashtags: 5 broad, 5 niche, 5 high-intent community tags.",
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

GEMINI_SYSTEM_PROMPT = """You are the creative director for @thesentimentclub — an Instagram brand
that explores philosophy, deep Japanese concepts, psychology, and the hidden
nuances of language.

Generate ONE unique post concept that is:
- High-retention, save-worthy, and share-worthy
- Emotionally resonant without being cliché
- Rooted in authentic cultural or psychological depth

Rules:
- definition MUST be 12 words or fewer
- subtitle MUST be kanji or native script for the concept (e.g. 侘寂, 物の哀れ)
- viral_hook MUST be a dictionary phonetic line with valid IPA, e.g. "[wa:bɪ 'sa:bɪ] noun • Japanese"
- word MUST be the romanized/lowercase entry headword (e.g. wabi-sabi)
- image_prompt MUST be an empty string ""
- caption MUST include: a punchy first-line hook, 3-4 story lines with emojis,
  and a clear CTA to SAVE and SHARE
- caption MUST be plain text only — never use markdown (**bold**, *italic*, etc.)
- hashtags MUST be exactly 15 (5 broad, 5 niche, 5 high-intent community)
- Output ONLY valid JSON matching the schema — no markdown, no commentary
"""

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


@dataclass
class PostContent:
    word: str
    subtitle: str
    definition: str
    viral_hook: str
    image_prompt: str
    caption: str
    hashtags: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostContent:
        hashtags = [tag.strip().lstrip("#") for tag in data["hashtags"]]
        if len(hashtags) != 15:
            raise ValueError(f"Expected 15 hashtags, got {len(hashtags)}")
        definition_words = data["definition"].split()
        if len(definition_words) > 12:
            raise ValueError(
                f"Definition exceeds 12 words ({len(definition_words)}): {data['definition']}"
            )
        return cls(
            word=_strip_markdown(data["word"].strip()),
            subtitle=_strip_markdown(data["subtitle"].strip()),
            definition=_strip_markdown(data["definition"].strip()),
            viral_hook=_strip_markdown(data["viral_hook"].strip()),
            image_prompt=data["image_prompt"].strip(),
            caption=_strip_markdown(data["caption"].strip()),
            hashtags=hashtags,
        )

    def full_caption(self) -> str:
        tags = " ".join(f"#{tag}" for tag in self.hashtags)
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
    """Primary model first, then configured fallbacks."""
    models = [GEMINI_MODEL]
    for model in GEMINI_FALLBACK_MODELS:
        if model not in models:
            models.append(model)
    return models


def _is_transient_gemini_error(exc: Exception) -> bool:
    """Return True for temporary Gemini capacity/rate-limit failures."""
    if isinstance(exc, genai.errors.ServerError):
        code = getattr(exc, "status_code", None)
        if code in {429, 500, 503}:
            return True
    message = str(exc).upper()
    return any(token in message for token in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def generate_content() -> PostContent:
    """Call Gemini and return validated post content."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        temperature=0.9,
        response_mime_type="application/json",
        response_json_schema=CONTENT_JSON_SCHEMA,
    )
    last_error: Exception | None = None

    for model in _gemini_models_to_try():
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                logger.info(
                    "Generating content with %s (attempt %s/%s)...",
                    model,
                    attempt,
                    GEMINI_MAX_RETRIES,
                )
                response = client.models.generate_content(
                    model=model,
                    contents=GEMINI_SYSTEM_PROMPT,
                    config=config,
                )
                raw = response.text
                if not raw:
                    raise ValueError("Gemini returned an empty response")
                data = json.loads(raw)
                content = PostContent.from_dict(data)
                logger.info("[✓] Gemini Content Generated — concept: %s", content.word)
                return content
            except json.JSONDecodeError as exc:
                logger.exception("Gemini returned invalid JSON")
                raise RuntimeError("Failed to parse Gemini JSON output") from exc
            except genai.errors.ClientError as exc:
                if "API key not valid" in str(exc) or "API_KEY_INVALID" in str(exc):
                    raise RuntimeError(
                        "Invalid GEMINI_API_KEY. Create a key at "
                        "https://aistudio.google.com/apikey (starts with 'AIza') "
                        "and set it in your .env file."
                    ) from exc
                if _is_transient_gemini_error(exc) and attempt < GEMINI_MAX_RETRIES:
                    wait = GEMINI_RETRY_BASE_SECONDS * attempt
                    logger.warning("Gemini transient error on %s — retrying in %ss...", model, wait)
                    time.sleep(wait)
                    last_error = exc
                    continue
                logger.exception("Gemini API request failed")
                raise RuntimeError(f"Gemini API error: {exc}") from exc
            except genai.errors.ServerError as exc:
                if _is_transient_gemini_error(exc) and attempt < GEMINI_MAX_RETRIES:
                    wait = GEMINI_RETRY_BASE_SECONDS * attempt
                    logger.warning(
                        "Gemini high demand (503) on %s — retrying in %ss...",
                        model,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue
                logger.exception("Gemini server error on model %s", model)
                last_error = exc
                break
            except Exception as exc:
                logger.exception("Gemini content generation failed")
                raise RuntimeError("Gemini content generation failed") from exc

        if model != _gemini_models_to_try()[-1]:
            logger.warning("Switching Gemini model fallback from %s...", model)

    raise RuntimeError(
        "Gemini is temporarily unavailable (503 high demand). "
        "Wait a minute and run again, or set GEMINI_MODEL to a fallback in .env."
    ) from last_error


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


def _ipa_font_candidates() -> list[Path]:
    """Fonts with IPA / extended Latin coverage for phonetic pronunciation lines."""
    if sys.platform == "darwin":
        return [
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
        ]
    if sys.platform.startswith("linux"):
        return [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        ]
    return [
        Path("C:/Windows/Fonts/ARIALUNI.TTF"),
        Path("C:/Windows/Fonts/arialuni.ttf"),
    ]


def _load_custom_font(
    size: int,
    *,
    is_kanji: bool = False,
    italic: bool = False,
    ipa: bool = False,
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
    if ipa:
        candidates = _ipa_font_candidates()
        label = "IPA"
    elif is_kanji:
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


def _format_phonetic_line(content: PostContent) -> str:
    """Build dictionary meta line from viral_hook or a sensible fallback."""
    hook = content.viral_hook.strip()
    if hook and ("•" in hook or hook.startswith("[")):
        return hook
    return f"[{content.word.lower()}] noun • Japanese"


def create_post_image(content: PostContent) -> Path:
    """Compose a minimalist dictionary-style Instagram image on pure white."""
    logger.info("Creating dictionary-style post image...")
    try:
        canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), COLOR_WHITE)
        draw = ImageDraw.Draw(canvas)

        font_kanji = _load_custom_font(60, is_kanji=True)
        font_word = _load_custom_font(110)
        font_meta = _load_custom_font(40, ipa=True)
        font_definition = _load_custom_font(34)
        font_watermark = _load_custom_font(WATERMARK_FONT_SIZE, display=True)

        max_width = CANVAS_WIDTH - (MARGIN_X * 2)
        y = DICT_START_Y

        # 1) Subtitle / kanji — top of dictionary entry block
        draw.text((MARGIN_X, y), content.subtitle, font=font_kanji, fill=COLOR_TEXT)
        y += _line_height(font_kanji) + 28

        # 2) Main headword — large lowercase serif
        y = _draw_left_text(
            draw,
            MARGIN_X,
            y,
            content.word,
            font_word,
            COLOR_TEXT,
            max_width,
            line_spacing=6,
            lowercase=True,
        )
        y += 40

        # 3) IPA phonetic + part of speech + language
        meta_line = _format_phonetic_line(content)
        draw.text((MARGIN_X, y), meta_line, font=font_meta, fill=COLOR_TEXT)
        y += _line_height(font_meta) + 36

        # 4) Subtle horizontal rule
        draw.line(
            (MARGIN_X, y, CANVAS_WIDTH - MARGIN_X, y),
            fill=COLOR_LINE,
            width=2,
        )
        y += 32

        # 5) Definition — lowercase, wrapped
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

        # 6) Bottom-left watermark
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


def upload_to_catbox(image_path: Path) -> str:
    """Upload image to catbox.moe and return direct HTTPS URL."""
    logger.info("Uploading image to Catbox...")
    try:
        with image_path.open("rb") as handle:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (image_path.name, handle, "image/jpeg")},
                timeout=60,
            )
        response.raise_for_status()
        url = response.text.strip()
        if not url.startswith("https://"):
            raise ValueError(f"Unexpected Catbox response: {url[:200]}")
        logger.info("[✓] Hosted at Catbox — %s", url)
        return url
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Catbox upload failed")
        raise RuntimeError("Catbox upload failed") from exc


def upload_to_imgur(image_path: Path) -> str:
    """Fallback upload to Imgur (requires IMGUR_CLIENT_ID)."""
    if not IMGUR_CLIENT_ID:
        raise RuntimeError("IMGUR_CLIENT_ID not set — cannot use Imgur fallback")
    logger.info("Uploading image to Imgur (fallback)...")
    try:
        with image_path.open("rb") as handle:
            response = requests.post(
                "https://api.imgur.com/3/upload",
                headers={"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"},
                files={"image": handle},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        url = payload["data"]["link"]
        logger.info("[✓] Hosted at Imgur — %s", url)
        return url
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.exception("Imgur upload failed")
        raise RuntimeError("Imgur upload failed") from exc


def host_image(image_path: Path) -> str:
    """Upload to Catbox with Imgur fallback."""
    try:
        return upload_to_catbox(image_path)
    except RuntimeError:
        logger.warning("Catbox failed — attempting Imgur fallback...")
        return upload_to_imgur(image_path)


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


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--lookup-ig-id":
        load_dotenv()
        try:
            lookup_instagram_account_id()
            return 0
        except (EnvironmentError, RuntimeError) as exc:
            logger.error("%s", exc)
            return 1

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
