#!/usr/bin/env python3
"""
The Sentiment Club — Instagram Story Automation

Generates dictionary-style vertical Story images (1080x1920) and publishes
via the same Instagram token as the feed post pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from main import (
    COLOR_LINE,
    COLOR_TEXT,
    COLOR_WHITE,
    COLOR_WORD,
    OUTPUT_STORY,
    STORY_HEIGHT,
    STORY_MARGIN_X,
    STORY_START_Y,
    STORY_WIDTH,
    WATERMARK_FONT_SIZE,
    WATERMARK_TEXT,
    PostContent,
    _draw_left_text,
    _line_height,
    _load_custom_font,
    generate_content,
    host_image,
    publish_story_to_instagram,
    record_published_word,
    validate_env,
)

logger = logging.getLogger("sentimentclub_story")


def create_story_image(content: PostContent) -> Path:
    """Compose a vertical dictionary-style Story image on pure white."""
    logger.info("Creating dictionary-style Story image...")
    canvas = Image.new("RGB", (STORY_WIDTH, STORY_HEIGHT), COLOR_WHITE)
    draw = ImageDraw.Draw(canvas)

    font_kanji = _load_custom_font(70, is_kanji=True)
    font_word = _load_custom_font(120, display=True)
    font_definition = _load_custom_font(38)
    font_watermark = _load_custom_font(WATERMARK_FONT_SIZE, display=True)

    max_width = STORY_WIDTH - (STORY_MARGIN_X * 2)
    y = STORY_START_Y

    draw.text((STORY_MARGIN_X, y), content.subtitle, font=font_kanji, fill=COLOR_TEXT)
    y += _line_height(font_kanji) + 32

    y = _draw_left_text(
        draw,
        STORY_MARGIN_X,
        y,
        content.word,
        font_word,
        COLOR_WORD,
        max_width,
        line_spacing=8,
        lowercase=False,
    )
    y += 48

    draw.line(
        (STORY_MARGIN_X, y, STORY_WIDTH - STORY_MARGIN_X, y),
        fill=COLOR_LINE,
        width=2,
    )
    y += 36

    _draw_left_text(
        draw,
        STORY_MARGIN_X,
        y,
        content.definition,
        font_definition,
        COLOR_TEXT,
        max_width,
        line_spacing=14,
        lowercase=True,
    )

    draw.text(
        (STORY_MARGIN_X, STORY_HEIGHT - 220),
        WATERMARK_TEXT,
        font=font_watermark,
        fill=COLOR_TEXT,
    )

    canvas.save(OUTPUT_STORY, format="JPEG", quality=95, optimize=True)
    logger.info("[✓] Story Image Created — saved to %s", OUTPUT_STORY)
    return OUTPUT_STORY


def run_story_pipeline() -> None:
    """Execute the full Story content → image → host → publish pipeline."""
    logger.info("=" * 60)
    logger.info("The Sentiment Club — Instagram Story Pipeline")
    logger.info("=" * 60)

    validate_env()

    content = generate_content()
    image_path = create_story_image(content)
    public_url = host_image(image_path)
    media_id = publish_story_to_instagram(public_url)
    record_published_word(content)

    logger.info("=" * 60)
    logger.info("Story pipeline complete!")
    logger.info("  Concept : %s", content.word)
    logger.info("  Image   : %s", image_path.resolve())
    logger.info("  URL     : %s", public_url)
    logger.info("  Media ID: %s", media_id)
    logger.info("=" * 60)


def main() -> int:
    try:
        run_story_pipeline()
        return 0
    except (EnvironmentError, RuntimeError) as exc:
        logger.error("Story pipeline failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Story pipeline interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
