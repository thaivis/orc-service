"""Diagonal text watermark overlay for ID document images."""

import io
import math

from PIL import Image, ImageDraw, ImageFont

_OPACITY = 102  # 40% of 255
_ANGLE = -30
_FONT_DIVISOR = 25  # image_width / this = font size
_MIN_FONT_SIZE = 14
_TEXT_PADDING_X = 30
_TEXT_PADDING_Y = 40


def apply_watermark(webp_bytes: bytes, hotel_name: str, iso_date: str) -> bytes:
    """Composite a diagonal text watermark onto an image at 40% opacity.

    Args:
        webp_bytes: Source image bytes (any PIL-readable format).
        hotel_name: Hotel name included in the watermark text.
        iso_date: ISO-8601 date string (e.g. "2026-05-19").

    Returns:
        WebP bytes with the watermark baked in.

    Raises:
        ValueError: If ``webp_bytes`` cannot be decoded as an image.
    """
    text = f"{hotel_name} • For hotel check-in only • {iso_date}"

    try:
        src = Image.open(io.BytesIO(webp_bytes)).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    w, h = src.size
    font_size = max(_MIN_FONT_SIZE, w // _FONT_DIVISOR)
    font = ImageFont.load_default(size=font_size)

    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Build a large canvas, tile the text, then rotate and crop to original size.
    # The canvas must be large enough that after rotation no corners are blank.
    canvas_side = int(math.sqrt(w**2 + h**2)) * 2
    tile = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    step_x = text_w + _TEXT_PADDING_X
    step_y = text_h + _TEXT_PADDING_Y

    for y in range(0, canvas_side, step_y):
        for x in range(0, canvas_side, step_x):
            draw.text((x, y), text, font=font, fill=(255, 255, 255, _OPACITY))

    rotated = tile.rotate(_ANGLE, expand=False)
    left = (canvas_side - w) // 2
    top = (canvas_side - h) // 2
    watermark = rotated.crop((left, top, left + w, top + h))

    combined = Image.alpha_composite(src, watermark).convert("RGB")

    out = io.BytesIO()
    combined.save(out, format="WEBP")
    return out.getvalue()
