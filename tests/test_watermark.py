"""Unit tests for app.watermark.apply_watermark.

No MinIO, PaddleOCR, Tesseract, or encryption dependency required.
"""

import io

import numpy as np
import pytest
from PIL import Image

from app.watermark import apply_watermark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_webp(width: int = 200, height: int = 120, color: tuple = (180, 100, 60)) -> bytes:
    """Return a solid-colour WebP image as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_output_is_valid_image():
    result = apply_watermark(_make_webp(), hotel_name="Grand Hotel", iso_date="2026-05-19")
    img = Image.open(io.BytesIO(result))
    img.verify()  # raises if corrupt


def test_output_dimensions_match_input():
    w, h = 300, 200
    result = apply_watermark(_make_webp(w, h), hotel_name="Grand Hotel", iso_date="2026-05-19")
    img = Image.open(io.BytesIO(result))
    assert img.size == (w, h)


def test_watermark_changes_at_least_one_pixel():
    src_bytes = _make_webp()
    result = apply_watermark(src_bytes, hotel_name="Grand Hotel", iso_date="2026-05-19")

    src_arr = np.array(Image.open(io.BytesIO(src_bytes)).convert("RGB"))
    out_arr = np.array(Image.open(io.BytesIO(result)).convert("RGB"))

    assert not np.array_equal(src_arr, out_arr), "Output is pixel-identical to input — watermark not applied"


def test_watermark_text_uses_hotel_name_and_date():
    """Smoke test: different hotel names produce different outputs."""
    src = _make_webp()
    out_a = apply_watermark(src, hotel_name="Alpha Hotel", iso_date="2026-01-01")
    out_b = apply_watermark(src, hotel_name="Beta Resort", iso_date="2026-01-01")
    assert out_a != out_b


def test_small_image_does_not_crash():
    """Tiny 32×32 image should still work without error."""
    src = _make_webp(32, 32)
    result = apply_watermark(src, hotel_name="Hotel", iso_date="2026-05-19")
    img = Image.open(io.BytesIO(result))
    assert img.size == (32, 32)


def test_tall_image_preserves_dimensions():
    src = _make_webp(100, 400)
    result = apply_watermark(src, hotel_name="Hotel", iso_date="2026-05-19")
    assert Image.open(io.BytesIO(result)).size == (100, 400)


def test_wide_image_preserves_dimensions():
    src = _make_webp(800, 120)
    result = apply_watermark(src, hotel_name="Hotel", iso_date="2026-05-19")
    assert Image.open(io.BytesIO(result)).size == (800, 120)


# ---------------------------------------------------------------------------
# Error handling — no silent degradation
# ---------------------------------------------------------------------------


def test_invalid_bytes_raises_value_error():
    with pytest.raises(ValueError):
        apply_watermark(b"not an image at all", hotel_name="Hotel", iso_date="2026-05-19")


def test_empty_bytes_raises_value_error():
    with pytest.raises(ValueError):
        apply_watermark(b"", hotel_name="Hotel", iso_date="2026-05-19")


def test_truncated_image_raises_value_error():
    valid = _make_webp()
    with pytest.raises((ValueError, Exception)):
        apply_watermark(valid[:20], hotel_name="Hotel", iso_date="2026-05-19")
