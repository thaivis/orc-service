from pathlib import Path

import cv2
import numpy as np

from app.preprocessing import (
    _order_corners,
    decode_image,
    detect_document_boundary,
    enhance_contrast,
    normalize_size,
    perspective_correct,
    rotations,
)

FIXTURES = Path(__file__).parent.parent / ".test-fixtures"


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_decode_image_returns_array_for_valid_png():
    src = np.full((20, 30, 3), 200, dtype=np.uint8)
    out = decode_image(_encode_png(src))
    assert out is not None
    assert out.shape == (20, 30, 3)


def test_decode_image_returns_none_for_garbage():
    assert decode_image(b"this is not an image") is None


def test_decode_image_applies_exif_orientation():
    """IMG_2730.jpg is a real iPhone 12 photo with EXIF orientation=6 (raw pixels are
    landscape 4032x3024; correctly displayed it's portrait 3024x4032). Without
    exif_transpose, decode_image would return the raw sideways array."""
    raw = (FIXTURES / "IMG_2730.jpg").read_bytes()
    out = decode_image(raw)
    assert out is not None
    h, w = out.shape[:2]
    assert (h, w) == (4032, 3024)


def test_normalize_size_no_upscale():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = normalize_size(img, max_dim=2000)
    assert out.shape == (100, 100, 3)


def test_normalize_size_downscales_to_cap():
    img = np.zeros((4000, 3000, 3), dtype=np.uint8)
    out = normalize_size(img, max_dim=2000)
    h, w = out.shape[:2]
    assert max(h, w) == 2000
    # aspect ratio preserved (within 1px)
    assert abs((h / w) - (4000 / 3000)) < 0.01


def test_order_corners_handles_shuffled_input():
    # Square corners delivered in random order
    raw = np.array([[100, 100], [10, 100], [100, 10], [10, 10]], dtype=np.float32)
    ordered = _order_corners(raw)
    assert tuple(ordered[0]) == (10.0, 10.0)   # TL
    assert tuple(ordered[1]) == (100.0, 10.0)  # TR
    assert tuple(ordered[2]) == (100.0, 100.0) # BR
    assert tuple(ordered[3]) == (10.0, 100.0)  # BL


def test_detect_document_boundary_finds_white_rect_on_black():
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (80, 60), (520, 340), (255, 255, 255), thickness=-1)
    corners = detect_document_boundary(canvas)
    assert corners is not None
    assert corners.shape == (4, 2)
    # TL should be near (80,60), BR near (520,340)
    assert abs(corners[0][0] - 80) < 5
    assert abs(corners[0][1] - 60) < 5
    assert abs(corners[2][0] - 520) < 5
    assert abs(corners[2][1] - 340) < 5


def test_detect_document_boundary_returns_none_on_uniform_image():
    canvas = np.full((300, 300, 3), 128, dtype=np.uint8)
    assert detect_document_boundary(canvas) is None


def test_perspective_correct_unwarps_rectangle():
    # Synthesise a known warped rectangle and verify it comes back near upright
    src_rect = np.array([[50, 60], [350, 30], [380, 250], [40, 280]], dtype=np.float32)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.fillPoly(img, [src_rect.astype(np.int32)], (255, 255, 255))
    out = perspective_correct(img, src_rect)
    h, w = out.shape[:2]
    # Output should be a near-rectangular crop with expected size from corner distances
    assert w > 200 and h > 100
    # Most pixels should be white (the filled poly)
    white_ratio = (out > 200).all(axis=2).mean()
    assert white_ratio > 0.85


def test_enhance_contrast_preserves_shape_and_dtype():
    img = np.random.randint(0, 256, size=(100, 150, 3), dtype=np.uint8)
    out = enhance_contrast(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_rotations_yields_four_orientations():
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[0, 0] = 255  # marker pixel at TL
    rot = rotations(img)
    assert [r[0] for r in rot] == [0, 90, 180, 270]
    # 0° preserves shape; 90°/270° swap dims
    assert rot[0][1].shape == (20, 30, 3)
    assert rot[1][1].shape == (30, 20, 3)
    assert rot[2][1].shape == (20, 30, 3)
    assert rot[3][1].shape == (30, 20, 3)
    # 180° moves the TL marker to BR
    assert rot[2][1][-1, -1, 0] == 255
