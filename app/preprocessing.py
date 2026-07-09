from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import get_settings


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode bytes → BGR uint8 array. PIL first (JPEG/PNG) — some Docker+pip OpenCV mixes lack cv2.imdecode."""
    # PIL path avoids broken/partial cv2 wheels after uninstalling extra opencv-* packages
    try:
        pil = Image.open(BytesIO(image_bytes))
        pil.load()
        # Bake in EXIF orientation — otherwise a portrait phone photo decodes as raw sideways
        # sensor data, guaranteeing the 0° OCR pass fails and every rotation retry pass runs.
        pil = ImageOps.exif_transpose(pil)
        rgb = pil.convert("RGB")
        return np.array(rgb)[:, :, ::-1].copy()
    except Exception:
        pass
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    imdecode = getattr(cv2, "imdecode", None)
    if imdecode is not None:
        img = imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return None


def normalize_size(img: np.ndarray, max_dim: int | None = None) -> np.ndarray:
    """Downscale (never upscale) so longest side ≤ max_dim. Caps OCR cost on huge phone photos."""
    if max_dim is None:
        max_dim = get_settings().max_image_dimension
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary corner points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_document_boundary(img: np.ndarray) -> np.ndarray | None:
    """Find the document outline as a 4-corner polygon. Returns ordered TL/TR/BR/BL or None."""
    h, w = img.shape[:2]
    image_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        if cv2.contourArea(c) < image_area * 0.2:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_corners(approx)
    return None


def perspective_correct(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp `img` so the polygon `corners` (TL,TR,BR,BL) becomes axis-aligned."""
    tl, tr, br, bl = corners
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if out_w < 50 or out_h < 50:
        return img
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on luminance to recover text under uneven lighting (common with phone shots)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def rotations(img: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Yield (degrees, rotated image) for the 4 cardinal orientations."""
    return [
        (0, img),
        (90, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(img, cv2.ROTATE_180)),
        (270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]


def preprocess(image_bytes: bytes) -> np.ndarray | None:
    """Decode → normalize → optional document crop + perspective → CLAHE.
    Returns None only if decode itself fails. Falls through gracefully when no document is detected."""
    img = decode_image(image_bytes)
    if img is None:
        return None
    img = normalize_size(img)
    corners = detect_document_boundary(img)
    if corners is not None:
        img = perspective_correct(img, corners)
    return enhance_contrast(img)
