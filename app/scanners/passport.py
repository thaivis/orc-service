import cv2
import numpy as np
import pytesseract

from app.mrz import MrzParsed, parse_td3
from app.preprocessing import preprocess, rotations
from app.scan_error import ScanError
from app.schemas import ConfidenceScores, DocumentType, ScanResponse, Sex

_mrz_detector = None


def _get_detector():
    global _mrz_detector
    if _mrz_detector is None:
        from fastmrz import FastMRZ

        _mrz_detector = FastMRZ()
    return _mrz_detector


def _normalise_mrz_line(line: str) -> str:
    """Replace spaces and common OCR misreads with '<' to match MRZ charset."""
    return line.strip().replace(" ", "<").replace("«", "<").replace(">", "<")


def _find_mrz_window(line: str) -> str | None:
    """Extract a 44-char MRZ substring from a noisy OCR line.
    For line 1 (passport TD3), finds the P< anchor.
    For line 2, takes the first clean 44-char run of valid MRZ chars."""
    line = _normalise_mrz_line(line)
    # TD3 line 1 always starts with P<
    if "P<" in line:
        idx = line.index("P<")
        candidate = line[idx:idx + 44]
        if len(candidate) == 44:
            return candidate
    # Generic: first 44-char window that contains < and no lowercase/spaces
    for start in range(len(line)):
        window = line[start:start + 44]
        if len(window) < 44:
            break
        if "<" in window and window == window.upper():
            return window
    # Fallback: pad or trim to 44
    if len(line) >= 30:
        return (line + "<" * 44)[:44]
    return None


def _extract_mrz_lines(raw_mrz_text: str) -> tuple[str, str] | None:
    raw_lines = [l for l in raw_mrz_text.split("\n") if l.strip()]
    candidates: list[str] = []
    for raw in raw_lines:
        window = _find_mrz_window(raw)
        if window and len(window) == 44 and "<" in window:
            candidates.append(window)
    if len(candidates) < 2:
        return None
    return candidates[-2], candidates[-1]


def _try_parse(detector, img: np.ndarray) -> MrzParsed | None:
    raw = detector.get_details(img, input_type="numpy", ignore_parse=True)
    pair = _extract_mrz_lines(raw or "")
    if pair is None:
        return None
    parsed = parse_td3(pair[0], pair[1])
    if parsed is None or parsed.document_number is None:
        return None
    return parsed


def _to_response(parsed: MrzParsed) -> ScanResponse:
    sex = Sex(parsed.sex) if parsed.sex in ("M", "F") else None
    base_score = 1.0 if parsed.valid else 0.8
    return ScanResponse(
        type=DocumentType.PASSPORT,
        first_name=parsed.given_names,
        last_name=parsed.surname,
        document_number=parsed.document_number,
        date_of_birth=parsed.date_of_birth_iso,
        sex=sex,
        country=parsed.nationality,
        document_valid=parsed.valid,
        confidence=ConfidenceScores(
            overall=base_score,
            first_name=base_score if parsed.given_names else 0.0,
            last_name=base_score if parsed.surname else 0.0,
            document_number=base_score,
            date_of_birth=base_score if parsed.date_of_birth_iso else 0.0,
            sex=base_score if parsed.sex else 0.0,
            country=base_score if parsed.nationality else 0.0,
        ),
        warnings=[] if parsed.valid else ["mrz_check_digits_failed"],
    )


def _try_parse_bottom_crop(img: np.ndarray) -> MrzParsed | None:
    """Fallback: crop bottom 22% of image and run tesseract directly, bypassing ONNX detection."""
    h = img.shape[0]
    crop = img[int(h * 0.78):, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = pytesseract.image_to_string(thresh, lang="mrz", config="--oem 3 --psm 6")
    pair = _extract_mrz_lines(raw or "")
    if pair is None:
        return None
    parsed = parse_td3(pair[0], pair[1])
    if parsed is None or parsed.document_number is None:
        return None
    return parsed


def looks_like_passport(img: np.ndarray) -> bool:
    """Quick MRZ presence check on a preprocessed image. Used by thai_id scanner for type-mismatch fallback."""
    parsed = _try_parse(_get_detector(), img)
    return parsed is not None


def scan_passport(image_bytes: bytes) -> tuple[ScanResponse | None, ScanError | None]:
    img = preprocess(image_bytes)
    if img is None:
        return None, ScanError("image_invalid")

    detector = _get_detector()
    fallback: MrzParsed | None = None
    for _deg, rotated in rotations(img):
        parsed = _try_parse(detector, rotated)
        if parsed is None:
            continue
        if parsed.valid:
            return _to_response(parsed), None
        if fallback is None:
            fallback = parsed

    if fallback is not None:
        return _to_response(fallback), None

    # ONNX detection missed — try direct bottom-crop tesseract on each rotation
    for _deg, rotated in rotations(img):
        parsed = _try_parse_bottom_crop(rotated)
        if parsed is None:
            continue
        if parsed.valid:
            return _to_response(parsed), None
        if fallback is None:
            fallback = parsed

    if fallback is not None:
        return _to_response(fallback), None

    return None, ScanError("no_document_detected")


def scan_passport_from_text(raw_mrz_text: str) -> tuple[ScanResponse | None, ScanError | None]:
    """Bypass OCR; parse MRZ text directly. Used for testing."""
    pair = _extract_mrz_lines(raw_mrz_text)
    if pair is None:
        return None, ScanError("no_document_detected")
    parsed = parse_td3(pair[0], pair[1])
    if parsed is None or parsed.document_number is None:
        return None, ScanError("no_document_detected")
    return _to_response(parsed), None
