import numpy as np

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


def _extract_mrz_lines(raw_mrz_text: str) -> tuple[str, str] | None:
    lines = [line for line in raw_mrz_text.split("\n") if line.strip()]
    candidates = [line.strip() for line in lines if "<" in line and len(line.strip()) >= 30]
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
