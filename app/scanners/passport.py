import numpy as np

from app.mrz import MrzParsed, _parse_name, parse_td3
from app.preprocessing import decode_image, preprocess, rotations
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
    if parsed.valid:
        warnings: list[str] = []
    elif parsed.document_number is None:
        warnings = ["mrz_incomplete"]  # line 1 read (name/country) but line 2 missing/unreadable
    else:
        warnings = ["mrz_check_digits_failed"]
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
            document_number=base_score if parsed.document_number else 0.0,
            date_of_birth=base_score if parsed.date_of_birth_iso else 0.0,
            sex=base_score if parsed.sex else 0.0,
            country=base_score if parsed.nationality else 0.0,
        ),
        warnings=warnings,
    )


def looks_like_passport(img: np.ndarray) -> bool:
    """Quick MRZ presence check on a preprocessed image. Used by thai_id scanner for type-mismatch fallback."""
    parsed = _try_parse(_get_detector(), img)
    return parsed is not None


_MRZ_OCR_CONFIG = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def _ocr_mrz_text(img: np.ndarray) -> str:
    """Direct Tesseract MRZ read — fallback for when fastmrz's ROI detector can't locate the
    band (small scans, tight crops). Tries the full frame, then the bottom strip."""
    import cv2
    import pytesseract

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(gray, lang="mrz", config=_MRZ_OCR_CONFIG)
    if _extract_mrz_lines(txt) is None:
        h = gray.shape[0]
        txt = pytesseract.image_to_string(gray[int(h * 0.6):, :], lang="mrz", config=_MRZ_OCR_CONFIG)
    return txt


def _pad_mrz(line: str) -> str:
    """Normalise an OCR'd MRZ line to the fixed 44-char TD3 width. Tesseract often drops the
    trailing '<' filler, which parse_td3 (strict len==44) would otherwise reject outright."""
    return (line + "<" * 44)[:44]


def _try_parse_direct(img: np.ndarray) -> MrzParsed | None:
    pair = _extract_mrz_lines(_ocr_mrz_text(img))
    if pair is None:
        return None
    parsed = parse_td3(_pad_mrz(pair[0]), _pad_mrz(pair[1]))
    if parsed is None or parsed.document_number is None:
        return None
    return parsed


def _parse_line1(line1: str) -> MrzParsed | None:
    """Salvage name + nationality from MRZ line 1 alone, for passports whose second line is
    missing/redacted. Gated on a well-formed 'P<CCC' + surname so OCR garbage can't slip through."""
    line1 = _pad_mrz(line1)
    if not line1.startswith("P"):
        return None
    country = line1[2:5].replace("<", "").strip()
    if len(country) != 3 or not country.isalpha():
        return None
    surname, given_names = _parse_name(line1[5:44])
    if not surname:
        return None
    return MrzParsed(
        document_number=None, surname=surname, given_names=given_names,
        nationality=country, date_of_birth_iso=None, sex=None, valid=False,
    )


def _try_parse_line1(img: np.ndarray) -> MrzParsed | None:
    for line in _ocr_mrz_text(img).splitlines():
        line = line.strip()
        if len(line) >= 25 and "<" in line:
            parsed = _parse_line1(line)
            if parsed is not None:
                return parsed
    return None


def scan_passport(image_bytes: bytes) -> tuple[ScanResponse | None, ScanError | None]:
    img = preprocess(image_bytes)
    if img is None:
        return None, ScanError("image_invalid")
    raw = decode_image(image_bytes)

    detector = _get_detector()
    fallback: MrzParsed | None = None

    # Pass 1: fastmrz on the preprocessed image (locates + OCRs the MRZ band; fast when it works).
    # Pass 2: direct Tesseract MRZ OCR on the *raw* image — recovers scans where fastmrz's ROI
    #   detector finds nothing. Raw (not preprocessed) because CLAHE/warp can smear a clean MRZ.
    for source, parser in ((img, lambda r: _try_parse(detector, r)), (raw, _try_parse_direct)):
        if source is None:
            continue
        for _deg, rotated in rotations(source):
            parsed = parser(rotated)
            if parsed is None:
                continue
            if parsed.valid:
                return _to_response(parsed), None
            if fallback is None:
                fallback = parsed

    if fallback is not None:
        return _to_response(fallback), None

    # Pass 3: no full MRZ parse — salvage name + nationality from a clean line 1 (line 2
    # missing/redacted) so we still return what's legible instead of a bare 422.
    for _deg, rotated in rotations(raw):
        parsed = _try_parse_line1(rotated)
        if parsed is not None:
            return _to_response(parsed), None

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
