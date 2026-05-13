import os
import re
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.preprocessing import preprocess
from app.scan_error import ScanError
from app.schemas import ConfidenceScores, DocumentType, ScanResponse, Sex
from app.validators import thai_id_checksum

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        # Default off: MKL-DNN in paddle can segfault in slim Linux containers.
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import PaddleOCR

        # PP-OCRv5 (paddleocr 3.x) is the first version with Thai in the default rec dict.
        _ocr = PaddleOCR(
            lang="th",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ocr


@dataclass
class OcrLine:
    text: str
    confidence: float
    cx: float = 0.0
    cy: float = 0.0


def _run_ocr(img: np.ndarray) -> list[OcrLine]:
    raw = _get_ocr().predict(img)
    if not raw:
        return []
    out: list[OcrLine] = []
    for res in raw:
        for text, conf, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            out.append(
                OcrLine(
                    text=str(text).strip(),
                    confidence=float(conf),
                    cx=sum(xs) / len(xs),
                    cy=sum(ys) / len(ys),
                )
            )
    return out


def _find_below(anchor: OcrLine, lines: list[OcrLine], max_dx: float = 100.0) -> OcrLine | None:
    candidates = [
        ln for ln in lines if ln.cy > anchor.cy + 1 and abs(ln.cx - anchor.cx) < max_dx
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda ln: ln.cy - anchor.cy)


# --- ID number ---

def extract_id_number(lines: list[OcrLine]) -> tuple[str | None, float, bool]:
    """Returns (id_str_13_digits, confidence, checksum_valid). Prefers checksum-valid match."""
    fallback: tuple[str, float] | None = None
    for line in lines:
        digits = re.sub(r"\D", "", line.text)
        if len(digits) == 13:
            if thai_id_checksum(digits):
                return digits, line.confidence, True
            if fallback is None:
                fallback = (digits, line.confidence)
    if fallback is not None:
        return fallback[0], fallback[1], False
    return None, 0.0, False


# --- Sex (derived from Thai title prefix) ---

THAI_TITLE_TO_SEX: dict[str, str] = {
    "เด็กชาย": "M",
    "เด็กหญิง": "F",
    "นางสาว": "F",  # check before "นาง" since it's a longer prefix containing "นาง"
    "นาย": "M",
    "นาง": "F",
}

def extract_sex(lines: list[OcrLine]) -> tuple[Sex | None, float]:
    for line in lines:
        for prefix, s in THAI_TITLE_TO_SEX.items():
            if prefix in line.text:
                return Sex(s), line.confidence
    return None, 0.0


# --- Names ---

EN_TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Miss|Master|Mister)\.?\s+([A-Z][A-Za-z\-]+)", re.IGNORECASE)


def _strip_title(text: str) -> str:
    m = EN_TITLE_RE.search(text)
    if m:
        return text[m.start():].split(None, 1)[-1].strip()
    return text.strip()


def _first_alpha_token(text: str) -> str | None:
    for tok in text.split():
        cleaned = re.sub(r"[^A-Za-z\-]", "", tok)
        if cleaned and cleaned.upper() not in {"MR", "MRS", "MISS", "MASTER", "MISTER"}:
            return cleaned.upper()
    return None


def extract_first_name(lines: list[OcrLine]) -> tuple[str | None, float]:
    # Strategy A: explicit English title prefix anywhere
    for line in lines:
        m = EN_TITLE_RE.search(line.text)
        if m:
            return m.group(1).upper(), line.confidence
    # Strategy B: anchor on "Name" label (excluding "Last Name")
    for line in lines:
        low = line.text.lower()
        if "name" in low and "last" not in low:
            after = re.sub(r"^.*?name\s*[:\-]?\s*", "", line.text, count=1, flags=re.IGNORECASE)
            tok = _first_alpha_token(after)
            if tok:
                return tok, line.confidence
            below = _find_below(line, lines)
            if below:
                tok = _first_alpha_token(_strip_title(below.text))
                if tok:
                    return tok, below.confidence
    return None, 0.0


def extract_last_name(lines: list[OcrLine]) -> tuple[str | None, float]:
    for line in lines:
        low = line.text.lower()
        if "last name" in low or "lastname" in low or "last  name" in low:
            after = re.sub(r"^.*?last\s*name\s*[:\-]?\s*", "", line.text, count=1, flags=re.IGNORECASE)
            tok = _first_alpha_token(after)
            if tok:
                return tok, line.confidence
            below = _find_below(line, lines)
            if below:
                tok = _first_alpha_token(below.text)
                if tok:
                    return tok, below.confidence
    return None, 0.0


# --- DOB ---

EN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
TH_MONTHS = {
    "ม.ค.": 1, "ม.ค": 1, "มกราคม": 1,
    "ก.พ.": 2, "ก.พ": 2, "กุมภาพันธ์": 2,
    "มี.ค.": 3, "มี.ค": 3, "มีนาคม": 3,
    "เม.ย.": 4, "เม.ย": 4, "เมษายน": 4,
    "พ.ค.": 5, "พ.ค": 5, "พฤษภาคม": 5,
    "มิ.ย.": 6, "มิ.ย": 6, "มิถุนายน": 6,
    "ก.ค.": 7, "ก.ค": 7, "กรกฎาคม": 7,
    "ส.ค.": 8, "ส.ค": 8, "สิงหาคม": 8,
    "ก.ย.": 9, "ก.ย": 9, "กันยายน": 9,
    "ต.ค.": 10, "ต.ค": 10, "ตุลาคม": 10,
    "พ.ย.": 11, "พ.ย": 11, "พฤศจิกายน": 11,
    "ธ.ค.": 12, "ธ.ค": 12, "ธันวาคม": 12,
}

DATE_EN_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,})\.?\s+(\d{4})\b")


def _be_to_ce(year: int) -> int:
    return year - 543 if year > 2400 else year


def _try_make_date(year: int, month: int, day: int) -> date | None:
    try:
        d = date(year, month, day)
    except ValueError:
        return None
    if d > date.today():
        return None
    return d


def _parse_thai_date(text: str) -> date | None:
    m = re.search(r"(\d{1,2})\s+(\S+?)\s+(\d{4})", text)
    if not m:
        return None
    day = int(m.group(1))
    month = TH_MONTHS.get(m.group(2))
    if month is None:
        return None
    year = _be_to_ce(int(m.group(3)))
    return _try_make_date(year, month, day)


def extract_dob(lines: list[OcrLine]) -> tuple[date | None, float]:
    # English date first (more reliable on Thai ID)
    for line in lines:
        m = DATE_EN_RE.search(line.text)
        if m:
            month = EN_MONTHS.get(m.group(2)[:3].lower())
            if month is None:
                continue
            day = int(m.group(1))
            year = _be_to_ce(int(m.group(3)))
            d = _try_make_date(year, month, day)
            if d:
                return d, line.confidence
    # Thai date fallback
    for line in lines:
        d = _parse_thai_date(line.text)
        if d:
            return d, line.confidence
    return None, 0.0


# --- Orchestration ---

def looks_like_thai_id(img: np.ndarray) -> bool:
    """Quick check on a preprocessed image: contains either a valid 13-digit ID or a Thai title prefix.
    Used by the passport scanner for type-mismatch fallback."""
    lines = _run_ocr(img)
    if not lines:
        return False
    for line in lines:
        digits = re.sub(r"\D", "", line.text)
        if len(digits) == 13 and thai_id_checksum(digits):
            return True
    for line in lines:
        if any(prefix in line.text for prefix in THAI_TITLE_TO_SEX):
            return True
    return False


def scan_thai_id_from_lines(lines: list[OcrLine]) -> tuple[ScanResponse | None, ScanError | None]:
    """Build ScanResponse from already-OCR'd lines. Pure function — used directly by tests."""
    id_num, id_conf, id_valid = extract_id_number(lines)
    if id_num is None:
        return None, ScanError("no_document_detected")

    first, first_conf = extract_first_name(lines)
    last, last_conf = extract_last_name(lines)
    dob, dob_conf = extract_dob(lines)
    sex, sex_conf = extract_sex(lines)

    field_confs = [id_conf, first_conf, last_conf, dob_conf, sex_conf]
    populated = [c for c in field_confs if c > 0]
    overall = sum(populated) / len(populated) if populated else 0.0

    warnings: list[str] = []
    if not id_valid:
        warnings.append("thai_id_checksum_failed")

    return (
        ScanResponse(
            type=DocumentType.THAI_ID,
            first_name=first,
            last_name=last,
            document_number=id_num,
            date_of_birth=dob,
            sex=sex,
            country="THA",
            document_valid=id_valid,
            confidence=ConfidenceScores(
                overall=overall,
                first_name=first_conf,
                last_name=last_conf,
                document_number=id_conf,
                date_of_birth=dob_conf,
                sex=sex_conf,
                country=1.0,
            ),
            warnings=warnings,
        ),
        None,
    )


def scan_thai_id(image_bytes: bytes) -> tuple[ScanResponse | None, ScanError | None]:
    img = preprocess(image_bytes)
    if img is None:
        return None, ScanError("image_invalid")
    lines = _run_ocr(img)
    return scan_thai_id_from_lines(lines)
