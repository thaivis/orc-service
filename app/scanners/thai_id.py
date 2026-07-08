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
    """Returns (id_str_13_digits, confidence, checksum_valid). Prefers checksum-valid match.

    Handles two OCR shapes: a single line holding the whole number, and a number that
    PaddleOCR split across several detections (e.g. '1' '8399' '01779' '295') — common on
    sharp, straight scans where each digit group becomes its own text box."""
    fallback: tuple[str, float] | None = None
    # Fast path: one line already contains all 13 digits.
    for line in lines:
        digits = re.sub(r"\D", "", line.text)
        if len(digits) == 13:
            if thai_id_checksum(digits):
                return digits, line.confidence, True
            if fallback is None:
                fallback = (digits, line.confidence)
    # Fragmented path: concatenate consecutive digit runs and accept the first checksum-valid
    # 13-digit window. The checksum gates against false joins (dates, card/laser numbers).
    runs: list[tuple[str, float]] = [
        (m.group(), line.confidence)
        for line in lines
        for m in re.finditer(r"\d+", line.text)
    ]
    for i in range(len(runs)):
        joined = ""
        confs: list[float] = []
        for digits, conf in runs[i:]:
            joined += digits
            confs.append(conf)
            if len(joined) >= 13:
                if len(joined) == 13 and thai_id_checksum(joined):
                    return joined, sum(confs) / len(confs), True
                break
    if fallback is not None:
        return fallback[0], fallback[1], False
    return None, 0.0, False


# --- Sex (derived from Thai title prefix) ---

THAI_TITLE_TO_SEX: dict[str, str] = {
    "เด็กชาย": "M",
    "เด็กหญิง": "F",
    "นางสาว": "F",  # check before "นาง" since it's a longer prefix containing "นาง"
    "น.ส.": "F",
    "นาย": "M",
    "นาง": "F",
}

EN_TITLE_TO_SEX = {"mr": "M", "mister": "M", "master": "M", "mrs": "F", "miss": "F"}


def extract_sex(lines: list[OcrLine]) -> tuple[Sex | None, float]:
    for line in lines:
        for prefix, s in THAI_TITLE_TO_SEX.items():
            if prefix in line.text:
                return Sex(s), line.confidence
    # English title fallback — some cards OCR the Thai title poorly (e.g. 'น.ส.' → 'Miss')
    for line in lines:
        m = re.search(r"\b(mr|mrs|miss|master|mister)\b", line.text, re.IGNORECASE)
        if m:
            return Sex(EN_TITLE_TO_SEX[m.group(1).lower()]), line.confidence
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


NAME_STOPWORDS = {"MR", "MRS", "MISS", "MASTER", "MISTER", "LAST", "NAME", "LASTNAME"}


GLUED_TITLE_RE = re.compile(r"^(?:mr|mrs|miss|master|mister)\.?", re.IGNORECASE)


def _name_token(text: str) -> str | None:
    """First A-Za-z token in `text`, upper-cased, skipping titles and label words.

    Also peels a title glued to the name when OCR drops the space ("Mr.Channarong")."""
    for tok in _strip_title(text).split():
        cleaned = re.sub(r"[^A-Za-z\-]", "", tok)
        deglued = GLUED_TITLE_RE.sub("", cleaned)
        if len(deglued) >= 2:  # only peel when a real name remains
            cleaned = deglued
        cleaned = cleaned.upper()
        if cleaned and cleaned not in NAME_STOPWORDS:
            return cleaned
    return None


def _row_tol(lines: list[OcrLine]) -> float:
    """Vertical tolerance for 'same row', scaled to the image so it holds at any resolution."""
    cys = [ln.cy for ln in lines]
    if not cys:
        return 8.0
    return max(8.0, (max(cys) - min(cys)) * 0.02)


def _same_row_right(anchor: OcrLine, lines: list[OcrLine], tol: float) -> list[OcrLine]:
    """Lines on the same text row as `anchor`, to its right, ordered left-to-right.

    PaddleOCR often emits a field's label and value as separate boxes with the value to the
    right; searching 'below' misses those, so we scan rightward on the row first."""
    same = [
        ln for ln in lines
        if ln is not anchor and abs(ln.cy - anchor.cy) <= tol and ln.cx > anchor.cx
    ]
    return sorted(same, key=lambda ln: ln.cx)


def _name_candidates(anchor: OcrLine, lines: list[OcrLine], tol: float) -> list[OcrLine]:
    """Value boxes to try for a label: same-row-right first, then the line below."""
    cands = _same_row_right(anchor, lines, tol)
    below = _find_below(anchor, lines)
    if below is not None:
        cands.append(below)
    return cands


def extract_first_name(lines: list[OcrLine]) -> tuple[str | None, float]:
    # Strategy A: explicit English title prefix inline ("Mr. KITTIKHUN")
    for line in lines:
        m = EN_TITLE_RE.search(line.text)
        if m:
            return m.group(1).upper(), line.confidence
    # Strategy B: anchor on the "Name" label (excluding "Last Name"); value may be inline,
    # to the right on the same row, or on the line below.
    tol = _row_tol(lines)
    for line in lines:
        low = line.text.lower()
        if "name" in low and "last" not in low:
            after = re.sub(r"^.*?name\s*[:\-]?\s*", "", line.text, count=1, flags=re.IGNORECASE)
            tok = _name_token(after)
            if tok:
                return tok, line.confidence
            for cand in _name_candidates(line, lines, tol):
                tok = _name_token(cand.text)
                if tok:
                    return tok, cand.confidence
    return None, 0.0


def extract_last_name(lines: list[OcrLine]) -> tuple[str | None, float]:
    tol = _row_tol(lines)
    for line in lines:
        low = line.text.lower()
        if re.search(r"last\s*nam", low):  # tolerate OCR dropping the trailing 'e' ("Last nam")
            after = re.sub(r"^.*?last\s*nam(?:e)?\s*[:\-]?\s*", "", line.text, count=1, flags=re.IGNORECASE)
            tok = _name_token(after)
            if tok:
                return tok, line.confidence
            for cand in _name_candidates(line, lines, tol):
                tok = _name_token(cand.text)
                if tok:
                    return tok, cand.confidence
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


def _parse_en_date(text: str) -> date | None:
    m = DATE_EN_RE.search(text)
    if not m:
        return None
    month = EN_MONTHS.get(m.group(2)[:3].lower())
    if month is None:
        return None
    return _try_make_date(_be_to_ce(int(m.group(3))), month, int(m.group(1)))


def extract_dob(lines: list[OcrLine]) -> tuple[date | None, float]:
    tol = _row_tol(lines)
    # Anchored: reconstruct the date from the "Date of Birth" / "เกิด" row. This both avoids
    # grabbing the issue/expiry date and stitches a date PaddleOCR split into separate tokens.
    for line in lines:
        if "birth" in line.text.lower() or "เกิด" in line.text:
            row_text = " ".join([line.text] + [c.text for c in _same_row_right(line, lines, tol)])
            d = _parse_en_date(row_text) or _parse_thai_date(row_text)
            if d:
                return d, line.confidence
    # Fallback: first parseable date anywhere (English preferred, then Thai).
    for line in lines:
        d = _parse_en_date(line.text)
        if d:
            return d, line.confidence
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


def _has_id_card_markers(lines: list[OcrLine]) -> bool:
    """True when the lines carry unmistakable Thai-ID header text, even if the number itself
    didn't OCR. Lets low-res cards still return the fields that *were* read instead of a bare 422."""
    for line in lines:
        if "บัตรประจำตัวประชาชน" in line.text or "thai national id" in line.text.lower():
            return True
    return False


def scan_thai_id_from_lines(lines: list[OcrLine]) -> tuple[ScanResponse | None, ScanError | None]:
    """Build ScanResponse from already-OCR'd lines. Pure function — used directly by tests."""
    id_num, id_conf, id_valid = extract_id_number(lines)
    first, first_conf = extract_first_name(lines)
    last, last_conf = extract_last_name(lines)
    dob, dob_conf = extract_dob(lines)
    sex, sex_conf = extract_sex(lines)

    # No ID number: only accept the card if it clearly *is* a Thai ID (header present) AND we
    # recovered at least one other field — otherwise it's genuinely not a readable document.
    if id_num is None and not (_has_id_card_markers(lines) and any([first, last, dob, sex])):
        return None, ScanError("no_document_detected")

    field_confs = [id_conf, first_conf, last_conf, dob_conf, sex_conf]
    populated = [c for c in field_confs if c > 0]
    overall = sum(populated) / len(populated) if populated else 0.0

    warnings: list[str] = []
    if id_num is None:
        warnings.append("id_number_unreadable")
    elif not id_valid:
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
