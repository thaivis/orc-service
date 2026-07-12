import os
import re
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.confidence import tier_from_id_checksum, tier_from_ocr_score
from app.preprocessing import preprocess, rotations
from app.scan_error import ScanError
from app.schemas import ConfidenceScores, DocumentType, ScanResponse, Sex
from app.validators import thai_id_checksum

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        # Enabled: oneDNN-accelerated conv kernels cut per-scan OCR time ~20-27% (measured: 10
        # consecutive predict() calls on the mobile det/rec models, 32.3s -> 23.6s avg, tighter
        # variance too) with byte-identical detection/recognition output. paddlepaddle is pinned
        # to 3.2.x specifically because 3.3.x has an unrelated oneDNN/PIR regression (see
        # requirements.txt) — on this pinned version, with these mobile models, MKL-DNN ran
        # crash-free across repeated trials.
        os.environ.setdefault("FLAGS_use_mkldnn", "1")
        from paddleocr import PaddleOCR

        # PP-OCRv5 (paddleocr 3.x) is the first version with Thai in the default rec dict.
        # `lang="th"` alone picks the heavy PP-OCRv5_server_det backbone (85MB, ResNet-vd) for
        # detection. The mobile variant (4.8MB, PP-LCNet) is ~17x smaller and measurably lighter
        # on both CPU and memory per scan; validated against tools/accuracy.py before shipping.
        # Recognition must be pinned explicitly too — passing text_detection_model_name alone
        # makes paddleocr silently drop `lang` and fall back recognition to the heavy
        # PP-OCRv5_server_rec (82MB) instead of the Thai mobile model.
        _ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="th_PP-OCRv5_mobile_rec",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ocr


_det_cache: tuple | None = None

_DIGIT_TRANSLATION = str.maketrans({
    "๐": "0",
    "๑": "1",
    "๒": "2",
    "๓": "3",
    "๔": "4",
    "๕": "5",
    "๖": "6",
    "๗": "7",
    "๘": "8",
    "๙": "9",
})


def _ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_TRANSLATION)


def _detect_boxes(img: np.ndarray):
    """Detected text boxes for `img`, reusing the previous result when it's the very same array.

    The orientation probe and the first OCR pass both detect on the un-rotated card, and detection
    scales with image area (0.4s on a 274px demo card, 3.8s on an uncropped 1500x2000 phone photo)
    — paying it twice is pure waste. Keyed on object identity, not contents: a stale entry from a
    concurrent scan can only miss, never return another image's boxes."""
    global _det_cache
    if _det_cache is not None and _det_cache[0] is img:
        return _det_cache[1]
    inner, det_params = _get_stages()
    polys = list(inner.text_det_model([img], **det_params))[0]["dt_polys"]
    _det_cache = (img, polys)
    return polys


def _text_reads_horizontally(img: np.ndarray) -> bool | None:
    """True when the detected text lines are wider than tall, i.e. the card is at 0 or 180 degrees.

    Detection is cheap next to recognition (~1-4s against ~17s for a full pass), and it separates
    the {0,180} family from {90,270} cleanly: measured across every fixture, upright cards score
    0.92-1.00 of boxes wider than tall, sideways ones 0.00-0.17. It cannot tell 0 from 180 (both
    are horizontal) — that stays the rotation loop's job. Textline-orientation angles look like
    they should answer that, but measured on the fixtures they don't: physically flipping
    IMG_2729 moves its 180-degree share from 0.19 to 0.13, and nattaya's from 0.00 to 0.00.

    Deliberately not decided from the image's own aspect ratio: when `detect_document_boundary`
    fails to find the card, the frame keeps the phone's portrait shape while the text inside is
    perfectly upright (IMG_2730 in the fixtures), and an aspect-ratio test gets that backwards.

    Returns None when detection can't answer, so callers keep the original all-rotations order."""
    inner, _det_params = _get_stages()
    if inner is None:
        return None
    try:
        polys = _detect_boxes(img)
    except Exception:
        return None
    if len(polys) == 0:
        return None
    wide = sum(
        1 for p in polys
        if (max(q[0] for q in p) - min(q[0] for q in p))
        > (max(q[1] for q in p) - min(q[1] for q in p))
    )
    return wide * 2 >= len(polys)


def _rotation_plan(img: np.ndarray) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
    """Split the 4 cardinal rotations into (try first, fall back to) by detected text direction.

    A wrong rotation can produce a checksum-valid but wholly wrong ID number (measured: IMG_2729
    at 270 degrees reads 1242565242022 and satisfies `_is_complete`), so the fallback half is
    never discarded outright — it is skipped only once the preferred half has actually read
    something off the card, which is evidence the detector called the direction correctly.

    When detection can't answer, everything lands in the first list and behaviour is unchanged."""
    rots = rotations(img)
    horizontal = _text_reads_horizontally(img)
    if horizontal is None:
        return rots, []
    preferred = (0, 180) if horizontal else (90, 270)
    return (
        [r for r in rots if r[0] in preferred],
        [r for r in rots if r[0] not in preferred],
    )


@dataclass
class OcrLine:
    text: str
    confidence: float
    cx: float = 0.0
    cy: float = 0.0


_stages: tuple | None = None
_REC_BATCH = 6


def _get_stages():
    """The pipeline's own detection / textline-orientation / recognition models, plus the exact
    detection parameters `predict()` would pass them.

    Recognition dominates a scan (~85% of the time, and it scales with the number of detected
    boxes, not the image's size), but `predict()` only recognizes all-or-nothing. Driving the
    sub-models directly lets recognition stop once every field has been read — a Thai ID detects
    16-36 text lines and the six fields we return all sit in the first dozen, so the rest are the
    address, religion, issue/expiry dates and signature, recognized and then thrown away.

    Reusing the pipeline's own model objects (rather than constructing new ones) is what keeps
    detection faithful: a standalone TextDetection silently uses the model's default thresholds
    while the pipeline configures limit_side_len=64/limit_type=min/box_thresh=0.6/unclip_ratio=1.5,
    and different boxes mean different line merges and different text.

    Returns (None, None) if paddlex's internals ever stop matching the pinned 3.5.x layout, so
    callers transparently fall back to the whole-image `predict()` path."""
    global _stages
    if _stages is None:
        try:
            inner = _get_ocr().paddlex_pipeline._pipeline
            params = inner.get_text_det_params(
                inner.text_det_limit_side_len,
                inner.text_det_limit_type,
                inner.text_det_max_side_limit,
                inner.text_det_thresh,
                inner.text_det_box_thresh,
                inner.text_det_unclip_ratio,
            )
            # Touch every attribute the fast path needs, so a layout change fails here (and falls
            # back) rather than halfway through a scan.
            _ = (inner.text_det_model, inner.text_rec_model,
                 inner.textline_orientation_model, inner._sort_boxes,
                 inner._crop_by_polys, inner.rotate_image)
            _stages = (inner, params)
        except Exception:
            _stages = (None, None)
    return _stages


def _line_from(text: str, score: float, poly) -> OcrLine:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return OcrLine(
        text=str(text).strip(),
        confidence=float(score),
        cx=sum(xs) / len(xs),
        cy=sum(ys) / len(ys),
    )


def _run_ocr_full(img: np.ndarray) -> list[OcrLine]:
    """Whole-image OCR. Fallback for when the sub-model fast path is unavailable."""
    raw = _get_ocr().predict(img)
    if not raw:
        return []
    out: list[OcrLine] = []
    for res in raw:
        for text, conf, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
            out.append(_line_from(text, conf, poly))
    return out


def _run_ocr(img: np.ndarray) -> list[OcrLine]:
    """Detect every text box, then recognize them top-to-bottom only until the fields are all read.

    `_sort_boxes` orders boxes down the card, and a Thai ID puts the ID number, name, surname,
    title (which gives sex) and date of birth above everything else, so completeness is reached
    long before the last box. Cards that never complete — an ID number too blurred to pass its
    checksum, a card with no legible surname — recognize every box, exactly as before.

    Recognizing a subset necessarily changes how boxes are batched, and recognition pads each
    batch to its widest crop, so a few characters can differ from what whole-image `predict()`
    returns (measured: '24 May 2022' vs '24.May2022' on an issue-date line we don't read). Field
    accuracy is validated against tools/accuracy.py rather than assumed from text equality."""
    inner, _det_params = _get_stages()
    if inner is None:
        return _run_ocr_full(img)

    polys = inner._sort_boxes(_detect_boxes(img))
    if len(polys) == 0:
        return []

    crops, boxes = [], []
    for crop, poly in zip(inner._crop_by_polys(img, polys), polys):
        if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
            crops.append(crop)
            boxes.append(poly)

    lines: list[OcrLine] = []
    for start in range(0, len(crops), _REC_BATCH):
        chunk = crops[start:start + _REC_BATCH]
        angles = [
            int(np.asarray(info["class_ids"], dtype=np.int64).ravel()[0])
            for info in inner.textline_orientation_model(chunk)
        ]
        chunk = inner.rotate_image(chunk, angles)
        for res, poly in zip(
            inner.text_rec_model(chunk, batch_size=len(chunk)),
            boxes[start:start + _REC_BATCH],
        ):
            lines.append(_line_from(res["rec_text"], res["rec_score"], poly))

        response, _error = scan_thai_id_from_lines(lines)
        if response is not None and _is_complete(response):
            break
    return lines


def _run_tesseract_ocr(img: np.ndarray) -> list[OcrLine]:
    """English-oriented fallback for low-res Thai IDs whose Latin fields are clearer than Thai.

    Paddle's Thai mobile recognizer can collapse small, compressed mixed Thai/English ID cards
    into high-confidence symbol garbage. Tesseract's English model reads the Latin labels, names,
    DOB, and Arabic-digit ID groups on those cards well enough to recover the API fields. Kept as
    a fallback so normal Paddle reads stay on the faster tuned path.
    """
    import cv2
    import pytesseract

    scaled = cv2.resize(
        img,
        (img.shape[1] * 2, img.shape[0] * 2),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(
        gray,
        lang="eng",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    grouped: dict[tuple[int, int, int], list[int]] = {}
    order: list[tuple[int, int, int]] = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except ValueError:
            continue
        if conf <= 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(i)

    lines: list[OcrLine] = []
    for key in order:
        idxs = grouped[key]
        text = " ".join(data["text"][i].strip() for i in idxs)
        confs = [float(data["conf"][i]) / 100.0 for i in idxs if float(data["conf"][i]) > 0]
        left = min(float(data["left"][i]) for i in idxs) / 2.0
        top = min(float(data["top"][i]) for i in idxs) / 2.0
        right = max(float(data["left"][i] + data["width"][i]) for i in idxs) / 2.0
        bottom = max(float(data["top"][i] + data["height"][i]) for i in idxs) / 2.0
        lines.append(OcrLine(
            text=text,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            cx=(left + right) / 2.0,
            cy=(top + bottom) / 2.0,
        ))
    return lines


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
        digits = re.sub(r"[^0-9]", "", _ascii_digits(line.text))
        if len(digits) == 13:
            if thai_id_checksum(digits):
                return digits, line.confidence, True
            if fallback is None:
                fallback = (digits, line.confidence)

    # Fragmented path: concatenate digit runs on the same OCR row only. The old implementation
    # walked every digit run in page order, which let dates/laser numbers from unrelated rows
    # accidentally combine into a checksum-valid 13-digit value.
    indexed = list(enumerate(lines))
    tol = _row_tol(lines)
    indexed.sort(key=lambda item: (item[1].cy, item[1].cx, item[0]))
    rows: list[list[tuple[int, OcrLine]]] = []
    for item in indexed:
        _idx, line = item
        if not rows or abs(line.cy - rows[-1][0][1].cy) > tol:
            rows.append([item])
        else:
            rows[-1].append(item)

    for row in rows:
        runs: list[tuple[str, float]] = []
        for _idx, line in sorted(row, key=lambda item: (item[1].cx, item[0])):
            runs.extend(
                (m.group(), line.confidence)
                for m in re.finditer(r"[0-9]+", _ascii_digits(line.text))
            )
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


NAME_STOPWORDS = {
    "MR", "MRS", "MISS", "MASTER", "MISTER", "LAST", "NAME", "LASTNAME",
    # Other field labels a loose positional fallback could otherwise mistake for a name.
    "DATE", "BIRTH", "OF", "ISSUE", "EXPIRY", "IDENTIFICATION", "NUMBER",
    "RELIGION", "ADDRESS", "SEX",
}


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


def _column_candidates(anchor: OcrLine, lines: list[OcrLine], max_dx: float = 100.0) -> list[OcrLine]:
    """Other lines roughly in the same vertical column as `anchor`, nearest first.

    Fallback for when a label OCR'd cleanly but its value isn't to the right or directly
    below — seen on skewed/rotated captures where "below" on the physical card doesn't map to
    "below" in pixel space. Excludes lines that mention "name" so a Name/Last name label row
    can't be mistaken for another label's value."""
    others = [
        ln for ln in lines
        if ln is not anchor and abs(ln.cx - anchor.cx) < max_dx and "name" not in ln.text.lower()
    ]
    return sorted(others, key=lambda ln: abs(ln.cy - anchor.cy))


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
            # Label found but its value isn't to the right or below — widen to the column.
            for cand in _column_candidates(line, lines):
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

DATE_EN_RE = re.compile(r"\b(\d{1,2})\s*([A-Za-z]{3,})\.?\s+(\d{4})\b")


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

    # document_number's checksum is a stronger correctness signal than PaddleOCR's own score —
    # it dominates the tier (always MAX when valid) instead of blending with it. The other
    # fields have no independent verification, so their tier IS the OCR engine's own score,
    # just bucketed onto the same 5-level scale for comparability.
    id_tier = tier_from_id_checksum(id_num is not None, id_valid, id_conf)
    first_tier = tier_from_ocr_score(first_conf)
    last_tier = tier_from_ocr_score(last_conf)
    dob_tier = tier_from_ocr_score(dob_conf)
    sex_tier = tier_from_ocr_score(sex_conf)

    field_confs = [id_tier, first_tier, last_tier, dob_tier, sex_tier]
    populated = [c for c in field_confs if c > 0]
    overall = min(populated) if populated else 0.0

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
                first_name=first_tier,
                last_name=last_tier,
                document_number=id_tier,
                date_of_birth=dob_tier,
                sex=sex_tier,
                country=1.0,
            ),
            warnings=warnings,
        ),
        None,
    )


def _is_complete(response: ScanResponse) -> bool:
    """True once nothing further could be gained from trying more rotations.

    date_of_birth is deliberately excluded from this check: a missing DOB is almost always a
    genuine OCR miss (small font, calendar-format ambiguity) rather than an orientation problem,
    so waiting on it before stopping just buys 3 extra full OCR passes that re-confirm the same
    miss. Measured on a real fixture image: DOB stayed unreadable across all 4 rotations, so the
    extra passes cost ~40s and recovered nothing.
    """
    return response.document_valid and all(
        v is not None
        for v in (response.first_name, response.last_name, response.sex)
    )


def _supporting_field_count(response: ScanResponse) -> int:
    """How much non-ID evidence this rotation read from the card.

    A random 13-digit run can pass the Thai checksum by chance, especially when OCR is pointed at
    the wrong rotation and sees dates/laser numbers as noise. Names, DOB, and title-derived sex
    are independent evidence that the rotation actually read the card face.
    """
    return sum(
        v is not None
        for v in (response.first_name, response.last_name, response.date_of_birth, response.sex)
    )


def _merge_rotation_reads(responses: list[ScanResponse]) -> ScanResponse:
    """Combine reads from multiple rotations of the same physical card, keeping the
    highest-confidence value for each field independently.

    A card photographed at an angle often OCRs one field cleanly in one rotation (e.g. the ID
    number) and a different field cleanly in another (e.g. the surname, whose label is small
    and easily misread) — picking a single "best" rotation would discard whichever fields lost
    out in that rotation even though another rotation read them fine."""
    valid = [r for r in responses if r.document_valid]
    id_pool = valid or responses
    # Prefer the id number multiple independent rotations agree on over the single most
    # "confident" read — a checksum only guards against random digit errors, so one rotation can
    # still land on a coincidentally checksum-valid misread (e.g. unrelated digits printed
    # elsewhere on the card) with high per-line OCR confidence. Agreement across rotations that
    # each re-detected and re-read the number from scratch is the stronger correctness signal.
    id_agreement: dict[str | None, int] = {}
    for r in id_pool:
        id_agreement[r.document_number] = id_agreement.get(r.document_number, 0) + 1
    id_source = max(
        id_pool,
        key=lambda r: (
            id_agreement[r.document_number],
            _supporting_field_count(r),
            r.confidence.document_number,
        ),
    )

    def pick(field: str, conf_field: str) -> tuple[object | None, float]:
        candidates = [r for r in responses if getattr(r, field) is not None]
        if not candidates:
            return None, 0.0
        best = max(candidates, key=lambda r: getattr(r.confidence, conf_field))
        return getattr(best, field), getattr(best.confidence, conf_field)

    first, first_conf = pick("first_name", "first_name")
    last, last_conf = pick("last_name", "last_name")
    dob, dob_conf = pick("date_of_birth", "date_of_birth")
    sex, sex_conf = pick("sex", "sex")

    field_confs = [id_source.confidence.document_number, first_conf, last_conf, dob_conf, sex_conf]
    populated = [c for c in field_confs if c > 0]
    overall = min(populated) if populated else 0.0

    warnings: list[str] = []
    if id_source.document_number is None:
        warnings.append("id_number_unreadable")
    elif not id_source.document_valid:
        warnings.append("thai_id_checksum_failed")

    return ScanResponse(
        type=DocumentType.THAI_ID,
        first_name=first,
        last_name=last,
        document_number=id_source.document_number,
        date_of_birth=dob,
        sex=sex,
        country="THA",
        document_valid=id_source.document_valid,
        confidence=ConfidenceScores(
            overall=overall,
            first_name=first_conf,
            last_name=last_conf,
            document_number=id_source.confidence.document_number,
            date_of_birth=dob_conf,
            sex=sex_conf,
            country=1.0,
        ),
        warnings=warnings,
    )


def scan_thai_id(image_bytes: bytes) -> tuple[ScanResponse | None, ScanError | None]:
    img = preprocess(image_bytes)
    if img is None:
        return None, ScanError("image_invalid")

    responses: list[ScanResponse] = []

    # Cheap first pass: many Thai IDs expose the API fields in Latin text. If Tesseract reads a
    # complete, checksum-valid card, skip Paddle entirely (and avoid model cold-start cost).
    for _deg, rotated in rotations(img):
        lines = _run_tesseract_ocr(rotated)
        response, _error = scan_thai_id_from_lines(lines)
        if response is None:
            continue
        responses.append(response)
        if _is_complete(response):
            return response, None

    # Try the cardinal orientations — a card photographed upside-down or sideways otherwise OCRs
    # as noise. A cheap detection pass (~1-4s, against ~17s for a full OCR pass) splits them into
    # the direction the text actually runs and the perpendicular one. Stops as soon as a rotation
    # reads every field; otherwise every rotation tried gets merged field-by-field so a field
    # missed in one rotation can still be recovered from another.
    preferred, fallback = _rotation_plan(img)
    for _deg, rotated in preferred:
        lines = _run_ocr(rotated)
        response, _error = scan_thai_id_from_lines(lines)
        if response is None:
            continue
        responses.append(response)
        if _is_complete(response):
            break

    # The perpendicular rotations are worth their OCR passes when the preferred direction read
    # nothing, or when it only produced a bare checksum-valid number with no independent fields.
    # That bare-number case is exactly how a wrong rotation can turn dates/laser digits into a
    # confident-looking but wrong document_number.
    if not responses or not any(_supporting_field_count(r) > 0 for r in responses):
        for _deg, rotated in fallback:
            lines = _run_ocr(rotated)
            response, _error = scan_thai_id_from_lines(lines)
            if response is None:
                continue
            responses.append(response)
            if _is_complete(response):
                break

    if not responses:
        return None, ScanError("no_document_detected")
    return _merge_rotation_reads(responses), None
