import re

import numpy as np

from app.confidence import (
    name_confidence_floor,
    tier_from_cross_engine_agreement,
    tier_from_field_checksum,
    tier_from_line2_score_proxy,
    tier_from_ocr_score,
    tier_from_unchecksummed_field,
)
from app.mrz import MrzParsed, _parse_name, parse_td3, td3_line2_checks
from app.preprocessing import decode_image, normalize_size, preprocess, rotations
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
    line1, line2 = candidates[-2], candidates[-1]
    if line1 == line2:
        # A genuine TD3 line 2 always differs from line 1 — it encodes document number, date of
        # birth, expiry, and checksums, never the name. Identical lines mean the source printed
        # (or a crop/OCR artifact duplicated) the same text twice, e.g. a mockup/specimen graphic
        # with no real second line — so document_number/date_of_birth/sex can't be trusted from
        # it even if it happens to satisfy a checksum or two by coincidence.
        return None
    return line1, line2


def _normalize_doc_type(line1: str) -> str:
    """Force TD3 slot 0 to 'P' when slot 1 is the '<' filler. This module only ever parses
    passports, and OCR commonly misreads the tiny passport marker 'P' as 'F' or similar at
    low resolution (e.g. screenshotted scans) — slot 1 being '<' is a strong enough signal
    that this is the doc-type field to safely correct it."""
    if len(line1) >= 2 and line1[1] == "<" and line1[0] != "P":
        return "P" + line1[1:]
    return line1


def _name_conf_from_supplementary_ocr(img: np.ndarray, actual_line1: str, fallback_proxy: float) -> float:
    """fastmrz's public API discards Tesseract's own per-word OCR confidence (it only calls
    `image_to_string`, never `image_to_data`), so line1 (name) fields read through it have no
    real confidence signal by default. A checksum-score proxy is NOT a safe substitute: ICAO
    9303 never checksums line 1 at all, so a perfectly clean line 2 says nothing about whether
    line 1's name was read correctly.

    Independently re-OCR the same source image with our own Tesseract call (the same technique
    the direct-OCR fallback pass already uses) and compare its line 1 against `actual_line1`
    (the text fastmrz actually returned, and that the response is built from). Confidence is
    only meaningful when it describes the exact text being returned — if the two independent
    reads disagree, that disagreement is itself strong evidence the name is unreliable and must
    dominate, *regardless* of how confident either engine was about its own (different) answer.
    (Confirmed in practice: fastmrz misread one character as a different, similarly-shaped
    letter while our independent read got it right — the wrong text alone still self-reported
    ~90% confidence, which is exactly the false signal this comparison exists to catch.)

    Falls back to the checksum-score proxy only when the independent read can't itself locate a
    plausible line 1 at all (e.g. a rotation where fastmrz's specialized ROI detector finds the
    MRZ band but a generic full-frame/bottom-strip OCR pass doesn't)."""
    text, conf_by_line = _ocr_mrz_text_and_conf(img)
    pair = _extract_mrz_lines(text)
    if pair is None:
        return fallback_proxy
    agree = pair[0] == actual_line1
    return tier_from_cross_engine_agreement(agree, conf_by_line.get(pair[0], 65.0) / 100.0)


def _try_parse(detector, img: np.ndarray) -> tuple[MrzParsed, tuple[bool, ...], float] | None:
    raw = detector.get_details(img, input_type="numpy", ignore_parse=True)
    pair = _extract_mrz_lines(raw or "")
    if pair is None:
        return None
    line2 = _realign_line2(pair[1])
    parsed = parse_td3(_normalize_doc_type(pair[0]), line2)
    if parsed is None or parsed.document_number is None:
        return None
    checks = td3_line2_checks(line2)
    proxy = tier_from_line2_score_proxy(True, sum(checks))
    name_conf = _name_conf_from_supplementary_ocr(img, pair[0], proxy)
    return parsed, checks, name_conf


def _split_given_names(given_names: str | None) -> tuple[str | None, str | None]:
    """MRZ given_names is one space-joined string (e.g. "ADAM MICHAEL"); the API surfaces the
    first token as first_name and everything after it as middle_name."""
    if not given_names:
        return None, None
    parts = given_names.split(maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def _to_response(
    parsed: MrzParsed,
    checks: tuple[bool, bool, bool, bool, bool] | None,
    name_conf: float,
) -> ScanResponse:
    """Build the API response with per-field confidence graded by the strongest signal actually
    available for that field, instead of one blended score copied onto every field:
      - document_number / date_of_birth carry their own TD3 check digit — graded against that
        plus the line's composite check digit (tier_from_field_checksum).
      - sex / country (nationality) have no check digit of their own in TD3 — graded on how many
        of the *other* four line-2 checks passed, as a proxy for overall read quality
        (tier_from_unchecksummed_field). Only meaningful when `checks` came from a real line 2;
        when the response was salvaged from line 1 alone (`checks is None`), nationality is read
        from line1 instead and shares line1's own OCR-confidence-derived `name_conf`.
      - first_name / middle_name / last_name always come from line 1, which ICAO never
        checksums, so they always use `name_conf` (real per-word OCR confidence when available,
        otherwise the line2-score proxy — see `_try_parse` / `_try_parse_direct` / `name_conf`),
        raised to `name_confidence_floor` when the document is *fully* checksum-valid — Tesseract's
        self-reported confidence for the specialized MRZ font model runs low even on correct
        reads, so a full 5/5 checksum pass (strong, hard-to-fake evidence the whole OCR pass over
        this image was accurate) is allowed to raise a floor under an under-confident real score.
    `overall` is the minimum of whichever fields are actually populated — a single bad field
    should pull the headline number down, not get diluted by an average.
    """
    sex = Sex(parsed.sex) if parsed.sex in ("M", "F") else None
    first_name, middle_name = _split_given_names(parsed.given_names)
    name_conf = max(name_conf, name_confidence_floor(parsed.valid))
    if parsed.valid:
        warnings: list[str] = []
    elif parsed.document_number is None:
        warnings = ["mrz_incomplete"]  # line 1 read (name/country) but line 2 missing/unreadable
    else:
        warnings = ["mrz_check_digits_failed"]

    if checks is not None:
        doc_ok, dob_ok, expiry_ok, personal_ok, composite_ok = checks
        document_number_conf = tier_from_field_checksum(
            parsed.document_number is not None, doc_ok, composite_ok
        )
        date_of_birth_conf = tier_from_field_checksum(
            parsed.date_of_birth_iso is not None, dob_ok, composite_ok
        )
        unchecksummed = tier_from_unchecksummed_field(
            True, sum([doc_ok, dob_ok, expiry_ok, personal_ok])
        )
        sex_conf = unchecksummed if parsed.sex else 0.0
        country_conf = unchecksummed if parsed.nationality else 0.0
    else:
        # Line-1-only salvage: no line 2 recovered, so document_number/date_of_birth/sex are
        # structurally absent, and nationality here was read off line 1 alongside the name — it
        # shares line 1's own OCR confidence rather than a line-2-derived proxy that doesn't apply.
        document_number_conf = 0.0
        date_of_birth_conf = 0.0
        sex_conf = 0.0
        country_conf = name_conf if parsed.nationality else 0.0

    first_name_conf = name_conf if first_name else 0.0
    middle_name_conf = name_conf if middle_name else 0.0
    last_name_conf = name_conf if parsed.surname else 0.0

    populated = [
        v for v in (
            first_name_conf, middle_name_conf, last_name_conf,
            document_number_conf, date_of_birth_conf, sex_conf, country_conf,
        ) if v > 0.0
    ]
    overall = min(populated) if populated else 0.0

    return ScanResponse(
        type=DocumentType.PASSPORT,
        first_name=first_name,
        middle_name=middle_name,
        last_name=parsed.surname,
        document_number=parsed.document_number,
        date_of_birth=parsed.date_of_birth_iso,
        sex=sex,
        country=parsed.nationality,
        document_valid=parsed.valid,
        confidence=ConfidenceScores(
            overall=overall,
            first_name=first_name_conf,
            middle_name=middle_name_conf,
            last_name=last_name_conf,
            document_number=document_number_conf,
            date_of_birth=date_of_birth_conf,
            sex=sex_conf,
            country=country_conf,
        ),
        warnings=warnings,
    )


def looks_like_passport(img: np.ndarray) -> bool:
    """Quick MRZ presence check on a preprocessed image. Used by thai_id scanner for type-mismatch fallback."""
    parsed = _try_parse(_get_detector(), img)
    return parsed is not None


_MRZ_OCR_CONFIG = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_MIN_OCR_DIM = 1600


def _upscale_if_small(img: np.ndarray) -> np.ndarray:
    """Enlarge tiny scans (e.g. screenshotted/re-compressed passport photos, ~400-500px) before
    OCR. Tesseract needs a minimum MRZ character height to read reliably; below it, character
    shapes blur together (e.g. 'P' misread as 'F', 'G' as '6') and OCR degrades sharply."""
    import cv2

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest >= _MIN_OCR_DIM:
        return img
    scale = _MIN_OCR_DIM / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _group_ocr_data_into_lines(data: dict) -> list[tuple[str, float]]:
    """Reconstruct text lines (and each line's average word confidence) from a pytesseract
    `image_to_data` DICT result, in reading order. MRZ's fixed-width font has no real
    inter-word gaps (aside from the '<' filler), so each physical MRZ line is almost always a
    single Tesseract "word" already; joining without spaces preserves that either way."""
    grouped: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
    order: list[tuple[int, int, int]] = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((word, float(data["conf"][i])))

    lines: list[tuple[str, float]] = []
    for key in order:
        words = grouped[key]
        text = "".join(w for w, _ in words)
        confs = [c for _, c in words if c >= 0]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        lines.append((text, avg_conf))
    return lines


def _ocr_mrz_text_and_conf(img: np.ndarray) -> tuple[str, dict[str, float]]:
    """Direct Tesseract MRZ read — fallback for when fastmrz's ROI detector can't locate the
    band (small scans, tight crops). Tries the full frame, then the bottom strip. Unlike a plain
    `image_to_string` call, this also returns each reconstructed line's average per-word OCR
    confidence (0-100), keyed by the line's own text, so callers can grade line1-derived fields
    (name, and line1-salvaged country) on real OCR confidence instead of a proxy."""
    import cv2
    import pytesseract

    img = _upscale_if_small(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _run(region: np.ndarray) -> tuple[str, dict[str, float]]:
        data = pytesseract.image_to_data(
            region, lang="mrz", config=_MRZ_OCR_CONFIG, output_type=pytesseract.Output.DICT
        )
        lines = _group_ocr_data_into_lines(data)
        text = "\n".join(t for t, _ in lines)
        conf_by_line = {t: c for t, c in lines}
        return text, conf_by_line

    text, conf_by_line = _run(gray)
    if _extract_mrz_lines(text) is None:
        h = gray.shape[0]
        text, conf_by_line = _run(gray[int(h * 0.6):, :])
    return text, conf_by_line


def _pad_mrz(line: str) -> str:
    """Normalise an OCR'd MRZ line to the fixed 44-char TD3 width. Tesseract often drops the
    trailing '<' filler, which parse_td3 (strict len==44) would otherwise reject outright."""
    return (line + "<" * 44)[:44]


_DIGIT_CONFUSABLES = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})
# (start, end) spans of TD3 line 2 that ICAO 9303 mandates as digits-only, so any OCR'd letter
# there is certainly a misread: date-of-birth field + its check digit, expiry field + its check
# digit, and the standalone document-number/personal-number/composite check digits (those check
# digits are always 0-9 even though the *fields* they validate, e.g. document_number, may
# legitimately contain letters — so the fields themselves are left untouched).
_NUMERIC_ZONES = ((9, 10), (13, 20), (21, 28), (42, 44))


def _fix_numeric_confusables(line2: str) -> str:
    """OCR frequently swaps visually similar letters for digits (O/0, I/1, S/5, B/8, Z/2) inside
    TD3 line 2's strictly-numeric zones. Correcting only those zones (not document_number or the
    personal-number field, which may legitimately contain letters) avoids erasing real data."""
    if len(line2) != 44:
        return line2
    chars = list(line2)
    for start, end in _NUMERIC_ZONES:
        chars[start:end] = "".join(chars[start:end]).translate(_DIGIT_CONFUSABLES)
    return "".join(chars)


def _realign_line2(raw_line2: str) -> str:
    """Recover from a single OCR-inserted/dropped character in MRZ line 2 (the direct-Tesseract
    fallback is noisy enough to occasionally gain or lose one character), which otherwise shifts
    every fixed-width field after the error — e.g. turning nationality "THA" into "2TH". Tries
    every single-character deletion/insertion and keeps whichever candidate makes all five TD3
    check digits internally consistent; falls back to the untouched padding if none do.

    Deliberately single-pass, not iterative: chaining more edit rounds on top of each other was
    tried and reverted — greedily accepting a partial-score improvement and then searching edits
    *of that already-lucky string* compounds false positives (each round both explores a wider
    net and starts from an already-biased base), and empirically let pure OCR garbage cross the
    same check-digit score that a real partial recovery reaches. A single pass keeps the
    candidate pool anchored to the actual OCR read, so a coincidental checksum match stays rare."""
    original = _fix_numeric_confusables(_pad_mrz(raw_line2))
    score = sum(td3_line2_checks(original))
    if score == 5:
        return original

    best, best_score = original, score
    limit = min(len(raw_line2), 44)
    candidates = [raw_line2[:i] + raw_line2[i + 1:] for i in range(limit)]
    candidates += [raw_line2[:i] + "<" + raw_line2[i:] for i in range(limit + 1)]
    for cand in candidates:
        padded = _fix_numeric_confusables(_pad_mrz(cand))
        cand_score = sum(td3_line2_checks(padded))
        if cand_score > best_score:
            best, best_score = padded, cand_score
            if best_score == 5:
                break
    return best


def _try_parse_direct(img: np.ndarray) -> tuple[MrzParsed, tuple[bool, ...], float] | None:
    text, conf_by_line = _ocr_mrz_text_and_conf(img)
    pair = _extract_mrz_lines(text)
    if pair is None:
        return None
    line2 = _realign_line2(pair[1])
    parsed = parse_td3(_normalize_doc_type(_pad_mrz(pair[0])), line2)
    if parsed is None or parsed.document_number is None:
        return None
    checks = td3_line2_checks(line2)
    name_conf = tier_from_ocr_score(conf_by_line.get(pair[0], 65.0) / 100.0)
    return parsed, checks, name_conf


def _parse_line1(line1: str) -> MrzParsed | None:
    """Salvage name + nationality from MRZ line 1 alone, for passports whose second line is
    missing/redacted. Gated on a well-formed 'P<CCC' + surname so OCR garbage can't slip through."""
    line1 = _normalize_doc_type(_pad_mrz(line1))
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


def _try_parse_line1(img: np.ndarray) -> tuple[MrzParsed, float] | None:
    text, conf_by_line = _ocr_mrz_text_and_conf(img)
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 25 and "<" in line:
            parsed = _parse_line1(line)
            if parsed is not None:
                name_conf = tier_from_ocr_score(conf_by_line.get(line, 65.0) / 100.0)
                return parsed, name_conf
    return None


_SEX_LABEL_RE = re.compile(r"^sex$", re.IGNORECASE)
_VIZ_OCR_TARGET_DIM = 1600


def _normalize_for_viz_ocr(img: np.ndarray) -> np.ndarray:
    """Resize the longest side to a fixed target before general-purpose (unrestricted,
    whole-page) OCR — small source images (downloaded/screenshotted passports, a few hundred
    px) need upscaling or printed text blurs together, same as `_upscale_if_small` does for
    MRZ; large ones (direct phone-camera photos, 3000px+) need downscaling or full automatic
    page segmentation over the entire data page takes multiple seconds per rotation for no
    accuracy benefit on a label as large as "Sex"."""
    import cv2

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest == _VIZ_OCR_TARGET_DIM:
        return img
    scale = _VIZ_OCR_TARGET_DIM / longest
    # INTER_AREA empirically reads this label reliably in both directions — CUBIC/LINEAR
    # upscaling introduces edge ringing that Tesseract's general model is more prone to
    # misread than the smoother AREA resampling, even though AREA is conventionally the
    # downscale-only choice.
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _read_sex_label(img: np.ndarray) -> tuple[Sex | None, float]:
    """OCR the printed data-page text (not the MRZ) for a 'Sex' label and the M/F value below
    it. Unlike MRZ OCR this uses the general-purpose English model with no character whitelist,
    since it's reading a page layout rather than a fixed-font code."""
    import pytesseract

    img = _normalize_for_viz_ocr(img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = [
        (data["text"][i].strip(), data["left"][i], data["top"][i], data["width"][i], data["height"][i], float(data["conf"][i]))
        for i in range(len(data["text"]))
        if data["text"][i].strip() and float(data["conf"][i]) > 0
    ]
    label = next((w for w in words if _SEX_LABEL_RE.match(w[0])), None)
    if label is not None:
        _text, lx, ly, lw, lh, _conf = label
        label_cx = lx + lw / 2
        label_bottom = ly + lh
        # The value sits below the label (this card's field layout is label-above-value),
        # roughly in the same column — generous tolerance since label/value widths differ a lot.
        candidates = [
            w for w in words
            if w[0].upper() in ("M", "F")
            and w[2] >= label_bottom - 2
            and abs((w[1] + w[3] / 2) - label_cx) < lw * 6
        ]
        if candidates:
            nearest = min(candidates, key=lambda w: w[2])
            return Sex(nearest[0].upper()), nearest[5] / 100.0
        return None, 0.0

    # No "Sex" label recognized at all — its small/faint/italic font often OCRs far worse than
    # the bold value next to it. A bare single-letter "M"/"F" token essentially never occurs
    # elsewhere on a passport data page (names, dates, and codes are always multi-character), so
    # if the page yields exactly one, it's unambiguous enough to trust without the label. (Only
    # reached when the label itself wasn't found — if it *was* found but its column search came
    # up empty, that's a deliberate rejection of an out-of-column letter, not a missing label.)
    bare = [w for w in words if w[0].upper() in ("M", "F")]
    if len(bare) == 1:
        only = bare[0]
        return Sex(only[0].upper()), only[5] / 100.0
    return None, 0.0


def _fill_sex_from_visual_zone(response: ScanResponse, raw: np.ndarray | None) -> ScanResponse:
    """Patch in a sex value read straight off the passport data page when the MRZ didn't carry
    one — e.g. line 2 is redacted/unreadable, or the response was salvaged from line 1 alone,
    both of which leave sex structurally unrecoverable from the MRZ."""
    if response.sex is not None or raw is None:
        return response
    for _deg, rotated in rotations(raw):
        sex, conf = _read_sex_label(rotated)
        if sex is not None:
            return response.model_copy(update={
                "sex": sex,
                "confidence": response.confidence.model_copy(update={"sex": tier_from_ocr_score(conf)}),
            })
    return response


_MIN_FALLBACK_SCORE = 2  # out of 5 checksum digits; 0-1 is statistically indistinguishable from
# a pure-garbage OCR read matching by chance (~10% odds per digit), so treating it as usable
# fabricates confident-looking but meaningless name/document_number/sex fields.


def scan_passport(image_bytes: bytes) -> tuple[ScanResponse | None, ScanError | None]:
    img = preprocess(image_bytes)
    if img is None:
        return None, ScanError("image_invalid")
    raw = decode_image(image_bytes)
    if raw is not None:
        # Same cap `img` already goes through — Pass 1 (fastmrz on `img`) succeeds most of the
        # time at this resolution, so the uncapped-resolution fallback passes below gain nothing
        # from a huge original photo except unbounded CPU/RAM on Tesseract.
        raw = normalize_size(raw)

    detector = _get_detector()
    fallback: MrzParsed | None = None
    fallback_checks: tuple[bool, ...] | None = None
    fallback_name_conf = 0.0
    fallback_score = -1

    # Pass 1: fastmrz on the preprocessed image (locates + OCRs the MRZ band; fast when it works).
    # Pass 2: direct Tesseract MRZ OCR on the *raw* image — recovers scans where fastmrz's ROI
    #   detector finds nothing. Raw (not preprocessed) because CLAHE/warp can smear a clean MRZ.
    # Keep the best-scoring fallback across every pass/rotation (not just the first one found) —
    # an earlier pass can land a low-confidence misaligned read (e.g. nationality "THA" -> "2TH")
    # before a later pass/rotation recovers the correctly-aligned line.
    for source, parser in ((img, lambda r: _try_parse(detector, r)), (raw, _try_parse_direct)):
        if source is None:
            continue
        for _deg, rotated in rotations(source):
            result = parser(rotated)
            if result is None:
                continue
            parsed, checks, name_conf = result
            if parsed.valid:
                return _fill_sex_from_visual_zone(_to_response(parsed, checks, name_conf), raw), None
            score = sum(checks)
            if score > fallback_score:
                fallback, fallback_checks, fallback_name_conf, fallback_score = parsed, checks, name_conf, score

    if fallback is not None and fallback_score >= _MIN_FALLBACK_SCORE:
        response = _to_response(fallback, fallback_checks, fallback_name_conf)
        return _fill_sex_from_visual_zone(response, raw), None

    # Pass 3: no full MRZ parse — salvage name + nationality from a clean line 1 (line 2
    # missing/redacted) so we still return what's legible instead of a bare 422.
    for _deg, rotated in rotations(raw):
        result = _try_parse_line1(rotated)
        if result is not None:
            parsed, name_conf = result
            return _fill_sex_from_visual_zone(_to_response(parsed, None, name_conf), raw), None

    return None, ScanError("no_document_detected")


def scan_passport_from_text(raw_mrz_text: str) -> tuple[ScanResponse | None, ScanError | None]:
    """Bypass OCR; parse MRZ text directly. Used for testing."""
    pair = _extract_mrz_lines(raw_mrz_text)
    if pair is None:
        return None, ScanError("no_document_detected")
    parsed = parse_td3(pair[0], pair[1])
    if parsed is None or parsed.document_number is None:
        return None, ScanError("no_document_detected")
    checks = td3_line2_checks(pair[1])
    return _to_response(parsed, checks, 1.0), None
