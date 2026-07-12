from datetime import date

from app.scanners import thai_id as thai_id_module
from app.scanners.thai_id import (
    OcrLine,
    extract_dob,
    extract_first_name,
    extract_id_number,
    extract_last_name,
    extract_sex,
    scan_thai_id,
    scan_thai_id_from_lines,
)
from app.schemas import DocumentType, Sex
from app.validators import thai_id_checksum


def _valid_id() -> str:
    base = "110170015764"
    total = sum(int(base[i]) * (13 - i) for i in range(12))
    check = (11 - (total % 11)) % 10
    return base + str(check)


def _invalid_id() -> str:
    valid = _valid_id()
    wrong = (int(valid[-1]) + 1) % 10
    return valid[:-1] + str(wrong)


# --- ID number ---

def test_extract_id_number_strips_separators_and_validates():
    valid = _valid_id()
    spaced = f"{valid[0]} {valid[1:5]} {valid[5:10]} {valid[10:12]} {valid[12]}"
    id_str, conf, ok = extract_id_number([OcrLine(text=spaced, confidence=0.95)])
    assert id_str == valid
    assert ok is True
    assert conf == 0.95


def test_extract_id_number_normalizes_thai_digits_to_ascii():
    valid = _valid_id()
    thai_digits = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
    text = valid.translate(thai_digits)

    id_str, _conf, ok = extract_id_number([OcrLine(text=text, confidence=0.95)])

    assert id_str == valid
    assert ok is True


def test_extract_id_number_prefers_checksum_valid_over_invalid():
    valid = _valid_id()
    invalid = _invalid_id()
    lines = [
        OcrLine(text=invalid, confidence=0.99),  # higher confidence but bad checksum
        OcrLine(text=valid, confidence=0.7),
    ]
    id_str, _, ok = extract_id_number(lines)
    assert id_str == valid
    assert ok is True


def test_extract_id_number_falls_back_to_invalid_checksum():
    invalid = _invalid_id()
    id_str, _, ok = extract_id_number([OcrLine(text=invalid, confidence=0.9)])
    assert id_str == invalid
    assert ok is False


def test_extract_id_number_returns_none_when_no_13_digit_run():
    id_str, conf, ok = extract_id_number([OcrLine(text="hello world", confidence=0.9)])
    assert id_str is None
    assert conf == 0.0
    assert ok is False


def test_extract_id_number_does_not_join_digits_across_rows():
    """Dates, laser numbers, and addresses can each contribute digit groups. A checksum-valid
    value assembled across different OCR rows is not the ID-number field."""
    valid = _valid_id()
    lines = [
        OcrLine(text=valid[:6], confidence=0.95, cx=100, cy=10),
        OcrLine(text=valid[6:], confidence=0.95, cx=100, cy=80),
    ]

    id_str, conf, ok = extract_id_number(lines)

    assert id_str is None
    assert conf == 0.0
    assert ok is False


# --- Sex ---

def test_extract_sex_for_each_thai_title():
    assert extract_sex([OcrLine(text="นาย JOHN", confidence=0.9)])[0] == Sex.M
    assert extract_sex([OcrLine(text="นาง JANE", confidence=0.9)])[0] == Sex.F
    assert extract_sex([OcrLine(text="นางสาว JANE", confidence=0.9)])[0] == Sex.F
    assert extract_sex([OcrLine(text="เด็กชาย Bobby", confidence=0.9)])[0] == Sex.M
    assert extract_sex([OcrLine(text="เด็กหญิง Susie", confidence=0.9)])[0] == Sex.F


def test_extract_sex_returns_none_when_no_prefix():
    sex, conf = extract_sex([OcrLine(text="JOHN DOE", confidence=0.9)])
    assert sex is None
    assert conf == 0.0


def test_extract_sex_does_not_misclassify_นางสาว_as_นาง():
    # "นางสาว" contains "นาง" but should resolve to F via นางสาว — both are F so the value is fine,
    # but we want to make sure the longer prefix is matched first (regression guard).
    sex, _ = extract_sex([OcrLine(text="นางสาว SOMSRI", confidence=0.9)])
    assert sex == Sex.F


# --- Names ---

def test_extract_first_name_from_english_title():
    name, conf = extract_first_name([OcrLine(text="Mr. JOHN", confidence=0.88)])
    assert name == "JOHN"
    assert conf == 0.88


def test_extract_first_name_anchor_below():
    lines = [
        OcrLine(text="Name", confidence=0.95, cx=100, cy=50),
        OcrLine(text="Mr. SOMCHAI", confidence=0.9, cx=110, cy=80),
        OcrLine(text="Last name", confidence=0.95, cx=100, cy=120),
        OcrLine(text="JAIDEE", confidence=0.92, cx=110, cy=150),
    ]
    name, _ = extract_first_name(lines)
    assert name == "SOMCHAI"


def test_extract_first_name_anchor_inline():
    name, _ = extract_first_name([OcrLine(text="Name SOMCHAI", confidence=0.9)])
    assert name == "SOMCHAI"


def test_extract_last_name_anchor_below():
    lines = [
        OcrLine(text="Last name", confidence=0.95, cx=100, cy=120),
        OcrLine(text="JAIDEE", confidence=0.92, cx=110, cy=150),
    ]
    name, _ = extract_last_name(lines)
    assert name == "JAIDEE"


def test_extract_last_name_inline():
    name, _ = extract_last_name([OcrLine(text="Last name : JAIDEE", confidence=0.9)])
    assert name == "JAIDEE"


def test_extract_last_name_falls_back_to_same_column_above_label():
    """On some rotated/skewed captures the value box lands above its label instead of below
    or to the right — the strict same-row/below search must fall back to scanning the column."""
    lines = [
        OcrLine(text="Date of Birth 29 Aug. 2003", confidence=0.96, cx=442.8, cy=623.2),
        OcrLine(text="JAIDEE", confidence=1.0, cx=322.8, cy=625.2),
        OcrLine(text="NameMr. SOMCHAI", confidence=0.96, cx=262.5, cy=707.0),
        OcrLine(text="Last name", confidence=0.83, cx=327.5, cy=827.5),
    ]
    name, conf = extract_last_name(lines)
    assert name == "JAIDEE"
    assert conf == 1.0


def test_extract_last_name_column_fallback_ignores_other_labels():
    """The column fallback must not mistake a nearby field label (e.g. a Date of Birth line
    outside the column tolerance, or a Name line inside it) for the surname value."""
    lines = [
        OcrLine(text="Date of Birth 29 Aug. 2003", confidence=0.96, cx=900.0, cy=625.2),
        OcrLine(text="NameMr. SOMCHAI", confidence=0.96, cx=262.5, cy=707.0),
        OcrLine(text="Last name", confidence=0.83, cx=327.5, cy=827.5),
    ]
    name, _ = extract_last_name(lines)
    assert name is None


def test_extract_first_name_returns_none_when_absent():
    name, conf = extract_first_name([OcrLine(text="Identification", confidence=0.9)])
    assert name is None
    assert conf == 0.0


# --- DOB ---

def test_extract_dob_english_format():
    d, conf = extract_dob([OcrLine(text="Date of Birth 1 Jan. 1990", confidence=0.9)])
    assert d == date(1990, 1, 1)
    assert conf == 0.9


def test_extract_dob_english_format_without_space_between_day_and_month():
    d, conf = extract_dob([OcrLine(text="Date of Birth 2May 1990", confidence=0.9)])
    assert d == date(1990, 5, 2)
    assert conf == 0.9


def test_extract_dob_thai_be_format_converts_to_ce():
    d, _ = extract_dob([OcrLine(text="1 ม.ค. 2533", confidence=0.9)])
    assert d == date(1990, 1, 1)


def test_extract_dob_thai_full_month():
    d, _ = extract_dob([OcrLine(text="15 มีนาคม 2533", confidence=0.9)])
    assert d == date(1990, 3, 15)


def test_extract_dob_rejects_future_date():
    d, conf = extract_dob([OcrLine(text="1 Jan. 2999", confidence=0.9)])
    assert d is None
    assert conf == 0.0


def test_extract_dob_returns_none_when_no_date():
    d, conf = extract_dob([OcrLine(text="JOHN DOE", confidence=0.9)])
    assert d is None
    assert conf == 0.0


# --- Orchestration ---

def test_scan_thai_id_from_lines_happy_path():
    valid = _valid_id()
    lines = [
        OcrLine(text=valid, confidence=0.95),
        OcrLine(text="นาย SOMCHAI", confidence=0.9, cx=100, cy=50),
        OcrLine(text="Name SOMCHAI", confidence=0.9, cx=100, cy=80),
        OcrLine(text="Last name JAIDEE", confidence=0.92, cx=100, cy=120),
        OcrLine(text="Date of Birth 1 Jan. 1990", confidence=0.88),
    ]
    result, err = scan_thai_id_from_lines(lines)
    assert err is None
    assert result is not None
    assert result.type == DocumentType.THAI_ID
    assert result.document_number == valid
    assert result.first_name == "SOMCHAI"
    assert result.last_name == "JAIDEE"
    assert result.date_of_birth == date(1990, 1, 1)
    assert result.sex == Sex.M
    assert result.country == "THA"
    assert result.document_valid is True
    assert result.confidence.country == 1.0
    assert result.confidence.overall > 0.8
    assert result.warnings == []


def test_scan_thai_id_from_lines_no_id_returns_no_document():
    result, err = scan_thai_id_from_lines([OcrLine(text="hello", confidence=0.9)])
    assert result is None
    assert err is not None
    assert err.code == "no_document_detected"
    assert err.detected_type is None


def test_scan_thai_id_from_lines_invalid_checksum_warns():
    invalid = _invalid_id()
    lines = [OcrLine(text=invalid, confidence=0.95)]
    result, err = scan_thai_id_from_lines(lines)
    assert err is None
    assert result is not None
    assert result.document_valid is False
    assert "thai_id_checksum_failed" in result.warnings


def test_scan_thai_id_from_lines_partial_extraction_returns_nulls():
    valid = _valid_id()
    lines = [OcrLine(text=valid, confidence=0.95)]
    result, err = scan_thai_id_from_lines(lines)
    assert err is None
    assert result is not None
    assert result.first_name is None
    assert result.last_name is None
    assert result.sex is None
    assert result.date_of_birth is None
    assert result.confidence.first_name == 0.0
    assert result.confidence.last_name == 0.0
    assert result.confidence.sex == 0.0
    assert result.confidence.date_of_birth == 0.0


def test_synthetic_id_helper_passes_checksum():
    """Sanity check on the test helper itself."""
    assert thai_id_checksum(_valid_id())
    assert not thai_id_checksum(_invalid_id())


# --- scan_thai_id rotation retry ---

def test_scan_thai_id_recovers_via_rotation(monkeypatch):
    """A card photographed upside-down only OCRs cleanly at one of the 4 cardinal rotations —
    scan_thai_id must retry them instead of giving up after the unrotated pass."""
    valid = _valid_id()
    good_lines = [OcrLine(text=valid, confidence=0.95)]

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(
        thai_id_module, "rotations",
        lambda img: [(0, "r0"), (90, "r90"), (180, "r180"), (270, "r270")],
    )
    monkeypatch.setattr(
        thai_id_module, "_run_ocr",
        lambda rotated: good_lines if rotated == "r180" else [],
    )
    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", lambda rotated, **kwargs: [])

    result, err = scan_thai_id(b"fake-bytes")
    assert err is None
    assert result is not None
    assert result.document_number == valid
    assert result.document_valid is True


def test_scan_thai_id_short_circuits_on_first_complete_rotation(monkeypatch):
    """The common case (card already upright, every field read) must not pay for extra OCR passes."""
    valid = _valid_id()
    good_lines = [
        OcrLine(text=valid, confidence=0.95),
        OcrLine(text="นาย SOMCHAI", confidence=0.9, cx=100, cy=50),
        OcrLine(text="Name SOMCHAI", confidence=0.9, cx=100, cy=80),
        OcrLine(text="Last name JAIDEE", confidence=0.92, cx=100, cy=120),
        OcrLine(text="Date of Birth 1 Jan. 1990", confidence=0.88),
    ]
    calls: list[str] = []

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(
        thai_id_module, "rotations",
        lambda img: [(0, "r0"), (90, "r90"), (180, "r180"), (270, "r270")],
    )

    def fake_run_ocr(rotated):
        calls.append(rotated)
        return good_lines

    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", lambda rotated, **kwargs: [])
    monkeypatch.setattr(thai_id_module, "_run_ocr", fake_run_ocr)

    result, err = scan_thai_id(b"fake-bytes")
    assert err is None
    assert result is not None
    assert calls == ["r0"]


def test_scan_thai_id_returns_no_document_when_all_rotations_fail(monkeypatch):
    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(thai_id_module, "rotations", lambda img: [(0, "r0"), (180, "r180")])
    monkeypatch.setattr(thai_id_module, "_run_ocr", lambda rotated: [])
    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", lambda rotated, **kwargs: [])

    result, err = scan_thai_id(b"fake-bytes")
    assert result is None
    assert err is not None
    assert err.code == "no_document_detected"


def test_scan_thai_id_merges_fields_across_rotations(monkeypatch):
    """A checksum-valid ID number can appear in more than one rotation while different fields
    read cleanly in each — e.g. the surname's label OCRs correctly only when the card lands at
    a different angle. The final result must not lose a field a later rotation did recover."""
    valid = _valid_id()
    # r0: valid checksum, first name reads fine, last name label never detected.
    r0_lines = [
        OcrLine(text=valid, confidence=0.95),
        OcrLine(text="Name Mr. SOMCHAI", confidence=0.95, cx=100, cy=80),
    ]
    # r90: also a valid checksum (same document, different rotation), last name reads fine
    # but first name does not.
    r90_lines = [
        OcrLine(text=valid, confidence=0.9),
        OcrLine(text="Last name", confidence=0.95, cx=100, cy=120),
        OcrLine(text="JAIDEE", confidence=0.97, cx=110, cy=150),
    ]

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(
        thai_id_module, "rotations",
        lambda img: [(0, "r0"), (90, "r90"), (180, "r180"), (270, "r270")],
    )
    monkeypatch.setattr(
        thai_id_module, "_run_ocr",
        lambda rotated: {"r0": r0_lines, "r90": r90_lines}.get(rotated, []),
    )
    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", lambda rotated, **kwargs: [])

    result, err = scan_thai_id(b"fake-bytes")
    assert err is None
    assert result is not None
    assert result.document_number == valid
    assert result.document_valid is True
    assert result.first_name == "SOMCHAI"
    assert result.last_name == "JAIDEE"


def test_scan_thai_id_prefers_id_number_multiple_rotations_agree_on(monkeypatch):
    """A single rotation can land on a *different*, coincidentally checksum-valid 13-digit
    number read with high per-line confidence (e.g. unrelated digits printed elsewhere on the
    card) — this must not outrank the number two independent rotations agree on."""
    correct = _valid_id()
    # Construct a second, different checksum-valid number to act as the confident-but-wrong read.
    other_base = "110170015765"
    other_total = sum(int(other_base[i]) * (13 - i) for i in range(12))
    other_check = (11 - (other_total % 11)) % 10
    wrong = other_base + str(other_check)
    assert wrong != correct

    lines_by_rotation = {
        # Higher single-line confidence, but only one rotation reads this number.
        "r0": [OcrLine(text=wrong, confidence=1.0)],
        "r90": [OcrLine(text=correct, confidence=0.85)],
        "r180": [OcrLine(text=correct, confidence=0.85)],
    }

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(
        thai_id_module, "rotations",
        lambda img: [(0, "r0"), (90, "r90"), (180, "r180"), (270, "r270")],
    )
    monkeypatch.setattr(
        thai_id_module, "_run_ocr",
        lambda rotated: lines_by_rotation.get(rotated, []),
    )
    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", lambda rotated, **kwargs: [])

    result, err = scan_thai_id(b"fake-bytes")
    assert err is None
    assert result is not None
    assert result.document_number == correct


def test_scan_thai_id_tries_fallback_when_preferred_reads_only_bare_id(monkeypatch):
    """A wrong orientation can hallucinate a checksum-valid number from unrelated digits. If the
    preferred orientation reads no name/DOB/sex, scan the fallback orientations before trusting it."""
    correct = _valid_id()
    other_base = "110170015765"
    other_total = sum(int(other_base[i]) * (13 - i) for i in range(12))
    wrong = other_base + str((11 - (other_total % 11)) % 10)
    assert wrong != correct

    preferred_lines = [OcrLine(text=wrong, confidence=1.0)]
    fallback_lines = [
        OcrLine(text=correct, confidence=0.9),
        OcrLine(text="Name Mr. SOMCHAI", confidence=0.95),
        OcrLine(text="Last name JAIDEE", confidence=0.94),
        OcrLine(text="Date of Birth 1 Jan. 1990", confidence=0.93),
    ]

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(thai_id_module, "rotations", lambda img: [(0, "r0"), (90, "r90")])
    monkeypatch.setattr(thai_id_module, "_rotation_plan", lambda img: ([(0, "r0")], [(90, "r90")]))
    monkeypatch.setattr(
        thai_id_module,
        "_run_ocr",
        lambda rotated: {"r0": preferred_lines, "r90": fallback_lines}.get(rotated, []),
    )
    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", lambda rotated, **kwargs: [])

    result, err = scan_thai_id(b"fake-bytes")

    assert err is None
    assert result is not None
    assert result.document_number == correct
    assert result.first_name == "SOMCHAI"
    assert result.last_name == "JAIDEE"


def test_scan_thai_id_uses_tesseract_when_paddle_reads_no_supported_fields(monkeypatch):
    valid = _valid_id()
    tesseract_lines = [
        OcrLine(text=f"Identification Number {valid}", confidence=0.9),
        OcrLine(text="Name Mr. SOMCHAI", confidence=0.95),
        OcrLine(text="Last name JAIDEE", confidence=0.94),
        OcrLine(text="Date of Birth 2May 1990", confidence=0.93),
    ]
    calls: list[str] = []

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(thai_id_module, "rotations", lambda img: [(0, "r0"), (90, "r90")])
    monkeypatch.setattr(thai_id_module, "_rotation_plan", lambda img: ([(0, "r0")], [(90, "r90")]))
    monkeypatch.setattr(thai_id_module, "_run_ocr", lambda rotated: [])

    def fake_tesseract(rotated, **kwargs):
        calls.append((rotated, kwargs["scale"], kwargs["psm"]))
        return tesseract_lines if rotated == "r0" else []

    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", fake_tesseract)

    result, err = scan_thai_id(b"fake-bytes")

    assert err is None
    assert result is not None
    assert calls == [("r0", 1, 11)]
    assert result.document_number == valid
    assert result.first_name == "SOMCHAI"
    assert result.last_name == "JAIDEE"
    assert result.date_of_birth == date(1990, 5, 2)


def test_scan_thai_id_tries_next_tesseract_config_before_paddle(monkeypatch):
    """Some cards read correctly only at a different Tesseract scale/PSM. Try the next
    Tesseract config before paying for Paddle's heavier model path."""
    valid = _valid_id()
    complete_lines = [
        OcrLine(text=valid, confidence=0.95),
        OcrLine(text="Name Mr. SOMCHAI", confidence=0.95),
        OcrLine(text="Last name JAIDEE", confidence=0.94),
        OcrLine(text="Date of Birth 2May 1990", confidence=0.93),
    ]
    tess_calls: list[tuple[str, int, int]] = []
    paddle_calls: list[str] = []

    monkeypatch.setattr(thai_id_module, "preprocess", lambda image_bytes: "preprocessed")
    monkeypatch.setattr(thai_id_module, "rotations", lambda img: [(0, "r0")])
    monkeypatch.setattr(thai_id_module, "_rotation_plan", lambda img: ([(0, "r0")], []))

    def fake_tesseract(rotated, **kwargs):
        tess_calls.append((rotated, kwargs["scale"], kwargs["psm"]))
        if kwargs == {"scale": 2, "psm": 11}:
            return complete_lines
        return []

    def fake_run_ocr(rotated):
        paddle_calls.append(rotated)
        return []

    monkeypatch.setattr(thai_id_module, "_run_tesseract_ocr", fake_tesseract)
    monkeypatch.setattr(thai_id_module, "_run_ocr", fake_run_ocr)

    result, err = scan_thai_id(b"fake-bytes")

    assert err is None
    assert result is not None
    assert result.document_number == valid
    assert tess_calls == [("r0", 1, 11), ("r0", 2, 11)]
    assert paddle_calls == []
