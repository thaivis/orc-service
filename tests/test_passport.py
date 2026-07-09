import numpy as np
import pytesseract

from app.mrz import MrzParsed
from app.scanners import passport as passport_module
from app.schemas import ConfidenceScores, DocumentType, ScanResponse, Sex


def _fake_ocr_data(entries: list[tuple[str, int, int, int, int, float]]) -> dict:
    return {
        "text": [e[0] for e in entries],
        "left": [e[1] for e in entries],
        "top": [e[2] for e in entries],
        "width": [e[3] for e in entries],
        "height": [e[4] for e in entries],
        "conf": [e[5] for e in entries],
    }


def _blank_img() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_read_sex_label_finds_value_below_label(monkeypatch):
    data = _fake_ocr_data([
        ("Sex", 300, 200, 40, 20, 95.0),
        ("F", 310, 230, 15, 20, 90.0),
        ("Height", 500, 200, 60, 20, 92.0),  # unrelated label, different column
        ("1.65", 505, 230, 30, 20, 88.0),
    ])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda img, output_type: data)

    sex, conf = passport_module._read_sex_label(_blank_img())
    assert sex == Sex.F
    assert conf == 0.9


def test_read_sex_label_returns_none_without_label(monkeypatch):
    data = _fake_ocr_data([("Height", 400, 200, 60, 20, 92.0), ("1.65", 410, 230, 30, 20, 90.0)])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda img, output_type: data)

    sex, conf = passport_module._read_sex_label(_blank_img())
    assert sex is None
    assert conf == 0.0


def test_read_sex_label_ignores_out_of_column_letters(monkeypatch):
    """An 'M'/'F' far to the side (e.g. a stray initial in an unrelated field) must not be
    mistaken for the sex value just because it's below the label's row."""
    data = _fake_ocr_data([
        ("Sex", 300, 200, 40, 20, 95.0),
        ("M", 900, 230, 15, 20, 90.0),  # way outside the label's column
    ])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda img, output_type: data)

    sex, conf = passport_module._read_sex_label(_blank_img())
    assert sex is None
    assert conf == 0.0


def test_read_sex_label_uses_bare_token_when_label_not_found(monkeypatch):
    """The printed 'Sex' label is often a faint/italic font that OCRs worse than the bold value
    next to it. If the label itself never turns up but exactly one bare M/F token does, trust it."""
    data = _fake_ocr_data([
        ("Nationality", 300, 150, 90, 20, 40.0),
        ("MYANMAR", 300, 170, 90, 25, 94.0),
        ("M", 300, 220, 20, 22, 83.0),
        ("Date", 300, 250, 40, 20, 60.0),
    ])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda img, output_type: data)

    sex, conf = passport_module._read_sex_label(_blank_img())
    assert sex == Sex.M
    assert conf == 0.83


def test_read_sex_label_ambiguous_bare_tokens_without_label_returns_none(monkeypatch):
    """Two bare M/F tokens with no 'Sex' label to disambiguate — too risky to guess."""
    data = _fake_ocr_data([
        ("M", 300, 220, 20, 22, 83.0),
        ("F", 600, 400, 20, 22, 70.0),
    ])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda img, output_type: data)

    sex, conf = passport_module._read_sex_label(_blank_img())
    assert sex is None
    assert conf == 0.0


def test_fill_sex_from_visual_zone_skips_when_already_present():
    response = ScanResponse(
        type=DocumentType.PASSPORT, sex=Sex.M,
        confidence=ConfidenceScores(overall=0.8, sex=0.8),
    )
    result = passport_module._fill_sex_from_visual_zone(response, raw=_blank_img())
    assert result is response


def test_fill_sex_from_visual_zone_returns_original_when_raw_missing():
    response = ScanResponse(type=DocumentType.PASSPORT, sex=None, confidence=ConfidenceScores(overall=0.8))
    result = passport_module._fill_sex_from_visual_zone(response, raw=None)
    assert result is response


def test_fill_sex_from_visual_zone_patches_missing_sex(monkeypatch):
    monkeypatch.setattr(passport_module, "rotations", lambda img: [(0, img)])
    monkeypatch.setattr(passport_module, "_read_sex_label", lambda img: (Sex.F, 0.9))

    response = ScanResponse(
        type=DocumentType.PASSPORT, sex=None,
        confidence=ConfidenceScores(overall=0.8, sex=0.0),
    )
    result = passport_module._fill_sex_from_visual_zone(response, raw=_blank_img())

    assert result is not response
    assert result.sex == Sex.F
    assert result.confidence.sex == 0.9
    assert response.sex is None  # original untouched


def test_fill_sex_from_visual_zone_leaves_sex_none_when_no_rotation_recovers_it(monkeypatch):
    monkeypatch.setattr(passport_module, "rotations", lambda img: [(0, img), (90, img)])
    monkeypatch.setattr(passport_module, "_read_sex_label", lambda img: (None, 0.0))

    response = ScanResponse(type=DocumentType.PASSPORT, sex=None, confidence=ConfidenceScores(overall=0.8))
    result = passport_module._fill_sex_from_visual_zone(response, raw=_blank_img())

    assert result.sex is None


def test_extract_mrz_lines_rejects_identical_lines():
    """A mockup/specimen passport graphic can print the same line twice instead of a real TD3
    line 2 — since line 2 never legitimately equals line 1, that's a sign there's no usable
    document_number/date_of_birth/sex here, not just a checksum-fooling OCR misread."""
    line = "P<THACITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<"
    text = f"{line}\n{line}\n"

    assert passport_module._extract_mrz_lines(text) is None


def _garbage_parsed(score_fields: bool = True) -> MrzParsed:
    return MrzParsed(
        document_number="MITBTTSE" if score_fields else None,
        surname="ANFARAIAIGLES A", given_names="09  9 C",
        nationality="MRB", date_of_birth_iso=None, sex="F", valid=False,
    )


def test_scan_passport_rejects_low_score_fallback_and_salvages_line1(monkeypatch):
    """A checksum score of 1/5 is statistically close to a garbage OCR read matching by chance —
    it must not be trusted as a real document_number/name/sex; the scan should instead fall
    through to the line1-only salvage pass."""
    monkeypatch.setattr(passport_module, "preprocess", lambda b: "IMG")
    monkeypatch.setattr(passport_module, "decode_image", lambda b: "RAW")
    monkeypatch.setattr(passport_module, "normalize_size", lambda r: r)
    monkeypatch.setattr(passport_module, "_get_detector", lambda: None)
    monkeypatch.setattr(passport_module, "rotations", lambda src: [(0, src)])
    monkeypatch.setattr(
        passport_module, "_try_parse",
        lambda detector, img: (_garbage_parsed(), (True, False, False, False, False), 0.6),
    )
    monkeypatch.setattr(passport_module, "_try_parse_direct", lambda img: None)
    salvaged = MrzParsed(
        document_number=None, surname="HTUT", given_names="AUNG WIN",
        nationality="MMR", date_of_birth_iso=None, sex=None, valid=False,
    )
    monkeypatch.setattr(passport_module, "_try_parse_line1", lambda img: (salvaged, 0.7))
    monkeypatch.setattr(passport_module, "_fill_sex_from_visual_zone", lambda response, raw: response)

    result, err = passport_module.scan_passport(b"fake")

    assert err is None
    assert result.last_name == "HTUT"
    assert result.document_number is None
    assert result.sex is None


_VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
_VALID_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


class _FakeDetector:
    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    def get_details(self, img, input_type, ignore_parse):
        return self._raw_text


def test_try_parse_uses_real_ocr_confidence_over_line2_score_proxy(monkeypatch):
    """fastmrz never exposes real per-word OCR confidence, so line1 (name) fields used to fall
    back to a proxy derived from line2's checksum score alone — but line 1 has no checksum of
    its own in TD3, so a perfectly clean line 2 (which would proxy to 100%) says nothing about
    whether the name itself was read correctly. The independent supplementary OCR read's real
    (and here, much lower) confidence must win over that misleading proxy."""
    raw_text = _VALID_LINE1 + "\n" + _VALID_LINE2  # line2 is 5/5 clean -> proxy alone would be 1.0
    monkeypatch.setattr(
        passport_module, "_ocr_mrz_text_and_conf",
        lambda img: (raw_text, {_VALID_LINE1: 55.0, _VALID_LINE2: 92.0}),
    )

    result = passport_module._try_parse(_FakeDetector(raw_text), _blank_img())

    assert result is not None
    parsed, checks, name_conf = result
    assert sum(checks) == 5  # confirms the proxy path would otherwise have said 1.0
    assert name_conf == 0.6  # bucketed from the real (low) line1 OCR confidence instead


def test_try_parse_disagreement_between_engines_caps_confidence_low(monkeypatch):
    """Real bug caught by hand-testing: fastmrz misread one character in the name as a
    different, similarly-shaped letter. The independent supplementary OCR read the *correct*
    text instead — but its own self-reported confidence for that (different, correct) text was
    high (~90%+). Naively borrowing that confidence for fastmrz's (wrong) returned text would
    show a wrong name as 90% trustworthy. The two reads disagreeing must cap confidence low
    regardless of either engine's self-reported score."""
    fastmrz_line1 = "P<THASATJARAK<<KITTIKMWN<<<<<<<<<<<<<<<<<<<<"  # wrong: H misread as W
    supplementary_line1 = "P<THASATJARAK<<KITTIKHUN<<<<<<<<<<<<<<<<<<<<"  # correct
    raw_text = fastmrz_line1 + "\n" + _VALID_LINE2
    monkeypatch.setattr(
        passport_module, "_ocr_mrz_text_and_conf",
        lambda img: (
            supplementary_line1 + "\n" + _VALID_LINE2,
            {supplementary_line1: 92.0, _VALID_LINE2: 92.0},
        ),
    )

    result = passport_module._try_parse(_FakeDetector(raw_text), _blank_img())

    assert result is not None
    parsed, checks, name_conf = result
    assert parsed.given_names == "KITTIKMWN"  # the (wrong) text actually being returned
    assert name_conf == 0.6  # capped low by disagreement, not the supplementary read's 92%


def test_try_parse_falls_back_to_line2_score_proxy_when_supplementary_ocr_finds_no_line1(monkeypatch):
    """If fastmrz's specialized ROI detector reads the MRZ but a generic independent OCR pass
    can't locate any plausible line1 on the same image, fall back to the old proxy rather than
    silently reporting zero/missing confidence for a field that *is* present."""
    raw_text = _VALID_LINE1 + "\n" + _VALID_LINE2
    monkeypatch.setattr(passport_module, "_ocr_mrz_text_and_conf", lambda img: ("garbage, no mrz here", {}))

    result = passport_module._try_parse(_FakeDetector(raw_text), _blank_img())

    assert result is not None
    parsed, checks, name_conf = result
    assert sum(checks) == 5
    assert name_conf == 1.0  # falls back to tier_from_line2_score_proxy(True, 5)


def test_to_response_applies_name_confidence_floor_when_document_fully_valid():
    """Real bug caught by hand-testing: Tesseract's self-reported confidence for the MRZ font
    model can be very low (single digits) even on a byte-for-byte correct read. When the whole
    document passes all five line-2 checksums (strong independent evidence this OCR pass was
    accurate), name confidence should be raised to a floor instead of staying stuck at the raw
    (under-confident) OCR score."""
    parsed = MrzParsed(
        document_number="L898902C3", surname="ERIKSSON", given_names="ANNA MARIA",
        nationality="UTO", date_of_birth_iso="1974-08-12", sex="F", valid=True,
    )
    checks = (True, True, True, True, True)

    response = passport_module._to_response(parsed, checks, 0.6)

    assert response.confidence.first_name == 0.8
    assert response.confidence.middle_name == 0.8
    assert response.confidence.last_name == 0.8


def test_to_response_does_not_apply_floor_when_document_not_fully_valid():
    """A partial line-2 score (e.g. 4/5) is common even when line 1 has a real character-level
    error, so only a *full* 5/5 pass may raise the floor — anything less must leave the raw OCR
    score alone."""
    parsed = MrzParsed(
        document_number="L898902C3", surname="ERIKSSON", given_names="ANNA MARIA",
        nationality="UTO", date_of_birth_iso="1974-08-12", sex="F", valid=False,
    )
    checks = (True, True, True, True, False)  # composite fails -> not fully valid

    response = passport_module._to_response(parsed, checks, 0.6)

    assert response.confidence.first_name == 0.6
    assert response.confidence.last_name == 0.6


def test_scan_passport_accepts_fallback_at_min_score(monkeypatch):
    monkeypatch.setattr(passport_module, "preprocess", lambda b: "IMG")
    monkeypatch.setattr(passport_module, "decode_image", lambda b: "RAW")
    monkeypatch.setattr(passport_module, "normalize_size", lambda r: r)
    monkeypatch.setattr(passport_module, "_get_detector", lambda: None)
    monkeypatch.setattr(passport_module, "rotations", lambda src: [(0, src)])
    monkeypatch.setattr(
        passport_module, "_try_parse",
        lambda detector, img: (_garbage_parsed(), (True, True, False, False, False), 0.7),
    )
    monkeypatch.setattr(passport_module, "_try_parse_direct", lambda img: None)
    monkeypatch.setattr(passport_module, "_fill_sex_from_visual_zone", lambda response, raw: response)

    result, err = passport_module.scan_passport(b"fake")

    assert err is None
    assert result.document_number == "MITBTTSE"
