from app.mrz import parse_td3, td3_line2_check_score
from app.scanners.passport import _fix_numeric_confusables, _realign_line2, scan_passport_from_text


# Real-format synthetic TD3 from ICAO 9303 examples
LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_parse_td3_extracts_fields():
    p = parse_td3(LINE1, LINE2)
    assert p is not None
    assert p.surname == "ERIKSSON"
    assert p.given_names == "ANNA MARIA"
    assert p.document_number == "L898902C3"
    assert p.nationality == "UTO"
    assert p.date_of_birth_iso == "1974-08-12"
    assert p.sex == "F"


def test_parse_td3_valid_check_digits():
    p = parse_td3(LINE1, LINE2)
    assert p is not None
    assert p.valid is True


def test_parse_td3_rejects_wrong_length():
    assert parse_td3("P<", LINE2) is None
    assert parse_td3(LINE1, "X<") is None


def test_parse_td3_requires_passport_marker():
    swapped = "I" + LINE1[1:]
    assert parse_td3(swapped, LINE2) is None


def test_parse_td3_tampered_doc_number_marks_invalid():
    tampered = "L898902C46UTO7408122F1204159ZE184226B<<<<<10"
    p = parse_td3(LINE1, tampered)
    assert p is not None
    assert p.valid is False


def test_scan_passport_from_text_happy_path():
    raw = LINE1 + "\n" + LINE2
    result, err = scan_passport_from_text(raw)
    assert err is None
    assert result is not None
    assert result.first_name == "ANNA"
    assert result.middle_name == "MARIA"
    assert result.last_name == "ERIKSSON"
    assert result.document_number == "L898902C3"
    assert result.country == "UTO"
    assert result.sex.value == "F"
    assert result.document_valid is True


def test_scan_passport_from_text_clean_mrz_has_max_confidence_everywhere():
    # All five TD3 line-2 checksums pass and the (synthetic, no-OCR) name confidence is 1.0 —
    # every populated field, and therefore overall, should land on the top tier.
    raw = LINE1 + "\n" + LINE2
    result, err = scan_passport_from_text(raw)
    assert err is None
    assert result is not None
    assert result.confidence.overall == 1.0
    assert result.confidence.document_number == 1.0
    assert result.confidence.date_of_birth == 1.0
    assert result.confidence.sex == 1.0
    assert result.confidence.country == 1.0
    assert result.confidence.first_name == 1.0
    assert result.confidence.middle_name == 1.0
    assert result.confidence.last_name == 1.0


def test_scan_passport_from_text_tampered_doc_number_lowers_only_that_field():
    # Tampering document_number's content breaks both its own check digit and the composite
    # check (which recomputes over the same characters) — document_number should drop to the
    # lowest tier, but the other line-2 checks (dob/expiry/personal) still pass, so sex/country
    # (which have no check digit of their own) stay high via the unchecksummed-field proxy.
    tampered = "L898902C46UTO7408122F1204159ZE184226B<<<<<10"
    raw = LINE1 + "\n" + tampered
    result, err = scan_passport_from_text(raw)
    assert err is None
    assert result is not None
    assert result.document_valid is False
    assert result.confidence.document_number == 0.6
    # dob's own check digit still passes, but the composite check digit (which recomputes over
    # doc_field too) no longer does — tier_from_field_checksum grades that as HIGH, not MAX.
    assert result.confidence.date_of_birth == 0.9
    assert result.confidence.sex == 0.9
    assert result.confidence.country == 0.9
    assert result.confidence.overall == 0.6


def test_scan_passport_from_text_no_mrz_returns_error():
    result, err = scan_passport_from_text("not an mrz at all\nnope")
    assert result is None
    assert err is not None
    assert err.code == "no_document_detected"
    assert err.detected_type is None


def test_realign_line2_recovers_from_spurious_inserted_character():
    # Raw OCR read gains one stray leading character before _pad_mrz truncates it back to 44 —
    # shifting every field in the line by one position and dropping the true trailing composite
    # check digit. This is the exact mechanism behind the real "THA" -> "2TH" bug report: the
    # shifted document number ("L898902C3" -> "9L89890 2C") and nationality both come out wrong,
    # but every field's check digit still lines up against fixed ICAO offsets, so nothing but the
    # cross-field checksum comparison can reveal the misalignment.
    raw_ocr_line2 = "9" + LINE2
    assert td3_line2_check_score(LINE2) == 5
    assert td3_line2_check_score(raw_ocr_line2[:44]) < 5

    realigned = _realign_line2(raw_ocr_line2)
    assert realigned == LINE2
    p = parse_td3(LINE1, realigned)
    assert p is not None
    assert p.document_number == "L898902C3"
    assert p.nationality == "UTO"
    assert p.valid is True


def test_realign_line2_leaves_clean_line_untouched():
    assert _realign_line2(LINE2) == LINE2


def test_realign_line2_falls_back_when_unrecoverable():
    garbage = "X" * 44
    assert _realign_line2(garbage) == garbage


def test_fix_numeric_confusables_corrects_digit_only_zones():
    # ICAO 9303 mandates digits-only in the DOB/expiry fields and every check digit; OCR
    # frequently misreads a '0' as letter 'O' there (visually near-identical in most fonts).
    chars = list(LINE2)
    chars[15] = "O"  # inside the date-of-birth field
    chars[23] = "O"  # inside the expiry field
    chars[43] = "O"  # the composite check digit itself
    corrupted = "".join(chars)
    assert td3_line2_check_score(corrupted) < 5

    assert _fix_numeric_confusables(corrupted) == LINE2


def test_fix_numeric_confusables_leaves_document_number_letters_untouched():
    # document_number ("L898902C3") legitimately contains a letter — must survive unchanged even
    # though it looks superficially like the confusable set used to correct the numeric zones.
    assert _fix_numeric_confusables(LINE2)[:9] == LINE2[:9]


def test_realign_line2_recovers_from_confusable_digits_and_inserted_character():
    chars = list(LINE2)
    chars[15] = "O"
    chars[23] = "O"
    raw_ocr_line2 = "9" + "".join(chars)  # stray leading char on top of the digit misreads

    assert _realign_line2(raw_ocr_line2) == LINE2


def test_realign_line2_stays_single_pass_on_multi_character_drift():
    # A raw OCR read with two separate stray characters is beyond what a single-edit search can
    # fully realign. This is deliberate (see _realign_line2's docstring): chaining more rounds to
    # chase full recovery was tried and reverted because it let pure garbage cross the same
    # check-digit score bar as a real partial recovery. Confirm this stays capped, not "fixed".
    chars = list(LINE2)
    chars.insert(5, "9")
    chars.insert(32, "X")
    raw_ocr_line2 = "".join(chars)

    assert _realign_line2(raw_ocr_line2) != LINE2
