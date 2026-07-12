from datetime import date

from app.validators import (
    mrz_check_digit,
    mrz_check_digit_matches,
    mrz_dob_to_iso,
    thai_id_checksum,
)


def test_mrz_check_digit_known_vector():
    # ICAO 9303 spec example: D23145890 → 7
    assert mrz_check_digit("D23145890") == 7


def test_mrz_check_digit_filler():
    assert mrz_check_digit_matches("<<<<<<<<<", "<")
    assert not mrz_check_digit_matches("ABC123", "<")


def test_mrz_check_digit_non_digit_expected():
    assert not mrz_check_digit_matches("D23145890", "X")


def test_mrz_dob_to_iso_chooses_past_century():
    # YY=85 today → 1985 (2085 is in the future)
    assert mrz_dob_to_iso("850615") == "1985-06-15"


def test_mrz_dob_to_iso_recent_year_uses_2000s():
    # YY=05 today → 2005 (in the past, prefer 2000s)
    assert mrz_dob_to_iso("050101") == "2005-01-01"


def test_mrz_dob_to_iso_invalid_returns_none():
    assert mrz_dob_to_iso("999999") is None
    assert mrz_dob_to_iso("abc123") is None
    assert mrz_dob_to_iso("85061") is None


def test_thai_id_checksum_valid_synthetic():
    # Build a synthetic 13-digit ID that passes mod-11
    base = "110170015764"  # 12 digits
    total = sum(int(base[i]) * (13 - i) for i in range(12))
    check = (11 - (total % 11)) % 10
    assert thai_id_checksum(base + str(check))


def test_thai_id_checksum_invalid_when_check_digit_wrong():
    base = "110170015764"
    total = sum(int(base[i]) * (13 - i) for i in range(12))
    correct = (11 - (total % 11)) % 10
    wrong = (correct + 1) % 10
    assert not thai_id_checksum(base + str(wrong))


def test_thai_id_checksum_rejects_non_digit():
    assert not thai_id_checksum("11017001576A")
    assert not thai_id_checksum("123")


def test_thai_id_checksum_rejects_unicode_digits():
    assert not thai_id_checksum("๑๑๐๑๗๐๐๑๕๗๖๔๖")


def test_mrz_dob_to_iso_future_date_rejected():
    # If today is 2026-04-25, "260601" (Jun 2026) is in the future → reject
    today = date.today()
    if today < date(2026, 6, 1):
        assert mrz_dob_to_iso("260601") is None
