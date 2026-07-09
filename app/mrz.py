from dataclasses import dataclass

from app.validators import mrz_check_digit_matches, mrz_dob_to_iso


@dataclass
class MrzParsed:
    document_number: str | None
    surname: str | None
    given_names: str | None
    nationality: str | None
    date_of_birth_iso: str | None
    sex: str | None
    valid: bool


def _parse_name(name_field: str) -> tuple[str | None, str | None]:
    parts = name_field.split("<<", 1)
    surname = parts[0].replace("<", " ").strip() or None
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else None
    return surname, (given or None)


def _sex_from_mrz(c: str) -> str | None:
    if c == "M":
        return "M"
    if c == "F":
        return "F"
    return None


def td3_line2_checks(line2: str) -> tuple[bool, bool, bool, bool, bool]:
    """Validate TD3 line 2's five check digits (doc number, DOB, expiry, personal, composite).
    Assumes len(line2) == 44 — used by parse_td3, by realignment scoring, and by confidence
    scoring (each field's own check digit vs. the composite is a stronger correctness signal
    than the single aggregate `valid` bool)."""
    doc_field = line2[0:9]
    doc_check = line2[9]
    dob_field = line2[13:19]
    dob_check = line2[19]
    expiry_field = line2[21:27]
    expiry_check = line2[27]
    personal_field = line2[28:42]
    personal_check = line2[42]
    composite_check = line2[43]

    doc_ok = mrz_check_digit_matches(doc_field, doc_check)
    dob_ok = mrz_check_digit_matches(dob_field, dob_check)
    expiry_ok = mrz_check_digit_matches(expiry_field, expiry_check)
    if personal_field.replace("<", "") == "":
        personal_ok = personal_check in ("<", "0")
    else:
        personal_ok = mrz_check_digit_matches(personal_field, personal_check)
    composite_field = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]
    composite_ok = mrz_check_digit_matches(composite_field, composite_check)
    return doc_ok, dob_ok, expiry_ok, personal_ok, composite_ok


def td3_line2_check_score(line2: str) -> int:
    """Count how many of TD3 line 2's five check digits are internally consistent (0-5).
    Used to pick the best-aligned candidate when raw OCR may have inserted/dropped a character."""
    if len(line2) != 44:
        return 0
    return sum(td3_line2_checks(line2))


def parse_td3(line1: str, line2: str) -> MrzParsed | None:
    """Parse ICAO 9303 TD3 passport MRZ (two 44-char lines). Returns None if lines malformed."""
    line1 = line1.strip()
    line2 = line2.strip()
    if len(line1) != 44 or len(line2) != 44:
        return None
    if not line1.startswith("P"):
        return None

    surname, given_names = _parse_name(line1[5:44])

    doc_field = line2[0:9]
    nationality = line2[10:13].replace("<", "").strip() or None
    dob_field = line2[13:19]
    sex_char = line2[20]

    doc_ok, dob_ok, expiry_ok, personal_ok, composite_ok = td3_line2_checks(line2)

    document_number = doc_field.replace("<", "").strip() or None

    return MrzParsed(
        document_number=document_number,
        surname=surname,
        given_names=given_names,
        nationality=nationality,
        date_of_birth_iso=mrz_dob_to_iso(dob_field),
        sex=_sex_from_mrz(sex_char),
        valid=all([doc_ok, dob_ok, expiry_ok, personal_ok, composite_ok]),
    )
