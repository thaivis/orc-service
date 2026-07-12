from datetime import date


def _mrz_char_value(c: str) -> int:
    if "0" <= c <= "9":
        return ord(c) - ord("0")
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    return 0  # "<" or unexpected


def mrz_check_digit(s: str) -> int:
    """ICAO 9303 check digit: weights (7,3,1) cycle, sum mod 10."""
    weights = (7, 3, 1)
    return sum(_mrz_char_value(c) * weights[i % 3] for i, c in enumerate(s)) % 10


def mrz_check_digit_matches(field: str, expected: str) -> bool:
    """Verify that `expected` (single char or '<') matches check digit of `field`."""
    if expected == "<":
        return field.replace("<", "") == ""
    if not expected.isdigit():
        return False
    return mrz_check_digit(field) == int(expected)


def mrz_dob_to_iso(yymmdd: str) -> str | None:
    """Convert MRZ YYMMDD to ISO 8601, picking century such that date is in the past."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    today = date.today()
    year_20xx = 2000 + yy
    year = year_20xx if year_20xx <= today.year else 1900 + yy
    try:
        d = date(year, mm, dd)
    except ValueError:
        return None
    if d > today:
        return None
    return d.isoformat()


def thai_id_checksum(id_str: str) -> bool:
    """Verify Thai national ID 13-digit mod-11 checksum."""
    if len(id_str) != 13 or not id_str.isascii() or not id_str.isdigit():
        return False
    total = sum(int(id_str[i]) * (13 - i) for i in range(12))
    check = (11 - (total % 11)) % 10
    return check == int(id_str[12])
