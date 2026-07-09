from app.confidence import (
    name_confidence_floor,
    tier_from_cross_engine_agreement,
    tier_from_field_checksum,
    tier_from_id_checksum,
    tier_from_line2_score_proxy,
    tier_from_ocr_score,
    tier_from_unchecksummed_field,
)


def test_tier_from_ocr_score_ladder():
    assert tier_from_ocr_score(0.0) == 0.0
    assert tier_from_ocr_score(0.5) == 0.6
    assert tier_from_ocr_score(0.65) == 0.7
    assert tier_from_ocr_score(0.75) == 0.8
    assert tier_from_ocr_score(0.85) == 0.9
    assert tier_from_ocr_score(0.95) == 1.0
    assert tier_from_ocr_score(1.0) == 1.0


def test_tier_from_field_checksum_missing_when_absent():
    assert tier_from_field_checksum(False, True, True) == 0.0


def test_tier_from_field_checksum_both_pass_is_max():
    assert tier_from_field_checksum(True, own_ok=True, composite_ok=True) == 1.0


def test_tier_from_field_checksum_own_only():
    assert tier_from_field_checksum(True, own_ok=True, composite_ok=False) == 0.9


def test_tier_from_field_checksum_composite_only():
    assert tier_from_field_checksum(True, own_ok=False, composite_ok=True) == 0.7


def test_tier_from_field_checksum_neither():
    assert tier_from_field_checksum(True, own_ok=False, composite_ok=False) == 0.6


def test_tier_from_unchecksummed_field_counts_down():
    assert tier_from_unchecksummed_field(True, 4) == 1.0
    assert tier_from_unchecksummed_field(True, 3) == 0.9
    assert tier_from_unchecksummed_field(True, 2) == 0.8
    assert tier_from_unchecksummed_field(True, 1) == 0.7
    assert tier_from_unchecksummed_field(True, 0) == 0.6
    assert tier_from_unchecksummed_field(False, 4) == 0.0


def test_tier_from_line2_score_proxy():
    assert tier_from_line2_score_proxy(True, 5) == 1.0
    assert tier_from_line2_score_proxy(True, 4) == 0.9
    assert tier_from_line2_score_proxy(True, 3) == 0.8
    assert tier_from_line2_score_proxy(True, 2) == 0.7
    assert tier_from_line2_score_proxy(False, 5) == 0.0


def test_tier_from_id_checksum_valid_is_always_max():
    # Even a mediocre OCR read of a checksum-valid number is trustworthy — the checksum, not
    # the OCR engine's self-reported score, is the ground truth here.
    assert tier_from_id_checksum(True, id_valid=True, raw_conf_0_1=0.5) == 1.0


def test_tier_from_id_checksum_invalid_caps_below_max():
    # A failing checksum means the number is known-wrong; even a high raw OCR confidence must
    # not present it as fully trustworthy.
    assert tier_from_id_checksum(True, id_valid=False, raw_conf_0_1=0.99) == 0.9


def test_tier_from_id_checksum_missing():
    assert tier_from_id_checksum(False, id_valid=False, raw_conf_0_1=0.9) == 0.0


def test_tier_from_cross_engine_agreement_disagree_caps_low_regardless_of_confidence():
    # A self-confident wrong answer must not outrank a clear disagreement signal.
    assert tier_from_cross_engine_agreement(agree=False, raw_conf_0_1_if_agree=0.99) == 0.6


def test_tier_from_cross_engine_agreement_agree_trusts_the_ocr_score():
    assert tier_from_cross_engine_agreement(agree=True, raw_conf_0_1_if_agree=0.9) == 0.9
    assert tier_from_cross_engine_agreement(agree=True, raw_conf_0_1_if_agree=0.5) == 0.6


def test_name_confidence_floor_raises_low_score_when_document_fully_valid():
    assert name_confidence_floor(document_fully_valid=True) == 0.8


def test_name_confidence_floor_is_zero_when_not_fully_valid():
    assert name_confidence_floor(document_fully_valid=False) == 0.0
