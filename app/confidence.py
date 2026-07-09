"""Shared confidence-tier spec used by every scanner.

Every field confidence collapses onto one of five discrete tiers (0.6/0.7/0.8/0.9/1.0) rather
than a raw continuous float — the underlying signals (a checksum pass/fail, an OCR engine's own
score) are not precise enough to justify reporting more resolution than this, and discrete tiers
give callers a stable threshold to build "flag for manual review"/"safe to auto-fill" logic on.
"""

TIER_MISSING = 0.0
TIER_LOW = 0.6
TIER_LOWMED = 0.7
TIER_MEDIUM = 0.8
TIER_HIGH = 0.9
TIER_MAX = 1.0

_OCR_LADDER = (
    (0.95, TIER_MAX),
    (0.85, TIER_HIGH),
    (0.75, TIER_MEDIUM),
    (0.65, TIER_LOWMED),
)


def tier_from_ocr_score(raw_conf_0_1: float) -> float:
    """Snap a continuous OCR engine confidence (0-1) onto the tier scale. Used for fields with
    no independent way to verify correctness (names, and MRZ line1-salvage country) — the tier
    IS the OCR engine's own self-reported confidence, just quantized."""
    if raw_conf_0_1 <= 0:
        return TIER_MISSING
    for threshold, tier in _OCR_LADDER:
        if raw_conf_0_1 >= threshold:
            return tier
    return TIER_LOW


def tier_from_field_checksum(present: bool, own_ok: bool, composite_ok: bool) -> float:
    """Tier for an MRZ TD3 field backed by its own check digit (document_number, date_of_birth).
    Both the field's own check digit and the line's composite check digit are independent
    correctness signals — agreement on both is the strongest possible signal (MAX); agreement on
    only one is still meaningfully more trustworthy than agreement on neither."""
    if not present:
        return TIER_MISSING
    if own_ok and composite_ok:
        return TIER_MAX
    if own_ok:
        return TIER_HIGH
    if composite_ok:
        return TIER_LOWMED
    return TIER_LOW


def tier_from_unchecksummed_field(present: bool, checks_passed: int, total: int = 4) -> float:
    """Tier for an MRZ TD3 field with no check digit of its own (sex, nationality/country) — ICAO
    9303 doesn't cover either in a checksum. The only available signal is how many of the *other*
    four line-2 checks (doc/dob/expiry/personal) came out consistent, as a proxy for "was this
    whole OCR read of the line clean"."""
    if not present:
        return TIER_MISSING
    table = {4: TIER_MAX, 3: TIER_HIGH, 2: TIER_MEDIUM, 1: TIER_LOWMED, 0: TIER_LOW}
    return table.get(checks_passed, TIER_LOW) if total == 4 else TIER_LOW


def tier_from_line2_score_proxy(present: bool, score: int) -> float:
    """Tier for MRZ line1 fields (first_name/middle_name/last_name, and country when salvaged
    from line1) when no real per-word OCR confidence is available for that read — e.g. fastmrz's
    public API discards Tesseract's own confidence. Falls back to how clean the *rest* of the
    document's line-2 checksum came out (0-5) as the best available proxy for OCR quality on this
    same source image."""
    if not present:
        return TIER_MISSING
    table = {5: TIER_MAX, 4: TIER_HIGH, 3: TIER_MEDIUM, 2: TIER_LOWMED}
    return table.get(score, TIER_LOW)


def tier_from_cross_engine_agreement(agree: bool, raw_conf_0_1_if_agree: float) -> float:
    """Tier for a field independently read by two different OCR passes over the same image (e.g.
    fastmrz's internal OCR vs. our own supplementary Tesseract read) when neither read has a
    checksum to verify against — MRZ line 1 (names) is never checksummed under ICAO 9303, so this
    is the only cross-check available for it. Agreement between two independent reads is a much
    stronger correctness signal than either engine's own self-reported confidence: when they
    disagree, that disagreement dominates and caps the tier low regardless of how confident the
    second read was about its own (different) answer — a self-confident wrong answer must not
    outrank a clear sign that the two reads don't match. When they agree, the second read's own
    OCR confidence becomes trustworthy, since it's now describing the exact text being returned."""
    if not agree:
        return TIER_LOW
    return tier_from_ocr_score(raw_conf_0_1_if_agree)


def name_confidence_floor(document_fully_valid: bool) -> float:
    """A floor to raise MRZ line1 (name) confidence to when the *rest* of the document (all five
    TD3 line-2 checksums) verified cleanly. Exists because Tesseract's self-reported per-word
    confidence for the specialized "mrz" OCR-B font model has been observed to run low/near-zero
    even on byte-for-byte correct reads (empirically: a correct name read at 30% self-confidence,
    a checksum-verified-correct line 2 at 0% self-confidence) — so `tier_from_ocr_score` alone
    under-represents genuinely good reads through this model. A full 5/5 checksum pass is only
    ~1-in-100,000 to happen by chance, so it's strong (if indirect) evidence this particular OCR
    pass over this particular image was accurate, including line 1 read in the same pass.

    Deliberately only a *floor* (combine with `max(name_conf, this)`), and deliberately gated on
    *full* validity, not a partial score: a partial line-2 score (e.g. 4/5) is common even when
    line 1 has a real character-level error (confirmed in practice — a misread name coexisted
    with a mostly-clean-but-not-fully-valid line 2), so only the strongest, hardest-to-fake
    signal is allowed to override a low raw OCR score; anything less falls through to
    `tier_from_ocr_score`/`tier_from_cross_engine_agreement` unchanged."""
    return TIER_MEDIUM if document_fully_valid else TIER_MISSING


def tier_from_id_checksum(present: bool, id_valid: bool, raw_conf_0_1: float) -> float:
    """Tier for Thai ID document_number: a passing checksum is a much stronger correctness
    signal than PaddleOCR's own score, so it dominates (always MAX) rather than blending with
    it. A failing checksum falls back to the OCR engine's own confidence, bucketed — but capped
    below MAX, since a known-wrong number should never present as fully trustworthy no matter
    how confident the OCR engine was about reading its (wrong) digits."""
    if not present:
        return TIER_MISSING
    if id_valid:
        return TIER_MAX
    return min(tier_from_ocr_score(raw_conf_0_1), TIER_HIGH)