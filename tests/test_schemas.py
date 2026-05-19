from app.schemas import ConfidenceScores, DocumentType, ScanResponse


def _minimal_response() -> ScanResponse:
    return ScanResponse(
        type=DocumentType.PASSPORT,
        confidence=ConfidenceScores(overall=1.0),
    )


def test_image_url_defaults_to_none():
    resp = _minimal_response()
    assert resp.image_url is None


def test_image_url_accepts_string():
    resp = _minimal_response()
    resp.image_url = "https://example.com/scan.jpg"
    assert resp.image_url == "https://example.com/scan.jpg"


def test_scan_response_serialises_without_image_url():
    data = _minimal_response().model_dump(exclude_none=True)
    assert "image_url" not in data


def test_scan_response_serialises_with_image_url():
    resp = _minimal_response()
    resp.image_url = "https://example.com/scan.jpg"
    data = resp.model_dump(exclude_none=True)
    assert data["image_url"] == "https://example.com/scan.jpg"
