"""End-to-end tests against real fixtures in `.test-fixtures/`.

Skipped wholesale when paddleocr/fastmrz/httpx are not installed locally — the unit suite
keeps verifying logic. Run via `pytest tests/test_integration.py -v` after a full
`pip install -r requirements.txt` (heavy: paddlepaddle ~500MB)."""

from __future__ import annotations

from pathlib import Path

import importlib.util

import pytest

pytest.importorskip("fastmrz")
pytest.importorskip("httpx")


def _has_tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


HAS_PADDLE = importlib.util.find_spec("paddleocr") is not None
HAS_TESSERACT = _has_tesseract()
needs_paddle = pytest.mark.skipif(not HAS_PADDLE, reason="paddleocr not installed locally")
needs_tesseract = pytest.mark.skipif(
    not HAS_TESSERACT, reason="tesseract binary not on PATH (fastmrz needs it)"
)

from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent.parent / ".test-fixtures"
API_KEY = "test-key"  # set by conftest.py via env

PASSPORT_FIXTURES = ["fake1.jpg", "fake2.jpg", "fake3.png", "fake5.jpg"]
THAI_ID_FIXTURES = ["fake7.jpg", "fake8.png", "fake9.jpeg"]


def _content_type(name: str) -> str:
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _post_scan(client: TestClient, fixture: str, scan_type: str) -> tuple[int, dict, dict]:
    path = FIXTURES / fixture
    if not path.exists():
        pytest.skip(f"fixture missing: {fixture}")
    endpoint = f"/scan/{scan_type.replace('_', '-')}"
    with open(path, "rb") as f:
        resp = client.post(
            endpoint,
            headers={"X-API-Key": API_KEY},
            files={"image": (fixture, f, _content_type(fixture))},
        )
    return resp.status_code, resp.json(), dict(resp.headers)


# --- Passport happy path ---
# Contract test: response must be either a well-formed 200 ScanResponse OR a 422
# no_document_detected. We don't fail on per-image accuracy (that's an OCR-quality
# concern handled by manual fixture review). At least one passport fixture must
# extract successfully or the engine is broken — see test_passport_engine_extracts_at_least_one.

@needs_tesseract
@pytest.mark.parametrize("fixture", PASSPORT_FIXTURES)
def test_passport_response_is_well_formed(client: TestClient, fixture: str):
    status, body, headers = _post_scan(client, fixture, "passport")
    assert "x-request-id" in {k.lower() for k in headers}
    if status == 200:
        assert body["type"] == "passport"
        assert body["sex"] in ("M", "F", None)
        assert isinstance(body["document_valid"], bool)
        assert 0 <= body["confidence"]["overall"] <= 1
    else:
        assert status == 422, f"{fixture}: unexpected {status} {body}"
        assert body["error"] == "no_document_detected"


@needs_tesseract
def test_passport_engine_extracts_at_least_one(client: TestClient):
    """Smoke check that fastmrz works against at least one fixture — guards against engine
    breakage (model/data file missing, dep regression). Per-image accuracy is out of scope."""
    extracted = []
    for fixture in PASSPORT_FIXTURES:
        status, body, _ = _post_scan(client, fixture, "passport")
        if status == 200 and body.get("document_number"):
            extracted.append(fixture)
    assert extracted, "no passport fixture extracted — engine appears broken"


# --- Thai ID happy path ---

@needs_paddle
@pytest.mark.parametrize("fixture", THAI_ID_FIXTURES)
def test_thai_id_response_is_well_formed(client: TestClient, fixture: str):
    status, body, _ = _post_scan(client, fixture, "thai_id")
    if status == 200:
        assert body["type"] == "thai_id"
        assert body["country"] == "THA"
        if body["document_number"]:
            digits = body["document_number"]
            assert len(digits) == 13 and digits.isdigit(), f"{fixture}: bad id format {digits}"
        assert body["sex"] in ("M", "F", None)
        assert isinstance(body["document_valid"], bool)
    else:
        assert status == 422, f"{fixture}: unexpected {status} {body}"
        assert body["error"] == "no_document_detected"


@needs_paddle
def test_thai_id_engine_extracts_at_least_one(client: TestClient):
    extracted = []
    for fixture in THAI_ID_FIXTURES:
        status, body, _ = _post_scan(client, fixture, "thai_id")
        if status == 200 and body.get("document_number"):
            extracted.append(fixture)
    assert extracted, "no Thai ID fixture extracted — engine appears broken"


# --- Edge cases / input validation ---

def test_oversized_file_rejected(client: TestClient):
    status, body, _ = _post_scan(client, "large.jpg", "passport")
    assert status == 400
    assert body["error"] == "file_too_large"


def test_webp_rejected_unsupported_format(client: TestClient):
    status, body, _ = _post_scan(client, "fake6.webp", "passport")
    assert status == 400
    assert body["error"] == "unsupported_format"


def test_pdf_rejected_unsupported_format(client: TestClient):
    """fake.pdf is sent with image/jpeg content-type → fails magic-byte check → 400 image_invalid."""
    status, body, _ = _post_scan(client, "fake.pdf", "passport")
    assert status == 400
    assert body["error"] in ("unsupported_format", "image_invalid")


@pytest.mark.parametrize("fixture", ["tiny.jpg", "tiny.png", "fake.jpg"])
def test_garbage_files_rejected(client: TestClient, fixture: str):
    status, body, _ = _post_scan(client, fixture, "passport")
    assert status == 400, f"{fixture}: {body}"
    assert body["error"] == "image_invalid"


# --- Request ID flow ---

@needs_tesseract
def test_request_id_header_returned_on_success(client: TestClient):
    _, _, headers = _post_scan(client, PASSPORT_FIXTURES[0], "passport")
    rid = {k.lower(): v for k, v in headers.items()}.get("x-request-id")
    assert rid is not None and len(rid) >= 16


def test_request_id_returned_even_on_auth_failure(client: TestClient):
    resp = client.get("/health")  # public, no PII
    assert resp.status_code == 200
    resp2 = client.post("/scan/passport", headers={"X-API-Key": "wrong"})
    assert resp2.status_code == 401
    assert "x-request-id" in {k.lower() for k in resp2.headers}
