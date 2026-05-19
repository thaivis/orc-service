"""Integration tests for storage wiring in scan endpoints — no real MinIO or OCR engines required."""
from __future__ import annotations

import io
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas import ConfidenceScores, DocumentType, ScanResponse

API_KEY = "test-key"


def _make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(200, 100, 50))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _fake_scan_result(doc_type: DocumentType) -> tuple[ScanResponse, None]:
    return ScanResponse(
        type=doc_type,
        first_name="Test",
        last_name="User",
        document_number="AB1234567",
        date_of_birth=date(1990, 1, 1),
        sex=None,
        country="THA",
        document_valid=True,
        confidence=ConfidenceScores(overall=1.0),
    ), None


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _post(client: TestClient, endpoint: str, jpeg_bytes: bytes) -> tuple[int, dict]:
    resp = client.post(
        endpoint,
        headers={"X-API-Key": API_KEY},
        files={"image": ("doc.jpg", jpeg_bytes, "image/jpeg")},
    )
    return resp.status_code, resp.json()


# ---------------------------------------------------------------------------
# image_url is null when MINIO_ENDPOINT is not configured (default in tests)
# ---------------------------------------------------------------------------

def test_thai_id_image_url_null_when_no_minio(client: TestClient):
    jpeg = _make_jpeg_bytes()
    with patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] is None


def test_passport_image_url_null_when_no_minio(client: TestClient):
    jpeg = _make_jpeg_bytes()
    with patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert body["image_url"] is None


# ---------------------------------------------------------------------------
# image_url is populated when storage returns a URL
# ---------------------------------------------------------------------------

def test_thai_id_image_url_populated_when_storage_returns_url(client: TestClient):
    jpeg = _make_jpeg_bytes()
    expected_url = "http://minio:9000/bucket/thai-id/2026/05/some-uuid.webp"
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=expected_url),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] == expected_url


def test_passport_image_url_populated_when_storage_returns_url(client: TestClient):
    jpeg = _make_jpeg_bytes()
    expected_url = "http://minio:9000/bucket/passport/2026/05/some-uuid.webp"
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", return_value=expected_url),
    ):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert body["image_url"] == expected_url


# ---------------------------------------------------------------------------
# Scan still succeeds even when storage raises an unexpected exception
# ---------------------------------------------------------------------------

def test_thai_id_scan_succeeds_even_when_storage_raises(client: TestClient):
    jpeg = _make_jpeg_bytes()
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", side_effect=RuntimeError("minio unreachable")),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] is None


def test_passport_scan_succeeds_even_when_storage_raises(client: TestClient):
    jpeg = _make_jpeg_bytes()
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", side_effect=RuntimeError("minio unreachable")),
    ):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert body["image_url"] is None
