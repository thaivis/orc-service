"""Integration tests for storage wiring in scan endpoints — no real MinIO or OCR engines required."""
from __future__ import annotations

import io
from datetime import date
from unittest.mock import MagicMock, patch

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


def _make_webp_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))
    img.save(buf, format="WEBP")
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


def _settings_with_minio(**kwargs):
    from app.config import Settings

    defaults = dict(
        api_key=API_KEY,
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="thaivis-id-documents",
        minio_use_ssl=False,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


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
# image_url is /document/{object_key} when storage returns an object_key
# ---------------------------------------------------------------------------

def test_thai_id_image_url_populated_when_storage_returns_key(client: TestClient):
    jpeg = _make_jpeg_bytes()
    object_key = "thai-id/2026/05/some-uuid.webp"
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=object_key),
        patch("app.main.get_settings", return_value=_settings_with_minio()),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] == f"/document/{object_key}"


def test_passport_image_url_populated_when_storage_returns_key(client: TestClient):
    jpeg = _make_jpeg_bytes()
    object_key = "passport/2026/05/some-uuid.webp"
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", return_value=object_key),
        patch("app.main.get_settings", return_value=_settings_with_minio()),
    ):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert body["image_url"] == f"/document/{object_key}"


# ---------------------------------------------------------------------------
# image_url uses PUBLIC_BASE_URL prefix when configured
# ---------------------------------------------------------------------------

def test_thai_id_image_url_uses_public_base_url(client: TestClient):
    jpeg = _make_jpeg_bytes()
    object_key = "thai-id/2026/05/some-uuid.webp"
    settings = _settings_with_minio(public_base_url="https://api.example.com")
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=object_key),
        patch("app.main.get_settings", return_value=settings),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] == f"https://api.example.com/document/{object_key}"


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


# ---------------------------------------------------------------------------
# warnings contains "storage_upload_failed" when MinIO configured but upload fails
# ---------------------------------------------------------------------------

def test_thai_id_warning_when_minio_configured_and_upload_returns_none(client: TestClient):
    jpeg = _make_jpeg_bytes()
    settings = _settings_with_minio()
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=None),
        patch("app.main.get_settings", return_value=settings),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] is None
    assert "storage_upload_failed" in body["warnings"]


def test_passport_warning_when_minio_configured_and_upload_returns_none(client: TestClient):
    jpeg = _make_jpeg_bytes()
    settings = _settings_with_minio()
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", return_value=None),
        patch("app.main.get_settings", return_value=settings),
    ):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert body["image_url"] is None
    assert "storage_upload_failed" in body["warnings"]


def test_thai_id_warning_when_minio_configured_and_upload_raises(client: TestClient):
    jpeg = _make_jpeg_bytes()
    settings = _settings_with_minio()
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", side_effect=RuntimeError("boom")),
        patch("app.main.get_settings", return_value=settings),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert body["image_url"] is None
    assert "storage_upload_failed" in body["warnings"]


def test_passport_warning_when_minio_configured_and_upload_raises(client: TestClient):
    jpeg = _make_jpeg_bytes()
    settings = _settings_with_minio()
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", side_effect=RuntimeError("boom")),
        patch("app.main.get_settings", return_value=settings),
    ):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert body["image_url"] is None
    assert "storage_upload_failed" in body["warnings"]


def test_thai_id_no_warning_when_minio_not_configured(client: TestClient):
    jpeg = _make_jpeg_bytes()
    with patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert "storage_upload_failed" not in body["warnings"]


def test_passport_no_warning_when_minio_not_configured(client: TestClient):
    jpeg = _make_jpeg_bytes()
    with patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert "storage_upload_failed" not in body["warnings"]


def test_thai_id_no_warning_when_upload_succeeds(client: TestClient):
    jpeg = _make_jpeg_bytes()
    object_key = "thai-id/2026/05/some-uuid.webp"
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_scan_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=object_key),
    ):
        status, body = _post(client, "/scan/thai-id", jpeg)
    assert status == 200
    assert "storage_upload_failed" not in body["warnings"]


def test_passport_no_warning_when_upload_succeeds(client: TestClient):
    jpeg = _make_jpeg_bytes()
    object_key = "passport/2026/05/some-uuid.webp"
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_scan_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", return_value=object_key),
    ):
        status, body = _post(client, "/scan/passport", jpeg)
    assert status == 200
    assert "storage_upload_failed" not in body["warnings"]


# ---------------------------------------------------------------------------
# GET /document/{object_key} — 200 returns decrypted image/webp
# ---------------------------------------------------------------------------

def test_decrypt_endpoint_returns_webp(client: TestClient):
    from app.encryption import encrypt

    webp_bytes = _make_webp_bytes()
    encrypted_blob = encrypt(webp_bytes)

    body_mock = MagicMock()
    body_mock.read.return_value = encrypted_blob
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": body_mock}

    with (
        patch("app.main.get_settings", return_value=_settings_with_minio()),
        patch("boto3.client", return_value=mock_s3),
    ):
        resp = client.get(
            "/document/thai-id/2026/05/some-uuid.webp",
            headers={"X-API-Key": API_KEY},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"
    assert resp.content == webp_bytes


# ---------------------------------------------------------------------------
# GET /document/{object_key} — 401 when X-API-Key is absent
# ---------------------------------------------------------------------------

def test_decrypt_endpoint_returns_401_without_key(client: TestClient):
    resp = client.get("/document/thai-id/2026/05/some-uuid.webp")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /document/{object_key} — 404 when object key does not exist in MinIO
# ---------------------------------------------------------------------------

def test_decrypt_endpoint_returns_404_unknown_key(client: TestClient):
    from botocore.exceptions import ClientError

    error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        "GetObject",
    )
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = error

    with (
        patch("app.main.get_settings", return_value=_settings_with_minio()),
        patch("boto3.client", return_value=mock_s3),
    ):
        resp = client.get(
            "/document/thai-id/2026/05/nonexistent.webp",
            headers={"X-API-Key": API_KEY},
        )

    assert resp.status_code == 404
