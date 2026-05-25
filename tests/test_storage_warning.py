"""Integration-style tests for storage_upload_failed warning logic.

Each endpoint (/scan/thai-id, /scan/passport) is covered by four cases:
- upload failure   : minio_endpoint set, upload returns None  → warning present, image_url null
- upload exception : minio_endpoint set, upload raises        → warning present, image_url null
- upload success   : minio_endpoint set, upload returns key   → image_url populated, no warning
- storage disabled : minio_endpoint unset                     → image_url null, no warning

No assertions on internal call counts — all assertions target the HTTP response body.
"""
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


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _fake_result(doc_type: DocumentType) -> tuple[ScanResponse, None]:
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


def _minio_settings(**overrides):
    from app.config import Settings

    base = dict(
        api_key=API_KEY,
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="thaivis-id-documents",
        minio_use_ssl=False,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _post(client: TestClient, endpoint: str) -> dict:
    resp = client.post(
        endpoint,
        headers={"X-API-Key": API_KEY},
        files={"image": ("doc.jpg", _jpeg(), "image/jpeg")},
    )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# /scan/thai-id
# ---------------------------------------------------------------------------

def test_thai_id_upload_failure_adds_warning(client: TestClient):
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=None),
        patch("app.main.get_settings", return_value=_minio_settings()),
    ):
        body = _post(client, "/scan/thai-id")
    assert "storage_upload_failed" in body["warnings"]
    assert body["image_url"] is None


def test_thai_id_upload_exception_adds_warning(client: TestClient):
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", side_effect=RuntimeError("boom")),
        patch("app.main.get_settings", return_value=_minio_settings()),
    ):
        body = _post(client, "/scan/thai-id")
    assert "storage_upload_failed" in body["warnings"]
    assert body["image_url"] is None


def test_thai_id_upload_success_no_warning(client: TestClient):
    object_key = "thai-id/2026/05/abc.webp"
    with (
        patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_result(DocumentType.THAI_ID)),
        patch("app.storage.upload_document_image", return_value=object_key),
        patch("app.main.get_settings", return_value=_minio_settings()),
    ):
        body = _post(client, "/scan/thai-id")
    assert "storage_upload_failed" not in body["warnings"]
    assert body["image_url"] is not None and body["image_url"] != ""


def test_thai_id_storage_disabled_no_warning(client: TestClient):
    with patch("app.scanners.thai_id.scan_thai_id", return_value=_fake_result(DocumentType.THAI_ID)):
        body = _post(client, "/scan/thai-id")
    assert "storage_upload_failed" not in body["warnings"]
    assert body["image_url"] is None


# ---------------------------------------------------------------------------
# /scan/passport
# ---------------------------------------------------------------------------

def test_passport_upload_failure_adds_warning(client: TestClient):
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", return_value=None),
        patch("app.main.get_settings", return_value=_minio_settings()),
    ):
        body = _post(client, "/scan/passport")
    assert "storage_upload_failed" in body["warnings"]
    assert body["image_url"] is None


def test_passport_upload_exception_adds_warning(client: TestClient):
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", side_effect=RuntimeError("boom")),
        patch("app.main.get_settings", return_value=_minio_settings()),
    ):
        body = _post(client, "/scan/passport")
    assert "storage_upload_failed" in body["warnings"]
    assert body["image_url"] is None


def test_passport_upload_success_no_warning(client: TestClient):
    object_key = "passport/2026/05/abc.webp"
    with (
        patch("app.scanners.passport.scan_passport", return_value=_fake_result(DocumentType.PASSPORT)),
        patch("app.storage.upload_document_image", return_value=object_key),
        patch("app.main.get_settings", return_value=_minio_settings()),
    ):
        body = _post(client, "/scan/passport")
    assert "storage_upload_failed" not in body["warnings"]
    assert body["image_url"] is not None and body["image_url"] != ""


def test_passport_storage_disabled_no_warning(client: TestClient):
    with patch("app.scanners.passport.scan_passport", return_value=_fake_result(DocumentType.PASSPORT)):
        body = _post(client, "/scan/passport")
    assert "storage_upload_failed" not in body["warnings"]
    assert body["image_url"] is None
