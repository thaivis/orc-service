"""Unit tests for app/storage.py — no real MinIO instance required."""
import io
import re
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _make_jpeg_bytes() -> bytes:
    """Return minimal valid JPEG bytes via PIL."""
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(128, 64, 32))
    img.save(buf, format="JPEG")
    return buf.getvalue()


_WEBP_MAGIC = b"RIFF"
_WEBP_MARKER = b"WEBP"


def _is_webp(data: bytes) -> bool:
    return data[:4] == _WEBP_MAGIC and data[8:12] == _WEBP_MARKER


# ---------------------------------------------------------------------------
# Helper: patch settings so tests don't need a real .env
# ---------------------------------------------------------------------------

def _settings(endpoint: str | None = "localhost:9000", **kwargs):
    from app.config import Settings

    defaults = dict(
        api_key="test-key",
        minio_endpoint=endpoint,
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="thaivis-id-documents",
        minio_use_ssl=False,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Test: returns None when MINIO_ENDPOINT is not configured
# ---------------------------------------------------------------------------

def test_returns_none_when_no_endpoint():
    from app.storage import upload_document_image

    with patch("app.storage.get_settings", return_value=_settings(endpoint=None)):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert result is None


# ---------------------------------------------------------------------------
# Test: successful upload returns a URL string
# ---------------------------------------------------------------------------

def test_returns_url_on_success():
    from app.storage import upload_document_image

    mock_client = MagicMock()
    mock_client.put_object.return_value = {}

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("boto3.client", return_value=mock_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert isinstance(result, str)
    assert result.startswith("http://localhost:9000/thaivis-id-documents/")
    assert result.endswith(".webp")


# ---------------------------------------------------------------------------
# Test: object key matches {document_type}/{year}/{month}/{uuid}.webp
# ---------------------------------------------------------------------------

def test_object_key_pattern():
    from app.storage import upload_document_image

    captured: dict = {}

    def fake_put_object(**kwargs):
        captured.update(kwargs)
        return {}

    mock_client = MagicMock()
    mock_client.put_object.side_effect = fake_put_object

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("boto3.client", return_value=mock_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "thai-id")

    key = captured["Key"]
    pattern = r"^thai-id/\d{4}/\d{2}/[0-9a-f\-]{36}\.webp$"
    assert re.match(pattern, key), f"Key '{key}' does not match expected pattern"
    assert result is not None and key in result


# ---------------------------------------------------------------------------
# Test: bytes sent to MinIO are WebP format
# ---------------------------------------------------------------------------

def test_uploaded_bytes_are_webp():
    from app.storage import upload_document_image

    captured: dict = {}

    def fake_put_object(**kwargs):
        captured.update(kwargs)
        return {}

    mock_client = MagicMock()
    mock_client.put_object.side_effect = fake_put_object

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("boto3.client", return_value=mock_client),
    ):
        upload_document_image(_make_jpeg_bytes(), "passport")

    body = captured["Body"]
    assert _is_webp(body), "Uploaded bytes are not WebP format"
    assert captured["ContentType"] == "image/webp"


# ---------------------------------------------------------------------------
# Test: returns None when boto3 raises an exception (connection refused etc.)
# ---------------------------------------------------------------------------

def test_returns_none_on_boto3_exception():
    from app.storage import upload_document_image

    mock_client = MagicMock()
    mock_client.put_object.side_effect = Exception("connection refused")

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("boto3.client", return_value=mock_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert result is None


# ---------------------------------------------------------------------------
# Test: returns None when WebP conversion fails (corrupt image bytes)
# ---------------------------------------------------------------------------

def test_returns_none_on_conversion_failure():
    from app.storage import upload_document_image

    corrupt_bytes = b"not-an-image"

    with patch("app.storage.get_settings", return_value=_settings()):
        result = upload_document_image(corrupt_bytes, "passport")

    assert result is None


# ---------------------------------------------------------------------------
# Test: no temporary files are written to disk
# ---------------------------------------------------------------------------

def test_no_disk_writes():
    import builtins

    from app.storage import upload_document_image

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        # Allow reads of existing files (e.g. PIL internal reads via path)
        # but reject any write-mode open
        mode = args[0] if args else kwargs.get("mode", "r")
        if isinstance(mode, str) and ("w" in mode or "x" in mode or "a" in mode):
            raise AssertionError(f"Unexpected file write: {file!r} mode={mode!r}")
        return original_open(file, *args, **kwargs)

    mock_client = MagicMock()
    mock_client.put_object.return_value = {}

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("boto3.client", return_value=mock_client),
        patch("builtins.open", side_effect=guarded_open),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert result is not None


# ---------------------------------------------------------------------------
# Test: SSL flag switches scheme to https
# ---------------------------------------------------------------------------

def test_uses_https_when_ssl_enabled():
    from app.storage import upload_document_image

    mock_client = MagicMock()
    mock_client.put_object.return_value = {}

    with (
        patch("app.storage.get_settings", return_value=_settings(minio_use_ssl=True)),
        patch("boto3.client", return_value=mock_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert result is not None
    assert result.startswith("https://")
