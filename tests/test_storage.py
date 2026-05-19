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
# Test: successful upload returns an object_key string (not a raw URL)
# ---------------------------------------------------------------------------

def test_returns_object_key_on_success():
    from app.storage import upload_document_image

    mock_client = MagicMock()
    mock_client.put_object.return_value = {}

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", return_value=b"encrypted"),
        patch("boto3.client", return_value=mock_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert isinstance(result, str)
    assert re.match(r"^passport/\d{4}/\d{2}/[0-9a-f\-]{36}\.webp$", result)
    assert not result.startswith("http"), "result must be an object_key, not a URL"


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
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", return_value=b"encrypted"),
        patch("boto3.client", return_value=mock_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "thai-id")

    key = captured["Key"]
    pattern = r"^thai-id/\d{4}/\d{2}/[0-9a-f\-]{36}\.webp$"
    assert re.match(pattern, key), f"Key '{key}' does not match expected pattern"
    assert result == key, "upload_document_image must return the object_key directly"


# ---------------------------------------------------------------------------
# Test: bytes uploaded to MinIO are encrypted (not plain WebP)
# ---------------------------------------------------------------------------

def test_uploaded_bytes_are_encrypted():
    from app.storage import upload_document_image

    captured: dict = {}

    def fake_put_object(**kwargs):
        captured.update(kwargs)
        return {}

    mock_client = MagicMock()
    mock_client.put_object.side_effect = fake_put_object

    encrypted_sentinel = b"ENCRYPTED_SENTINEL"

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", return_value=encrypted_sentinel),
        patch("boto3.client", return_value=mock_client),
    ):
        upload_document_image(_make_jpeg_bytes(), "passport")

    assert captured["Body"] == encrypted_sentinel, "MinIO must receive encrypted bytes"
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
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", return_value=b"encrypted"),
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
        mode = args[0] if args else kwargs.get("mode", "r")
        if isinstance(mode, str) and ("w" in mode or "x" in mode or "a" in mode):
            raise AssertionError(f"Unexpected file write: {file!r} mode={mode!r}")
        return original_open(file, *args, **kwargs)

    mock_client = MagicMock()
    mock_client.put_object.return_value = {}

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", return_value=b"encrypted"),
        patch("boto3.client", return_value=mock_client),
        patch("builtins.open", side_effect=guarded_open),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert result is not None


# ---------------------------------------------------------------------------
# Test: SSL flag switches boto3 to https endpoint
# ---------------------------------------------------------------------------

def test_uses_https_when_ssl_enabled():
    from app.storage import upload_document_image

    mock_client = MagicMock()
    mock_client.put_object.return_value = {}

    captured_endpoint: dict = {}

    def fake_boto3_client(service, endpoint_url=None, **kwargs):
        captured_endpoint["url"] = endpoint_url
        return mock_client

    with (
        patch("app.storage.get_settings", return_value=_settings(minio_use_ssl=True)),
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", return_value=b"encrypted"),
        patch("boto3.client", side_effect=fake_boto3_client),
    ):
        result = upload_document_image(_make_jpeg_bytes(), "passport")

    assert result is not None
    assert captured_endpoint["url"].startswith("https://")


# ---------------------------------------------------------------------------
# Test: apply_watermark is called before encrypt and put_object
# ---------------------------------------------------------------------------

def test_watermark_called_before_put_object():
    from app.storage import upload_document_image

    call_order: list[str] = []

    def fake_watermark(webp_bytes, hotel_name, iso_date):
        call_order.append("watermark")
        return b"watermarked"

    def fake_encrypt(data):
        call_order.append("encrypt")
        return b"encrypted"

    mock_client = MagicMock()

    def fake_put_object(**kwargs):
        call_order.append("put_object")
        return {}

    mock_client.put_object.side_effect = fake_put_object

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", side_effect=fake_watermark),
        patch("app.encryption.encrypt", side_effect=fake_encrypt),
        patch("boto3.client", return_value=mock_client),
    ):
        upload_document_image(_make_jpeg_bytes(), "thai-id")

    assert call_order == ["watermark", "encrypt", "put_object"]


# ---------------------------------------------------------------------------
# Test: encrypt receives watermark output and put_object receives encrypt output
# ---------------------------------------------------------------------------

def test_encrypt_receives_watermark_output_and_put_object_receives_encrypted():
    from app.storage import upload_document_image

    watermark_output = b"watermarked_image_bytes"
    encrypt_output = b"encrypted_image_bytes"
    received_by_encrypt: list = []
    received_by_put_object: list = []

    def fake_watermark(webp_bytes, hotel_name, iso_date):
        return watermark_output

    def fake_encrypt(data):
        received_by_encrypt.append(data)
        return encrypt_output

    mock_client = MagicMock()

    def fake_put_object(**kwargs):
        received_by_put_object.append(kwargs.get("Body"))
        return {}

    mock_client.put_object.side_effect = fake_put_object

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", side_effect=fake_watermark),
        patch("app.encryption.encrypt", side_effect=fake_encrypt),
        patch("boto3.client", return_value=mock_client),
    ):
        upload_document_image(_make_jpeg_bytes(), "passport")

    assert received_by_encrypt == [watermark_output]
    assert received_by_put_object == [encrypt_output]


# ---------------------------------------------------------------------------
# Test: if apply_watermark raises, put_object is never called
# ---------------------------------------------------------------------------

def test_watermark_raise_propagates():
    from app.storage import upload_document_image

    mock_client = MagicMock()

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", side_effect=ValueError("bad image")),
        patch("boto3.client", return_value=mock_client),
    ):
        with pytest.raises(ValueError, match="bad image"):
            upload_document_image(_make_jpeg_bytes(), "thai-id")

    mock_client.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Test: if encrypt raises, put_object is never called
# ---------------------------------------------------------------------------

def test_encrypt_raise_propagates():
    from app.storage import upload_document_image

    mock_client = MagicMock()

    with (
        patch("app.storage.get_settings", return_value=_settings()),
        patch("app.watermark.apply_watermark", return_value=b"watermarked"),
        patch("app.encryption.encrypt", side_effect=ValueError("key error")),
        patch("boto3.client", return_value=mock_client),
    ):
        with pytest.raises(ValueError, match="key error"):
            upload_document_image(_make_jpeg_bytes(), "thai-id")

    mock_client.put_object.assert_not_called()
