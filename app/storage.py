import io
import logging
import uuid
from datetime import date as _date
from datetime import datetime, timezone

from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


def upload_document_image(image_bytes: bytes, document_type: str) -> str | None:
    """Upload document image to MinIO/S3. Returns object_key or None if disabled/failed."""
    settings = get_settings()

    if not settings.minio_endpoint:
        return None

    # Convert to WebP in memory — no disk writes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        webp_buf = io.BytesIO()
        img.save(webp_buf, format="WEBP")
        webp_bytes = webp_buf.getvalue()
    except Exception:
        logger.warning("storage: WebP conversion failed", exc_info=True)
        return None

    # Watermark then encrypt — exceptions propagate; never store plaintext
    from app.encryption import encrypt
    from app.watermark import apply_watermark

    watermarked = apply_watermark(webp_bytes, settings.hotel_name, _date.today().isoformat())
    payload = encrypt(watermarked)

    # Object key: {document_type}/{year}/{month}/{uuid}.webp
    now = datetime.now(tz=timezone.utc)
    object_key = f"{document_type}/{now.year}/{now.month:02d}/{uuid.uuid4()}.webp"

    scheme = "https" if settings.minio_use_ssl else "http"
    endpoint_url = f"{scheme}://{settings.minio_endpoint}"

    try:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
        )
        client.put_object(
            Bucket=settings.minio_bucket,
            Key=object_key,
            Body=payload,
            ContentType="image/webp",
        )
    except Exception:
        logger.warning("storage: MinIO upload failed", exc_info=True)
        return None

    return object_key
