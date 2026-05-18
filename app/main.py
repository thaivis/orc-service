import logging
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.logging_config import configure_logging, request_id_var
from app.scan_error import ScanError
from app.schemas import (
    ErrorCode,
    ErrorResponse,
    ScanResponse,
)

configure_logging(get_settings().log_level)
logger = logging.getLogger("orc-service")

app = FastAPI(title="orc-service", version="0.1.0")

PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _error(code: ErrorCode, message: str, http_status: int) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content=ErrorResponse(error=code, message=message).model_dump(exclude_none=True),
    )


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith("/docs/") or path.startswith("/redoc/"):
        return await call_next(request)

    settings = get_settings()
    provided = request.headers.get("X-API-Key")
    if not provided or provided != settings.api_key:
        return _error(
            ErrorCode.UNAUTHORIZED,
            "Missing or invalid X-API-Key",
            status.HTTP_401_UNAUTHORIZED,
        )
    return await call_next(request)


# Registered LAST → outermost → wraps every other middleware so 401/500 also carry X-Request-ID.
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status_code = 500
    response: Response | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "duration_ms": duration_ms,
            },
        )
        request_id_var.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    for err in exc.errors():
        loc = err.get("loc", ())
        if "image" in loc:
            return _error(
                ErrorCode.IMAGE_INVALID,
                "Field 'image' is required",
                status.HTTP_400_BAD_REQUEST,
            )
    return _error(
        ErrorCode.IMAGE_INVALID,
        "Invalid request payload",
        status.HTTP_400_BAD_REQUEST,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _validate_and_read_image(image: UploadFile) -> bytes:
    """Validate and read image file."""
    settings = get_settings()

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": ErrorCode.UNSUPPORTED_FORMAT.value,
                "message": f"Only image/jpeg and image/png allowed, got '{image.content_type}'",
            },
        )

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    contents = await image.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": ErrorCode.FILE_TOO_LARGE.value,
                "message": f"Image exceeds {settings.max_image_size_mb}MB limit",
            },
        )

    if not (contents.startswith(JPEG_MAGIC) or contents.startswith(PNG_MAGIC)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": ErrorCode.IMAGE_INVALID.value,
                "message": "Image bytes do not match JPEG or PNG signature",
            },
        )

    return contents


@app.post(
    "/scan/thai-id",
    response_model=ScanResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def scan_thai_id_card(
    image: UploadFile = File(...),
) -> ScanResponse:
    contents = await _validate_and_read_image(image)

    from app.scanners.thai_id import scan_thai_id

    result, err = scan_thai_id(contents)
    return _handle_scan_result(result, err, "No Thai ID detected in image")


@app.post(
    "/scan/passport",
    response_model=ScanResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def scan_passport_document(
    image: UploadFile = File(...),
) -> ScanResponse:
    contents = await _validate_and_read_image(image)

    # DEBUG: save incoming image to disk
    import os
    os.makedirs("/tmp/orc-debug", exist_ok=True)
    debug_path = f"/tmp/orc-debug/passport_{uuid.uuid4().hex[:8]}.jpg"
    with open(debug_path, "wb") as f:
        f.write(contents)
    logger.info("debug_image_saved", extra={"path": debug_path, "method": "GET", "path": "/debug", "status": 200, "duration_ms": 0})

    from app.scanners.passport import scan_passport

    result, err = scan_passport(contents)
    return _handle_scan_result(result, err, "No MRZ detected in image")


def _handle_scan_result(
    result: ScanResponse | None,
    err: ScanError | None,
    not_detected_message: str,
) -> ScanResponse:
    if err is None:
        assert result is not None
        return result
    if err.code == ErrorCode.IMAGE_INVALID.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": err.code,
                "message": "Image bytes could not be decoded",
            },
        )
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "error": err.code,
            "message": not_detected_message,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail and "message" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={"error": "internal_error", "message": str(detail)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so 500 responses still flow back through middleware (X-Request-ID, access log)
    instead of being short-circuited by Starlette's outer ServerErrorMiddleware."""
    logger.exception("unhandled exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "Internal server error"},
    )
