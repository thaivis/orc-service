from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    THAI_ID = "thai_id"
    PASSPORT = "passport"


class Sex(str, Enum):
    M = "M"
    F = "F"


class ConfidenceScores(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    first_name: float = Field(ge=0.0, le=1.0, default=0.0)
    last_name: float = Field(ge=0.0, le=1.0, default=0.0)
    document_number: float = Field(ge=0.0, le=1.0, default=0.0)
    date_of_birth: float = Field(ge=0.0, le=1.0, default=0.0)
    sex: float = Field(ge=0.0, le=1.0, default=0.0)
    country: float = Field(ge=0.0, le=1.0, default=0.0)


class ScanResponse(BaseModel):
    type: DocumentType
    first_name: str | None = None
    last_name: str | None = None
    document_number: str | None = None
    date_of_birth: date | None = None
    sex: Sex | None = None
    country: str | None = None
    document_valid: bool = False
    image_url: str | None = None
    confidence: ConfidenceScores
    warnings: list[str] = Field(default_factory=list)


class ErrorCode(str, Enum):
    UNAUTHORIZED = "unauthorized"
    INVALID_TYPE = "invalid_type"
    UNSUPPORTED_FORMAT = "unsupported_format"
    FILE_TOO_LARGE = "file_too_large"
    IMAGE_INVALID = "image_invalid"
    NO_DOCUMENT_DETECTED = "no_document_detected"
    TYPE_MISMATCH = "type_mismatch"
    INTERNAL_ERROR = "internal_error"


class ErrorResponse(BaseModel):
    error: ErrorCode
    message: str
    detected_type: DocumentType | None = None
