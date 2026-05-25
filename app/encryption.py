"""AES-256-GCM encryption primitives for storing ID document images."""

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

MAGIC = b"ORC\x01"
_IV_LEN = 12
_TAG_LEN = 16
_MIN_BLOB_LEN = len(MAGIC) + _IV_LEN + _TAG_LEN


def _load_key() -> bytes:
    from app.config import get_settings
    hex_key = get_settings().encryption_key or ""
    if len(hex_key) != 64:
        raise ValueError(
            f"ENCRYPTION_KEY must be exactly 64 hex characters (32 bytes); got {len(hex_key)}"
        )
    try:
        return bytes.fromhex(hex_key)
    except ValueError:
        raise ValueError("ENCRYPTION_KEY contains non-hex characters")


_key: bytes = _load_key()


def encrypt(plaintext: bytes) -> bytes:
    """Return MAGIC + IV (12 B) + ciphertext + GCM tag (16 B)."""
    iv = get_random_bytes(_IV_LEN)
    cipher = AES.new(_key, AES.MODE_GCM, nonce=iv, mac_len=_TAG_LEN)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return MAGIC + iv + ciphertext + tag


def decrypt(blob: bytes) -> bytes:
    """Validate magic header + GCM tag then return plaintext."""
    if len(blob) < _MIN_BLOB_LEN:
        raise ValueError("Blob is too short to be a valid encrypted payload")
    if blob[:4] != MAGIC:
        raise ValueError("Invalid magic header; expected ORC\\x01")
    iv = blob[4 : 4 + _IV_LEN]
    ciphertext = blob[4 + _IV_LEN : -_TAG_LEN]
    tag = blob[-_TAG_LEN:]
    cipher = AES.new(_key, AES.MODE_GCM, nonce=iv, mac_len=_TAG_LEN)
    return cipher.decrypt_and_verify(ciphertext, tag)
