"""Unit tests for app.encryption — AES-256-GCM module.

No MinIO, PaddleOCR, or Tesseract dependency is required to run these tests.
"""

import pytest

import app.encryption as enc
from app.encryption import MAGIC, _load_key, decrypt, encrypt

VALID_KEY = "a" * 64  # 32 bytes of 0xAA


class _MockSettings:
    """Minimal settings stub for key-loading tests."""
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key


# ---------------------------------------------------------------------------
# encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_returns_original_bytes():
    plaintext = b"Hello, hotel guest!"
    assert decrypt(encrypt(plaintext)) == plaintext


def test_roundtrip_empty_bytes():
    assert decrypt(encrypt(b"")) == b""


def test_roundtrip_large_payload():
    plaintext = b"x" * 100_000
    assert decrypt(encrypt(plaintext)) == plaintext


# ---------------------------------------------------------------------------
# Blob structure
# ---------------------------------------------------------------------------


def test_encrypted_output_starts_with_magic():
    blob = encrypt(b"test")
    assert blob[:4] == MAGIC


def test_encrypted_output_length():
    plaintext = b"abc"
    blob = encrypt(plaintext)
    # 4 (magic) + 12 (IV) + len(plaintext) + 16 (GCM tag)
    assert len(blob) == 4 + 12 + len(plaintext) + 16


def test_two_calls_produce_different_ciphertexts():
    plaintext = b"same input"
    assert encrypt(plaintext) != encrypt(plaintext)


# ---------------------------------------------------------------------------
# Tamper / wrong-key detection
# ---------------------------------------------------------------------------


def test_decrypt_wrong_key_raises(monkeypatch):
    blob = encrypt(b"secret")
    wrong_key = bytes(32)  # 32 zero bytes — different from 0xAA key in env
    monkeypatch.setattr(enc, "_key", wrong_key)
    with pytest.raises(Exception):
        decrypt(blob)


def test_decrypt_tampered_ciphertext_raises():
    blob = encrypt(b"important data")
    tampered = bytearray(blob)
    # Flip a byte in the ciphertext body (past magic + IV, before tag)
    tampered[4 + 12] ^= 0xFF
    with pytest.raises(Exception):
        decrypt(bytes(tampered))


def test_decrypt_tampered_tag_raises():
    blob = encrypt(b"data")
    tampered = bytearray(blob)
    tampered[-1] ^= 0x01  # flip last byte of GCM tag
    with pytest.raises(Exception):
        decrypt(bytes(tampered))


def test_decrypt_missing_magic_raises():
    blob = encrypt(b"data")
    # Replace magic with garbage
    bad_blob = b"XXXX" + blob[4:]
    with pytest.raises(ValueError, match="magic"):
        decrypt(bad_blob)


def test_decrypt_too_short_raises():
    with pytest.raises(ValueError):
        decrypt(b"ORC\x01" + b"\x00" * 10)  # 14 bytes, minimum is 32


# ---------------------------------------------------------------------------
# Key loading validation
# ---------------------------------------------------------------------------


def test_load_key_missing_raises(monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: _MockSettings(encryption_key=None))
    with pytest.raises(ValueError, match="64 hex"):
        _load_key()


def test_load_key_too_short_raises(monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: _MockSettings(encryption_key="abc123"))
    with pytest.raises(ValueError, match="64 hex"):
        _load_key()


def test_load_key_non_hex_raises(monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: _MockSettings(encryption_key="z" * 64))
    with pytest.raises(ValueError, match="non-hex"):
        _load_key()


def test_load_key_valid_returns_32_bytes(monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: _MockSettings(encryption_key=VALID_KEY))
    key = _load_key()
    assert isinstance(key, bytes)
    assert len(key) == 32
