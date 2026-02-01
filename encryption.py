"""
Encryption module for Dream Ferret.
Encrypts user-entered content (dream text, life context) at rest.
Uses Fernet symmetric encryption with a server-side key.
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Encryption key from environment, or generate a default for development
# In production, set ENCRYPTION_KEY environment variable on Render
_ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")

# If no key set, derive one from a passphrase (for development only)
if not _ENCRYPTION_KEY:
    # Use a fixed salt for consistency (in production, use env var instead)
    _DEFAULT_PASSPHRASE = "DreamFerret-Dev-Key-2026"
    _SALT = b"dreamferret_salt_v1"

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=100000,
    )
    _ENCRYPTION_KEY = base64.urlsafe_b64encode(
        kdf.derive(_DEFAULT_PASSPHRASE.encode())
    ).decode()

# Create Fernet instance
_fernet = Fernet(_ENCRYPTION_KEY.encode() if isinstance(_ENCRYPTION_KEY, str) else _ENCRYPTION_KEY)

# Prefix to identify encrypted content
ENCRYPTED_PREFIX = "ENC::"


def encrypt(plaintext: str) -> str:
    """
    Encrypt a string. Returns encrypted string with prefix.
    Returns original string if empty/None.
    """
    if not plaintext:
        return plaintext

    encrypted_bytes = _fernet.encrypt(plaintext.encode('utf-8'))
    return ENCRYPTED_PREFIX + encrypted_bytes.decode('utf-8')


def decrypt(ciphertext: str) -> str:
    """
    Decrypt a string. Returns decrypted string.
    If string is not encrypted (no prefix), returns as-is (for backward compatibility).
    Returns original string if empty/None.
    """
    if not ciphertext:
        return ciphertext

    # Check if this is encrypted content
    if not ciphertext.startswith(ENCRYPTED_PREFIX):
        # Not encrypted - return as-is (legacy data)
        return ciphertext

    # Remove prefix and decrypt
    encrypted_data = ciphertext[len(ENCRYPTED_PREFIX):]
    try:
        decrypted_bytes = _fernet.decrypt(encrypted_data.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')
    except Exception:
        # If decryption fails, return original (shouldn't happen in normal use)
        return ciphertext


def is_encrypted(text: str) -> bool:
    """Check if a string is encrypted."""
    return text is not None and text.startswith(ENCRYPTED_PREFIX)
