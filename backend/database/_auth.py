"""
backend/database/_auth.py — password hashing and verification.

Uses bcrypt directly (passlib is unmaintained and throws warnings with modern bcrypt).

IMPORTANT: No module calls load_settings() at import time.  All values are
passed in as parameters so that tests can inject fixtures freely.
"""

from __future__ import annotations

import bcrypt


def hash_password(password: str) -> str:
    """
    Hash *password* with bcrypt using the library's recommended cost factor.

    Returns the full bcrypt digest string (algorithm, cost, salt, hash).
    The plaintext password is never stored, logged, or returned.
    """
    # bcrypt.hashpw expects bytes
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    digest = bcrypt.hashpw(pw_bytes, salt)
    return digest.decode("utf-8")


def verify_password(plain_text: str, hashed: str) -> bool:
    """
    Verify *plain_text* against the stored bcrypt *hashed* digest.

    Returns True if the candidate matches, False otherwise.
    Raises TypeError / ValueError if the hash format is invalid.
    """
    candidate_bytes = plain_text.encode("utf-8")
    hashed_bytes = hashed.encode("utf-8")
    return bcrypt.checkpw(candidate_bytes, hashed_bytes)
