"""
Authentication module – JWT-based user auth with bcrypt password hashing.

In production, swap the in-memory store for PostgreSQL / DynamoDB.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# ---------------------------------------------------------------------------
# Simple password hashing (production: use bcrypt via passlib)
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, bytes]:
    if salt is None:
        salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + hashed).decode(), salt


def _verify_password(password: str, stored_hash: str) -> bool:
    raw = base64.b64decode(stored_hash)
    salt, expected = raw[:16], raw[16:]
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return hmac.compare_digest(hashed, expected)


# ---------------------------------------------------------------------------
# JWT helpers (HS256, no external dependency)
# ---------------------------------------------------------------------------

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-use-a-real-secret-key")
_ALGORITHM = "HS256"
_EXPIRY_SECONDS = int(os.getenv("JWT_EXPIRY_SECONDS", "86400"))  # 24 h


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def _create_jwt(payload: dict) -> str:
    header = _b64url(json.dumps({"alg": _ALGORITHM, "typ": "JWT"}).encode())
    payload_b64 = _b64url(json.dumps(payload).encode())
    signing_input = f"{header}.{payload_b64}"
    sig = hmac.new(_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def _decode_jwt(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(
            _SECRET.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        actual_sig = _b64url_decode(parts[2])
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64url_decode(parts[1]))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Auth handler
# ---------------------------------------------------------------------------

class AuthHandler:
    """Manages user registration, login, and token verification."""

    def __init__(self):
        # In-memory user store — swap for a database in production
        self._users: dict[str, str] = {}  # username -> password_hash

    def register(self, username: str, password: str) -> str | None:
        if username in self._users:
            return None
        hashed, _ = _hash_password(password)
        self._users[username] = hashed
        return self._issue_token(username)

    def login(self, username: str, password: str) -> str | None:
        stored = self._users.get(username)
        if stored is None or not _verify_password(password, stored):
            return None
        return self._issue_token(username)

    def decode_token(self, token: str) -> dict | None:
        return _decode_jwt(token)

    # -- internal ----------------------------------------------------------

    def _issue_token(self, username: str) -> str:
        payload = {
            "sub": username,
            "iat": int(time.time()),
            "exp": int(time.time()) + _EXPIRY_SECONDS,
        }
        return _create_jwt(payload)
