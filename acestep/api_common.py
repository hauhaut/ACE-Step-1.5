"""Shared API utilities for ACE-Step API servers.

Provides common authentication and response formatting used by:
- acestep/api_server.py
- acestep/gradio_ui/api_routes.py
- openrouter/openrouter_api_server.py
"""
import hmac
import os
import tempfile
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Header

# Module-level API key storage
_api_key: Optional[str] = None


def set_api_key(key: Optional[str]):
    """Set the API key for authentication."""
    global _api_key
    _api_key = key


def get_api_key() -> Optional[str]:
    """Get the current API key (for testing/introspection)."""
    return _api_key


def wrap_response(
    data: Any, code: int = 200, error: Optional[str] = None
) -> Dict[str, Any]:
    """Wrap response data in standard format compatible with CustomAceStep."""
    return {
        "data": data,
        "code": code,
        "error": error,
        "timestamp": int(time.time() * 1000),
        "extra": None,
    }


def validate_audio_path(path: str) -> str:
    """Validate audio file path to prevent path traversal attacks.

    Only allows access to files in the system temp directory or ACESTEP_TMPDIR.
    Raises HTTPException if path is outside allowed directory.
    """
    cache_root = os.path.expanduser("~/.cache/acestep")
    acestep_tmp = os.getenv("ACESTEP_TMPDIR") or os.path.join(cache_root, "tmp")
    allowed_dir = acestep_tmp.strip()
    system_temp = tempfile.gettempdir()

    abs_path = os.path.abspath(os.path.normpath(path))
    abs_allowed = os.path.abspath(os.path.normpath(allowed_dir))
    abs_system_temp = os.path.abspath(os.path.normpath(system_temp))

    is_in_allowed = (
        abs_path.startswith(abs_allowed + os.sep) or abs_path == abs_allowed
    )
    is_in_system_temp = (
        abs_path.startswith(abs_system_temp + os.sep) or abs_path == abs_system_temp
    )

    if not (is_in_allowed or is_in_system_temp):
        raise HTTPException(
            status_code=403, detail="Access denied: path outside allowed directory"
        )

    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found")

    return abs_path


def verify_token_from_request(
    body: dict, authorization: Optional[str] = None
) -> Optional[str]:
    """Verify API key from request body (ai_token) or Authorization header.

    Returns the token if valid, None if no auth required.
    Uses timing-safe comparison to prevent timing attacks.
    """
    if _api_key is None:
        return None

    ai_token = body.get("ai_token") if body else None
    if ai_token:
        if hmac.compare_digest(ai_token, _api_key):
            return ai_token
        raise HTTPException(status_code=401, detail="Invalid ai_token")

    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        if hmac.compare_digest(token, _api_key):
            return token
        raise HTTPException(status_code=401, detail="Invalid API key")

    raise HTTPException(
        status_code=401, detail="Missing ai_token or Authorization header"
    )


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from Authorization header (for non-body endpoints).

    Uses timing-safe comparison to prevent timing attacks.
    """
    if _api_key is None:
        return

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    if not hmac.compare_digest(token, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
