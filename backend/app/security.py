from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from threading import Lock

import jwt
from fastapi import Header, HTTPException
from jwt import ExpiredSignatureError, InvalidTokenError

from app.db.mongo import fetch_auth_user, upsert_auth_user

DEFAULT_JWT_SECRET = "deepshield-dev-secret-change-in-production"
JWT_ALGORITHM = "HS256"
DEFAULT_PBKDF2_ITERATIONS = 260000
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")

_bootstrap_lock = Lock()
_bootstrap_done = False

_login_guard_lock = Lock()
_login_guard_state: dict[str, dict] = {}


def _get_jwt_secret() -> str:
    return os.getenv("AUTH_JWT_SECRET", DEFAULT_JWT_SECRET).strip()


def _get_access_minutes() -> int:
    raw = os.getenv("AUTH_ACCESS_TOKEN_MINUTES", "720").strip()
    try:
        value = int(raw)
    except ValueError:
        return 720
    return max(15, min(value, 24 * 60))


def _get_refresh_days() -> int:
    raw = os.getenv("AUTH_REFRESH_TOKEN_DAYS", "14").strip()
    try:
        value = int(raw)
    except ValueError:
        return 14
    return max(1, min(value, 60))


def _get_login_guard_config() -> dict:
    def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(minimum, min(value, maximum))

    return {
        "max_attempts": _int_env("AUTH_LOGIN_MAX_ATTEMPTS", 5, 3, 15),
        "window_seconds": _int_env("AUTH_LOGIN_WINDOW_SECONDS", 300, 60, 3600),
        "lock_seconds": _int_env("AUTH_LOGIN_LOCK_SECONDS", 900, 60, 24 * 3600),
    }


def get_login_guard_config() -> dict:
    return _get_login_guard_config()


def _credential(role: str) -> tuple[str, str]:
    if role == "admin":
        return (
            os.getenv("AUTH_ADMIN_USERNAME", "admin").strip(),
            os.getenv("AUTH_ADMIN_PASSWORD", "Admin@123").strip(),
        )
    return (
        os.getenv("AUTH_USER_USERNAME", "user").strip(),
        os.getenv("AUTH_USER_PASSWORD", "User@123").strip(),
    )


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def is_jwt_secret_strong() -> bool:
    secret = _get_jwt_secret()
    return bool(secret) and secret != DEFAULT_JWT_SECRET and len(secret) >= 24


def is_admin_protection_enabled() -> bool:
    return True


def _hash_password(password: str, iterations: int = DEFAULT_PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        iterations,
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iter_raw, salt, digest = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iter_raw)
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            iterations,
        ).hex()
        return secrets.compare_digest(expected, digest)
    except Exception:
        return False


def _ensure_bootstrap_users() -> None:
    global _bootstrap_done
    if _bootstrap_done:
        return
    with _bootstrap_lock:
        if _bootstrap_done:
            return

        for role in ("admin", "user"):
            username, password = _credential(role)
            if not username or not password:
                continue

            password_hash = _hash_password(password)
            existing, _ = fetch_auth_user(username)
            if not existing:
                upsert_auth_user(
                    username=username,
                    role=role,
                    password_hash=password_hash,
                    seeded_from_env=True,
                )
                continue

            # Keep bootstrap users aligned with the configured env credentials
            # while they are still marked as env-seeded accounts.
            if bool(existing.get("seeded_from_env", False)):
                current_hash = str(existing.get("password_hash", ""))
                current_role = str(existing.get("role", "")).strip()
                if current_role != role or not _verify_password(password, current_hash):
                    upsert_auth_user(
                        username=username,
                        role=role,
                        password_hash=password_hash,
                        seeded_from_env=True,
                    )
        _bootstrap_done = True


def ensure_auth_users_bootstrapped() -> None:
    _ensure_bootstrap_users()


def _guard_keys(username: str, source_ip: str) -> list[str]:
    uname = username.strip().lower() or "_"
    ip = source_ip.strip() or "unknown"
    return [f"user:{uname}|ip:{ip}", f"ip:{ip}"]


def check_login_allowed(username: str, source_ip: str) -> tuple[bool, int]:
    config = _get_login_guard_config()
    now = int(time.time())
    keys = _guard_keys(username, source_ip)

    with _login_guard_lock:
        max_remaining = 0
        for key in keys:
            entry = _login_guard_state.get(key, {"failures": [], "lock_until": 0})
            lock_until = int(entry.get("lock_until", 0))
            if lock_until > now:
                remaining = lock_until - now
                if remaining > max_remaining:
                    max_remaining = remaining
                continue

            window = config["window_seconds"]
            failures = [ts for ts in entry.get("failures", []) if now - int(ts) <= window]
            entry["failures"] = failures
            entry["lock_until"] = 0
            _login_guard_state[key] = entry

        if max_remaining > 0:
            return False, max_remaining
    return True, 0


def register_failed_login(username: str, source_ip: str) -> None:
    config = _get_login_guard_config()
    now = int(time.time())
    keys = _guard_keys(username, source_ip)
    with _login_guard_lock:
        for key in keys:
            entry = _login_guard_state.get(key, {"failures": [], "lock_until": 0})
            window = config["window_seconds"]
            failures = [ts for ts in entry.get("failures", []) if now - int(ts) <= window]
            failures.append(now)
            entry["failures"] = failures
            if len(failures) >= config["max_attempts"]:
                entry["lock_until"] = now + config["lock_seconds"]
                entry["failures"] = []
            _login_guard_state[key] = entry


def register_successful_login(username: str, source_ip: str) -> None:
    keys = _guard_keys(username, source_ip)
    with _login_guard_lock:
        for key in keys:
            if key in _login_guard_state:
                del _login_guard_state[key]


def authenticate_user(username: str, password: str) -> dict | None:
    _ensure_bootstrap_users()
    normalized_username = _normalize_username(username)
    user, error = fetch_auth_user(normalized_username)
    if not user and normalized_username != username.strip():
        user, error = fetch_auth_user(username.strip())
    if error or not user:
        return None
    if not _verify_password(password, str(user.get("password_hash", ""))):
        return None
    role = str(user.get("role", "")).strip()
    if role not in {"admin", "user"}:
        return None
    return {"username": str(user.get("username", "")).strip(), "role": role}


def register_standard_user(username: str, password: str) -> tuple[dict | None, str | None]:
    _ensure_bootstrap_users()
    normalized_username = _normalize_username(username)
    if not USERNAME_PATTERN.match(normalized_username):
        return None, "Username must be 3-32 chars. Allowed: letters, numbers, underscore, dot, hyphen."
    if normalized_username.startswith((".", "-")) or normalized_username.endswith((".", "-")):
        return None, "Username cannot start or end with dot/hyphen."
    admin_username = _normalize_username(_credential("admin")[0])
    default_user_username = _normalize_username(_credential("user")[0])
    if normalized_username in {admin_username, default_user_username}:
        return None, "This username is reserved. Choose another one."
    if len(password or "") < 8:
        return None, "Password must be at least 8 characters."

    existing, error = fetch_auth_user(normalized_username)
    if error:
        return None, error
    if existing:
        return None, "Username already exists."

    password_hash = _hash_password(password)
    ok, save_error = upsert_auth_user(
        username=normalized_username,
        role="user",
        password_hash=password_hash,
        seeded_from_env=False,
    )
    if not ok:
        return None, save_error or "Unable to create account."
    return {"username": normalized_username, "role": "user"}, None


def change_authenticated_user_password(
    *,
    username: str,
    current_password: str,
    new_password: str,
) -> tuple[bool, str | None]:
    _ensure_bootstrap_users()
    normalized_username = _normalize_username(username)
    if not normalized_username:
        return False, "Invalid user."
    if len(new_password or "") < 8:
        return False, "New password must be at least 8 characters."
    if current_password == new_password:
        return False, "New password must be different from current password."

    user, error = fetch_auth_user(normalized_username)
    if error:
        return False, error
    if not user:
        return False, "User account not found."
    if not _verify_password(current_password, str(user.get("password_hash", ""))):
        return False, "Current password is incorrect."

    role = str(user.get("role", "user")).strip()
    if role not in {"admin", "user"}:
        return False, "Invalid user role."
    new_hash = _hash_password(new_password)
    ok, save_error = upsert_auth_user(
        username=normalized_username,
        role=role,
        password_hash=new_hash,
        seeded_from_env=False,
    )
    if not ok:
        return False, save_error or "Unable to update password."
    return True, None


def get_access_token_ttl_seconds() -> int:
    return _get_access_minutes() * 60


def _create_token(username: str, role: str, token_type: str, ttl_seconds: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_access_token(username: str, role: str) -> str:
    return _create_token(username, role, "access", get_access_token_ttl_seconds())


def create_refresh_token(username: str, role: str) -> str:
    return _create_token(username, role, "refresh", _get_refresh_days() * 24 * 60 * 60)


def _parse_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    prefix = "bearer "
    if not authorization.lower().startswith(prefix):
        raise HTTPException(status_code=401, detail="Authorization must use Bearer token.")
    token = authorization[len(prefix) :].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    return token


def _decode_token(token: str, expected_type: str | None = None) -> dict:
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired.") from exc
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token.") from exc
    token_type = str(payload.get("typ", "access")).strip()
    if expected_type and token_type != expected_type:
        raise HTTPException(status_code=401, detail=f"Invalid token type: expected {expected_type}.")
    return payload


def require_authenticated_user(authorization: str | None = Header(default=None)) -> dict:
    token = _parse_bearer_token(authorization)
    payload = _decode_token(token, expected_type="access")
    username = str(payload.get("sub", "")).strip()
    role = str(payload.get("role", "")).strip()
    if not username or role not in {"admin", "user"}:
        raise HTTPException(status_code=401, detail="Invalid token payload.")
    return {"username": username, "role": role}


def require_admin_user(authorization: str | None = Header(default=None)) -> dict:
    user = require_authenticated_user(authorization)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


def consume_refresh_token(refresh_token: str) -> dict:
    token = str(refresh_token or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing refresh token.")
    payload = _decode_token(token, expected_type="refresh")
    username = str(payload.get("sub", "")).strip()
    role = str(payload.get("role", "")).strip()
    if not username or role not in {"admin", "user"}:
        raise HTTPException(status_code=401, detail="Invalid refresh token payload.")
    return {"username": username, "role": role}
