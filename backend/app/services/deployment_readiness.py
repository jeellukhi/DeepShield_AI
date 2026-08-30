from __future__ import annotations

import os

from pymongo import MongoClient

from app.db.mongo import count_auth_users
from app.security import ensure_auth_users_bootstrapped, is_admin_protection_enabled, is_jwt_secret_strong
from app.security import get_login_guard_config
from app.services.model_readiness import get_model_readiness


def _get_env_mode() -> str:
    mode = os.getenv("APP_ENV", "development").strip().lower()
    if mode not in {"development", "production"}:
        return "development"
    return mode


def _get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    parts = [part.strip() for part in raw.split(",")]
    return [part for part in parts if part]


def _check_mongo() -> tuple[bool, str]:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)
    try:
        client.admin.command("ping")
        return True, "MongoDB connection ok."
    except Exception as exc:
        return False, f"MongoDB ping failed: {str(exc)}"
    finally:
        client.close()


def get_deployment_readiness() -> dict:
    mode = _get_env_mode()
    cors_origins = _get_cors_origins()
    admin_enabled = is_admin_protection_enabled()
    model_readiness = get_model_readiness()
    checks: list[dict] = []
    next_actions: list[str] = []

    checks.append(
        {
            "name": "environment_mode",
            "status": "pass" if mode in {"development", "production"} else "warn",
            "detail": f"APP_ENV={mode}",
        }
    )

    if "*" in cors_origins:
        checks.append(
            {
                "name": "cors_policy",
                "status": "fail",
                "detail": "CORS allows * (unsafe for production).",
            }
        )
        next_actions.append("Set CORS_ORIGINS to explicit frontend URL(s), not '*'.")
    elif mode == "production" and any("localhost" in origin for origin in cors_origins):
        checks.append(
            {
                "name": "cors_policy",
                "status": "warn",
                "detail": "Production mode still includes localhost origin.",
            }
        )
        next_actions.append("Remove localhost from CORS_ORIGINS in production.")
    else:
        checks.append(
            {
                "name": "cors_policy",
                "status": "pass",
                "detail": f"CORS origins configured: {len(cors_origins)}",
            }
        )

    if mode == "production" and not admin_enabled:
        checks.append(
            {
                "name": "admin_protection",
                "status": "fail",
                "detail": "Admin protection not enabled.",
            }
        )
        next_actions.append("Enable JWT auth and protect admin endpoints.")
    elif mode == "development" and not admin_enabled:
        checks.append(
            {
                "name": "admin_protection",
                "status": "warn",
                "detail": "Admin protection disabled (acceptable in development).",
            }
        )
    else:
        checks.append(
            {
                "name": "admin_protection",
                "status": "pass",
                "detail": "Admin protection enabled.",
            }
        )

    jwt_secret_strong = is_jwt_secret_strong()
    if mode == "production" and not jwt_secret_strong:
        checks.append(
            {
                "name": "jwt_secret_strength",
                "status": "fail",
                "detail": "AUTH_JWT_SECRET is weak/default in production.",
            }
        )
        next_actions.append("Set a strong AUTH_JWT_SECRET in backend/.env.")
    elif jwt_secret_strong:
        checks.append(
            {
                "name": "jwt_secret_strength",
                "status": "pass",
                "detail": "JWT secret strength check passed.",
            }
        )
    else:
        checks.append(
            {
                "name": "jwt_secret_strength",
                "status": "warn",
                "detail": "Using default dev JWT secret.",
            }
        )

    mongo_ok, mongo_detail = _check_mongo()
    checks.append(
        {
            "name": "mongodb_connection",
            "status": "pass" if mongo_ok else "fail",
            "detail": mongo_detail,
        }
    )
    if not mongo_ok:
        next_actions.append("Verify MongoDB service is running and MONGODB_URI is correct.")

    ensure_auth_users_bootstrapped()
    auth_users, auth_users_error = count_auth_users()
    if auth_users_error:
        checks.append(
            {
                "name": "auth_user_store",
                "status": "fail",
                "detail": f"Auth user check failed: {auth_users_error}",
            }
        )
        next_actions.append("Check MongoDB auth_users collection access.")
    else:
        admin_count = int(auth_users.get("admin_count", 0))
        user_count = int(auth_users.get("user_count", 0))
        if mode == "production" and (admin_count < 1 or user_count < 1):
            checks.append(
                {
                    "name": "auth_user_store",
                    "status": "fail",
                    "detail": f"Missing required users. admin_count={admin_count}, user_count={user_count}",
                }
            )
            next_actions.append("Ensure at least one admin and one user account exist in auth_users.")
        else:
            checks.append(
                {
                    "name": "auth_user_store",
                    "status": "pass",
                    "detail": f"Auth users ok. admin_count={admin_count}, user_count={user_count}",
                }
            )

    login_guard = get_login_guard_config()
    checks.append(
        {
            "name": "login_guard_policy",
            "status": "pass",
            "detail": (
                "max_attempts="
                f"{login_guard['max_attempts']}, window_seconds={login_guard['window_seconds']}, "
                f"lock_seconds={login_guard['lock_seconds']}"
            ),
        }
    )

    image_exists = bool(model_readiness["model_files"]["image_model_exists"])
    profile_exists = bool(model_readiness["model_files"]["profile_model_exists"])
    checks.append(
        {
            "name": "trained_models",
            "status": "pass" if image_exists and profile_exists else "warn",
            "detail": f"image_model={image_exists}, profile_model={profile_exists}",
        }
    )
    if not image_exists or not profile_exists:
        next_actions.append("Train missing models before production rollout.")

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    warn_count = sum(1 for check in checks if check["status"] == "warn")
    pass_count = sum(1 for check in checks if check["status"] == "pass")

    # Warnings are advisory; only failures block production readiness.
    ready_for_production = mode == "production" and fail_count == 0
    if mode != "production":
        next_actions.append("Set APP_ENV=production and configure CORS_ORIGINS for deployment.")

    return {
        "environment": mode,
        "cors_origins": cors_origins,
        "admin_protection_enabled": admin_enabled,
        "summary": {
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "ready_for_production": ready_for_production,
        },
        "checks": checks,
        "next_actions": next_actions,
    }
