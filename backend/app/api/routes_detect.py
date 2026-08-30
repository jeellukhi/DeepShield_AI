import re
import random
import shutil
import string
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
import csv
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.db.mongo import (
    count_auth_users,
    fetch_audit_logs,
    fetch_profile_feedback_export_rows,
    fetch_profile_reports_export_json,
    fetch_profile_feedback_stats,
    fetch_recent_reports_paginated,
    fetch_report_by_id,
    fetch_report_detail,
    fetch_report_stats,
    fetch_report_trend,
    save_audit_log,
    save_report,
    update_report_feedback,
)
from app.schemas.auth import ChangePasswordRequest, LoginRequest, RefreshRequest, SignupRequest
from app.schemas.report_feedback import ReportFeedbackRequest
from app.schemas.profile import ProfileAnalyzeRequest
from app.schemas.thresholds import ThresholdsUpdateRequest
from app.services.image_detector import analyze_image_authenticity
from app.services.dataset_manager import import_labeled_images_zip, save_labeled_image
from app.services.dataset_quality import get_dataset_quality_report
from app.services.image_trainer import train_image_model
from app.services.profile_detector import predict_profile_text_authenticity
from app.services.profile_trainer import train_profile_model
from app.services.model_evaluation import evaluate_image_model, evaluate_profile_model
from app.services.model_readiness import get_model_readiness
from app.services.model_summary import get_model_summary
from app.services.model_thresholds import get_effective_thresholds, save_threshold_overrides
from app.services.project_status import get_project_status
from app.services.recommendation import build_recommendation
from app.services.report_pdf import build_report_pdf
from app.services.trust_score import calculate_trust_score, classify_risk
from app.services.video_detector import analyze_video_authenticity
from app.services.deployment_readiness import get_deployment_readiness
from app.security import (
    authenticate_user,
    check_login_allowed,
    change_authenticated_user_password,
    consume_refresh_token,
    create_access_token,
    create_refresh_token,
    ensure_auth_users_bootstrapped,
    get_login_guard_config,
    get_access_token_ttl_seconds,
    is_admin_protection_enabled,
    is_jwt_secret_strong,
    register_failed_login,
    register_standard_user,
    register_successful_login,
    require_admin_user,
    require_authenticated_user,
)

router = APIRouter(tags=["Detection"])

PROTECTED_ROUTE_HINTS = [
    "GET /api/v1/auth/me",
    "POST /api/v1/auth/change-password",
    "GET /api/v1/system/audit-logs",
    "POST /api/v1/model/train-image",
    "POST /api/v1/model/train-profile",
    "GET /api/v1/model/evaluation",
    "GET /api/v1/model/thresholds",
    "POST /api/v1/model/thresholds",
    "GET /api/v1/model/export-feedback-csv",
    "GET /api/v1/model/export-reports-json",
    "POST /api/v1/dataset/upload-image",
    "POST /api/v1/dataset/upload-zip",
    "GET /api/v1/dataset/quality",
    "POST /api/v1/demo/generate-profiles",
    "PATCH /api/v1/reports/{report_id}/feedback",
]


def _request_ip(request: Request | None) -> str:
    if request is None:
        return "unknown"
    forwarded = str(request.headers.get("x-forwarded-for", "")).strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _write_audit_event(
    *,
    actor_username: str,
    actor_role: str,
    action: str,
    status: str,
    request: Request | None,
    target: str = "",
    detail: str = "",
) -> None:
    # Best-effort audit write: endpoint flow should continue even if audit insert fails.
    save_audit_log(
        {
            "actor_username": actor_username,
            "actor_role": actor_role,
            "action": action,
            "status": status,
            "endpoint": request.url.path if request else "",
            "method": request.method if request else "",
            "source_ip": _request_ip(request),
            "target": target,
            "detail": detail,
        }
    )


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "deepshield-backend"}


@router.post("/auth/login")
def auth_login(payload: LoginRequest, request: Request):
    username = payload.username.strip().lower()
    source_ip = _request_ip(request)
    allowed, retry_after = check_login_allowed(username, source_ip)
    if not allowed:
        _write_audit_event(
            actor_username=username,
            actor_role="",
            action="auth.login",
            status="locked",
            request=request,
            detail=f"Too many failed attempts. Retry after {retry_after}s.",
        )
        raise HTTPException(status_code=429, detail=f"Too many login attempts. Try again in {retry_after} seconds.")

    user = authenticate_user(payload.username, payload.password)
    if not user:
        register_failed_login(username, source_ip)
        _write_audit_event(
            actor_username=username,
            actor_role="",
            action="auth.login",
            status="denied",
            request=request,
            detail="Invalid credentials.",
        )
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    register_successful_login(username, source_ip)
    access_token = create_access_token(user["username"], user["role"])
    refresh_token = create_refresh_token(user["username"], user["role"])
    _write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        action="auth.login",
        status="success",
        request=request,
        detail="Login successful.",
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": get_access_token_ttl_seconds(),
        "user": user,
    }


@router.post("/auth/signup")
def auth_signup(payload: SignupRequest, request: Request):
    user, error = register_standard_user(payload.username, payload.password)
    if error or not user:
        _write_audit_event(
            actor_username=payload.username.strip().lower(),
            actor_role="",
            action="auth.signup",
            status="failed",
            request=request,
            detail=error or "Signup failed.",
        )
        raise HTTPException(status_code=400, detail=error or "Signup failed.")

    access_token = create_access_token(user["username"], user["role"])
    refresh_token = create_refresh_token(user["username"], user["role"])
    _write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        action="auth.signup",
        status="success",
        request=request,
        detail="User account created.",
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": get_access_token_ttl_seconds(),
        "user": user,
    }


@router.post("/auth/refresh")
def auth_refresh(payload: RefreshRequest, request: Request):
    try:
        user = consume_refresh_token(payload.refresh_token)
    except HTTPException as exc:
        _write_audit_event(
            actor_username="",
            actor_role="",
            action="auth.refresh",
            status="denied",
            request=request,
            detail=f"Refresh rejected: {exc.detail}",
        )
        raise
    access_token = create_access_token(user["username"], user["role"])
    refresh_token = create_refresh_token(user["username"], user["role"])
    _write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        action="auth.refresh",
        status="success",
        request=request,
        detail="Access token renewed.",
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": get_access_token_ttl_seconds(),
        "user": user,
    }


@router.get("/auth/me")
def auth_me(current_user: dict = Depends(require_authenticated_user)):
    return {"user": current_user}


@router.post("/auth/change-password")
def auth_change_password(
    payload: ChangePasswordRequest,
    request: Request,
    current_user: dict = Depends(require_authenticated_user),
):
    ok, error = change_authenticated_user_password(
        username=current_user["username"],
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    if not ok:
        _write_audit_event(
            actor_username=current_user["username"],
            actor_role=current_user["role"],
            action="auth.change_password",
            status="failed",
            request=request,
            detail=error or "Password update failed.",
        )
        raise HTTPException(status_code=400, detail=error or "Password update failed.")
    _write_audit_event(
        actor_username=current_user["username"],
        actor_role=current_user["role"],
        action="auth.change_password",
        status="success",
        request=request,
        detail="Password updated.",
    )
    return {"updated": True}


@router.get("/model/readiness")
def model_readiness(request: Request, admin_user: dict = Depends(require_admin_user)):
    result = get_model_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.readiness",
        status="success",
        request=request,
        detail="Model readiness loaded.",
    )
    return result


@router.get("/model/summary")
def model_summary(_: dict = Depends(require_admin_user)):
    return get_model_summary()


@router.get("/project/status")
def project_status(_: dict = Depends(require_admin_user)):
    return get_project_status()


@router.get("/system/deployment-readiness")
def deployment_readiness(request: Request, admin_user: dict = Depends(require_admin_user)):
    result = get_deployment_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="system.deployment_readiness",
        status="success",
        request=request,
        detail="Deployment readiness loaded.",
    )
    return result


@router.get("/security/status")
def security_status(request: Request, admin_user: dict = Depends(require_admin_user)):
    ensure_auth_users_bootstrapped()
    user_counts, user_error = count_auth_users()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="security.status",
        status="success" if not user_error else "failed",
        request=request,
        detail=user_error or "Security status loaded.",
    )
    return {
        "admin_protection_enabled": is_admin_protection_enabled(),
        "auth_mode": "jwt_bearer",
        "header_name": "Authorization: Bearer <token>",
        "jwt_secret_strong": is_jwt_secret_strong(),
        "login_guard": get_login_guard_config(),
        "auth_users": user_counts,
        "auth_users_error": user_error,
        "protected_routes": PROTECTED_ROUTE_HINTS,
    }


@router.post("/model/train-image")
def train_image_model_route(
    request: Request,
    max_per_class: int = 2000,
    admin_user: dict = Depends(require_admin_user),
):
    safe_max_per_class = max(50, min(max_per_class, 20000))
    try:
        result = train_image_model(max_per_class=safe_max_per_class)
    except ValueError as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.train_image",
            status="failed",
            request=request,
            detail=f"Validation error: {str(exc)}",
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.train_image",
            status="failed",
            request=request,
            detail=f"Runtime error: {str(exc)}",
        )
        raise HTTPException(status_code=500, detail=f"Training failed: {str(exc)}") from exc

    readiness = get_model_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.train_image",
        status="success",
        request=request,
        detail=f"Trained with max_per_class={safe_max_per_class}.",
    )
    return {
        "trained": True,
        "max_per_class": safe_max_per_class,
        **result,
        "readiness": readiness,
    }


@router.post("/model/train-profile")
def train_profile_model_route(
    request: Request,
    max_samples: int = 5000,
    admin_user: dict = Depends(require_admin_user),
):
    safe_max_samples = max(100, min(max_samples, 20000))
    try:
        result = train_profile_model(max_samples=safe_max_samples)
    except ValueError as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.train_profile",
            status="failed",
            request=request,
            detail=f"Validation error: {str(exc)}",
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.train_profile",
            status="failed",
            request=request,
            detail=f"Runtime error: {str(exc)}",
        )
        raise HTTPException(status_code=500, detail=f"Profile training failed: {str(exc)}") from exc

    readiness = get_model_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.train_profile",
        status="success",
        request=request,
        detail=f"Trained with max_samples={safe_max_samples}.",
    )
    return {
        "trained": True,
        "max_samples": safe_max_samples,
        **result,
        "readiness": readiness,
    }


@router.get("/model/profile-feedback-stats")
def model_profile_feedback_stats(_: dict = Depends(require_admin_user)):
    stats, error = fetch_profile_feedback_stats(min_per_class_required=20)
    if error:
        raise HTTPException(status_code=500, detail=error)
    return stats


@router.get("/model/evaluation")
def model_evaluation(
    request: Request,
    image_max_per_class: int = 2000,
    profile_max_samples: int = 6000,
    admin_user: dict = Depends(require_admin_user),
):
    safe_image = max(200, min(image_max_per_class, 10000))
    safe_profile = max(200, min(profile_max_samples, 20000))
    thresholds = get_effective_thresholds()
    values = thresholds.get("values", {})
    image_eval = evaluate_image_model(
        max_per_class=safe_image,
        threshold_pct=float(values.get("image_fake_probability_threshold", 50.0)),
    )
    profile_eval = evaluate_profile_model(
        max_samples=safe_profile,
        threshold_pct=float(values.get("profile_fake_probability_threshold", 50.0)),
    )
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.evaluation",
        status="success",
        request=request,
        detail=(
            f"Evaluated models with image_max_per_class={safe_image}, "
            f"profile_max_samples={safe_profile}."
        ),
    )
    return {
        "thresholds": thresholds,
        "image_model": image_eval,
        "profile_model": profile_eval,
    }


@router.get("/model/thresholds")
def model_thresholds_get(
    request: Request,
    admin_user: dict = Depends(require_admin_user),
):
    result = get_effective_thresholds()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.thresholds_get",
        status="success",
        request=request,
        detail="Loaded threshold configuration.",
    )
    return result


@router.post("/model/thresholds")
def model_thresholds_set(
    payload: ThresholdsUpdateRequest,
    request: Request,
    admin_user: dict = Depends(require_admin_user),
):
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No threshold values provided.")
    try:
        result = save_threshold_overrides(updates)
    except ValueError as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.thresholds_set",
            status="failed",
            request=request,
            detail=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.thresholds_set",
        status="success",
        request=request,
        detail=f"Updated keys: {', '.join(sorted(updates.keys()))}",
    )
    return result


@router.get("/model/export-feedback-csv")
def export_profile_feedback_csv(
    request: Request,
    labeled_only: bool = True,
    limit: int = 10000,
    admin_user: dict = Depends(require_admin_user),
):
    rows, error = fetch_profile_feedback_export_rows(labeled_only=labeled_only, limit=limit)
    if error:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.export_feedback_csv",
            status="failed",
            request=request,
            detail=error,
        )
        raise HTTPException(status_code=500, detail=error)

    headers = [
        "report_id",
        "created_at",
        "username",
        "bio",
        "followers",
        "following",
        "posts",
        "engagement_rate",
        "image_score",
        "trust_score",
        "risk_level",
        "feedback_label",
        "feedback_note",
        "profile_model_source",
        "profile_model_score",
        "profile_model_fake_probability",
    ]

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8")
    filename = "deepshield_profile_feedback.csv"
    content_headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.export_feedback_csv",
        status="success",
        request=request,
        detail=f"Exported {len(rows)} rows.",
    )
    return StreamingResponse(BytesIO(csv_bytes), media_type="text/csv", headers=content_headers)


@router.get("/model/export-reports-json")
def export_profile_reports_json(
    request: Request,
    limit: int = 10000,
    admin_user: dict = Depends(require_admin_user),
):
    rows, error = fetch_profile_reports_export_json(limit=limit)
    if error:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="model.export_reports_json",
            status="failed",
            request=request,
            detail=error,
        )
        raise HTTPException(status_code=500, detail=error)

    json_bytes = json.dumps(rows, ensure_ascii=True, indent=2).encode("utf-8")
    filename = "deepshield_profile_reports.json"
    content_headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="model.export_reports_json",
        status="success",
        request=request,
        detail=f"Exported {len(rows)} rows.",
    )
    return StreamingResponse(BytesIO(json_bytes), media_type="application/json", headers=content_headers)


@router.get("/system/audit-logs")
def system_audit_logs(
    request: Request,
    limit: int = 50,
    action: str | None = None,
    actor_username: str | None = None,
    admin_user: dict = Depends(require_admin_user),
):
    items, error = fetch_audit_logs(limit=limit, action=action, actor_username=actor_username)
    if error:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="system.audit_logs",
            status="failed",
            request=request,
            detail=error,
        )
        raise HTTPException(status_code=500, detail=error)
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="system.audit_logs",
        status="success",
        request=request,
        detail=f"Returned {len(items)} audit rows.",
    )
    return {"items": items, "count": len(items), "error": None}


SUSPICIOUS_PATTERNS = [
    "official",
    "support",
    "helpdesk",
    "free",
    "giveaway",
    "crypto",
    "bitcoin",
    "investment",
    "dm for",
    "click link",
    "urgent",
    "verify now",
    "loan",
]


def _contains_suspicious_pattern(text: str) -> int:
    lower_text = text.lower()
    return sum(1 for pattern in SUSPICIOUS_PATTERNS if pattern in lower_text)


def _matched_suspicious_patterns(text: str) -> list[str]:
    lower_text = text.lower()
    return sorted({pattern for pattern in SUSPICIOUS_PATTERNS if pattern in lower_text})


def _profile_text_score(username: str, bio: str) -> tuple[float, list[str]]:
    score = 100.0
    factors: list[str] = []
    combined = f"{username} {bio}"
    matched_patterns = _matched_suspicious_patterns(combined)
    suspicious_count = len(matched_patterns)

    score -= suspicious_count * 12
    if matched_patterns:
        factors.append(f"Suspicious keywords found: {', '.join(matched_patterns)}")
    if re.search(r"\d{4,}", username):
        score -= 8
        factors.append("Username contains long numeric sequence.")
    if len(set(username.lower())) <= 4:
        score -= 6
        factors.append("Username appears repetitive.")
    if bio and len(bio.strip()) < 10:
        score -= 5
        factors.append("Bio is very short and provides limited identity context.")

    return max(0.0, min(100.0, round(score, 2))), factors


def _engagement_score(followers: int, following: int, posts: int, engagement_rate: float) -> tuple[float, list[str]]:
    score = 100.0
    factors: list[str] = []
    ratio = followers / max(following, 1)

    if followers < 25 and following > 500:
        score -= 35
        factors.append("Very low followers with very high following.")
    elif ratio < 0.1:
        score -= 20
        factors.append("Follower/following ratio is unusually low.")

    if posts == 0:
        score -= 20
        factors.append("No posts available on profile.")
    elif posts < 3:
        score -= 10
        factors.append("Very few posts available on profile.")

    if engagement_rate < 0.3:
        score -= 25
        factors.append("Engagement rate is extremely low.")
    elif engagement_rate < 1.0:
        score -= 12
        factors.append("Engagement rate is lower than expected.")
    elif engagement_rate > 25:
        score -= 10
        factors.append("Engagement rate is unusually high (possible inorganic activity).")

    return max(0.0, min(100.0, round(score, 2))), factors


@router.post("/analyze/profile")
def analyze_profile(payload: ProfileAnalyzeRequest, current_user: dict = Depends(require_authenticated_user)):
    heuristic_profile_score, profile_factors = _profile_text_score(payload.username, payload.bio)
    profile_score = heuristic_profile_score
    profile_model = predict_profile_text_authenticity(payload.username, payload.bio)
    profile_model_source = "heuristic"
    profile_model_weight = 0.0
    profile_heuristic_weight = 1.0
    if profile_model:
        model_profile_score = float(profile_model["profile_score"])
        profile_model_weight = float(profile_model.get("recommended_weight", 0.6))
        profile_heuristic_weight = round(1.0 - profile_model_weight, 2)
        profile_score = round(
            (profile_model_weight * model_profile_score) + (profile_heuristic_weight * heuristic_profile_score),
            2,
        )
        profile_model_source = "trained_profile_model+heuristic_fusion"
        if abs(model_profile_score - heuristic_profile_score) >= 18:
            profile_factors.append("Profile NLP model disagrees with heuristic signals.")
    engagement_score, engagement_factors = _engagement_score(
        payload.followers,
        payload.following,
        payload.posts,
        payload.engagement_rate,
    )
    trust_score = calculate_trust_score(payload.image_score, profile_score, engagement_score)
    risk_level = classify_risk(trust_score)
    component_scores = {
        "image_score": round(payload.image_score, 2),
        "profile_score": profile_score,
        "engagement_score": engagement_score,
    }
    confidence_breakdown = {
        "image_contribution": round(0.45 * payload.image_score, 2),
        "profile_contribution": round(0.35 * profile_score, 2),
        "engagement_contribution": round(0.20 * engagement_score, 2),
    }
    risk_factors = [*profile_factors, *engagement_factors]
    if payload.image_score < 45:
        risk_factors.append("Media authenticity score is low.")
    if not risk_factors:
        risk_factors.append("No major risk factors detected.")

    recommendation_details = build_recommendation(trust_score, risk_level, component_scores)
    recommendation = recommendation_details["label"]
    report_payload = {
        "analysis_type": "profile",
        "owner_username": current_user["username"],
        "owner_role": current_user["role"],
        "input_profile": payload.model_dump(),
        "trust_score": trust_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "component_scores": component_scores,
        "confidence_breakdown": confidence_breakdown,
        "risk_factors": risk_factors,
        "recommendation_details": recommendation_details,
        "profile_model_source": profile_model_source,
        "profile_model_weight": profile_model_weight,
        "profile_heuristic_weight": profile_heuristic_weight,
        "profile_heuristic_score": heuristic_profile_score,
        "profile_model_score": profile_model["profile_score"] if profile_model else None,
        "profile_model_fake_probability": profile_model["fake_probability"] if profile_model else None,
    }
    report_saved, report_id, save_error = save_report(report_payload)

    return {
        "trust_score": trust_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "component_scores": component_scores,
        "confidence_breakdown": confidence_breakdown,
        "risk_factors": risk_factors,
        "recommendation_details": recommendation_details,
        "profile_model_source": profile_model_source,
        "profile_model_weight": profile_model_weight,
        "profile_heuristic_weight": profile_heuristic_weight,
        "profile_heuristic_score": heuristic_profile_score,
        "profile_model_score": profile_model["profile_score"] if profile_model else None,
        "profile_model_fake_probability": profile_model["fake_probability"] if profile_model else None,
        "report_saved": report_saved,
        "report_id": report_id,
        "save_error": save_error,
    }


@router.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...), _: dict = Depends(require_authenticated_user)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPG, PNG, or WEBP.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        image_result = analyze_image_authenticity(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        **image_result,
    }


@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), _: dict = Depends(require_authenticated_user)):
    allowed_types = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"}
    allowed_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    suffix = Path(file.filename or "").suffix.lower()

    if file.content_type not in allowed_types and suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video type. Use MP4, MOV, AVI, WEBM, or MKV.",
        )

    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp_file:
            tmp_file_path = tmp_file.name
            await file.seek(0)
            shutil.copyfileobj(file.file, tmp_file)

        video_result = analyze_video_authenticity(tmp_file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if tmp_file_path:
            Path(tmp_file_path).unlink(missing_ok=True)

    return {
        "filename": file.filename,
        **video_result,
    }


@router.post("/dataset/upload-image")
async def upload_dataset_image(
    request: Request,
    file: UploadFile = File(...),
    label: str = Form(...),
    admin_user: dict = Depends(require_admin_user),
):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    suffix = Path(file.filename or "").suffix.lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}

    if file.content_type not in allowed_types and suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPG, PNG, or WEBP.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        saved_path = save_labeled_image(
            file_bytes=file_bytes,
            label=label.lower().strip(),
            original_name=file.filename or "upload.jpg",
        )
    except ValueError as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="dataset.upload_image",
            status="failed",
            request=request,
            detail=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    readiness = get_model_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="dataset.upload_image",
        status="success",
        request=request,
        target=file.filename or "",
        detail=f"Uploaded label={label.lower().strip()}",
    )
    return {
        "saved": True,
        "label": label.lower().strip(),
        "filename": Path(saved_path).name,
        "saved_path": saved_path,
        "dataset_counts": readiness["dataset_counts"],
    }


@router.post("/dataset/upload-zip")
async def upload_dataset_zip(
    request: Request,
    file: UploadFile = File(...),
    admin_user: dict = Depends(require_admin_user),
):
    allowed_types = {"application/zip", "application/x-zip-compressed"}
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in allowed_types and suffix != ".zip":
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a ZIP file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty ZIP file.")

    try:
        import_result = import_labeled_images_zip(file_bytes)
    except ValueError as exc:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="dataset.upload_zip",
            status="failed",
            request=request,
            detail=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    readiness = get_model_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="dataset.upload_zip",
        status="success",
        request=request,
        target=file.filename or "",
        detail=(
            f"Imported real={import_result.get('imported_real', 0)}, "
            f"fake={import_result.get('imported_fake', 0)}."
        ),
    )
    return {
        "saved": True,
        "filename": file.filename,
        **import_result,
        "dataset_counts": readiness["dataset_counts"],
    }


@router.get("/dataset/quality")
def dataset_quality(
    request: Request,
    max_scan_per_class: int = 2000,
    admin_user: dict = Depends(require_admin_user),
):
    safe_scan = max(200, min(max_scan_per_class, 10000))
    result = get_dataset_quality_report(max_scan_per_class=safe_scan)
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="dataset.quality",
        status="success" if result.get("available") else "failed",
        request=request,
        detail=f"Dataset quality scan max_scan_per_class={safe_scan}.",
    )
    return result


@router.get("/reports")
def get_reports(
    limit: int = 20,
    page: int = 1,
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    current_user: dict = Depends(require_authenticated_user),
):
    safe_limit = max(1, min(limit, 50))
    safe_page = max(1, min(page, 100000))
    safe_risk = risk_level.title() if risk_level else None
    safe_feedback = feedback_label if feedback_label else None
    if safe_risk not in {None, "Low", "Medium", "High"}:
        raise HTTPException(status_code=400, detail="risk_level must be Low, Medium, or High.")
    if safe_feedback not in {None, "confirmed_real", "confirmed_fake", "unsure"}:
        raise HTTPException(status_code=400, detail="feedback_label must be confirmed_real, confirmed_fake, or unsure.")

    scoped_username = username
    username_exact = False
    owner_username = None
    owner_legacy_username = None
    if current_user["role"] != "admin":
        owner_username = current_user["username"]
        owner_legacy_username = current_user["username"]
        scoped_username = None

    items, total_count, error = fetch_recent_reports_paginated(
        page=safe_page,
        page_size=safe_limit,
        risk_level=safe_risk,
        username=scoped_username,
        feedback_label=safe_feedback,
        username_exact=username_exact,
        owner_username=owner_username,
        owner_legacy_username=owner_legacy_username,
    )
    total_pages = (total_count + safe_limit - 1) // safe_limit if total_count > 0 else 0
    return {
        "items": items,
        "count": len(items),
        "total_count": total_count,
        "page": safe_page,
        "page_size": safe_limit,
        "total_pages": total_pages,
        "has_prev": safe_page > 1,
        "has_next": safe_page < total_pages,
        "filters": {"risk_level": safe_risk, "username": scoped_username or "", "feedback_label": safe_feedback or ""},
        "error": error,
    }


@router.get("/reports/stats")
def get_reports_stats(
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    current_user: dict = Depends(require_authenticated_user),
):
    safe_risk = risk_level.title() if risk_level else None
    safe_feedback = feedback_label if feedback_label else None
    if safe_risk not in {None, "Low", "Medium", "High"}:
        raise HTTPException(status_code=400, detail="risk_level must be Low, Medium, or High.")
    if safe_feedback not in {None, "confirmed_real", "confirmed_fake", "unsure"}:
        raise HTTPException(status_code=400, detail="feedback_label must be confirmed_real, confirmed_fake, or unsure.")

    scoped_username = username
    username_exact = False
    owner_username = None
    owner_legacy_username = None
    if current_user["role"] != "admin":
        owner_username = current_user["username"]
        owner_legacy_username = current_user["username"]
        scoped_username = None

    stats, error = fetch_report_stats(
        safe_risk,
        scoped_username,
        safe_feedback,
        username_exact=username_exact,
        owner_username=owner_username,
        owner_legacy_username=owner_legacy_username,
    )
    return {
        **stats,
        "filters": {"risk_level": safe_risk, "username": scoped_username or "", "feedback_label": safe_feedback or ""},
        "error": error,
    }


@router.get("/reports/trend")
def get_reports_trend(
    limit: int = 30,
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    current_user: dict = Depends(require_authenticated_user),
):
    safe_limit = max(3, min(limit, 100))
    safe_risk = risk_level.title() if risk_level else None
    safe_feedback = feedback_label if feedback_label else None
    if safe_risk not in {None, "Low", "Medium", "High"}:
        raise HTTPException(status_code=400, detail="risk_level must be Low, Medium, or High.")
    if safe_feedback not in {None, "confirmed_real", "confirmed_fake", "unsure"}:
        raise HTTPException(status_code=400, detail="feedback_label must be confirmed_real, confirmed_fake, or unsure.")

    scoped_username = username
    username_exact = False
    owner_username = None
    owner_legacy_username = None
    if current_user["role"] != "admin":
        owner_username = current_user["username"]
        owner_legacy_username = current_user["username"]
        scoped_username = None

    items, error = fetch_report_trend(
        safe_limit,
        safe_risk,
        scoped_username,
        safe_feedback,
        username_exact=username_exact,
        owner_username=owner_username,
        owner_legacy_username=owner_legacy_username,
    )
    return {
        "items": items,
        "count": len(items),
        "filters": {"risk_level": safe_risk, "username": scoped_username or "", "feedback_label": safe_feedback or ""},
        "error": error,
    }


@router.get("/reports/{report_id}")
def get_report_detail(report_id: str, current_user: dict = Depends(require_authenticated_user)):
    detail, error = fetch_report_detail(report_id)
    if error == "Invalid report id.":
        raise HTTPException(status_code=400, detail=error)
    if error:
        raise HTTPException(status_code=404, detail=error)
    if current_user["role"] != "admin":
        owner_username = str(detail.get("owner_username", "")).strip()
        legacy_username = str(detail.get("input_profile", {}).get("username", "")).strip()
        if owner_username:
            if owner_username != current_user["username"]:
                raise HTTPException(status_code=403, detail="Access denied for this report.")
        elif legacy_username != current_user["username"]:
            raise HTTPException(status_code=403, detail="Access denied for this report.")
    return detail


@router.patch("/reports/{report_id}/feedback")
def patch_report_feedback(
    report_id: str,
    payload: ReportFeedbackRequest,
    request: Request,
    admin_user: dict = Depends(require_admin_user),
):
    ok, error = update_report_feedback(
        report_id,
        payload.feedback_label,
        payload.feedback_note.strip(),
    )
    if not ok and error == "Invalid report id.":
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="reports.feedback_update",
            status="failed",
            request=request,
            target=report_id,
            detail=error,
        )
        raise HTTPException(status_code=400, detail=error)
    if not ok and error == "Report not found.":
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="reports.feedback_update",
            status="failed",
            request=request,
            target=report_id,
            detail=error,
        )
        raise HTTPException(status_code=404, detail=error)
    if not ok:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="reports.feedback_update",
            status="failed",
            request=request,
            target=report_id,
            detail=error or "Unknown error.",
        )
        raise HTTPException(status_code=500, detail=error or "Unable to save feedback.")

    detail, detail_error = fetch_report_detail(report_id)
    if detail_error:
        _write_audit_event(
            actor_username=admin_user["username"],
            actor_role=admin_user["role"],
            action="reports.feedback_update",
            status="failed",
            request=request,
            target=report_id,
            detail=detail_error,
        )
        raise HTTPException(status_code=500, detail=detail_error)

    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="reports.feedback_update",
        status="success",
        request=request,
        target=report_id,
        detail=f"label={payload.feedback_label}",
    )
    return {"updated": True, "item": detail}


@router.get("/reports/{report_id}/pdf")
def download_report_pdf(report_id: str, current_user: dict = Depends(require_authenticated_user)):
    report, error = fetch_report_by_id(report_id)
    if error == "Invalid report id.":
        raise HTTPException(status_code=400, detail=error)
    if error:
        raise HTTPException(status_code=404, detail=error)
    if current_user["role"] != "admin":
        owner_username = str(report.get("owner_username", "")).strip()
        legacy_username = str(report.get("input_profile", {}).get("username", "")).strip()
        if owner_username:
            if owner_username != current_user["username"]:
                raise HTTPException(status_code=403, detail="Access denied for this report.")
        elif legacy_username != current_user["username"]:
            raise HTTPException(status_code=403, detail="Access denied for this report.")

    pdf_bytes = build_report_pdf(report)
    filename = f"deepshield_report_{report_id[:8]}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)


@router.post("/demo/generate-profiles")
def generate_demo_profiles(
    request: Request,
    count: int = 25,
    auto_feedback: bool = True,
    admin_user: dict = Depends(require_admin_user),
):
    safe_count = max(1, min(count, 200))
    normal_bios = [
        "Photographer and traveler. Sharing real moments.",
        "Software student. Building projects and learning daily.",
        "Fitness coach. Tips on healthy lifestyle.",
        "Food blogger. Honest reviews and recipes.",
    ]
    suspicious_bios = [
        "Official support. Verify now and claim giveaway.",
        "Crypto investment helper. DM for quick returns.",
        "Urgent account verify now. Click link in bio.",
        "Free bitcoin bonus, limited time offer.",
    ]

    generated = 0
    labeled_real = 0
    labeled_fake = 0

    for idx in range(safe_count):
        suspicious = idx % 2 == 0
        if suspicious:
            username = f"{random.choice(['support', 'official', 'helpdesk'])}{random.randint(1000, 999999)}"
            bio = random.choice(suspicious_bios)
            followers = random.randint(5, 120)
            following = random.randint(300, 1500)
            posts = random.randint(0, 8)
            engagement_rate = round(random.uniform(0.1, 45.0), 2)
            image_score = round(random.uniform(22, 62), 2)
        else:
            base_name = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
            username = f"{base_name}_{random.randint(10, 999)}"
            bio = random.choice(normal_bios)
            followers = random.randint(180, 5000)
            following = random.randint(60, 1400)
            posts = random.randint(12, 240)
            engagement_rate = round(random.uniform(1.2, 16.0), 2)
            image_score = round(random.uniform(58, 97), 2)

        payload = ProfileAnalyzeRequest(
            username=username,
            bio=bio,
            followers=followers,
            following=following,
            posts=posts,
            engagement_rate=engagement_rate,
            image_score=image_score,
        )

        result = analyze_profile(payload, current_user=admin_user)
        generated += 1

        report_id = result.get("report_id")
        if auto_feedback and report_id:
            trust_score = float(result.get("trust_score", 50.0))
            if trust_score >= 70:
                feedback_label = "confirmed_real"
                labeled_real += 1
            elif trust_score <= 55:
                feedback_label = "confirmed_fake"
                labeled_fake += 1
            else:
                feedback_label = "confirmed_fake" if suspicious else "confirmed_real"
                if feedback_label == "confirmed_real":
                    labeled_real += 1
                else:
                    labeled_fake += 1

            update_report_feedback(report_id, feedback_label, "auto-generated demo feedback")

    stats, stats_error = fetch_profile_feedback_stats(min_per_class_required=20)
    readiness = get_model_readiness()
    _write_audit_event(
        actor_username=admin_user["username"],
        actor_role=admin_user["role"],
        action="demo.generate_profiles",
        status="success",
        request=request,
        detail=f"generated={generated}, auto_feedback={auto_feedback}",
    )
    return {
        "generated": generated,
        "auto_feedback": auto_feedback,
        "auto_labeled_real": labeled_real,
        "auto_labeled_fake": labeled_fake,
        "feedback_stats": stats if not stats_error else None,
        "feedback_stats_error": stats_error,
        "readiness": readiness,
    }
