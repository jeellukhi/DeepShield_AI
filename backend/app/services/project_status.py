from __future__ import annotations

from app.db.mongo import fetch_profile_feedback_stats, fetch_report_stats
from app.services.model_readiness import get_model_readiness


def get_project_status() -> dict:
    readiness = get_model_readiness()
    feedback_stats, feedback_error = fetch_profile_feedback_stats(min_per_class_required=20)
    report_stats, report_error = fetch_report_stats()

    total_reports = int((report_stats or {}).get("total_reports", 0))
    total_images = int(readiness["dataset_counts"]["total_images"])
    image_model_exists = bool(readiness["model_files"]["image_model_exists"])
    profile_model_exists = bool(readiness["model_files"]["profile_model_exists"])
    profile_model_usable = bool(readiness["model_files"].get("profile_model_usable", False))
    feedback_ready = bool((feedback_stats or {}).get("ready_for_quality_gate", False))

    checklist = {
        "dataset_loaded": total_images > 0,
        "image_model_trained": image_model_exists,
        "profile_feedback_ready": feedback_ready,
        "profile_model_trained": profile_model_exists,
        "profile_model_usable": profile_model_usable,
        "reports_available": total_reports > 0,
    }

    completed_count = sum(1 for _, passed in checklist.items() if passed)
    total_count = len(checklist)
    completion_percent = round((completed_count / total_count) * 100, 2)

    mvp_ready = (
        checklist["dataset_loaded"]
        and checklist["image_model_trained"]
        and checklist["profile_model_trained"]
        and checklist["profile_model_usable"]
        and checklist["reports_available"]
    )

    if mvp_ready:
        next_action = "MVP complete. Continue with validation, security hardening, and deployment."
    elif not checklist["dataset_loaded"]:
        next_action = "Upload/import dataset images first."
    elif not checklist["image_model_trained"]:
        next_action = "Train image model."
    elif not checklist["profile_feedback_ready"]:
        next_action = "Collect more confirmed_real/confirmed_fake feedback labels."
    elif not checklist["profile_model_trained"] or not checklist["profile_model_usable"]:
        next_action = "Train/retrain profile model until quality gate passes."
    else:
        next_action = "Generate or analyze more profiles to populate report history."

    return {
        "mvp_ready": mvp_ready,
        "completion_percent": completion_percent,
        "completed_items": completed_count,
        "total_items": total_count,
        "checklist": checklist,
        "next_action": next_action,
        "snapshots": {
            "model_readiness": readiness,
            "feedback_stats": feedback_stats if not feedback_error else None,
            "report_stats": report_stats if not report_error else None,
        },
        "errors": {
            "feedback_error": feedback_error,
            "report_error": report_error,
        },
    }
