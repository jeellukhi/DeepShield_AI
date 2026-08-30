import os
import re
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

_client = None


def _to_iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _get_db():
    global _client
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "deepshield_ai")

    if _client is None:
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1500)

    return _client[db_name]


def _get_collection():
    return _get_db()["analysis_reports"]


def _get_users_collection():
    return _get_db()["auth_users"]


def _get_audit_collection():
    return _get_db()["audit_logs"]


def _apply_feedback_filter(query: dict, feedback_label: str | None) -> None:
    if not feedback_label:
        return
    if feedback_label == "unsure":
        query["$or"] = [
            {"feedback.label": "unsure"},
            {"feedback.label": {"$exists": False}},
            {"feedback.label": None},
            {"feedback": {"$exists": False}},
        ]
        return
    query["feedback.label"] = feedback_label


def _build_profile_reports_query(
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    username_exact: bool = False,
    owner_username: str | None = None,
    owner_legacy_username: str | None = None,
) -> dict:
    conditions: list[dict] = [{"analysis_type": "profile"}]
    if risk_level:
        conditions.append({"risk_level": risk_level})
    if username:
        safe_username = str(username).strip()
        if username_exact:
            conditions.append({"input_profile.username": {"$regex": f"^{re.escape(safe_username)}$", "$options": "i"}})
        else:
            conditions.append({"input_profile.username": {"$regex": safe_username, "$options": "i"}})

    if feedback_label == "unsure":
        conditions.append(
            {
                "$or": [
                    {"feedback.label": "unsure"},
                    {"feedback.label": {"$exists": False}},
                    {"feedback.label": None},
                    {"feedback": {"$exists": False}},
                ]
            }
        )
    elif feedback_label:
        conditions.append({"feedback.label": feedback_label})

    if owner_username:
        owner = str(owner_username).strip()
        legacy = str(owner_legacy_username or owner_username).strip()
        conditions.append(
            {
                "$or": [
                    {"owner_username": owner},
                    {"owner_username": {"$exists": False}, "input_profile.username": legacy},
                ]
            }
        )

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _format_report_item(doc: dict) -> dict:
    feedback = doc.get("feedback", {})
    feedback_updated = feedback.get("updated_at")
    return {
        "id": str(doc.get("_id")),
        "analysis_type": doc.get("analysis_type", "profile"),
        "trust_score": doc.get("trust_score"),
        "risk_level": doc.get("risk_level"),
        "recommendation": doc.get("recommendation"),
        "risk_factor_count": len(doc.get("risk_factors", [])),
        "feedback_label": feedback.get("label"),
        "feedback_updated_at": _to_iso(feedback_updated),
        "username": doc.get("input_profile", {}).get("username", ""),
        "created_at": _to_iso(doc.get("created_at")),
    }


def save_report(report: dict) -> tuple[bool, str | None, str | None]:
    try:
        collection = _get_collection()
        report["created_at"] = datetime.now(timezone.utc)
        inserted = collection.insert_one(report)
        return True, str(inserted.inserted_id), None
    except PyMongoError as exc:
        return False, None, str(exc)


def fetch_recent_reports(
    limit: int = 10,
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    username_exact: bool = False,
    owner_username: str | None = None,
    owner_legacy_username: str | None = None,
) -> tuple[list[dict], str | None]:
    try:
        collection = _get_collection()
        query = _build_profile_reports_query(
            risk_level,
            username,
            feedback_label,
            username_exact=username_exact,
            owner_username=owner_username,
            owner_legacy_username=owner_legacy_username,
        )

        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        items: list[dict] = []
        for doc in cursor:
            items.append(_format_report_item(doc))
        return items, None
    except PyMongoError as exc:
        return [], str(exc)


def fetch_recent_reports_paginated(
    page: int = 1,
    page_size: int = 20,
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    username_exact: bool = False,
    owner_username: str | None = None,
    owner_legacy_username: str | None = None,
) -> tuple[list[dict], int, str | None]:
    try:
        collection = _get_collection()
        safe_page = max(1, page)
        safe_page_size = max(1, min(page_size, 100))
        skip_count = (safe_page - 1) * safe_page_size
        query = _build_profile_reports_query(
            risk_level,
            username,
            feedback_label,
            username_exact=username_exact,
            owner_username=owner_username,
            owner_legacy_username=owner_legacy_username,
        )

        total_count = int(collection.count_documents(query))
        cursor = collection.find(query).sort("created_at", -1).skip(skip_count).limit(safe_page_size)
        items: list[dict] = []
        for doc in cursor:
            items.append(_format_report_item(doc))
        return items, total_count, None
    except PyMongoError as exc:
        return [], 0, str(exc)


def fetch_report_stats(
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    username_exact: bool = False,
    owner_username: str | None = None,
    owner_legacy_username: str | None = None,
) -> tuple[dict, str | None]:
    try:
        collection = _get_collection()
        match_query = _build_profile_reports_query(
            risk_level,
            username,
            feedback_label,
            username_exact=username_exact,
            owner_username=owner_username,
            owner_legacy_username=owner_legacy_username,
        )

        pipeline = [
            {"$match": match_query},
            {
                "$group": {
                    "_id": None,
                    "total_reports": {"$sum": 1},
                    "avg_trust_score": {"$avg": "$trust_score"},
                    "low_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "Low"]}, 1, 0]}},
                    "medium_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "Medium"]}, 1, 0]}},
                    "high_risk_count": {"$sum": {"$cond": [{"$eq": ["$risk_level", "High"]}, 1, 0]}},
                }
            },
        ]
        result = list(collection.aggregate(pipeline))
        if not result:
            return {
                "total_reports": 0,
                "avg_trust_score": 0.0,
                "low_risk_count": 0,
                "medium_risk_count": 0,
                "high_risk_count": 0,
            }, None

        row = result[0]
        return {
            "total_reports": int(row.get("total_reports", 0)),
            "avg_trust_score": round(float(row.get("avg_trust_score", 0.0)), 2),
            "low_risk_count": int(row.get("low_risk_count", 0)),
            "medium_risk_count": int(row.get("medium_risk_count", 0)),
            "high_risk_count": int(row.get("high_risk_count", 0)),
        }, None
    except PyMongoError as exc:
        return {}, str(exc)


def fetch_report_trend(
    limit: int = 30,
    risk_level: str | None = None,
    username: str | None = None,
    feedback_label: str | None = None,
    username_exact: bool = False,
    owner_username: str | None = None,
    owner_legacy_username: str | None = None,
) -> tuple[list[dict], str | None]:
    try:
        collection = _get_collection()
        query = _build_profile_reports_query(
            risk_level,
            username,
            feedback_label,
            username_exact=username_exact,
            owner_username=owner_username,
            owner_legacy_username=owner_legacy_username,
        )

        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        recent = list(cursor)
        recent.reverse()

        points: list[dict] = []
        for doc in recent:
            points.append(
                {
                    "id": str(doc.get("_id")),
                    "username": doc.get("input_profile", {}).get("username", ""),
                    "trust_score": doc.get("trust_score"),
                    "risk_level": doc.get("risk_level"),
                    "created_at": _to_iso(doc.get("created_at")),
                }
            )
        return points, None
    except PyMongoError as exc:
        return [], str(exc)


def fetch_report_by_id(report_id: str) -> tuple[dict | None, str | None]:
    try:
        object_id = ObjectId(report_id)
    except InvalidId:
        return None, "Invalid report id."

    try:
        collection = _get_collection()
        doc = collection.find_one({"_id": object_id})
        if not doc:
            return None, "Report not found."
        return doc, None
    except PyMongoError as exc:
        return None, str(exc)


def fetch_report_detail(report_id: str) -> tuple[dict | None, str | None]:
    report, error = fetch_report_by_id(report_id)
    if error:
        return None, error

    detail = {
        "id": str(report.get("_id")),
        "analysis_type": report.get("analysis_type", "profile"),
        "owner_username": report.get("owner_username", ""),
        "owner_role": report.get("owner_role", ""),
        "input_profile": report.get("input_profile", {}),
        "trust_score": report.get("trust_score"),
        "risk_level": report.get("risk_level"),
        "recommendation": report.get("recommendation"),
        "component_scores": report.get("component_scores", {}),
        "confidence_breakdown": report.get("confidence_breakdown", {}),
        "profile_model_source": report.get("profile_model_source"),
        "profile_model_weight": report.get("profile_model_weight"),
        "profile_heuristic_weight": report.get("profile_heuristic_weight"),
        "profile_heuristic_score": report.get("profile_heuristic_score"),
        "profile_model_score": report.get("profile_model_score"),
        "profile_model_fake_probability": report.get("profile_model_fake_probability"),
        "risk_factors": report.get("risk_factors", []),
        "recommendation_details": report.get("recommendation_details", {}),
        "feedback": {
            "label": report.get("feedback", {}).get("label"),
            "note": report.get("feedback", {}).get("note", ""),
            "updated_at": _to_iso(report.get("feedback", {}).get("updated_at")),
        },
        "created_at": _to_iso(report.get("created_at")),
    }
    return detail, None


def update_report_feedback(report_id: str, feedback_label: str, feedback_note: str) -> tuple[bool, str | None]:
    try:
        object_id = ObjectId(report_id)
    except InvalidId:
        return False, "Invalid report id."

    try:
        collection = _get_collection()
        updated = collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "feedback": {
                        "label": feedback_label,
                        "note": feedback_note,
                        "updated_at": datetime.now(timezone.utc),
                    }
                }
            },
        )
        if updated.matched_count == 0:
            return False, "Report not found."
        return True, None
    except PyMongoError as exc:
        return False, str(exc)


def fetch_profile_feedback_training_samples(limit: int = 5000) -> tuple[list[dict], str | None]:
    try:
        collection = _get_collection()
        query = {
            "analysis_type": "profile",
            "feedback.label": {"$in": ["confirmed_real", "confirmed_fake"]},
        }
        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        rows: list[dict] = []
        for doc in cursor:
            profile = doc.get("input_profile", {})
            feedback_label = doc.get("feedback", {}).get("label")
            rows.append(
                {
                    "username": str(profile.get("username", "")).strip(),
                    "bio": str(profile.get("bio", "")).strip(),
                    "feedback_label": feedback_label,
                }
            )
        return rows, None
    except PyMongoError as exc:
        return [], str(exc)


def fetch_profile_feedback_stats(min_per_class_required: int = 20) -> tuple[dict, str | None]:
    try:
        collection = _get_collection()
        query = {"analysis_type": "profile"}
        total_reports = int(collection.count_documents(query))
        confirmed_real = int(collection.count_documents({**query, "feedback.label": "confirmed_real"}))
        confirmed_fake = int(collection.count_documents({**query, "feedback.label": "confirmed_fake"}))
        unsure_or_missing = max(0, total_reports - confirmed_real - confirmed_fake)

        return {
            "total_reports": total_reports,
            "confirmed_real": confirmed_real,
            "confirmed_fake": confirmed_fake,
            "unsure_or_missing": unsure_or_missing,
            "min_per_class_required": min_per_class_required,
            "remaining_real": max(0, min_per_class_required - confirmed_real),
            "remaining_fake": max(0, min_per_class_required - confirmed_fake),
            "ready_for_quality_gate": (
                confirmed_real >= min_per_class_required and confirmed_fake >= min_per_class_required
            ),
        }, None
    except PyMongoError as exc:
        return {}, str(exc)


def fetch_profile_feedback_export_rows(
    labeled_only: bool = True,
    limit: int = 10000,
) -> tuple[list[dict], str | None]:
    try:
        collection = _get_collection()
        query: dict = {"analysis_type": "profile"}
        if labeled_only:
            query["feedback.label"] = {"$in": ["confirmed_real", "confirmed_fake"]}

        cursor = collection.find(query).sort("created_at", -1).limit(max(1, min(limit, 100000)))
        rows: list[dict] = []
        for doc in cursor:
            profile = doc.get("input_profile", {})
            feedback = doc.get("feedback", {})
            rows.append(
                {
                    "report_id": str(doc.get("_id")),
                    "created_at": _to_iso(doc.get("created_at")) or "",
                    "username": str(profile.get("username", "")),
                    "bio": str(profile.get("bio", "")),
                    "followers": profile.get("followers", ""),
                    "following": profile.get("following", ""),
                    "posts": profile.get("posts", ""),
                    "engagement_rate": profile.get("engagement_rate", ""),
                    "image_score": profile.get("image_score", ""),
                    "trust_score": doc.get("trust_score", ""),
                    "risk_level": doc.get("risk_level", ""),
                    "feedback_label": feedback.get("label", ""),
                    "feedback_note": feedback.get("note", ""),
                    "profile_model_source": doc.get("profile_model_source", ""),
                    "profile_model_score": doc.get("profile_model_score", ""),
                    "profile_model_fake_probability": doc.get("profile_model_fake_probability", ""),
                }
            )
        return rows, None
    except PyMongoError as exc:
        return [], str(exc)


def fetch_profile_reports_export_json(limit: int = 10000) -> tuple[list[dict], str | None]:
    try:
        collection = _get_collection()
        query = {"analysis_type": "profile"}
        cursor = collection.find(query).sort("created_at", -1).limit(max(1, min(limit, 100000)))
        rows: list[dict] = []
        for doc in cursor:
            profile = doc.get("input_profile", {})
            feedback = doc.get("feedback", {})
            rows.append(
                {
                    "report_id": str(doc.get("_id")),
                    "analysis_type": doc.get("analysis_type", "profile"),
                    "owner_username": doc.get("owner_username", ""),
                    "owner_role": doc.get("owner_role", ""),
                    "created_at": _to_iso(doc.get("created_at")) or "",
                    "username": str(profile.get("username", "")),
                    "bio": str(profile.get("bio", "")),
                    "followers": profile.get("followers", ""),
                    "following": profile.get("following", ""),
                    "posts": profile.get("posts", ""),
                    "engagement_rate": profile.get("engagement_rate", ""),
                    "image_score": profile.get("image_score", ""),
                    "trust_score": doc.get("trust_score", ""),
                    "risk_level": doc.get("risk_level", ""),
                    "recommendation": doc.get("recommendation", ""),
                    "component_scores": doc.get("component_scores", {}),
                    "confidence_breakdown": doc.get("confidence_breakdown", {}),
                    "risk_factors": doc.get("risk_factors", []),
                    "profile_model_source": doc.get("profile_model_source", ""),
                    "profile_model_weight": doc.get("profile_model_weight", ""),
                    "profile_heuristic_weight": doc.get("profile_heuristic_weight", ""),
                    "profile_heuristic_score": doc.get("profile_heuristic_score", ""),
                    "profile_model_score": doc.get("profile_model_score", ""),
                    "profile_model_fake_probability": doc.get("profile_model_fake_probability", ""),
                    "feedback": {
                        "label": feedback.get("label", ""),
                        "note": feedback.get("note", ""),
                        "updated_at": _to_iso(feedback.get("updated_at")) or "",
                    },
                }
            )
        return rows, None
    except PyMongoError as exc:
        return [], str(exc)


def upsert_auth_user(
    username: str,
    role: str,
    password_hash: str,
    seeded_from_env: bool = False,
) -> tuple[bool, str | None]:
    try:
        collection = _get_users_collection()
        collection.update_one(
            {"username": username},
            {
                "$set": {
                    "username": username,
                    "role": role,
                    "password_hash": password_hash,
                    "seeded_from_env": seeded_from_env,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
        return True, None
    except PyMongoError as exc:
        return False, str(exc)


def fetch_auth_user(username: str) -> tuple[dict | None, str | None]:
    try:
        collection = _get_users_collection()
        doc = collection.find_one({"username": username})
        if not doc:
            return None, None
        return {
            "id": str(doc.get("_id")),
            "username": str(doc.get("username", "")),
            "role": str(doc.get("role", "")),
            "password_hash": str(doc.get("password_hash", "")),
            "seeded_from_env": bool(doc.get("seeded_from_env", False)),
            "created_at": _to_iso(doc.get("created_at")),
            "updated_at": _to_iso(doc.get("updated_at")),
        }, None
    except PyMongoError as exc:
        return None, str(exc)


def count_auth_users() -> tuple[dict, str | None]:
    try:
        collection = _get_users_collection()
        total = int(collection.count_documents({}))
        admin_count = int(collection.count_documents({"role": "admin"}))
        user_count = int(collection.count_documents({"role": "user"}))
        return {
            "total_users": total,
            "admin_count": admin_count,
            "user_count": user_count,
        }, None
    except PyMongoError as exc:
        return {"total_users": 0, "admin_count": 0, "user_count": 0}, str(exc)


def save_audit_log(event: dict) -> tuple[bool, str | None]:
    try:
        collection = _get_audit_collection()
        payload = {
            "actor_username": event.get("actor_username", ""),
            "actor_role": event.get("actor_role", ""),
            "action": event.get("action", ""),
            "status": event.get("status", "info"),
            "endpoint": event.get("endpoint", ""),
            "method": event.get("method", ""),
            "source_ip": event.get("source_ip", ""),
            "target": event.get("target", ""),
            "detail": event.get("detail", ""),
            "created_at": datetime.now(timezone.utc),
        }
        collection.insert_one(payload)
        return True, None
    except PyMongoError as exc:
        return False, str(exc)


def fetch_audit_logs(limit: int = 50, action: str | None = None, actor_username: str | None = None) -> tuple[list[dict], str | None]:
    try:
        collection = _get_audit_collection()
        safe_limit = max(1, min(limit, 200))
        query: dict = {}
        if action:
            query["action"] = {"$regex": action, "$options": "i"}
        if actor_username:
            query["actor_username"] = {"$regex": actor_username, "$options": "i"}

        cursor = collection.find(query).sort("created_at", -1).limit(safe_limit)
        rows: list[dict] = []
        for doc in cursor:
            rows.append(
                {
                    "id": str(doc.get("_id")),
                    "actor_username": doc.get("actor_username", ""),
                    "actor_role": doc.get("actor_role", ""),
                    "action": doc.get("action", ""),
                    "status": doc.get("status", ""),
                    "endpoint": doc.get("endpoint", ""),
                    "method": doc.get("method", ""),
                    "source_ip": doc.get("source_ip", ""),
                    "target": doc.get("target", ""),
                    "detail": doc.get("detail", ""),
                    "created_at": _to_iso(doc.get("created_at")),
                }
            )
        return rows, None
    except PyMongoError as exc:
        return [], str(exc)
