from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from app.services.profile_text_features import build_profile_text
from app.services.model_thresholds import get_effective_thresholds


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _profile_model_path() -> Path:
    return _project_root() / "ml" / "models" / "profile_nlp.pkl"


def predict_profile_text_authenticity(username: str, bio: str) -> dict | None:
    model_path = _profile_model_path()
    if not model_path.exists():
        return None

    try:
        with model_path.open("rb") as f:
            artifact = pickle.load(f)
    except Exception:
        return None

    model = artifact.get("model")
    if model is None:
        return None
    quality_gate = artifact.get("quality_gate", {})
    if not quality_gate:
        return None
    if not quality_gate.get("passed", False):
        return None

    try:
        text = np.asarray([build_profile_text(username, bio)], dtype=object)
        if hasattr(model, "predict_proba"):
            fake_probability = float(model.predict_proba(text)[0][1]) * 100.0
        else:
            fake_probability = float(model.predict(text)[0]) * 100.0
    except Exception:
        return None

    raw_fake_probability = max(0.0, min(100.0, fake_probability))
    metrics = artifact.get("metrics", {})
    f1_score = float(metrics.get("f1_score", 60.0))
    class_counts = artifact.get("class_counts", {})
    total_samples = int(class_counts.get("confirmed_real", 0)) + int(class_counts.get("confirmed_fake", 0))
    count_real = max(1, int(class_counts.get("confirmed_real", 0)))
    count_fake = max(1, int(class_counts.get("confirmed_fake", 0)))
    balance_ratio = min(count_real, count_fake) / max(count_real, count_fake)

    f1_reliability = max(0.0, min(1.0, (f1_score - 60.0) / 40.0))
    size_reliability = max(0.0, min(1.0, total_samples / 200.0))
    reliability = max(0.0, min(1.0, (0.60 * f1_reliability) + (0.25 * size_reliability) + (0.15 * balance_ratio)))
    recommended_weight = round(0.35 + (0.45 * reliability), 2)
    decision_threshold = float(artifact.get("decision_threshold", 0.5))
    artifact_threshold_pct = round(decision_threshold * 100, 2)
    threshold_cfg = get_effective_thresholds()
    configured_threshold_pct = float(
        threshold_cfg.get("values", {}).get("profile_fake_probability_threshold", artifact_threshold_pct)
    )
    configured_threshold_pct = max(20.0, min(80.0, configured_threshold_pct))
    calibrated_fake_probability = max(
        0.0,
        min(100.0, raw_fake_probability + (50.0 - configured_threshold_pct)),
    )
    profile_score = max(0.0, min(100.0, 100.0 - calibrated_fake_probability))

    return {
        "profile_score": round(profile_score, 2),
        "fake_probability": round(calibrated_fake_probability, 2),
        "raw_fake_probability": round(raw_fake_probability, 2),
        "model_type": artifact.get("model_type", "profile_nlp_model"),
        "model_source": "trained_profile_model",
        "f1_score": round(f1_score, 2),
        "recommended_weight": recommended_weight,
        "decision_threshold": round(configured_threshold_pct, 2),
        "training_samples": total_samples,
    }
