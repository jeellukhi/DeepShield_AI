from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from app.services.profile_text_features import build_profile_text
from app.services.profile_numeric_features import extract_numeric_features
from app.services.model_thresholds import get_effective_thresholds


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _profile_model_path() -> Path:
    return _project_root() / "ml" / "models" / "profile_nlp.pkl"


def _load_artifact() -> dict | None:
    model_path = _profile_model_path()
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_profile_text_authenticity(
    username: str,
    bio: str,
    followers: int | float = 0,
    following: int | float = 0,
    posts: int | float = 0,
    engagement_rate: float = 0.0,
    image_score: float = 50.0,
) -> dict | None:
    artifact = _load_artifact()
    if artifact is None:
        return None

    # --- Quality gate ---
    quality_gate = artifact.get("quality_gate", {})
    if quality_gate and not quality_gate.get("passed", True):
        return None

    try:
        text_input = np.asarray([build_profile_text(username, bio)], dtype=object)
        num_input = np.asarray(
            [extract_numeric_features(followers, following, posts, engagement_rate, image_score)],
            dtype=np.float32,
        )

        # --- New ensemble model (text + rf_numeric + gb_numeric) ---
        model_text = artifact.get("model_text")
        model_rf = artifact.get("model_rf_numeric")
        model_gb = artifact.get("model_gb_numeric")
        weights = artifact.get("ensemble_weights", {"text": 0.30, "rf_numeric": 0.40, "gb_numeric": 0.30})

        if model_text is not None and model_rf is not None and model_gb is not None:
            p_text = float(model_text.predict_proba(text_input)[0][1])
            p_rf = float(model_rf.predict_proba(num_input)[0][1])
            p_gb = float(model_gb.predict_proba(num_input)[0][1])
            raw_prob = (
                weights.get("text", 0.30) * p_text
                + weights.get("rf_numeric", 0.40) * p_rf
                + weights.get("gb_numeric", 0.30) * p_gb
            )
            model_type = "ensemble_text+rf+gb"

        # --- Legacy single model fallback ---
        elif artifact.get("model") is not None:
            model = artifact["model"]
            raw_prob = float(model.predict_proba(text_input)[0][1])
            model_type = str(artifact.get("model_type", "legacy"))

        else:
            return None

        raw_fake_probability = max(0.0, min(100.0, raw_prob * 100.0))

    except Exception:
        return None

    # --- Compute reliability / weight ---
    metrics = artifact.get("metrics", {})
    f1_score_val = float(metrics.get("f1_score", 60.0))
    class_counts = artifact.get("class_counts", {})
    total_samples = int(artifact.get("total_samples", 0)) or (
        int(class_counts.get("confirmed_real", 0)) + int(class_counts.get("confirmed_fake", 0))
    )
    count_real = max(1, int(class_counts.get("confirmed_real", 0)))
    count_fake = max(1, int(class_counts.get("confirmed_fake", 0)))
    balance_ratio = min(count_real, count_fake) / max(count_real, count_fake)

    f1_reliability = max(0.0, min(1.0, (f1_score_val - 60.0) / 40.0))
    size_reliability = max(0.0, min(1.0, total_samples / 300.0))
    reliability = max(0.0, min(1.0, 0.55 * f1_reliability + 0.25 * size_reliability + 0.20 * balance_ratio))
    recommended_weight = round(0.40 + (0.45 * reliability), 2)

    # --- Apply configured threshold ---
    decision_threshold = float(artifact.get("decision_threshold", 0.50))
    artifact_threshold_pct = round(
        decision_threshold * 100 if decision_threshold <= 1.0 else decision_threshold, 2
    )
    threshold_cfg = get_effective_thresholds()
    configured_threshold_pct = float(
        threshold_cfg.get("values", {}).get("profile_fake_probability_threshold", artifact_threshold_pct)
    )
    configured_threshold_pct = max(20.0, min(80.0, configured_threshold_pct))

    calibrated_fake_probability = max(
        0.0, min(100.0, raw_fake_probability + (50.0 - configured_threshold_pct))
    )
    profile_score = max(0.0, min(100.0, 100.0 - calibrated_fake_probability))

    return {
        "profile_score": round(profile_score, 2),
        "fake_probability": round(calibrated_fake_probability, 2),
        "raw_fake_probability": round(raw_fake_probability, 2),
        "model_type": model_type,
        "model_source": "trained_profile_model",
        "f1_score": round(f1_score_val, 2),
        "recommended_weight": recommended_weight,
        "decision_threshold": round(configured_threshold_pct, 2),
        "training_samples": total_samples,
    }
