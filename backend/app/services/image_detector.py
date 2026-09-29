import pickle
from pathlib import Path

import cv2
import numpy as np

from app.services.image_features import extract_image_feature
from app.services.model_thresholds import get_effective_thresholds

_MODEL_CACHE: dict | None = None
_MODEL_CACHE_MTIME: float | None = None


def _normalize(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    ratio = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, ratio))


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _trained_model_path() -> Path:
    return _project_root() / "ml" / "models" / "image_cnn.pkl"


def _load_model_artifact() -> dict | None:
    global _MODEL_CACHE, _MODEL_CACHE_MTIME
    model_path = _trained_model_path()
    if not model_path.exists():
        _MODEL_CACHE = None
        _MODEL_CACHE_MTIME = None
        return None

    try:
        mtime = model_path.stat().st_mtime
    except Exception:
        return None

    if _MODEL_CACHE is not None and _MODEL_CACHE_MTIME == mtime:
        return _MODEL_CACHE

    try:
        with model_path.open("rb") as f:
            artifact = pickle.load(f)
    except Exception:
        return None

    _MODEL_CACHE = artifact
    _MODEL_CACHE_MTIME = mtime
    return artifact


def _predict_with_trained_model(image: np.ndarray) -> dict | None:
    artifact = _load_model_artifact()
    if artifact is None:
        return None

    model = artifact.get("model")
    model_cal = artifact.get("model_calibration")  # new calibration ensemble partner
    input_size = int(artifact.get("input_size", 64))
    feature_mode = str(artifact.get("feature_mode", "v3_multicue_stack"))
    raw_artifact_threshold = float(artifact.get("decision_threshold", 0.5))
    artifact_threshold_pct = raw_artifact_threshold * 100.0 if raw_artifact_threshold <= 1.0 else raw_artifact_threshold
    threshold_cfg = get_effective_thresholds()
    decision_threshold_pct = float(
        threshold_cfg.get("values", {}).get("image_fake_probability_threshold", artifact_threshold_pct)
    )
    decision_threshold = max(0.2, min(0.8, decision_threshold_pct / 100.0))
    if model is None:
        return None

    try:
        x = extract_image_feature(image, input_size, feature_mode=feature_mode).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            prob_main = float(model.predict_proba(x)[0][1])
        else:
            prob_main = float(model.predict(x)[0])

        # Blend with calibration model if available
        if model_cal is not None and hasattr(model_cal, "predict_proba"):
            weights = artifact.get("ensemble_weights", {"main": 0.75, "calibration": 0.25})
            prob_cal = float(model_cal.predict_proba(x)[0][1])
            fake_prob_raw = (
                weights.get("main", 0.75) * prob_main
                + weights.get("calibration", 0.25) * prob_cal
            )
        else:
            fake_prob_raw = prob_main

        fake_probability = round(max(0.0, min(100.0, fake_prob_raw * 100.0)), 2)
    except Exception:
        return None

    authenticity_score = round(max(0.0, min(100.0, 100.0 - fake_probability)), 2)
    label = "Real-like" if fake_probability < (decision_threshold * 100.0) else "Fake-like"

    return {
        "authenticity_score": authenticity_score,
        "fake_probability": fake_probability,
        "label": label,
        "signals": {
            "model_type": artifact.get("model_type", "trained_model"),
            "input_size": input_size,
            "feature_mode": feature_mode,
            "decision_threshold": round(decision_threshold * 100.0, 2),
        },
        "model_source": "trained_image_model",
    }


def analyze_image_frame(image: np.ndarray) -> dict:
    model_result = _predict_with_trained_model(image)
    if model_result is not None:
        return model_result
    return analyze_image_array(image)


def analyze_image_array(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    noise = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    noise_std = float(np.std(noise))

    blur_score = _normalize(lap_var, 40.0, 450.0)
    texture_score = _normalize(edge_density, 0.02, 0.22)
    noise_score = _normalize(noise_std, 2.0, 20.0)

    authenticity_score = round((0.4 * blur_score + 0.35 * texture_score + 0.25 * noise_score) * 100, 2)
    label = "Real-like" if authenticity_score >= 50 else "Fake-like"
    fake_probability = round(100 - authenticity_score, 2)

    return {
        "authenticity_score": authenticity_score,
        "fake_probability": fake_probability,
        "label": label,
        "signals": {
            "laplacian_variance": round(lap_var, 2),
            "edge_density": round(edge_density, 4),
            "noise_std": round(noise_std, 2),
        },
        "model_source": "heuristic",
    }


def analyze_image_authenticity(file_bytes: bytes) -> dict:
    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file.")
    model_result = analyze_image_frame(image)
    if model_result.get("model_source") != "trained_image_model":
        return model_result

    heuristic_result = analyze_image_array(image)
    model_score = float(model_result.get("authenticity_score", 50.0))
    heuristic_score = float(heuristic_result.get("authenticity_score", 50.0))

    fused_score = (0.72 * model_score) + (0.28 * heuristic_score)
    disagreement_penalty = max(0.0, (model_score - heuristic_score - 10.0) * 0.40)
    heuristic_suspicion_penalty = max(0.0, (50.0 - heuristic_score) * 0.35)
    final_score = max(0.0, min(100.0, fused_score - disagreement_penalty - heuristic_suspicion_penalty))

    threshold_pct = float(model_result.get("signals", {}).get("decision_threshold", 50.0))
    base_fake_probability = max(0.0, min(100.0, 100.0 - final_score))
    uncertainty_boost = max(0.0, (disagreement_penalty * 0.12) + (heuristic_suspicion_penalty * 0.08))
    fake_probability = round(max(0.0, min(100.0, base_fake_probability + uncertainty_boost)), 2)
    label = "Fake-like" if fake_probability >= threshold_pct else "Real-like"

    warning_flags: list[str] = []
    if disagreement_penalty > 0:
        warning_flags.append("Model score and heuristic score disagree.")
    if heuristic_suspicion_penalty > 0:
        warning_flags.append("Heuristic texture quality appears suspicious.")

    return {
        **model_result,
        "authenticity_score": round(final_score, 2),
        "fake_probability": fake_probability,
        "label": label,
        "raw_fake_probability": round(max(0.0, min(100.0, 100.0 - model_score)), 2),
        "base_fake_probability": round(base_fake_probability, 2),
        "model_source": "trained_image_model+heuristic_fusion",
        "warning_flags": warning_flags,
        "signals": {
            **model_result.get("signals", {}),
            "heuristic_authenticity_score": round(heuristic_score, 2),
            "fusion": {
                "model_weight": 0.72,
                "heuristic_weight": 0.28,
                "disagreement_penalty": round(disagreement_penalty, 2),
                "heuristic_suspicion_penalty": round(heuristic_suspicion_penalty, 2),
                "uncertainty_boost": round(uncertainty_boost, 2),
            },
        },
    }
