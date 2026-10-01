from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from app.services.image_features import extract_image_feature, is_usable_image
from app.services.model_thresholds import get_effective_thresholds

_cached_artifact: dict | None = None
_cached_mtime: float = -1.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _model_path() -> Path:
    return _project_root() / "ml" / "models" / "image_cnn.pkl"


def _load_model_artifact() -> dict | None:
    global _cached_artifact, _cached_mtime
    path = _model_path()
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _cached_artifact is not None and abs(mtime - _cached_mtime) < 0.5:
        return _cached_artifact
    try:
        with path.open("rb") as f:
            artifact = pickle.load(f)
        _cached_artifact = artifact
        _cached_mtime = mtime
        return artifact
    except Exception:
        return None


def _predict_with_cnn(image: np.ndarray, artifact: dict) -> float | None:
    """Run EfficientNet CNN inference. Returns fake_probability 0-100 or None."""
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        import torch.nn as nn
        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )
        model.load_state_dict(artifact["model_state_dict"])
        model = model.to(device)
        model.eval()

        mean = artifact.get("normalize_mean", [0.485, 0.456, 0.406])
        std = artifact.get("normalize_std", [0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(tensor).squeeze()
            raw_prob = float(torch.sigmoid(logit).cpu().item())

        # --- Recalibrate to the model's own trained threshold ---
        # The CNN was trained with threshold T (e.g. 0.31).
        # We recalibrate so that T maps to 0.50 (the natural midpoint),
        # preserving the relative ordering of all predictions.
        # Formula: recalibrated = raw_prob / (2 * T)  → capped at 1.0
        # Effect: raw_prob=T → 0.50 | raw_prob=0 → 0.0 | raw_prob=2T → 1.0
        cnn_threshold = float(artifact.get("decision_threshold", 0.5))
        if cnn_threshold > 0.0 and cnn_threshold != 0.5:
            recalibrated = min(1.0, raw_prob / (2.0 * cnn_threshold))
        else:
            recalibrated = raw_prob

        return recalibrated * 100.0

    except Exception:
        return None


def _predict_with_sklearn(image: np.ndarray, artifact: dict) -> float | None:
    """Run sklearn ensemble inference. Returns fake_probability 0-100 or None."""
    try:
        model = artifact.get("model")
        model_cal = artifact.get("model_calibration")
        input_size = int(artifact.get("input_size", 64))
        feature_mode = str(artifact.get("feature_mode", "v4_lbp_ela"))

        if model is None:
            return None

        x = extract_image_feature(image, input_size, feature_mode=feature_mode).reshape(1, -1)

        if hasattr(model, "predict_proba"):
            prob_main = float(model.predict_proba(x)[0][1])
        else:
            prob_main = float(model.predict(x)[0])

        if model_cal is not None and hasattr(model_cal, "predict_proba"):
            weights = artifact.get("ensemble_weights", {"main": 0.80, "calibration": 0.20})
            prob_cal = float(model_cal.predict_proba(x)[0][1])
            fake_prob_raw = weights.get("main", 0.80) * prob_main + weights.get("calibration", 0.20) * prob_cal
        else:
            fake_prob_raw = prob_main

        return max(0.0, min(100.0, fake_prob_raw * 100.0))

    except Exception:
        return None


def _predict_with_trained_model(image: np.ndarray) -> dict | None:
    artifact = _load_model_artifact()
    if artifact is None:
        return None

    arch = artifact.get("model_architecture", "")
    model_type = artifact.get("model_type", "")
    is_cnn = arch == "efficientnet_b0" or "efficientnet" in model_type

    # Threshold for final label
    threshold_cfg = get_effective_thresholds()
    decision_threshold_pct = float(
        threshold_cfg.get("values", {}).get("image_fake_probability_threshold", 50.0)
    )
    decision_threshold_pct = max(20.0, min(80.0, decision_threshold_pct))

    if is_cnn:
        fake_probability = _predict_with_cnn(image, artifact)
        source = "cnn_efficientnet_b0"
    else:
        fake_probability = _predict_with_sklearn(image, artifact)
        source = "sklearn_ensemble"

    if fake_probability is None:
        return None

    fake_probability = round(max(0.0, min(100.0, fake_probability)), 2)
    authenticity_score = round(100.0 - fake_probability, 2)
    label = "Fake-like" if fake_probability >= decision_threshold_pct else "Real-like"

    return {
        "authenticity_score": authenticity_score,
        "fake_probability": fake_probability,
        "label": label,
        "signals": {
            "model_type": model_type or source,
            "model_source": source,
            "decision_threshold": round(decision_threshold_pct, 2),
        },
        "model_source": source,
    }


def analyze_image_array(image: np.ndarray) -> dict:
    """Heuristic-only analysis (blur + edge density + noise). Always available."""
    if not is_usable_image(image, min_side=32):
        return {
            "authenticity_score": 50.0,
            "fake_probability": 50.0,
            "label": "Uncertain",
            "signals": {},
            "model_source": "heuristic_only",
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    gray_f = gray.astype(np.float32) / 255.0
    noise_map = gray_f - cv2.GaussianBlur(gray_f, (5, 5), 0)
    noise_std = float(np.std(noise_map))

    blur_score = min(1.0, lap_var / 350.0)
    edge_score = min(1.0, edge_density / 0.18)
    noise_score = min(1.0, noise_std / 0.12)
    raw = 0.45 * blur_score + 0.30 * edge_score + 0.25 * noise_score
    authenticity_score = round(max(0.0, min(100.0, raw * 100.0)), 2)
    fake_probability = round(100.0 - authenticity_score, 2)

    return {
        "authenticity_score": authenticity_score,
        "fake_probability": fake_probability,
        "label": "Real-like" if fake_probability < 50 else "Fake-like",
        "signals": {
            "lap_var": round(lap_var, 2),
            "edge_density": round(edge_density, 4),
            "noise_std": round(noise_std, 4),
        },
        "model_source": "heuristic_only",
    }


def analyze_image_frame(frame: np.ndarray) -> dict:
    """Used by video detector for per-frame analysis."""
    result = _predict_with_trained_model(frame)
    if result:
        return result
    return analyze_image_array(frame)


def analyze_image_authenticity(image_bytes: bytes) -> dict:
    """Main entry point for image analysis API."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not is_usable_image(image, min_side=32):
        return {
            "authenticity_score": 50.0,
            "fake_probability": 50.0,
            "label": "Uncertain",
            "signals": {"error": "Could not decode image or image too small"},
            "model_source": "error",
        }

    model_result = _predict_with_trained_model(image)
    heuristic_result = analyze_image_array(image)

    if model_result is None:
        return {**heuristic_result, "model_source": "heuristic_fallback"}

    model_fake = float(model_result["fake_probability"])
    heuristic_fake = float(heuristic_result["fake_probability"])
    model_source = model_result.get("model_source", "trained")

    # CNN gets 85% weight — it learned from 10k face images
    # Heuristic gets 15% — only a sanity check
    if "cnn" in model_source:
        weight_model = 0.85
    else:
        weight_model = 0.72

    fused_fake = round(weight_model * model_fake + (1.0 - weight_model) * heuristic_fake, 2)
    fused_score = round(100.0 - fused_fake, 2)

    threshold_cfg = get_effective_thresholds()
    threshold = float(threshold_cfg.get("values", {}).get("image_fake_probability_threshold", 50.0))
    label = "Fake-like" if fused_fake >= threshold else "Real-like"

    warning_flags = []
    if abs(model_fake - heuristic_fake) > 35:
        warning_flags.append("Model and heuristic scores strongly disagree — result may be uncertain.")

    return {
        "authenticity_score": fused_score,
        "fake_probability": fused_fake,
        "label": label,
        "model_source": f"{model_source}+heuristic_fusion",
        "calibration": {
            "model_fake_prob": round(model_fake, 2),
            "heuristic_fake_prob": round(heuristic_fake, 2),
            "model_weight": weight_model,
        },
        "signals": {**model_result.get("signals", {}), **heuristic_result.get("signals", {})},
        "warning_flags": warning_flags,
    }
