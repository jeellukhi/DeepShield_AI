from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from app.services.image_features import extract_image_feature, is_usable_image
from app.services.model_thresholds import get_effective_thresholds

# ── Artifact cache (pickle file) ────────────────────────────────────────────
_cached_artifact: dict | None = None
_cached_mtime: float = -1.0

# ── PyTorch model cache (loaded once, reused for every image/frame) ──────────
# Reloading EfficientNet for every frame adds ~0.1s x 64 frames = 6+ sec wasted
_torch_model_cache: dict | None = None   # {"model": model, "device": device, "transform": transform}


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
        # Invalidate torch model cache when pkl changes
        global _torch_model_cache
        _torch_model_cache = None
        return artifact
    except Exception:
        return None


def _get_torch_model(artifact: dict):
    """Load EfficientNet once and cache it globally. Returns (model, device, transform) or None."""
    global _torch_model_cache
    if _torch_model_cache is not None:
        return _torch_model_cache

    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        import torch.nn as nn

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

        _torch_model_cache = {"model": model, "device": device, "transform": transform}
        return _torch_model_cache

    except Exception:
        return None


def _detect_face_region(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Use Haar cascade to detect and crop the main face.
    Returns cropped face (BGR) or None if no face found.
    The CNN was trained on cropped face images — feeding it a face crop gives
    the most reliable result.
    """
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # 20% padding around face
        pad_x = int(w * 0.20)
        pad_y = int(h * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_bgr.shape[1], x + w + pad_x)
        y2 = min(image_bgr.shape[0], y + h + pad_y)
        crop = image_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
    except Exception:
        return None


def _run_cnn_on_image(image_bgr: np.ndarray, artifact: dict) -> float | None:
    """
    Run EfficientNet on image_bgr. Returns raw sigmoid output (0-1) or None.
    Uses the global cached model — does NOT reload weights.
    """
    cache = _get_torch_model(artifact)
    if cache is None:
        return None
    try:
        import torch
        from PIL import Image as PILImage

        model = cache["model"]
        device = cache["device"]
        transform = cache["transform"]

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(tensor).squeeze()
            raw_prob = float(torch.sigmoid(logit).cpu().item())

        return raw_prob

    except Exception:
        return None


def _predict_with_cnn(image_bgr: np.ndarray, artifact: dict) -> dict | None:
    """
    Smart CNN prediction:
    1. Try face crop first — CNN was trained on cropped faces
    2. If no face found, also run CNN on full image as fallback
    3. Use CNN's stored threshold (0.31) for labeling — NOT admin threshold
    Returns dict with fake_probability, confidence, face_detected, or None.
    """
    cnn_threshold = float(artifact.get("decision_threshold", 0.5))

    face_crop = _detect_face_region(image_bgr)
    face_detected = face_crop is not None

    if face_detected:
        raw = _run_cnn_on_image(face_crop, artifact)
    else:
        raw = _run_cnn_on_image(image_bgr, artifact)

    if raw is None:
        return None

    # Use the CNN's OWN trained threshold for this prediction
    # (e.g. threshold=0.31 means: raw > 0.31 = fake, raw < 0.31 = real)
    fake_probability = round(raw * 100.0, 2)

    # Confidence: how far is the raw output from the decision boundary?
    # Near the threshold = low confidence; far from it = high confidence
    distance_from_threshold = abs(raw - cnn_threshold)
    confidence = round(min(100.0, distance_from_threshold / (1.0 - cnn_threshold) * 100.0), 1)

    return {
        "raw_prob": raw,
        "fake_probability": fake_probability,
        "cnn_threshold": cnn_threshold,
        "face_detected": face_detected,
        "confidence": confidence,
    }


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


def analyze_image_array(image: np.ndarray) -> dict:
    """Heuristic-only analysis (blur + edge + noise). Always available as fallback."""
    if not is_usable_image(image, min_side=32):
        return {"authenticity_score": 50.0, "fake_probability": 50.0, "label": "Uncertain", "signals": {}, "model_source": "heuristic_only"}

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
    auth = round(max(0.0, min(100.0, raw * 100.0)), 2)

    return {
        "authenticity_score": auth,
        "fake_probability": round(100.0 - auth, 2),
        "label": "Real-like" if auth >= 50 else "Fake-like",
        "signals": {"lap_var": round(lap_var, 2), "edge_density": round(edge_density, 4), "noise_std": round(noise_std, 4)},
        "model_source": "heuristic_only",
    }


def analyze_image_frame(frame: np.ndarray) -> dict:
    """Per-frame analysis for video detector. Uses cached CNN model."""
    artifact = _load_model_artifact()
    if artifact is None:
        return analyze_image_array(frame)

    arch = artifact.get("model_architecture", "")
    model_type = artifact.get("model_type", "")
    is_cnn = arch == "efficientnet_b0" or "efficientnet" in model_type

    if is_cnn:
        cnn_result = _predict_with_cnn(frame, artifact)
        if cnn_result is not None:
            raw = cnn_result["raw_prob"]
            cnn_threshold = cnn_result["cnn_threshold"]
            fake_prob = cnn_result["fake_probability"]
            auth = round(100.0 - fake_prob, 2)
            label = "Fake-like" if raw >= cnn_threshold else "Real-like"
            return {
                "authenticity_score": auth,
                "fake_probability": fake_prob,
                "label": label,
                "model_source": "cnn_efficientnet_b0",
                "face_detected": cnn_result["face_detected"],
            }

    # Sklearn fallback
    sklearn_prob = _predict_with_sklearn(frame, artifact)
    if sklearn_prob is not None:
        auth = round(100.0 - sklearn_prob, 2)
        return {"authenticity_score": auth, "fake_probability": sklearn_prob, "label": "Fake-like" if sklearn_prob >= 50 else "Real-like", "model_source": "sklearn"}

    return analyze_image_array(frame)


def analyze_image_authenticity(image_bytes: bytes) -> dict:
    """Main entry point for the image analysis API."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not is_usable_image(image, min_side=32):
        return {"authenticity_score": 50.0, "fake_probability": 50.0, "label": "Uncertain", "signals": {"error": "Could not decode image or image too small"}, "model_source": "error"}

    artifact = _load_model_artifact()
    warning_flags: list[str] = []

    # ── No model available: heuristic only ──
    if artifact is None:
        result = analyze_image_array(image)
        return {**result, "model_source": "heuristic_fallback", "warning_flags": ["No trained model found — using texture heuristics only."]}

    arch = artifact.get("model_architecture", "")
    model_type = artifact.get("model_type", "")
    is_cnn = arch == "efficientnet_b0" or "efficientnet" in model_type

    # ── CNN path ──────────────────────────────────────────────────────────────
    if is_cnn:
        cnn_result = _predict_with_cnn(image, artifact)

        if cnn_result is None:
            # CNN failed completely — fall back to heuristic
            h = analyze_image_array(image)
            return {**h, "model_source": "heuristic_fallback", "warning_flags": ["CNN inference failed — using texture heuristics."]}

        raw = cnn_result["raw_prob"]
        cnn_threshold = cnn_result["cnn_threshold"]
        face_detected = cnn_result["face_detected"]
        confidence = cnn_result["confidence"]

        # Use CNN's own trained threshold for the label
        fake_probability = round(raw * 100.0, 2)
        auth_score = round(100.0 - fake_probability, 2)
        label = "Fake-like" if raw >= cnn_threshold else "Real-like"

        if not face_detected:
            warning_flags.append("No face detected — CNN accuracy may be reduced for non-face images.")
        if confidence < 30:
            warning_flags.append("Low confidence prediction — result is near the decision boundary.")

        return {
            "authenticity_score": auth_score,
            "fake_probability": fake_probability,
            "label": label,
            "model_source": "cnn_efficientnet_b0",
            "face_detected": face_detected,
            "confidence": confidence,
            "calibration": {
                "raw_cnn_prob": round(raw, 4),
                "cnn_threshold": cnn_threshold,
            },
            "warning_flags": warning_flags,
        }

    # ── Sklearn path ──────────────────────────────────────────────────────────
    sklearn_prob = _predict_with_sklearn(image, artifact)
    if sklearn_prob is not None:
        auth = round(100.0 - sklearn_prob, 2)
        threshold_cfg = get_effective_thresholds()
        threshold = float(threshold_cfg.get("values", {}).get("image_fake_probability_threshold", 50.0))
        label = "Fake-like" if sklearn_prob >= threshold else "Real-like"
        return {"authenticity_score": auth, "fake_probability": round(sklearn_prob, 2), "label": label, "model_source": "sklearn_ensemble", "warning_flags": warning_flags}

    # ── Final fallback ────────────────────────────────────────────────────────
    h = analyze_image_array(image)
    return {**h, "model_source": "heuristic_fallback", "warning_flags": warning_flags}
