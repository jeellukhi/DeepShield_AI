from __future__ import annotations
import pickle
from pathlib import Path
import cv2
import numpy as np
from app.services.image_features import extract_image_feature, is_usable_image
from app.services.model_thresholds import get_effective_thresholds

_cached_artifact: dict | None = None
_cached_mtime: float = -1.0
_torch_model_cache: dict | None = None


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
        global _torch_model_cache
        _torch_model_cache = None
        return artifact
    except Exception:
        return None

def _get_torch_model(artifact: dict):
    global _torch_model_cache
    if _torch_model_cache is not None:
        return _torch_model_cache
    try:
        import torch, torchvision.models as models, torchvision.transforms as transforms, torch.nn as nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_features, 256),
            nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, 1),
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
    """Detect and return cropped face region, or None if no face found."""
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x, pad_y = int(w * 0.20), int(h * 0.20)
        x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
        x2 = min(image_bgr.shape[1], x + w + pad_x)
        y2 = min(image_bgr.shape[0], y + h + pad_y)
        crop = image_bgr[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
    except Exception:
        return None

def _run_cnn_on_crop(face_crop: np.ndarray, artifact: dict) -> float | None:
    """Run CNN ONLY on a face crop. Returns raw sigmoid output 0-1 or None."""
    cache = _get_torch_model(artifact)
    if cache is None:
        return None
    try:
        import torch
        from PIL import Image as PILImage
        model, device, transform = cache["model"], cache["device"], cache["transform"]
        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = transform(PILImage.fromarray(img_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = float(torch.sigmoid(model(tensor).squeeze()).cpu().item())
        return raw
    except Exception:
        return None

def analyze_image_array(image: np.ndarray) -> dict:
    """Heuristic-only analysis. Works on any image (fallback when no face)."""
    if not is_usable_image(image, min_side=32):
        return {"authenticity_score": 50.0, "fake_probability": 50.0,
                "label": "Uncertain", "signals": {}, "model_source": "heuristic_only"}
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
    return {"authenticity_score": auth, "fake_probability": round(100.0 - auth, 2),
            "label": "Real-like" if auth >= 50 else "Fake-like",
            "signals": {"lap_var": round(lap_var, 2), "edge_density": round(edge_density, 4),
                        "noise_std": round(noise_std, 4)},
            "model_source": "heuristic_only"}

def analyze_image_frame(frame: np.ndarray) -> dict:
    """Per-frame analysis for video detector. Uses cached model."""
    artifact = _load_model_artifact()
    if artifact is None:
        return analyze_image_array(frame)

    arch = artifact.get("model_architecture", "")
    is_cnn = arch == "efficientnet_b0" or "efficientnet" in artifact.get("model_type", "")

    if is_cnn:
        face_crop = _detect_face_region(frame)
        if face_crop is not None:
            raw = _run_cnn_on_crop(face_crop, artifact)
            if raw is not None:
                cnn_thresh = float(artifact.get("decision_threshold", 0.31))
                fake_prob = round(raw * 100.0, 2)
                return {
                    "authenticity_score": round(100.0 - fake_prob, 2),
                    "fake_probability": fake_prob,
                    "label": "Fake-like" if raw >= cnn_thresh else "Real-like",
                    "model_source": "cnn_efficientnet_b0",
                    "face_detected": True,
                }
        # No face in this frame → heuristic (reliable for any image)
        h = analyze_image_array(frame)
        h["face_detected"] = False
        h["model_source"] = "heuristic_no_face"
        return h

    return analyze_image_array(frame)

def analyze_image_authenticity(image_bytes: bytes) -> dict:
    """Main API entry point for /analyze/image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not is_usable_image(image, min_side=32):
        return {"authenticity_score": 50.0, "fake_probability": 50.0, "label": "Uncertain",
                "signals": {"error": "Cannot decode image or image too small"}, "model_source": "error",
                "face_detected": False, "warning_flags": ["Image could not be read."]}

    artifact = _load_model_artifact()
    threshold_cfg = get_effective_thresholds()
    img_threshold = float(threshold_cfg.get("values", {}).get("image_fake_probability_threshold", 31.0))

    # ── Step 1: Try to detect face ────────────────────────────────────────────
    face_crop = _detect_face_region(image)
    face_detected = face_crop is not None

    # ── Step 2: CNN path (ONLY if face detected + CNN model available) ────────
    if face_detected and artifact is not None:
        arch = artifact.get("model_architecture", "")
        is_cnn = arch == "efficientnet_b0" or "efficientnet" in artifact.get("model_type", "")
        if is_cnn:
            raw = _run_cnn_on_crop(face_crop, artifact)
            if raw is not None:
                cnn_thresh = float(artifact.get("decision_threshold", 0.31))
                fake_probability = round(raw * 100.0, 2)
                auth_score = round(100.0 - fake_probability, 2)
                label = "Fake-like" if raw >= cnn_thresh else "Real-like"
                confidence = round(abs(raw - cnn_thresh) / max(cnn_thresh, 1 - cnn_thresh) * 100.0, 1)
                warning_flags = []
                if confidence < 20:
                    warning_flags.append("Low confidence — result is near the decision boundary.")
                return {
                    "authenticity_score": auth_score,
                    "fake_probability": fake_probability,
                    "label": label,
                    "model_source": "cnn_efficientnet_b0",
                    "face_detected": True,
                    "confidence": confidence,
                    "calibration": {"raw_cnn_prob": round(raw, 4), "cnn_threshold": cnn_thresh},
                    "warning_flags": warning_flags,
                }

    # ── Step 3: No face detected — return neutral + heuristic for reference ──
    if not face_detected:
        heuristic = analyze_image_array(image)
        return {
            "authenticity_score": 50.0,
            "fake_probability": 50.0,
            "label": "Uncertain",
            "model_source": "no_face_detected",
            "face_detected": False,
            "confidence": 0,
            "heuristic_reference": {
                "authenticity_score": heuristic["authenticity_score"],
                "fake_probability": heuristic["fake_probability"],
            },
            "warning_flags": [
                "No face detected in this image.",
                "The AI model is trained specifically on face images (AI-generated vs real faces).",
                "For non-face images, the result is Uncertain. Upload a clear face photo for accurate detection.",
            ],
        }

    # ── Step 4: Face detected but no CNN → sklearn fallback ──────────────────
    if artifact is not None:
        model = artifact.get("model")
        if model is not None:
            try:
                input_size = int(artifact.get("input_size", 64))
                feature_mode = str(artifact.get("feature_mode", "v4_lbp_ela"))
                x = extract_image_feature(face_crop, input_size, feature_mode=feature_mode).reshape(1, -1)
                prob = float(model.predict_proba(x)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(x)[0])
                fake_prob = round(prob * 100.0, 2)
                label = "Fake-like" if fake_prob >= img_threshold else "Real-like"
                return {"authenticity_score": round(100.0 - fake_prob, 2), "fake_probability": fake_prob,
                        "label": label, "model_source": "sklearn_ensemble", "face_detected": True,
                        "warning_flags": []}
            except Exception:
                pass

    return {**analyze_image_array(image), "face_detected": False,
            "warning_flags": ["No trained model found — using texture heuristics only."]}
