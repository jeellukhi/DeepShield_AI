from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.db.mongo import fetch_profile_feedback_training_samples
from app.services.image_features import extract_image_feature, is_usable_image
from app.services.profile_text_features import build_profile_text

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    real_precision = round((tn / max(1, tn + fn)) * 100.0, 2)
    real_recall = round((tn / max(1, tn + fp)) * 100.0, 2)
    fake_precision = round((tp / max(1, tp + fp)) * 100.0, 2)
    fake_recall = round((tp / max(1, tp + fn)) * 100.0, 2)

    return {
        "metrics": {
            "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100.0, 2),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100.0, 2),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100.0, 2),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100.0, 2),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "class_metrics": {
            "real_precision": real_precision,
            "real_recall": real_recall,
            "fake_precision": fake_precision,
            "fake_recall": fake_recall,
        },
    }


def _safe_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return round(float(roc_auc_score(y_true, probs)) * 100.0, 2)
    except Exception:
        return None


def evaluate_image_model(max_per_class: int = 2000, threshold_pct: float | None = None) -> dict:
    model_path = _project_root() / "ml" / "models" / "image_cnn.pkl"
    if not model_path.exists():
        return {"available": False, "message": "Image model not found. Train image model first."}

    with model_path.open("rb") as f:
        artifact = pickle.load(f)

    model = artifact.get("model")
    if model is None:
        return {"available": False, "message": "Image model artifact is invalid."}

    input_size = int(artifact.get("input_size", 48))
    feature_mode = str(artifact.get("feature_mode", "v2_texture_stack"))
    raw_default_threshold = float(artifact.get("decision_threshold", 0.5))
    default_threshold = raw_default_threshold * 100.0 if raw_default_threshold <= 1.0 else raw_default_threshold
    applied_threshold = float(threshold_pct if threshold_pct is not None else default_threshold)
    applied_threshold = max(20.0, min(80.0, applied_threshold))

    project_root = _project_root()
    real_dir = project_root / "ml" / "datasets" / "deepfake_images" / "real"
    fake_dir = project_root / "ml" / "datasets" / "deepfake_images" / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        return {"available": False, "message": "Image dataset folders are missing."}

    rng = np.random.default_rng(42)

    def _scan(folder: Path, label: int) -> tuple[list[np.ndarray], list[int], int]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        skipped = 0
        paths = [
            p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not paths:
            return xs, ys, skipped
        sample_count = min(len(paths), max(50, max_per_class))
        if len(paths) > sample_count:
            sampled_idx = rng.choice(len(paths), size=sample_count, replace=False)
            paths = [paths[int(i)] for i in np.sort(sampled_idx)]
        for path in paths:
            image = cv2.imread(str(path))
            if not is_usable_image(image, min_side=24):
                skipped += 1
                continue
            xs.append(extract_image_feature(image, input_size, feature_mode))
            ys.append(label)
        return xs, ys, skipped

    real_x, real_y, real_skipped = _scan(real_dir, 0)
    fake_x, fake_y, fake_skipped = _scan(fake_dir, 1)
    if len(real_x) < 50 or len(fake_x) < 50:
        return {
            "available": False,
            "message": f"Not enough evaluation images. real={len(real_x)}, fake={len(fake_x)} (need >=50 each).",
        }

    x = np.asarray(real_x + fake_x, dtype=np.float32)
    y = np.asarray(real_y + fake_y, dtype=np.int32)

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    probs = model.predict_proba(x_test)[:, 1]
    y_pred = (probs >= (applied_threshold / 100.0)).astype(np.int32)

    summary = _binary_metrics(y_test, y_pred)
    summary["metrics"]["balanced_accuracy"] = round(float(balanced_accuracy_score(y_test, y_pred)) * 100.0, 2)
    summary["metrics"]["roc_auc"] = _safe_roc_auc(y_test, probs)
    return {
        "available": True,
        "model_type": str(artifact.get("model_type", "image_model")),
        "threshold_pct": round(applied_threshold, 2),
        "dataset": {
            "real_count": len(real_x),
            "fake_count": len(fake_x),
            "real_skipped": real_skipped,
            "fake_skipped": fake_skipped,
            "test_count": int(len(y_test)),
        },
        **summary,
    }


def evaluate_profile_model(max_samples: int = 6000, threshold_pct: float | None = None) -> dict:
    model_path = _project_root() / "ml" / "models" / "profile_nlp.pkl"
    if not model_path.exists():
        return {"available": False, "message": "Profile model not found. Train profile model first."}

    with model_path.open("rb") as f:
        artifact = pickle.load(f)

    model = artifact.get("model")
    if model is None:
        return {"available": False, "message": "Profile model artifact is invalid."}

    rows, error = fetch_profile_feedback_training_samples(limit=max_samples)
    if error:
        return {"available": False, "message": f"Unable to read profile feedback rows: {error}"}
    if not rows:
        return {"available": False, "message": "No profile feedback rows found."}

    x_text: list[str] = []
    y: list[int] = []
    for row in rows:
        label = row.get("feedback_label")
        if label not in {"confirmed_real", "confirmed_fake"}:
            continue
        x_text.append(build_profile_text(row.get("username", ""), row.get("bio", "")))
        y.append(0 if label == "confirmed_real" else 1)

    if len(y) < 20:
        return {"available": False, "message": "Need at least 20 labeled samples to evaluate profile model."}

    x = np.asarray(x_text, dtype=object)
    y_arr = np.asarray(y, dtype=np.int32)

    _, x_test, _, y_test = train_test_split(
        x,
        y_arr,
        test_size=0.2,
        random_state=42,
        stratify=y_arr,
    )

    raw_default_threshold = float(artifact.get("decision_threshold", 0.5))
    default_threshold = raw_default_threshold * 100.0 if raw_default_threshold <= 1.0 else raw_default_threshold
    applied_threshold = float(threshold_pct if threshold_pct is not None else default_threshold)
    applied_threshold = max(20.0, min(80.0, applied_threshold))

    probs = model.predict_proba(x_test)[:, 1]
    y_pred = (probs >= (applied_threshold / 100.0)).astype(np.int32)
    summary = _binary_metrics(y_test, y_pred)
    summary["metrics"]["balanced_accuracy"] = round(float(balanced_accuracy_score(y_test, y_pred)) * 100.0, 2)
    summary["metrics"]["roc_auc"] = _safe_roc_auc(y_test, probs)

    return {
        "available": True,
        "model_type": str(artifact.get("model_type", "profile_model")),
        "threshold_pct": round(applied_threshold, 2),
        "dataset": {
            "total_samples": int(len(y_arr)),
            "test_count": int(len(y_test)),
            "real_count": int(np.sum(y_arr == 0)),
            "fake_count": int(np.sum(y_arr == 1)),
        },
        "quality_gate": artifact.get("quality_gate", {}),
        **summary,
    }
