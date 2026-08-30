from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.image_features import augment_image, extract_image_feature, is_usable_image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _collect_image_paths(folder: Path, max_files: int, rng: np.random.Generator) -> tuple[list[Path], int]:
    image_paths = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    total_found = len(image_paths)
    if max_files > 0 and total_found > max_files:
        sampled_idx = rng.choice(total_found, size=max_files, replace=False)
        image_paths = [image_paths[int(i)] for i in np.sort(sampled_idx)]
    return image_paths, total_found


def _load_base_samples(
    image_paths: list[Path],
    label: int,
    input_size: int,
    feature_mode: str,
    min_side: int = 24,
) -> tuple[list[np.ndarray], list[int], list[str], int]:
    items: list[np.ndarray] = []
    labels: list[int] = []
    used_paths: list[str] = []
    skipped = 0

    for path in image_paths:
        image = cv2.imread(str(path))
        if not is_usable_image(image, min_side=min_side):
            skipped += 1
            continue
        items.append(extract_image_feature(image, input_size, feature_mode))
        labels.append(label)
        used_paths.append(str(path))

    return items, labels, used_paths, skipped


def _augment_training_features(
    train_paths: np.ndarray,
    y_train: np.ndarray,
    input_size: int,
    feature_mode: str,
    rng: np.random.Generator,
    augment_per_image: int,
    augment_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if augment_per_image <= 0 or augment_fraction <= 0.0:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            {"augmented_real": 0, "augmented_fake": 0, "augment_failures": 0},
        )

    x_aug_list: list[np.ndarray] = []
    y_aug_list: list[int] = []
    augmented_real = 0
    augmented_fake = 0
    augment_failures = 0

    for raw_path, label in zip(train_paths.tolist(), y_train.tolist()):
        if float(rng.random()) > augment_fraction:
            continue
        image = cv2.imread(str(raw_path))
        if not is_usable_image(image, min_side=24):
            augment_failures += 1
            continue
        for _ in range(augment_per_image):
            aug = augment_image(image, rng)
            x_aug_list.append(extract_image_feature(aug, input_size, feature_mode))
            y_aug_list.append(int(label))
            if int(label) == 0:
                augmented_real += 1
            else:
                augmented_fake += 1

    if x_aug_list:
        x_aug = np.asarray(x_aug_list, dtype=np.float32)
        y_aug = np.asarray(y_aug_list, dtype=np.int32)
    else:
        x_aug = np.empty((0, 0), dtype=np.float32)
        y_aug = np.empty((0,), dtype=np.int32)

    return (
        x_aug,
        y_aug,
        {
            "augmented_real": augmented_real,
            "augmented_fake": augmented_fake,
            "augment_failures": augment_failures,
        },
    )


def _best_threshold(y_true: np.ndarray, fake_probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.arange(0.30, 0.71, 0.01):
        pred = (fake_probs >= threshold).astype(np.int32)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def train_image_model(max_per_class: int = 2000, input_size: int = 48, test_size: float = 0.2) -> dict:
    if max_per_class < 50:
        raise ValueError("max_per_class must be at least 50.")
    feature_mode = "v3_multicue_stack"

    project_root = _project_root()
    real_dir = project_root / "ml" / "datasets" / "deepfake_images" / "real"
    fake_dir = project_root / "ml" / "datasets" / "deepfake_images" / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise ValueError("Dataset folders not found. Expected real/ and fake/ folders.")

    rng = np.random.default_rng(42)
    if max_per_class <= 3000:
        augment_per_image = 1
        augment_fraction = 1.0
    elif max_per_class <= 7000:
        augment_per_image = 1
        augment_fraction = 0.6
    else:
        augment_per_image = 0
        augment_fraction = 0.0

    real_paths, real_total_found = _collect_image_paths(real_dir, max_per_class, rng)
    fake_paths, fake_total_found = _collect_image_paths(fake_dir, max_per_class, rng)

    real_x, real_y, real_used_paths, real_skipped = _load_base_samples(
        real_paths,
        0,
        input_size,
        feature_mode,
    )
    fake_x, fake_y, fake_used_paths, fake_skipped = _load_base_samples(
        fake_paths,
        1,
        input_size,
        feature_mode,
    )

    if len(real_x) < 50 or len(fake_x) < 50:
        raise ValueError("Need at least 50 valid images in both real and fake classes.")

    x_base = np.asarray(real_x + fake_x, dtype=np.float32)
    y_base = np.asarray(real_y + fake_y, dtype=np.int32)
    source_paths = np.asarray(real_used_paths + fake_used_paths, dtype=object)

    x_train_base, x_test, y_train_base, y_test, train_paths, _ = train_test_split(
        x_base,
        y_base,
        source_paths,
        test_size=test_size,
        random_state=42,
        stratify=y_base,
    )

    min_train_class = int(min(np.bincount(y_train_base))) if len(y_train_base) > 0 else 0
    use_validation_split = len(y_train_base) >= 80 and min_train_class >= 20
    candidates: list[dict] = []
    candidates.append(
        {
            "name": "logreg_scaled_texture",
            "pipeline": Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=900,
                            solver="liblinear",
                            class_weight="balanced",
                            C=1.1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }
    )
    candidates.append(
        {
            "name": "logreg_lbfgs_texture",
            "pipeline": Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1200,
                            solver="lbfgs",
                            class_weight="balanced",
                            C=1.0,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }
    )
    pca_components = min(256, x_train_base.shape[1], max(2, x_train_base.shape[0] - 1))
    if pca_components > 4:
        candidates.append(
            {
                "name": "pca_logreg_texture",
                "pipeline": Pipeline(
                    steps=[
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=pca_components, random_state=42)),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=1200,
                                solver="lbfgs",
                                class_weight="balanced",
                                C=1.4,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            }
        )

    best_name = ""
    best_model = None
    best_threshold = 0.50
    best_val_f1 = -1.0
    if use_validation_split:
        x_train_sub, x_val, y_train_sub, y_val, train_paths_sub, _ = train_test_split(
            x_train_base,
            y_train_base,
            train_paths,
            test_size=0.2,
            random_state=42,
            stratify=y_train_base,
        )
        x_aug_sub, y_aug_sub, _ = _augment_training_features(
            train_paths_sub,
            y_train_sub,
            input_size,
            feature_mode,
            rng,
            augment_per_image,
            augment_fraction,
        )
        if len(y_aug_sub) > 0:
            x_train = np.vstack([x_train_sub, x_aug_sub]).astype(np.float32)
            y_train = np.concatenate([y_train_sub, y_aug_sub]).astype(np.int32)
        else:
            x_train = x_train_sub
            y_train = y_train_sub
        for candidate in candidates:
            model = candidate["pipeline"]
            model.fit(x_train, y_train)
            val_probs = model.predict_proba(x_val)[:, 1]
            threshold, val_f1 = _best_threshold(y_val, val_probs)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold = threshold
                best_name = str(candidate["name"])
                best_model = model
    else:
        best_name = str(candidates[0]["name"])
        best_model = candidates[0]["pipeline"]
        best_val_f1 = float("nan")

    assert best_model is not None
    x_aug_full, y_aug_full, aug_stats = _augment_training_features(
        train_paths,
        y_train_base,
        input_size,
        feature_mode,
        rng,
        augment_per_image,
        augment_fraction,
    )
    if len(y_aug_full) > 0:
        x_train_final = np.vstack([x_train_base, x_aug_full]).astype(np.float32)
        y_train_final = np.concatenate([y_train_base, y_aug_full]).astype(np.int32)
    else:
        x_train_final = x_train_base
        y_train_final = y_train_base

    best_model.fit(x_train_final, y_train_final)
    test_probs = best_model.predict_proba(x_test)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(np.int32)
    tn = int(np.sum((y_test == 0) & (y_pred == 0)))
    fp = int(np.sum((y_test == 0) & (y_pred == 1)))
    fn = int(np.sum((y_test == 1) & (y_pred == 0)))
    tp = int(np.sum((y_test == 1) & (y_pred == 1)))

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)) * 100, 2),
    }

    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "image_cnn.pkl"

    artifact = {
        "model": best_model,
        "input_size": input_size,
        "feature_mode": feature_mode,
        "model_type": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_map": {"0": "real", "1": "fake"},
        "metrics": metrics,
        "decision_threshold": round(best_threshold, 2),
        "validation_f1": (round(best_val_f1 * 100, 2) if best_val_f1 == best_val_f1 else None),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "augmentation": {
            "augment_per_image": augment_per_image,
            "augment_fraction": round(float(augment_fraction), 2),
            **aug_stats,
        },
        "dataset_summary": {
            "real_total_found": real_total_found,
            "fake_total_found": fake_total_found,
            "real_sampled_paths": len(real_paths),
            "fake_sampled_paths": len(fake_paths),
            "real_valid_base": len(real_x),
            "fake_valid_base": len(fake_x),
            "real_skipped": real_skipped,
            "fake_skipped": fake_skipped,
        },
    }

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    return {
        "model_path": str(model_path),
        "model_type": artifact["model_type"],
        "dataset_used": {
            "real_count": len(real_x),
            "fake_count": len(fake_x),
            "total_count": len(real_x) + len(fake_x),
            "real_total_found": real_total_found,
            "fake_total_found": fake_total_found,
            "real_sampled_paths": len(real_paths),
            "fake_sampled_paths": len(fake_paths),
            "real_skipped": real_skipped,
            "fake_skipped": fake_skipped,
            "real_augmented": aug_stats["augmented_real"],
            "fake_augmented": aug_stats["augmented_fake"],
            "augment_failures": aug_stats["augment_failures"],
            "augment_per_image": augment_per_image,
            "augment_fraction": round(float(augment_fraction), 2),
        },
        "split": {
            "train_base_count": int(len(y_train_base)),
            "train_augmented_count": int(len(y_aug_full)),
            "train_count": int(len(y_train_final)),
            "test_count": len(y_test),
        },
        "metrics": metrics,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "trained_at": artifact["trained_at"],
        "decision_threshold": round(best_threshold * 100, 2),
        "validation_f1": artifact["validation_f1"],
    }
