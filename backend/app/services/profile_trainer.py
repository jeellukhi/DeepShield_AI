from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.db.mongo import fetch_profile_feedback_training_samples
from app.services.profile_text_features import build_profile_text


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _best_threshold(y_true: np.ndarray, fake_probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.arange(0.30, 0.71, 0.02):
        pred = (fake_probs >= threshold).astype(np.int32)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def train_profile_model(max_samples: int = 5000, min_per_class: int = 3) -> dict:
    rows, error = fetch_profile_feedback_training_samples(limit=max_samples)
    if error:
        raise ValueError(f"Unable to read feedback data: {error}")
    if not rows:
        raise ValueError("No feedback training data found. Save some confirmed_real/confirmed_fake feedback first.")

    x_text: list[str] = []
    y: list[int] = []
    class_counts = {"confirmed_real": 0, "confirmed_fake": 0}

    for row in rows:
        username = row.get("username", "")
        bio = row.get("bio", "")
        feedback_label = row.get("feedback_label")
        if feedback_label not in {"confirmed_real", "confirmed_fake"}:
            continue

        x_text.append(build_profile_text(username, bio))
        y.append(0 if feedback_label == "confirmed_real" else 1)
        class_counts[feedback_label] += 1

    if class_counts["confirmed_real"] < min_per_class or class_counts["confirmed_fake"] < min_per_class:
        raise ValueError(
            f"Need at least {min_per_class} samples for each class. "
            f"Current: real={class_counts['confirmed_real']}, fake={class_counts['confirmed_fake']}."
        )

    x = np.asarray(x_text, dtype=object)
    y_arr = np.asarray(y, dtype=np.int32)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y_arr,
        test_size=0.2,
        random_state=42,
        stratify=y_arr,
    )

    candidates = [
        {
            "name": "tfidf_logreg_bi",
            "pipeline": Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12000)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=700,
                            solver="liblinear",
                            class_weight="balanced",
                            C=1.2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        },
        {
            "name": "tfidf_logreg_tri",
            "pipeline": Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=18000)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=800,
                            solver="liblinear",
                            class_weight="balanced",
                            C=0.9,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        },
    ]

    best_name = ""
    best_pipeline = None
    best_threshold = 0.50
    best_val_f1 = -1.0
    min_train_class = int(min(np.bincount(y_train_full))) if len(y_train_full) > 0 else 0
    use_validation_split = len(y_train_full) >= 30 and min_train_class >= 5

    if use_validation_split:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full,
        )
        for candidate in candidates:
            pipeline = candidate["pipeline"]
            pipeline.fit(x_train, y_train)
            val_probs = pipeline.predict_proba(x_val)[:, 1]
            threshold, val_f1 = _best_threshold(y_val, val_probs)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_name = str(candidate["name"])
                best_pipeline = pipeline
                best_threshold = threshold
    else:
        best_name = str(candidates[0]["name"])
        best_pipeline = candidates[0]["pipeline"]
        best_threshold = 0.50
        best_val_f1 = float("nan")

    assert best_pipeline is not None
    best_pipeline.fit(x_train_full, y_train_full)
    test_probs = best_pipeline.predict_proba(x_test)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(np.int32)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
    }
    quality_gate = {
        "min_per_class_required": 20,
        "min_f1_required": 60.0,
        "passed": (
            class_counts["confirmed_real"] >= 20
            and class_counts["confirmed_fake"] >= 20
            and metrics["f1_score"] >= 60.0
        ),
    }

    models_dir = _project_root() / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "profile_nlp.pkl"
    artifact = {
        "model": best_pipeline,
        "model_type": f"{best_name}_feedback",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_map": {"0": "real", "1": "fake"},
        "metrics": metrics,
        "class_counts": class_counts,
        "quality_gate": quality_gate,
        "decision_threshold": round(best_threshold, 2),
        "validation_f1": (round(best_val_f1 * 100, 2) if best_val_f1 == best_val_f1 else None),
    }

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    return {
        "model_path": str(model_path),
        "model_type": artifact["model_type"],
        "trained_at": artifact["trained_at"],
        "dataset_used": {
            "total_samples": int(len(y_arr)),
            "confirmed_real": int(class_counts["confirmed_real"]),
            "confirmed_fake": int(class_counts["confirmed_fake"]),
        },
        "split": {
            "train_count": int(len(y_train)),
            "test_count": int(len(y_test)),
        },
        "metrics": metrics,
        "quality_gate": quality_gate,
    }
