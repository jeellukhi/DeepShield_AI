from __future__ import annotations

import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.db.mongo import fetch_profile_feedback_training_samples
from app.services.profile_text_features import build_profile_text
from app.services.profile_numeric_features import extract_numeric_features


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_synthetic_profiles() -> list[dict]:
    """Load bootstrap synthetic profiles from CSV (ships with the project)."""
    csv_path = _project_root() / "ml" / "datasets" / "synthetic_profiles" / "bootstrap_profiles.csv"
    if not csv_path.exists():
        return []
    rows = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "")
                if label not in {"confirmed_real", "confirmed_fake"}:
                    continue
                rows.append({
                    "username": row.get("username", ""),
                    "bio": row.get("bio", ""),
                    "followers": float(row.get("followers", 0) or 0),
                    "following": float(row.get("following", 0) or 0),
                    "posts": float(row.get("posts", 0) or 0),
                    "engagement_rate": float(row.get("engagement_rate", 0) or 0),
                    "image_score": float(row.get("image_score", 50) or 50),
                    "feedback_label": label,
                })
    except Exception:
        return []
    return rows


def _best_threshold(y_true: np.ndarray, fake_probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.arange(0.25, 0.76, 0.01):
        pred = (fake_probs >= threshold).astype(np.int32)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _build_features(rows: list[dict]) -> tuple[list[str], list[list[float]], list[int]]:
    """Build text features and numeric features from profile rows."""
    texts: list[str] = []
    numerics: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        username = str(row.get("username", ""))
        bio = str(row.get("bio", ""))
        feedback_label = row.get("feedback_label", "")
        if feedback_label not in {"confirmed_real", "confirmed_fake"}:
            continue
        texts.append(build_profile_text(username, bio))
        numerics.append(extract_numeric_features(
            followers=row.get("followers", 0),
            following=row.get("following", 0),
            posts=row.get("posts", 0),
            engagement_rate=row.get("engagement_rate", 0.0),
            image_score=row.get("image_score", 50.0),
        ))
        labels.append(0 if feedback_label == "confirmed_real" else 1)
    return texts, numerics, labels


def train_profile_model(max_samples: int = 5000, min_per_class: int = 3) -> dict:
    # --- Load MongoDB feedback data ---
    db_rows, error = fetch_profile_feedback_training_samples(limit=max_samples)
    if error:
        db_rows = []

    # --- Load synthetic bootstrap data ---
    synthetic_rows = _load_synthetic_profiles()

    # Merge: MongoDB data takes priority, synthetic fills the gap
    all_rows = list(db_rows) + list(synthetic_rows)

    if not all_rows:
        raise ValueError("No training data found. No MongoDB feedback and no synthetic profiles CSV.")

    texts, numerics, labels = _build_features(all_rows)

    if not texts:
        raise ValueError("No valid labeled samples after processing.")

    class_counts = {
        "confirmed_real": int(labels.count(0)),
        "confirmed_fake": int(labels.count(1)),
    }
    db_counts = {"db_real": 0, "db_fake": 0, "synthetic_real": 0, "synthetic_fake": 0}
    for r in db_rows:
        if r.get("feedback_label") == "confirmed_real":
            db_counts["db_real"] += 1
        elif r.get("feedback_label") == "confirmed_fake":
            db_counts["db_fake"] += 1
    for r in synthetic_rows:
        if r.get("feedback_label") == "confirmed_real":
            db_counts["synthetic_real"] += 1
        elif r.get("feedback_label") == "confirmed_fake":
            db_counts["synthetic_fake"] += 1

    if class_counts["confirmed_real"] < min_per_class or class_counts["confirmed_fake"] < min_per_class:
        raise ValueError(
            f"Need at least {min_per_class} samples per class. "
            f"Got real={class_counts['confirmed_real']}, fake={class_counts['confirmed_fake']}."
        )

    x_text = np.asarray(texts, dtype=object)
    x_num = np.asarray(numerics, dtype=np.float32)
    y_arr = np.asarray(labels, dtype=np.int32)

    # --- Train/test split ---
    indices = np.arange(len(y_arr))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_arr
    )
    x_text_train, x_text_test = x_text[idx_train], x_text[idx_test]
    x_num_train, x_num_test = x_num[idx_train], x_num[idx_test]
    y_train, y_test = y_arr[idx_train], y_arr[idx_test]

    # --- Model 1: TF-IDF + Logistic Regression (text signals) ---
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=20000)),
        ("clf", LogisticRegression(
            max_iter=1000, solver="liblinear",
            class_weight="balanced", C=1.0, random_state=42,
        )),
    ])
    text_pipeline.fit(x_text_train, y_train)
    text_train_probs = text_pipeline.predict_proba(x_text_train)[:, 1].reshape(-1, 1)
    text_test_probs = text_pipeline.predict_proba(x_text_test)[:, 1].reshape(-1, 1)

    # --- Model 2: RandomForest on numeric features ---
    rf_numeric = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    rf_numeric.fit(x_num_train, y_train)
    num_train_probs = rf_numeric.predict_proba(x_num_train)[:, 1].reshape(-1, 1)
    num_test_probs = rf_numeric.predict_proba(x_num_test)[:, 1].reshape(-1, 1)

    # --- Model 3: GradientBoosting on numeric features ---
    gb_numeric = Pipeline([
        ("scale", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            min_samples_leaf=3, random_state=42,
        )),
    ])
    gb_numeric.fit(x_num_train, y_train)
    gb_train_probs = gb_numeric.predict_proba(x_num_train)[:, 1].reshape(-1, 1)
    gb_test_probs = gb_numeric.predict_proba(x_num_test)[:, 1].reshape(-1, 1)

    # --- Ensemble: weighted average of all 3 models ---
    # Text model is useful for bio/username signals
    # Numeric models capture follower/engagement patterns better
    # Weights: text=0.30, RF_numeric=0.40, GB_numeric=0.30
    train_ensemble = (0.30 * text_train_probs + 0.40 * num_train_probs + 0.30 * gb_train_probs).ravel()
    test_ensemble = (0.30 * text_test_probs + 0.40 * num_test_probs + 0.30 * gb_test_probs).ravel()

    # --- Find best threshold ---
    best_threshold, best_val_f1 = _best_threshold(y_train, train_ensemble)

    # --- Final evaluation on test set ---
    y_pred = (test_ensemble >= best_threshold).astype(np.int32)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_test, test_ensemble)) * 100, 2),
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

    # --- Save model artifact ---
    models_dir = _project_root() / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "profile_nlp.pkl"

    artifact = {
        "model_text": text_pipeline,
        "model_rf_numeric": rf_numeric,
        "model_gb_numeric": gb_numeric,
        "ensemble_weights": {"text": 0.30, "rf_numeric": 0.40, "gb_numeric": 0.30},
        "model_type": "text_tfidf_logreg+rf_numeric+gb_numeric_ensemble",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "class_map": {"0": "real", "1": "fake"},
        "metrics": metrics,
        "class_counts": class_counts,
        "data_sources": db_counts,
        "quality_gate": quality_gate,
        "decision_threshold": round(best_threshold, 4),
        "validation_f1": round(best_val_f1 * 100, 2),
        "total_samples": len(y_arr),
    }

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    return {
        "model_path": str(model_path),
        "model_type": artifact["model_type"],
        "trained_at": artifact["trained_at"],
        "dataset_used": {
            "total_samples": len(y_arr),
            "confirmed_real": class_counts["confirmed_real"],
            "confirmed_fake": class_counts["confirmed_fake"],
            **db_counts,
        },
        "split": {
            "train_count": int(len(y_train)),
            "test_count": int(len(y_test)),
        },
        "metrics": metrics,
        "quality_gate": quality_gate,
        "decision_threshold": round(best_threshold * 100, 2),
        "validation_f1": artifact["validation_f1"],
    }
