from __future__ import annotations

import pickle
from pathlib import Path


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _safe_load_pickle(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def get_model_summary() -> dict:
    models_dir = _project_root() / "ml" / "models"
    image_pickle_path = models_dir / "image_cnn.pkl"
    profile_pickle_path = models_dir / "profile_nlp.pkl"
    image_keras_path = models_dir / "image_cnn.keras"

    image_artifact = _safe_load_pickle(image_pickle_path)
    profile_artifact = _safe_load_pickle(profile_pickle_path)

    image_info = {
        "exists": image_pickle_path.exists() or image_keras_path.exists(),
        "path": str(image_pickle_path if image_pickle_path.exists() else image_keras_path),
        "type": image_artifact.get("model_type") if image_artifact else ("keras_model" if image_keras_path.exists() else None),
        "trained_at": image_artifact.get("trained_at") if image_artifact else None,
        "metrics": image_artifact.get("metrics", {}) if image_artifact else {},
    }

    profile_quality_gate = profile_artifact.get("quality_gate", {}) if profile_artifact else {}
    profile_info = {
        "exists": profile_pickle_path.exists(),
        "path": str(profile_pickle_path),
        "type": profile_artifact.get("model_type") if profile_artifact else None,
        "trained_at": profile_artifact.get("trained_at") if profile_artifact else None,
        "metrics": profile_artifact.get("metrics", {}) if profile_artifact else {},
        "quality_gate": profile_quality_gate,
    }

    return {
        "image_model": image_info,
        "profile_model": profile_info,
    }
