import pickle
from pathlib import Path


def _count_files(folder: Path, patterns: tuple[str, ...]) -> int:
    if not folder.exists():
        return 0
    count = 0
    for pattern in patterns:
        count += len(list(folder.glob(pattern)))
    return count


def _profile_model_quality(profile_model_path: Path) -> dict:
    if not profile_model_path.exists():
        return {"usable": False, "reason": "profile model not found"}
    try:
        with profile_model_path.open("rb") as f:
            artifact = pickle.load(f)
        quality_gate = artifact.get("quality_gate", {})
        if quality_gate:
            passed = bool(quality_gate.get("passed", False))
            if passed:
                return {"usable": True, "reason": "quality gate passed"}
            return {"usable": False, "reason": "needs more balanced feedback/training quality"}
        return {"usable": False, "reason": "legacy profile model; retrain required with quality gate"}
    except Exception:
        return {"usable": False, "reason": "unable to load profile model"}


def get_model_readiness() -> dict:
    # backend/app/services -> app -> backend -> project root
    project_root = Path(__file__).resolve().parents[3]
    dataset_root = project_root / "ml" / "datasets"
    models_root = project_root / "ml" / "models"

    image_real_dir = dataset_root / "deepfake_images" / "real"
    image_fake_dir = dataset_root / "deepfake_images" / "fake"

    image_real_count = _count_files(image_real_dir, ("*.jpg", "*.jpeg", "*.png", "*.webp"))
    image_fake_count = _count_files(image_fake_dir, ("*.jpg", "*.jpeg", "*.png", "*.webp"))
    total_images = image_real_count + image_fake_count

    image_model_keras_path = models_root / "image_cnn.keras"
    image_model_pickle_path = models_root / "image_cnn.pkl"
    profile_model_path = models_root / "profile_nlp.pkl"
    profile_quality = _profile_model_quality(profile_model_path)
    image_model_exists = image_model_keras_path.exists() or image_model_pickle_path.exists()
    trained_models_found = int(image_model_exists) + int(profile_model_path.exists())

    if image_model_pickle_path.exists():
        active_image_model_path = image_model_pickle_path
    else:
        active_image_model_path = image_model_keras_path

    if trained_models_found == 0:
        mode = "heuristic"
        status = "Training required for production-grade accuracy."
    else:
        mode = "hybrid"
        status = "At least one trained model is available."
        if profile_model_path.exists() and not profile_quality["usable"]:
            status = f"{status} Profile model is currently disabled ({profile_quality['reason']})."

    return {
        "mode": mode,
        "status": status,
        "dataset_paths": {
            "image_real_dir": str(image_real_dir),
            "image_fake_dir": str(image_fake_dir),
        },
        "dataset_counts": {
            "image_real_count": image_real_count,
            "image_fake_count": image_fake_count,
            "total_images": total_images,
        },
        "model_files": {
            "image_model_path": str(active_image_model_path),
            "image_model_keras_path": str(image_model_keras_path),
            "image_model_pickle_path": str(image_model_pickle_path),
            "profile_model_path": str(profile_model_path),
            "image_model_exists": image_model_exists,
            "profile_model_exists": profile_model_path.exists(),
            "profile_model_usable": profile_quality["usable"],
            "profile_model_status": profile_quality["reason"],
        },
    }
