from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

THRESHOLD_KEYS = {
    "image_fake_probability_threshold",
    "video_fake_probability_threshold",
    "profile_fake_probability_threshold",
}

_cache_data: dict | None = None
_cache_expiry_ts: float = 0.0


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _models_dir() -> Path:
    return _project_root() / "ml" / "models"


def _thresholds_path() -> Path:
    return _models_dir() / "thresholds.json"


def _read_pickle_threshold(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            artifact = pickle.load(f)
        raw_threshold = float(artifact.get("decision_threshold", 0.5))
        threshold = raw_threshold * 100.0 if raw_threshold <= 1.0 else raw_threshold
        return round(max(20.0, min(80.0, threshold)), 2)
    except Exception:
        return None


def _load_overrides() -> dict:
    path = _thresholds_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        cleaned: dict[str, float] = {}
        for key in THRESHOLD_KEYS:
            if key not in data:
                continue
            value = float(data[key])
            cleaned[key] = round(max(20.0, min(80.0, value)), 2)
        return cleaned
    except Exception:
        return {}


def get_effective_thresholds() -> dict:
    global _cache_data, _cache_expiry_ts
    now = time.monotonic()
    if _cache_data is not None and now < _cache_expiry_ts:
        return _cache_data

    image_default = _read_pickle_threshold(_models_dir() / "image_cnn.pkl") or 50.0
    profile_default = _read_pickle_threshold(_models_dir() / "profile_nlp.pkl") or 50.0
    video_default = 55.0

    overrides = _load_overrides()
    values = {
        "image_fake_probability_threshold": float(overrides.get("image_fake_probability_threshold", image_default)),
        "video_fake_probability_threshold": float(overrides.get("video_fake_probability_threshold", video_default)),
        "profile_fake_probability_threshold": float(overrides.get("profile_fake_probability_threshold", profile_default)),
    }

    sources = {
        "image_fake_probability_threshold": (
            "override" if "image_fake_probability_threshold" in overrides else "model_default"
        ),
        "video_fake_probability_threshold": (
            "override" if "video_fake_probability_threshold" in overrides else "system_default"
        ),
        "profile_fake_probability_threshold": (
            "override" if "profile_fake_probability_threshold" in overrides else "model_default"
        ),
    }
    result = {"values": values, "sources": sources, "overrides": overrides}
    _cache_data = result
    _cache_expiry_ts = now + 3.0
    return result


def save_threshold_overrides(updates: dict) -> dict:
    global _cache_data, _cache_expiry_ts
    overrides = _load_overrides()
    next_overrides = dict(overrides)

    for key, value in updates.items():
        if key not in THRESHOLD_KEYS:
            raise ValueError(f"Unsupported threshold key: {key}")
        if value is None:
            next_overrides.pop(key, None)
            continue
        numeric = float(value)
        if numeric < 20.0 or numeric > 80.0:
            raise ValueError(f"{key} must be between 20 and 80.")
        next_overrides[key] = round(numeric, 2)

    path = _thresholds_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(next_overrides, f, ensure_ascii=True, indent=2)

    _cache_data = None
    _cache_expiry_ts = 0.0
    return get_effective_thresholds()
