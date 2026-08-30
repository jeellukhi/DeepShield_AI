from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _collect_paths(folder: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def _sample_paths(paths: list[Path], max_scan: int, rng: np.random.Generator) -> list[Path]:
    if len(paths) <= max_scan:
        return paths
    sampled_idx = rng.choice(len(paths), size=max_scan, replace=False)
    return [paths[int(i)] for i in np.sort(sampled_idx)]


def _safe_read(path: Path) -> tuple[np.ndarray | None, str]:
    try:
        image = cv2.imread(str(path))
        if image is None:
            return None, "decode_failed"
        return image, ""
    except Exception:
        return None, "decode_exception"


def _file_signature(path: Path, max_bytes: int = 262144) -> str:
    digest = hashlib.md5()
    try:
        with path.open("rb") as f:
            digest.update(f.read(max_bytes))
        return digest.hexdigest()
    except Exception:
        return "io_error"


def _scan_class(paths: list[Path], max_scan: int, rng: np.random.Generator) -> dict:
    sampled = _sample_paths(paths, max_scan=max_scan, rng=rng)
    corrupt = 0
    too_small = 0
    blurry = 0
    low_texture = 0
    duplicate_candidates = 0
    widths: list[int] = []
    heights: list[int] = []
    lap_vars: list[float] = []
    hashes: dict[str, int] = {}

    for path in sampled:
        file_hash = _file_signature(path)
        if file_hash != "io_error":
            hashes[file_hash] = hashes.get(file_hash, 0) + 1

        image, err = _safe_read(path)
        if image is None:
            corrupt += 1
            continue

        h, w = image.shape[:2]
        widths.append(int(w))
        heights.append(int(h))
        if min(h, w) < 128:
            too_small += 1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lap_vars.append(lap_var)
        if lap_var < 45.0:
            blurry += 1

        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
        if edge_density < 0.015:
            low_texture += 1

    for _, count in hashes.items():
        if count > 1:
            duplicate_candidates += count - 1

    scanned_count = len(sampled)
    valid_count = max(0, scanned_count - corrupt)
    avg_w = round(float(np.mean(widths)), 2) if widths else 0.0
    avg_h = round(float(np.mean(heights)), 2) if heights else 0.0
    avg_lap = round(float(np.mean(lap_vars)), 2) if lap_vars else 0.0

    return {
        "total_files": len(paths),
        "scanned_files": scanned_count,
        "valid_files": valid_count,
        "corrupt_files": corrupt,
        "small_resolution_files": too_small,
        "blurry_files": blurry,
        "low_texture_files": low_texture,
        "duplicate_candidates": duplicate_candidates,
        "avg_width": avg_w,
        "avg_height": avg_h,
        "avg_laplacian_variance": avg_lap,
    }


def _class_quality_score(stats: dict) -> float:
    scanned = max(1, int(stats.get("scanned_files", 0)))
    corrupt_rate = float(stats.get("corrupt_files", 0)) / scanned
    small_rate = float(stats.get("small_resolution_files", 0)) / scanned
    blur_rate = float(stats.get("blurry_files", 0)) / scanned
    low_texture_rate = float(stats.get("low_texture_files", 0)) / scanned
    dup_rate = float(stats.get("duplicate_candidates", 0)) / scanned

    score = 100.0
    score -= min(25.0, corrupt_rate * 220.0)
    score -= min(20.0, small_rate * 120.0)
    score -= min(20.0, blur_rate * 80.0)
    score -= min(15.0, low_texture_rate * 70.0)
    score -= min(20.0, dup_rate * 180.0)
    return round(max(0.0, min(100.0, score)), 2)


def get_dataset_quality_report(max_scan_per_class: int = 2000) -> dict:
    safe_max_scan = max(200, min(max_scan_per_class, 10000))
    root = _project_root()
    real_dir = root / "ml" / "datasets" / "deepfake_images" / "real"
    fake_dir = root / "ml" / "datasets" / "deepfake_images" / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        return {
            "available": False,
            "message": "Dataset folders not found. Expected ml/datasets/deepfake_images/{real,fake}.",
        }

    rng = np.random.default_rng(42)
    real_paths = _collect_paths(real_dir)
    fake_paths = _collect_paths(fake_dir)

    real_stats = _scan_class(real_paths, safe_max_scan, rng)
    fake_stats = _scan_class(fake_paths, safe_max_scan, rng)
    real_score = _class_quality_score(real_stats)
    fake_score = _class_quality_score(fake_stats)

    total_real = int(real_stats["total_files"])
    total_fake = int(fake_stats["total_files"])
    imbalance_ratio = round((max(total_real, total_fake) / max(1, min(total_real, total_fake))), 3)
    balanced = imbalance_ratio <= 1.3
    quality_score = round((real_score + fake_score) / 2.0, 2)

    recommendations: list[str] = []
    if not balanced:
        recommendations.append("Class imbalance is high. Keep real/fake counts closer (<=1.3x ratio).")
    if real_stats["duplicate_candidates"] > 0 or fake_stats["duplicate_candidates"] > 0:
        recommendations.append("Remove duplicate/near-duplicate images before retraining.")
    if real_stats["blurry_files"] > 0 or fake_stats["blurry_files"] > 0:
        recommendations.append("Filter very blurry samples; keep clearer face/profile media.")
    if real_stats["small_resolution_files"] > 0 or fake_stats["small_resolution_files"] > 0:
        recommendations.append("Drop low-resolution images (<128px min side) for better model quality.")
    if real_stats["corrupt_files"] > 0 or fake_stats["corrupt_files"] > 0:
        recommendations.append("Clean unreadable/corrupted files in dataset folders.")
    if not recommendations:
        recommendations.append("Dataset quality checks look good for current scan window.")

    return {
        "available": True,
        "scan_config": {"max_scan_per_class": safe_max_scan, "seed": 42},
        "dataset_paths": {"real_dir": str(real_dir), "fake_dir": str(fake_dir)},
        "counts": {
            "real_total": total_real,
            "fake_total": total_fake,
            "imbalance_ratio": imbalance_ratio,
            "balanced": balanced,
        },
        "real": real_stats,
        "fake": fake_stats,
        "quality_score": quality_score,
        "quality_grade": "A" if quality_score >= 85 else ("B" if quality_score >= 70 else ("C" if quality_score >= 55 else "D")),
        "recommendations": recommendations,
    }
