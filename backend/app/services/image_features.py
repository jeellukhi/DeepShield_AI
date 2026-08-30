from __future__ import annotations

import cv2
import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def is_usable_image(image: np.ndarray | None, min_side: int = 24) -> bool:
    if image is None:
        return False
    if len(image.shape) < 2:
        return False
    h, w = image.shape[:2]
    return min(h, w) >= min_side


def _jpeg_blockiness_score(gray_norm: np.ndarray) -> float:
    if gray_norm.ndim != 2:
        return 0.0
    if gray_norm.shape[0] < 10 or gray_norm.shape[1] < 10:
        return 0.0

    diff_x = np.abs(np.diff(gray_norm, axis=1))
    diff_y = np.abs(np.diff(gray_norm, axis=0))
    boundary_cols = np.arange(7, diff_x.shape[1], 8)
    boundary_rows = np.arange(7, diff_y.shape[0], 8)
    if len(boundary_cols) == 0 or len(boundary_rows) == 0:
        return 0.0

    boundary_x = float(np.mean(diff_x[:, boundary_cols])) if len(boundary_cols) > 0 else 0.0
    boundary_y = float(np.mean(diff_y[boundary_rows, :])) if len(boundary_rows) > 0 else 0.0
    global_x = float(np.mean(diff_x))
    global_y = float(np.mean(diff_y))
    boundary_avg = 0.5 * (boundary_x + boundary_y)
    global_avg = 0.5 * (global_x + global_y)
    return boundary_avg - global_avg


def extract_image_feature(image: np.ndarray, input_size: int, feature_mode: str = "v3_multicue_stack") -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    gray_norm = resized.astype(np.float32) / 255.0
    if feature_mode == "v1_grayscale":
        return gray_norm.reshape(-1)

    edges = cv2.Canny(resized, 80, 160).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray_norm, cv2.CV_32F)
    lap_abs = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX)
    base_feature = np.concatenate([gray_norm.reshape(-1), edges.reshape(-1), lap_abs.reshape(-1)], axis=0)
    if feature_mode == "v2_texture_stack":
        return base_feature

    # v3_multicue_stack: texture map + chroma maps + frequency/artifact summary.
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    half_size = max(16, input_size // 2)

    cr = cv2.resize(ycrcb[:, :, 1], (half_size, half_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cb = cv2.resize(ycrcb[:, :, 2], (half_size, half_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sat = cv2.resize(hsv[:, :, 1], (half_size, half_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    gray_hist = cv2.calcHist([resized], [0], None, [32], [0, 256]).reshape(-1).astype(np.float32)
    hist_sum = float(np.sum(gray_hist))
    if hist_sum > 0:
        gray_hist /= hist_sum

    dct = cv2.dct(gray_norm.astype(np.float32))
    abs_dct = np.abs(dct)
    h, w = abs_dct.shape
    low_end = max(4, min(8, h // 4, w // 4))
    mid_end = max(low_end + 2, min(20, h // 2, w // 2))
    low_energy = float(np.mean(abs_dct[:low_end, :low_end]))
    mid_energy = (
        float(np.mean(abs_dct[low_end:mid_end, low_end:mid_end])) if mid_end > low_end else 0.0
    )
    high_energy = float(np.mean(abs_dct[mid_end:, mid_end:])) if mid_end < h and mid_end < w else 0.0
    total_energy = low_energy + mid_energy + high_energy + 1e-6
    low_ratio = low_energy / total_energy
    mid_ratio = mid_energy / total_energy
    high_ratio = high_energy / total_energy

    lap_var = float(cv2.Laplacian(resized, cv2.CV_64F).var())
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    noise_map = gray_norm - cv2.GaussianBlur(gray_norm, (5, 5), 0)
    noise_std = float(np.std(noise_map))
    blockiness = _jpeg_blockiness_score(gray_norm)
    brightness_mean = float(np.mean(gray_norm))
    brightness_std = float(np.std(gray_norm))
    chroma_std = float((np.std(cr) + np.std(cb)) * 0.5)
    sat_mean = float(np.mean(sat))

    summary = np.asarray(
        [
            _clamp01(lap_var / 450.0),
            _clamp01(edge_density / 0.22),
            _clamp01(noise_std / 0.14),
            _clamp01((blockiness + 0.08) / 0.18),
            _clamp01(brightness_mean),
            _clamp01(brightness_std / 0.32),
            _clamp01(chroma_std / 0.28),
            _clamp01(sat_mean),
            _clamp01(low_ratio),
            _clamp01(mid_ratio),
            _clamp01(high_ratio),
            _clamp01((high_energy + 1e-6) / (low_energy + 1e-6)),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            base_feature,
            cr.reshape(-1),
            cb.reshape(-1),
            sat.reshape(-1),
            gray_hist,
            summary,
        ],
        axis=0,
    )


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    aug = image.copy()
    if float(rng.random()) < 0.5:
        aug = cv2.flip(aug, 1)

    if float(rng.random()) < 0.55:
        h, w = aug.shape[:2]
        scale = float(rng.uniform(0.7, 0.96))
        new_w = max(24, int(w * scale))
        new_h = max(24, int(h * scale))
        small = cv2.resize(aug, (new_w, new_h), interpolation=cv2.INTER_AREA)
        aug = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if float(rng.random()) < 0.75:
        quality = int(rng.integers(35, 96))
        ok, encoded = cv2.imencode(".jpg", aug, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                aug = decoded

    if float(rng.random()) < 0.6:
        kernel = 3 if float(rng.random()) < 0.65 else 5
        aug = cv2.GaussianBlur(aug, (kernel, kernel), 0)

    if float(rng.random()) < 0.45:
        noise_sigma = float(rng.uniform(2.0, 10.0))
        noise = rng.normal(0.0, noise_sigma, aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if float(rng.random()) < 0.45:
        alpha = float(rng.uniform(0.82, 1.2))
        beta = float(rng.uniform(-18, 18))
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    if float(rng.random()) < 0.35:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= float(rng.uniform(0.75, 1.25))
        hsv[:, :, 2] *= float(rng.uniform(0.82, 1.18))
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return aug
