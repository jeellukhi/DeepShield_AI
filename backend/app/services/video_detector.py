from __future__ import annotations

import cv2
import numpy as np

from app.services.image_detector import analyze_image_array, analyze_image_frame
from app.services.model_thresholds import get_effective_thresholds


def _optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute optical flow between two grayscale frames.
    Returns a score 0-100: low = suspicious (unnatural motion), high = natural motion.
    Deepfakes often have subtle but detectable motion inconsistencies.
    """
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        # Real video: smooth consistent motion (low std relative to mean)
        # Deepfake: erratic or zero motion around face region
        if mean_mag < 0.001:
            return 40.0   # Completely static — slight suspicion
        consistency_ratio = 1.0 - min(1.0, std_mag / max(mean_mag, 0.001))
        return round(50.0 + 50.0 * consistency_ratio, 2)
    except Exception:
        return 50.0  # neutral if flow fails


def _temporal_consistency_score(frame_scores: list[float]) -> float:
    """
    Check if authenticity scores vary wildly between frames.
    Real videos are consistent; deepfakes can have sudden drops on face cuts.
    """
    if len(frame_scores) < 3:
        return 75.0
    diffs = [abs(frame_scores[i] - frame_scores[i - 1]) for i in range(1, len(frame_scores))]
    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))
    # Penalize large frame-to-frame swings
    swing_penalty = min(50.0, mean_diff * 1.5 + max_diff * 0.3)
    return round(max(0.0, 100.0 - swing_penalty), 2)


def analyze_video_authenticity(video_path: str, max_frames: int = 64) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    duration_sec = total_frames / max(fps, 1.0)

    # Adaptive frame step: sample evenly across entire video
    frame_step = max(1, total_frames // max_frames) if total_frames > 0 else 1

    frame_index = 0
    frame_scores: list[float] = []
    frame_scores_model: list[float] = []
    frame_scores_heuristic: list[float] = []
    optical_flow_scores: list[float] = []
    model_sources: set[str] = set()
    prev_gray: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            # --- Image authenticity ---
            model_analysis = analyze_image_frame(frame)
            heuristic_analysis = analyze_image_array(frame)

            model_score = float(model_analysis["authenticity_score"])
            heuristic_score = float(heuristic_analysis["authenticity_score"])
            model_source = str(model_analysis.get("model_source", "unknown"))
            model_sources.add(model_source)

            if model_source == "trained_image_model":
                fused_score = 0.70 * model_score + 0.30 * heuristic_score
            else:
                fused_score = heuristic_score

            frame_scores_model.append(model_score)
            frame_scores_heuristic.append(heuristic_score)
            frame_scores.append(fused_score)

            # --- Optical flow between consecutive sampled frames ---
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray_small = cv2.resize(curr_gray, (160, 120))
            if prev_gray is not None:
                flow_score = _optical_flow_score(prev_gray, curr_gray_small)
                optical_flow_scores.append(flow_score)
            prev_gray = curr_gray_small

            if len(frame_scores) >= max_frames:
                break

        frame_index += 1

    cap.release()

    if not frame_scores:
        raise ValueError("No readable frames found in the video.")

    # --- Aggregate scores ---
    threshold_cfg = get_effective_thresholds()
    decision_threshold = float(threshold_cfg.get("values", {}).get("video_fake_probability_threshold", 55.0))
    decision_threshold = max(20.0, min(80.0, decision_threshold))

    avg_model_score = float(np.mean(frame_scores_model))
    avg_heuristic_score = float(np.mean(frame_scores_heuristic))
    avg_fused_score = float(np.mean(frame_scores))
    score_std = float(np.std(frame_scores))

    # Temporal consistency: penalize erratic frame-to-frame swings
    temporal_score = _temporal_consistency_score(frame_scores)

    # Optical flow: penalize unnatural or absent motion
    avg_flow_score = float(np.mean(optical_flow_scores)) if optical_flow_scores else 50.0

    # --- Final score fusion ---
    # Weights: fused_image=0.55, temporal=0.25, optical_flow=0.20
    if len(optical_flow_scores) >= 2:
        final_score = (
            0.55 * avg_fused_score
            + 0.25 * temporal_score
            + 0.20 * avg_flow_score
        )
    else:
        # Not enough frames for flow — use image + temporal only
        final_score = 0.70 * avg_fused_score + 0.30 * temporal_score

    final_score = max(0.0, min(100.0, final_score))

    # Small penalty when model and heuristic strongly disagree
    disagreement_penalty = max(0.0, (avg_model_score - avg_heuristic_score - 10.0) * 0.25)
    final_score = max(0.0, final_score - disagreement_penalty)

    avg_score = round(final_score, 2)
    fake_probability = round(max(0.0, min(100.0, 100.0 - avg_score)), 2)
    label = "Fake-like" if fake_probability >= decision_threshold else "Real-like"
    stability_index = round(max(0.0, min(100.0, 100.0 - score_std * 3.0)), 2)

    if len(model_sources) == 1:
        model_source_label = next(iter(model_sources))
    else:
        model_source_label = "mixed"
    if model_source_label == "trained_image_model":
        model_source_label = "trained_image_model+heuristic_fusion"

    warning_flags: list[str] = []
    if disagreement_penalty > 0:
        warning_flags.append("Trained model score disagrees with heuristic texture analysis.")
    if score_std > 20:
        warning_flags.append("Frame-to-frame authenticity is highly unstable.")
    if avg_heuristic_score < 45:
        warning_flags.append("Heuristic frame quality appears suspicious.")
    if avg_flow_score < 45 and len(optical_flow_scores) >= 2:
        warning_flags.append("Optical flow shows unnatural or erratic motion between frames.")
    if temporal_score < 50:
        warning_flags.append("Large authenticity swings detected across video frames.")

    return {
        "authenticity_score": avg_score,
        "fake_probability": fake_probability,
        "label": label,
        "frames_analyzed": len(frame_scores),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 1),
        "frame_score_std": round(score_std, 2),
        "stability_index": stability_index,
        "temporal_consistency_score": round(temporal_score, 2),
        "optical_flow_score": round(avg_flow_score, 2),
        "model_source": model_source_label,
        "decision_threshold": decision_threshold,
        "calibration": {
            "avg_model_score": round(avg_model_score, 2),
            "avg_heuristic_score": round(avg_heuristic_score, 2),
            "avg_fused_score": round(avg_fused_score, 2),
            "temporal_score": round(temporal_score, 2),
            "optical_flow_score": round(avg_flow_score, 2),
            "disagreement_penalty": round(disagreement_penalty, 2),
        },
        "warning_flags": warning_flags,
    }
