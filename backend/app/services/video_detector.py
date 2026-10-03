from __future__ import annotations

import cv2
import numpy as np

from app.services.image_detector import analyze_image_array, analyze_image_frame
from app.services.model_thresholds import get_effective_thresholds


def _temporal_consistency_score(frame_scores: list[float]) -> float:
    """
    Returns 0-100: 100=perfectly consistent frames, 0=wildly erratic.
    Large swings between frames indicate potential deepfake cuts/blending.
    """
    if len(frame_scores) < 3:
        return 75.0
    diffs = [abs(frame_scores[i] - frame_scores[i - 1]) for i in range(1, len(frame_scores))]
    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))
    swing_penalty = min(40.0, mean_diff * 1.0 + max_diff * 0.2)
    return round(max(0.0, 100.0 - swing_penalty), 2)


def analyze_video_authenticity(video_path: str, max_frames: int = 64) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    duration_sec = total_frames / max(fps, 1.0)
    frame_step = max(1, total_frames // max_frames) if total_frames > 0 else 1

    frame_index = 0
    frame_scores: list[float] = []
    face_detected_count = 0
    model_sources: set[str] = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            result = analyze_image_frame(frame)
            auth_score = float(result.get("authenticity_score", 50.0))
            face_found = result.get("face_detected", False)
            src = result.get("model_source", "unknown")

            if face_found:
                # Face detected — CNN result is reliable
                face_detected_count += 1
                frame_scores.append(auth_score)
                model_sources.add(src)
            else:
                # No face — run heuristic only; CNN is unreliable on non-face frames
                heuristic = analyze_image_array(frame)
                h_auth = float(heuristic.get("authenticity_score", 50.0))
                frame_scores.append(h_auth)
                model_sources.add("heuristic_only")

            if len(frame_scores) >= max_frames:
                break

        frame_index += 1

    cap.release()

    if not frame_scores:
        raise ValueError("No readable frames found in the video.")

    # --- Score aggregation ---
    threshold_cfg = get_effective_thresholds()
    decision_threshold = float(threshold_cfg.get("values", {}).get("video_fake_probability_threshold", 45.0))
    decision_threshold = max(20.0, min(80.0, decision_threshold))

    median_score = float(np.median(frame_scores))
    mean_score = float(np.mean(frame_scores))
    score_std = float(np.std(frame_scores))

    temporal_score = _temporal_consistency_score(frame_scores)

    # 80% image score + 20% temporal consistency
    final_score = 0.80 * median_score + 0.20 * temporal_score
    final_score = round(max(0.0, min(100.0, final_score)), 2)

    fake_probability = round(max(0.0, min(100.0, 100.0 - final_score)), 2)
    label = "Fake-like" if fake_probability >= decision_threshold else "Real-like"
    stability_index = round(max(0.0, min(100.0, 100.0 - score_std * 2.5)), 2)

    face_detection_rate = round(face_detected_count / max(len(frame_scores), 1) * 100.0, 1)
    model_source_label = next(iter(model_sources)) if len(model_sources) == 1 else "mixed_cnn+heuristic"

    warning_flags: list[str] = []
    if face_detection_rate < 30:
        warning_flags.append(
            f"Only {face_detection_rate}% of frames had a detectable face. "
            "Results may be inaccurate — the model works best with face-forward videos."
        )
    if score_std > 25:
        warning_flags.append("Frame-to-frame authenticity is highly unstable.")
    if temporal_score < 50:
        warning_flags.append("Large authenticity swings detected across video frames.")
    if len(frame_scores) < 5:
        warning_flags.append("Very few frames analyzed — result may be unreliable.")

    return {
        "authenticity_score": final_score,
        "fake_probability": fake_probability,
        "label": label,
        "frames_analyzed": len(frame_scores),
        "face_detected_frames": face_detected_count,
        "face_detection_rate_pct": face_detection_rate,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 1),
        "frame_score_std": round(score_std, 2),
        "stability_index": stability_index,
        "temporal_consistency_score": round(temporal_score, 2),
        "model_source": model_source_label,
        "decision_threshold": decision_threshold,
        "calibration": {
            "median_frame_score": round(median_score, 2),
            "mean_frame_score": round(mean_score, 2),
            "temporal_score": round(temporal_score, 2),
        },
        "warning_flags": warning_flags,
    }
