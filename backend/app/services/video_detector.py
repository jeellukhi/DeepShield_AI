import cv2
import numpy as np

from app.services.image_detector import analyze_image_array, analyze_image_frame
from app.services.model_thresholds import get_effective_thresholds


def analyze_video_authenticity(video_path: str, max_frames: int = 24) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_step = max(1, total_frames // max_frames) if total_frames > 0 else 1

    frame_index = 0
    frame_scores: list[float] = []
    frame_scores_model: list[float] = []
    frame_scores_heuristic: list[float] = []
    model_sources: set[str] = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            model_analysis = analyze_image_frame(frame)
            heuristic_analysis = analyze_image_array(frame)

            model_score = float(model_analysis["authenticity_score"])
            heuristic_score = float(heuristic_analysis["authenticity_score"])
            model_source = str(model_analysis.get("model_source", "unknown"))
            model_sources.add(model_source)

            if model_source == "trained_image_model":
                # Conservative fusion to reduce false negatives on manipulated videos.
                fused_score = (0.65 * model_score) + (0.35 * heuristic_score)
            else:
                fused_score = heuristic_score

            frame_scores_model.append(model_score)
            frame_scores_heuristic.append(heuristic_score)
            frame_scores.append(fused_score)
            if len(frame_scores) >= max_frames:
                break

        frame_index += 1

    cap.release()

    if not frame_scores:
        raise ValueError("No readable frames found in the video.")

    threshold_cfg = get_effective_thresholds()
    decision_threshold = float(threshold_cfg.get("values", {}).get("video_fake_probability_threshold", 55.0))
    decision_threshold = max(20.0, min(80.0, decision_threshold))
    avg_model_score = float(np.mean(frame_scores_model))
    avg_heuristic_score = float(np.mean(frame_scores_heuristic))
    avg_fused_score = float(np.mean(frame_scores))
    score_std = float(np.std(frame_scores))

    disagreement_penalty = max(0.0, (avg_model_score - avg_heuristic_score - 8.0) * 0.35)
    instability_penalty = max(0.0, (score_std - 12.0) * 1.25)
    heuristic_suspicion_penalty = max(0.0, (55.0 - avg_heuristic_score) * 0.45)
    final_score = max(
        0.0,
        min(100.0, avg_fused_score - disagreement_penalty - instability_penalty - heuristic_suspicion_penalty),
    )

    avg_score = round(final_score, 2)
    raw_fake_probability = max(0.0, min(100.0, 100.0 - avg_score))
    risk_boost = max(0.0, (disagreement_penalty * 0.18) + (instability_penalty * 0.25) + (heuristic_suspicion_penalty * 0.22))
    calibrated_fake_probability = max(0.0, min(100.0, raw_fake_probability + risk_boost))
    fake_probability = round(calibrated_fake_probability, 2)
    raw_fake_probability = round(raw_fake_probability, 2)
    label = "Fake-like" if fake_probability >= decision_threshold else "Real-like"
    stability_index = round(max(0.0, min(100.0, 100 - (score_std * 4))), 2)
    if len(model_sources) == 1:
        model_source = next(iter(model_sources))
    else:
        model_source = "mixed"

    if model_source == "trained_image_model":
        model_source = "trained_image_model+heuristic_fusion"

    warning_flags: list[str] = []
    if disagreement_penalty > 0:
        warning_flags.append("Trained score disagrees with heuristic artifacts.")
    if instability_penalty > 0:
        warning_flags.append("Frame-to-frame authenticity is unstable.")
    if avg_heuristic_score < 50:
        warning_flags.append("Heuristic frame quality appears suspicious.")

    return {
        "authenticity_score": avg_score,
        "fake_probability": fake_probability,
        "label": label,
        "frames_analyzed": len(frame_scores),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "frame_score_std": round(score_std, 2),
        "stability_index": stability_index,
        "model_source": model_source,
        "decision_threshold": decision_threshold,
        "raw_fake_probability": raw_fake_probability,
        "base_fake_probability": raw_fake_probability,
        "risk_boost": round(risk_boost, 2),
        "calibration": {
            "avg_model_score": round(avg_model_score, 2),
            "avg_heuristic_score": round(avg_heuristic_score, 2),
            "avg_fused_score": round(avg_fused_score, 2),
            "disagreement_penalty": round(disagreement_penalty, 2),
            "instability_penalty": round(instability_penalty, 2),
            "heuristic_suspicion_penalty": round(heuristic_suspicion_penalty, 2),
        },
        "warning_flags": warning_flags,
    }
