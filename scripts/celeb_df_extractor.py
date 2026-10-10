"""
Celeb-DF v2 Frame Extractor (v2 — thread-safe, with resume support)
====================================================================
Extracts face crops from Celeb-DF video files and saves them as JPG images
in real/ and fake/ folders — ready for CNN training.

Fixes vs v1:
  - Thread-safe: cascade created inside each thread (not shared global)
  - Minimum image size guard before detectMultiScale
  - try/except around every OpenCV call (no crash on bad frame)
  - Resume support: skip videos whose output already exists
  - Better progress reporting

Usage:
    python celeb_df_extractor.py --celeb_df_root <path> --output_root <path>
"""
import argparse
import pathlib
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def detect_and_crop_face(frame: np.ndarray, min_face_px: int = 60) -> np.ndarray | None:
    """
    Detect largest face and return crop with 20% padding.
    Thread-safe: creates its own CascadeClassifier per call.
    Returns None if no face found or frame too small.
    """
    try:
        h, w = frame.shape[:2]
        # Skip frames too small for the cascade to work reliably
        if h < 100 or w < 100:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create cascade inside the function — NOT a shared global (thread-safe)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_face_px, min_face_px),
        )

        if len(faces) == 0:
            return None

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_x, pad_y = int(fw * 0.20), int(fh * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    except cv2.error:
        return None
    except Exception:
        return None


def extract_faces_from_video(
    video_path: pathlib.Path,
    output_dir: pathlib.Path,
    max_frames: int = 15,
    prefix: str = "",
    existing_stems: set | None = None,
) -> int:
    """
    Extract up to max_frames face crops from a single video.
    Skips if output files already exist (resume support).
    Returns count of newly saved faces.
    """
    # Resume: skip if we already have files for this video
    if existing_stems and video_path.stem in existing_stems:
        return 0

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, total // max_frames) if total > 0 else 1
        saved = 0
        frame_idx = 0

        while saved < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                crop = detect_and_crop_face(frame)
                if crop is not None:
                    fname = (
                        output_dir
                        / f"{prefix}_{video_path.stem}_f{frame_idx:05d}.jpg"
                    )
                    cv2.imwrite(
                        str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    saved += 1
            frame_idx += 1

        cap.release()
        return saved

    except Exception:
        return 0


def find_videos(folder: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from Celeb-DF v2 videos"
    )
    parser.add_argument(
        "--celeb_df_root", required=True, help="Path to Celeb-DF-v2 root folder"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output folder (creates real/ and fake/ subfolders)",
    )
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=15,
        help="Max face frames to extract per video (default: 15)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="Max videos per class, 0=all (default: 0=all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers (default: 2 — reduced to avoid OpenCV memory issues)",
    )
    parser.add_argument(
        "--skip_real",
        action="store_true",
        help="Skip real video extraction (use if already done)",
    )
    parser.add_argument(
        "--skip_fake",
        action="store_true",
        help="Skip fake video extraction (use if already done)",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.celeb_df_root)
    out_root = pathlib.Path(args.output_root)
    out_real = out_root / "real"
    out_fake = out_root / "fake"
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    # Celeb-DF v2 folder mapping
    real_folder_names = ["Celeb-real", "YouTube-real"]
    fake_folder_names = ["Celeb-synthesis"]

    real_videos: list[pathlib.Path] = []
    for folder_name in real_folder_names:
        folder = root / folder_name
        if folder.exists():
            vids = find_videos(folder)
            real_videos.extend(vids)
            print(f"  Found {len(vids)} real videos in {folder_name}")

    fake_videos: list[pathlib.Path] = []
    for folder_name in fake_folder_names:
        folder = root / folder_name
        if folder.exists():
            vids = find_videos(folder)
            fake_videos.extend(vids)
            print(f"  Found {len(vids)} fake videos in {folder_name}")

    if args.max_videos > 0:
        real_videos = real_videos[: args.max_videos]
        fake_videos = fake_videos[: args.max_videos]

    print(
        f"\nExtracting from {len(real_videos)} real + {len(fake_videos)} fake videos"
    )
    print(f"Frames per video: {args.frames_per_video}")
    print(f"Workers: {args.workers}")
    print(f"Output: {out_root}")
    print()

    # Build resume sets (stems of videos already processed)
    def get_existing_stems(out_dir: pathlib.Path) -> set:
        stems = set()
        for f in out_dir.glob("*.jpg"):
            # filename format: prefix_videostem_fNNNNN.jpg
            # stem of original video is between first _ and last _fNNNNN
            parts = f.stem.split("_")
            # rejoin everything except first (prefix) and last (frame number)
            if len(parts) >= 3:
                stems.add("_".join(parts[1:-1]))
        return stems

    def process_video(video_path, out_dir, label, existing_stems):
        count = extract_faces_from_video(
            video_path,
            out_dir,
            max_frames=args.frames_per_video,
            prefix=label,
            existing_stems=existing_stems,
        )
        return video_path.name, count

    # ── REAL ─────────────────────────────────────────────────────────────────
    if not args.skip_real:
        existing_real = get_existing_stems(out_real)
        existing_real_count = len(list(out_real.glob("*.jpg")))
        print(f"=== Extracting REAL faces (resume: {existing_real_count} already saved) ===")
        total_real = existing_real_count
        done = 0

        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {
                exe.submit(process_video, v, out_real, "real", existing_real): v
                for v in real_videos
            }
            for fut in as_completed(futures):
                try:
                    _, count = fut.result()
                    total_real += count
                    done += 1
                    if done % 50 == 0 or done == len(real_videos):
                        print(
                            f"  {done}/{len(real_videos)} videos done,"
                            f" {total_real} real faces saved",
                            flush=True,
                        )
                except Exception as e:
                    done += 1
                    print(f"  [SKIP] Error on video: {e}")

        print(f"  REAL done: {total_real} faces\n")
    else:
        total_real = len(list(out_real.glob("*.jpg")))
        print(f"=== Skipping REAL (already have {total_real} faces) ===\n")

    # ── FAKE ─────────────────────────────────────────────────────────────────
    if not args.skip_fake:
        existing_fake = get_existing_stems(out_fake)
        existing_fake_count = len(list(out_fake.glob("*.jpg")))
        print(f"=== Extracting FAKE faces (resume: {existing_fake_count} already saved) ===")
        total_fake = existing_fake_count
        done = 0

        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {
                exe.submit(process_video, v, out_fake, "fake", existing_fake): v
                for v in fake_videos
            }
            for fut in as_completed(futures):
                try:
                    _, count = fut.result()
                    total_fake += count
                    done += 1
                    if done % 100 == 0 or done == len(fake_videos):
                        print(
                            f"  {done}/{len(fake_videos)} videos done,"
                            f" {total_fake} fake faces saved",
                            flush=True,
                        )
                except Exception as e:
                    done += 1
                    print(f"  [SKIP] Error on video: {e}")

        print(f"  FAKE done: {total_fake} faces\n")
    else:
        total_fake = len(list(out_fake.glob("*.jpg")))
        print(f"=== Skipping FAKE (already have {total_fake} faces) ===\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"  Real faces: {total_real}  ->  {out_real}")
    print(f"  Fake faces: {total_fake}  ->  {out_fake}")
    print()
    print("Next step — train the video CNN:")
    print(
        f'  python -c "import sys; sys.path.insert(0,\'backend\'); '
        f"from app.services.cnn_trainer import train_cnn_model; import json; "
        f"r=train_cnn_model(dataset_root=r'{out_root}', "
        f"model_save_name='video_cnn.pkl', max_per_class=5000, epochs=10); "
        f'print(json.dumps(r[\'metrics\'], indent=2))"'
    )


if __name__ == "__main__":
    main()
