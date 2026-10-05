"""
Celeb-DF v2 Frame Extractor
============================
This script extracts face crops from Celeb-DF video files and saves them
as JPG images in real/ and fake/ folders — ready for CNN training.

Usage:
    python celeb_df_extractor.py --celeb_df_root <path_to_celeb_df> --output_root <output_path>

Celeb-DF v2 folder structure expected:
    Celeb-DF-v2/
        Celeb-real/         <- real celebrity videos
        YouTube-real/       <- additional real videos
        Celeb-synthesis/    <- deepfake videos (face-swapped)
        List_of_testing_videos.txt
"""
import argparse
import pathlib
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def detect_and_crop_face(frame: np.ndarray, min_size: int = 80) -> np.ndarray | None:
    """Detect largest face, add 20% padding, return crop or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * 0.20), int(h * 0.20)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def extract_faces_from_video(
    video_path: pathlib.Path,
    output_dir: pathlib.Path,
    max_frames: int = 20,
    prefix: str = "",
) -> int:
    """Extract up to max_frames face crops from a single video. Returns count saved."""
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
                fname = output_dir / f"{prefix}_{video_path.stem}_f{frame_idx:05d}.jpg"
                cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
        frame_idx += 1

    cap.release()
    return saved


def find_videos(folder: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_VIDEO_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from Celeb-DF v2 videos")
    parser.add_argument("--celeb_df_root", required=True,
                        help="Path to Celeb-DF-v2 root folder")
    parser.add_argument("--output_root", required=True,
                        help="Output folder (will create real/ and fake/ subfolders)")
    parser.add_argument("--frames_per_video", type=int, default=20,
                        help="Max face frames to extract per video (default: 20)")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos per class, 0=all (default: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    args = parser.parse_args()

    root = pathlib.Path(args.celeb_df_root)
    out_root = pathlib.Path(args.output_root)

    out_real = out_root / "real"
    out_fake = out_root / "fake"
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    # Celeb-DF v2 real folders
    real_folders = ["Celeb-real", "YouTube-real"]
    fake_folders = ["Celeb-synthesis"]

    real_videos = []
    for folder_name in real_folders:
        folder = root / folder_name
        if folder.exists():
            real_videos.extend(find_videos(folder))
            print(f"  Found {len(find_videos(folder))} real videos in {folder_name}")

    fake_videos = []
    for folder_name in fake_folders:
        folder = root / folder_name
        if folder.exists():
            fake_videos.extend(find_videos(folder))
            print(f"  Found {len(find_videos(folder))} fake videos in {folder_name}")

    if args.max_videos > 0:
        real_videos = real_videos[:args.max_videos]
        fake_videos = fake_videos[:args.max_videos]

    print(f"\nExtracting from {len(real_videos)} real + {len(fake_videos)} fake videos")
    print(f"Frames per video: {args.frames_per_video}")
    print(f"Output: {out_root}")
    print()

    def process_video(video_path, out_dir, label):
        count = extract_faces_from_video(video_path, out_dir,
                                          max_frames=args.frames_per_video,
                                          prefix=label)
        return video_path.name, count

    total_real = 0
    total_fake = 0

    print("=== Extracting REAL faces ===")
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_video, v, out_real, "real"): v for v in real_videos}
        for i, fut in enumerate(as_completed(futures), 1):
            name, count = fut.result()
            total_real += count
            if i % 10 == 0 or i == len(real_videos):
                print(f"  {i}/{len(real_videos)} videos done, {total_real} faces saved")

    print(f"\n=== Extracting FAKE faces ===")
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_video, v, out_fake, "fake"): v for v in fake_videos}
        for i, fut in enumerate(as_completed(futures), 1):
            name, count = fut.result()
            total_fake += count
            if i % 10 == 0 or i == len(fake_videos):
                print(f"  {i}/{len(fake_videos)} videos done, {total_fake} faces saved")

    print(f"\n=== DONE ===")
    print(f"Real faces saved: {total_real} -> {out_real}")
    print(f"Fake faces saved: {total_fake} -> {out_fake}")
    print(f"\nNext step: run cnn_trainer.py pointing to {out_root}")


if __name__ == "__main__":
    main()
