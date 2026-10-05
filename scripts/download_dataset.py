"""
Dataset Download Helper for DeepShield AI
==========================================
Downloads additional datasets to improve detection accuracy.

Available datasets:
  1. real-vs-fake-faces-10k  — Small diverse face dataset (287 MB, Kaggle)
  2. Instructions for Celeb-DF v2 video dataset

Requirements:
  pip install kaggle
  Set up Kaggle API key: https://www.kaggle.com/settings -> API -> Create New Token
  Save kaggle.json to C:\\Users\\<you>\\.kaggle\\kaggle.json
"""

import argparse
import pathlib
import subprocess
import sys
import shutil


def check_kaggle_installed():
    try:
        import kaggle
        return True
    except ImportError:
        print("Kaggle not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        return True


def check_kaggle_credentials():
    import os
    cred_path = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print("\n ERROR: Kaggle API key not found!")
        print(f"  Expected at: {cred_path}")
        print("\n  Steps to set up:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'API' section -> 'Create New Token'")
        print("  3. Save the downloaded kaggle.json to:", cred_path)
        print("  4. Run this script again")
        return False
    return True


def download_real_vs_fake_10k(output_dir: pathlib.Path):
    """Download Real vs Fake Faces 10k dataset from Kaggle (~287 MB)."""
    import kaggle
    dataset_id = "sachchitkunichetty/real-vs-fake-faces-10k"
    print(f"\n Downloading: Real vs Fake Faces 10k ({dataset_id})")
    print(f" Output: {output_dir}")
    print(" Size: ~287 MB")
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_and_extract(dataset_id, path=str(output_dir))
    print(" Download complete!")

    # Find and show structure
    print("\n Downloaded files:")
    for p in sorted(output_dir.rglob("*"))[:20]:
        if p.is_dir():
            count = len(list(p.glob("*.*")))
            print(f"   {p.relative_to(output_dir)}/ ({count} files)")


def merge_into_training_dataset(source_dir: pathlib.Path, dest_base: pathlib.Path, max_per_class: int = 5000):
    """
    Merge a new dataset's real/ and fake/ folders into the main training dataset.
    Copies up to max_per_class images from each class.
    """
    for class_name in ["real", "fake"]:
        src = source_dir / class_name
        if not src.exists():
            # Try nested structure
            matches = list(source_dir.rglob(class_name))
            if matches:
                src = matches[0]
            else:
                print(f"  WARNING: No {class_name}/ folder found in {source_dir}")
                continue

        dst = dest_base / class_name
        dst.mkdir(parents=True, exist_ok=True)

        files = [f for f in src.rglob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        files = files[:max_per_class]
        copied = 0
        for f in files:
            dest_file = dst / f"extra_{f.name}"
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
                copied += 1
        print(f"  Copied {copied} {class_name} images to {dst}")


def print_celebdf_instructions():
    print("""
 ============================================================
  Celeb-DF v2 Download Instructions (for VIDEO detection)
 ============================================================

 Celeb-DF v2 is a face-swap deepfake video dataset (~2 GB).
 It requires official registration (free).

 STEP 1: Request access
   Go to: https://github.com/yuezunli/celeb-deepfakeforensics
   Fill the Google Form linked on that page.
   You will receive a download link via email (usually same day).

 STEP 2: Download the dataset
   Download all 3 ZIP files:
     - Celeb-real.zip       (~590 real videos)
     - Celeb-synthesis.zip  (~5639 deepfake videos)
     - YouTube-real.zip     (~300 real videos)

 STEP 3: Extract to a folder, e.g.:
   C:\\Users\\HP\\Desktop\\Celeb-DF-v2\\
     Celeb-real\\
     Celeb-synthesis\\
     YouTube-real\\

 STEP 4: Run the frame extractor:
   cd "C:\\Users\\HP\\Desktop\\DeepShield_AI"
   .\\backend\\.venv\\Scripts\\python.exe scripts\\celeb_df_extractor.py `
     --celeb_df_root "C:\\Users\\HP\\Desktop\\Celeb-DF-v2" `
     --output_root "ml\\datasets\\celebdf_faces" `
     --frames_per_video 20

 STEP 5: Train the video model:
   .\\backend\\.venv\\Scripts\\python.exe -c "
   import sys; sys.path.insert(0, 'backend')
   from app.services.cnn_trainer import train_cnn_model
   import json
   result = train_cnn_model(
       dataset_root='ml/datasets/celebdf_faces',
       model_save_name='video_cnn.pkl',
       max_per_class=5000, epochs=10
   )
   print(json.dumps(result['metrics'], indent=2))
   "

 STEP 6: Done! The video detector will automatically use video_cnn.pkl
         if it exists, instead of the image model.

 ============================================================
""")


def main():
    parser = argparse.ArgumentParser(description="DeepShield dataset download helper")
    parser.add_argument("--dataset", choices=["real-vs-fake-10k", "celebdf-info"],
                        default="celebdf-info", help="Dataset to download")
    parser.add_argument("--output", default="ml/datasets/extra_faces",
                        help="Output directory for downloaded dataset")
    parser.add_argument("--merge", action="store_true",
                        help="Merge downloaded dataset into main training dataset")
    args = parser.parse_args()

    if args.dataset == "celebdf-info":
        print_celebdf_instructions()
        return

    if args.dataset == "real-vs-fake-10k":
        check_kaggle_installed()
        if not check_kaggle_credentials():
            return
        output_dir = pathlib.Path(args.output)
        download_real_vs_fake_10k(output_dir)
        if args.merge:
            print("\n Merging into main training dataset...")
            main_ds = pathlib.Path("ml/datasets/deepfake_images")
            merge_into_training_dataset(output_dir, main_ds, max_per_class=3000)
            print("\n Done! Retrain the model for improved accuracy.")


if __name__ == "__main__":
    main()
