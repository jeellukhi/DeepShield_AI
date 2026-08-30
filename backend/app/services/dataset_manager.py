from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _project_root() -> Path:
    # backend/app/services -> app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def _is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _target_dir_for_label(label: str) -> Path:
    return _project_root() / "ml" / "datasets" / "deepfake_images" / label


def _save_image_bytes(file_bytes: bytes, label: str, ext: str) -> str:
    target_dir = _target_dir_for_label(label)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{label}_{uuid4().hex}{ext}"
    target_path = target_dir / filename
    target_path.write_bytes(file_bytes)
    return str(target_path)


def save_labeled_image(file_bytes: bytes, label: str, original_name: str) -> str:
    if label not in {"real", "fake"}:
        raise ValueError("Label must be 'real' or 'fake'.")

    ext = Path(original_name).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image extension.")

    return _save_image_bytes(file_bytes, label, ext)


def import_labeled_images_zip(zip_bytes: bytes) -> dict:
    imported_real = 0
    imported_fake = 0
    skipped = 0

    try:
        zip_file = ZipFile(BytesIO(zip_bytes))
    except BadZipFile as exc:
        raise ValueError("Invalid ZIP file.") from exc

    with zip_file as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue

            normalized_path = item.filename.replace("\\", "/")
            parts = [part.strip() for part in normalized_path.split("/") if part.strip()]
            if not parts:
                skipped += 1
                continue

            detected_label = None
            for part in parts:
                lower_part = part.lower()
                if lower_part in {"real", "fake"}:
                    detected_label = lower_part
                    break

            if detected_label is None:
                skipped += 1
                continue

            file_name = parts[-1]
            if not _is_supported_image(file_name):
                skipped += 1
                continue

            file_bytes = archive.read(item)
            if not file_bytes:
                skipped += 1
                continue

            ext = Path(file_name).suffix.lower()
            _save_image_bytes(file_bytes, detected_label, ext)

            if detected_label == "real":
                imported_real += 1
            else:
                imported_fake += 1

    total_imported = imported_real + imported_fake
    if total_imported == 0:
        raise ValueError("No valid labeled images found. ZIP must include folders named 'real' and/or 'fake'.")

    return {
        "imported_real": imported_real,
        "imported_fake": imported_fake,
        "total_imported": total_imported,
        "skipped_files": skipped,
    }
