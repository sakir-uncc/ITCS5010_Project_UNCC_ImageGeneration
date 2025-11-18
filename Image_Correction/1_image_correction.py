import os
from pathlib import Path
import traceback
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
#                 CONFIGURATION VARIABLES
# ============================================================

# Root folder of your project dataset
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

# Output folder
PROCESSED_ROOT = DATASET_ROOT / "processed"

# Rotation angle (+2 degrees CCW = best from your tests)
ROTATION_DEG = 2.0

# Horizontal crop width (keep 86% of width)
CROP_WIDTH_RATIO = 0.86

# Final resize resolution (Stable Diffusion 1.5 optimal)
TARGET_WIDTH = 768
TARGET_HEIGHT = 512

# Number of workers for multiprocessing
NUM_WORKERS = 8

# Whether to rewrite CSV paths (set to CSV path or None)
CSV_TO_REWRITE = None
# Example:
# CSV_TO_REWRITE = DATASET_ROOT / "metadata_rewritten.csv"


# ============================================================
#                 IMAGE PROCESSING FUNCTION
# ============================================================

def process_single_image(
    src_path_str: str,
    src_root_str: str,
    dst_root_str: str,
    rotation_degrees: float,
    crop_width_ratio: float,
    target_width: int,
    target_height: int,
):
    """Rotate, crop, resize, and save a single image."""
    src_root = Path(src_root_str)
    dst_root = Path(dst_root_str)
    src_path = Path(src_path_str)

    rel_path = src_path.relative_to(src_root)
    dst_path = dst_root / rel_path

    img = cv2.imread(str(src_path))
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {src_path}")

    h, w = img.shape[:2]

    # ---- 1) Rotation ----
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # ---- 2) Horizontal center crop ----
    new_w = int(rotated.shape[1] * crop_width_ratio)
    x1 = (rotated.shape[1] - new_w) // 2
    x2 = x1 + new_w
    cropped = rotated[:, x1:x2]

    # ---- 3) Resize ----
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # ---- 4) Save ----
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(dst_path), resized)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for {dst_path}")

    return str(rel_path)


# ============================================================
#       HELPERS: FILE COLLECTION + OPTIONAL CSV REWRITE
# ============================================================

def collect_image_paths(dataset_root: Path, processed_root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []

    for p in dataset_root.rglob("*"):
        if p.is_dir(): 
            continue
        if p.suffix.lower() not in exts:
            continue
        if processed_root in p.parents:
            continue
        image_paths.append(p)

    return image_paths


def rewrite_two_column_csv_paths(csv_in: Path, processed_prefix="processed/"):
    """CSV format: image_path, caption (2 columns)."""
    csv_out = csv_in.with_name(csv_in.stem + "_processed.csv")

    import csv
    with csv_in.open("r", encoding="utf-8", newline="") as fin, \
         csv_out.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            if not row:
                continue
            new_path = processed_prefix + row[0].lstrip("./")
            writer.writerow([new_path] + row[1:])

    print(f"CSV rewritten → {csv_out}")


# ============================================================
#                           MAIN
# ============================================================

def main():
    processed_root = PROCESSED_ROOT
    processed_root.mkdir(exist_ok=True)

    print("\n===== UNC CHARLOTTE DATASET PREPROCESSING =====")
    print(f"Dataset root:   {DATASET_ROOT}")
    print(f"Processed root: {processed_root}")
    print(f"Rotation:       {ROTATION_DEG}° CCW")
    print(f"Crop width:     {CROP_WIDTH_RATIO}")
    print(f"Output size:    {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Workers:        {NUM_WORKERS}")
    print("================================================\n")

    # --- Collect images ---
    image_paths = sorted(collect_image_paths(DATASET_ROOT, processed_root))
    if not image_paths:
        print("No images found! Check dataset path.")
        return

    print(f"Discovered {len(image_paths)} images.\n")

    # --- Error logging ---
    error_log = open("preprocess_errors.log", "w", encoding="utf-8")

    use_tqdm = tqdm is not None
    pbar = tqdm(total=len(image_paths), desc="Processing", unit="img") if use_tqdm else None

    processed = 0
    failed = 0

    # --- Parallel processing ---
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {
            ex.submit(
                process_single_image,
                str(p),
                str(DATASET_ROOT),
                str(processed_root),
                ROTATION_DEG,
                CROP_WIDTH_RATIO,
                TARGET_WIDTH,
                TARGET_HEIGHT,
            ): p
            for p in image_paths
        }

        for fut in as_completed(futures):
            path = futures[fut]
            try:
                fut.result()
                processed += 1
            except Exception as e:
                failed += 1
                error_log.write(f"\nFAILED: {path}\n{e}\n{traceback.format_exc()}\n")
            finally:
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()
    error_log.close()

    # --- Summary ---
    print("\nDONE.")
    print(f"Processed successfully: {processed}")
    print(f"Failed: {failed}")
    print("Errors (if any) → preprocess_errors.log")

    # --- CSV rewriting ---
    if CSV_TO_REWRITE:
        print("\nRewriting CSV paths...")
        rewrite_two_column_csv_paths(CSV_TO_REWRITE)


if __name__ == "__main__":
    main()
