import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

PROCESSED_DEDUP_ROOT = DATASET_ROOT / "processed_dedup"
METADATA_TOKENIZED = DATASET_ROOT / "metadata_tokenized.csv"

OUT_ROOT = DATASET_ROOT / "sd_lora_dataset"
OUT_IMAGES = OUT_ROOT / "images"
OUT_CAPTIONS = OUT_ROOT / "captions"
OUT_REPORT = OUT_ROOT / "dataset_report.txt"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def make_flat_filename(rel_path: str) -> str:
    """
    Turn a relative path like:
        'academic_buildings/Storrs_Hall/001100.jpg'
    into:
        'academic_buildings_Storrs_Hall_001100.jpg'
    preserving capitalization and underscores.
    """
    parts = rel_path.split("/")
    if len(parts) < 3:
        # fallback: just replace slashes
        return rel_path.replace("/", "_")

    category = parts[0]         # e.g. academic_buildings
    cls = parts[1]              # e.g. Storrs_Hall
    filename = parts[-1]        # e.g. 001100.jpg

    stem = Path(filename).stem  # '001100'
    ext = Path(filename).suffix # '.jpg'

    flat_name = f"{category}_{cls}_{stem}{ext}"
    return flat_name


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Building SD LoRA dataset...")
    print("Dataset root:", DATASET_ROOT)
    print("Using images from:", PROCESSED_DEDUP_ROOT)
    print("Using captions from:", METADATA_TOKENIZED)
    print("Output dataset folder:", OUT_ROOT)
    print()

    # Create output folders
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_CAPTIONS.mkdir(parents=True, exist_ok=True)

    # Load tokenized metadata
    df = pd.read_csv(METADATA_TOKENIZED)
    if not {"img_path", "caption_tokenized"}.issubset(df.columns):
        raise ValueError("metadata_tokenized.csv must contain 'img_path' and 'caption_tokenized' columns.")

    total_rows = len(df)
    print(f"Loaded {total_rows} rows from metadata_tokenized.csv")

    # Stats
    copied = 0
    missing_images = []
    per_class_counts = defaultdict(int)

    # Process each row
    for _, row in tqdm(df.iterrows(), total=total_rows, desc="Processing images+captions"):
        rel_path = str(row["img_path"])
        caption = str(row["caption_tokenized"])

        src_img = PROCESSED_DEDUP_ROOT / rel_path
        if not src_img.exists():
            missing_images.append(rel_path)
            continue

        # Build flat filename
        flat_name = make_flat_filename(rel_path)
        img_out_path = OUT_IMAGES / flat_name
        txt_out_path = OUT_CAPTIONS / (Path(flat_name).stem + ".txt")

        # Copy image
        shutil.copy2(src_img, img_out_path)

        # Write caption file
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(caption.strip() + "\n")

        copied += 1

        # Collect per-class stats (by folder)
        class_key = "/".join(rel_path.split("/")[:2])  # e.g. academic_buildings/Storrs_Hall
        per_class_counts[class_key] += 1

    # -----------------------------------------------------------------
    # WRITE REPORT
    # -----------------------------------------------------------------
    print("\nWriting dataset_report.txt...")
    lines = []
    lines.append("Stable Diffusion LoRA Training Dataset Report")
    lines.append("===========================================")
    lines.append(f"Dataset root: {DATASET_ROOT}")
    lines.append(f"Processed images source: {PROCESSED_DEDUP_ROOT}")
    lines.append(f"Tokenized metadata: {METADATA_TOKENIZED}")
    lines.append(f"Output dataset: {OUT_ROOT}")
    lines.append("")
    lines.append(f"Total metadata rows: {total_rows}")
    lines.append(f"Total images successfully copied: {copied}")
    lines.append(f"Total missing images (present in CSV but not in processed_dedup/): {len(missing_images)}")
    lines.append("")

    if missing_images:
        lines.append("First few missing image paths:")
        for mp in missing_images[:20]:
            lines.append(f"  - {mp}")
        lines.append("")

    lines.append("Per-class image counts in sd_lora_dataset:")
    for cls, n in sorted(per_class_counts.items()):
        lines.append(f"  - {cls}: {n} images")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("Report written to:", OUT_REPORT)

    print("\nDone.")
    print(f"Images folder:   {OUT_IMAGES}")
    print(f"Captions folder: {OUT_CAPTIONS}")


if __name__ == "__main__":
    main()
