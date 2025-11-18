import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

PROCESSED_ROOT = DATASET_ROOT / "processed"
DEDUP_ROOT = DATASET_ROOT / "processed_dedup"

REMOVED_CSV = DATASET_ROOT / "dedup_removed_images.csv"
METADATA_IN = DATASET_ROOT / "metadata_rewritten.csv"
METADATA_OUT = DATASET_ROOT / "metadata_rewritten_dedup_final.csv"

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    print("Reading removed image list...")
    removed_df = pd.read_csv(REMOVED_CSV)
    removed_paths = set(removed_df["image_path_processed"].astype(str))

    print("Building list of kept images...")
    all_images = []
    for p in PROCESSED_ROOT.rglob("*.jpg"):
        rel = p.relative_to(PROCESSED_ROOT)
        all_images.append(rel)

    kept_images = [p for p in all_images if str(p) not in removed_paths]
    print(f"Total images: {len(all_images)}, kept: {len(kept_images)}, removed: {len(removed_paths)}")

    # -----------------------------------------------------------------
    # COPY KEPT IMAGES TO NEW FOLDER
    # -----------------------------------------------------------------
    print("\nCreating new deduplicated folder:", DEDUP_ROOT)
    if not DEDUP_ROOT.exists():
        DEDUP_ROOT.mkdir(parents=True)

    for rel in tqdm(kept_images, desc="Copying kept images"):
        src = PROCESSED_ROOT / rel
        dst = DEDUP_ROOT / rel

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # -----------------------------------------------------------------
    # BUILD NEW METADATA
    # -----------------------------------------------------------------
    print("\nLoading metadata:", METADATA_IN)
    meta = pd.read_csv(METADATA_IN)

    # Convert metadata img_path column to match relative paths (no "processed/" prefix)
    meta_paths = meta["img_path"].astype(str)

    kept_set = set(str(p) for p in kept_images)

    print("Filtering metadata...")
    meta_dedup = meta[meta_paths.isin(kept_set)].copy()
    print(f"Metadata rows before: {len(meta)}, after: {len(meta_dedup)}")

    meta_dedup.to_csv(METADATA_OUT, index=False)
    print("\nSaved deduplicated metadata to:", METADATA_OUT)

    # -----------------------------------------------------------------
    # FINAL CONSISTENCY CHECKS
    # -----------------------------------------------------------------
    print("\nPerforming sanity checks...")

    missing_images = []
    for rel in meta_dedup["img_path"]:
        if not (DEDUP_ROOT / rel).exists():
            missing_images.append(rel)

    if missing_images:
        print("WARNING: metadata refers to missing images!")
        for m in missing_images[:10]:
            print("  ", m)
    else:
        print("✓ Every metadata entry has a corresponding image.")

    orphan_images = []
    meta_set = set(meta_dedup["img_path"])
    for p in DEDUP_ROOT.rglob("*.jpg"):
        rel = str(p.relative_to(DEDUP_ROOT))
        if rel not in meta_set:
            orphan_images.append(rel)

    if orphan_images:
        print("WARNING: Some images have no metadata entries!")
        for o in orphan_images[:10]:
            print("  ", o)
    else:
        print("✓ Every image has a metadata entry.")

    print("\nDone!")


if __name__ == "__main__":
    main()
