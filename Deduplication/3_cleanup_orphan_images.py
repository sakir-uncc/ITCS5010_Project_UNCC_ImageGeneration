import os
from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

DEDUP_ROOT = DATASET_ROOT / "processed_dedup"
METADATA = DATASET_ROOT / "metadata_rewritten_dedup_final.csv"

def main():

    df = pd.read_csv(METADATA)
    meta_paths = set(df["img_path"].astype(str))

    orphan_paths = []

    for p in DEDUP_ROOT.rglob("*.jpg"):
        rel = str(p.relative_to(DEDUP_ROOT))
        if rel not in meta_paths:
            orphan_paths.append(p)

    print(f"Found {len(orphan_paths)} orphan images.")
    for o in orphan_paths[:10]:
        print("  ", o)

    print("\nRemoving orphan images...")
    for o in orphan_paths:
        o.unlink()

    print("Done! All images in processed_dedup now have metadata.")

if __name__ == "__main__":
    main()
