import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

METADATA_ORIGINAL = DATASET_ROOT / "metadata.csv"
METADATA_REWRITTEN = DATASET_ROOT / "metadata_rewritten.csv"
METADATA_FINAL = DATASET_ROOT / "metadata_rewritten_dedup_final.csv"

DEDUP_SUMMARY = DATASET_ROOT / "dedup_summary.csv"
DEDUP_REMOVED = DATASET_ROOT / "dedup_removed_images.csv"

RAW_DIRS = [
    DATASET_ROOT / "academic_buildings",
    DATASET_ROOT / "administrative_buildings",
    DATASET_ROOT / "athletic_facilities",
    DATASET_ROOT / "monuments",
    DATASET_ROOT / "student_life_building",
]

PROCESSED_DIR = DATASET_ROOT / "processed"
DEDUP_DIR = DATASET_ROOT / "processed_dedup"

REPORT_PATH = DATASET_ROOT / "dataset_preparation_report.txt"


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def count_images_by_class(root_dir: Path):
    """
    Return dict class_path(str) -> count for all jpg under root_dir.
    class_path is relative path parent of the file.
    """
    counts = defaultdict(int)
    for p in root_dir.rglob("*.jpg"):
        rel = p.relative_to(root_dir)
        class_key = str(rel.parent)  # e.g. academic_buildings/Fretwell_hall
        counts[class_key] += 1
    return counts


def group_by_top_category(class_counts):
    """
    From class_counts: 'academic_buildings/Fretwell_hall' -> n
    Returns dict top_category -> (num_classes, total_images)
    """
    agg = defaultdict(lambda: {"classes": set(), "images": 0})
    for cls, n in class_counts.items():
        top = cls.split("/")[0]
        agg[top]["classes"].add(cls)
        agg[top]["images"] += n
    result = {}
    for top, info in agg.items():
        result[top] = {
            "num_classes": len(info["classes"]),
            "total_images": info["images"],
        }
    return result


def safe_read_csv(path, **kwargs):
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    lines = []

    # -----------------------------------------------------------------
    # 1. HEADER
    # -----------------------------------------------------------------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("UNC Charlotte Campus Dataset Preparation Report")
    lines.append("================================================")
    lines.append(f"Generated on: {now}")
    lines.append(f"Dataset root: {DATASET_ROOT}")
    lines.append("")
    lines.append("Project goal: Prepare a high-quality, lens-corrected and deduplicated")
    lines.append("image–caption dataset of UNC Charlotte campus buildings and landmarks")
    lines.append("for Stable Diffusion / generative model fine-tuning.")
    lines.append("")

    # -----------------------------------------------------------------
    # 2. DIRECTORY OVERVIEW
    # -----------------------------------------------------------------
    lines.append("1. Directory Overview")
    lines.append("---------------------")
    children = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()])
    lines.append("Top-level folders under dataset root:")
    for c in children:
        lines.append(f"  - {c.name}/")
    lines.append("")

    # Final deduped image counts by class & by top category
    if DEDUP_DIR.exists():
        dedup_class_counts = count_images_by_class(DEDUP_DIR)
        dedup_top = group_by_top_category(dedup_class_counts)

        total_final_images = sum(dedup_class_counts.values())
        total_final_classes = len(dedup_class_counts)

        lines.append("Final deduplicated image counts (processed_dedup/):")
        lines.append(f"  Total images: {total_final_images}")
        lines.append(f"  Total location classes (folders under processed_dedup): {total_final_classes}")
        lines.append("")
        lines.append("Per top-level category in processed_dedup:")
        for top, info in sorted(dedup_top.items()):
            lines.append(
                f"  - {top}: {info['num_classes']} classes, {info['total_images']} images"
            )
        lines.append("")
    else:
        lines.append("processed_dedup/ not found; dedup summary will be limited.")
        lines.append("")

    # -----------------------------------------------------------------
    # 3. ORIGINAL DATASET & CAPTIONS
    # -----------------------------------------------------------------
    lines.append("2. Original Dataset and Captions")
    lines.append("--------------------------------")

    # Raw image counts
    raw_class_counts = {}
    for d in RAW_DIRS:
        if d.exists():
            raw_class_counts.update(count_images_by_class(d))
    raw_top = group_by_top_category(raw_class_counts)
    total_raw_images = sum(raw_class_counts.values())
    total_raw_classes = len(raw_class_counts)

    lines.append(f"Original raw images (before lens correction and deduplication):")
    lines.append(f"  Total images (all raw folders): {total_raw_images}")
    lines.append(f"  Total location classes (raw folders): {total_raw_classes}")
    lines.append("")
    lines.append("Per top-level category (raw):")
    for top, info in sorted(raw_top.items()):
        lines.append(
            f"  - {top}: {info['num_classes']} classes, {info['total_images']} images"
        )
    lines.append("")

    meta_orig = safe_read_csv(METADATA_ORIGINAL)
    if meta_orig is not None:
        lines.append(f"Original captions file: {METADATA_ORIGINAL}")
        lines.append(f"  Rows (image–caption pairs): {len(meta_orig)}")
        lines.append(f"  Columns: {', '.join(meta_orig.columns)}")
        lines.append("")

        # Check how many img_path entries actually existed in raw folders
        existing_count = 0
        for p in meta_orig["img_path"]:
            # raw images are under category folders at dataset root
            img_path = DATASET_ROOT / p
            if img_path.exists():
                existing_count += 1
        lines.append(
            f"Sanity check: {existing_count} / {len(meta_orig)} metadata rows "
            "had a matching raw image file at the time of report generation."
        )
        lines.append("")
    else:
        lines.append("WARNING: metadata.csv not found; cannot summarise original captions.")
        lines.append("")

    # -----------------------------------------------------------------
    # 4. CAPTION REWRITING STAGE
    # -----------------------------------------------------------------
    lines.append("3. Caption Rewriting Stage (DeepSeek-VL2)")
    lines.append("-----------------------------------------")

    meta_rew = safe_read_csv(METADATA_REWRITTEN)
    if meta_rew is not None:
        lines.append(f"Rewritten captions file: {METADATA_REWRITTEN}")
        lines.append(f"  Rows: {len(meta_rew)}")
        lines.append(f"  Columns: {', '.join(meta_rew.columns)}")
        lines.append("")

        has_caption_rew = "caption_rewritten" in meta_rew.columns
        lines.append(f"  Contains 'caption_rewritten' column: {has_caption_rew}")
        if has_caption_rew:
            non_empty = meta_rew["caption_rewritten"].astype(str).str.strip().ne("").sum()
            lines.append(f"  Non-empty rewritten captions: {non_empty}")
        lines.append("")

    else:
        lines.append("WARNING: metadata_rewritten.csv not found.")
        lines.append("")

    # Narrative description of the rewrite pipeline
    lines.append("Description of caption rewriting pipeline:")
    lines.append("  - Base model: DeepSeek-VL2 vision-language model.")
    lines.append("  - For each image, the model predicts a set of visual attributes")
    lines.append("    from a fixed vocabulary: trees, grass, path, road, water,")
    lines.append("    building, brick_building, open_plaza, parking_lot, bench, lamp_post.")
    lines.append("  - The script also parses the original caption to detect which")
    lines.append("    of these attributes are explicitly mentioned in text.")
    lines.append("  - It then takes the intersection of:")
    lines.append("        (attributes detected by DeepSeek-VL2)")
    lines.append("        ∩ (attributes inferred from the original caption)")
    lines.append("    and falls back to model-detected attributes if the intersection is empty.")
    lines.append("  - Attributes are ranked by global importance and at most three are kept,")
    lines.append("    prioritising trees/buildings/paths over minor details.")
    lines.append("  - If both 'building' and 'brick_building' are present, a deterministic")
    lines.append("    pseudo-random choice based on the MD5 hash of the image path keeps")
    lines.append("    exactly one of them to avoid redundancy while preserving diversity.")
    lines.append("  - A short, single-sentence caption is then generated using a template:")
    lines.append("        '<BuildingName> at UNC Charlotte <scene phrase>.'")
    lines.append("    where the scene phrase encodes the selected attributes, e.g.:")
    lines.append("        'near trees and a brick building',")
    lines.append("        'beside trees, a campus building, and a campus walkway'.")
    lines.append("  - This produces consistent, SD-friendly captions that always include")
    lines.append("    the building/landmark name and a small set of grounded visual cues.")
    lines.append("")

    # -----------------------------------------------------------------
    # 5. IMAGE PREPROCESSING (LENS CORRECTION)
    # -----------------------------------------------------------------
    lines.append("4. Image Pre-processing (Lens Correction)")
    lines.append("----------------------------------------")
    if PROCESSED_DIR.exists():
        proc_counts = count_images_by_class(PROCESSED_DIR)
        proc_top = group_by_top_category(proc_counts)
        total_proc_images = sum(proc_counts.values())
        total_proc_classes = len(proc_counts)

        lines.append(f"Lens-corrected images folder: {PROCESSED_DIR}")
        lines.append(f"  Total images in processed/: {total_proc_images}")
        lines.append(f"  Total classes in processed/: {total_proc_classes}")
        lines.append("")
        lines.append("Per top-level category (processed):")
        for top, info in sorted(proc_top.items()):
            lines.append(
                f"  - {top}: {info['num_classes']} classes, {info['total_images']} images"
            )
        lines.append("")
    else:
        lines.append("processed/ directory not found. Lens-corrected counts unavailable.")
        lines.append("")

    lines.append("Lens correction was applied to images extracted from GoPro-style video")
    lines.append("footage before any deduplication or model training. The corrected images")
    lines.append("are stored under processed/, mirroring the raw folder hierarchy.")
    lines.append("")

    # -----------------------------------------------------------------
    # 6. DEDUPLICATION STAGE
    # -----------------------------------------------------------------
    lines.append("5. Deduplication Stage (Hybrid Hash + CLIP)")
    lines.append("-------------------------------------------")

    dedup_summary = safe_read_csv(DEDUP_SUMMARY)
    dedup_removed_df = safe_read_csv(DEDUP_REMOVED)

    if dedup_summary is not None:
        lines.append(f"Dedup summary file: {DEDUP_SUMMARY}")
        lines.append(f"  Rows (classes): {len(dedup_summary)}")
        lines.append("")

        total_before = dedup_summary["original_count"].sum()
        total_after = dedup_summary["final_count"].sum()
        total_removed = dedup_summary["removed"].sum()
        overall_rate = 100.0 * total_removed / max(total_before, 1)

        lines.append(f"Total images before deduplication: {total_before}")
        lines.append(f"Total images after deduplication:  {total_after}")
        lines.append(f"Total removed as near-duplicates:  {total_removed}")
        lines.append(f"Overall removal rate: {overall_rate:.1f}%")
        lines.append("")

        # Per-class removal rates
        dedup_summary["removal_rate_pct"] = (
            100.0 * dedup_summary["removed"] / dedup_summary["original_count"].clip(lower=1)
        )
        max_row = dedup_summary.loc[dedup_summary["removal_rate_pct"].idxmax()]
        min_row = dedup_summary.loc[dedup_summary["removal_rate_pct"].idxmin()]
        mean_rate = dedup_summary["removal_rate_pct"].mean()

        lines.append(f"Mean per-class removal rate: {mean_rate:.1f}%")
        lines.append(
            f"Highest removal rate: {max_row['class']} "
            f"({max_row['removal_rate_pct']:.1f}% removed)"
        )
        lines.append(
            f"Lowest removal rate:  {min_row['class']} "
            f"({min_row['removal_rate_pct']:.1f}% removed)"
        )
        lines.append("")
        lines.append("Per-class deduplication summary (class, original, final, removed):")
        for _, r in dedup_summary.sort_values("class").iterrows():
            lines.append(
                f"  - {r['class']}: {int(r['original_count'])} -> "
                f"{int(r['final_count'])} (removed {int(r['removed'])}, "
                f"{r['removal_rate_pct']:.1f}% reduction)"
            )
        lines.append("")
    else:
        lines.append("WARNING: dedup_summary.csv not found; dedup stats unavailable.")
        lines.append("")

    if dedup_removed_df is not None:
        lines.append(f"Dedup removed images file: {DEDUP_REMOVED}")
        lines.append(f"  Total removed image paths listed: {len(dedup_removed_df)}")
        lines.append("")
    else:
        lines.append("WARNING: dedup_removed_images.csv not found.")
        lines.append("")

    # Narrative description of dedup algorithm
    lines.append("Description of deduplication algorithm:")
    lines.append("  - Per-class (per building/landmark folder) deduplication.")
    lines.append("  - Step 1: Perceptual hash (phash) grouping:")
    lines.append("      * Compute a phash for each image.")
    lines.append("      * Images whose Hamming distance is ≤ 4 are considered")
    lines.append("        near-identical and merged into the same candidate group.")
    lines.append("  - Step 2: CLIP embedding similarity:")
    lines.append("      * Compute CLIP ViT-B/32 image embeddings for all images in the class.")
    lines.append("      * Cosine similarity matrix is computed; image pairs with similarity")
    lines.append("        ≥ 0.985 are merged into the same group.")
    lines.append("  - Step 3: Quality-based representative selection within each group:")
    lines.append("      * For each group, a hybrid quality score is computed per image:")
    lines.append("            quality = 0.5 * sharpness + 0.3 * entropy + 0.2 * centerness")
    lines.append("        where:")
    lines.append("          - sharpness = variance of Laplacian (focus measure),")
    lines.append("          - entropy   = information content of the grayscale histogram,")
    lines.append("          - centerness = similarity to the CLIP centroid of the class.")
    lines.append("      * Exactly one image (the highest-quality one) is kept per group;")
    lines.append("        the rest are removed as near-duplicates.")
    lines.append("  - Step 4: A list of removed images is written to dedup_removed_images.csv,")
    lines.append("    and class-level before/after counts are written to dedup_summary.csv.")
    lines.append("")

    # -----------------------------------------------------------------
    # 7. FINAL DATASET
    # -----------------------------------------------------------------
    lines.append("6. Final Dataset for Model Training")
    lines.append("-----------------------------------")

    meta_final = safe_read_csv(METADATA_FINAL)
    if meta_final is not None:
        lines.append(f"Final metadata file (after dedup): {METADATA_FINAL}")
        lines.append(f"  Rows (image–caption pairs): {len(meta_final)}")
        lines.append(f"  Columns: {', '.join(meta_final.columns)}")
        lines.append("")

        # Check consistency with processed_dedup/
        missing_images = []
        for rel in meta_final["img_path"]:
            if not (DEDUP_DIR / rel).exists():
                missing_images.append(rel)

        if missing_images:
            lines.append(
                f"WARNING: {len(missing_images)} metadata rows refer to missing images in processed_dedup/."
            )
            lines.append("  First few missing:")
            for m in missing_images[:10]:
                lines.append(f"    - {m}")
        else:
            lines.append("Sanity check: every metadata row has a corresponding image")
            lines.append("in processed_dedup/.")
        lines.append("")

        # Check for images without metadata
        meta_set = set(meta_final["img_path"])
        orphan_images = []
        if DEDUP_DIR.exists():
            for p in DEDUP_DIR.rglob("*.jpg"):
                rel = str(p.relative_to(DEDUP_DIR))
                if rel not in meta_set:
                    orphan_images.append(rel)
        if orphan_images:
            lines.append(
                f"WARNING: {len(orphan_images)} images in processed_dedup/ have no metadata entry."
            )
            lines.append("  First few orphans:")
            for o in orphan_images[:10]:
                lines.append(f"    - {o}")
        else:
            lines.append("Sanity check: every image in processed_dedup/ has a metadata entry.")
        lines.append("")

        # Per-category breakdown from final metadata
        cat_counts = defaultdict(int)
        for rel in meta_final["img_path"]:
            top = str(rel).split("/")[0]
            cat_counts[top] += 1
        lines.append("Final image–caption pairs per top-level category (from metadata):")
        for top, n in sorted(cat_counts.items()):
            lines.append(f"  - {top}: {n} pairs")
        lines.append("")

    else:
        lines.append("WARNING: metadata_rewritten_dedup_final.csv not found.")
        lines.append("")

    # -----------------------------------------------------------------
    # 8. PIPELINE SUMMARY
    # -----------------------------------------------------------------
    lines.append("7. End-to-End Pipeline Summary")
    lines.append("------------------------------")
    lines.append("In summary, the dataset preparation pipeline consists of:")
    lines.append("  1) Raw data collection: walking through the UNC Charlotte campus")
    lines.append("     with an action camera, extracting frames into class-specific")
    lines.append("     folders such as academic_buildings/Fretwell_hall,")
    lines.append("     monuments/Aperture_Sculpture, etc.")
    lines.append("  2) Initial captions: automatically generated but verbose captions")
    lines.append("     stored in metadata.csv (img_path, caption).")
    lines.append("  3) Caption rewriting: DeepSeek-VL2-based rewriting that extracts")
    lines.append("     visual attributes, intersects them with the original text, and")
    lines.append("     produces short, standardised captions of the form:")
    lines.append("         '<BuildingName> at UNC Charlotte <scene phrase>.'")
    lines.append("     written to metadata_rewritten.csv.")
    lines.append("  4) Lens correction: geometric and lens distortion correction applied")
    lines.append("     to all images, saved under processed/ while preserving the folder")
    lines.append("     hierarchy.")
    lines.append("  5) Deduplication: a hybrid perceptual-hash + CLIP + quality-scoring")
    lines.append("     pipeline run per class to remove redundant frames from walk-through")
    lines.append("     videos. This yields processed_dedup/, dedup_removed_images.csv and")
    lines.append("     dedup_summary.csv.")
    lines.append("  6) Final alignment: metadata rows corresponding to removed images are")
    lines.append("     dropped, resulting in metadata_rewritten_dedup_final.csv, which is")
    lines.append("     fully aligned with the contents of processed_dedup/.")
    lines.append("")
    lines.append("The resulting dataset is a lens-corrected, deduplicated, and consistently")
    lines.append("captioned collection of UNC Charlotte campus images suitable for")
    lines.append("fine-tuning Stable Diffusion or other generative models on campus imagery.")
    lines.append("")

    # -----------------------------------------------------------------
    # WRITE REPORT
    # -----------------------------------------------------------------
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("Report written to:", REPORT_PATH)


if __name__ == "__main__":
    main()
