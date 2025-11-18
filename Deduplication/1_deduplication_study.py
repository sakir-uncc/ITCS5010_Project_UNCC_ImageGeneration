import os
import math
import cv2
import hashlib
import imagehash
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import open_clip


# ---------------------------------------------------------------------
# CONFIG – adjust these paths for your machine
# ---------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

PROCESSED_ROOT = DATASET_ROOT / "processed"          # lens-corrected images
METADATA_IN = DATASET_ROOT / "metadata_rewritten.csv"
METADATA_OUT = DATASET_ROOT / "metadata_rewritten_dedup.csv"

REMOVED_LIST_CSV = DATASET_ROOT / "dedup_removed_images.csv"
SUMMARY_CSV = DATASET_ROOT / "dedup_summary.csv"

# Do NOT delete anything by default – set to True once you're happy.
DELETE_FILES = False

# Thresholds
PHASH_THRESHOLD = 4          # Hamming distance (< = 4 -> near-identical)
CLIP_SIM_THRESHOLD = 0.985   # Cosine similarity (> = 0.985 -> duplicate)


# ---------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(DEVICE).eval()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def walk_classes(processed_root: Path):
    """
    Yield (class_key, [list of image paths]) where class_key is
    'academic_buildings/Fretwell_hall', etc.
    """
    class_to_imgs = {}
    for path in processed_root.rglob("*.jpg"):
        rel = path.relative_to(processed_root)
        class_key = str(rel.parent)  # folder path without filename
        class_to_imgs.setdefault(class_key, []).append(path)
    return class_to_imgs


def compute_phashes(image_paths):
    hashes = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            h = imagehash.phash(img)
        except Exception as e:
            print(f"[phash] Error on {p}: {e}")
            h = None
        hashes.append(h)
    return hashes


def compute_clip_embeddings(image_paths):
    """
    Returns an (N, D) tensor of normalized CLIP embeddings.
    """
    all_embeds = []
    batch = []
    batch_paths = []
    BATCH_SIZE = 32

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(clip_preprocess(img))
        except Exception as e:
            print(f"[clip] Error loading {p}: {e}")
            batch.append(None)

        batch_paths.append(p)

        if len(batch) == BATCH_SIZE:
            valid_idx = [i for i, t in enumerate(batch) if t is not None]
            if valid_idx:
                imgs = torch.stack([batch[i] for i in valid_idx]).to(DEVICE)
                with torch.no_grad():
                    emb = clip_model.encode_image(imgs)
                emb = F.normalize(emb, dim=1)
                # Put back into full batch positions
                full_emb = torch.zeros(len(batch), emb.shape[1], device=DEVICE)
                for j, i in enumerate(valid_idx):
                    full_emb[i] = emb[j]
            else:
                full_emb = torch.zeros(len(batch), 512, device=DEVICE)

            all_embeds.append(full_emb.cpu())
            batch, batch_paths = [], []

    # last partial
    if batch:
        valid_idx = [i for i, t in enumerate(batch) if t is not None]
        if valid_idx:
            imgs = torch.stack([batch[i] for i in valid_idx]).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_image(imgs)
            emb = F.normalize(emb, dim=1)
            full_emb = torch.zeros(len(batch), emb.shape[1], device=DEVICE)
            for j, i in enumerate(valid_idx):
                full_emb[i] = emb[j]
        else:
            full_emb = torch.zeros(len(batch), 512, device=DEVICE)
        all_embeds.append(full_emb.cpu())

    if not all_embeds:
        return torch.zeros(0, 512)

    embeds = torch.cat(all_embeds, dim=0)  # (N, D)
    return embeds


def compute_quality_scores(image_paths, embeddings):
    """
    Hybrid quality score per image:
      score = 0.5 * sharp_norm + 0.3 * entropy_norm + 0.2 * centerness_norm
    Returns np.array of scores (N,)
    """
    N = len(image_paths)
    sharp_vals = np.zeros(N, dtype=np.float32)
    ent_vals = np.zeros(N, dtype=np.float32)

    for i, p in enumerate(image_paths):
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # sharpness via Laplacian variance
            sharp_vals[i] = cv2.Laplacian(img, cv2.CV_64F).var()

            # entropy
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist.ravel()
            hist /= (hist.sum() + 1e-8)
            ent = -np.sum(hist * np.log2(hist + 1e-8))
            ent_vals[i] = ent
        except Exception as e:
            print(f"[quality] Error on {p}: {e}")

    # centerness: similarity to folder centroid in CLIP space
    if embeddings.shape[0] > 0:
        embeds = F.normalize(torch.from_numpy(embeddings.numpy()), dim=1)
        centroid = embeds.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, dim=1)
        center_scores = (embeds * centroid).sum(dim=1).numpy()
    else:
        center_scores = np.zeros(N, dtype=np.float32)

    def norm(x):
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    sharp_n = norm(sharp_vals)
    ent_n = norm(ent_vals)
    center_n = norm(center_scores)

    scores = 0.5 * sharp_n + 0.3 * ent_n + 0.2 * center_n
    return scores


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def deduplicate_class(image_paths):
    """
    For one class (folder), perform:
      1) phash-based grouping
      2) CLIP similarity grouping
      3) select best representative per group using quality scores
    Returns:
      kept_paths, removed_paths
    """
    N = len(image_paths)
    if N <= 1:
        return image_paths, []

    print(f"  Images in class: {N}")
    uf = UnionFind(N)

    # --- phash pass ---
    print("   > computing perceptual hashes...")
    hashes = compute_phashes(image_paths)

    print("   > phash dedup...")
    for i in range(N):
        if hashes[i] is None:
            continue
        for j in range(i + 1, N):
            if hashes[j] is None:
                continue
            dist = hashes[i] - hashes[j]
            if dist <= PHASH_THRESHOLD:
                uf.union(i, j)

    # --- CLIP embeddings pass ---
    print("   > computing CLIP embeddings...")
    embeds = compute_clip_embeddings(image_paths)  # (N, D)

    print("   > CLIP similarity dedup...")
    if embeds.shape[0] == N and N <= 1500:
        # normalize (should already be normalized, but just in case)
        embeds = F.normalize(embeds, dim=1)
        sims = embeds @ embeds.T  # (N, N)
        sims = sims.numpy()
        for i in range(N):
            for j in range(i + 1, N):
                if sims[i, j] >= CLIP_SIM_THRESHOLD:
                    uf.union(i, j)
    else:
        print("   > skipped CLIP similarity (embedding mismatch or too large N)")

    # --- quality scores ---
    print("   > computing quality scores...")
    scores = compute_quality_scores(image_paths, embeds)

    # --- build clusters ---
    clusters = {}
    for i in range(N):
        root = uf.find(i)
        clusters.setdefault(root, []).append(i)

    kept_indices = []
    removed_indices = []

    for root, idxs in clusters.items():
        if len(idxs) == 1:
            kept_indices.extend(idxs)
            continue
        # choose best by score
        best = max(idxs, key=lambda k: scores[k])
        kept_indices.append(best)
        for k in idxs:
            if k != best:
                removed_indices.append(k)

    kept_paths = [image_paths[i] for i in sorted(kept_indices)]
    removed_paths = [image_paths[i] for i in sorted(removed_indices)]
    return kept_paths, removed_paths


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Scanning processed images under:", PROCESSED_ROOT)
    class_to_imgs = walk_classes(PROCESSED_ROOT)
    print("Found", len(class_to_imgs), "classes/folders.")

    all_removed = []
    summary_rows = []

    for class_key, img_paths in sorted(class_to_imgs.items()):
        print(f"\n=== Class: {class_key} ===")
        before = len(img_paths)
        kept, removed = deduplicate_class(sorted(img_paths))
        after = len(kept)
        removed_count = len(removed)

        print(f"  -> kept {after} / {before}, removed {removed_count}")
        for p in removed:
            rel = p.relative_to(PROCESSED_ROOT)
            all_removed.append({
                "class": class_key,
                "image_path_processed": str(rel),
            })

        summary_rows.append({
            "class": class_key,
            "original_count": before,
            "final_count": after,
            "removed": removed_count,
        })

        if DELETE_FILES and removed:
            for p in removed:
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"Error deleting {p}: {e}")

    # Save removed list & summary
    removed_df = pd.DataFrame(all_removed)
    removed_df.to_csv(REMOVED_LIST_CSV, index=False)
    print("\nSaved removed image list to:", REMOVED_LIST_CSV)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print("Saved dedup summary to:", SUMMARY_CSV)

    # -----------------------------------------------------------------
    # Update metadata_rewritten.csv -> metadata_rewritten_dedup.csv
    # We assume metadata paths are relative like "academic_buildings/...".
    # Processed images are in "processed/<that path>".
    # -----------------------------------------------------------------
    if METADATA_IN.exists():
        print("\nUpdating metadata CSV…")
        meta = pd.read_csv(METADATA_IN)

        # Build set of relative paths to drop (without 'processed/' prefix)
        rel_to_drop = set()
        for row in all_removed:
            rel_proc = Path(row["image_path_processed"])  # e.g. academic_buildings/...
            rel_to_drop.add(str(rel_proc))

        before_rows = len(meta)
        meta_dedup = meta[~meta["img_path"].isin(rel_to_drop)].copy()
        after_rows = len(meta_dedup)

        meta_dedup.to_csv(METADATA_OUT, index=False)
        print(f"Metadata rows before: {before_rows}, after: {after_rows}")
        print("Deduplicated metadata written to:", METADATA_OUT)
    else:
        print("WARNING: metadata file not found at", METADATA_IN)


if __name__ == "__main__":
    main()
