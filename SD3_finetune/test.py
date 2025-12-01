import os
import re
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------------------
# GPU / DEVICE
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# PATHS (FINAL CONFIRMED)
# ---------------------------------------------------------------------
DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

REAL_IMAGES_DIR = DATASET_ROOT / "sd_lora_dataset/images"
REAL_CAPTIONS_DIR = DATASET_ROOT / "sd_lora_dataset/captions"

METADATA_TOKENIZED = DATASET_ROOT / "metadata_tokenized.csv"
LOCATION_TOKENS = DATASET_ROOT / "location_tokens.csv"

# Generated images
SD15_BASE_GEN = DATASET_ROOT / "runs/base_sd15/inference_outputs/all_locations"
SD3_BASE_GEN  = DATASET_ROOT / "runs/base_sd3/inference_outputs/all_locations"

SD15_FT_GEN = DATASET_ROOT / "runs/sd15_uncc_fullfinetune/inference_outputs/all_locations"
SD3_FT_GEN  = DATASET_ROOT / "runs/inference_outputs/all_locations"

OUTPUT_DIR = DATASET_ROOT / "runs/eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# LOAD TOKEN + CLASS MAPPING
# ---------------------------------------------------------------------
def normalize_class_name(cls: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", cls.lower())


def build_token_maps():
    token_df = pd.read_csv(LOCATION_TOKENS)
    token_to_class = dict(zip(token_df["token"], token_df["class_name"]))

    # Build class→slug
    class_to_slug = {cls: normalize_class_name(cls) for cls in token_to_class.values()}

    return token_to_class, class_to_slug


# ---------------------------------------------------------------------
# BUILD REAL DATASET INDEX
# ---------------------------------------------------------------------
def build_real_reference_sets(token_to_class, class_to_slug):
    df = pd.read_csv(METADATA_TOKENIZED)

    # dict: slug → list of {img_path, caption}
    refs = {}

    for _, row in df.iterrows():
        img_rel = row["img_path"]               # e.g., academic_buildings/Fretwell_Hall/xxxx.jpg
        caption = row["caption_tokenized"]

        # Extract class_name from path
        parts = str(img_rel).split("/")
        if len(parts) < 2:
            continue
        cls_name = parts[1]
        slug = class_to_slug.get(cls_name)
        if slug is None:
            continue

        # The flat version of the filename in sd_lora_dataset
        flat_name = img_rel.replace("/", "_")
        img_path = REAL_IMAGES_DIR / flat_name

        if not img_path.exists():
            continue

        refs.setdefault(slug, []).append({
            "img_path": img_path,
            "caption": caption,
            "class_name": cls_name
        })

    return refs


# ---------------------------------------------------------------------
# LOAD METRIC MODELS: CLIP + DINOv2
# ---------------------------------------------------------------------
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    clip_model.eval()
    return clip_model, clip_processor


def load_dino():
    # DINOv2 ViT-B/14
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2.to(device)
    dinov2.eval()

    # Transform from their official repo
    dino_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return dinov2, dino_transform


# ---------------------------------------------------------------------
# FEATURE EXTRACTION HELPERS
# ---------------------------------------------------------------------
def compute_dino_feature(path, model, transform, cache):
    if path in cache:
        return cache[path]

    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True)

    cache[path] = feat
    return feat


def compute_clip_score(img_path, captions, clip_model, clip_processor):
    img = Image.open(img_path).convert("RGB")

    best = -1e9

    for cap in captions:
        inputs = clip_processor(
            text=[cap],
            images=[img],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            out = clip_model(**inputs)

        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)

        sim = (img_emb * txt_emb).sum(dim=-1).item()
        best = max(best, sim)

    return best


# ---------------------------------------------------------------------
# LOAD GENERATED IMAGES
# ---------------------------------------------------------------------
def load_generated_images(gen_root, slug):
    folder = gen_root / slug
    if not folder.exists():
        return []
    return sorted(list(folder.glob("image_*.png")))


# ---------------------------------------------------------------------
# EVALUATE ONE MODEL
# ---------------------------------------------------------------------
def evaluate_model(model_name, gen_root, refs, clip_model, clip_processor, dino_model, dino_transform):
    results = []

    # Cache for features
    dino_cache = {}

    # Precompute global reference features for location accuracy
    global_real_feats = []
    global_real_slugs = []

    for slug, items in refs.items():
        for info in items:
            feat = compute_dino_feature(info["img_path"], dino_model, dino_transform, dino_cache)
            global_real_feats.append(feat)
            global_real_slugs.append(slug)

    global_real_feats = torch.cat(global_real_feats, dim=0)   # [N, D]

    # Evaluate per location
    for slug, items in tqdm(refs.items(), desc=f"Evaluating {model_name}"):
        real_imgs = [x["img_path"] for x in items]
        real_caps = [x["caption"] for x in items]

        # Load generated images
        gen_imgs = load_generated_images(gen_root, slug)
        if len(gen_imgs) == 0:
            continue

        # 1. Find best generated image via DINO max-match
        best_dino = -1e9
        best_gen = None
        best_real = None

        for g in gen_imgs:
            g_feat = compute_dino_feature(g, dino_model, dino_transform, dino_cache)

            # Compare with all real images of this location
            sims = []
            for r in real_imgs:
                r_feat = compute_dino_feature(r, dino_model, dino_transform, dino_cache)
                sim = (g_feat * r_feat).sum().item()
                sims.append(sim)

            local_best = max(sims)
            if local_best > best_dino:
                best_dino = local_best
                best_gen = g
                best_real = real_imgs[np.argmax(sims)]

        # 2. CLIPScore (best over real captions)
        clip_score = compute_clip_score(best_gen, real_caps, clip_model, clip_processor)

        # 3. Location accuracy: DINO global retrieval
        best_gen_feat = compute_dino_feature(best_gen, dino_model, dino_transform, dino_cache)
        sims_global = (best_gen_feat * global_real_feats).sum(dim=-1)
        idx = sims_global.argmax().item()
        pred_slug = global_real_slugs[idx]
        loc_acc = 1.0 if pred_slug == slug else 0.0

        results.append({
            "model": model_name,
            "location_slug": slug,
            "clipscore": clip_score,
            "dino": best_dino,
            "loc_acc": loc_acc,
            "best_gen_path": str(best_gen),
            "best_real_path": str(best_real),
        })

    return results


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    print("\nLoading token maps...")
    token_to_class, class_to_slug = build_token_maps()

    print("Building real dataset index...")
    refs = build_real_reference_sets(token_to_class, class_to_slug)

    print("Loading metric models (CLIP + DINOv2)...")
    clip_model, clip_processor = load_clip_model()
    dino_model, dino_transform = load_dino()

    model_specs = {
        "sd15_base": SD15_BASE_GEN,
        "sd15_finetuned": SD15_FT_GEN,
        "sd3_base": SD3_BASE_GEN,
        "sd3_finetuned": SD3_FT_GEN,
    }

    all_results = []

    for model_name, path in model_specs.items():
        res = evaluate_model(
            model_name,
            path,
            refs,
            clip_model,
            clip_processor,
            dino_model,
            dino_transform
        )

        df = pd.DataFrame(res)
        df.to_csv(OUTPUT_DIR / f"{model_name}_per_location.csv", index=False)
        print(f"Saved per-location results for {model_name}")

        all_results.extend(res)

    # Compute means
    df_all = pd.DataFrame(all_results)
    means = df_all.groupby("model")[["clipscore", "dino", "loc_acc"]].mean()
    means.to_csv(OUTPUT_DIR / "model_means.csv")
    print("Saved model-wise mean metrics.")


if __name__ == "__main__":
    main()
