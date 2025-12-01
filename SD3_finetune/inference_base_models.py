import re
from pathlib import Path
from typing import Dict, List
from PIL import Image

import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

# Diffusers imports
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

# Dataset helpers
import pandas as pd

# -----------------------------------------------------------------------------
#               CONFIG: MODIFY ONLY THESE TWO PATHS
# -----------------------------------------------------------------------------

DATASET_ROOT = Path("/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1")

# Where to save base model generations
OUT_SD15 = DATASET_ROOT / "runs/base_sd15/inference_outputs/all_locations"
OUT_SD3  = DATASET_ROOT / "runs/base_sd3/inference_outputs/all_locations"

# Input metadata for class mappings
METADATA_TOKENIZED = DATASET_ROOT / "metadata_tokenized.csv"
LOCATION_TOKENS = DATASET_ROOT / "location_tokens.csv"


# -----------------------------------------------------------------------------
#               LOAD LOCATION + CLASS MAPPINGS
# -----------------------------------------------------------------------------

def build_token_and_keyword_maps():
    """
    Returns:
      token_to_noun:      <loc_xxx> -> noun ("building", "lake", etc.)
      token_to_classname: <loc_xxx> -> class_name ("Fretwell_Hall")
    """
    # ---- Load nouns from metadata_tokenized.csv ----
    df_meta = pd.read_csv(METADATA_TOKENIZED)
    token_to_noun = {}

    pattern = re.compile(
        r"a photo of\s+(<loc_[^>]+>)\s+([a-zA-Z]+)\s+at UNC Charlotte",
        flags=re.IGNORECASE,
    )

    for cap in df_meta["caption_tokenized"].astype(str):
        m = pattern.search(cap)
        if m:
            tok = m.group(1)
            noun = m.group(2).lower()
            token_to_noun.setdefault(tok, noun)

    # ---- Load class_name from location_tokens.csv ----
    df_tok = pd.read_csv(LOCATION_TOKENS)
    token_to_classname = dict(zip(df_tok["token"], df_tok["class_name"]))

    return token_to_noun, token_to_classname


def normalize_class_name(cls: str) -> str:
    """
    Match your inference scripts: lowercase + remove non-alphanumerics.
    """
    return re.sub(r"[^a-z0-9]+", "", cls.lower())


# -----------------------------------------------------------------------------
#                  SAVE IMAGES
# -----------------------------------------------------------------------------

def save_images(images: torch.Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, output_dir / f"image_{i+1}.png")


# -----------------------------------------------------------------------------
#               BASE SD1.5 GENERATION (NO CHECKPOINT LOAD)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_sd15_images(prompt: str, num_images: int = 5, steps: int = 50):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    images_pil = pipe(
        prompt=[prompt] * num_images,
        num_inference_steps=steps,
    ).images

    tensors = [TF.to_tensor(im) for im in images_pil]
    return torch.stack(tensors, dim=0)


# -----------------------------------------------------------------------------
#               BASE SD3 GENERATION (NO CHECKPOINT LOAD)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_sd3_images(prompt: str, num_images: int = 5, steps: int = 50):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Replace scheduler to match your FT path (FlowMatchEulerDiscreteScheduler)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    images_pil = pipe(
        prompt=[prompt] * num_images,
        num_inference_steps=steps,
        guidance_scale=3.5,
        output_type="pil",
    ).images

    # Match training resolution by resizing
    # SD3 uses square VAE; use 1024 or 768 depending on your config
    RES = 768
    images_resized = [im.resize((RES, RES), Image.BICUBIC) for im in images_pil]

    tensors = [TF.to_tensor(im) for im in images_resized]
    return torch.stack(tensors, dim=0)


# -----------------------------------------------------------------------------
#                  GENERATE ALL LOCATIONS
# -----------------------------------------------------------------------------

def generate_for_all_locations_base():
    token_to_noun, token_to_classname = build_token_and_keyword_maps()

    print(f"Found {len(token_to_classname)} locations.")
    print("Generating base SD1.5 and SD3 images...\n")

    for token, cls_name in token_to_classname.items():
        noun = token_to_noun.get(token, "location")
        cls_slug = normalize_class_name(cls_name)

        prompt = f"a photo of {token} {noun} at UNC Charlotte"

        print(f"[{cls_slug}]  Prompt: {prompt}")

        # ---- SD1.5 ----
        out_dir_sd15 = OUT_SD15 / cls_slug
        imgs15 = generate_sd15_images(prompt)
        save_images(imgs15, out_dir_sd15)
        print(f"   SD1.5 → {out_dir_sd15}")

        # ---- SD3 ----
        out_dir_sd3 = OUT_SD3 / cls_slug
        imgs3 = generate_sd3_images(prompt)
        save_images(imgs3, out_dir_sd3)
        print(f"   SD3   → {out_dir_sd3}")

    print("\nDONE: Base model images saved.")


# -----------------------------------------------------------------------------
#                                   MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_for_all_locations_base()
