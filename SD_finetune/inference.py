import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import torch
from torch.cuda.amp import autocast
from torchvision.utils import save_image
import pandas as pd

from config import get_default_config
from model import build_sd15_model, create_optimizers, load_checkpoint


# ------------------------------------------------------------
#  Build mappings from dataset artifacts
# ------------------------------------------------------------
def build_token_and_keyword_maps(paths) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      - token_to_noun:      <loc_xxx> -> noun ('building', 'lake', 'sculpture', etc.)
      - token_to_classname: <loc_xxx> -> original class_name ('Fretwell_hall', ...)
      - keyword_to_token:   keyword   -> <loc_xxx>, for matching user prompts
    """

    # 1) Build token -> noun from metadata_tokenized.csv
    metadata_tokenized = paths.dataset_root / "metadata_tokenized.csv"
    if not metadata_tokenized.exists():
        raise FileNotFoundError(
            f"metadata_tokenized.csv not found at {metadata_tokenized}. "
            "Make sure 1_tokenize.py and 2_SD_dataset_prep.py have been run."
        )

    df_meta = pd.read_csv(metadata_tokenized)
    if "caption_tokenized" not in df_meta.columns:
        raise ValueError("metadata_tokenized.csv must contain 'caption_tokenized' column.")

    token_to_noun: Dict[str, str] = {}

    # Captions look like: "a photo of <loc_xxx> noun at UNC Charlotte ..."
    pattern = re.compile(
        r"a photo of\s+(<loc_[^>]+>)\s+([a-zA-Z]+)\s+at UNC Charlotte",
        flags=re.IGNORECASE,
    )

    for cap in df_meta["caption_tokenized"].astype(str):
        m = pattern.search(cap)
        if m:
            token = m.group(1)
            noun = m.group(2).lower()
            if token not in token_to_noun:
                token_to_noun[token] = noun

    # 2) Build token -> class_name and keyword -> token from location_tokens.csv
    token_df = pd.read_csv(paths.location_tokens_csv)
    if not {"class_name", "token"}.issubset(token_df.columns):
        raise ValueError("location_tokens.csv must contain 'class_name' and 'token' columns.")

    token_to_classname: Dict[str, str] = {}
    keyword_to_token: Dict[str, str] = {}

    # Generally unhelpful / generic tokens for keyword matching
    GENERIC_WORDS = {
        "building",
        "hall",
        "center",
        "union",
        "lake",
        "statue",
        "sculpture",
        "memorial",
        "gym",
        "arena",
        "stadium",
        "field",
        "facility",
        "college",
        "library",
        "quad",
        "tower",
        "deck",
        "rec",
        "complex",
    }

    def normalize_class_name(cls: str) -> str:
        """Match the training-time class normalization (lowercase + strip non-alnum)."""
        return re.sub(r"[^a-z0-9]+", "", cls.lower())

    for _, row in token_df.iterrows():
        cls = str(row["class_name"])
        tok = str(row["token"])

        token_to_classname[tok] = cls

        # Full normalized class as a keyword
        cls_norm = normalize_class_name(cls)
        if cls_norm:
            keyword_to_token.setdefault(cls_norm, tok)

        # Individual words inside class name
        parts = re.split(r"[^a-z0-9]+", cls.lower())
        for w in parts:
            if not w or w in GENERIC_WORDS:
                continue
            keyword_to_token.setdefault(w, tok)

    return token_to_noun, token_to_classname, keyword_to_token


def find_token_for_prompt(user_prompt: str, keyword_to_token: Dict[str, str]) -> str:
    """
    Try to find a single location token that matches the user prompt,
    based on keyword_to_token mapping. Returns token or None.
    """
    text = re.sub(r"[^a-z0-9]+", " ", user_prompt.lower())
    words = text.split()

    found_tokens = []
    for w in words:
        if w in keyword_to_token:
            found_tokens.append(keyword_to_token[w])

    if not found_tokens:
        return None

    # Deduplicate, preserve order
    uniq = list(dict.fromkeys(found_tokens))
    if len(uniq) > 1:
        print(f"Warning: multiple location matches found in prompt: {uniq}. Using {uniq[0]}.")

    return uniq[0]


def build_prompt_from_token(user_prompt: str, token: str, noun: str) -> str:
    """
    Option C:
      - Use class noun automatically (building / lake / sculpture / ...)
      - Match the training-time caption style:
          "a photo of <loc_xxx> noun at UNC Charlotte"
      - If the user adds more description, append it as a suffix.
    """
    user_prompt = user_prompt.strip()

    if not user_prompt:
        return f"a photo of {token} {noun} at UNC Charlotte"

    # Avoid adding "all" as scene text
    if user_prompt.lower() == "all":
        return f"a photo of {token} {noun} at UNC Charlotte"

    # Append full user text as additional scene description
    return f"a photo of {token} {noun} at UNC Charlotte, {user_prompt}"


# ------------------------------------------------------------
#  Core image generation
# ------------------------------------------------------------
@torch.no_grad()
def generate_images(
    model_bundle,
    prompt: str,
    num_images: int = 5,
    steps: int = 50,
) -> torch.Tensor:
    device = model_bundle.device
    tokenizer = model_bundle.tokenizer
    text_encoder = model_bundle.text_encoder
    unet = model_bundle.unet
    vae = model_bundle.vae
    noise_scheduler = model_bundle.noise_scheduler

    train_cfg = get_default_config()["training"]

    prompts = [prompt] * num_images

    # Encode text
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with autocast(enabled=train_cfg.use_fp16 and device.type == "cuda"):
        text_embeds = text_encoder(text_inputs.input_ids)[0]

    # Latent noise
    latents = torch.randn(
        (num_images, unet.in_channels, 64, 64),
        device=device,
    )

    # Sampling loop
    noise_scheduler.set_timesteps(steps, device=device)
    for t in noise_scheduler.timesteps:
        with autocast(enabled=train_cfg.use_fp16 and device.type == "cuda"):
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeds).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    latents = latents / 0.18215
    with autocast(enabled=train_cfg.use_fp16 and device.type == "cuda"):
        images = vae.decode(latents).sample

    images = (images.clamp(-1, 1) + 1) / 2.0
    return images


def save_images(images: torch.Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, output_dir / f"image_{i+1}.png")


# ------------------------------------------------------------
#  Generate 5 images for every location
# ------------------------------------------------------------
def generate_for_all_locations(
    model_bundle,
    token_to_noun: Dict[str, str],
    token_to_classname: Dict[str, str],
    run_dir: Path,
    steps: int = 50,
    num_images: int = 5,
):
    out_root = run_dir / "inference_outputs" / "all_locations"
    out_root.mkdir(parents=True, exist_ok=True)

    def normalize_class_name(cls: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", cls.lower())

    for token, noun in token_to_noun.items():
        cls_name = token_to_classname.get(token, token)
        cls_slug = normalize_class_name(cls_name)

        prompt = f"a photo of {token} {noun} at UNC Charlotte"
        print(f"\n[ALL] {cls_name} ({token}) → {prompt}")

        images = generate_images(
            model_bundle,
            prompt=prompt,
            num_images=num_images,
            steps=steps,
        )

        out_dir = out_root / cls_slug
        save_images(images, out_dir)
        print(f"Saved {num_images} images for {cls_name} to: {out_dir}")


# ------------------------------------------------------------
#  Interactive console loop
# ------------------------------------------------------------
def main():
    cfg = get_default_config()
    paths = cfg["paths"]
    exp = cfg["experiment"]

    # Build mappings from dataset artifacts
    print("Building token and keyword maps from dataset metadata...")
    token_to_noun, token_to_classname, keyword_to_token = build_token_and_keyword_maps(paths)
    print(f"Found {len(token_to_noun)} tokens with nouns.")
    print(f"Keyword map contains {len(keyword_to_token)} entries.")

    # Experiment dirs
    run_dir = paths.runs_root / exp.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found at: {best_ckpt}")

    # Load model
    print("\nLoading model...")
    model_bundle = build_sd15_model(cfg)
    optimizer, scheduler = create_optimizers(model_bundle, cfg)
    _ = load_checkpoint(model_bundle, optimizer, scheduler, best_ckpt)

    model_bundle.unet.eval()
    model_bundle.text_encoder.eval()

    print("\nModel loaded.")
    print("Type a prompt containing a location (e.g. 'fretwell at sunset'),")
    print("or type 'all' to generate 5 images for every location.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_prompt = input("\nEnter prompt (or 'all' / 'exit'): ").strip()

        if user_prompt.lower() in {"exit", "quit"}:
            print("Exiting inference.")
            break

        # Special mode: generate 5 images per location for every token
        if user_prompt.lower() == "all":
            generate_for_all_locations(
                model_bundle,
                token_to_noun,
                token_to_classname,
                run_dir,
                steps=50,
                num_images=5,
            )
            continue

        # Try to find a location token in the user prompt
        token = find_token_for_prompt(user_prompt, keyword_to_token)

        if token is not None:
            noun = token_to_noun.get(token, "location")
            full_prompt = build_prompt_from_token(user_prompt, token, noun)
            cls_name = token_to_classname.get(token, token)

            print(f"Matched location token: {token} ({cls_name}), noun='{noun}'")
            print(f"Final prompt: {full_prompt}")

        else:
            # No token found; use user's prompt as-is
            full_prompt = user_prompt
            print("No known location keyword found in prompt.")
            print(f"Using prompt as-is: {full_prompt}")

        # Generate 5 images
        print("Generating images...")
        images = generate_images(
            model_bundle,
            prompt=full_prompt,
            num_images=5,
            steps=50,
        )

        # Save under a timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if token is not None:
            # Use location-based folder
            cls_name = token_to_classname.get(token, token)
            cls_slug = re.sub(r"[^a-z0-9]+", "", cls_name.lower())
            out_dir = run_dir / "inference_outputs" / cls_slug / timestamp
        else:
            # Generic prompts go in "generic" folder
            out_dir = run_dir / "inference_outputs" / "generic" / timestamp

        save_images(images, out_dir)
        print(f"Saved 5 images to: {out_dir}")


if __name__ == "__main__":
    main()
