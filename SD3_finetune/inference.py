import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import pandas as pd
from PIL import Image

from SD3_finetune.config import get_default_config
from SD3_finetune.model import build_sd3_model, create_optimizers, load_checkpoint
from SD3_finetune.model import SD3ModelBundle  # for type hints
from SD3_finetune.train import encode_prompt


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
    metadata_tokenized = Path(paths.csv_metadata_path)
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
    token_df = pd.read_csv(paths.location_token_path)
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


def find_token_for_prompt(user_prompt: str, keyword_to_token: Dict[str, str]) -> Optional[str]:
    """
    Try to find a single location token that matches the user prompt,
    based on keyword_to_token mapping. Returns token or None.
    """
    text = re.sub(r"[^a-z0-9]+", " ", user_prompt.lower())
    words = text.split()

    found_tokens: List[str] = []
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
    Match the training-time caption style:
        "a photo of <loc_xxx> noun at UNC Charlotte"
    and append any extra user description as a suffix.
    """
    user_prompt = user_prompt.strip()

    if not user_prompt or user_prompt.lower() == "all":
        return f"a photo of {token} {noun} at UNC Charlotte"

    # Append full user text as additional scene description
    return f"a photo of {token} {noun} at UNC Charlotte, {user_prompt}"


# ------------------------------------------------------------
#  SD3 text + location conditioning
# ------------------------------------------------------------
def _extract_location_ids_from_prompts(
    prompts: List[str],
    token_to_id: Dict[str, int],
    default_id: int = 0,
) -> List[int]:
    """
    Very small inference-time helper: look for any known <loc_xxx> token
    in each prompt string and map to its integer ID.
    """
    loc_ids: List[int] = []
    for text in prompts:
        if text is None:
            text = ""
        text = str(text)

        chosen = default_id
        # First explicit <loc_xxx> wins
        m = re.search(r"(<loc_[^>]+>)", text)
        if m:
            tok = m.group(1)
            if tok in token_to_id:
                chosen = token_to_id[tok]
        else:
            # Fallback: scan all known tokens to see if any occurs
            for tok, idx in token_to_id.items():
                if tok in text:
                    chosen = idx
                    break

        loc_ids.append(chosen)

    return loc_ids


@torch.no_grad()
def _generate_images_sd3(
    cfg,
    bundle: SD3ModelBundle,
    prompts: List[str],
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
) -> List[Image.Image]:
    """
    Mirror the SD3 training conditioning:
      - joint encoding through 3 text encoders
      - optional location embedding bias
      - Flow-matching SD3 sampler via StableDiffusion3Pipeline
    """
    train_cfg = cfg["training"]
    device = bundle.device

    text_encoders = [bundle.text_encoder_g, bundle.text_encoder_clip_l, bundle.text_encoder_t5]
    tokenizers = [bundle.tokenizer_g, bundle.tokenizer_clip_l, bundle.tokenizer_t5]
    max_seq_len = getattr(train_cfg, "max_sequence_length", 256)

    # Encode text with multi-encoder helper from train.py
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        prompt=prompts,
        max_sequence_length=max_seq_len,
        device=device,
        num_images_per_prompt=1,
    )

    # Inject location bias (same idea as in train_step_sd3)
    if (
        getattr(train_cfg, "use_location_embeddings", False)
        and bundle.location_embeddings is not None
        and bundle.location_proj is not None
        and bundle.location_token_to_id
    ):
        loc_id_list = _extract_location_ids_from_prompts(prompts, bundle.location_token_to_id, default_id=0)
        loc_ids = torch.tensor(loc_id_list, dtype=torch.long, device=device)  # [B]

        # [B, loc_dim] -> [B, t5_hidden_dim]
        loc_emb = bundle.location_embeddings(loc_ids)
        loc_vec = bundle.location_proj(loc_emb)  # [B, D_t5]

        loc_vec = loc_vec.to(device=device, dtype=prompt_embeds.dtype)

        # Add as a bias across all tokens
        prompt_embeds = prompt_embeds + loc_vec.unsqueeze(1)

    images = bundle.pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
    ).images

    # Resize to training resolution for convenience
    res = train_cfg.image_resolution
    return [im.resize((res, res), Image.BICUBIC) for im in images]


@torch.no_grad()
def generate_images(
    cfg,
    bundle: SD3ModelBundle,
    prompt: str,
    num_images: int = 5,
    steps: int = 50,
    guidance_scale: float = 3.5,
) -> torch.Tensor:
    """
    User-facing helper that:
      - repeats the prompt `num_images` times
      - runs SD3 sampling with location conditioning
      - returns a batch tensor in [0,1] range suitable for torchvision.save_image
    """
    prompts = [prompt] * num_images
    images_pil = _generate_images_sd3(
        cfg=cfg,
        bundle=bundle,
        prompts=prompts,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )

    tensors = [TF.to_tensor(im) for im in images_pil]
    return torch.stack(tensors, dim=0)


def save_images(images: torch.Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # images expected in [N, 3, H, W], [0,1]
    for i, img in enumerate(images):
        save_image(img, output_dir / f"image_{i+1}.png")


# ------------------------------------------------------------
#  Generate images for every location
# ------------------------------------------------------------
def generate_for_all_locations(
    cfg,
    bundle: SD3ModelBundle,
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
            cfg,
            bundle,
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

    if getattr(paths, "model_type", "sd3") != "sd3":
        raise ValueError("This inference script is configured for SD3 only (paths.model_type != 'sd3').")

    # Build mappings from dataset artifacts
    print("Building token and keyword maps from dataset metadata...")
    token_to_noun, token_to_classname, keyword_to_token = build_token_and_keyword_maps(paths)
    print(f"Found {len(token_to_noun)} tokens with nouns.")
    print(f"Keyword map contains {len(keyword_to_token)} entries.")

    # Experiment dirs
    run_dir = Path(paths.output_dir)
    ckpt_dir = run_dir
    best_ckpt = ckpt_dir / "checkpoint_best.pt"

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found at: {best_ckpt}")

    # Load SD3 model + LoRA + location embeddings
    print("\nLoading SD3 model and best checkpoint...")
    bundle = build_sd3_model(cfg)
    optimizer = create_optimizers(cfg, bundle)
    _ = load_checkpoint(cfg, bundle, optimizer, best_ckpt)

    # Put pipeline in eval mode
    bundle.pipe.to(bundle.device)
    bundle.pipe.set_progress_bar_config(disable=True)

    print("\nModel loaded.")
    print("Type a prompt containing a location (e.g. 'fretwell at sunset'),")
    print("or type 'all' to generate images for every known location.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_prompt = input("\nEnter prompt (or 'all' / 'exit'): ").strip()
        except EOFError:
            print("\nEOF received, exiting inference.")
            break

        if user_prompt.lower() in {"exit", "quit"}:
            print("Exiting inference.")
            break

        # Special mode: generate images per location for every token
        if user_prompt.lower() == "all":
            generate_for_all_locations(
                cfg,
                bundle,
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

        # Generate images
        print("Generating images...")
        images = generate_images(
            cfg,
            bundle,
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
        print(f"Saved {images.shape[0]} images to: {out_dir}")


if __name__ == "__main__":
    main()
