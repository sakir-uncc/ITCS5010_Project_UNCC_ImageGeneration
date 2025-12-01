import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from SD3_finetune.config import get_default_config
from SD3_finetune.dataset_loader import get_train_val_dataloaders
from SD3_finetune.model import (
    build_sd3_model,
    build_sd15_model,
    create_optimizers,
    save_checkpoint,
    load_checkpoint,
    SD3ModelBundle,
    SD15ModelBundle,
)

# Visualization / metrics libs
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# CLIP for CLIPScore
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# BATCH HELPER
# ============================================================

def _select_pixel_and_caption(batch: Union[Dict[str, Any], Tuple[torch.Tensor, List[str]]]):
    """
    Robustly extract pixel tensors and caption strings from a batch produced by our dataset loader.
    Supports both dict-style batches and (images, captions) tuples.
    """
    pixel_values = None
    captions = None

    if isinstance(batch, dict):
        # Images
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]
        elif "images" in batch:
            pixel_values = batch["images"]
        elif "image" in batch:
            pixel_values = batch["image"]

        # Captions / prompts
        if "captions" in batch:
            captions = batch["captions"]
        elif "caption" in batch:
            captions = batch["caption"]
        elif "prompts" in batch:
            captions = batch["prompts"]
        elif "prompt" in batch:
            captions = batch["prompt"]
    else:
        # Assume (images, captions)
        pixel_values, captions = batch

    if pixel_values is None:
        raise ValueError("Could not find image tensor in batch. Expected key 'pixel_values' or 'images'.")

    if captions is None:
        # Fallback: if no captions provided, use empty strings
        bsz = pixel_values.shape[0]
        captions = [""] * bsz

    return pixel_values, captions


# ============================================================
# SD3 TEXT ENCODING HELPERS
# (adapted from diffusers train_dreambooth_lora_sd3)
# ============================================================

from typing import Any as AnyType, List as ListType


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length: int,
    prompt: Union[str, ListType[str]],
    num_images_per_prompt: int = 1,
    device: torch.device = None,
    text_input_ids: torch.LongTensor = None,
) -> torch.FloatTensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: Union[str, ListType[str]],
    device: torch.device = None,
    text_input_ids: torch.LongTensor = None,
    num_images_per_prompt: int = 1,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    outputs = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    # For CLIP text encoders, use the penultimate hidden layer as token features
    # and the [CLS] token from the last layer as pooled embedding.
    if hasattr(outputs, "last_hidden_state"):
        last_hidden_state = outputs.last_hidden_state
    else:
        last_hidden_state = outputs[0]

    prompt_embeds = outputs.hidden_states[-2]
    pooled_prompt_embeds = last_hidden_state[:, 0]  # [B, hidden]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    # duplicate for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders: ListType[nn.Module],
    tokenizers: ListType[AnyType],
    prompt: Union[str, ListType[str]],
    max_sequence_length: int,
    device: torch.device,
    num_images_per_prompt: int = 1,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Jointly encode prompts for the 3 SD3 text encoders (OpenCLIP-G, CLIP-L, T5-XXL).

    Returns:
        prompt_embeds: [B*, N_total, D_t5]
        pooled_prompt_embeds: [B*, D_clip_total]
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []

    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoder=text_encoders[-1],
        tokenizer=tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    # Pad CLIP embeddings along feature dim to match T5 embed dim if needed
    if t5_prompt_embed.shape[-1] > clip_prompt_embeds.shape[-1]:
        pad = t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]
        clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, pad))

    # Concatenate along sequence dimension: [B*, N_clip+N_t5, D_t5]
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


def _get_sigmas(scheduler, timesteps: torch.LongTensor, n_dim: int, dtype: torch.dtype, device: torch.device):
    """
    Helper to obtain σ(t) values for SD3 FlowMatchEulerDiscreteScheduler,
    adapted from the official DreamBooth SD3 training script.
    """
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    # map each timestep to its index in the scheduler's discrete time grid
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ============================================================
# LOCATION ID EXTRACTION
# ============================================================

def _extract_location_ids(
    captions: List[str],
    token_to_id: Dict[str, int],
    default_id: int = 0,
) -> List[int]:
    """
    Determine a location ID per caption by scanning for known location tokens.
    Simple heuristic: first token that appears in the caption wins.
    """
    loc_ids: List[int] = []
    tokens = list(token_to_id.items())

    for text in captions:
        if text is None:
            text = ""
        text = str(text)
        found = False
        for token, idx in tokens:
            if token in text:
                loc_ids.append(idx)
                found = True
                break
        if not found:
            loc_ids.append(default_id)
    return loc_ids


# ============================================================
# TRAIN STEP: SD3
# ============================================================

def train_step_sd3(
    cfg: Dict[str, Any],
    bundle: SD3ModelBundle,
    batch: Union[Dict[str, Any], Tuple[torch.Tensor, List[str]]],
    scaler: GradScaler,
    global_step: int,
) -> Tuple[torch.Tensor, int]:
    """
    Single training step for SD3 flow-matching + LoRA + location embeddings.

    - Encodes text via 3 SD3 text encoders
    - Optionally injects location embeddings as a bias into token-level embeddings
    - Encodes images via SD3 VAE
    - Uses FlowMatchEulerDiscreteScheduler for rectified flow training
    """
    train_cfg = cfg["training"]
    device = bundle.device

    transformer = bundle.transformer
    vae = bundle.vae
    scheduler = bundle.scheduler

    text_encoders = [bundle.text_encoder_g, bundle.text_encoder_clip_l, bundle.text_encoder_t5]
    tokenizers = [bundle.tokenizer_g, bundle.tokenizer_clip_l, bundle.tokenizer_t5]

    pixel_values, captions = _select_pixel_and_caption(batch)
    pixel_values = pixel_values.to(device=device, dtype=vae.dtype)

    # Ensure captions is a list of strings
    if isinstance(captions, str):
        captions = [captions]
    else:
        captions = list(captions)

    max_seq_len = getattr(train_cfg, "max_sequence_length", 256)

    with autocast(enabled=train_cfg.use_fp16):
        # Encode text (3 encoders)
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            prompt=captions,
            max_sequence_length=max_seq_len,
            device=device,
            num_images_per_prompt=1,
        )

        # ----------------------------------------
        # Inject location embeddings (if available)
        # ----------------------------------------
        if bundle.location_embeddings is not None and bundle.location_proj is not None and bundle.location_token_to_id:
            # Compute loc_id per caption
            loc_id_list = _extract_location_ids(captions, bundle.location_token_to_id, default_id=0)
            loc_ids = torch.tensor(loc_id_list, dtype=torch.long, device=device)  # [B]

            # [B, loc_dim] -> [B, t5_hidden_dim]
            loc_emb = bundle.location_embeddings(loc_ids)
            loc_vec = bundle.location_proj(loc_emb)  # [B, D_t5]

            loc_vec = loc_vec.to(device=device, dtype=prompt_embeds.dtype)

            # Add as a bias across all tokens
            prompt_embeds = prompt_embeds + loc_vec.unsqueeze(1)

        # ----------------------------------------
        # Encode images to latents and apply SD3 VAE scaling
        # ----------------------------------------
        latents = vae.encode(pixel_values).latent_dist.sample()
        model_input = (latents - bundle.vae_shift) * bundle.vae_scale
        model_input = model_input.to(dtype=transformer.dtype)

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Uniform timestep sampling over scheduler's training grid
        num_train_steps = len(scheduler.timesteps)
        indices = torch.randint(
            low=0,
            high=num_train_steps,
            size=(bsz,),
            device=device,
        )
        timesteps = scheduler.timesteps.to(device=device)[indices]

        sigmas = _get_sigmas(scheduler, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype, device=device)

        # Flow-matching interpolation: z_t = (1 - σ) * x + σ * z_1
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # DiT forward
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # Simple flow-matching loss: regress velocity field (noise - x)
        target = noise - model_input
        loss = torch.mean((model_pred.float() - target.float()) ** 2)

    scaler.scale(loss).backward()

    return loss.detach(), global_step + 1


# ============================================================
# TRAIN STEP: SD1.5 (legacy)
# ============================================================

def train_step_sd15(
    cfg: Dict[str, Any],
    bundle: SD15ModelBundle,
    batch: Union[Dict[str, Any], Tuple[torch.Tensor, List[str]]],
    scaler: GradScaler,
    global_step: int,
) -> Tuple[torch.Tensor, int]:
    """
    Minimal SD1.5 + LoRA training step kept for backwards compatibility.
    Uses standard noise-prediction objective.
    """
    from diffusers import DDPMScheduler  # local import to avoid unused warnings

    train_cfg = cfg["training"]
    device = bundle.device

    unet = bundle.unet
    vae = bundle.vae
    noise_scheduler: DDPMScheduler = bundle.noise_scheduler
    tokenizer = bundle.tokenizer
    text_encoder = bundle.text_encoder

    pixel_values, captions = _select_pixel_and_caption(batch)
    pixel_values = pixel_values.to(device=device, dtype=vae.dtype)

    with autocast(enabled=train_cfg.use_fp16):
        # encode text
        captions = [captions] if isinstance(captions, str) else captions
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        encoder_hidden_states = text_encoder(text_input_ids)[0]

        # encode images to latents
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * noise_scheduler.init_noise_sigma

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        target = noise
        loss = torch.mean((model_pred - target) ** 2)

    scaler.scale(loss).backward()
    return loss.detach(), global_step + 1


# ============================================================
# EVAL MODEL HELPERS (CLIP + DINOv2)
# ============================================================

def _load_clip_model(device: torch.device):
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.to(device)
        clip_model.eval()
        return clip_model, clip_processor
    except Exception as e:
        print(f"[WARN] Failed to load CLIP model for CLIPScore: {e}")
        return None, None


def _load_dinov2(eval_cfg, device: torch.device):
    if not eval_cfg.use_dinov2:
        return None, None

    try:
        model = torch.hub.load("facebookresearch/dinov2", eval_cfg.dinov2_model_name)
        model.to(device)
        model.eval()

        # Basic DINOv2 pre-processing
        dino_transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return model, dino_transform
    except Exception as e:
        print(f"[WARN] Failed to load DINOv2 model: {e}")
        return None, None


def _compute_clipscore_batch(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    images_pil: List[Image.Image],
    texts: List[str],
    device: torch.device,
) -> float:
    if clip_model is None or clip_processor is None:
        return float("nan")
    if len(images_pil) == 0:
        return float("nan")

    with torch.no_grad():
        inputs = clip_processor(
            text=texts,
            images=images_pil,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds  # [B, D]
        text_embeds = outputs.text_embeds    # [B, D]

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        sims = (image_embeds * text_embeds).sum(dim=-1)  # cosine sim
        return sims.mean().item()


def _compute_dino_similarity_batch(
    dino_model,
    dino_transform,
    real_images_pil: List[Image.Image],
    gen_images_pil: List[Image.Image],
    device: torch.device,
) -> float:
    if dino_model is None or dino_transform is None:
        return float("nan")
    if len(real_images_pil) == 0 or len(gen_images_pil) == 0:
        return float("nan")

    k = min(len(real_images_pil), len(gen_images_pil))
    real_images_pil = real_images_pil[:k]
    gen_images_pil = gen_images_pil[:k]

    with torch.no_grad():
        real_batch = torch.stack([dino_transform(im) for im in real_images_pil]).to(device)
        gen_batch = torch.stack([dino_transform(im) for im in gen_images_pil]).to(device)

        feats_real = dino_model(real_batch)
        feats_gen = dino_model(gen_batch)

        if isinstance(feats_real, (tuple, list)):
            feats_real = feats_real[0]
        if isinstance(feats_gen, (tuple, list)):
            feats_gen = feats_gen[0]

        feats_real = feats_real / feats_real.norm(dim=-1, keepdim=True)
        feats_gen = feats_gen / feats_gen.norm(dim=-1, keepdim=True)

        sims = (feats_real * feats_gen).sum(dim=-1)
        return sims.mean().item()


# ============================================================
# IMAGE GENERATION FOR VALIDATION (SD3)
# ============================================================

def _generate_images_sd3(
    cfg: Dict[str, Any],
    bundle: SD3ModelBundle,
    prompts: List[str],
    num_inference_steps: int,
) -> List[Image.Image]:
    """
    Use SD3 pipeline with the same multi-encoder + location embedding conditioning
    as training, to generate images for the given prompts.
    """
    train_cfg = cfg["training"]
    device = bundle.device

    text_encoders = [bundle.text_encoder_g, bundle.text_encoder_clip_l, bundle.text_encoder_t5]
    tokenizers = [bundle.tokenizer_g, bundle.tokenizer_clip_l, bundle.tokenizer_t5]
    max_seq_len = getattr(train_cfg, "max_sequence_length", 256)

    prompts = [str(p) for p in prompts]

    with torch.no_grad(), autocast(enabled=train_cfg.use_fp16):
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            prompt=prompts,
            max_sequence_length=max_seq_len,
            device=device,
            num_images_per_prompt=1,
        )

        # Inject location embeddings if available
        if bundle.location_embeddings is not None and bundle.location_proj is not None and bundle.location_token_to_id:
            loc_id_list = _extract_location_ids(prompts, bundle.location_token_to_id, default_id=0)
            loc_ids = torch.tensor(loc_id_list, dtype=torch.long, device=device)
            loc_emb = bundle.location_embeddings(loc_ids)
            loc_vec = bundle.location_proj(loc_emb)
            loc_vec = loc_vec.to(device=device, dtype=prompt_embeds.dtype)
            prompt_embeds = prompt_embeds + loc_vec.unsqueeze(1)

        images = bundle.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            output_type="pil",
        ).images

    res = cfg["training"].image_resolution
    return [im.resize((res, res), Image.BICUBIC) for im in images]


def _save_real_vs_fake_grid(
    real_images_pil,
    gen_images_pil,
    out_dir,
    epoch,
    image_resolution,
    max_images=4,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not real_images_pil or not gen_images_pil:
        return

    k = min(len(real_images_pil), len(gen_images_pil), max_images)

    target_size = (image_resolution, image_resolution)

    real_tensors = [
        TF.to_tensor(TF.resize(im, target_size))
        for im in real_images_pil[:k]
    ]

    gen_tensors = [
        TF.to_tensor(TF.resize(im, target_size))
        for im in gen_images_pil[:k]
    ]

    # 2 rows: first real, then generated
    grid_tensors = real_tensors + gen_tensors
    grid = make_grid(torch.stack(grid_tensors), nrow=k)

    grid_img = TF.to_pil_image(grid)
    save_path = out_dir / f"epoch_{epoch+1:03d}_real_vs_generated.png"
    grid_img.save(save_path)


# ============================================================
# VALIDATION LOOP (SD3)
# ============================================================

def run_validation_sd3(
    cfg: Dict[str, Any],
    bundle: SD3ModelBundle,
    val_loader: DataLoader,
    epoch: int,
    out_dir: Path,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    dino_model,
    dino_transform,
) -> Dict[str, float]:
    """
    Run lightweight validation on a few batches:
      - generate SD3 images for val captions
      - compute CLIPScore (prompt vs generated)
      - compute DINOv2 similarity (real vs generated)
      - save real vs generated grids
    """
    eval_cfg = cfg["eval"]
    device = bundle.device

    num_val_batches_clip = eval_cfg.num_val_batches if eval_cfg.use_clip_score else 0
    num_val_batches_dino = eval_cfg.num_val_batches_dinov2 if eval_cfg.use_dinov2 else 0
    max_batches = max(num_val_batches_clip, num_val_batches_dino)

    if max_batches == 0:
        return {"clip_score": float("nan"), "dino_score": float("nan")}

    bundle.transformer.eval()
    if bundle.clip_l_lora_params:
        bundle.text_encoder_clip_l.eval()

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    clip_scores: List[float] = []
    dino_scores: List[float] = []
    vis_done = False

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_batches:
                break

            pixel_values, captions = _select_pixel_and_caption(batch)
            if isinstance(captions, str):
                captions = [captions]
            else:
                captions = list(captions)

            bsz = pixel_values.shape[0]
            # Convert real images back to [0,1] PIL
            real_images_pil: List[Image.Image] = []
            for i in range(bsz):
                img = pixel_values[i]
                img = (img * 0.5 + 0.5).clamp(0, 1)
                real_images_pil.append(TF.to_pil_image(img))

            # Generate images with SD3 for the same prompts
            gen_images_pil = _generate_images_sd3(
                cfg=cfg,
                bundle=bundle,
                prompts=captions,
                num_inference_steps=eval_cfg.num_inference_steps,
            )

            # CLIPScore
            if eval_cfg.use_clip_score and clip_model is not None and clip_processor is not None and b_idx < num_val_batches_clip:
                k = min(len(real_images_pil), len(gen_images_pil))
                score_clip = _compute_clipscore_batch(
                    clip_model,
                    clip_processor,
                    gen_images_pil[:k],
                    captions[:k],
                    device,
                )
                if not np.isnan(score_clip):
                    clip_scores.append(score_clip)

            # DINOv2 similarity
            if eval_cfg.use_dinov2 and dino_model is not None and dino_transform is not None and b_idx < num_val_batches_dino:
                k = min(len(real_images_pil), len(gen_images_pil))
                score_dino = _compute_dino_similarity_batch(
                    dino_model,
                    dino_transform,
                    real_images_pil[:k],
                    gen_images_pil[:k],
                    device,
                )
                if not np.isnan(score_dino):
                    dino_scores.append(score_dino)

            # Save one visualization grid per epoch
            if not vis_done:
                _save_real_vs_fake_grid(
                    real_images_pil,
                    gen_images_pil,
                    samples_dir,
                    epoch,
                    cfg["training"].image_resolution,
                    max_images=4,
                )
                vis_done = True

    clip_mean = float(np.mean(clip_scores)) if clip_scores else float("nan")
    dino_mean = float(np.mean(dino_scores)) if dino_scores else float("nan")

    return {
        "clip_score": clip_mean,
        "dino_score": dino_mean,
    }


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_training_curves(
    out_dir: Path,
    train_loss_per_epoch: List[float],
    clip_scores: List[float],
    dino_scores: List[float],
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_loss_per_epoch) + 1)

    # 1) Train loss per epoch
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_per_epoch, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss per epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "train_loss_per_epoch.png")
    plt.close()

    # 2) CLIP + DINO metrics
    if clip_scores or dino_scores:
        plt.figure(figsize=(6, 4))
        if clip_scores:
            cs = [np.nan if x is None else x for x in clip_scores]
            plt.plot(epochs[: len(cs)], cs, label="CLIPScore")
        if dino_scores:
            ds = [np.nan if x is None else x for x in dino_scores]
            plt.plot(epochs[: len(ds)], ds, label="DINOv2 similarity")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Validation metrics over epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "val_clip_dino_curves.png")
        plt.close()


def append_metrics_csv(
    metrics_dir: Path,
    epoch: int,
    train_loss: float,
    clip_score: float,
    dino_score: float,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "train_metrics.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("epoch,train_loss,clip_score,dino_score\n")
        f.write(f"{epoch+1},{train_loss},{clip_score},{dino_score}\n")


# ============================================================
# MAIN TRAIN LOOP
# ============================================================

def train(cfg: Dict[str, Any]) -> None:
    """
    Main training entry point. Chooses SD3 or SD1.5 path based on cfg['paths'].model_type,
    and logs train-time visualizations plus CLIP + DINO metrics if enabled.
    """
    paths = cfg["paths"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["eval"]

    out_dir = Path(paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloaders
    train_loader, val_loader = get_train_val_dataloaders(cfg)

    # Build model bundle
    if getattr(paths, "model_type", "sd3") == "sd3":
        bundle = build_sd3_model(cfg)
    else:
        bundle = build_sd15_model(cfg)

    optimizer = create_optimizers(cfg, bundle)
    scaler = GradScaler(enabled=train_cfg.use_fp16)

    start_epoch = 0
    global_step = 0
    best_metric = -float("inf")
    best_ckpt_path = out_dir / "checkpoint_best.pt"

    # Optional resume
    resume_ckpt = getattr(paths, "resume_checkpoint", None)
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if ckpt_path.is_file():
            info = load_checkpoint(cfg, bundle, optimizer, ckpt_path)
            start_epoch = info["epoch"]
            global_step = info["step"]
            best_metric = info["best_metric"]

    device = bundle.device
    model_type = getattr(paths, "model_type", "sd3")

    # Load eval models if needed
    if model_type == "sd3":
        clip_model, clip_processor = _load_clip_model(device) if eval_cfg.use_clip_score else (None, None)
        dino_model, dino_transform = _load_dinov2(eval_cfg, device) if eval_cfg.use_dinov2 else (None, None)
    else:
        clip_model, clip_processor, dino_model, dino_transform = None, None, None, None

    num_epochs = train_cfg.num_train_epochs
    grad_accum = train_cfg.gradient_accumulation_steps

    # Tracking for plots
    train_loss_per_epoch: List[float] = []
    clip_scores_per_epoch: List[float] = []
    dino_scores_per_epoch: List[float] = []

    for epoch in range(start_epoch, num_epochs):
        if model_type == "sd3":
            bundle.transformer.train()
            if bundle.clip_l_lora_params:
                bundle.text_encoder_clip_l.train()
        else:
            bundle.unet.train()

        running_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in pbar:
            if model_type == "sd3":
                loss, global_step = train_step_sd3(cfg, bundle, batch, scaler, global_step)
            else:
                loss, global_step = train_step_sd15(cfg, bundle, batch, scaler, global_step)

            running_loss += loss.item()

            if (step_idx + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = running_loss / (step_idx + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_loss = running_loss / len(train_loader)
        train_loss_per_epoch.append(epoch_loss)

        # Simple checkpointing each epoch
        ckpt_path = out_dir / f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(cfg, bundle, optimizer, global_step, epoch + 1, best_metric, ckpt_path)

        print(f"[Epoch {epoch+1}] Avg train loss: {epoch_loss:.4f}")

        # -------------------------------
        # Validation + visualizations
        # -------------------------------
        clip_score_epoch = float("nan")
        dino_score_epoch = float("nan")

        if model_type == "sd3" and eval_cfg.run_validation_during_training:
            metrics = run_validation_sd3(
                cfg,
                bundle,
                val_loader,
                epoch,
                out_dir,
                clip_model,
                clip_processor,
                dino_model,
                dino_transform,
            )
            clip_score_epoch = metrics["clip_score"]
            dino_score_epoch = metrics["dino_score"]

            print(
                f"[Epoch {epoch+1}] "
                f"CLIPScore: {clip_score_epoch:.4f}  "
                f"DINOv2 similarity: {dino_score_epoch:.4f}"
            )

        clip_scores_per_epoch.append(clip_score_epoch)
        dino_scores_per_epoch.append(dino_score_epoch)

        # ------------------------------
        # BEST CHECKPOINT (DINOv2-based)
        # ------------------------------
        # Only evaluate if DINO score is finite (not nan)
        if dino_score_epoch == dino_score_epoch:   # nan-safe check
            if dino_score_epoch > best_metric:
                print(f"[Epoch {epoch+1}] New BEST checkpoint! DINOv2 improved from {best_metric:.4f} → {dino_score_epoch:.4f}")
                best_metric = dino_score_epoch
                save_checkpoint(
                    cfg,
                    bundle,
                    optimizer,
                    global_step,
                    epoch + 1,
                    best_metric,
                    best_ckpt_path
                )

        # Append metrics CSV row
        append_metrics_csv(
            metrics_dir=metrics_dir,
            epoch=epoch,
            train_loss=epoch_loss,
            clip_score=clip_score_epoch,
            dino_score=dino_score_epoch,
        )

        # Update plots
        plot_training_curves(out_dir, train_loss_per_epoch, clip_scores_per_epoch, dino_scores_per_epoch)


    # Final save
    final_ckpt = out_dir / "checkpoint_final.pt"
    save_checkpoint(cfg, bundle, optimizer, global_step, num_epochs, best_metric, final_ckpt)


if __name__ == "__main__":
    cfg = get_default_config()
    train(cfg)
