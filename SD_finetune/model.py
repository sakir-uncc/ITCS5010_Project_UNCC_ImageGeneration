from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
from torch import nn, optim

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from transformers import CLIPTextModel, CLIPTokenizer

from SD_finetune.config import CONFIG


@dataclass
class SDModelBundle:
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    device: torch.device
    # NEW: holds LoRA attention processors when use_lora=True
    lora_layers: Optional[nn.Module] = None


def build_sd15_model(cfg=CONFIG) -> SDModelBundle:
    paths = cfg["paths"]
    train_cfg = cfg["training"]
    base_model = paths.base_model_name

    # Device selection (respect GPU index)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{train_cfg.device_index}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load components
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # We never train the VAE
    vae.requires_grad_(False)

    lora_layers: Optional[nn.Module] = None

    if train_cfg.use_lora:
        # LoRA mode: freeze base UNet + text encoder, train only LoRA weights
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # Add LoRA processors to attention layers (same pattern as HuggingFace example script)
        lora_attn_procs: Dict[str, LoRAAttnProcessor] = {}

        for name in unet.attn_processors.keys():
            # self-attention blocks (attn1) don't use cross-attention
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                # Fallback; shouldn't normally happen for SD1.5, but be safe
                hidden_size = unet.config.block_out_channels[0]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=train_cfg.lora_rank,
            )

        unet.set_attn_processor(lora_attn_procs)
        # Wrap all attention processors into one module for easy optimization
        lora_layers = AttnProcsLayers(unet.attn_processors)
        print(f"LoRA enabled: rank={train_cfg.lora_rank}, {len(lora_attn_procs)} attention processors adapted.")

    else:
        # Full fine-tune: UNet always trainable, text encoder optionally
        if not train_cfg.train_text_encoder:
            text_encoder.requires_grad_(False)
        print(
            f"Full fine-tune mode. train_text_encoder={train_cfg.train_text_encoder} "
            f"(UNet always trainable)."
        )

    # Move to device
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)

    return SDModelBundle(
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        device=device,
        lora_layers=lora_layers,
    )


def create_optimizers(
    model_bundle: SDModelBundle,
    cfg=CONFIG,
) -> Tuple[optim.Optimizer, Any]:
    """
    Two modes:
      - Full fine-tune: optimize UNet + (optionally) text encoder.
      - LoRA: optimize only LoRA attention processors (and optionally text encoder if enabled).
    """
    train_cfg = cfg["training"]

    param_groups = []

    if train_cfg.use_lora:
        if model_bundle.lora_layers is None:
            raise ValueError("LoRA is enabled in config but lora_layers is None in model bundle.")

        # LoRA weights on UNet attention
        lora_params = list(model_bundle.lora_layers.parameters())
        param_groups.append({"params": lora_params, "lr": train_cfg.learning_rate_unet})

        # (Optional) also train text encoder fully even in LoRA mode if requested.
        if train_cfg.train_text_encoder:
            text_encoder_params = [p for p in model_bundle.text_encoder.parameters() if p.requires_grad]
            if text_encoder_params:
                param_groups.append(
                    {"params": text_encoder_params, "lr": train_cfg.learning_rate_text_encoder}
                )
        print(
            f"Optimizer in LoRA mode: {sum(p.numel() for p in lora_params)} LoRA params"
            + (" + text encoder params" if train_cfg.train_text_encoder else "")
        )

    else:
        # Full fine-tune
        unet_params = list(model_bundle.unet.parameters())
        param_groups.append({"params": unet_params, "lr": train_cfg.learning_rate_unet})

        if train_cfg.train_text_encoder:
            text_encoder_params = list(model_bundle.text_encoder.parameters())
            param_groups.append(
                {"params": text_encoder_params, "lr": train_cfg.learning_rate_text_encoder}
            )
            print("Optimizer in full FT mode: UNet + text encoder.")
        else:
            print("Optimizer in full FT mode: UNet only (text encoder frozen).")

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=0.01,
    )

    # Simple scheduler: cosine or constant with warmup (warmup handled externally in future if needed)
    if train_cfg.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.max_train_steps,
        )
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return optimizer, scheduler


def save_checkpoint(
    model_bundle: SDModelBundle,
    optimizer: optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    best_metric: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "unet": model_bundle.unet.state_dict(),
        "text_encoder": model_bundle.text_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(state, out_path)
    print(f"Saved checkpoint to {out_path}")


def load_checkpoint(
    model_bundle: SDModelBundle,
    optimizer: optim.Optimizer,
    scheduler: Any,
    ckpt_path: Path,
) -> Dict[str, Any]:
    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model_bundle.unet.load_state_dict(state["unet"])
    model_bundle.text_encoder.load_state_dict(state["text_encoder"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return {
        "step": state.get("step", 0),
        "epoch": state.get("epoch", 0),
        "best_metric": state.get("best_metric", -1e9),
    }
