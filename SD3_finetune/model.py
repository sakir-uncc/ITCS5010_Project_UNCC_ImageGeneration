from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import csv

import torch
from torch import nn, optim

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from transformers import CLIPTextModel, CLIPTokenizer

from SD3_finetune.config import get_default_config


# ============================================================
# SD1.5 BUNDLE (legacy)
# ============================================================

@dataclass
class SD15ModelBundle:
    """Container for SD1.5-style components used in the legacy path."""
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    device: torch.device
    lora_layers: Optional[AttnProcsLayers] = None


# ============================================================
# SD3 BUNDLE
# ============================================================

@dataclass
class SD3ModelBundle:
    """Container for SD3 components and LoRA/location objects."""
    pipe: StableDiffusion3Pipeline
    transformer: nn.Module
    vae: AutoencoderKL

    # Three SD3 text encoders + tokenizers
    text_encoder_clip_l: nn.Module
    tokenizer_clip_l: CLIPTokenizer

    text_encoder_g: nn.Module
    text_encoder_t5: nn.Module
    tokenizer_g: CLIPTokenizer
    tokenizer_t5: Any

    scheduler: FlowMatchEulerDiscreteScheduler
    vae_shift: float
    vae_scale: float

    # Location conditioning
    location_embeddings: Optional[nn.Embedding]
    location_proj: Optional[nn.Module]
    location_token_to_id: Dict[str, int]

    device: torch.device

    # LoRA parameter collections (filtered to requires_grad == True)
    transformer_lora_params: Optional[List[nn.Parameter]] = None
    clip_l_lora_params: Optional[List[nn.Parameter]] = None
    location_params: Optional[List[nn.Parameter]] = None


# ============================================================
# SD1.5 BUILDER (unchanged legacy)
# ============================================================

def build_sd15_model(cfg: Optional[Dict[str, Any]] = None) -> SD15ModelBundle:
    """
    Legacy SD1.5 model builder.
    Kept for backwards compatibility; primary project path is SD3.
    """
    if cfg is None:
        cfg = get_default_config()

    paths = cfg["paths"]
    train_cfg = cfg["training"]

    base_model = paths.base_model_name
    device = torch.device(f"cuda:{train_cfg.device_index}" if torch.cuda.is_available() else "cpu")

    # Load SD1.5 style components
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)

    lora_layers = None
    if train_cfg.use_lora:
        # Freeze base UNet and text encoder weights, then add LoRA adapters to attention.
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)

        attn_procs = {}
        for name, module in unet.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v"):
                hidden_size = module.to_q.in_features
                attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, rank=train_cfg.lora_rank)

        unet.set_attn_processor(attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)
        lora_layers.to(device)

    return SD15ModelBundle(
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        device=device,
        lora_layers=lora_layers,
    )


# ============================================================
# LOCATION TOKEN HELPERS
# ============================================================

def _load_location_token_to_id(location_csv_path: Path) -> Dict[str, int]:
    """
    Load mapping from location token string (e.g. "<loc_fretwellhall>")
    to a contiguous integer ID, based on location_tokens.csv.

    Expected CSV columns (from 1_tokenize.py):
        class_name, token
    """
    token_to_id: Dict[str, int] = {}
    if not location_csv_path.exists():
        return token_to_id

    with location_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        idx = 0
        for row in reader:
            token = row.get("token")
            if not token:
                continue
            token = token.strip()
            if token and token not in token_to_id:
                token_to_id[token] = idx
                idx += 1

    return token_to_id


# ============================================================
# SD3 BUILDER
# ============================================================

def build_sd3_model(cfg: Optional[Dict[str, Any]] = None) -> SD3ModelBundle:
    """
    Build SD3 Medium backbone using StableDiffusion3Pipeline and
    prepare high-level objects for LoRA fine-tuning and location conditioning.
    """
    if cfg is None:
        cfg = get_default_config()

    paths = cfg["paths"]
    train_cfg = cfg["training"]

    base_model = paths.base_model_name
    device = torch.device(f"cuda:{train_cfg.device_index}" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float16 if train_cfg.use_fp16 and torch.cuda.is_available() else torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model,
        torch_dtype=weight_dtype,
    )

    # Replace scheduler with FlowMatchEulerDiscreteScheduler (recommended for SD3)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    transformer = pipe.transformer
    vae = pipe.vae

    # SD3 uses three text encoders; by convention:
    #   text_encoder      -> OpenCLIP-G
    #   text_encoder_2    -> CLIP-L
    #   text_encoder_3    -> T5-XXL
    text_encoder_g = pipe.text_encoder
    text_encoder_clip_l = pipe.text_encoder_2
    text_encoder_t5 = pipe.text_encoder_3

    tokenizer_g = pipe.tokenizer
    tokenizer_clip_l = pipe.tokenizer_2
    tokenizer_t5 = pipe.tokenizer_3

    vae_shift = float(getattr(vae.config, "shift_factor", 0.0))
    vae_scale = float(getattr(vae.config, "scaling_factor", 1.0))

    # Freeze base weights; LoRA will re-enable training for its adapters.
    transformer.requires_grad_(False)
    text_encoder_g.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    if not train_cfg.train_clip_l_via_lora:
        text_encoder_clip_l.requires_grad_(False)

    # ------------------------------------
    # Location embeddings + projection
    # ------------------------------------
    loc_csv = Path(getattr(paths, "location_token_path", "location_tokens.csv"))
    token_to_id = _load_location_token_to_id(loc_csv)

    location_embeddings: Optional[nn.Embedding] = None
    location_proj: Optional[nn.Module] = None
    location_params: Optional[List[nn.Parameter]] = None

    if train_cfg.use_location_embeddings and len(token_to_id) > 0:
        num_locations = len(token_to_id)
        loc_dim = train_cfg.location_embedding_dim
        location_embeddings = nn.Embedding(
            num_embeddings=num_locations,
            embedding_dim=loc_dim,
        )

        # Project location embedding to T5 hidden size so it can be added to prompt_embeds
        t5_hidden_dim = int(getattr(text_encoder_t5.config, "d_model", 4096))
        location_proj = nn.Linear(loc_dim, t5_hidden_dim, bias=False)

        location_embeddings.to(device)
        location_proj.to(device)

        location_params = list(location_embeddings.parameters()) + list(location_proj.parameters())

        for p in location_params:
            p.data = p.data.float()

    else:
        token_to_id = {}

    bundle = SD3ModelBundle(
        pipe=pipe,
        transformer=transformer,
        vae=vae,
        text_encoder_clip_l=text_encoder_clip_l,
        tokenizer_clip_l=tokenizer_clip_l,
        text_encoder_g=text_encoder_g,
        text_encoder_t5=text_encoder_t5,
        tokenizer_g=tokenizer_g,
        tokenizer_t5=tokenizer_t5,
        scheduler=pipe.scheduler,
        vae_shift=vae_shift,
        vae_scale=vae_scale,
        location_embeddings=location_embeddings,
        location_proj=location_proj,
        location_token_to_id=token_to_id,
        device=device,
    )

    # Inject LoRA adapters and collect trainable parameters.
    _setup_sd3_lora(bundle, train_cfg)

    # Attach location params if any
    bundle.location_params = location_params

    return bundle


# ============================================================
# SD3 LoRA SETUP
# ============================================================

def _setup_sd3_lora(bundle: SD3ModelBundle, train_cfg) -> None:
    """
    Add LoRA adapters to the SD3 transformer and (optionally) the CLIP-L text encoder.
    Collects references to trainable parameters in the bundle.
    """
    from peft import LoraConfig  # lazy import

    transformer = bundle.transformer
    text_encoder_clip_l = bundle.text_encoder_clip_l

    transformer_lora_params: List[nn.Parameter] = []
    clip_l_lora_params: List[nn.Parameter] = []

    if train_cfg.use_lora:
        # LoRA on the transformer attention modules.
        transformer_lora_config = LoraConfig(
            r=train_cfg.lora_rank,
            lora_alpha=train_cfg.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)
        transformer_lora_params = [p for p in transformer.parameters() if p.requires_grad]

    if train_cfg.train_clip_l_via_lora:
        text_lora_config = LoraConfig(
            r=train_cfg.lora_rank,
            lora_alpha=train_cfg.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_clip_l.add_adapter(text_lora_config)
        clip_l_lora_params = [p for p in text_encoder_clip_l.parameters() if p.requires_grad]

    bundle.transformer_lora_params = transformer_lora_params
    bundle.clip_l_lora_params = clip_l_lora_params

    for p in transformer_lora_params:
        p.data = p.data.float()
    for p in clip_l_lora_params:
        p.data = p.data.float()


# ============================================================
# OPTIMIZER CREATION
# ============================================================

def create_optimizers(
    cfg: Dict[str, Any],
    model_bundle: Any,
) -> optim.Optimizer:
    """
    Create optimizer for either SD1.5 or SD3, depending on cfg['paths'].model_type.

    For SD1.5, only LoRA attention processors are optimized (if enabled).
    For SD3, we optimize transformer LoRA params, optional CLIP-L LoRA params,
    and optional location embedding/projection parameters.
    """
    train_cfg = cfg["training"]
    model_type = getattr(cfg["paths"], "model_type", "sd3")

    params: List[nn.Parameter] = []

    if model_type == "sd3":
        assert isinstance(model_bundle, SD3ModelBundle)
        if model_bundle.transformer_lora_params:
            params.extend(model_bundle.transformer_lora_params)
        if model_bundle.clip_l_lora_params:
            params.extend(model_bundle.clip_l_lora_params)
        if model_bundle.location_params:
            params.extend(model_bundle.location_params)
    else:
        # SD1.5 path
        assert isinstance(model_bundle, SD15ModelBundle)
        if model_bundle.lora_layers is not None:
            params.extend([p for p in model_bundle.lora_layers.parameters() if p.requires_grad])

    optimizer = optim.AdamW(
        params,
        lr=train_cfg.learning_rate_transformer,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.adam_weight_decay,
        eps=train_cfg.adam_epsilon,
    )
    return optimizer


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
    cfg: Dict[str, Any],
    model_bundle: Any,
    optimizer: optim.Optimizer,
    step: int,
    epoch: int,
    best_metric: float,
    out_path: Path,
) -> None:
    """
    Save a minimal checkpoint for SD3 or SD1.5 fine-tuning.
    For SD3, only LoRA weights and location embeddings/projection (if used) are stored.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_type = getattr(cfg["paths"], "model_type", "sd3")

    state: Dict[str, Any] = dict(step=step, epoch=epoch, best_metric=best_metric)

    if model_type == "sd3":
        from peft.utils import get_peft_model_state_dict

        assert isinstance(model_bundle, SD3ModelBundle)
        # Save only LoRA adapter weights for the transformer (and CLIP-L if present).
        transformer_lora = get_peft_model_state_dict(model_bundle.transformer)
        state["transformer_lora"] = transformer_lora

        if model_bundle.clip_l_lora_params:
            clip_l_lora = get_peft_model_state_dict(model_bundle.text_encoder_clip_l)
            state["clip_l_lora"] = clip_l_lora

        if model_bundle.location_embeddings is not None:
            state["location_embeddings"] = model_bundle.location_embeddings.state_dict()
        if model_bundle.location_proj is not None:
            state["location_proj"] = model_bundle.location_proj.state_dict()
    else:
        assert isinstance(model_bundle, SD15ModelBundle)
        state["unet"] = model_bundle.unet.state_dict()
        state["text_encoder"] = model_bundle.text_encoder.state_dict()

    state["optimizer"] = optimizer.state_dict()

    torch.save(state, out_path)


def load_checkpoint(
    cfg: Dict[str, Any],
    model_bundle: Any,
    optimizer: optim.Optimizer,
    ckpt_path: Path,
) -> Dict[str, Any]:
    """
    Load checkpoint into SD3 or SD1.5 bundles.
    Returns bookkeeping dict with step, epoch, and best_metric.
    """
    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model_type = getattr(cfg["paths"], "model_type", "sd3")

    if model_type == "sd3":
        from peft import set_peft_model_state_dict

        assert isinstance(model_bundle, SD3ModelBundle)
        # Restore LoRA weights into the transformer and CLIP-L (if present).
        if "transformer_lora" in state:
            set_peft_model_state_dict(
                model_bundle.transformer,
                state["transformer_lora"],
                adapter_name="default",
            )
        if "clip_l_lora" in state:
            set_peft_model_state_dict(
                model_bundle.text_encoder_clip_l,
                state["clip_l_lora"],
                adapter_name="default",
            )
        if model_bundle.location_embeddings is not None and "location_embeddings" in state:
            model_bundle.location_embeddings.load_state_dict(state["location_embeddings"])
        if model_bundle.location_proj is not None and "location_proj" in state:
            model_bundle.location_proj.load_state_dict(state["location_proj"])
    else:
        assert isinstance(model_bundle, SD15ModelBundle)
        model_bundle.unet.load_state_dict(state["unet"])
        model_bundle.text_encoder.load_state_dict(state["text_encoder"])

    optimizer.load_state_dict(state["optimizer"])

    return {
        "step": state.get("step", 0),
        "epoch": state.get("epoch", 0),
        "best_metric": state.get("best_metric", -1e9),
    }
