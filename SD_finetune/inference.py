from pathlib import Path
from typing import List

import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid, save_image

from config import CONFIG, get_default_config
from model import build_sd15_model, create_optimizers, load_checkpoint
from dataset_loader import CampusLoraDataset


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def generate_images_for_prompts(
    model_bundle,
    prompts: List[str],
    num_inference_steps: int = 50,
    cfg_scale: float = 7.5,
) -> torch.Tensor:
    """
    Simple classifier-free guidance-less sampler (like validation one).
    For simplicity, we ignore cfg_scale here and just do unconditional generation from prompts.
    """
    device = model_bundle.device
    tokenizer = model_bundle.tokenizer
    text_encoder = model_bundle.text_encoder
    unet = model_bundle.unet
    vae = model_bundle.vae
    noise_scheduler = model_bundle.noise_scheduler

    train_cfg = CONFIG["training"]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with autocast(device_type="cuda", enabled=train_cfg.use_fp16 and device.type == "cuda"):
        text_embeds = text_encoder(text_inputs.input_ids)[0]

    batch_size = len(prompts)
    latents = torch.randn(
        (batch_size, unet.in_channels, 64, 64),
        device=device,
    )

    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    for t in noise_scheduler.timesteps:
        with autocast(device_type="cuda", enabled=train_cfg.use_fp16 and device.type == "cuda"):
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeds).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    with autocast(device_type="cuda", enabled=train_cfg.use_fp16 and device.type == "cuda"):
        images = vae.decode(latents).sample
    images = (images.clamp(-1, 1) + 1) / 2.0
    return images


def save_grid(images: torch.Tensor, out_path: Path, nrow: int = 4):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, nrow=nrow)
    save_image(grid, out_path)


def main():
    cfg = get_default_config()
    paths = cfg["paths"]
    train_cfg = cfg["training"]
    exp_cfg = cfg["experiment"]

    set_seed(train_cfg.seed)

    run_dir = paths.runs_root / exp_cfg.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    figs_dir = run_dir / "figures"
    inf_dir = figs_dir / "inference_grids"
    inf_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    # build model and load best checkpoint
    model_bundle = build_sd15_model(cfg)
    optimizer, scheduler = create_optimizers(model_bundle, cfg)
    _ = load_checkpoint(model_bundle, optimizer, scheduler, best_ckpt)

    model_bundle.unet.eval()
    model_bundle.text_encoder.eval()

    # Example prompts for some tokens (you should customize with your actual <loc_...> tokens)
    example_prompts = [
        "a photo of <loc_storrshall> building at UNC Charlotte at sunset",
        "a photo of <loc_fretwellhall> building at UNC Charlotte on a cloudy day",
        "a watercolor painting of <loc_aperturesculpture> sculpture at UNC Charlotte",
        "a photo of <loc_hechenbleiknerlake> lake at UNC Charlotte during autumn",
    ]

    images = generate_images_for_prompts(model_bundle, example_prompts, num_inference_steps=50)
    grid_path = inf_dir / "example_prompts_grid.png"
    save_grid(images, grid_path, nrow=2)
    print(f"Saved example inference grid to {grid_path}")


if __name__ == "__main__":
    main()
