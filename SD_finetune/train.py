import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from SD_finetune.config import CONFIG, get_default_config
from SD_finetune.dataset_loader import build_train_val_dataloaders
from SD_finetune.model import build_sd15_model, create_optimizers, save_checkpoint, load_checkpoint
from transformers import CLIPProcessor, CLIPModel


def set_seed(seed: int):
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_text(text_encoder, tokenizer, captions: List[str], device: torch.device):
    batch = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch.input_ids.to(device)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]
    return encoder_hidden_states


def run_validation(
    model_bundle,
    val_loader,
    clip_model,
    clip_processor,
    device,
    num_batches: int,
    global_step: int,
    out_dir: Path,
) -> float:
    """
    Simple validation:
      - Generate images from a fixed number of batches of captions
      - Compute CLIP image-text similarity as a rough metric
    """

    model_bundle.unet.eval()
    model_bundle.text_encoder.eval()

    clip_model.eval()
    clip_model.to(device)

    noise_scheduler = model_bundle.noise_scheduler
    vae = model_bundle.vae
    tokenizer = model_bundle.tokenizer
    unet = model_bundle.unet

    all_scores = []

    val_iter = iter(val_loader)

    @torch.no_grad()
    def generate_images_from_captions(captions: List[str]) -> torch.Tensor:
        # encode text
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with autocast(enabled=CONFIG["training"].use_fp16 and device.type == "cuda"):
            text_embeds = model_bundle.text_encoder(text_inputs.input_ids)[0]

        # random latents
        batch_size = len(captions)
        latents = torch.randn(
            (batch_size, unet.in_channels, 64, 64),  # 512/8 = 64
            device=device,
        )

        # timesteps for generation
        noise_scheduler.set_timesteps(50, device=device)
        for t in noise_scheduler.timesteps:
            with autocast(enabled=CONFIG["training"].use_fp16 and device.type == "cuda"):
                noise_pred = unet(latents, t, encoder_hidden_states=text_embeds).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # decode
        latents = latents / 0.18215
        with autocast(enabled=CONFIG["training"].use_fp16 and device.type == "cuda"):
            images = vae.decode(latents).sample
        images = (images.clamp(-1, 1) + 1) / 2.0  # [0, 1]
        return images

    for _ in range(num_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        captions = batch["caption"]
        images_gen = generate_images_from_captions(captions)

        # CLIP scoring
        clip_inputs = clip_processor(
            text=captions,
            images=[img for img in images_gen],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad(), autocast(
            enabled=CONFIG["training"].use_fp16 and device.type == "cuda"
        ):
            clip_outputs = clip_model(**clip_inputs)
            text_embeds = clip_outputs.text_embeds  # (B, D)
            image_embeds = clip_outputs.image_embeds  # (B, D)

        # cosine similarity per sample
        sim = F.cosine_similarity(text_embeds, image_embeds)
        all_scores.extend(sim.detach().cpu().tolist())

    if not all_scores:
        return 0.0

    mean_score = float(sum(all_scores) / len(all_scores))

    # Save histogram for this val step
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(all_scores, bins=20)
    plt.title(f"CLIP score distribution @ step {global_step}")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"val_clip_hist_step_{global_step}.png")
    plt.close()

    return mean_score


def save_loss_curve(train_losses: List[float], val_metrics: List[float], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(steps, train_losses, label="Train loss")
    if val_metrics:
        val_steps = [
            s for s in range(0, len(train_losses), max(1, len(train_losses) // len(val_metrics)))
        ][: len(val_metrics)]
        plt.plot(val_steps, val_metrics, label="Val CLIP mean", linestyle="--")
    plt.xlabel("Logging step")
    plt.legend()
    plt.title("Training loss and validation CLIP metric")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png")
    plt.close()


def save_val_image_grid(images: torch.Tensor, out_path: Path, nrow: int = 4):
    """
    Save a grid of images [B, C, H, W] in [0, 1] to disk.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, nrow=nrow)
    save_image(grid, out_path)


@torch.no_grad()
def run_quick_inference(model_bundle, captions: List[str]) -> torch.Tensor:
    """
    Quick helper used inside training to visualize current model.
    """
    device = model_bundle.device
    tokenizer = model_bundle.tokenizer
    text_encoder = model_bundle.text_encoder
    unet = model_bundle.unet
    vae = model_bundle.vae
    noise_scheduler = model_bundle.noise_scheduler

    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with autocast(enabled=CONFIG["training"].use_fp16 and device.type == "cuda"):
        text_embeds = text_encoder(text_inputs.input_ids)[0]

    batch_size = len(captions)
    latents = torch.randn(
        (batch_size, unet.in_channels, 64, 64),  # 512/8
        device=device,
    )

    noise_scheduler.set_timesteps(50, device=device)
    for t in noise_scheduler.timesteps:
        with autocast(enabled=CONFIG["training"].use_fp16 and device.type == "cuda"):
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeds).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    with autocast(enabled=CONFIG["training"].use_fp16 and device.type == "cuda"):
        images = vae.decode(latents).sample
    images = (images.clamp(-1, 1) + 1) / 2.0
    return images


def main():
    cfg = get_default_config()
    paths = cfg["paths"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["eval"]
    exp_cfg = cfg["experiment"]

    set_seed(train_cfg.seed)

    # experiment dirs
    run_dir = paths.runs_root / exp_cfg.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    figs_dir = run_dir / "figures"
    grids_dir = figs_dir / "val_grids"
    for d in [run_dir, ckpt_dir, logs_dir, figs_dir, grids_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # data
    train_loader, val_loader = build_train_val_dataloaders(cfg)

    # model
    model_bundle = build_sd15_model(cfg)
    device = model_bundle.device

    # optimizer & scheduler
    optimizer, scheduler = create_optimizers(model_bundle, cfg)

    # mixed precision
    scaler = GradScaler(enabled=train_cfg.use_fp16 and device.type == "cuda")

    # CLIP for validation
    clip_model = CLIPModel.from_pretrained(eval_cfg.clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(eval_cfg.clip_model_name)

    # ---- Resume from latest checkpoint (if available and enabled) ----
    global_step = 0
    epoch = 0
    best_val_metric = -1e9

    if train_cfg.resume_from_latest and ckpt_dir.exists():
        latest_ckpt = None
        latest_step = -1

        for ckpt in ckpt_dir.glob("step_*.pt"):
            name = ckpt.stem  # e.g. "step_500"
            try:
                step = int(name.split("_")[-1])
            except ValueError:
                continue
            if step > latest_step:
                latest_step = step
                latest_ckpt = ckpt

        # If no step_* checkpoint, try best.pt
        if latest_ckpt is None:
            best_ckpt = ckpt_dir / "best.pt"
            if best_ckpt.exists():
                latest_ckpt = best_ckpt

        if latest_ckpt is not None:
            state = load_checkpoint(model_bundle, optimizer, scheduler, latest_ckpt)
            global_step = state.get("step", 0)
            epoch = state.get("epoch", 0)
            best_val_metric = state.get("best_metric", -1e9)
            print(
                f"Resumed from checkpoint {latest_ckpt} "
                f"(step={global_step}, epoch={epoch}, best_metric={best_val_metric:.4f})"
            )
        else:
            print("No existing checkpoint found in ckpt_dir; starting from scratch.")
    else:
        if not train_cfg.resume_from_latest:
            print("Resume disabled in config; starting from scratch.")
        else:
            print("Checkpoint directory does not exist yet; starting from scratch.")

    # Progress bar
    progress_bar = tqdm(total=train_cfg.max_train_steps, desc="Training", dynamic_ncols=True)
    if global_step > 0:
        progress_bar.update(global_step)

    train_losses_log: List[float] = []
    val_metrics_log: List[float] = []

    noise_scheduler = model_bundle.noise_scheduler
    vae = model_bundle.vae
    unet = model_bundle.unet
    text_encoder = model_bundle.text_encoder
    tokenizer = model_bundle.tokenizer

    unet.train()
    if train_cfg.train_text_encoder:
        text_encoder.train()

    while global_step < train_cfg.max_train_steps and epoch < train_cfg.num_epochs:
        epoch += 1
        for batch in train_loader:
            if global_step >= train_cfg.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            captions = batch["caption"]

            # encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            # sample noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # encode text
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with autocast(enabled=train_cfg.use_fp16 and device.type == "cuda"):
                encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

            # predict noise
            with autocast(enabled=train_cfg.use_fp16 and device.type == "cuda"):
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
                ).sample
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

            # backward with grad accumulation
            loss = loss / train_cfg.grad_accumulation
            scaler.scale(loss).backward()

            if (global_step + 1) % train_cfg.grad_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            train_losses_log.append(loss.detach().item() * train_cfg.grad_accumulation)

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{train_losses_log[-1]:.4f}"})
            global_step += 1

            # validation
            if global_step % train_cfg.val_every_n_steps == 0:
                val_metric = run_validation(
                    model_bundle,
                    val_loader,
                    clip_model,
                    clip_processor,
                    device,
                    num_batches=eval_cfg.num_val_batches,
                    global_step=global_step,
                    out_dir=figs_dir,
                )
                val_metrics_log.append(val_metric)
                print(f"\n[Validation] Step {global_step} | Mean CLIP score: {val_metric:.4f}")

                # if best, save checkpoint + sample grid
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_ckpt_path = ckpt_dir / "best.pt"
                    save_checkpoint(
                        model_bundle,
                        optimizer,
                        scheduler,
                        step=global_step,
                        epoch=epoch,
                        best_metric=best_val_metric,
                        out_path=best_ckpt_path,
                    )

                    # also save a visual grid from a small subset
                    batch_vis = next(iter(val_loader))
                    sample_captions = batch_vis["caption"][: eval_cfg.num_inference_images_per_class]
                    # generate images
                    model_bundle.unet.eval()
                    model_bundle.text_encoder.eval()
                    with torch.no_grad():
                        images_gen = run_quick_inference(model_bundle, sample_captions)
                    model_bundle.unet.train()
                    if train_cfg.train_text_encoder:
                        model_bundle.text_encoder.train()

                    grid_path = grids_dir / f"best_step_{global_step}.png"
                    save_val_image_grid(images_gen, grid_path, nrow=4)
                    print(f"Saved best validation grid to {grid_path}")

            # periodic checkpoints
            if global_step % train_cfg.save_every_n_steps == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                save_checkpoint(
                    model_bundle,
                    optimizer,
                    scheduler,
                    step=global_step,
                    epoch=epoch,
                    best_metric=best_val_metric,
                    out_path=ckpt_path,
                )

        # end epoch
        save_loss_curve(train_losses_log, val_metrics_log, figs_dir)

    progress_bar.close()
    print("Training finished.")
    save_loss_curve(train_losses_log, val_metrics_log, figs_dir)


if __name__ == "__main__":
    main()
