from pathlib import Path
from typing import List

import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from config import get_default_config
from dataset_loader import build_train_val_dataloaders
from model import build_sd15_model, create_optimizers, load_checkpoint
from transformers import CLIPProcessor, CLIPModel

from train import run_validation, run_quick_inference, set_seed


def save_test_image_grid(images: torch.Tensor, out_path: Path, nrow: int = 4):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images, nrow=nrow)
    save_image(grid, out_path)


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
    figs_dir = run_dir / "figures"
    test_figs_dir = figs_dir / "test"
    test_grids_dir = test_figs_dir / "grids"

    for d in [run_dir, ckpt_dir, figs_dir, test_figs_dir, test_grids_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # data (reuse train/val split; here we evaluate on the val split)
    _, val_loader = build_train_val_dataloaders(cfg)

    # model
    model_bundle = build_sd15_model(cfg)
    device = model_bundle.device

    # optimizer & scheduler (required for checkpoint loading, though not used for testing)
    optimizer, scheduler = create_optimizers(model_bundle, cfg)

    # load best checkpoint
    best_ckpt = ckpt_dir / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {best_ckpt}. Run train.py first.")

    state = load_checkpoint(model_bundle, optimizer, scheduler, best_ckpt)
    step = state.get("step", 0)
    best_metric = state.get("best_metric", float("nan"))
    print(f"Loaded best checkpoint from step={step}, best_val_metric={best_metric:.4f}")

    model_bundle.unet.eval()
    model_bundle.text_encoder.eval()

    # CLIP for evaluation
    eval_cfg = cfg["eval"]
    clip_model = CLIPModel.from_pretrained(eval_cfg.clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(eval_cfg.clip_model_name)

    # Evaluate on the entire val set
    mean_clip_score = run_validation(
        model_bundle,
        val_loader,
        clip_model,
        clip_processor,
        device,
        num_batches=len(val_loader),
        global_step=step,
        out_dir=test_figs_dir,
    )

    print(f"[TEST] Mean CLIP similarity over val set: {mean_clip_score:.4f}")

    # Also generate a grid of images from a subset of captions
    batch = next(iter(val_loader))
    sample_captions: List[str] = batch["caption"][: eval_cfg.num_inference_images_per_class]
    with torch.no_grad():
        images_gen = run_quick_inference(model_bundle, sample_captions)

    grid_path = test_grids_dir / f"best_step_{step}.png"
    save_test_image_grid(images_gen, grid_path, nrow=4)
    print(f"[TEST] Saved test image grid to {grid_path}")


if __name__ == "__main__":
    main()
