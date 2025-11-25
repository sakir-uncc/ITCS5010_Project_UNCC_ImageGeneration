from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathsConfig:
    # Root of the dataset and outputs
    dataset_root: Path = Path(
        "/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1"
    )
    sd_lora_dataset: Path = dataset_root / "sd_lora_dataset"
    images_dir: Path = sd_lora_dataset / "images"
    captions_dir: Path = sd_lora_dataset / "captions"

    # Where to save experiments (checkpoints, logs, plots)
    runs_root: Path = dataset_root / "runs"

    # Base model (Stable Diffusion 1.5)
    base_model_name: str = "runwayml/stable-diffusion-v1-5"

    # Optional: token table if you want to inspect tokens
    location_tokens_csv: Path = dataset_root / "location_tokens.csv"


@dataclass
class TrainingConfig:
    # Reproducibility
    seed: int = 42

    # Data / transforms
    image_resolution: int = 512
    center_crop: bool = True

    # Batching
    batch_size: int = 8
    grad_accumulation: int = 8  # effective batch size = batch_size * grad_accumulation

    # Training length
    max_train_steps: int = 20000
    num_epochs: int = 10  # loop cap; training stops earlier if max_train_steps reached

    # Learning rates
    learning_rate_unet: float = 1e-4
    learning_rate_text_encoder: float = 5e-5

    # LR scheduler
    lr_scheduler: str = "cosine"  # "cosine" or "constant"
    warmup_steps: int = 500

    # Checkpoint / validation frequency
    save_every_n_steps: int = 500
    val_every_n_steps: int = 500

    # Mixed precision
    use_fp16: bool = True

    # -------- NEW: Fine-tuning mode switches --------
    # If False -> full fine-tune (UNet + optional text encoder)
    # If True  -> LoRA on UNet attention layers (text encoder frozen by default)
    use_lora: bool = False
    lora_rank: int = 8

    # Whether to train the text encoder (only meaningful for full fine-tune)
    train_text_encoder: bool = True

    # Which GPU index to use when multiple CUDA devices are visible
    device_index: int = 0

    # Resume from latest checkpoint in runs/<exp>/checkpoints if available
    resume_from_latest: bool = True


@dataclass
class EvalConfig:
    # CLIP model for validation / test metrics
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # Validation settings
    num_val_batches: int = 5  # how many batches to evaluate per validation run
    num_inference_images_per_class: int = 4  # for visualization grids


@dataclass
class ExperimentConfig:
    experiment_name: str = "sd15_uncc_fullfinetune"
    notes: str = "Stable Diffusion 1.5 fine-tuned on UNC Charlotte campus dataset."


def get_default_config() -> dict:
    paths = PathsConfig()
    training = TrainingConfig()
    eval_cfg = EvalConfig()
    exp = ExperimentConfig()

    return {
        "paths": paths,
        "training": training,
        "eval": eval_cfg,
        "experiment": exp,
    }


# convenience global
CONFIG = get_default_config()
