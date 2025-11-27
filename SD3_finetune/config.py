# config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================
#  PATHS AND MODEL SELECTION
# ============================================================

# Root folder used by ALL dataset preparation scripts
DATASET_ROOT = "/home/sakir/Development/PHD_courses/ITCS5010/Dataset/cv_final_project_group1"


@dataclass
class PathsConfig:
    # SD3 as default backbone
    base_model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers"

    # "sd3" or "sd15"
    model_type: str = "sd3"

    # Dataset paths — MUST match 1_tokenize.py and 2_SD_dataset_prep.py
    raw_data_dir: str = f"{DATASET_ROOT}/processed_dedup"
    processed_data_dir: str = f"{DATASET_ROOT}/sd_lora_dataset"

    csv_metadata_path: str = f"{DATASET_ROOT}/metadata_tokenized.csv"
    location_token_path: str = f"{DATASET_ROOT}/location_tokens.csv"

    # Checkpoint + logs
    output_dir: str = f"{DATASET_ROOT}/runs"
    logging_dir: str = f"{DATASET_ROOT}/runs/logs"
    checkpoint_dir: str = f"{DATASET_ROOT}/runs/checkpoints"
    inference_dir: str = f"{DATASET_ROOT}/runs/inference_outputs"

    # Optional resume
    resume_checkpoint: Optional[str] = None


# ============================================================
#  TRAINING CONFIG
# ============================================================

@dataclass
class TrainingConfig:
    num_train_epochs: int = 20
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    max_train_steps: Optional[int] = None

    learning_rate_transformer: float = 1e-4
    learning_rate_clip_lora: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    lr_scheduler_type: str = "cosine"
    lr_warmup_steps: int = 500

    image_resolution: int = 512
    center_crop: bool = False

    use_fp16: bool = True

    use_lora: bool = True
    lora_rank: int = 16

    train_text_encoder: bool = True
    train_clip_l_via_lora: bool = True
    train_openclip_g: bool = False
    train_t5: bool = False

    use_location_embeddings: bool = True
    location_embedding_dim: int = 128

    device_index: int = 0


# ============================================================
#  VALIDATION / EVALUATION CONFIG
# ============================================================

@dataclass
class EvalConfig:
    run_validation_during_training: bool = True
    num_val_batches: int = 5

    use_clip_score: bool = True

    use_dinov2: bool = True
    dinov2_model_name: str = "dinov2_vitb14"
    dinov2_device: str = "cuda"
    num_val_batches_dinov2: int = 8

    num_inference_steps: int = 28


# ============================================================
#  EXPERIMENT CONFIG
# ============================================================

@dataclass
class ExperimentConfig:
    experiment_name: str = "sd3_uncc_lora"
    seed: int = 42
    report_to: Optional[str] = None


# ============================================================
# MASTER CONFIG
# ============================================================

def get_default_config() -> Dict[str, Any]:
    return {
        "paths": PathsConfig(),
        "training": TrainingConfig(),
        "eval": EvalConfig(),
        "experiment": ExperimentConfig(),
    }
