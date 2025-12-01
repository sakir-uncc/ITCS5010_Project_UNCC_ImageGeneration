# dataset_loader.py
#
# Updated for SD3 / 1024 training.
# Clean, minimal, model-agnostic dataset builder.
#

import os
from typing import List, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from SD3_finetune.config import get_default_config


# ============================================================
#  DATASET CLASS
# ============================================================

class CampusLoraDataset(Dataset):
    """
    Loads:
      - images from processed_data_dir/images
      - matching .txt captions from processed_data_dir/captions
    Returns:
      {
        "pixel_values": FloatTensor [C,H,W] normalized to [-1,1],
        "caption": str
      }
    """

    def __init__(self, processed_data_dir: str, image_resolution: int = 1024):
        self.processed_data_dir = processed_data_dir

        self.image_dir = os.path.join(processed_data_dir, "images")
        self.caption_dir = os.path.join(processed_data_dir, "captions")

        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not os.path.isdir(self.caption_dir):
            raise ValueError(f"Caption directory does not exist: {self.caption_dir}")

        # Collect image paths
        self.image_paths = sorted(
            [
                os.path.join(self.image_dir, f)
                for f in os.listdir(self.image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]
        )

        # Require caption for each image
        for img_path in self.image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            caption_path = os.path.join(self.caption_dir, base + ".txt")
            if not os.path.isfile(caption_path):
                raise FileNotFoundError(f"Missing caption file: {caption_path}")

        self.transform = T.Compose(
            [
                T.Resize(image_resolution, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        caption_path = os.path.join(self.caption_dir, base + ".txt")

        # Load image
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Load caption text
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {
            "pixel_values": pixel_values,  # tensor [-1,1]
            "caption": caption,
        }


# ============================================================
#  DATALOADER BUILDERS
# ============================================================

def get_train_val_dataloaders(
    cfg=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train/val DataLoader pairs using the processed dataset.

    - Splits the dataset 90/10 by default.
    - Uses image_resolution from config (1024 for SD3).
    """

    if cfg is None:
        cfg = get_default_config()

    paths = cfg["paths"]
    train_cfg = cfg["training"]

    dataset = CampusLoraDataset(
        processed_data_dir=paths.processed_data_dir,
        image_resolution=train_cfg.image_resolution,
    )

    # Basic 90/10 split
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.train_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================
#  STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    cfg = get_default_config()
    train_loader, val_loader = get_train_val_dataloaders(cfg)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size:   {len(val_loader.dataset)}")

    batch = next(iter(train_loader))
    print("Pixel batch:", batch["pixel_values"].shape)
    print("Caption sample:", batch["caption"][0])
